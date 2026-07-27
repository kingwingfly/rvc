//! The RVC fine-tuning loop (Burn autodiff on CUDA).

use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;

use anyhow::{Context, Result, anyhow};
use burn::backend::Autodiff;
use burn::backend::cuda::{Cuda, CudaDevice};
use burn::module::{AutodiffModule, Module, ModuleMapper, ModuleVisitor, Param};
use burn::optim::{AdamWConfig, GradientsParams, Optimizer};
use burn::tensor::{Int, Tensor, TensorData, TensorPrimitive};
use burn_rvc::{MultiPeriodDiscriminator, Synthesizer, SynthesizerConfig};

use crate::TrainRequest;
use crate::dashboard::Dashboard;
use crate::dataset::{CONTENT_DIM, Clip, HOP, Rng, clip_weights, sample_batch};
use crate::losses::{disc_loss, feature_matching, gen_adv, kl, mel_l1};
use crate::spectral::{Spectral, SpectralConfig};

/// Autodiff GPU backend.
type AB = Autodiff<Cuda>;
/// Inner (non-autodiff) backend — gradients and the saved/EMA weights live here.
type IB = Cuda;

const SEGMENT_FRAMES: usize = 36; // 17280 samples / 480 hop
/// Context window (frames) fed to enc_q/flow each step; clips must be at least
/// this long.
pub const WINDOW_FRAMES: usize = 48;
const C_MEL: f64 = 45.0;
const C_KL: f64 = 1.0;
const FM_WEIGHT: f64 = 2.0;

/// Run fine-tuning and return the path to the saved (safetensors) weights.
pub fn run(req: &TrainRequest, clips: Vec<Clip>) -> Result<PathBuf> {
    anyhow::ensure!(
        req.settings.sample_rate == 48_000,
        "native training currently supports only --model-sr 48000"
    );
    let device = CudaDevice::default();
    let cfg = SynthesizerConfig::v2_48k();

    // ---- Generator: resume a prior run, else warm-start, else scratch -------
    let mut net_g = Synthesizer::<AB>::new(&cfg, &device);
    if let Some(requested) = &req.resume {
        // Resume from a generator .safetensors written by an earlier run,
        // preferring the raw (non-EMA) live twin when it exists: it's the actual
        // last-step generator that co-evolved with the saved discriminator, so
        // `raw-G <-> live-D` is the faithful GAN-resume pairing (the EMA snapshot
        // is a smoothed average that never itself faced D).
        let p = prefer_raw_twin(requested);
        if p != *requested {
            tracing::info!(
                "resume: preferring raw live weights {} over the EMA snapshot {}",
                p.display(),
                requested.display()
            );
        }
        let res = net_g
            .load_weights(&p)
            .map_err(|e| anyhow!("resuming G from {}: {e}", p.display()))?;
        anyhow::ensure!(
            res.missing.is_empty(),
            "G resume incomplete ({} missing): {} is not a full generator checkpoint",
            res.missing.len(),
            p.display()
        );
        tracing::info!("resumed generator from {}", p.display());
    } else if let Some(p) = &req.pretrained_g {
        let res = net_g
            .load_pytorch(p)
            .map_err(|e| anyhow!("loading G {}: {e}", p.display()))?;
        anyhow::ensure!(
            res.missing.is_empty(),
            "G warm-start incomplete: {} missing",
            res.missing.len()
        );
        tracing::info!("warm-started generator from {}", p.display());
    } else {
        tracing::warn!(
            "no --pretrained-g: training the generator from scratch (poor on small data)"
        );
    }

    // ---- Discriminator: resume from the sidecar if present, else warm-start -
    let mut disc = MultiPeriodDiscriminator::<AB>::new(&device);
    let disc_resume = req
        .resume
        .as_deref()
        .map(disc_sidecar_path)
        .filter(|p| p.exists());
    if let Some(p) = &disc_resume {
        let res = disc
            .load_safetensors(p)
            .map_err(|e| anyhow!("resuming D from {}: {e}", p.display()))?;
        anyhow::ensure!(
            res.missing.is_empty(),
            "D resume incomplete: {} missing",
            res.missing.len()
        );
        tracing::info!("resumed discriminator from {}", p.display());
    } else if let Some(p) = &req.pretrained_d {
        let res = disc
            .load_pytorch(p)
            .map_err(|e| anyhow!("loading D {}: {e}", p.display()))?;
        anyhow::ensure!(
            res.missing.is_empty(),
            "D warm-start incomplete: {} missing",
            res.missing.len()
        );
        tracing::info!("warm-started discriminator from {}", p.display());
    } else if req.resume.is_some() {
        tracing::warn!(
            "resuming without a discriminator checkpoint or --pretrained-d: \
             the discriminator starts fresh (adversarial training will lag)"
        );
    }

    let spectral = Spectral::<AB>::new(&SpectralConfig::v2_48k(), &device);

    let mut opt_g = AdamWConfig::new()
        .with_beta_1(0.8)
        .with_beta_2(0.99)
        .with_epsilon(1e-9)
        .with_weight_decay(0.01)
        .init::<AB, Synthesizer<AB>>();
    let mut opt_d = AdamWConfig::new()
        .with_beta_1(0.8)
        .with_beta_2(0.99)
        .with_epsilon(1e-9)
        .with_weight_decay(0.01)
        .init::<AB, MultiPeriodDiscriminator<AB>>();
    let base_lr = req.settings.lr;
    let lr_final = req.settings.lr_final;
    let accum = req.settings.grad_accum.max(1);
    let d_lr_ratio = req.settings.d_lr_ratio;
    let d_interval = req.settings.d_interval.max(1);

    let batch = req.settings.batch_size.max(1);
    let total_frames: usize = clips.iter().map(|c| c.frames).sum();
    let steps_per_epoch = (total_frames / (batch * WINDOW_FRAMES)).max(1);
    let total_steps = req.settings.epochs as usize * steps_per_epoch;

    // Derive the per-step EMA decay from a smoothing window = ema_frac of the run,
    // so it stays sensible for any epoch count. `ema_frac == 0` disables EMA.
    let ema_window = req.settings.ema_frac * total_steps as f64;
    let ema_decay = if ema_window >= 1.0 {
        (1.0 - 1.0 / ema_window).min(0.9999)
    } else {
        0.0
    };
    // Clip-sampling bias (None = uniform); computed once from each clip's SNR.
    let cdf = clip_weights(&clips, req.settings.snr_weight);
    tracing::info!(
        "training: {} clips, batch {}x{} accum, {} steps/epoch, {} epochs -> {} steps; \
         lr {:.1e}->{:.1e} (final {}x), ema window {:.0} steps (decay {:.4}), \
         d-lr-ratio {} interval {}, snr-weight {}",
        clips.len(),
        batch,
        accum,
        steps_per_epoch,
        req.settings.epochs,
        total_steps,
        base_lr,
        base_lr * lr_final,
        lr_final,
        ema_window.max(0.0),
        ema_decay,
        d_lr_ratio,
        d_interval,
        req.settings.snr_weight,
    );

    let mut rng = Rng::new(0x51D_u64.wrapping_mul(req.settings.epochs as u64 + 1));
    let sid = req.settings.speaker_id;
    let seg_len = SEGMENT_FRAMES * HOP;

    // Generator weight EMA (kept on the inner backend). The saved model is the
    // EMA — averaged over the adversarial oscillation, so cleaner. `None` when
    // disabled (`--ema-frac 0`), in which case the raw live weights are saved.
    let mut ema: Option<Synthesizer<IB>> = (ema_decay > 0.0).then(|| net_g.valid());

    let out = req.out.with_extension("safetensors");

    // "Best" checkpointing (on unless `--no-save-best`): the per-step mel is noisy enough
    // that its single-step minimum is mostly luck, so we compare the *mean* over a
    // window sized to give ~20 evaluations over a run of any length. Each hit
    // rewrites a full G + D pair, so keeping the count bounded also keeps the I/O
    // negligible next to the compute.
    let best_path = req.settings.save_best.then(|| best_path(&out));
    let best_window = (total_steps / 20).max(1);
    let mut best_mel = f32::INFINITY;
    let mut best_step: Option<usize> = None;
    let (mut win_sum, mut win_n) = (0.0f32, 0usize);

    let mut dash = Dashboard::new(
        req.settings.use_tui,
        steps_per_epoch,
        req.settings.epochs as usize,
    );

    let mut stopped_early = false;
    let mut last_d = 0.0f32; // carried across steps that skip the D update
    for step in 0..total_steps {
        // Early stop: SIGINT (non-TUI) or `q` in the dashboard. The model saved
        // below reflects the last completed step.
        if req.stop.load(Ordering::Relaxed) || dash.interrupted() {
            stopped_early = true;
            break;
        }
        // Exponential LR schedule over the whole run: base_lr at step 0 decaying
        // to base_lr * lr_final at the final step (epoch-count independent).
        let progress = step as f64 / total_steps.max(1) as f64;
        let cur_lr = base_lr * lr_final.powf(progress);
        let update_d = step % d_interval == 0;

        // Accumulate gradients over `accum` micro-batches (effective batch =
        // batch * accum) before a single optimizer step: steadier gradients
        // without the VRAM of a larger real batch. `accum == 1` is plain SGD.
        let inv = 1.0 / accum as f32;
        let mut acc_g = GradientsParams::new();
        let mut acc_d = GradientsParams::new();
        let mut g_sum = 0.0f32;
        let mut d_sum = 0.0f32;
        let mut mel_sum = 0.0f32;

        let b = batch;
        for _ in 0..accum {
            let data = sample_batch(&clips, b, WINDOW_FRAMES, &mut rng, cdf.as_deref());

            let phone = Tensor::<AB, 3>::from_data(
                TensorData::new(data.phone, [b, WINDOW_FRAMES, CONTENT_DIM]),
                &device,
            );
            let pitch = Tensor::<AB, 2, Int>::from_data(
                TensorData::new(data.coarse, [b, WINDOW_FRAMES]),
                &device,
            );
            let nsff0 = Tensor::<AB, 2>::from_data(
                TensorData::new(data.nsff0, [b, WINDOW_FRAMES]),
                &device,
            );
            let gt = Tensor::<AB, 2>::from_data(
                TensorData::new(data.gt, [b, WINDOW_FRAMES * HOP]),
                &device,
            );

            // enc_q input spectrogram (a constant w.r.t. autodiff).
            let spec = spectral.linear(gt.clone()).detach();

            // Random decode segment per sample.
            let ids: Vec<usize> = (0..b)
                .map(|_| rng.below(WINDOW_FRAMES - SEGMENT_FRAMES + 1))
                .collect();

            let tf = net_g.forward_train(phone, pitch, nsff0, spec, sid, &ids, SEGMENT_FRAMES);

            // Ground-truth audio segment matching `ids`.
            let mut segs = Vec::with_capacity(b);
            for (i, &s) in ids.iter().enumerate() {
                segs.push(gt.clone().narrow(0, i, 1).narrow(1, s * HOP, seg_len));
            }
            let y = Tensor::cat(segs, 0); // [b, seg_len]
            let y3 = y.clone().reshape([b, 1, seg_len]);
            let y_hat = tf.y_hat; // [b, 1, seg_len]

            // ---- Discriminator (fake detached) ------------------------------
            if update_d {
                let y_hat_d = y_hat.clone().detach();
                let mut d_loss = disc_loss(
                    disc.scale.forward(y3.clone()).0,
                    disc.scale.forward(y_hat_d.clone()).0,
                );
                for p in &disc.periods {
                    d_loss =
                        d_loss + disc_loss(p.forward(y3.clone()).0, p.forward(y_hat_d.clone()).0);
                }
                d_sum += scalar(&d_loss);
                let d_grads = GradientsParams::from_grads(d_loss.backward(), &disc);
                acc_d = accumulate(acc_d, d_grads, &disc, inv);
            }

            // ---- Generator --------------------------------------------------
            let mel_loss = mel_l1(
                spectral.mel(y.clone()),
                spectral.mel(y_hat.clone().reshape([b, seg_len])),
            )
            .mul_scalar(C_MEL);
            let kl_loss = kl(tf.z_p, tf.logs_q, tf.m_p, tf.logs_p).mul_scalar(C_KL);

            let (sf_s, ff_s) = disc.scale.forward(y_hat.clone().reshape([b, 1, seg_len]));
            let (_, fr_s) = disc.scale.forward(y3.clone());
            let mut g_adv = gen_adv(sf_s);
            let mut g_fm = feature_matching(&fr_s, &ff_s);
            for p in &disc.periods {
                let (sf, ff) = p.forward(y_hat.clone().reshape([b, 1, seg_len]));
                let (_, fr) = p.forward(y3.clone());
                g_adv = g_adv + gen_adv(sf);
                g_fm = g_fm + feature_matching(&fr, &ff);
            }
            let g_loss = g_adv + g_fm.mul_scalar(FM_WEIGHT) + mel_loss.clone() + kl_loss.clone();
            g_sum += scalar(&g_loss);
            mel_sum += scalar(&mel_loss);
            let g_grads = GradientsParams::from_grads(g_loss.backward(), &net_g);
            acc_g = accumulate(acc_g, g_grads, &net_g, inv);
        }

        // ---- Optimizer steps (once per accumulated batch) -------------------
        if update_d {
            disc = opt_d.step(cur_lr * d_lr_ratio, disc, acc_d);
            last_d = d_sum * inv;
        }
        net_g = opt_g.step(cur_lr, net_g, acc_g);

        // Track the EMA of the just-updated generator weights.
        if let Some(e) = ema.take() {
            ema = Some(ema_update(e, &net_g, ema_decay));
        }

        let g_scalar = g_sum * inv;
        let mel_scalar = mel_sum * inv;
        dash.update(step, g_scalar, last_d, mel_scalar, cur_lr);
        // Always log the losses (throttled). With the TUI, `init_logging` routes
        // tracing to `{work_dir}/train.log` (not stderr), so this stays off the
        // dashboard while still recording every run's curves.
        if step % 20 == 0 || step + 1 == total_steps {
            tracing::info!(
                "step {}/{}  lr={:.2e} g={:.3} d={:.3} mel={:.3}",
                step,
                total_steps,
                cur_lr,
                g_scalar,
                last_d,
                mel_scalar
            );
        }

        // Snapshot the best-so-far weights. What we save is what a deploy would
        // use (the EMA when enabled), even though the mel we score is the live
        // generator's — the EMA trails it, which is the point.
        if let Some(p) = &best_path {
            win_sum += mel_scalar;
            win_n += 1;
            if win_n >= best_window {
                let mean = win_sum / win_n as f32;
                win_sum = 0.0;
                win_n = 0;
                if mean.is_finite() && mean < best_mel {
                    match save_checkpoint(ema.as_ref(), &net_g, &disc, p) {
                        // Only claim the new best once it's actually on disk, so
                        // a failed write doesn't block a later (slightly worse)
                        // minimum from being saved.
                        Ok(()) => {
                            best_mel = mean;
                            best_step = Some(step);
                            tracing::info!(
                                "new best mel {:.3} at step {}; saved {}",
                                mean,
                                step,
                                p.display()
                            );
                        }
                        Err(e) => tracing::warn!("could not save best checkpoint: {e:#}"),
                    }
                }
            }
        }
    }

    // Close the TUI (restores the terminal) before we log/save.
    dash.finish();
    if stopped_early {
        tracing::info!("stopped early; saving current weights");
    }

    // Save the fine-tuned generator (inference-backend weights). With EMA on,
    // the saved model is the EMA — averaged over the adversarial oscillation, so
    // cleaner and less staticky than the raw final step.
    save_generator(ema.as_ref(), &net_g, &out).with_context(|| "saving trained weights")?;
    if ema.is_some() {
        tracing::info!("saved EMA generator to {}", out.display());
        // Keep the raw (non-EMA) live weights alongside for resume/debug.
        let raw = out.with_extension("raw.safetensors");
        if let Err(e) = net_g.valid().save_safetensors(&raw) {
            tracing::warn!("could not write raw weights {}: {e}", raw.display());
        }
    } else {
        tracing::info!("saved generator to {}", out.display());
    }

    // Save the discriminator alongside it so a later `--continue <out>` resumes
    // the adversary too (a best-effort sidecar; a failure here must not discard
    // the generator we already wrote).
    let disc_out = disc_sidecar_path(&out);
    match disc.valid().save_safetensors(&disc_out) {
        Ok(()) => tracing::info!("saved discriminator checkpoint to {}", disc_out.display()),
        Err(e) => tracing::warn!(
            "could not save discriminator checkpoint {}: {e} (resume will fall back to --pretrained-d)",
            disc_out.display()
        ),
    }

    if let (Some(p), Some(step)) = (&best_path, best_step) {
        tracing::info!(
            "best checkpoint: mel {:.3} (step {}) at {} (+ {})",
            best_mel,
            step,
            p.display(),
            disc_sidecar_path(p).display()
        );
    } else if best_path.is_some() {
        tracing::warn!(
            "no best checkpoint was written (the run was too short to complete a window)"
        );
    }
    Ok(out)
}

/// Sidecar path for the discriminator checkpoint that pairs with a generator
/// safetensors: `voice.safetensors` -> `voice.disc.safetensors`.
///
/// A run also writes the raw (non-EMA) live weights as `voice.raw.safetensors`,
/// which is the natural thing to `--resume` from. That twin must resolve to the
/// *same* sidecar as its EMA output, so we drop a trailing `.raw` stem first —
/// otherwise `--resume voice.raw.safetensors` would look for the non-existent
/// `voice.raw.disc.safetensors` and the discriminator would silently start fresh.
fn disc_sidecar_path(generator: &std::path::Path) -> PathBuf {
    let base = match generator.file_stem().and_then(|s| s.to_str()) {
        Some(stem) if stem.ends_with(".raw") => {
            generator.with_file_name(&stem[..stem.len() - ".raw".len()])
        }
        _ => generator.to_path_buf(),
    };
    base.with_extension("disc.safetensors")
}

/// Path of the *best* checkpoint that pairs with a run's output weights:
/// `models/voice.safetensors` -> `models/checkpoint/voice.best.safetensors`.
///
/// It lives in its own `checkpoint/` subdirectory so it never collides with the
/// final output, and keeps the run's suffix so [`disc_sidecar_path`] yields the
/// matching `voice.best.disc.safetensors` — i.e. `--resume` accepts it as-is.
fn best_path(out: &Path) -> PathBuf {
    let dir = out.parent().unwrap_or(Path::new(".")).join("checkpoint");
    let name = out.file_stem().and_then(|s| s.to_str()).unwrap_or("voice");
    let suffix = out
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or("safetensors");
    dir.join(format!("{name}.best.{suffix}"))
}

/// Save a generator + discriminator pair to `path` and its `.disc` sidecar.
///
/// Unlike the end-of-run save this is all-or-nothing: a half-written pair is
/// worse than none, since the two are only useful together on `--resume`.
fn save_checkpoint(
    ema: Option<&Synthesizer<IB>>,
    net_g: &Synthesizer<AB>,
    disc: &MultiPeriodDiscriminator<AB>,
    path: &Path,
) -> Result<()> {
    save_generator(ema, net_g, path)?;
    let d = disc_sidecar_path(path);
    disc.valid()
        .save_safetensors(&d)
        .map_err(|e| anyhow!("saving {}: {e}", d.display()))
}

/// Resolve which generator to actually resume from, preferring the raw
/// (non-EMA) live twin when it exists: `voice.safetensors` -> the sibling
/// `voice.raw.safetensors`, if present. The raw weights are the real last-step
/// generator that co-evolved with the saved discriminator, so `raw-G <-> live-D`
/// is the faithful GAN-resume pairing. A path that is already the raw twin (or
/// has no sibling twin) is returned unchanged.
fn prefer_raw_twin(resume: &std::path::Path) -> PathBuf {
    let already_raw = resume
        .file_stem()
        .and_then(|s| s.to_str())
        .is_some_and(|stem| stem.ends_with(".raw"));
    if already_raw {
        return resume.to_path_buf();
    }
    let raw = resume.with_extension("raw.safetensors");
    if raw.exists() {
        raw
    } else {
        resume.to_path_buf()
    }
}

/// Extract a scalar loss value to `f32` for logging.
fn scalar(t: &Tensor<AB, 1>) -> f32 {
    t.clone()
        .into_data()
        .to_vec::<f32>()
        .map(|v| v.first().copied().unwrap_or(f32::NAN))
        .unwrap_or(f32::NAN)
}

/// A [`ModuleVisitor`] that sums `incoming` (scaled) into `acc`, matching
/// gradients by [`ParamId`]. Gradients live on the inner backend [`IB`] even
/// though the module is autodiff-wrapped, so lookups use `IB`.
struct GradAccum {
    acc: GradientsParams,
    incoming: GradientsParams,
    scale: f32,
}

impl ModuleVisitor<AB> for GradAccum {
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<AB, D>>) {
        let id = param.id;
        if let Some(g) = self.incoming.remove::<IB, D>(id) {
            let g = if (self.scale - 1.0).abs() > f32::EPSILON {
                g.mul_scalar(self.scale)
            } else {
                g
            };
            let merged = match self.acc.remove::<IB, D>(id) {
                Some(a) => a + g,
                None => g,
            };
            self.acc.register::<IB, D>(id, merged);
        }
    }
}

/// Add `incoming * scale` into `acc`, param by param (for gradient
/// accumulation). `module` supplies the traversal over parameter ids.
fn accumulate<M: Module<AB>>(
    acc: GradientsParams,
    incoming: GradientsParams,
    module: &M,
    scale: f32,
) -> GradientsParams {
    let mut v = GradAccum {
        acc,
        incoming,
        scale,
    };
    module.visit(&mut v);
    v.acc
}

/// Collects a module's float parameters (in traversal order) as rank-erased
/// primitives, so a same-typed module can be blended against them element-wise.
struct ParamCollector {
    prims: Vec<TensorPrimitive<IB>>,
}

impl ModuleVisitor<IB> for ParamCollector {
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<IB, D>>) {
        self.prims.push(param.val().into_primitive());
    }
}

/// [`ModuleMapper`] applying `ema = keep*ema + (1-keep)*src`, consuming the
/// collected source primitives in the same traversal order.
struct EmaBlend {
    prims: Vec<TensorPrimitive<IB>>,
    keep: f64,
}

impl ModuleMapper<IB> for EmaBlend {
    fn map_float<const D: usize>(&mut self, param: Param<Tensor<IB, D>>) -> Param<Tensor<IB, D>> {
        let (id, dst, mapper) = param.consume();
        let src = Tensor::<IB, D>::from_primitive(self.prims.pop().expect("ema source underflow"));
        let blended = dst.mul_scalar(self.keep) + src.mul_scalar(1.0 - self.keep);
        Param::from_mapped_value(id, blended, mapper)
    }
}

/// Update the generator weight EMA toward the current live weights.
fn ema_update(ema: Synthesizer<IB>, net_g: &Synthesizer<AB>, keep: f64) -> Synthesizer<IB> {
    let src = net_g.valid();
    let mut collect = ParamCollector { prims: Vec::new() };
    src.visit(&mut collect);
    // `map` traverses in the same order as `visit`; reverse so `pop()` yields
    // the source params in that order.
    collect.prims.reverse();
    let mut blend = EmaBlend {
        prims: collect.prims,
        keep,
    };
    ema.map(&mut blend)
}

/// Save the generator: the EMA weights if present, else the raw live weights.
fn save_generator(
    ema: Option<&Synthesizer<IB>>,
    net_g: &Synthesizer<AB>,
    path: &Path,
) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    let res = match ema {
        Some(e) => e.save_safetensors(path),
        None => net_g.valid().save_safetensors(path),
    };
    res.map_err(|e| anyhow!("saving {}: {e}", path.display()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{best_path, disc_sidecar_path, prefer_raw_twin};
    use std::path::{Path, PathBuf};

    #[test]
    fn best_checkpoint_lands_in_the_checkpoint_dir_with_a_sidecar() {
        let best = best_path(Path::new("models/voice.safetensors"));
        assert_eq!(
            best,
            PathBuf::from("models/checkpoint/voice.best.safetensors")
        );
        // It must be resumable as-is: its sidecar is the *best* discriminator,
        // not the final run's.
        assert_eq!(
            disc_sidecar_path(&best),
            PathBuf::from("models/checkpoint/voice.best.disc.safetensors")
        );
        // A bare output name (no directory) still gets a checkpoint dir.
        assert_eq!(
            best_path(Path::new("voice.safetensors")),
            PathBuf::from("checkpoint/voice.best.safetensors")
        );
    }

    #[test]
    fn disc_sidecar_matches_ema_and_raw() {
        // The EMA output and its raw twin must resolve to the *same* sidecar,
        // so `--resume voice.raw.safetensors` finds the discriminator saved
        // next to `voice.safetensors`.
        let want = PathBuf::from("out/voice.disc.safetensors");
        assert_eq!(disc_sidecar_path(Path::new("out/voice.safetensors")), want);
        assert_eq!(
            disc_sidecar_path(Path::new("out/voice.raw.safetensors")),
            want
        );
    }

    #[test]
    fn prefer_raw_twin_switches_only_when_twin_exists() {
        let dir = std::env::temp_dir().join(format!("rvc-twin-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let ema = dir.join("voice.safetensors");
        let raw = dir.join("voice.raw.safetensors");

        // No raw twin yet: resuming from the EMA path stays put.
        assert_eq!(prefer_raw_twin(&ema), ema);

        // Once the raw twin exists, the EMA path resolves to it...
        std::fs::write(&raw, b"x").unwrap();
        assert_eq!(prefer_raw_twin(&ema), raw);
        // ...while an already-raw path is returned unchanged (no double `.raw`).
        assert_eq!(prefer_raw_twin(&raw), raw);

        std::fs::remove_dir_all(&dir).ok();
    }
}
