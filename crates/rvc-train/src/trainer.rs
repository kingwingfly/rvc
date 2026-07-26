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
    if let Some(p) = &req.resume {
        // Resume from a generator .safetensors written by an earlier run.
        let res = net_g
            .load_weights(p)
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
    let lr_decay = req.settings.lr_decay;
    let ema_decay = req.settings.ema_decay;
    let accum = req.settings.grad_accum.max(1);
    let d_lr_ratio = req.settings.d_lr_ratio;
    let d_interval = req.settings.d_interval.max(1);

    let batch = req.settings.batch_size.max(1);
    let total_frames: usize = clips.iter().map(|c| c.frames).sum();
    let steps_per_epoch = (total_frames / (batch * WINDOW_FRAMES)).max(1);
    let total_steps = req.settings.epochs as usize * steps_per_epoch;
    // Clip-sampling bias (None = uniform); computed once from each clip's SNR.
    let cdf = clip_weights(&clips, req.settings.snr_weight);
    tracing::info!(
        "training: {} clips, batch {}x{} accum, {} steps/epoch, {} epochs -> {} steps; \
         lr {} decay {}, ema {}, d-lr-ratio {} interval {}, snr-weight {}",
        clips.len(),
        batch,
        accum,
        steps_per_epoch,
        req.settings.epochs,
        total_steps,
        base_lr,
        lr_decay,
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
    // disabled (`--ema 0`), in which case the raw live weights are saved.
    let mut ema: Option<Synthesizer<IB>> = (ema_decay > 0.0).then(|| net_g.valid());

    let mut dash = Dashboard::new(
        req.settings.use_tui,
        total_steps,
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
        let epoch = step / steps_per_epoch;
        let cur_lr = base_lr * lr_decay.powi(epoch as i32);
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
    }

    // Close the TUI (restores the terminal) before we log/save.
    dash.finish();
    if stopped_early {
        tracing::info!("stopped early; saving current weights");
    }

    // Save the fine-tuned generator (inference-backend weights). With EMA on,
    // the saved model is the EMA — averaged over the adversarial oscillation, so
    // cleaner and less staticky than the raw final step.
    let out = req.out.with_extension("safetensors");
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
    Ok(out)
}

/// Sidecar path for the discriminator checkpoint next to a generator
/// safetensors: `voice.safetensors` -> `voice.disc.safetensors`.
fn disc_sidecar_path(generator: &std::path::Path) -> PathBuf {
    generator.with_extension("disc.safetensors")
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
