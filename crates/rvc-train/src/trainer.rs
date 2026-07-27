//! The RVC fine-tuning loop (Burn autodiff), generic over the compute backend.
//!
//! `AB` runs the loop; `AB::InnerBackend` holds gradients, the EMA and every
//! saved weight. `AutodiffBackend` guarantees both share a device type, so one
//! `device` serves the whole loop. [`crate::train`] picks `AB` at run time.

use std::marker::PhantomData;
use std::path::PathBuf;
use std::sync::atomic::Ordering;

use anyhow::{Context, Result, anyhow};
use burn::module::{AutodiffModule, Module, ModuleMapper, ModuleVisitor, Param};
use burn::optim::{AdamWConfig, GradientsParams, Optimizer};
use burn::tensor::backend::{AutodiffBackend, Backend};
use burn::tensor::{Int, Tensor, TensorData, TensorPrimitive};
use burn_rvc::{MultiPeriodDiscriminator, Synthesizer, SynthesizerConfig};

use crate::TrainRequest;
use crate::checkpoint::{BestMeta, Checkpoint};
use crate::dashboard::Dashboard;
use crate::dataset::{CONTENT_DIM, Clip, HOP, Rng, clip_weights, sample_batch};
use crate::losses::{disc_loss, feature_matching, gen_adv, kl, mel_l1};
use crate::spectral::{Spectral, SpectralConfig};

const SEGMENT_FRAMES: usize = 36; // 17280 samples / 480 hop
/// Context window (frames) fed to enc_q/flow each step; clips must be at least
/// this long.
pub const WINDOW_FRAMES: usize = 48;
const C_MEL: f64 = 45.0;
const C_KL: f64 = 1.0;
const FM_WEIGHT: f64 = 2.0;

/// Run fine-tuning and return the path to the saved (safetensors) weights.
pub fn run<AB: AutodiffBackend>(
    req: &TrainRequest,
    clips: Vec<Clip>,
    device: &AB::Device,
) -> Result<PathBuf> {
    anyhow::ensure!(
        req.settings.sample_rate == 48_000,
        "native training currently supports only --model-sr 48000"
    );
    let cfg = SynthesizerConfig::v2_48k();

    // ---- Generator: resume a prior run, else warm-start, else scratch -------
    let mut net_g = Synthesizer::<AB>::new(&cfg, device);
    let resume = req.resume.as_deref().map(Checkpoint::new);
    if let Some(ck) = &resume {
        let p = ck.resume_from();
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
    let mut disc = MultiPeriodDiscriminator::<AB>::new(device);
    let disc_resume = resume.as_ref().map(Checkpoint::disc).filter(|p| p.exists());
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
    } else if resume.is_some() {
        tracing::warn!(
            "resuming without a discriminator checkpoint or --pretrained-d: \
             the discriminator starts fresh (adversarial training will lag)"
        );
    }

    let spectral = Spectral::<AB>::new(&SpectralConfig::v2_48k(), device);

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

    // Generator weight EMA (kept on the inner backend): averaged over the
    // adversarial oscillation, so cleaner than any single step, and what every
    // checkpoint deploys. `None` when disabled (`--ema-frac 0`).
    let mut ema: Option<Synthesizer<AB::InnerBackend>> = (ema_decay > 0.0).then(|| net_g.valid());

    let out = Checkpoint::new(&req.out);

    // Best-so-far checkpointing (on unless `--no-save-best`). The per-step mel is
    // noisy enough that its minimum is mostly luck, so we compare the *mean* over a
    // window. The cap matters more than the fraction: `total_steps` is the
    // *scheduled* count, and a run that is stopped by hand — the way this trainer
    // is meant to be used — would otherwise exit before its first window ever
    // closed, leaving the in-loop save dead code.
    let best = req.settings.save_best.then(|| out.best());
    let best_window = (total_steps / 20).clamp(1, 50);
    // Inherit the score the last run left, so a fresh process can only *improve* on
    // it; starting from infinity makes every run's first window a clobber. A
    // sidecar whose weights are gone is ignored — it would veto every save.
    let prev = best
        .as_ref()
        .filter(|ck| ck.generator().exists())
        .and_then(Checkpoint::load_meta);
    let mut best_mel = prev.map_or(f32::INFINITY, |m| m.mel);
    // `Some` only once this run has written one, which is what the closing report
    // distinguishes: an inherited best is not evidence that this run produced one.
    let mut best_step: Option<usize> = None;
    let (mut win_sum, mut win_n) = (0.0f32, 0usize);
    if let Some(m) = prev {
        tracing::info!(
            "best so far: mel {:.3} (step {}) from a prior run",
            m.mel,
            m.step
        );
    }

    let mut dash = Dashboard::new(
        req.settings.use_tui,
        steps_per_epoch,
        req.settings.epochs as usize,
    );

    let mut stopped_early = false;
    let mut last_d = 0.0f32; // carried across steps that skip the D update
    let mut last_step = 0usize; // the last step actually executed, for the tail window
    for step in 0..total_steps {
        // Early stop: SIGINT (non-TUI) or `q` in the dashboard. The model saved
        // below reflects the last completed step.
        if req.stop.load(Ordering::Relaxed) || dash.interrupted() {
            stopped_early = true;
            break;
        }
        last_step = step;
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
                device,
            );
            let pitch = Tensor::<AB, 2, Int>::from_data(
                TensorData::new(data.coarse, [b, WINDOW_FRAMES]),
                device,
            );
            let nsff0 =
                Tensor::<AB, 2>::from_data(TensorData::new(data.nsff0, [b, WINDOW_FRAMES]), device);
            let gt = Tensor::<AB, 2>::from_data(
                TensorData::new(data.gt, [b, WINDOW_FRAMES * HOP]),
                device,
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

        if let Some(ck) = &best {
            win_sum += mel_scalar;
            win_n += 1;
            if win_n == best_window {
                let mean = win_sum / win_n as f32;
                (win_sum, win_n) = (0.0, 0);
                if keep_best(ck, &mut best_mel, mean, step, ema.as_ref(), &net_g, &disc) {
                    best_step = Some(step);
                }
            }
        }
    }

    // Close the TUI (restores the terminal) before we log/save.
    dash.finish();
    if stopped_early {
        tracing::info!("stopped early; saving current weights");
    }

    // Judge the partial window an early stop (or a ragged tail) leaves behind, but
    // only when there is enough of it to mean anything: a one-step mean carries
    // many times the variance of a full window, so a lucky tail would otherwise
    // unseat a genuinely better best. Under that bar it still stands when nothing
    // is on disk — a stopped-early run must leave *something* rather than nothing.
    if let Some(ck) = &best {
        let trustworthy = win_n * 2 >= best_window || best_mel.is_infinite();
        if win_n > 0 && trustworthy {
            let mean = win_sum / win_n as f32;
            if keep_best(
                ck,
                &mut best_mel,
                mean,
                last_step,
                ema.as_ref(),
                &net_g,
                &disc,
            ) {
                best_step = Some(last_step);
            }
        }
    }

    out.save(ema.as_ref(), &net_g.valid(), &disc.valid())
        .context("saving trained weights")?;
    if let Some(ck) = &best {
        match best_step {
            Some(step) => tracing::info!(
                "best checkpoint (mel {best_mel:.3}, step {step}): {}",
                ck.generator().display()
            ),
            // A run that contributes no best is invisible otherwise: say plainly
            // that this one added nothing, and whether anything is there at all.
            None if best_mel.is_finite() => tracing::info!(
                "no new best this run; keeping mel {best_mel:.3} in {}",
                ck.generator().display()
            ),
            None => tracing::warn!(
                "no best checkpoint was written (the run was too short to complete a window)"
            ),
        }
    }
    Ok(out.generator())
}

/// Save `ck` when `mean` beats `best`, reporting whether it did. `best` advances
/// only once the whole family is on disk, so a failed write can't block a later
/// minimum from being saved; the score sidecar is written last for the same
/// reason — a later run must not inherit a best whose weights never landed.
fn keep_best<AB: AutodiffBackend>(
    ck: &Checkpoint,
    best: &mut f32,
    mean: f32,
    step: usize,
    ema: Option<&Synthesizer<AB::InnerBackend>>,
    net_g: &Synthesizer<AB>,
    disc: &MultiPeriodDiscriminator<AB>,
) -> bool {
    if mean.is_nan() || mean >= *best {
        return false;
    }
    tracing::info!("new best mel {mean:.3} (step {step})");
    match ck.save(ema, &net_g.valid(), &disc.valid()) {
        Ok(()) => {
            *best = mean;
            if let Err(e) = ck.save_meta(BestMeta { mel: mean, step }) {
                // The weights are the checkpoint; losing the score only costs the
                // *next* run its memory of what to beat.
                tracing::warn!("could not record the best score: {e:#}");
            }
            true
        }
        Err(e) => {
            tracing::warn!("could not save best checkpoint: {e:#}");
            false
        }
    }
}

/// Extract a scalar loss value to `f32` for logging.
fn scalar<AB: AutodiffBackend>(t: &Tensor<AB, 1>) -> f32 {
    t.clone()
        .into_data()
        .to_vec::<f32>()
        .map(|v| v.first().copied().unwrap_or(f32::NAN))
        .unwrap_or(f32::NAN)
}

/// Sums `incoming` (scaled) into `acc`, matched by `ParamId`. Gradients live on
/// `AB::InnerBackend` even though the module is autodiff-wrapped, which is why
/// the lookups below name it.
struct GradAccum<AB> {
    acc: GradientsParams,
    incoming: GradientsParams,
    scale: f32,
    /// `AB` appears only in the trait we implement, never in a field.
    _ab: PhantomData<AB>,
}

impl<AB: AutodiffBackend> ModuleVisitor<AB> for GradAccum<AB> {
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<AB, D>>) {
        let id = param.id;
        if let Some(g) = self.incoming.remove::<AB::InnerBackend, D>(id) {
            let g = if (self.scale - 1.0).abs() > f32::EPSILON {
                g.mul_scalar(self.scale)
            } else {
                g
            };
            let merged = match self.acc.remove::<AB::InnerBackend, D>(id) {
                Some(a) => a + g,
                None => g,
            };
            self.acc.register::<AB::InnerBackend, D>(id, merged);
        }
    }
}

/// Add `incoming * scale` into `acc`, param by param (for gradient
/// accumulation). `module` supplies the traversal over parameter ids.
fn accumulate<AB: AutodiffBackend, M: Module<AB>>(
    acc: GradientsParams,
    incoming: GradientsParams,
    module: &M,
    scale: f32,
) -> GradientsParams {
    let mut v = GradAccum::<AB> {
        acc,
        incoming,
        scale,
        _ab: PhantomData,
    };
    module.visit(&mut v);
    v.acc
}

/// Collects a module's float parameters (in traversal order) as rank-erased
/// primitives, so a same-typed module can be blended against them element-wise.
struct ParamCollector<B: Backend> {
    prims: Vec<TensorPrimitive<B>>,
}

impl<B: Backend> ModuleVisitor<B> for ParamCollector<B> {
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
        self.prims.push(param.val().into_primitive());
    }
}

/// [`ModuleMapper`] applying `ema = keep*ema + (1-keep)*src`, consuming the
/// collected source primitives in the same traversal order.
struct EmaBlend<B: Backend> {
    prims: Vec<TensorPrimitive<B>>,
    keep: f64,
}

impl<B: Backend> ModuleMapper<B> for EmaBlend<B> {
    fn map_float<const D: usize>(&mut self, param: Param<Tensor<B, D>>) -> Param<Tensor<B, D>> {
        let (id, dst, mapper) = param.consume();
        let src = Tensor::<B, D>::from_primitive(self.prims.pop().expect("ema source underflow"));
        let blended = dst.mul_scalar(self.keep) + src.mul_scalar(1.0 - self.keep);
        Param::from_mapped_value(id, blended, mapper)
    }
}

/// Update the generator weight EMA toward the current live weights.
fn ema_update<AB: AutodiffBackend>(
    ema: Synthesizer<AB::InnerBackend>,
    net_g: &Synthesizer<AB>,
    keep: f64,
) -> Synthesizer<AB::InnerBackend> {
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
    let blended = ema.map(&mut blend);
    // Pairing is positional, so a `map`/`visit` order drift would silently blend
    // the wrong weights together. Leftovers prove the two disagreed on the count.
    assert!(blend.prims.is_empty(), "ema source overflow");
    blended
}
