//! The RVC fine-tuning loop (Burn autodiff), generic over the compute backend.
//!
//! `AB` runs the loop; `AB::InnerBackend` holds gradients, the EMA and every
//! saved weight. `AutodiffBackend` guarantees both share a device type, so one
//! `device` serves the whole loop. [`crate::train`] picks `AB` at run time.

use std::path::PathBuf;
use std::sync::atomic::Ordering;

use anyhow::{Context, Result, anyhow};
use burn::module::{AutodiffModule, Module};
use burn::optim::{AdamWConfig, GradientsParams, Optimizer};
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::{Int, Tensor, TensorData};
use burn_rvc::{MultiPeriodDiscriminator, Synthesizer, SynthesizerConfig};

use crate::TrainRequest;
use crate::dataset::{CONTENT_DIM, Clip, HOP, clip_weights, sample_batch};
use burn_vits::{Spectral, SpectralConfig, disc_loss, feature_matching, gen_adv, kl, mel_l1};
use train_kit::{
    Best, Checkpoint, Dashboard, Rng, Schedule, accumulate, ema_update, human, materialize, scalar,
};

/// Latent frames rendered per step: 17280 samples / 480 hop. The default for
/// `TrainSettings::segment_frames`, which is what the loop actually reads.
pub const SEGMENT_FRAMES: usize = 36;
/// Context window (frames) fed to enc_q/flow each step; clips must be at least
/// this long. The default for `TrainSettings::window_frames`.
pub const WINDOW_FRAMES: usize = 48;
const C_MEL: f64 = 45.0;
const C_KL: f64 = 1.0;
const FM_WEIGHT: f64 = 2.0;

/// Run fine-tuning and return the path to the saved (safetensors) weights.
pub fn run<AB: AutodiffBackend>(
    req: &TrainRequest,
    clips: Vec<Clip>,
    devices: &[AB::Device],
) -> Result<PathBuf> {
    anyhow::ensure!(!devices.is_empty(), "no compute device selected");
    // Device 0 owns the weights, the optimizers and the EMA; the others only
    // ever produce gradients.
    let device = &devices[0];
    let main_device = device;
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
        burn_kit::check_coverage(&format!("G resume from {}", p.display()), &res)
            .map_err(anyhow::Error::msg)?;
        tracing::info!("resumed generator from {}", p.display());
    } else if let Some(p) = &req.pretrained_g {
        let res = net_g
            .load_pytorch(p)
            .map_err(|e| anyhow!("loading G {}: {e}", p.display()))?;
        burn_kit::check_coverage(&format!("G warm-start from {}", p.display()), &res)
            .map_err(anyhow::Error::msg)?;
        tracing::info!("warm-started generator from {}", p.display());
    } else {
        tracing::warn!(
            "no --pretrained-g: training the generator from scratch (poor on small data)"
        );
    }

    // ---- Discriminator: resume from the sidecar if present, else warm-start -
    let mut disc = MultiPeriodDiscriminator::<AB>::new(&burn_rvc::RVC_V2_PERIODS, device);
    let disc_resume = resume.as_ref().map(Checkpoint::disc).filter(|p| p.exists());
    if let Some(p) = &disc_resume {
        let res = disc
            .load_safetensors(p)
            .map_err(|e| anyhow!("resuming D from {}: {e}", p.display()))?;
        burn_kit::check_coverage(&format!("D resume from {}", p.display()), &res)
            .map_err(anyhow::Error::msg)?;
        tracing::info!("resumed discriminator from {}", p.display());
    } else if let Some(p) = &req.pretrained_d {
        let res = disc
            .load_pytorch(p, Some("model"))
            .map_err(|e| anyhow!("loading D {}: {e}", p.display()))?;
        burn_kit::check_coverage(&format!("D warm-start from {}", p.display()), &res)
            .map_err(anyhow::Error::msg)?;
        tracing::info!("warm-started discriminator from {}", p.display());
    } else if resume.is_some() {
        tracing::warn!(
            "resuming without a discriminator checkpoint or --pretrained-d: \
             the discriminator starts fresh (adversarial training will lag)"
        );
    }

    // One per device: the STFT kernels are constants, but a tensor can only be
    // used on the device it lives on.
    let spectrals: Vec<Spectral<AB>> = devices
        .iter()
        .map(|d| Spectral::<AB>::new(&SpectralConfig::v2_48k(), d))
        .collect();

    let weight_decay = req.settings.weight_decay;
    // Read once, so the loop and the tensor shapes below cannot disagree about
    // them. `window_frames` is the clip-length floor as well as the context
    // width — `load_clips` was given the same value, and a shorter clip was
    // discarded there rather than truncated here.
    let window_frames = req.settings.window_frames;
    let segment_frames = req.settings.segment_frames;
    let mut opt_g = AdamWConfig::new()
        .with_beta_1(0.8)
        .with_beta_2(0.99)
        .with_epsilon(1e-9)
        .with_weight_decay(weight_decay)
        .init::<AB, Synthesizer<AB>>();
    let mut opt_d = AdamWConfig::new()
        .with_beta_1(0.8)
        .with_beta_2(0.99)
        .with_epsilon(1e-9)
        .with_weight_decay(weight_decay)
        .init::<AB, MultiPeriodDiscriminator<AB>>();
    let accum = req.settings.grad_accum.max(1);
    let d_lr_ratio = req.settings.d_lr_ratio;
    let d_interval = req.settings.d_interval.max(1);

    let batch = req.settings.batch_size.max(1);
    let total_frames: usize = clips.iter().map(|c| c.frames).sum();
    let steps_per_epoch = (total_frames / (batch * window_frames)).max(1);
    let total_steps = req.settings.epochs as usize * steps_per_epoch;

    // LR decay and EMA window, both as fractions of the whole run so they stay
    // sensible for any epoch count. `ema_frac == 0` disables the EMA.
    let sched = Schedule::new(
        req.settings.lr,
        req.settings.lr_final,
        req.settings.ema_frac,
        total_steps,
    );
    // Clip-sampling bias (None = uniform); computed once from each clip's SNR.
    let cdf = clip_weights(&clips, req.settings.snr_weight);
    tracing::info!(
        "run:   {total_steps} steps ({} epochs x {steps_per_epoch}), batch {batch}{}",
        req.settings.epochs,
        match accum {
            1 => String::new(),
            n => format!(" x{n} accum"),
        }
    );
    tracing::info!(
        "sched: lr {:.1e} -> {:.1e}, ema over {:.0} steps{}{}",
        req.settings.lr,
        sched.final_lr(),
        sched.ema_window,
        match d_lr_ratio {
            r if (r - 1.0).abs() < f64::EPSILON => String::new(),
            r => format!(", d-lr x{r}"),
        },
        match d_interval {
            1 => String::new(),
            n => format!(", d every {n} steps"),
        }
    );

    let mut rng = Rng::new(0x51D_u64.wrapping_mul(req.settings.epochs as u64 + 1));
    let sid = req.settings.speaker_id;
    let seg_len = segment_frames * HOP;

    // Generator weight EMA (kept on the inner backend): averaged over the
    // adversarial oscillation, so cleaner than any single step, and what every
    // checkpoint deploys. `None` when disabled (`--ema-frac 0`).
    let mut ema: Option<Synthesizer<AB::InnerBackend>> =
        (sched.ema_decay > 0.0).then(|| net_g.valid());

    let out = Checkpoint::new(&req.out);
    // Best-so-far checkpointing, on unless `--no-save-best`.
    let mut best = Best::new(
        &out,
        req.settings.save_best,
        total_steps,
        req.settings.best_window,
    );

    let mut dash = Dashboard::new(
        req.settings.use_tui,
        steps_per_epoch,
        req.settings.epochs as usize,
        &["g_loss", "d_loss", "mel_loss"],
    );

    // Burn initialises parameters lazily, and a replica cloned from a module
    // whose parameters have not materialised gets fresh `ParamId`s — its
    // gradients would then never match the master's and would be dropped in
    // silence. Warm-start and resume materialise on load, but training from
    // scratch does not, so force it before any replica is made.
    if devices.len() > 1 {
        materialize(&net_g);
        materialize(&disc);
    }

    let started = std::time::Instant::now();
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
        let cur_lr = sched.lr(step);
        let update_d = step % d_interval == 0;

        // Accumulate over `accum` micro-batches per device before one optimizer
        // step — steadier gradients without the VRAM of a larger real batch. The
        // effective batch is `batch * accum * devices`, and `inv` averages over
        // all of it.
        let micros = accum * devices.len();
        let inv = 1.0 / micros as f32;
        let mut acc_g = GradientsParams::new();
        let mut acc_d = GradientsParams::new();
        // Losses stay on the device that produced them until the whole step has
        // been dispatched. Reading one back is a hard sync, so summing them here
        // would make each device wait for the previous one and serialise exactly
        // what data parallelism is for.
        let mut g_losses = Vec::with_capacity(micros);
        let mut d_losses = Vec::with_capacity(micros);
        let mut mel_losses = Vec::with_capacity(micros);

        // Replicas of the current weights for the non-master devices. Rebuilt
        // each step because only the master is optimized; this clone-and-copy is
        // the cost of data parallelism without an all-reduce.
        let replicas: Vec<(Synthesizer<AB>, MultiPeriodDiscriminator<AB>)> = devices[1..]
            .iter()
            .map(|d| (net_g.clone().to_device(d), disc.clone().to_device(d)))
            .collect();

        let b = batch;
        for _ in 0..accum {
            // Every device gets its own micro-batch, so the effective batch is
            // `batch * accum * devices` — the usual data-parallel bargain.
            for (i, dev) in devices.iter().enumerate() {
                let data = sample_batch(&clips, b, window_frames, &mut rng, cdf.as_deref());
                let ids: Vec<usize> = (0..b)
                    .map(|_| rng.below(window_frames - segment_frames + 1))
                    .collect();
                // Device 0 is the master and owns the live weights; the rest work
                // on replicas made above. `replicas` is empty when N == 1, so that
                // path is exactly the single-device one, with no copy.
                let (g_ref, d_ref) = match i.checked_sub(1) {
                    None => (&net_g, &disc),
                    Some(r) => (&replicas[r].0, &replicas[r].1),
                };
                let out = micro_step(MicroIn {
                    net_g: g_ref,
                    disc: d_ref,
                    spectral: &spectrals[i],
                    device: dev,
                    data,
                    ids: &ids,
                    batch: b,
                    sid,
                    seg_len,
                    window_frames,
                    segment_frames,
                    update_d,
                });

                g_losses.push(out.g);
                mel_losses.push(out.mel);
                d_losses.extend(out.d);

                // Gradients come home to the master before they are summed; only
                // the master's optimizer state and weights ever advance.
                let g_grads = match i {
                    0 => out.g_grads,
                    _ => out.g_grads.to_device(main_device, &net_g),
                };
                acc_g = accumulate(acc_g, g_grads, &net_g, inv);
                if let Some(d_grads) = out.d_grads {
                    let d_grads = match i {
                        0 => d_grads,
                        _ => d_grads.to_device(main_device, &disc),
                    };
                    acc_d = accumulate(acc_d, d_grads, &disc, inv);
                }
            }
        }

        // ---- Optimizer steps (once per accumulated batch) -------------------
        if update_d {
            disc = opt_d.step(cur_lr * d_lr_ratio, disc, acc_d);
        }
        net_g = opt_g.step(cur_lr, net_g, acc_g);

        // Track the EMA of the just-updated generator weights.
        if let Some(e) = ema.take() {
            ema = Some(ema_update(e, &net_g, sched.ema_decay));
        }

        // The step's only sync, now that every device and both optimizers have
        // been dispatched. `last_d` keeps its previous value on the steps that
        // leave D alone, which is what the dashboard should show.
        let g_scalar = g_losses.into_iter().map(scalar).sum::<f32>() * inv;
        let mel_scalar = mel_losses.into_iter().map(scalar).sum::<f32>() * inv;
        if !d_losses.is_empty() {
            last_d = d_losses.into_iter().map(scalar).sum::<f32>() * inv;
        }
        // The dashboard also emits the throttled progress line, whether or not
        // the TUI is up.
        dash.update(step, &[g_scalar, last_d, mel_scalar], cur_lr);
        best.observe(step, mel_scalar, ema.as_ref(), &net_g, Some(&disc));
    }

    // Close the TUI (restores the terminal) before we log/save.
    dash.finish();
    if stopped_early {
        tracing::info!("stopped early; saving current weights");
    }

    best.finish(last_step, ema.as_ref(), &net_g, Some(&disc));

    out.save(ema.as_ref(), &net_g.valid(), Some(&disc.valid()))
        .context("saving trained weights")?;
    tracing::info!(
        "done:  {} steps in {}{}",
        last_step + 1,
        human(started.elapsed()),
        if stopped_early {
            " (stopped early)"
        } else {
            ""
        }
    );
    tracing::info!("       weights -> {}", out.generator().display());
    best.report();
    Ok(out.generator())
}

/// Everything one micro-batch needs. A struct because the list is long enough
/// that positional arguments stop being readable.
struct MicroIn<'a, AB: AutodiffBackend> {
    net_g: &'a Synthesizer<AB>,
    disc: &'a MultiPeriodDiscriminator<AB>,
    spectral: &'a Spectral<AB>,
    device: &'a AB::Device,
    data: crate::dataset::Batch,
    ids: &'a [usize],
    batch: usize,
    sid: i64,
    seg_len: usize,
    /// Carried per micro-batch rather than read from a constant, because both
    /// are now settings — and the tensor shapes below are built from them, so a
    /// stale copy is a shape mismatch rather than a wrong number.
    window_frames: usize,
    segment_frames: usize,
    update_d: bool,
}

/// What it produces: gradients on the device it ran on, plus the losses the
/// dashboard reports — still as device tensors, and detached from the autodiff
/// graph so holding them costs nothing. Reading them here would sync the device
/// before the next one is even dispatched.
struct MicroOut<AB: AutodiffBackend> {
    g_grads: GradientsParams,
    d_grads: Option<GradientsParams>,
    g: Tensor<AB::InnerBackend, 1>,
    d: Option<Tensor<AB::InnerBackend, 1>>,
    mel: Tensor<AB::InnerBackend, 1>,
}

/// One forward pass and both backward passes, entirely on `input.device`.
///
/// Split out of the step loop so the same code serves the master device and each
/// replica — data parallelism is then just "call this once per device and sum".
fn micro_step<AB: AutodiffBackend>(input: MicroIn<'_, AB>) -> MicroOut<AB> {
    let MicroIn {
        net_g,
        disc,
        spectral,
        device,
        data,
        ids,
        batch: b,
        sid,
        seg_len,
        window_frames,
        segment_frames,
        update_d,
    } = input;

    let phone = Tensor::<AB, 3>::from_data(
        TensorData::new(data.phone, [b, window_frames, CONTENT_DIM]),
        device,
    );
    let pitch =
        Tensor::<AB, 2, Int>::from_data(TensorData::new(data.coarse, [b, window_frames]), device);
    let nsff0 = Tensor::<AB, 2>::from_data(TensorData::new(data.nsff0, [b, window_frames]), device);
    let gt = Tensor::<AB, 2>::from_data(TensorData::new(data.gt, [b, window_frames * HOP]), device);

    // enc_q input spectrogram (a constant w.r.t. autodiff).
    let spec = spectral.linear(gt.clone()).detach();
    let tf = net_g.forward_train(phone, pitch, nsff0, spec, sid, ids, segment_frames);

    // Ground-truth audio segment matching `ids`.
    let mut segs = Vec::with_capacity(b);
    for (i, &s) in ids.iter().enumerate() {
        segs.push(gt.clone().narrow(0, i, 1).narrow(1, s * HOP, seg_len));
    }
    let y = Tensor::cat(segs, 0); // [b, seg_len]
    let y3 = y.clone().reshape([b, 1, seg_len]);
    let y_hat = tf.y_hat; // [b, 1, seg_len]

    // ---- Discriminator (fake detached) --------------------------------------
    let mut d = None;
    let d_grads = update_d.then(|| {
        let y_hat_d = y_hat.clone().detach();
        let mut d_loss = disc_loss(
            disc.scale.forward(y3.clone()).0,
            disc.scale.forward(y_hat_d.clone()).0,
        );
        for p in &disc.periods {
            d_loss = d_loss + disc_loss(p.forward(y3.clone()).0, p.forward(y_hat_d.clone()).0);
        }
        d = Some(d_loss.clone().inner());
        GradientsParams::from_grads(d_loss.backward(), disc)
    });

    // ---- Generator ----------------------------------------------------------
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
    let g_loss = g_adv + g_fm.mul_scalar(FM_WEIGHT) + mel_loss.clone() + kl_loss;

    MicroOut {
        g: g_loss.clone().inner(),
        mel: mel_loss.inner(),
        d,
        g_grads: GradientsParams::from_grads(g_loss.backward(), net_g),
        d_grads,
    }
}
