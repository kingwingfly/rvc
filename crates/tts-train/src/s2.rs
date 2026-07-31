//! Fine-tuning `s2`, the SoVITS waveform stage.
//!
//! This is what adapts *timbre*. Until it exists a cloned voice's colour comes
//! entirely from the reference clip's speaker vector, which is a few seconds of
//! audio averaged into one 512-wide vector — enough to approximate a voice, not
//! enough to be it. `s1` (see [`crate::s1`]) adapts the other half, delivery.
//!
//! The loop is `rvc-train`'s VITS GAN with the RVC-specific parts swapped out,
//! because both models are the same architecture from the same source: the same
//! four losses in the same proportions (mel-L1 x45, KL x1, feature-matching x2,
//! LSGAN), the same D-before-G interleaving, the same EMA and decaying schedule.
//!
//! What differs is the unit of work. `rvc-train` samples fixed 0.48 s windows,
//! which it can because RVC's conditioning is frame-aligned content features
//! with no sequence structure. Here the conditioning is *text*, attended to
//! through the MRTE cross-attention, so a window of frames no longer matches the
//! phonemes it was transcribed from. Each micro-batch is therefore one whole
//! utterance — `enc_p`, `enc_q` and the flow see all of it, and only the decoder
//! runs on a random segment, exactly as upstream's `rand_slice_segments` does.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};

use burn::module::{AutodiffModule, Module};
use burn::optim::{AdamWConfig, GradientsParams, Optimizer};
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::{Int, Tensor, TensorData};
use burn_gptsovits::{GPTSOVITS_V2_PERIODS, SovitsConfig, SovitsPartial};
use burn_vits::{
    MultiPeriodDiscriminator, Spectral, SpectralConfig, disc_loss, feature_matching, gen_adv, kl,
    mel_l1,
};
use train_kit::{
    Best, Checkpoint, Dashboard, Rng, Schedule, accumulate, ema_update, human, materialize, scalar,
};

use crate::dataset::{Clip, SAMPLES_PER_FRAME};
use crate::error::{Result, TrainError};

/// Loss weights, matching RVC and upstream GPT-SoVITS alike — the two inherited
/// them from VITS together.
const C_MEL: f64 = 45.0;
const C_KL: f64 = 1.0;
const FM_WEIGHT: f64 = 2.0;

/// What a fine-tune is asked to do.
#[derive(Debug, Clone)]
pub struct S2Settings {
    pub epochs: u32,
    /// Clips per optimizer step, per device. They are processed one at a time
    /// and the gradients accumulated — utterances differ in length and this loop
    /// does not pad, so this *is* the gradient accumulation knob rather than a
    /// separate one.
    pub batch_size: usize,
    pub lr: f64,
    /// End-of-run learning rate as a fraction of `lr`.
    pub lr_final: f64,
    /// EMA window as a fraction of the run; `0` saves the raw weights.
    pub ema_frac: f64,
    pub use_tui: bool,
    /// Discriminator learning rate as a multiple of the generator's. Below 1
    /// holds off a discriminator that is winning.
    pub d_lr_ratio: f64,
    /// Update the discriminator every N steps. Above 1 is the coarser version of
    /// the same lever.
    pub d_interval: usize,
    /// Latent frames the decoder renders per step. 32 frames is 0.64 s at the
    /// 640-sample hop, which is what upstream's 20480-sample segment comes to.
    pub segment_frames: usize,
    /// Skip clips longer than this many latent frames. `enc_q` and the flow run
    /// over the whole utterance, so memory grows with the longest clip rather
    /// than with the batch — this is the cap that keeps a 6 GB card alive.
    pub max_frames: usize,
    /// Track a best-so-far checkpoint by mean mel loss.
    pub save_best: bool,
}

impl Default for S2Settings {
    fn default() -> Self {
        Self {
            epochs: 10,
            batch_size: 1,
            lr: 1e-4,
            lr_final: 0.1,
            ema_frac: 0.1,
            use_tui: true,
            d_lr_ratio: 1.0,
            d_interval: 1,
            segment_frames: 32,
            max_frames: 1600,
            save_best: true,
        }
    }
}

/// Fine-tune `s2` on a prepared corpus, returning where the weights were saved.
///
/// `s2` is the generator to warm-start from — an upstream `s2G*.pth` or a
/// `.safetensors` from an earlier run. `s2d` is the matching discriminator; its
/// absence is survivable but not advised, since a fresh adversary spends the
/// early steps learning what real audio is instead of critiquing this voice.
pub fn run<AB: AutodiffBackend>(
    s2: &Path,
    s2d: Option<&Path>,
    clips: &[Clip],
    settings: &S2Settings,
    out: &Checkpoint,
    stop: &AtomicBool,
    devices: &[AB::Device],
) -> Result<PathBuf> {
    if devices.is_empty() {
        return Err(TrainError::Device("no compute device selected".into()));
    }
    // Device 0 owns the weights, the optimizers and the EMA; the rest only ever
    // produce gradients.
    let device = &devices[0];
    let weights =
        |what: &str, e: Box<dyn std::error::Error>| TrainError::Weights(format!("{what}: {e}"));

    let cfg = SovitsConfig::default();
    let mut net_g = SovitsPartial::<AB>::new(&cfg, device);
    let applied = net_g.load_weights(s2).map_err(|e| weights("s2", e))?;
    if !applied.missing.is_empty() {
        return Err(TrainError::Weights(format!(
            "s2 warm-start incomplete ({} missing): {} is not a full generator checkpoint",
            applied.missing.len(),
            s2.display()
        )));
    }
    tracing::info!("warm-started s2 from {}", s2.display());

    let mut disc = MultiPeriodDiscriminator::<AB>::new(&GPTSOVITS_V2_PERIODS, device);
    match s2d {
        Some(p) => {
            // Dispatch on extension the way the generator does: an upstream
            // `s2D*.pth` on the first run, our own sidecar when resuming.
            let res = match p.extension().and_then(|e| e.to_str()) {
                Some("safetensors") => disc.load_safetensors(p),
                // GPT-SoVITS keeps the discriminator under `weight`; RVC keeps
                // it under `model`, and passing the wrong one applies nothing.
                _ => disc.load_pytorch(p, Some("weight")),
            }
            .map_err(|e| weights("s2 discriminator", e))?;
            if !res.missing.is_empty() {
                return Err(TrainError::Weights(format!(
                    "discriminator warm-start incomplete: {} missing from {}",
                    res.missing.len(),
                    p.display()
                )));
            }
            tracing::info!("warm-started the discriminator from {}", p.display());
        }
        None => tracing::warn!(
            "no s2 discriminator found: the adversary starts from scratch, so the \
             early steps teach it what speech is rather than what this voice is"
        ),
    }

    // Usable clips: long enough to slice a decoder segment from, short enough
    // that the full-sequence encoders fit.
    let usable: Vec<&Clip> = clips
        .iter()
        .filter(|c| {
            let frames = c.frames();
            frames >= settings.segment_frames && frames <= settings.max_frames
        })
        .collect();
    if usable.is_empty() {
        return Err(TrainError::Corpus(format!(
            "no clip is between {} and {} latent frames ({:.2}-{:.0} s) — \
             nothing to train s2 on",
            settings.segment_frames,
            settings.max_frames,
            settings.segment_frames as f32 / 50.0,
            settings.max_frames as f32 / 50.0,
        )));
    }
    if usable.iter().any(|c| c.audio.is_empty()) {
        return Err(TrainError::Corpus(
            "s2 needs the corpus prepared with waveforms; this one has none".into(),
        ));
    }
    let skipped = clips.len() - usable.len();
    if skipped > 0 {
        tracing::warn!("{skipped} clips outside the frame bounds were left out");
    }

    // One per device: the STFT kernels are constants, but a tensor can only be
    // used on the device it lives on.
    let spectrals: Vec<Spectral<AB>> = devices
        .iter()
        .map(|d| Spectral::<AB>::new(&SpectralConfig::gptsovits_v2_32k(), d))
        .collect();

    let adam = || {
        AdamWConfig::new()
            .with_beta_1(0.8)
            .with_beta_2(0.99)
            .with_epsilon(1e-9)
            .with_weight_decay(0.01)
    };
    let mut opt_g = adam().init::<AB, SovitsPartial<AB>>();
    let mut opt_d = adam().init::<AB, MultiPeriodDiscriminator<AB>>();

    let batch = settings.batch_size.max(1);
    let d_interval = settings.d_interval.max(1);
    let steps_per_epoch = usable.len().div_ceil(batch * devices.len()).max(1);
    let total_steps = steps_per_epoch * settings.epochs as usize;

    // LR decay and EMA window, both fractions of the run so they stay sensible
    // whatever the epoch count.
    let sched = Schedule::new(
        settings.lr,
        settings.lr_final,
        settings.ema_frac,
        total_steps,
    );
    let mut ema: Option<SovitsPartial<AB::InnerBackend>> =
        (sched.ema_decay > 0.0).then(|| net_g.valid());

    tracing::info!(
        "fine-tuning s2 on {} clips ({:.1} min), {total_steps} steps ({} epochs x {steps_per_epoch})",
        usable.len(),
        usable.iter().map(|c| c.seconds()).sum::<f32>() / 60.0,
        settings.epochs,
    );
    tracing::info!(
        "sched: lr {:.1e} -> {:.1e}, ema over {:.0} steps, segment {} frames ({:.2} s)",
        settings.lr,
        sched.final_lr(),
        sched.ema_window,
        settings.segment_frames,
        settings.segment_frames as f32 / 50.0,
    );

    let mut dash = Dashboard::new(
        settings.use_tui,
        steps_per_epoch,
        settings.epochs as usize,
        &["g_loss", "d_loss", "mel_loss"],
    );

    // Burn allocates parameters lazily, and a replica cloned from a module whose
    // parameters have not materialised gets fresh `ParamId`s — its gradients
    // would then never match the master's and would be dropped in silence, so
    // every extra device would contribute nothing while the run looked healthy.
    // Loading weights materialises, but do not rely on that from here.
    if devices.len() > 1 {
        materialize(&net_g);
        materialize(&disc);
    }

    let mut best = Best::new(out, settings.save_best, total_steps);

    let mut rng = Rng::new(0x505_1725_u64.wrapping_mul(settings.epochs as u64 + 1));
    let started = std::time::Instant::now();
    let mut stopped_early = false;
    let mut last_d = 0.0f32;
    let mut last_step = 0usize;

    for step in 0..total_steps {
        if stop.load(Ordering::Relaxed) || dash.interrupted() {
            stopped_early = true;
            break;
        }
        last_step = step;
        let cur_lr = sched.lr(step);
        let update_d = step % d_interval == 0;

        let micros = batch * devices.len();
        let inv = 1.0 / micros as f32;
        let mut acc_g = GradientsParams::new();
        let mut acc_d = GradientsParams::new();
        // Losses stay on the device that produced them until the whole step has
        // been dispatched: reading one back is a hard sync, so summing here would
        // make each device wait for the previous and serialise exactly what data
        // parallelism is for.
        let mut g_losses = Vec::with_capacity(micros);
        let mut d_losses = Vec::with_capacity(micros);
        let mut mel_losses = Vec::with_capacity(micros);

        // Replicas of the current weights for the non-master devices, rebuilt
        // each step because only the master is optimized. Empty when there is
        // one device, so that path is exactly the single-device one with no copy.
        let replicas: Vec<(SovitsPartial<AB>, MultiPeriodDiscriminator<AB>)> = devices[1..]
            .iter()
            .map(|d| (net_g.clone().to_device(d), disc.clone().to_device(d)))
            .collect();

        for _ in 0..batch {
            for (i, dev) in devices.iter().enumerate() {
                let clip = usable[rng.below(usable.len())];
                let (g_ref, d_ref) = match i.checked_sub(1) {
                    None => (&net_g, &disc),
                    Some(r) => (&replicas[r].0, &replicas[r].1),
                };
                let out = micro_step(MicroIn {
                    net_g: g_ref,
                    disc: d_ref,
                    spectral: &spectrals[i],
                    device: dev,
                    clip,
                    segment_frames: settings.segment_frames,
                    update_d,
                    rng: &mut rng,
                });

                g_losses.push(out.g);
                mel_losses.push(out.mel);
                d_losses.extend(out.d);

                // Gradients come home to the master before they are summed; only
                // the master's optimizer state and weights ever advance.
                let g_grads = match i {
                    0 => out.g_grads,
                    _ => out.g_grads.to_device(device, &net_g),
                };
                acc_g = accumulate(acc_g, g_grads, &net_g, inv);
                if let Some(d_grads) = out.d_grads {
                    let d_grads = match i {
                        0 => d_grads,
                        _ => d_grads.to_device(device, &disc),
                    };
                    acc_d = accumulate(acc_d, d_grads, &disc, inv);
                }
            }
        }

        if update_d {
            disc = opt_d.step(cur_lr * settings.d_lr_ratio, disc, acc_d);
        }
        net_g = opt_g.step(cur_lr, net_g, acc_g);

        if let Some(e) = ema.take() {
            ema = Some(ema_update(e, &net_g, sched.ema_decay));
        }

        // The step's only sync, now that every device and both optimizers have
        // been dispatched. `last_d` keeps its previous value on steps that leave
        // D alone, which is what the dashboard should show.
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

    dash.finish();
    if stopped_early {
        tracing::info!("stopped early; saving current weights");
    }

    best.finish(last_step, ema.as_ref(), &net_g, Some(&disc));

    out.save(ema.as_ref(), &net_g.valid(), Some(&disc.valid()))
        .map_err(|e| TrainError::Weights(e.to_string()))?;
    tracing::info!(
        "done: {} steps in {}{}",
        last_step + 1,
        human(started.elapsed()),
        match stopped_early {
            true => " (stopped early)",
            false => "",
        }
    );
    let path = out.generator();
    tracing::info!("weights -> {}", path.display());
    best.report();
    Ok(path)
}

/// Everything one micro-batch needs.
struct MicroIn<'a, AB: AutodiffBackend> {
    net_g: &'a SovitsPartial<AB>,
    disc: &'a MultiPeriodDiscriminator<AB>,
    spectral: &'a Spectral<AB>,
    device: &'a AB::Device,
    clip: &'a Clip,
    segment_frames: usize,
    update_d: bool,
    rng: &'a mut Rng,
}

/// What it produces: gradients on the device it ran on, plus the losses the
/// dashboard reports — still as device tensors and detached from the autodiff
/// graph, so holding them costs nothing. Reading them here would sync the device
/// before the next one is even dispatched.
struct MicroOut<AB: AutodiffBackend> {
    g_grads: GradientsParams,
    d_grads: Option<GradientsParams>,
    g: Tensor<AB::InnerBackend, 1>,
    d: Option<Tensor<AB::InnerBackend, 1>>,
    mel: Tensor<AB::InnerBackend, 1>,
}

/// One utterance: a full forward pass and both backward passes, entirely on
/// `input.device`.
///
/// Split out of the step loop so the same code serves the master device and each
/// replica — data parallelism is then just "call this once per device and sum".
fn micro_step<AB: AutodiffBackend>(input: MicroIn<'_, AB>) -> MicroOut<AB> {
    let MicroIn {
        net_g,
        disc,
        spectral,
        device,
        clip,
        segment_frames,
        update_d,
        rng,
    } = input;

    let frames = clip.frames();
    let samples = frames * SAMPLES_PER_FRAME;
    let seg_len = segment_frames * SAMPLES_PER_FRAME;
    let start = rng.below(frames - segment_frames + 1);

    let gt = Tensor::<AB, 2>::from_data(
        TensorData::new(clip.audio[..samples].to_vec(), [1, samples]),
        device,
    );
    let codes = Tensor::<AB, 2, Int>::from_data(
        TensorData::new(
            clip.tokens[..frames / 2]
                .iter()
                .map(|&t| t as i32)
                .collect::<Vec<_>>(),
            [1, frames / 2],
        ),
        device,
    );
    let text = Tensor::<AB, 2, Int>::from_data(
        TensorData::new(
            clip.phones.iter().map(|&p| p as i32).collect::<Vec<_>>(),
            [1, clip.phones.len()],
        ),
        device,
    );

    // The posterior encoder's input is a constant w.r.t. autodiff: it is the
    // ground truth, not something the generator produced.
    let spec = spectral.linear(gt.clone()).detach();
    let tf = net_g.forward_train(codes, text, spec, &[start], segment_frames);

    let y = gt.narrow(1, start * SAMPLES_PER_FRAME, seg_len); // [1, seg_len]
    let y3 = y.clone().reshape([1, 1, seg_len]);
    let y_hat = tf.y_hat; // [1, 1, seg_len]

    // ---- Discriminator (fake detached) --------------------------------------
    let mut d = None;
    let d_grads = update_d.then(|| {
        let y_hat_d = y_hat.clone().detach();
        let mut loss = disc_loss(
            disc.scale.forward(y3.clone()).0,
            disc.scale.forward(y_hat_d.clone()).0,
        );
        for p in &disc.periods {
            loss = loss + disc_loss(p.forward(y3.clone()).0, p.forward(y_hat_d.clone()).0);
        }
        d = Some(loss.clone().inner());
        GradientsParams::from_grads(loss.backward(), disc)
    });

    // ---- Generator ----------------------------------------------------------
    let mel_loss = mel_l1(
        spectral.mel(y.clone()),
        spectral.mel(y_hat.clone().reshape([1, seg_len])),
    )
    .mul_scalar(C_MEL);
    let kl_loss = kl(tf.z_p, tf.logs_q, tf.m_p, tf.logs_p).mul_scalar(C_KL);

    let (sf_s, ff_s) = disc.scale.forward(y_hat.clone());
    let (_, fr_s) = disc.scale.forward(y3.clone());
    let mut g_adv = gen_adv(sf_s);
    let mut g_fm = feature_matching(&fr_s, &ff_s);
    for p in &disc.periods {
        let (sf, ff) = p.forward(y_hat.clone());
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
