//! The RVC fine-tuning loop (Burn autodiff on wgpu).

use std::path::PathBuf;
use std::sync::atomic::Ordering;

use anyhow::{Context, Result, anyhow};
use burn::backend::Autodiff;
use burn::backend::wgpu::{Wgpu, WgpuDevice};
use burn::module::AutodiffModule;
use burn::optim::{AdamWConfig, GradientsParams, Optimizer};
use burn::tensor::{Int, Tensor, TensorData};
use burn_rvc::{MultiPeriodDiscriminator, Synthesizer, SynthesizerConfig};

use crate::TrainRequest;
use crate::dashboard::Dashboard;
use crate::dataset::{CONTENT_DIM, Clip, HOP, Rng, sample_batch};
use crate::losses::{disc_loss, feature_matching, gen_adv, kl, mel_l1};
use crate::spectral::{Spectral, SpectralConfig};

/// Autodiff GPU backend.
type AB = Autodiff<Wgpu>;

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
    let device = WgpuDevice::default();
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
    let lr = 1e-4;

    let batch = req.settings.batch_size.max(1);
    let total_frames: usize = clips.iter().map(|c| c.frames).sum();
    let steps_per_epoch = (total_frames / (batch * WINDOW_FRAMES)).max(1);
    let total_steps = req.settings.epochs as usize * steps_per_epoch;
    tracing::info!(
        "training: {} clips, batch {}, {} steps/epoch, {} epochs -> {} steps",
        clips.len(),
        batch,
        steps_per_epoch,
        req.settings.epochs,
        total_steps
    );

    let mut rng = Rng::new(0x51D_u64.wrapping_mul(req.settings.epochs as u64 + 1));
    let sid = req.settings.speaker_id;
    let seg_len = SEGMENT_FRAMES * HOP;

    let mut dash = Dashboard::new(
        req.settings.use_tui,
        total_steps,
        steps_per_epoch,
        req.settings.epochs as usize,
    );

    let mut stopped_early = false;
    for step in 0..total_steps {
        // Early stop: SIGINT (non-TUI) or `q` in the dashboard. The model saved
        // below reflects the last completed step.
        if req.stop.load(Ordering::Relaxed) || dash.interrupted() {
            stopped_early = true;
            break;
        }
        let b = batch;
        let data = sample_batch(&clips, b, WINDOW_FRAMES, &mut rng);

        let phone = Tensor::<AB, 3>::from_data(
            TensorData::new(data.phone, [b, WINDOW_FRAMES, CONTENT_DIM]),
            &device,
        );
        let pitch = Tensor::<AB, 2, Int>::from_data(
            TensorData::new(data.coarse, [b, WINDOW_FRAMES]),
            &device,
        );
        let nsff0 =
            Tensor::<AB, 2>::from_data(TensorData::new(data.nsff0, [b, WINDOW_FRAMES]), &device);
        let gt =
            Tensor::<AB, 2>::from_data(TensorData::new(data.gt, [b, WINDOW_FRAMES * HOP]), &device);

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

        // ---- Discriminator step (fake detached) -----------------------------
        let y_hat_d = y_hat.clone().detach();
        let mut d_loss = disc_loss(
            disc.scale.forward(y3.clone()).0,
            disc.scale.forward(y_hat_d.clone()).0,
        );
        for p in &disc.periods {
            d_loss = d_loss + disc_loss(p.forward(y3.clone()).0, p.forward(y_hat_d.clone()).0);
        }
        let d_scalar = scalar(&d_loss);
        let d_grads = GradientsParams::from_grads(d_loss.backward(), &disc);
        disc = opt_d.step(lr, disc, d_grads);

        // ---- Generator step -------------------------------------------------
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
        let g_scalar = scalar(&g_loss);
        let mel_scalar = scalar(&mel_loss);
        let g_grads = GradientsParams::from_grads(g_loss.backward(), &net_g);
        net_g = opt_g.step(lr, net_g, g_grads);

        dash.update(step, g_scalar, d_scalar, mel_scalar);
        if !dash.is_active() && (step % 20 == 0 || step + 1 == total_steps) {
            tracing::info!(
                "step {}/{}  g={:.3} d={:.3} mel={:.3}",
                step,
                total_steps,
                g_scalar,
                d_scalar,
                mel_scalar
            );
        }
    }

    // Close the TUI (restores the terminal) before we log/save.
    dash.finish();
    if stopped_early {
        tracing::info!("stopped early; saving current weights");
    }

    // Save the fine-tuned generator (inference-backend weights).
    std::fs::create_dir_all(
        req.out
            .parent()
            .unwrap_or_else(|| std::path::Path::new(".")),
    )
    .ok();
    let out = req.out.with_extension("safetensors");
    net_g
        .valid()
        .save_safetensors(&out)
        .map_err(|e| anyhow!("saving {}: {e}", out.display()))
        .with_context(|| "saving trained weights")?;

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
