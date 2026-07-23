//! The RVC fine-tuning loop (Burn autodiff on wgpu).

use std::path::PathBuf;

use anyhow::{anyhow, Context, Result};
use burn::backend::wgpu::{Wgpu, WgpuDevice};
use burn::backend::Autodiff;
use burn::module::AutodiffModule;
use burn::optim::{AdamWConfig, GradientsParams, Optimizer};
use burn::tensor::{Int, Tensor, TensorData};
use burn_rvc::{MultiPeriodDiscriminator, Synthesizer, SynthesizerConfig};

use crate::dataset::{sample_batch, Clip, Rng, CONTENT_DIM, HOP};
use crate::losses::{disc_loss, feature_matching, gen_adv, kl, mel_l1};
use crate::spectral::{Spectral, SpectralConfig};
use crate::TrainRequest;

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

    // ---- Models (warm-started from the public pretrained bases) -------------
    let mut net_g = Synthesizer::<AB>::new(&cfg, &device);
    if let Some(p) = &req.pretrained_g {
        let res = net_g.load_pytorch(p).map_err(|e| anyhow!("loading G {}: {e}", p.display()))?;
        anyhow::ensure!(res.missing.is_empty(), "G warm-start incomplete: {} missing", res.missing.len());
        tracing::info!("warm-started generator from {}", p.display());
    } else {
        tracing::warn!("no --pretrained-g: training the generator from scratch (poor on small data)");
    }

    let mut disc = MultiPeriodDiscriminator::<AB>::new(&device);
    if let Some(p) = &req.pretrained_d {
        let res = disc.load_pytorch(p).map_err(|e| anyhow!("loading D {}: {e}", p.display()))?;
        anyhow::ensure!(res.missing.is_empty(), "D warm-start incomplete: {} missing", res.missing.len());
        tracing::info!("warm-started discriminator from {}", p.display());
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

    for step in 0..total_steps {
        let b = batch;
        let data = sample_batch(&clips, b, WINDOW_FRAMES, &mut rng);

        let phone = Tensor::<AB, 3>::from_data(
            TensorData::new(data.phone, [b, WINDOW_FRAMES, CONTENT_DIM]),
            &device,
        );
        let pitch =
            Tensor::<AB, 2, Int>::from_data(TensorData::new(data.coarse, [b, WINDOW_FRAMES]), &device);
        let nsff0 =
            Tensor::<AB, 2>::from_data(TensorData::new(data.nsff0, [b, WINDOW_FRAMES]), &device);
        let gt = Tensor::<AB, 2>::from_data(
            TensorData::new(data.gt, [b, WINDOW_FRAMES * HOP]),
            &device,
        );

        // enc_q input spectrogram (a constant w.r.t. autodiff).
        let spec = spectral.linear(gt.clone()).detach();

        // Random decode segment per sample.
        let ids: Vec<usize> = (0..b).map(|_| rng.below(WINDOW_FRAMES - SEGMENT_FRAMES + 1)).collect();

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
        let mut d_loss = disc_loss(disc.scale.forward(y3.clone()).0, disc.scale.forward(y_hat_d.clone()).0);
        for p in &disc.periods {
            d_loss = d_loss + disc_loss(p.forward(y3.clone()).0, p.forward(y_hat_d.clone()).0);
        }
        let d_scalar = scalar(&d_loss);
        let d_grads = GradientsParams::from_grads(d_loss.backward(), &disc);
        disc = opt_d.step(lr, disc, d_grads);

        // ---- Generator step -------------------------------------------------
        let mel_loss = mel_l1(spectral.mel(y.clone()), spectral.mel(y_hat.clone().reshape([b, seg_len])))
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

        if step % 20 == 0 || step + 1 == total_steps {
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

    // Save the fine-tuned generator (inference-backend weights).
    std::fs::create_dir_all(req.out.parent().unwrap_or_else(|| std::path::Path::new(".")))
        .ok();
    let out = req.out.with_extension("safetensors");
    net_g.valid()
        .save_safetensors(&out)
        .map_err(|e| anyhow!("saving {}: {e}", out.display()))
        .with_context(|| "saving trained weights")?;
    Ok(out)
}

/// Extract a scalar loss value to `f32` for logging.
fn scalar(t: &Tensor<AB, 1>) -> f32 {
    t.clone().into_data().to_vec::<f32>().map(|v| v.first().copied().unwrap_or(f32::NAN)).unwrap_or(f32::NAN)
}
