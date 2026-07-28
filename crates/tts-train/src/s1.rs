//! Fine-tuning `s1`, the text-to-semantic transformer.
//!
//! Plain next-token prediction: the model is shown a clip's phonemes followed by
//! its semantic tokens, and asked to predict each token from everything before
//! it. This is what adapts *delivery* — pacing, emphasis, where a speaker
//! breathes — because those are properties of the token sequence rather than of
//! the waveform `s2` renders.
//!
//! Simpler than `rvc-train`'s GAN in every way that matters: one model, one
//! optimizer, one loss, and a number that means something on its own.

use burn::module::AutodiffModule;
use burn::optim::{AdamWConfig, GradientsParams, Optimizer};
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::{Int, Tensor, TensorData};
use burn_gptsovits::{T2s, T2sConfig};
use train_kit::{Checkpoint, Dashboard, ema_update, human, scalar};

use crate::dataset::Clip;
use crate::error::Result;

/// What a fine-tune is asked to do.
#[derive(Debug, Clone)]
pub struct S1Settings {
    pub epochs: u32,
    /// Clips per optimizer step. One is the honest default: clips differ in
    /// length and this loop does not pad, so a batch is processed one at a time
    /// and accumulated.
    pub batch_size: usize,
    pub lr: f64,
    /// End-of-run learning rate as a fraction of `lr`.
    pub lr_final: f64,
    /// EMA window as a fraction of the run; `0` saves the raw weights.
    pub ema_frac: f64,
    pub use_tui: bool,
    /// Longest token sequence to train on. A clip past this is skipped rather
    /// than truncated — a cut-off sequence teaches the model to stop early.
    pub max_tokens: usize,
}

impl Default for S1Settings {
    fn default() -> Self {
        Self {
            epochs: 10,
            batch_size: 1,
            lr: 1e-5,
            lr_final: 0.1,
            ema_frac: 0.1,
            use_tui: true,
            max_tokens: 1500,
        }
    }
}

/// Fine-tune `s1` on a prepared corpus, returning where the weights were saved.
pub fn run<AB: AutodiffBackend>(
    s1_path: &std::path::Path,
    clips: &[Clip],
    settings: &S1Settings,
    out: &Checkpoint,
    device: &AB::Device,
) -> Result<std::path::PathBuf> {
    let cfg = T2sConfig::default();
    let mut model = T2s::<AB>::new(&cfg, device);
    model
        .load_pytorch(s1_path)
        .map_err(|e| crate::error::TrainError::Weights(format!("s1: {e}")))?;

    let usable: Vec<&Clip> = clips
        .iter()
        .filter(|c| c.tokens.len() <= settings.max_tokens && !c.tokens.is_empty())
        .collect();
    if usable.is_empty() {
        return Err(crate::error::TrainError::Corpus(format!(
            "every clip is longer than --max-tokens ({}) or empty",
            settings.max_tokens
        )));
    }
    let skipped = clips.len() - usable.len();
    if skipped > 0 {
        tracing::warn!("{skipped} clips over the token cap were left out");
    }

    let steps_per_epoch = usable.len().div_ceil(settings.batch_size.max(1));
    let total_steps = steps_per_epoch * settings.epochs as usize;
    tracing::info!(
        "fine-tuning s1 on {} clips ({:.1} min), {total_steps} steps",
        usable.len(),
        usable.iter().map(|c| c.seconds()).sum::<f32>() / 60.0
    );

    let mut optim = AdamWConfig::new().init();
    let mut dash = Dashboard::new(
        settings.use_tui,
        steps_per_epoch,
        settings.epochs as usize,
        &["loss"],
    );

    // The EMA is what gets deployed, as in `rvc-train`: it is markedly steadier
    // than the live weights on a corpus this small.
    let ema_keep = ema_keep(settings.ema_frac, total_steps);
    let mut ema = ema_keep.map(|_| model.valid());

    let started = std::time::Instant::now();
    let mut step = 0usize;
    let mut stopped_early = false;

    'training: for epoch in 0..settings.epochs {
        for chunk in usable.chunks(settings.batch_size.max(1)) {
            if dash.interrupted() {
                tracing::info!("stopping early at step {step}");
                stopped_early = true;
                break 'training;
            }

            // Exponential decay across the whole run, the same shape `rvc-train`
            // uses — a small corpus overfits fast at a flat rate.
            let progress = step as f64 / total_steps.max(1) as f64;
            let lr = settings.lr * settings.lr_final.powf(progress);

            let mut grads = GradientsParams::new();
            let mut total = 0.0f32;
            for clip in chunk {
                let loss = clip_loss(&model, clip, device);
                total += scalar(loss.clone().inner());
                let g = GradientsParams::from_grads(loss.backward(), &model);
                grads = train_kit::accumulate(grads, g, &model, 1.0 / chunk.len() as f32);
            }
            model = optim.step(lr, model, grads);

            if let (Some(keep), Some(current)) = (ema_keep, ema.take()) {
                ema = Some(ema_update(current, &model, keep));
            }

            let mean = total / chunk.len() as f32;
            dash.update(step, &[mean], lr);
            if step % 20 == 0 || step + 1 == total_steps {
                tracing::info!(
                    "{step:>6}/{total_steps} {:>3.0}%  loss {mean:8.4}  lr {lr:.1e}  eta {}",
                    100.0 * step as f64 / total_steps.max(1) as f64,
                    human(eta(started.elapsed(), step, total_steps)),
                );
            }
            step += 1;
        }
        tracing::debug!("epoch {} of {}", epoch + 1, settings.epochs);
    }

    dash.finish();
    tracing::info!(
        "done: {step} steps in {}{}",
        human(started.elapsed()),
        if stopped_early {
            " (stopped early)"
        } else {
            ""
        }
    );

    // `s1` has no adversary, so the discriminator slot goes empty.
    out.save::<AB::InnerBackend, _, T2s<AB::InnerBackend>>(ema.as_ref(), &model.valid(), None)
        .map_err(|e| crate::error::TrainError::Weights(e.to_string()))?;
    let path = out.generator();
    tracing::info!("weights -> {}", path.display());
    Ok(path)
}

/// Cross-entropy over one clip.
///
/// The target is the token sequence shifted by one with `EOS` appended, so every
/// position predicts what comes next and the last predicts the end. Teaching the
/// stop is the whole reason `EOS` is in the vocabulary.
fn clip_loss<AB: AutodiffBackend>(
    model: &T2s<AB>,
    clip: &Clip,
    device: &AB::Device,
) -> Tensor<AB, 1> {
    let cfg = T2sConfig::default();
    let n_phones = clip.phones.len();
    let n_tokens = clip.tokens.len();

    let phones: Tensor<AB, 2, Int> = Tensor::from_data(
        TensorData::new(
            clip.phones.iter().map(|&i| i as i32).collect::<Vec<_>>(),
            [1, n_phones],
        ),
        device,
    );
    let bert: Tensor<AB, 3> = Tensor::from_data(
        TensorData::new(clip.bert.clone(), [1, n_phones, clip.bert_dim]),
        device,
    );
    let tokens: Tensor<AB, 2, Int> = Tensor::from_data(
        TensorData::new(
            clip.tokens.iter().map(|&t| t as i32).collect::<Vec<_>>(),
            [1, n_tokens],
        ),
        device,
    );

    let text = model.embed_text(phones, bert);
    let audio = model.embed_audio(tokens, 0);

    let logits = model.forward_prompt_all(text, audio);
    // One row per audio position: what each predicts about the next.
    let targets: Vec<i32> = clip.tokens[1..]
        .iter()
        .map(|&t| t as i32)
        .chain([cfg.eos() as i32])
        .collect();
    let targets: Tensor<AB, 1, Int> =
        Tensor::from_data(TensorData::new(targets, [n_tokens]), device);

    cross_entropy(logits, targets)
}

/// Mean cross-entropy of `logits` `[positions, vocab]` against `targets`.
///
/// Mean rather than upstream's sum: a sum makes the gradient scale with clip
/// length, so long clips would dominate a corpus of mixed lengths and the
/// learning rate would mean something different for every one.
fn cross_entropy<B: burn::tensor::backend::Backend>(
    logits: Tensor<B, 2>,
    targets: Tensor<B, 1, Int>,
) -> Tensor<B, 1> {
    let [n, _] = logits.dims();
    let log_probs = burn::tensor::activation::log_softmax(logits, 1);
    let picked = log_probs.gather(1, targets.reshape([n, 1]));
    -picked.mean()
}

/// How much of the EMA to keep each step, or `None` when it is disabled.
fn ema_keep(frac: f64, total_steps: usize) -> Option<f64> {
    if frac <= 0.0 || total_steps == 0 {
        return None;
    }
    let window = (frac * total_steps as f64).max(1.0);
    Some(1.0 - 1.0 / window)
}

fn eta(elapsed: std::time::Duration, step: usize, total: usize) -> std::time::Duration {
    if step == 0 {
        return std::time::Duration::ZERO;
    }
    let per_step = elapsed.as_secs_f64() / step as f64;
    std::time::Duration::from_secs_f64(per_step * (total.saturating_sub(step)) as f64)
}
