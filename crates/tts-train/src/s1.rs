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

use burn::module::{AutodiffModule, Module};
use burn::optim::{AdamWConfig, GradientsParams, Optimizer};
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::{Int, Tensor, TensorData};
use burn_gptsovits::{T2s, T2sConfig};
use train_kit::{Checkpoint, Dashboard, Schedule, ema_update, human, scalar};

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
///
/// `devices[0]` is the master: it owns the weights, the optimizer and the EMA,
/// and the others only ever produce gradients for it. A clip is the unit of
/// work, so with N devices each step processes N clips at once.
pub fn run<AB: AutodiffBackend>(
    s1_path: &std::path::Path,
    clips: &[Clip],
    settings: &S1Settings,
    out: &Checkpoint,
    stop: &std::sync::atomic::AtomicBool,
    devices: &[AB::Device],
) -> Result<std::path::PathBuf> {
    if devices.is_empty() {
        return Err(crate::error::TrainError::Device(
            "no compute device selected".into(),
        ));
    }
    let device = &devices[0];
    let cfg = T2sConfig::default();
    let mut model = T2s::<AB>::new(&cfg, device);
    let applied = model
        .load_weights(s1_path)
        .map_err(|e| crate::error::TrainError::Weights(format!("s1: {e}")))?;
    // The loader allows a partial apply so a coverage report can be inspected,
    // which makes an empty one a success unless somebody checks — and an
    // unchecked warm-start silently trains a randomly initialised model for
    // however long the run lasts. `s2` refuses the same way.
    if !applied.errors.is_empty() || !applied.missing.is_empty() {
        return Err(crate::error::TrainError::Weights(format!(
            "s1 warm-start incomplete ({} missing, {} failed to apply): {} is \
             not a full T2S checkpoint",
            applied.missing.len(),
            applied.errors.len(),
            s1_path.display()
        )));
    }

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

    // A step consumes `batch_size` clips per device, so more devices make each
    // step wider rather than the run longer.
    let per_step = settings.batch_size.max(1) * devices.len();
    let steps_per_epoch = usable.len().div_ceil(per_step);
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

    // The EMA is what gets deployed, as in the adversarial loops: it is markedly
    // steadier than the live weights on a corpus this small. The LR decay comes
    // from the same schedule they use — a small corpus overfits fast at a flat
    // rate.
    let sched = Schedule::new(
        settings.lr,
        settings.lr_final,
        settings.ema_frac,
        total_steps,
    );
    let mut ema = (sched.ema_decay > 0.0).then(|| model.valid());

    // Burn allocates parameters lazily, and a replica cloned before they
    // materialise gets fresh `ParamId`s — its gradients would then never match
    // the master's and would be dropped in silence, leaving every extra device
    // contributing nothing while the run looked healthy.
    if devices.len() > 1 {
        train_kit::materialize(&model);
    }

    let started = std::time::Instant::now();
    let mut step = 0usize;
    let mut stopped_early = false;

    'training: for epoch in 0..settings.epochs {
        for chunk in usable.chunks(per_step) {
            if stop.load(std::sync::atomic::Ordering::Relaxed) || dash.interrupted() {
                tracing::info!("stopping early at step {step}");
                stopped_early = true;
                break 'training;
            }

            let lr = sched.lr(step);

            // Replicas for the non-master devices, rebuilt each step because only
            // the master is optimized. Empty when there is one device, so that
            // path is exactly the single-device one with no copy.
            let replicas: Vec<T2s<AB>> = devices[1..]
                .iter()
                .map(|d| model.clone().to_device(d))
                .collect();

            let mut grads = GradientsParams::new();
            let mut total = 0.0f32;
            let inv = 1.0 / chunk.len() as f32;
            for (i, clip) in chunk.iter().enumerate() {
                // Round-robin across devices, so a chunk of N*batch clips is N
                // independent forward/backward passes running side by side.
                let d = i % devices.len();
                let m = match d.checked_sub(1) {
                    None => &model,
                    Some(r) => &replicas[r],
                };
                let loss = clip_loss(m, clip, &devices[d]);
                total += scalar(loss.clone().inner());
                let g = GradientsParams::from_grads(loss.backward(), m);
                // Gradients come home to the master before they are summed; only
                // the master's optimizer state and weights ever advance.
                let g = match d {
                    0 => g,
                    _ => g.to_device(device, &model),
                };
                grads = train_kit::accumulate(grads, g, &model, inv);
            }
            model = optim.step(lr, model, grads);

            if let Some(current) = ema.take() {
                ema = Some(ema_update(current, &model, sched.ema_decay));
            }

            // The dashboard also emits the throttled progress line, whether or
            // not the TUI is up.
            dash.update(step, &[total * inv], lr);
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
