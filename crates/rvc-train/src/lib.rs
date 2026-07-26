//! Native RVC generator fine-tuning (Rust + Burn).
//!
//! Pipeline: decode the corpus and extract ContentVec + RMVPE features (reusing
//! the ONNX extractors the inference path uses), then adversarially fine-tune
//! the [`burn_rvc`] generator on the GPU (CUDA), warm-started from the public
//! pretrained bases, and save the result as safetensors. Deploy either directly
//! (`rvc convert --backend burn`) or via the ONNX export helper.

mod dashboard;
mod dataset;
mod losses;
mod spectral;
mod trainer;

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use anyhow::{Context, Result};

/// Generator hyperparameters.
#[derive(Debug, Clone)]
pub struct TrainSettings {
    /// Generator output sample rate (48000 supported).
    pub sample_rate: u32,
    /// Number of training epochs.
    pub epochs: u32,
    /// Mini-batch size.
    pub batch_size: usize,
    /// Speaker id embedded in the generator.
    pub speaker_id: i64,
    /// Base learning rate (AdamW), before decay.
    pub lr: f64,
    /// Per-epoch exponential LR decay factor (`lr_epoch = lr * decay^epoch`).
    /// `1.0` disables decay; smaller settles the late-training oscillation that
    /// keeps `mel_loss` shaking on a constant LR.
    pub lr_decay: f64,
    /// Generator weight EMA decay (`ema = decay*ema + (1-decay)*g` each step).
    /// The EMA weights — averaged over the adversarial oscillation, so cleaner
    /// and less staticky — are what gets saved. `0.0` disables EMA (save the raw
    /// live weights instead).
    pub ema_decay: f64,
    /// Micro-batches accumulated per optimizer step. Raises the *effective*
    /// batch size without extra VRAM (more stable gradients on a small GPU).
    /// `1` = plain batching.
    pub grad_accum: usize,
    /// Discriminator LR multiplier relative to the generator. `< 1.0` weakens an
    /// over-eager discriminator whose gradients inject high-frequency buzz.
    pub d_lr_ratio: f64,
    /// Update the discriminator only every N steps (`1` = every step). Another
    /// lever to stop the discriminator overpowering the generator.
    pub d_interval: usize,
    /// SNR-based clip-sampling bias (`0.0` = uniform). Weights each clip by
    /// `snr^alpha` (noise-floor SNR, **not** loudness) so cleaner clips are drawn
    /// more often while soft/breathy passages are preserved.
    pub snr_weight: f32,
    /// Show Burn's interactive TUI dashboard (the caller must have routed logs
    /// off stderr; disable for non-TTY output).
    pub use_tui: bool,
}

/// A full training request: corpus, output, feature models, and warm-start.
#[derive(Debug, Clone)]
pub struct TrainRequest {
    /// Corpus audio files of the target voice.
    pub data: Vec<PathBuf>,
    /// Destination path for the trained weights (`.safetensors` is written).
    pub out: PathBuf,
    /// Scratch directory.
    pub work_dir: PathBuf,
    /// ContentVec encoder ONNX (already resolved/downloaded).
    pub content: PathBuf,
    /// RMVPE F0 ONNX (already resolved/downloaded).
    pub rmvpe: PathBuf,
    /// Resume from a prior run's generator `.safetensors` (`--continue`),
    /// instead of a pretrained base. The discriminator resumes from the sibling
    /// `<stem>.disc.safetensors` if present. Takes priority over `pretrained_g`.
    pub resume: Option<PathBuf>,
    /// Pretrained generator (`f0G48k.pth`) for warm-start.
    pub pretrained_g: Option<PathBuf>,
    /// Pretrained discriminator (`f0D48k.pth`) for warm-start.
    pub pretrained_d: Option<PathBuf>,
    /// Generator hyperparameters.
    pub settings: TrainSettings,
    /// Early-stop flag: set it (e.g. from a SIGINT handler) to stop after the
    /// current step and save the model. In the TUI, `q` stops too.
    pub stop: Arc<AtomicBool>,
}

/// Fine-tune a generator and return the path to the saved weights.
///
/// Blocking (GPU compute); run it off the async runtime (the CLI uses
/// `spawn_blocking`).
pub fn train(req: TrainRequest) -> Result<PathBuf> {
    anyhow::ensure!(!req.data.is_empty(), "no training audio provided");
    anyhow::ensure!(
        req.settings.sample_rate == 48_000,
        "native training currently supports only --model-sr 48000 (got {})",
        req.settings.sample_rate
    );
    std::fs::create_dir_all(&req.work_dir)
        .with_context(|| format!("creating work dir {}", req.work_dir.display()))?;

    // A self-contained runtime drains the async decode streams; GPU training is
    // synchronous.
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .context("building the training runtime")?;

    let clips = rt.block_on(dataset::prepare_clips(
        &req.data,
        &req.content,
        &req.rmvpe,
        req.settings.sample_rate,
        trainer::WINDOW_FRAMES,
    ))?;

    trainer::run(&req, clips)
}
