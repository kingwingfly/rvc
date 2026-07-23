//! Native RVC generator fine-tuning (Rust + Burn).
//!
//! Pipeline: decode the corpus and extract ContentVec + RMVPE features (reusing
//! the ONNX extractors the inference path uses), then adversarially fine-tune
//! the [`burn_rvc`] generator on the GPU (wgpu), warm-started from the public
//! pretrained bases, and save the result as safetensors. Deploy either directly
//! (`asmr convert --backend burn`) or via the ONNX export helper.

mod dataset;
mod losses;
mod spectral;
mod trainer;

use std::path::PathBuf;

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
    /// Pretrained generator (`f0G48k.pth`) for warm-start.
    pub pretrained_g: Option<PathBuf>,
    /// Pretrained discriminator (`f0D48k.pth`) for warm-start.
    pub pretrained_d: Option<PathBuf>,
    /// Generator hyperparameters.
    pub settings: TrainSettings,
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
