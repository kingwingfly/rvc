//! Native RVC generator fine-tuning (Rust + Burn).
//!
//! Pipeline: decode the corpus and extract ContentVec + RMVPE features (reusing
//! the ONNX extractors the inference path uses), then adversarially fine-tune
//! the [`burn_rvc`] generator on a GPU, warm-started from the public pretrained
//! bases, and save the result as safetensors. Deploy either directly
//! (`rvc convert --backend cuda`) or via the ONNX export helper.
//!
//! The loop is generic over the Burn compute backend ([`trainer::run`]); this
//! module picks the concrete one from [`TrainRequest::backend`]. Weights are
//! backend-independent: a model trained on LibTorch loads on CubeCL and back.

mod checkpoint;
mod dashboard;
mod dataset;
mod losses;
mod spectral;
mod trainer;

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use anyhow::{Context, Result};
pub use rvc_core::DeviceSpec;

/// Which Burn compute backend runs the training loop.
///
/// All can be linked into one binary; this is the run-time choice. Nothing about
/// the saved weights depends on it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TrainBackend {
    /// Fastest available: LibTorch on CUDA, else CubeCL/CUDA, else LibTorch CPU.
    #[default]
    Auto,
    /// CubeCL/CUDA kernels. NVIDIA only.
    Cuda,
    /// LibTorch (tch): CUDA, MPS, Vulkan or CPU.
    LibTorch,
    /// WebGPU (wgpu → Vulkan/Metal/DX12). Vendor-neutral, no toolkit needed.
    Wgpu,
}

impl std::fmt::Display for TrainBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Auto => "auto",
            Self::Cuda => "cuda",
            Self::LibTorch => "tch",
            Self::Wgpu => "wgpu",
        })
    }
}

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
    /// End-of-run learning rate as a fraction of [`Self::lr`]. The LR decays
    /// exponentially from `lr` to `lr * lr_final` over the whole scheduled run
    /// (`lr * lr_final^(step/total_steps)`), so the schedule is independent of the
    /// epoch count. `1.0` disables decay; smaller settles the late-training
    /// oscillation that keeps `mel_loss` shaking on a constant LR.
    pub lr_final: f64,
    /// Generator weight EMA smoothing window as a fraction of the whole run. The
    /// per-step decay is derived as `1 - 1/(ema_frac * total_steps)`, so the EMA
    /// averages over `ema_frac` of the run regardless of epoch count. The EMA
    /// weights — averaged over the adversarial oscillation, so cleaner and less
    /// staticky — are what gets saved (the raw live weights are saved alongside).
    /// `0.0` disables EMA (save only the raw live weights).
    pub ema_frac: f64,
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
    /// Also keep a *best* checkpoint (on by default; `rvc train --no-save-best`):
    /// whenever the windowed-mean `mel` loss hits a new minimum, write the
    /// generator and its discriminator sidecar to
    /// `<out-dir>/checkpoint/<name>.best[.disc].safetensors`. `false` = only the
    /// final weights are written.
    pub save_best: bool,
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
    /// Compute backend for the training loop.
    pub backend: TrainBackend,
    /// Devices to train on. One entry is ordinary single-device training; more
    /// than one runs data-parallel with device 0 as the master.
    pub devices: Vec<DeviceSpec>,
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
    // Reject an unusable backend before the corpus is decoded, not after.
    resolve_backend(&req)?;
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

    dispatch(&req, clips)
}

/// The backend a request resolves to, rejecting one that cannot work.
///
/// Separate from [`dispatch`] so `train` can call it *before* decoding the
/// corpus: feature extraction takes minutes, and finding out afterwards that the
/// backend was never going to run is a poor trade.
fn resolve_backend(req: &TrainRequest) -> Result<TrainBackend> {
    anyhow::ensure!(!req.devices.is_empty(), "no --device given");
    // Duplicates would silently double a device's share of the batch.
    for (i, d) in req.devices.iter().enumerate() {
        anyhow::ensure!(
            !req.devices[..i].contains(d),
            "device {d} is listed twice in --devices"
        );
    }
    let backend = match req.backend {
        TrainBackend::Auto if rvc_core::auto_prefers_libtorch() => TrainBackend::LibTorch,
        TrainBackend::Auto => TrainBackend::Cuda,
        explicit => explicit,
    };
    Ok(backend)
}

/// Instantiate the resolved compute backend and hand off to [`trainer::run`].
///
/// An *explicit* backend is never silently substituted: asking for one this
/// build lacks, or a device it cannot see, is an error with a reason.
fn dispatch(req: &TrainRequest, clips: Vec<dataset::Clip>) -> Result<PathBuf> {
    let backend = resolve_backend(req)?;
    let list = req
        .devices
        .iter()
        .map(|d| d.to_string())
        .collect::<Vec<_>>()
        .join(", ");
    tracing::info!("training backend: {backend} (devices: {list})");

    match backend {
        #[cfg(feature = "tch")]
        TrainBackend::LibTorch => {
            use burn::backend::{Autodiff, libtorch::LibTorch};
            let devices = req
                .devices
                .iter()
                .map(|d| rvc_core::libtorch_device(*d))
                .collect::<rvc_core::Result<Vec<_>>>()?;
            rvc_core::guard_init("tch", || {
                trainer::run::<Autodiff<LibTorch<f32>>>(req, clips, &devices)
            })?
        }
        #[cfg(feature = "cuda")]
        TrainBackend::Cuda => {
            use burn::backend::{Autodiff, cuda::Cuda};
            let devices = req
                .devices
                .iter()
                .map(|d| rvc_core::cuda_device(*d))
                .collect::<rvc_core::Result<Vec<_>>>()?;
            rvc_core::guard_init("cuda", || {
                trainer::run::<Autodiff<Cuda>>(req, clips, &devices)
            })?
        }
        #[cfg(feature = "wgpu")]
        TrainBackend::Wgpu => {
            use burn::backend::{Autodiff, wgpu::Wgpu};
            let devices = req
                .devices
                .iter()
                .map(|d| rvc_core::wgpu_device(*d))
                .collect::<rvc_core::Result<Vec<_>>>()?;
            rvc_core::guard_init("wgpu", || {
                trainer::run::<Autodiff<Wgpu>>(req, clips, &devices)
            })?
        }
        // `Auto` is resolved above, so reaching here means the chosen backend was
        // compiled out — which only happens on a `--no-default-features` build.
        #[allow(unreachable_patterns)]
        other => anyhow::bail!(
            "this build has no `{other}` training backend \
             (rebuild with `--features {other}`)"
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn req(devices: &[DeviceSpec], backend: TrainBackend) -> TrainRequest {
        TrainRequest {
            data: vec!["a.wav".into()],
            out: "out".into(),
            work_dir: "wd".into(),
            content: "c.onnx".into(),
            rmvpe: "r.onnx".into(),
            resume: None,
            pretrained_g: None,
            pretrained_d: None,
            settings: TrainSettings {
                sample_rate: 48_000,
                epochs: 1,
                batch_size: 1,
                speaker_id: 0,
                lr: 1e-4,
                lr_final: 0.1,
                ema_frac: 0.1,
                grad_accum: 1,
                d_lr_ratio: 1.0,
                d_interval: 1,
                snr_weight: 0.0,
                save_best: true,
                use_tui: false,
            },
            backend,
            devices: devices.to_vec(),
            stop: Arc::new(AtomicBool::new(false)),
        }
    }

    #[test]
    fn a_repeated_device_is_rejected() {
        // Listing one twice would silently double its share of the batch, which
        // looks like training working and quietly isn't.
        let r = req(
            &[DeviceSpec::Gpu(0), DeviceSpec::Gpu(0)],
            TrainBackend::Cuda,
        );
        let err = resolve_backend(&r).unwrap_err().to_string();
        assert!(err.contains("twice"), "{err}");

        let ok = req(
            &[DeviceSpec::Gpu(0), DeviceSpec::Gpu(1)],
            TrainBackend::Cuda,
        );
        assert_eq!(resolve_backend(&ok).unwrap(), TrainBackend::Cuda);
    }

    #[test]
    fn an_empty_device_list_is_rejected() {
        let err = resolve_backend(&req(&[], TrainBackend::Cuda))
            .unwrap_err()
            .to_string();
        assert!(err.contains("no --device"), "{err}");
    }

    #[test]
    fn every_backend_is_accepted_for_training() {
        for b in [
            TrainBackend::Cuda,
            TrainBackend::LibTorch,
            TrainBackend::Wgpu,
        ] {
            assert_eq!(resolve_backend(&req(&[DeviceSpec::Auto], b)).unwrap(), b);
        }
    }
}
