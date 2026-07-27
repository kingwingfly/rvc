//! Native RVC generator fine-tuning (Rust + Burn).
//!
//! Pipeline: decode the corpus and extract ContentVec + RMVPE features (reusing
//! the ONNX extractors the inference path uses), then adversarially fine-tune
//! the [`burn_rvc`] generator on a GPU, warm-started from the public pretrained
//! bases, and save the result as safetensors. Deploy either directly
//! (`rvc convert --backend cuda`) or via the ONNX export helper.
//!
//! The loop is generic over the Burn compute backend ([`trainer::run`]); this
//! module picks the concrete one from [`TrainRequest::backend`]. In practice
//! that is always CubeCL/CUDA today — LibTorch cannot run the backward pass, see
//! [`TCH_TRAINING_UNSUPPORTED`] — but the genericity is what lets that change
//! with a one-line dispatch arm once the upstream bug is fixed.

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
/// Both can be linked into one binary; this is the run-time choice. Nothing
/// about the saved weights depends on it — but see
/// [`TCH_TRAINING_UNSUPPORTED`]: only CubeCL/CUDA can train at present.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TrainBackend {
    /// Whatever can actually train. Today that is always [`Self::Cuda`]:
    /// LibTorch is faster for inference but cannot run the backward pass.
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
    /// Which device that backend should use (`DeviceSpec::Auto` = fastest).
    pub device: DeviceSpec,
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
    let backend = match req.backend {
        TrainBackend::Auto => auto_backend(req.device),
        explicit => explicit,
    };
    anyhow::ensure!(
        backend != TrainBackend::LibTorch,
        "{TCH_TRAINING_UNSUPPORTED}"
    );
    Ok(backend)
}

/// Instantiate the resolved compute backend and hand off to [`trainer::run`].
///
/// An *explicit* backend is never silently substituted: asking for one this
/// build lacks, or a device it cannot see, is an error with a reason.
fn dispatch(req: &TrainRequest, clips: Vec<dataset::Clip>) -> Result<PathBuf> {
    let backend = resolve_backend(req)?;
    tracing::info!("training backend: {backend} (device {})", req.device);

    match backend {
        TrainBackend::LibTorch => anyhow::bail!(TCH_TRAINING_UNSUPPORTED),
        #[cfg(feature = "cuda")]
        TrainBackend::Cuda => {
            use burn::backend::{Autodiff, cuda::Cuda};
            let device = rvc_core::cuda_device(req.device)?;
            rvc_core::guard_init("cuda", || {
                trainer::run::<Autodiff<Cuda>>(req, clips, &device)
            })?
        }
        #[cfg(feature = "wgpu")]
        TrainBackend::Wgpu => {
            use burn::backend::{Autodiff, wgpu::Wgpu};
            let device = rvc_core::wgpu_device(req.device)?;
            rvc_core::guard_init("wgpu", || {
                trainer::run::<Autodiff<Wgpu>>(req, clips, &device)
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

/// Why LibTorch cannot train, as of `burn 0.21` / `burn-tch 0.21`.
///
/// `Autodiff<LibTorch>` panics in `conv1d`'s backward whenever a convolution is
/// **grouped** *and* its padded input length is not a multiple of the stride:
/// the weight gradient comes back one or more kernel-taps too long, and
/// LibTorch's strict `copy_` rejects it. `MultiPeriodDiscriminator`'s scale
/// branch is exactly that shape (`k=41, s=4, groups=4..256` over a 17280-sample
/// segment), so every training step hits it.
///
/// Minimal repro, in `examples/convgrad.rs`:
/// `conv1d(16 -> 64, k=4, s=2, p=1, groups=4)` on length 101. Ungrouped is fine,
/// grouped-but-evenly-divisible is fine, and CubeCL/CUDA is fine on all of them —
/// so this is a `burn-tch` backward bug, not a flaw in the port. Inference is
/// unaffected (no backward pass), which is why `--backend tch` converts happily.
pub const TCH_TRAINING_UNSUPPORTED: &str = "\
the LibTorch (tch) backend cannot train: burn 0.21's autodiff panics in the \
backward pass of grouped strided conv1d, which the discriminator uses on every \
step (see `cargo run -p rvc-train --example convgrad --features tch,cuda`).\n\
Use `--backend cuda` to train; `--backend tch` is still the fastest choice for \
`rvc convert` and `rvc serve`.";

/// Resolve `Auto` to a backend that is both compiled in and usable.
///
/// Unlike inference, this never prefers LibTorch: it cannot run a backward pass
/// (see [`TCH_TRAINING_UNSUPPORTED`]). Kept as a separate decision from
/// `rvc_core::auto_prefers_libtorch` for exactly that reason.
fn auto_backend(_spec: DeviceSpec) -> TrainBackend {
    TrainBackend::Cuda
}
