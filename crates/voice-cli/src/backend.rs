//! Backend selection for the engines `voice` adds on top of `rvc`.
//!
//! The same shape as `rvc-cli`'s: an enum of compute backends, an `auto` that
//! defers to [`burn_kit::auto_backend`] so every subcommand of both binaries
//! agrees, and constructors that hand back a boxed trait object so nothing here
//! names a Burn type.

use std::path::Path;

use anyhow::{Context, Result};
use clap::ValueEnum;
use voice_stt::Transcribe;

/// Which Burn compute backend runs recognition. There is no ONNX path.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, ValueEnum)]
pub enum SttBackend {
    /// Fastest available: LibTorch on a GPU, else CubeCL/CUDA, else WebGPU,
    /// else LibTorch on CPU.
    #[default]
    Auto,
    #[value(name = "cuda", alias = "burn-cuda")]
    Cuda,
    #[value(name = "tch", alias = "libtorch", alias = "burn-tch")]
    Tch,
    #[value(name = "wgpu", alias = "webgpu", alias = "burn-wgpu")]
    Wgpu,
}

/// Load a Whisper checkpoint onto the chosen backend.
///
/// Naming a backend that isn't compiled in, or a device it can't drive, is an
/// error with a reason — only `auto` substitutes.
pub fn load_transcriber(
    dir: &Path,
    backend: SttBackend,
    device: burn_kit::DeviceSpec,
) -> Result<Box<dyn Transcribe>> {
    let backend = match backend {
        SttBackend::Auto => match burn_kit::auto_backend() {
            burn_kit::AutoBackend::LibTorch => SttBackend::Tch,
            burn_kit::AutoBackend::Cuda => SttBackend::Cuda,
            burn_kit::AutoBackend::Wgpu => SttBackend::Wgpu,
        },
        explicit => explicit,
    };
    tracing::info!(
        "loading Whisper from {} ({:?}, device {device})",
        dir.display(),
        backend
    );

    match backend {
        #[cfg(feature = "tch")]
        SttBackend::Tch => voice_stt::libtorch_transcriber(dir, device)
            .context("failed to load Whisper on the LibTorch backend"),
        #[cfg(feature = "cuda")]
        SttBackend::Cuda => voice_stt::cuda_transcriber(dir, device)
            .context("failed to load Whisper on the CubeCL/CUDA backend"),
        #[cfg(feature = "wgpu")]
        SttBackend::Wgpu => voice_stt::wgpu_transcriber(dir, device)
            .context("failed to load Whisper on the WebGPU backend"),
        SttBackend::Auto => unreachable!("resolved above"),
        // Only reachable on a `--no-default-features` build.
        #[allow(unreachable_patterns)]
        other => anyhow::bail!(
            "this binary was built without the {other:?} backend (rebuild with `--features …`)"
        ),
    }
}
