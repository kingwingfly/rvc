//! Backend selection for the engines `voice` adds on top of `rvc`.
//!
//! The same shape as `rvc-cli`'s: an enum of compute backends, an `auto` that
//! defers to [`burn_kit::auto_backend`] so every subcommand of both binaries
//! agrees, and constructors that hand back a boxed trait object so nothing here
//! names a Burn type.

use std::path::Path;

use anyhow::{Context, Result};
use clap::ValueEnum;
use stt_core::Transcriber;

/// Which runtime performs recognition.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, ValueEnum)]
pub enum SttBackend {
    /// Pick by what the model directory holds — an ONNX export runs on ONNX
    /// Runtime, `model.safetensors` runs on the fastest available Burn backend
    /// (LibTorch on a GPU, else CubeCL/CUDA, else WebGPU, else LibTorch on CPU).
    #[default]
    Auto,
    /// ONNX Runtime, from an `optimum`-style export.
    Onnx,
    /// Native Burn, CubeCL/CUDA compute.
    #[value(name = "cuda", alias = "burn-cuda")]
    Cuda,
    /// Native Burn, LibTorch compute — CUDA, MPS, Vulkan or CPU.
    #[value(name = "tch", alias = "libtorch", alias = "burn-tch")]
    Tch,
    /// Native Burn, WebGPU compute.
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
) -> Result<Transcriber> {
    let backend = match backend {
        // Weights first, hardware second — the same order `rvc --backend auto`
        // uses, and for the same reason: an ONNX export cannot run on Burn and
        // safetensors cannot run on ONNX Runtime, so the files decide before
        // preference does.
        SttBackend::Auto if is_onnx_export(dir) => SttBackend::Onnx,
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
        SttBackend::Tch => Transcriber::libtorch(dir, device)
            .context("failed to load Whisper on the LibTorch backend"),
        #[cfg(feature = "cuda")]
        SttBackend::Cuda => Transcriber::cuda(dir, device)
            .context("failed to load Whisper on the CubeCL/CUDA backend"),
        #[cfg(feature = "wgpu")]
        SttBackend::Wgpu => {
            Transcriber::wgpu(dir, device).context("failed to load Whisper on the WebGPU backend")
        }
        #[cfg(feature = "onnx")]
        SttBackend::Onnx => {
            Transcriber::onnx(dir).context("failed to load Whisper on ONNX Runtime")
        }
        SttBackend::Auto => unreachable!("resolved above"),
        // Only reachable on a `--no-default-features` build.
        #[allow(unreachable_patterns)]
        other => anyhow::bail!(
            "this binary was built without the {other:?} backend (rebuild with `--features …`)"
        ),
    }
}

/// Whether `dir` holds an ONNX export rather than Burn weights.
fn is_onnx_export(dir: &Path) -> bool {
    ["onnx/encoder_model.onnx", "encoder_model.onnx"]
        .iter()
        .any(|p| dir.join(p).exists())
}
