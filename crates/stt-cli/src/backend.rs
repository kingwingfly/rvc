//! Backend selection for recognition.
//!
//! The enum, its aliases and the `auto` rule are [`cli_kit::Backend`], shared
//! with every other binary; what is local is the one thing that cannot be —
//! whether *these* weights are an ONNX export — plus constructors that hand back
//! a boxed trait object, so nothing here names a Burn type.

use std::path::Path;

use anyhow::{Context, Result};
use cli_kit::Backend;
use stt_core::Transcriber;

/// Load a Whisper checkpoint onto the chosen backend.
///
/// Naming a backend that isn't compiled in, or a device it can't drive, is an
/// error with a reason — only `auto` substitutes.
pub fn load_transcriber(
    dir: &Path,
    backend: Backend,
    device: burn_kit::DeviceSpec,
) -> Result<Transcriber> {
    let backend = backend.resolve(is_onnx_export(dir));
    tracing::info!(
        "loading Whisper from {} ({backend}, device {device})",
        dir.display(),
    );

    match backend {
        #[cfg(feature = "tch")]
        Backend::Tch => Transcriber::libtorch(dir, device)
            .context("failed to load Whisper on the LibTorch backend"),
        #[cfg(feature = "cuda")]
        Backend::Cuda => Transcriber::cuda(dir, device)
            .context("failed to load Whisper on the CubeCL/CUDA backend"),
        #[cfg(feature = "wgpu")]
        Backend::Wgpu => {
            Transcriber::wgpu(dir, device).context("failed to load Whisper on the WebGPU backend")
        }
        #[cfg(feature = "onnx")]
        Backend::Onnx => Transcriber::onnx(dir).context("failed to load Whisper on ONNX Runtime"),
        Backend::Auto => unreachable!("resolved above"),
        // Only reachable on a `--no-default-features` build.
        #[allow(unreachable_patterns)]
        other => Err(other.unavailable()),
    }
}

/// Whether `dir` holds an ONNX export rather than Burn weights.
fn is_onnx_export(dir: &Path) -> bool {
    ["onnx/encoder_model.onnx", "encoder_model.onnx"]
        .iter()
        .any(|p| dir.join(p).exists())
}
