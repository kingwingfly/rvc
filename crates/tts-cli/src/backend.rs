//! Backend selection for synthesis.
//!
//! Same shape as `stt-cli`'s: an enum, an `auto` that defers to
//! [`burn_kit::auto_backend`] so every subcommand of every binary agrees, and a
//! loader that hands back a concrete type so nothing here names a Burn type.
//!
//! There is no `onnx` here as there is for `stt`. `s1` and `s2` are fine-tuned,
//! so they must be Burn; the only ONNX in this path is the frozen prosody
//! encoder, which is not a choice the user makes.

use std::path::Path;

use anyhow::{Context, Result};
use clap::ValueEnum;
use tts_core::{ProsodyEncoder, Synthesizer};

/// Which Burn compute backend runs synthesis.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, ValueEnum)]
pub enum TtsBackend {
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

/// Everything a synthesizer is loaded from.
pub struct ModelPaths<'a> {
    pub hubert: &'a Path,
    pub s1: &'a Path,
    pub s2: &'a Path,
}

/// A loaded synthesizer with its backend erased.
///
/// `Synthesizer<B>` is generic and the CLI must not be, so the concrete backends
/// are folded into one enum here. A trait object would need the reference type
/// erased too, and a reference is tied to the backend that produced it.
///
/// Boxed per variant: a `Synthesizer` holds three loaded networks, so the
/// variants differ by tens of kilobytes and the enum would otherwise be as large
/// as its biggest.
pub enum Loaded {
    #[cfg(feature = "tch")]
    Tch(Box<Synthesizer<burn::backend::LibTorch<f32>>>),
    #[cfg(feature = "cuda")]
    Cuda(Box<Synthesizer<burn::backend::Cuda>>),
    #[cfg(feature = "wgpu")]
    Wgpu(Box<Synthesizer<burn::backend::Wgpu>>),
}

/// Load the models onto the chosen backend.
pub fn load(
    paths: ModelPaths<'_>,
    prosody: Option<Box<dyn ProsodyEncoder>>,
    backend: TtsBackend,
    device: burn_kit::DeviceSpec,
) -> Result<Loaded> {
    let backend = match backend {
        TtsBackend::Auto => match burn_kit::auto_backend() {
            burn_kit::AutoBackend::LibTorch => TtsBackend::Tch,
            burn_kit::AutoBackend::Cuda => TtsBackend::Cuda,
            burn_kit::AutoBackend::Wgpu => TtsBackend::Wgpu,
        },
        explicit => explicit,
    };
    tracing::info!("loading GPT-SoVITS ({backend:?}, device {device})");

    match backend {
        #[cfg(feature = "tch")]
        TtsBackend::Tch => {
            let device = burn_kit::libtorch_device(device)?;
            let model = burn_kit::guard_init("tch", || {
                Synthesizer::load(paths.hubert, paths.s1, paths.s2, prosody, &device)
            })?
            .context("failed to load GPT-SoVITS on the LibTorch backend")?;
            Ok(Loaded::Tch(Box::new(model)))
        }
        #[cfg(feature = "cuda")]
        TtsBackend::Cuda => {
            let device = burn_kit::cuda_device(device)?;
            let model = burn_kit::guard_init("cuda", || {
                Synthesizer::load(paths.hubert, paths.s1, paths.s2, prosody, &device)
            })?
            .context("failed to load GPT-SoVITS on the CubeCL/CUDA backend")?;
            Ok(Loaded::Cuda(Box::new(model)))
        }
        #[cfg(feature = "wgpu")]
        TtsBackend::Wgpu => {
            let device = burn_kit::wgpu_device(device)?;
            let model = burn_kit::guard_init("wgpu", || {
                Synthesizer::load(paths.hubert, paths.s1, paths.s2, prosody, &device)
            })?
            .context("failed to load GPT-SoVITS on the WebGPU backend")?;
            Ok(Loaded::Wgpu(Box::new(model)))
        }
        TtsBackend::Auto => unreachable!("resolved above"),
        // Only reachable on a `--no-default-features` build.
        #[allow(unreachable_patterns)]
        other => anyhow::bail!(
            "this binary was built without the {other:?} backend (rebuild with `--features …`)"
        ),
    }
}
