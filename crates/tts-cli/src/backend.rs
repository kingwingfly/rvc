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

/// Everything a fine-tune needs, gathered so the per-backend arms below stay
/// one line each.
pub struct TrainInputs<'a> {
    pub hubert: &'a Path,
    pub s1: &'a Path,
    pub s2: &'a Path,
    /// The `s2` discriminator to warm-start from, when the bundle has one.
    pub s2d: Option<&'a Path>,
    pub prosody: Option<&'a Path>,
    pub pairs: &'a [(std::path::PathBuf, std::path::PathBuf)],
    pub language: text_kit::Language,
    pub s1_settings: &'a tts_train::S1Settings,
    pub s2_settings: &'a tts_train::S2Settings,
    /// Output stem; each stage derives its own checkpoint family from it.
    pub out: &'a Path,
    pub stage: crate::train::Stage,
    pub stop: &'a std::sync::atomic::AtomicBool,
}

impl TrainInputs<'_> {
    /// The checkpoint family one stage writes: `models/mine` -> `models/mine.s1`.
    ///
    /// Built by appending to the file name rather than with
    /// [`Path::with_extension`], which sees only the last dot and would turn
    /// `voice.v2` into `voice.s1`, silently merging two runs' outputs.
    fn checkpoint(&self, stage: &str) -> train_kit::Checkpoint {
        let name = self
            .out
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| "voice".into());
        let dir = self.out.parent().unwrap_or(Path::new(""));
        train_kit::Checkpoint::new(&dir.join(format!("{name}.{stage}")))
    }
}

/// Prepare the corpus once and fine-tune whichever stages were asked for.
///
/// The frozen encoders run on the *inner* backend — preparation needs no
/// gradients, and building them under autodiff would tape a forward pass per
/// clip for nothing.
pub fn train(
    inputs: TrainInputs<'_>,
    backend: TtsBackend,
    devices: &[burn_kit::DeviceSpec],
) -> Result<()> {
    anyhow::ensure!(!devices.is_empty(), "no --device given");
    let backend = match backend {
        TtsBackend::Auto => match burn_kit::auto_backend() {
            burn_kit::AutoBackend::LibTorch => TtsBackend::Tch,
            burn_kit::AutoBackend::Cuda => TtsBackend::Cuda,
            burn_kit::AutoBackend::Wgpu => TtsBackend::Wgpu,
        },
        explicit => explicit,
    };
    let list = devices
        .iter()
        .map(|d| d.to_string())
        .collect::<Vec<_>>()
        .join(", ");
    tracing::info!("fine-tuning ({backend:?}, devices: {list})");

    macro_rules! train {
        ($inner:ty, $resolve:expr, $name:literal) => {{
            let devices: Vec<_> = devices
                .iter()
                .map(|d| $resolve(*d))
                .collect::<std::result::Result<Vec<_>, _>>()?;
            let devices = distinct(devices)?;
            let device = devices[0].clone();
            let prosody = load_prosody(inputs.prosody);

            // Preparation is the expensive half of a fine-tune — cnhubert, the
            // quantiser and the prosody BERT over every clip — so it happens
            // once even when both stages train. Waveforms are decoded only when
            // `s2` will actually read them.
            let clips = burn_kit::guard_init($name, || -> Result<_> {
                let (hubert, quantizer) =
                    tts_train::encoders::<$inner>(inputs.hubert, inputs.s2, &device)?;
                let mut prosody = prosody;
                let clips = tts_train::prepare(
                    inputs.pairs,
                    &hubert,
                    &quantizer,
                    &mut prosody,
                    inputs.language,
                    inputs.stage.wants_s2(),
                    &device,
                )?;
                // Never unwound, for the reason `args::run` gives at its own
                // `mem::forget`: dropping an ONNX Runtime CUDA session beside a
                // CUDA Burn backend corrupts the heap, and the abort lands at
                // exit — after the fine-tuned weights are safely on disk, which
                // makes it look like training crashed when it did not. Verified
                // both ways: this same run without `--prosody` exits 0.
                // The cost is the encoder's memory held for the rest of the run;
                // it is inference-only and idle from here on.
                std::mem::forget(prosody);
                Ok(clips)
            })??;

            if inputs.stage.wants_s1() {
                let out = inputs.checkpoint("s1");
                burn_kit::guard_init($name, || {
                    tts_train::s1::run::<burn::backend::Autodiff<$inner>>(
                        inputs.s1,
                        &clips,
                        inputs.s1_settings,
                        &out,
                        inputs.stop,
                        &devices,
                    )
                })??;
            }
            if inputs.stage.wants_s2() {
                let out = inputs.checkpoint("s2");
                burn_kit::guard_init($name, || {
                    tts_train::s2::run::<burn::backend::Autodiff<$inner>>(
                        inputs.s2,
                        inputs.s2d,
                        &clips,
                        inputs.s2_settings,
                        &out,
                        inputs.stop,
                        &devices,
                    )
                })??;
            }
        }};
    }

    match backend {
        #[cfg(feature = "tch")]
        TtsBackend::Tch => train!(
            burn::backend::LibTorch<f32>,
            burn_kit::libtorch_device,
            "tch"
        ),
        #[cfg(feature = "cuda")]
        TtsBackend::Cuda => train!(burn::backend::Cuda, burn_kit::cuda_device, "cuda"),
        #[cfg(feature = "wgpu")]
        TtsBackend::Wgpu => train!(burn::backend::Wgpu, burn_kit::wgpu_device, "wgpu"),
        TtsBackend::Auto => unreachable!("resolved above"),
        #[allow(unreachable_patterns)]
        other => anyhow::bail!(
            "this binary was built without the {other:?} backend (rebuild with `--features …`)"
        ),
    }
    Ok(())
}

/// Reject a device list with repeats, *after* resolution.
///
/// Checking the `--device` strings is not enough: `auto,gpu:0` are different
/// spellings that name device 0 twice, and on WebGPU `auto`, `vulkan` and `mps`
/// all resolve to the default adapter. A repeat would silently double that
/// device's share of the work — and its memory.
fn distinct<D: PartialEq + std::fmt::Debug>(devices: Vec<D>) -> Result<Vec<D>> {
    for (i, d) in devices.iter().enumerate() {
        anyhow::ensure!(
            !devices[..i].contains(d),
            "--device lists {d:?} more than once (different spellings can name one device)"
        );
    }
    Ok(devices)
}

/// Load the prosody encoder if one was found; its absence costs expressiveness
/// rather than correctness, so it is a warning.
fn load_prosody(dir: Option<&Path>) -> Option<Box<dyn ProsodyEncoder>> {
    let dir = dir?;
    #[cfg(feature = "onnx")]
    match tts_core::OnnxProsody::load(dir, 1024) {
        Ok(p) => return Some(Box::new(p) as Box<dyn ProsodyEncoder>),
        Err(e) => tracing::warn!("prosody encoder failed to load ({e}); training without it"),
    }
    #[cfg(not(feature = "onnx"))]
    let _ = dir;
    None
}
