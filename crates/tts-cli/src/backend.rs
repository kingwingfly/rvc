//! Backend selection for synthesis.
//!
//! The enum, its aliases and the `auto` rule are [`cli_kit::Backend`], shared
//! with every other binary; what is local is whether *these* models are an ONNX
//! export, and a loader that hands back a [`Synthesizer`] so nothing here names
//! a Burn type.
//!
//! The two halves of this file resolve the same `--backend` differently, and
//! deliberately. [`load`] erases the runtime behind `tts_core::Engine`, because
//! inference is a free choice and `--backend onnx` runs the graphs
//! `export/export_gptsovits.py` writes — including from a fine-tune, which is
//! the only reason that exporter exists. [`train`] cannot: ONNX Runtime has no
//! training path, so it stays generic over a concrete Burn `AutodiffBackend` and
//! rejects `onnx` with that reason.

use std::path::Path;

use anyhow::{Context, Result};
use cli_kit::Backend;
use tts_core::{Engine, ProsodyEncoder, Synthesizer};

/// Everything a synthesizer is loaded from.
pub struct ModelPaths<'a> {
    /// The `--models` directory itself, which is where an ONNX export lives.
    pub dir: &'a Path,
    pub hubert: &'a Path,
    pub s1: &'a Path,
    pub s2: &'a Path,
}

/// Load the models onto the chosen backend.
///
/// Naming a backend that isn't compiled in, or one the model directory has
/// nothing for, is an error with a reason — only `auto` substitutes.
pub fn load(
    paths: ModelPaths<'_>,
    prosody: Option<Box<dyn ProsodyEncoder>>,
    backend: Backend,
    device: burn_kit::DeviceSpec,
) -> Result<Synthesizer> {
    let backend = backend.resolve(has_onnx_export(paths.dir));
    tracing::info!("loading GPT-SoVITS ({backend}, device {device})");

    // Each arm builds one engine, boxes it and stops there; the synthesizer
    // around it is the same object either way.
    macro_rules! burn_engine {
        ($inner:ty, $device:expr, $name:literal, $what:literal) => {{
            let device = $device;
            let engine = burn_kit::guard_init($name, || {
                tts_core::BurnEngine::<$inner>::load(paths.hubert, paths.s1, paths.s2, &device)
            })?
            .context(concat!(
                "failed to load GPT-SoVITS on the ",
                $what,
                " backend"
            ))?;
            Box::new(engine) as Box<dyn Engine>
        }};
    }

    let engine: Box<dyn Engine> = match backend {
        #[cfg(feature = "tch")]
        Backend::Tch => burn_engine!(
            burn::backend::LibTorch<f32>,
            burn_kit::libtorch_device(device)?,
            "tch",
            "LibTorch"
        ),
        #[cfg(feature = "cuda")]
        Backend::Cuda => burn_engine!(
            burn::backend::Cuda,
            burn_kit::cuda_device(device)?,
            "cuda",
            "CubeCL/CUDA"
        ),
        #[cfg(feature = "wgpu")]
        Backend::Wgpu => burn_engine!(
            burn::backend::Wgpu,
            burn_kit::wgpu_device(device)?,
            "wgpu",
            "WebGPU"
        ),
        #[cfg(feature = "onnx")]
        Backend::Onnx => Box::new(
            tts_core::OnnxEngine::load(paths.dir)
                .context("failed to load GPT-SoVITS on ONNX Runtime")?,
        ),
        Backend::Auto => unreachable!("resolved above"),
        // Only reachable on a `--no-default-features` build.
        #[allow(unreachable_patterns)]
        other => return Err(other.unavailable()),
    };
    Ok(Synthesizer::new(engine, prosody))
}

/// Whether `dir` holds an ONNX export beside (or instead of) the checkpoints.
///
/// Only compiled in with the feature that could use one, so a build without ONNX
/// Runtime never silently prefers a directory it cannot read.
#[cfg(feature = "onnx")]
fn has_onnx_export(dir: &Path) -> bool {
    tts_core::OnnxEngine::find(dir).is_some()
}

#[cfg(not(feature = "onnx"))]
fn has_onnx_export(_dir: &Path) -> bool {
    false
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
    backend: Backend,
    devices: &[burn_kit::DeviceSpec],
) -> Result<()> {
    anyhow::ensure!(!devices.is_empty(), "no --device given");
    // Nothing on disk to consult: there is no ONNX training path to resolve to.
    let backend = backend.resolve(false);
    let list = devices
        .iter()
        .map(|d| d.to_string())
        .collect::<Vec<_>>()
        .join(", ");
    tracing::info!("fine-tuning ({backend}, devices: {list})");

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
        Backend::Tch => train!(
            burn::backend::LibTorch<f32>,
            burn_kit::libtorch_device,
            "tch"
        ),
        #[cfg(feature = "cuda")]
        Backend::Cuda => train!(burn::backend::Cuda, burn_kit::cuda_device, "cuda"),
        #[cfg(feature = "wgpu")]
        Backend::Wgpu => train!(burn::backend::Wgpu, burn_kit::wgpu_device, "wgpu"),
        // Not "unsupported yet": ONNX Runtime has no training path at all, and
        // the graphs `--backend onnx` runs are what a fine-tune is *exported to*
        // afterwards. Saying so is more use than the generic message below.
        Backend::Onnx => anyhow::bail!(
            "ONNX Runtime cannot train — fine-tune on a Burn backend \
             (`--backend auto|cuda|tch|wgpu`), then export the result with \
             `export/export_gptsovits.py --s1/--s2 <weights>` to run it on ONNX Runtime"
        ),
        Backend::Auto => unreachable!("resolved above"),
        #[allow(unreachable_patterns)]
        other => return Err(other.unavailable()),
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
