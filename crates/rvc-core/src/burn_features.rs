//! Native Burn feature extraction — the peer of the ORT `OnnxContentEncoder`
//! and `OnnxPitchEstimator`.
//!
//! ContentVec and RMVPE were ONNX-only for as long as `rvc` had one runtime for
//! them, which made `ORT_DYLIB_PATH` a hard requirement of every invocation.
//! These two types give them the run-time choice the generator has had since
//! [`Generator`](crate::Generator) was introduced — **an addition, never a
//! replacement**: `content_vec.onnx` and `rmvpe.onnx` stay supported deployment
//! targets and gain company.
//!
//! Both are generic over the Burn compute backend, so one binary carries all of
//! them and `--backend` chooses at run time. The concrete constructors at the
//! bottom keep every Burn type inside this crate, exactly as `burn_backend`'s
//! do — `rvc-cli` names none of them and this crate gains no backend enum.
//!
//! # What is *not* here
//!
//! Neither the RMVPE mel front end nor the salience → Hz decode. Both are pure
//! Rust already (`mel::RmvpeMel`, `dsp::rmvpe_decode`), backend-agnostic, and shared
//! with the ONNX path — so the two runtimes cannot disagree about anything
//! outside the network itself, which is what makes comparing their F0 a test of
//! the port rather than of the front end.

use std::path::Path;

use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, TensorData};
use burn_rmvpe::{Rmvpe, RmvpeConfig};

use crate::analysis::{ContentEncoder, PitchEstimator};
use crate::config::RMVPE_BINS;
use crate::dsp::rmvpe_decode;
use crate::error::{Result, VcError};
use crate::mel::RmvpeMel;
use burn_kit::{DeviceSpec, guard_init};

/// RMVPE on the Burn compute backend `B`, yielding an F0 (Hz) contour at 100 Hz.
pub struct BurnPitchEstimator<B: Backend> {
    model: Rmvpe<B>,
    mel: RmvpeMel,
    /// Resolved once at load: `B::Device::default()` per clip would silently
    /// fall to device 0 — and for LibTorch, whose default is `Cpu`, off the GPU.
    device: B::Device,
    threshold: f32,
}

impl<B: Backend> BurnPitchEstimator<B> {
    /// Load `rmvpe.pt` (Hugging Face `lj1995/VoiceConversionWebUI`).
    ///
    /// `threshold` is the voicing floor — pass [`FeatureExtractor::F0_THRESHOLD`]
    /// unless you have a reason not to, since it decides which frames come out
    /// as 0 and 0 is what switches the generator to its noise branch.
    ///
    /// [`FeatureExtractor::F0_THRESHOLD`]: crate::FeatureExtractor::F0_THRESHOLD
    pub fn load(weights: &Path, threshold: f32, device: &B::Device) -> Result<Self> {
        let cfg = RmvpeConfig::default();
        tracing::info!("burn rmvpe: {}", B::name(device));
        let mut model = Rmvpe::<B>::new(&cfg, device);
        let res = model.load_pytorch(weights).map_err(|e| {
            VcError::Burn(format!("loading rmvpe weights {}: {e}", weights.display()))
        })?;
        if !res.missing.is_empty() {
            return Err(VcError::Burn(format!(
                "rmvpe weights {} are incomplete: {} params missing (first: {:?})",
                weights.display(),
                res.missing.len(),
                res.missing.first()
            )));
        }
        tracing::info!(
            "loaded {} rmvpe params from {}",
            res.applied.len(),
            weights.display()
        );
        Ok(Self {
            model,
            mel: RmvpeMel::new(),
            device: device.clone(),
            threshold,
        })
    }
}

impl<B: Backend> PitchEstimator for BurnPitchEstimator<B> {
    /// Estimate the F0 (Hz) contour from a mono 16 kHz `f32` buffer.
    ///
    /// The frame count is deliberately **not** aligned here, where
    /// `f0::OnnxPitchEstimator` reflect-pads to a
    /// multiple of 32 by hand: `Rmvpe::forward` pads and trims itself, because
    /// the multiple-of-32 requirement is a property of that network's five
    /// halvings rather than of any caller. It pads with **zeros**, which is
    /// upstream's `F.pad(..., mode="constant")` and therefore what the published
    /// weights were run with; the ONNX path's reflect padding is the one that
    /// diverges from upstream. Both trim afterwards, so the two runtimes can
    /// differ only over the last ≤ 31 frames and only by what leaks in through
    /// the convolution stack's receptive field.
    fn extract(&mut self, wav16k: &[f32]) -> Result<Vec<f32>> {
        let (n_mels, time, data) = self.mel.compute(wav16k);
        if time == 0 {
            return Ok(Vec::new());
        }

        // `compute` returns mel-major `data[d * time + t]`, which is `[1, n_mels,
        // time]` row-major — the shape `forward` wants, with no transpose.
        let mel = Tensor::<B, 3>::from_data(TensorData::new(data, [1, n_mels, time]), &self.device);
        let salience = self.model.forward(mel);
        debug_assert_eq!(salience.dims(), [1, time, RMVPE_BINS]);
        let flat = salience
            .into_data()
            .into_vec::<f32>()
            .map_err(|e| VcError::Burn(format!("reading rmvpe salience: {e:?}")))?;

        let rows: Vec<Vec<f32>> = flat.chunks_exact(RMVPE_BINS).map(<[f32]>::to_vec).collect();
        Ok(rmvpe_decode(&rows, self.threshold))
    }
}

/// ContentVec on the Burn compute backend `B`, yielding time-major `[T][768]`
/// features at 50 Hz.
pub struct BurnContentEncoder<B: Backend> {
    model: burn_rvc::ContentVec<B>,
}

impl<B: Backend> BurnContentEncoder<B> {
    /// Load ContentVec from either the directory
    /// `hub_kit::fetch_contentvec(WeightFormat::Torch, …)` returns or a direct
    /// `pytorch_model.bin` — the same two spellings
    /// [`burn_rvc::ContentVec::load`] accepts, since deciding between them is
    /// its business and not this crate's.
    pub fn load(weights: &Path, device: &B::Device) -> Result<Self> {
        tracing::info!("burn contentvec: {}", B::name(device));
        let (model, res) = burn_rvc::ContentVec::<B>::load(weights, device).map_err(|e| {
            VcError::Burn(format!(
                "loading contentvec weights {}: {e}",
                weights.display()
            ))
        })?;
        if !res.missing.is_empty() {
            return Err(VcError::Burn(format!(
                "contentvec weights {} are incomplete: {} params missing (first: {:?})",
                weights.display(),
                res.missing.len(),
                res.missing.first()
            )));
        }
        tracing::info!(
            "loaded {} contentvec params from {}",
            res.applied.len(),
            weights.display()
        );
        Ok(Self { model })
    }
}

impl<B: Backend> ContentEncoder for BurnContentEncoder<B> {
    fn extract(&mut self, wav16k: &[f32]) -> Result<Vec<Vec<f32>>> {
        Ok(self.model.extract(wav16k))
    }
}

// The constructors below return `Box<dyn _>` where `burn_backend`'s generator
// ones return `impl Generator`, and the difference is not an inconsistency. A
// generator is chosen by a single `match` whose arms are boxed once at the end;
// a content encoder and a pitch estimator are chosen *independently*, so
// `rvc-cli` has two matches whose arms must all agree on one type — and two
// `impl Trait` returns from two functions are two distinct opaque types, which
// no `match` can unify. Boxing at the constructor is where it was going to
// happen anyway: `FeatureExtractor::from_parts` takes boxes.

/// Load the Burn RMVPE on CubeCL/CUDA.
///
/// Returns an erased [`PitchEstimator`], which is what lets `rvc-cli` pick a
/// backend at run time without depending on `burn`.
#[cfg(feature = "cuda")]
pub fn cuda_pitch_estimator(
    weights: &Path,
    threshold: f32,
    device: DeviceSpec,
) -> Result<Box<dyn PitchEstimator>> {
    use burn::backend::cuda::Cuda;

    let device = burn_kit::cuda_device(device)?;
    let model = guard_init("cuda", || {
        BurnPitchEstimator::<Cuda>::load(weights, threshold, &device)
    })??;
    Ok(Box::new(model))
}

/// Load the Burn RMVPE on WebGPU — the portable option: no vendor toolkit, and
/// the only one that runs on AMD, Intel or Apple GPUs.
#[cfg(feature = "wgpu")]
pub fn wgpu_pitch_estimator(
    weights: &Path,
    threshold: f32,
    device: DeviceSpec,
) -> Result<Box<dyn PitchEstimator>> {
    use burn::backend::wgpu::Wgpu;

    let device = burn_kit::wgpu_device(device)?;
    let model = guard_init("wgpu", || {
        BurnPitchEstimator::<Wgpu>::load(weights, threshold, &device)
    })??;
    Ok(Box::new(model))
}

/// Load the Burn RMVPE on the LibTorch backend.
#[cfg(feature = "tch")]
pub fn libtorch_pitch_estimator(
    weights: &Path,
    threshold: f32,
    device: DeviceSpec,
) -> Result<Box<dyn PitchEstimator>> {
    use burn::backend::libtorch::LibTorch;

    let device = burn_kit::libtorch_device(device)?;
    let model = guard_init("tch", || {
        BurnPitchEstimator::<LibTorch<f32>>::load(weights, threshold, &device)
    })??;
    Ok(Box::new(model))
}

/// Load Burn ContentVec on CubeCL/CUDA.
#[cfg(feature = "cuda")]
pub fn cuda_content_encoder(weights: &Path, device: DeviceSpec) -> Result<Box<dyn ContentEncoder>> {
    use burn::backend::cuda::Cuda;

    let device = burn_kit::cuda_device(device)?;
    let model = guard_init("cuda", || {
        BurnContentEncoder::<Cuda>::load(weights, &device)
    })??;
    Ok(Box::new(model))
}

/// Load Burn ContentVec on WebGPU.
#[cfg(feature = "wgpu")]
pub fn wgpu_content_encoder(weights: &Path, device: DeviceSpec) -> Result<Box<dyn ContentEncoder>> {
    use burn::backend::wgpu::Wgpu;

    let device = burn_kit::wgpu_device(device)?;
    let model = guard_init("wgpu", || {
        BurnContentEncoder::<Wgpu>::load(weights, &device)
    })??;
    Ok(Box::new(model))
}

/// Load Burn ContentVec on the LibTorch backend.
#[cfg(feature = "tch")]
pub fn libtorch_content_encoder(
    weights: &Path,
    device: DeviceSpec,
) -> Result<Box<dyn ContentEncoder>> {
    use burn::backend::libtorch::LibTorch;

    let device = burn_kit::libtorch_device(device)?;
    let model = guard_init("tch", || {
        BurnContentEncoder::<LibTorch<f32>>::load(weights, &device)
    })??;
    Ok(Box::new(model))
}
