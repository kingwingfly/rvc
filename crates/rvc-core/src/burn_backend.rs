//! Native Burn generator backend — the peer of the ORT [`crate::RvcModel`].
//!
//! Feature extraction (ContentVec + RMVPE) reuses [`FeatureExtractor`]; the
//! generator is the [`burn_rvc`] port. Mirrors [`RvcModel::convert_segment`]'s
//! DSP: content features are upsampled ×2 to the F0 rate (inside
//! [`FeatureExtractor::extract_aligned`]), F0 is pitch-shifted, then the two are
//! aligned and fed to [`Synthesizer::infer`]. Wants a GPU: the HiFiGAN decoder
//! is far too slow on CPU.
//!
//! Generic over the Burn compute backend, so one binary carries all of them and
//! `--backend` chooses at run time. The concrete constructors at the bottom keep
//! every Burn type inside this crate.

use std::path::Path;

use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor, TensorData};
use burn_rvc::{Synthesizer, SynthesizerConfig};

use crate::backend::Generator;
use crate::config::{CONTENT_DIM, ConvertParams};
use crate::dsp::{f0_to_coarse, shift_pitch};
use crate::error::{Result, VcError};
use crate::features::{DEFAULT_CHUNK, FeatureExtractor};
use burn_kit::{DeviceSpec, guard_init};

/// A loaded native-Burn conversion pipeline on the compute backend `B`.
pub struct BurnGenerator<B: Backend> {
    extractor: FeatureExtractor,
    model: Synthesizer<B>,
    /// Resolved once at load: `B::Device::default()` per segment would silently
    /// fall to device 0 — and for LibTorch, whose default is `Cpu`, off the GPU.
    device: B::Device,
    model_sr: u32,
    speaker_id: i64,
}

impl<B: Backend> BurnGenerator<B> {
    /// Load the feature extractors and the generator weights
    /// (`.pth`/`.safetensors`).
    pub fn load(
        content: &Path,
        rmvpe: &Path,
        weights: &Path,
        model_sr: u32,
        speaker_id: i64,
        device: &B::Device,
    ) -> Result<Self> {
        let cfg = match model_sr {
            40_000 => SynthesizerConfig::v2_40k(),
            _ => SynthesizerConfig::v2_48k(),
        };
        tracing::info!("burn backend: {}", B::name(device));
        let mut model = Synthesizer::<B>::new(&cfg, device);
        let res = model.load_weights(weights).map_err(|e| {
            VcError::Burn(format!(
                "loading generator weights {}: {e}",
                weights.display()
            ))
        })?;
        if !res.missing.is_empty() {
            return Err(VcError::Burn(format!(
                "generator weights {} are incomplete: {} params missing (first: {:?})",
                weights.display(),
                res.missing.len(),
                res.missing.first()
            )));
        }
        tracing::info!(
            "loaded {} generator params from {}",
            res.applied.len(),
            weights.display()
        );

        let extractor = FeatureExtractor::load(content, rmvpe)?;
        Ok(Self {
            extractor,
            model,
            device: device.clone(),
            model_sr,
            speaker_id,
        })
    }
}

impl<B: Backend> Generator for BurnGenerator<B> {
    fn output_sr(&self) -> u32 {
        self.model_sr
    }

    fn convert_segment(&mut self, wav16k: &[f32], params: ConvertParams) -> Result<Vec<f32>> {
        // Chunked extraction bounds ONNX memory on long segments; content is
        // upsampled to the 100 Hz F0 grid and aligned to f0.
        let (content, mut f0) = self.extractor.extract_aligned(wav16k, DEFAULT_CHUNK)?;
        shift_pitch(&mut f0, params.transpose);

        let n = content.len().min(f0.len());
        if n == 0 {
            return Ok(Vec::new());
        }
        let coarse = f0_to_coarse(&f0[..n]);
        let pitchf = f0[..n].to_vec();

        let mut phone_flat = Vec::with_capacity(n * CONTENT_DIM);
        for row in &content[..n] {
            phone_flat.extend_from_slice(row);
        }

        let device = &self.device;
        let phone =
            Tensor::<B, 3>::from_data(TensorData::new(phone_flat, [1, n, CONTENT_DIM]), device);
        let pitch = Tensor::<B, 2, Int>::from_data(TensorData::new(coarse, [1, n]), device);
        let nsff0 = Tensor::<B, 2>::from_data(TensorData::new(pitchf, [1, n]), device);

        let audio = self.model.infer(phone, pitch, nsff0, self.speaker_id);
        audio
            .into_data()
            .into_vec::<f32>()
            .map_err(|e| VcError::Burn(format!("reading generator output: {e:?}")))
    }
}

/// Load the Burn generator on CubeCL/CUDA.
///
/// Returns an opaque [`Generator`], which is what lets `rvc-cli` pick a backend
/// at run time without depending on `burn`.
#[cfg(feature = "cuda")]
pub fn cuda_generator(
    content: &Path,
    rmvpe: &Path,
    weights: &Path,
    model_sr: u32,
    speaker_id: i64,
    device: DeviceSpec,
) -> Result<impl Generator + 'static> {
    use burn::backend::cuda::Cuda;

    let device = burn_kit::cuda_device(device)?;
    guard_init("cuda", || {
        BurnGenerator::<Cuda>::load(content, rmvpe, weights, model_sr, speaker_id, &device)
    })?
}

/// Load the Burn generator on WebGPU — the portable option: no vendor toolkit,
/// and the only one that runs on AMD, Intel or Apple GPUs.
#[cfg(feature = "wgpu")]
pub fn wgpu_generator(
    content: &Path,
    rmvpe: &Path,
    weights: &Path,
    model_sr: u32,
    speaker_id: i64,
    device: DeviceSpec,
) -> Result<impl Generator + 'static> {
    use burn::backend::wgpu::Wgpu;

    let device = burn_kit::wgpu_device(device)?;
    guard_init("wgpu", || {
        BurnGenerator::<Wgpu>::load(content, rmvpe, weights, model_sr, speaker_id, &device)
    })?
}

/// Load the Burn generator on the LibTorch backend.
#[cfg(feature = "tch")]
pub fn libtorch_generator(
    content: &Path,
    rmvpe: &Path,
    weights: &Path,
    model_sr: u32,
    speaker_id: i64,
    device: DeviceSpec,
) -> Result<impl Generator + 'static> {
    use burn::backend::libtorch::LibTorch;

    let device = burn_kit::libtorch_device(device)?;
    guard_init("tch", || {
        BurnGenerator::<LibTorch<f32>>::load(content, rmvpe, weights, model_sr, speaker_id, &device)
    })?
}
