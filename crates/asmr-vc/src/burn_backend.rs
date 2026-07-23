//! Native Burn generator backend — the peer of the ORT [`crate::RvcModel`].
//!
//! Feature extraction (ContentVec + RMVPE) reuses [`FeatureExtractor`]; the
//! generator is the [`burn_rvc`] port. Mirrors [`RvcModel::convert_segment`]'s
//! DSP: content features are upsampled ×2 to the F0 rate (inside
//! [`FeatureExtractor::extract_aligned`]), F0 is pitch-shifted, then the two are
//! aligned and fed to [`Synthesizer::infer`]. Runs on the GPU (wgpu/Vulkan);
//! the HiFiGAN decoder is far too slow on CPU.

use std::path::Path;

use burn::backend::wgpu::{Wgpu, WgpuDevice};
use burn::tensor::{Int, Tensor, TensorData};
use burn_rvc::{Synthesizer, SynthesizerConfig};

use crate::backend::Generator;
use crate::config::{CONTENT_DIM, ConvertParams};
use crate::dsp::{f0_to_coarse, shift_pitch};
use crate::error::{Result, VcError};
use crate::features::{DEFAULT_CHUNK, FeatureExtractor};

/// The GPU (wgpu/Vulkan) backend.
type B = Wgpu;

/// A loaded native-Burn conversion pipeline.
pub struct BurnGenerator {
    extractor: FeatureExtractor,
    model: Synthesizer<B>,
    model_sr: u32,
    speaker_id: i64,
}

impl BurnGenerator {
    /// Load the feature extractors and the generator weights
    /// (`.pth`/`.safetensors`).
    pub fn load(
        content: &Path,
        rmvpe: &Path,
        weights: &Path,
        model_sr: u32,
        speaker_id: i64,
    ) -> Result<Self> {
        let cfg = match model_sr {
            40_000 => SynthesizerConfig::v2_40k(),
            _ => SynthesizerConfig::v2_48k(),
        };
        let device = WgpuDevice::default();
        let mut model = Synthesizer::<B>::new(&cfg, &device);
        let res = model.load_weights(weights).map_err(|e| {
            VcError::Burn(format!("loading generator weights {}: {e}", weights.display()))
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
        Ok(Self { extractor, model, model_sr, speaker_id })
    }
}

impl Generator for BurnGenerator {
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

        let device = WgpuDevice::default();
        let phone =
            Tensor::<B, 3>::from_data(TensorData::new(phone_flat, [1, n, CONTENT_DIM]), &device);
        let pitch = Tensor::<B, 2, Int>::from_data(TensorData::new(coarse, [1, n]), &device);
        let nsff0 = Tensor::<B, 2>::from_data(TensorData::new(pitchf, [1, n]), &device);

        let audio = self.model.infer(phone, pitch, nsff0, self.speaker_id);
        audio
            .into_data()
            .into_vec::<f32>()
            .map_err(|e| VcError::Burn(format!("reading generator output: {e:?}")))
    }
}
