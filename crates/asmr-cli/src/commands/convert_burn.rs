//! Native Burn generator inference for `asmr convert`.
//!
//! Feature extraction (ContentVec + RMVPE) reuses `asmr-vc`; the generator is
//! the `burn-rvc` port. Mirrors the ORT `RvcModel::convert_segment` DSP: content
//! features are upsampled ×2 to the F0 rate, F0 is pitch-shifted, then the two
//! are aligned and fed to `Synthesizer::infer`.

use std::path::Path;

use anyhow::{anyhow, Context, Result};
use asmr_vc::{f0_to_coarse, shift_pitch, FeatureExtractor, CONTENT_DIM, DEFAULT_CHUNK};
use burn::backend::wgpu::{Wgpu, WgpuDevice};
use burn::tensor::{Int, Tensor, TensorData};
use burn_rvc::{Synthesizer, SynthesizerConfig};

/// The GPU (wgpu/Vulkan) backend — the HiFiGAN decoder is far too slow on CPU.
type B = Wgpu;

/// A loaded Burn conversion pipeline.
pub struct BurnConverter {
    extractor: FeatureExtractor,
    model: Synthesizer<B>,
    model_sr: u32,
    transpose: i32,
    speaker_id: i64,
}

impl BurnConverter {
    /// Load feature extractors and the generator weights (`.pth`/`.safetensors`).
    pub fn load(
        content: &Path,
        rmvpe: &Path,
        weights: &Path,
        model_sr: u32,
        transpose: i32,
        speaker_id: i64,
    ) -> Result<Self> {
        let cfg = match model_sr {
            40_000 => SynthesizerConfig::v2_40k(),
            _ => SynthesizerConfig::v2_48k(),
        };
        let device = WgpuDevice::default();
        let mut model = Synthesizer::<B>::new(&cfg, &device);
        let res = model
            .load_weights(weights)
            .map_err(|e| anyhow!("loading generator weights {}: {e}", weights.display()))?;
        if !res.missing.is_empty() {
            return Err(anyhow!(
                "generator weights {} are incomplete: {} params missing (first: {:?})",
                weights.display(),
                res.missing.len(),
                res.missing.first()
            ));
        }
        tracing::info!("loaded {} generator params from {}", res.applied.len(), weights.display());

        let extractor = FeatureExtractor::load(content, rmvpe)
            .map_err(|e| anyhow!("loading ContentVec/RMVPE ONNX: {e}"))?;

        Ok(Self { extractor, model, model_sr, transpose, speaker_id })
    }

    /// The generator's output sample rate.
    pub fn output_sr(&self) -> u32 {
        self.model_sr
    }

    /// Convert one mono 16 kHz clip, returning audio at [`Self::output_sr`].
    pub fn convert(&mut self, wav16k: &[f32]) -> Result<Vec<f32>> {
        // Chunked extraction bounds ONNX memory on long clips; content is
        // upsampled to the 100 Hz F0 grid and aligned to f0.
        let (content, mut f0) = self
            .extractor
            .extract_aligned(wav16k, DEFAULT_CHUNK)
            .map_err(|e| anyhow!("feature extraction: {e}"))?;
        shift_pitch(&mut f0, self.transpose);

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
        let phone = Tensor::<B, 3>::from_data(TensorData::new(phone_flat, [1, n, CONTENT_DIM]), &device);
        let pitch = Tensor::<B, 2, Int>::from_data(TensorData::new(coarse, [1, n]), &device);
        let nsff0 = Tensor::<B, 2>::from_data(TensorData::new(pitchf, [1, n]), &device);

        let audio = self.model.infer(phone, pitch, nsff0, self.speaker_id);
        audio
            .into_data()
            .into_vec::<f32>()
            .map_err(|e| anyhow!("reading generator output: {e:?}"))
            .context("converting segment")
    }
}
