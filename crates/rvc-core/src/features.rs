//! Reusable content + F0 feature extraction.
//!
//! The trainer needs exactly the analysis features the inference path derives:
//! per-frame ContentVec content vectors and an RMVPE F0 contour. This wraps the
//! same two ONNX sessions so training and inference share one implementation.

use std::path::Path;

use crate::config::ANALYSIS_SR;
use crate::dsp::upsample_rows;
use crate::encoder::ContentEncoder;
use crate::error::Result;
use crate::f0::F0Estimator;
use crate::session::build_session;

/// Default window (16 kHz samples ≈ 10 s) for chunked extraction — keeps the
/// ContentVec/RMVPE convolutions small enough for a modest GPU. Multiple of the
/// content hop (320) so per-chunk content×2 and F0 frame counts line up.
pub const DEFAULT_CHUNK: usize = 160_000;

/// RMVPE voicing threshold used across the toolkit.
const F0_THRESHOLD: f32 = 0.03;

/// Content vectors and F0 for one clip, sampled on the analysis frame grid.
pub struct Features {
    /// Time-major content features, `[T][768]`.
    pub content: Vec<Vec<f32>>,
    /// F0 in Hz at 100 Hz frame rate, `[T_f0]`.
    pub f0: Vec<f32>,
}

/// Builds and holds the ContentVec + RMVPE ONNX sessions for repeated use.
pub struct FeatureExtractor {
    content: ContentEncoder,
    f0: F0Estimator,
}

impl FeatureExtractor {
    /// The analysis sample rate feature extraction expects (16 kHz mono).
    pub const ANALYSIS_SR: u32 = ANALYSIS_SR;

    /// Load the ContentVec and RMVPE ONNX models from disk.
    pub fn load(content_onnx: &Path, rmvpe_onnx: &Path) -> Result<Self> {
        let content = ContentEncoder::new(build_session(content_onnx)?);
        let f0 = F0Estimator::new(build_session(rmvpe_onnx)?, F0_THRESHOLD);
        Ok(Self { content, f0 })
    }

    /// Extract content vectors and F0 from a mono **16 kHz** `f32` buffer.
    ///
    /// Runs the whole buffer through the ONNX models in one pass — fine for
    /// short clips, but long audio can exceed GPU memory. Use
    /// [`Self::extract_aligned`] for arbitrary-length input.
    pub fn extract(&mut self, wav16k: &[f32]) -> Result<Features> {
        let content = self.content.extract(wav16k)?;
        let f0 = self.f0.extract(wav16k)?;
        Ok(Features { content, f0 })
    }

    /// Extract content (upsampled ×2 to the 100 Hz F0 grid) and F0, processing
    /// the audio in `chunk`-sample windows so the ONNX convolutions stay within
    /// GPU memory regardless of clip length. Returns **equal-length** aligned
    /// frames `(content[T][768], f0[T])`.
    pub fn extract_aligned(
        &mut self,
        wav16k: &[f32],
        chunk: usize,
    ) -> Result<(Vec<Vec<f32>>, Vec<f32>)> {
        let chunk = chunk.max(ANALYSIS_SR as usize); // at least 1 s per window
        let mut content: Vec<Vec<f32>> = Vec::new();
        let mut f0: Vec<f32> = Vec::new();

        let mut start = 0;
        while start < wav16k.len() {
            let end = (start + chunk).min(wav16k.len());
            // Skip a tiny trailing window that can't yield frames.
            if end - start < ANALYSIS_SR as usize / 5 {
                break;
            }
            let feats = self.extract(&wav16k[start..end])?;
            let up = upsample_rows(&feats.content, 2); // 50 Hz -> 100 Hz
            let n = up.len().min(feats.f0.len());
            content.extend_from_slice(&up[..n]);
            f0.extend_from_slice(&feats.f0[..n]);
            start = end;
        }
        Ok((content, f0))
    }
}
