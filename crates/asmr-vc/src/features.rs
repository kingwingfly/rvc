//! Reusable content + F0 feature extraction.
//!
//! The trainer needs exactly the analysis features the inference path derives:
//! per-frame ContentVec content vectors and an RMVPE F0 contour. This wraps the
//! same two ONNX sessions so training and inference share one implementation.

use std::path::Path;

use crate::config::ANALYSIS_SR;
use crate::encoder::ContentEncoder;
use crate::error::Result;
use crate::f0::F0Estimator;
use crate::session::build_session;

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
    pub fn extract(&mut self, wav16k: &[f32]) -> Result<Features> {
        let content = self.content.extract(wav16k)?;
        let f0 = self.f0.extract(wav16k)?;
        Ok(Features { content, f0 })
    }
}
