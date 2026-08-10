//! Reusable content + F0 feature extraction.
//!
//! The trainer needs exactly the analysis features the inference path derives:
//! per-frame ContentVec content vectors and an RMVPE F0 contour. This pairs one
//! [`ContentEncoder`] with one [`PitchEstimator`] so training and inference
//! share a single implementation — whichever runtime each of the two is on.

use std::path::Path;

use crate::analysis::{ContentEncoder, PitchEstimator};
use crate::config::ANALYSIS_SR;
use crate::dsp::upsample_rows;
use crate::encoder::OnnxContentEncoder;
use crate::error::Result;
use crate::f0::OnnxPitchEstimator;
use crate::session::build_session;

/// Default window (16 kHz samples ≈ 10 s) for chunked extraction — keeps the
/// ContentVec/RMVPE convolutions small enough for a modest GPU. Multiple of the
/// content hop (320) so per-chunk content×2 and F0 frame counts line up.
pub const DEFAULT_CHUNK: usize = 160_000;

/// Default RMVPE voicing threshold — upstream's, and what every runtime here
/// draws the line at unless a caller says otherwise.
const F0_THRESHOLD: f32 = 0.03;

/// Build the ONNX ContentVec encoder from a `.onnx` file.
///
/// The peer of `burn_features`' `cuda_content_encoder` and friends, and boxed
/// for the same reason: the caller picks one of several constructors in a
/// `match` and every arm has to yield the same type. This one is not behind a
/// feature, because ONNX Runtime is not optional in this crate.
pub fn onnx_content_encoder(path: &Path) -> Result<Box<dyn ContentEncoder>> {
    Ok(Box::new(OnnxContentEncoder::new(build_session(path)?)))
}

/// Build the ONNX RMVPE pitch estimator from a `.onnx` file.
///
/// `threshold` is the voicing floor — pass [`FeatureExtractor::F0_THRESHOLD`]
/// unless a caller is deliberately moving it, so every runtime in the toolkit
/// draws the voiced/unvoiced line in the same place. It matters more than it
/// looks: a frame below it comes out as 0, and 0 is what switches the generator
/// to the noise branch that renders breath. It is a parameter rather than the
/// constant because 0.03 is upstream's guess about ordinary speech, and the
/// material this toolkit exists for is quiet enough that the guess is worth
/// arguing with.
pub fn onnx_pitch_estimator(path: &Path, threshold: f32) -> Result<Box<dyn PitchEstimator>> {
    Ok(Box::new(OnnxPitchEstimator::new(
        build_session(path)?,
        threshold,
    )))
}

/// Content vectors and F0 for one clip, sampled on the analysis frame grid.
pub struct Features {
    /// Time-major content features, `[T][768]`.
    pub content: Vec<Vec<f32>>,
    /// F0 in Hz at 100 Hz frame rate, `[T_f0]`.
    pub f0: Vec<f32>,
}

/// Holds whichever content encoder and pitch estimator the caller chose.
///
/// The two are independent: nothing here requires them to run on the same
/// backend, which is the whole point of `--content-vec-backend` and
/// `--rmvpe-backend` being separate flags.
pub struct FeatureExtractor {
    content: Box<dyn ContentEncoder>,
    f0: Box<dyn PitchEstimator>,
}

impl FeatureExtractor {
    /// The analysis sample rate feature extraction expects (16 kHz mono).
    pub const ANALYSIS_SR: u32 = ANALYSIS_SR;

    /// Load the ContentVec and RMVPE **ONNX** models from disk.
    ///
    /// Kept as the convenience it always was, since the pure-ORT path builds
    /// exactly this pair. Anything else goes through [`Self::from_parts`].
    pub fn load(content_onnx: &Path, rmvpe_onnx: &Path) -> Result<Self> {
        Ok(Self::from_parts(
            onnx_content_encoder(content_onnx)?,
            onnx_pitch_estimator(rmvpe_onnx, F0_THRESHOLD)?,
        ))
    }

    /// Assemble from two already-built models.
    ///
    /// Deciding *which* implementations these are belongs to the binary, not to
    /// this crate — the same division `stt-cli` and `tts-cli` use, where the
    /// core defines the trait and the CLI constructs the engine. It is what
    /// keeps `rvc-core` free of any backend enum.
    pub fn from_parts(content: Box<dyn ContentEncoder>, f0: Box<dyn PitchEstimator>) -> Self {
        Self { content, f0 }
    }

    /// The RMVPE voicing threshold this toolkit **defaults** to (0.03).
    ///
    /// Exposed so a caller building a non-ONNX [`PitchEstimator`] starts from
    /// the same number rather than picking its own, and so a CLI flag's default
    /// cannot drift away from it. The threshold decides which frames come out as
    /// 0, and 0 is what switches the generator to its noise branch.
    pub const F0_THRESHOLD: f32 = F0_THRESHOLD;

    /// Extract content vectors and F0 from a mono **16 kHz** `f32` buffer.
    ///
    /// Runs the whole buffer through both models in one pass — fine for short
    /// clips, but long audio can exceed GPU memory on any backend. Use
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
