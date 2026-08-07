//! Configuration for the RVC pipeline.
//!
//! Defaults follow the canonical `Retrieval-based-Voice-Conversion-WebUI` ONNX
//! export (`SynthesizerTrnMs768NSFsid`), whose generator takes the inputs
//! `phone`, `phone_lengths`, `pitch`, `pitchf`, `ds`, `rnd` and returns `audio`.
//! Everything is overridable so alternative exports can be accommodated without
//! code changes.

use std::path::PathBuf;

/// Analysis sample rate for the content encoder and F0 extractor (fixed by the
/// pretrained HuBERT/ContentVec + RMVPE models).
pub const ANALYSIS_SR: u32 = 16_000;

/// ContentVec hop: 320 samples @ 16 kHz => 50 frames/s.
pub const CONTENT_HOP: usize = 320;

/// F0 hop: 160 samples @ 16 kHz => 100 frames/s. Content features are upsampled
/// x2 (50 -> 100 Hz) to align with this.
pub const F0_HOP: usize = 160;

/// Feature width of ContentVec v2 (`vec-768-layer-12`).
pub const CONTENT_DIM: usize = 768;

/// RMVPE salience width (cents bins).
pub const RMVPE_BINS: usize = 360;

/// Names of the generator's ONNX inputs/outputs. Override to match a
/// non-standard export.
#[derive(Debug, Clone)]
pub struct GeneratorIo {
    pub phone: String,
    pub phone_lengths: String,
    pub pitch: String,
    pub pitchf: String,
    pub ds: String,
    pub rnd: String,
    pub audio_out: String,
    /// Whether the generator expects a random noise `rnd` input (most NSF-HiFiGAN
    /// exports do). Set false for exports without it.
    pub needs_rnd: bool,
    /// Channel width of the `rnd` noise input (flow latent dim, usually 192).
    pub rnd_dim: usize,
}

impl Default for GeneratorIo {
    fn default() -> Self {
        Self {
            phone: "phone".into(),
            phone_lengths: "phone_lengths".into(),
            pitch: "pitch".into(),
            pitchf: "pitchf".into(),
            ds: "ds".into(),
            rnd: "rnd".into(),
            audio_out: "audio".into(),
            needs_rnd: true,
            rnd_dim: 192,
        }
    }
}

/// Paths to the three ONNX graphs the **fused** pure-ORT pipeline needs.
///
/// Deliberately all-or-nothing, and that is why it is not the general shape:
/// [`RvcModel`](crate::RvcModel) is one pipeline built from all three at once,
/// so it cannot describe an ONNX generator whose RMVPE runs on LibTorch. A mix
/// composes a [`FeatureExtractor`](crate::FeatureExtractor) and a
/// [`Generator`](crate::Generator) separately instead — **except when the
/// generator itself is the ONNX one**, since every [`Generator`] constructor
/// that takes a prebuilt extractor is a Burn one. That combination has nowhere
/// to go and is a caller-side error; `rvc-cli`'s `build_converter` is where it
/// is refused.
#[derive(Debug, Clone)]
pub struct ModelPaths {
    /// ContentVec / HuBERT encoder (`vec-768-layer-12.onnx`).
    pub content: PathBuf,
    /// RMVPE F0 estimator (`rmvpe.onnx`).
    pub rmvpe: PathBuf,
    /// The trained RVC generator (`voice.onnx`).
    pub generator: PathBuf,
}

/// Full RVC pipeline configuration.
#[derive(Debug, Clone)]
pub struct RvcConfig {
    pub models: ModelPaths,
    pub io: GeneratorIo,
    /// Output sample rate of the generator (40_000 or 48_000 depending on the
    /// trained model).
    pub model_sr: u32,
    /// Speaker id fed as `ds` (single-speaker models use 0).
    pub speaker_id: i64,
    /// RMVPE voicing threshold; frames below this salience are treated unvoiced.
    pub f0_threshold: f32,
}

impl RvcConfig {
    /// Construct with default generator I/O names and thresholds.
    pub fn new(models: ModelPaths, model_sr: u32) -> Self {
        Self {
            models,
            io: GeneratorIo::default(),
            model_sr,
            speaker_id: 0,
            f0_threshold: 0.03,
        }
    }
}

/// Per-conversion runtime parameters (independent of which model is loaded).
#[derive(Debug, Clone, Copy, Default)]
pub struct ConvertParams {
    /// Pitch shift in semitones applied to the extracted F0.
    pub transpose: i32,
}
