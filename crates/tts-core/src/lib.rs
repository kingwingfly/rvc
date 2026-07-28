//! Speech synthesis for the `voice` toolkit — GPT-SoVITS.
//!
//! Two stages, and a text front-end in front of them:
//!
//! 1. [`text_kit`] turns text into phonemes, plus a per-character phoneme count,
//! 2. [`prosody`] turns the same text into a feature per phoneme,
//! 3. the T2S transformer predicts semantic tokens from both, and
//! 4. the SoVITS stage turns those tokens into waveform.
//!
//! Stages 3 and 4 are trained here, so they are Burn ports. The prosody encoder
//! is frozen and published as ONNX, so it runs on ONNX Runtime — see
//! [`prosody`] for why, and for the trait that leaves room to change that.

mod error;
pub mod prosody;
mod sample;
mod synth;

pub use error::{Result, TtsError};
#[cfg(feature = "onnx")]
pub use prosody::OnnxProsody;
pub use prosody::{ProsodyEncoder, ProsodyFeatures};
pub use sample::{Rng, SampleOptions};
pub use synth::{ANALYSIS_SR, OUTPUT_SR, Reference, SynthOptions, Synthesizer};
