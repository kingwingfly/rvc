//! Speech synthesis for the `voice` toolkit — GPT-SoVITS.
//!
//! Two stages, and a text front-end in front of them:
//!
//! 1. [`text_kit`] turns text into phonemes, plus a per-character phoneme count,
//! 2. [`prosody`] turns the same text into a feature per phoneme,
//! 3. the T2S transformer predicts semantic tokens from both, and
//! 4. the SoVITS stage turns those tokens into waveform.
//!
//! Stages 3 and 4 are trained here, so a Burn port is the only thing that can
//! fine-tune them — but a *frozen* copy of either runs on whatever the user
//! chose, so both sit behind [`Engine`], with a Burn implementation and an ONNX
//! Runtime one. [`engine`] says what that boundary is and why it falls there.
//! The prosody encoder is frozen and published as ONNX; [`prosody`] says why it
//! has no Burn implementation and what would justify writing one.

mod burn_engine;
mod engine;
mod error;
#[cfg(feature = "onnx")]
mod onnx_engine;
pub mod prosody;
mod sample;
mod synth;

pub use burn_engine::BurnEngine;
pub use engine::{EOS, Engine, PRIOR_CHANNELS};
pub use error::{Result, TtsError};
#[cfg(feature = "onnx")]
pub use onnx_engine::OnnxEngine;
#[cfg(feature = "onnx")]
pub use prosody::OnnxProsody;
pub use prosody::{ProsodyEncoder, ProsodyFeatures};
pub use sample::{Rng, SampleOptions};
pub use synth::{ANALYSIS_SR, OUTPUT_SR, Reference, SynthOptions, Synthesizer};
