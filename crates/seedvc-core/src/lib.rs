//! Zero-shot voice conversion — the engine around [`burn_seedvc`].
//!
//! What separates this engine from `rvc` and `tts` is that **there is nothing to
//! train**. A 1–30 s reference clip is the entire speaker specification, so the
//! same binary converts into any voice you can produce a recording of, and there
//! is no `train` subcommand anywhere in the crate.
//!
//! The reference conditions the transformer **three ways**, which is the part
//! worth knowing before reading any of this:
//!
//! - as a **timbre vector**, a Kaldi filterbank of the clip through CAMPPlus;
//! - as a **mel prefix** the output is generated *after*, in context — so the
//!   transformer is asked to continue the reference rather than to imitate it,
//!   and the prompt-length prefix has to be sliced back off afterwards;
//! - as its own **length-regulated content**, prepended to the source's, so the
//!   prompt it is asked to continue is one whose audio and content agree.
//!
//! Everything downstream of that is the same shape `rvc-core` has: a
//! [`Converter`] that owns the model and turns a `futures::Stream` of source PCM
//! into a stream of converted PCM, with a batch wrapper over it.
//!
//! # Rates
//!
//! Three, and they are not interchangeable. The content encoder eats **16 kHz**
//! and its log-mel is Whisper's own, not the VITS one. CAMPPlus eats a **Kaldi
//! filterbank** at 16 kHz in `[batch, frames, bins]` — the opposite axis order to
//! everything else. Everything from the diffusion transformer onward is
//! **22.05 kHz**. Every pairing has matching frame counts and 80 bands, so
//! substituting one for another runs happily and computes something else.

pub mod backend;
pub mod convert;
pub mod error;
pub mod model;
#[cfg(feature = "onnx")]
pub mod onnx_model;
pub mod reference;
pub mod stream;

pub use backend::load;
/// The sampler's knobs — Euler steps and the classifier-free guidance scale.
///
/// Re-exported because they are user flags that reach [`Model::convert`]
/// unchanged, and a caller should not need `burn-seedvc` in its manifest to name
/// two numbers.
pub use burn_seedvc::flow::Sampler;
pub use convert::{ConvertOptions, convert, convert_path};
pub use error::{Error, Result};
pub use model::{BurnModel, CONTENT_SR, Model, ModelPaths, Reference};
#[cfg(feature = "onnx")]
pub use onnx_model::OnnxModel;
pub use stream::{Converter, StreamParams, convert_stream};
