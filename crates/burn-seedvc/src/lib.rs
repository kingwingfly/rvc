//! The Seed-VC network in [Burn](https://burn.dev).
//!
//! Seed-VC converts a voice **without training on it**: a 1–30 s reference clip
//! is the whole speaker specification, where RVC and GPT-SoVITS each want a
//! fine-tune. That is the reason this port exists beside them rather than
//! instead of them.
//!
//! The signal path, in the order the pieces depend on each other:
//!
//! 1. [`content`] runs the source audio through a frozen Whisper encoder, which
//!    is what carries *what was said* while dropping most of who said it,
//! 2. [`length_regulator`] resamples those features to the mel frame rate,
//! 3. [`campplus`] turns the reference clip into one timbre vector, from its own
//!    separate checkpoint — [`style_encoder`] looks like the module that does
//!    this and is a fossil the released inference path never builds,
//! 4. [`dit`] is a diffusion transformer that predicts a mel, conditioned on the
//!    content stream and that timbre vector, with [`wavenet`] as its final block,
//! 5. [`flow`] is the sampler that drives the transformer over N steps,
//! 6. [`bigvgan`] vocodes the mel back to a waveform.
//!
//! Only the transformer is generative; everything either side of it is frozen.
//!
//! Nothing here names a compute backend, and there are no app dependencies.
//!
//! # Provenance
//!
//! Ported from Seed-VC (<https://github.com/Plachtaa/seed-vc>, GPL-3.0), read as
//! a reference and never run or vendored. The target preset is
//! `seed-uvit-whisper-small-wavenet` — see [`config`] for why that one, and for
//! every dimension the checkpoint expects.
//!
//! **This is why the whole workspace is GPL-3.0.** A port written from reading
//! GPL source is a derivative work, and the licence carries forward to anything
//! that links it.

pub mod config;

mod bigvgan;
pub mod content;
mod dit;
pub mod flow;
// Public where the unported siblings are private: the weight-coverage example
// is this crate's only test, and it has to construct the modules it checks.
pub mod campplus;
pub mod length_regulator;
pub mod style_encoder;
pub mod vq;
mod wavenet;

pub use config::SeedVcConfig;
pub use length_regulator::InterpolateRegulator;
pub use vq::{ResidualVq, VqConfig};
