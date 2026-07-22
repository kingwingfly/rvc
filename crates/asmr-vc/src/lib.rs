//! RVC v2 voice conversion for the ASMR toolkit.
//!
//! The pipeline is three ONNX models run via `ort` (ContentVec + RMVPE +
//! trained generator). It converts a source voice's **timbre** to the target
//! while preserving content and pitch — which is exactly why breathy/expressive
//! vocalizations survive: they live in the content and F0 streams, not the
//! timbre the generator replaces.
//!
//! Audio crosses every boundary as mono `f32`, and the conversion path is
//! [`futures::Stream`]-in → [`futures::Stream`]-out ([`convert::convert_stream`]),
//! so `asmr serve` is a plain Unix filter and a realtime service is a thin wrapper.

mod config;
mod convert;
mod dsp;
mod encoder;
mod error;
mod f0;
mod mel;
mod rvc;
mod session;

#[cfg(test)]
mod model_probe;

pub use config::{
    ConvertParams, GeneratorIo, ModelPaths, RvcConfig, ANALYSIS_SR, CONTENT_DIM, CONTENT_HOP,
    F0_HOP, RMVPE_BINS,
};
pub use convert::{convert_stream, Converter, StreamParams};
pub use error::{Result, VcError};
pub use rvc::RvcModel;
pub use session::build_session;
