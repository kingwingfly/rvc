//! RVC v2 voice conversion — the rvc toolkit's conversion pipeline.
//!
//! The pipeline is three ONNX models run via `ort` (ContentVec + RMVPE +
//! trained generator). It converts a source voice's **timbre** to the target
//! while preserving content and pitch — which is exactly why breathy/expressive
//! speech survives: it lives in the content and F0 streams, not the timbre
//! the generator replaces.
//!
//! Audio crosses every boundary as mono `f32`, and the conversion path is
//! [`futures::Stream`]-in → [`futures::Stream`]-out ([`convert::convert_stream`]),
//! so the conversion CLI is a plain Unix filter and a realtime service is a thin
//! wrapper.

mod analysis;
mod backend;
mod config;
mod convert;
mod denoise;
mod dsp;
mod encoder;
mod error;
mod f0;
mod features;
mod mel;
mod rvc;
mod session;

#[cfg(feature = "burn")]
mod burn_backend;
#[cfg(feature = "burn")]
mod burn_features;

#[cfg(test)]
mod model_probe;

pub use analysis::{ContentEncoder, PitchEstimator};
pub use backend::Generator;
#[cfg(feature = "burn")]
pub use burn_backend::BurnGenerator;
#[cfg(feature = "cuda")]
pub use burn_backend::cuda_generator;
#[cfg(feature = "tch")]
pub use burn_backend::libtorch_generator;
#[cfg(feature = "wgpu")]
pub use burn_backend::wgpu_generator;
#[cfg(feature = "burn")]
pub use burn_features::{BurnContentEncoder, BurnPitchEstimator};
#[cfg(feature = "cuda")]
pub use burn_features::{cuda_content_encoder, cuda_pitch_estimator};
#[cfg(feature = "tch")]
pub use burn_features::{libtorch_content_encoder, libtorch_pitch_estimator};
#[cfg(feature = "wgpu")]
pub use burn_features::{wgpu_content_encoder, wgpu_pitch_estimator};
pub use config::{
    ANALYSIS_SR, CONTENT_DIM, CONTENT_HOP, ConvertParams, F0_HOP, GeneratorIo, ModelPaths,
    RMVPE_BINS, RvcConfig,
};
pub use convert::{Converter, StreamParams, convert_stream};
pub use denoise::{DenoiseParams, Denoiser};
pub use dsp::{f0_to_coarse, shift_pitch, upsample_rows};
pub use error::{Result, VcError};
pub use features::{
    DEFAULT_CHUNK, FeatureExtractor, Features, onnx_content_encoder, onnx_pitch_estimator,
};
pub use rvc::RvcModel;
pub use session::build_session;
