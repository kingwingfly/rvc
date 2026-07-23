//! RVC v2 (Retrieval-based Voice Conversion) synthesizer, implemented in
//! [Burn](https://burn.dev).
//!
//! A faithful port of the RVC-Project reference generator
//! (`SynthesizerTrnMs768NSFsid`) so that the public pretrained weights load
//! unchanged and warm-start native training. Modules mirror the reference
//! `state_dict` layout:
//!
//! - [`TextEncoder`] — `enc_p` (prior/content encoder).
//! - [`PosteriorEncoder`] — `enc_q` (training-only).
//! - [`ResidualCouplingBlock`] — `flow`.
//! - [`GeneratorNsf`] — `dec` (NSF-HiFiGAN decoder).
//!
//! The model is generic over the Burn [`Backend`](burn::tensor::backend::Backend).

mod config;
mod discriminator;
mod flow;
mod generator;
mod nn;
mod posterior;
mod store;
mod synthesizer;
mod text_encoder;
mod wavenet;
mod weightnorm;

pub use config::SynthesizerConfig;
pub use discriminator::{DiscriminatorP, DiscriminatorS, MultiPeriodDiscriminator};
pub use flow::{ResidualCouplingBlock, ResidualCouplingLayer};
pub use generator::{GeneratorNsf, ResBlock1, SourceModule};
pub use nn::RvcLayerNorm;
pub use posterior::PosteriorEncoder;
pub use synthesizer::{Synthesizer, TrainForward};
pub use text_encoder::{Encoder, Ffn, MultiHeadAttention, TextEncoder};
pub use wavenet::Wn;
pub use weightnorm::{WeightNormConv1d, WeightNormConv2d, WeightNormConvTranspose1d};
