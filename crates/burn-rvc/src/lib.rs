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
//! [`ContentVec`] joins them as the *input* side: the content encoder whose
//! frames `enc_p` consumes. It is a HuBERT checkpoint, so the network itself
//! lives in `burn-hubert` — shared with GPT-SoVITS's cnhubert — and this crate
//! contributes only RVC's readout of it.
//!
//! The model is generic over the Burn [`Backend`](burn::tensor::backend::Backend).

mod config;
mod contentvec;
mod generator;
mod synthesizer;
mod text_encoder;

pub use config::{RVC_V2_PERIODS, SynthesizerConfig};
pub use contentvec::ContentVec;
pub use generator::{GeneratorNsf, SourceModule};
pub use synthesizer::{Synthesizer, TrainForward};
pub use text_encoder::TextEncoder;

// The VITS blocks live in `burn-vits`, shared with GPT-SoVITS. Re-exported so
// this crate still reads as one model to its consumers.
pub use burn_vits::{
    DiscriminatorP, DiscriminatorS, Encoder, Ffn, MultiHeadAttention, MultiPeriodDiscriminator,
    PosteriorEncoder, ResBlock1, ResidualCouplingBlock, ResidualCouplingLayer, VitsLayerNorm,
    WeightNormConv1d, WeightNormConv2d, WeightNormConvTranspose1d, Wn,
};
