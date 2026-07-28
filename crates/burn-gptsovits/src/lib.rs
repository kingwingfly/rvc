//! The GPT-SoVITS network in [Burn](https://burn.dev).
//!
//! Built in the order the pieces depend on each other:
//!
//! 1. the two encoders — [`hubert`] for the audio side, [`bert`] for prosody,
//! 2. the VQ + SoVITS stage that turns semantic tokens into waveform,
//! 3. the autoregressive T2S transformer that produces those tokens.
//!
//! Most of the second stage is a modified VITS, so it is assembled from
//! `burn-vits` rather than written again; both projects descend from the same
//! source, which is why their `state_dict` names line up.
//!
//! Nothing here names a compute backend, and there are no app dependencies.

mod decoder;
mod hubert;
mod quantizer;
mod reference;
mod sovits;
mod t2s;
mod text_encoder;

pub use decoder::{Decoder, DecoderConfig};
pub use hubert::{Hubert, HubertConfig};
pub use quantizer::{Codebook, Quantizer, QuantizerConfig, Vq};
pub use reference::{ReferenceConfig, ReferenceEncoder};
pub use sovits::{SovitsConfig, SovitsPartial};
pub use t2s::{T2s, T2sConfig, T2sState};
pub use text_encoder::{TextEncoder, TextEncoderConfig};
