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

mod hubert;
mod quantizer;
mod sovits;

pub use hubert::{Hubert, HubertConfig};
pub use quantizer::{Codebook, Quantizer, QuantizerConfig, Vq};
pub use sovits::{SovitsConfig, SovitsPartial};
