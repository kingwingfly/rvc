//! The RVC prior encoder (`TextEncoder768`).
//!
//! The one piece of RVC's encoder that is RVC's: a 768-dim ContentVec
//! projection and a pitch embedding in front of the shared VITS attention stack.

use burn::module::Module;
use burn::nn::conv::{Conv1d, Conv1dConfig};
use burn::nn::{Embedding, EmbeddingConfig, Linear, LinearConfig};
use burn::tensor::Tensor;
use burn::tensor::backend::Backend;
use burn_vits::{Encoder, leaky_relu};

use crate::config::SynthesizerConfig;

/// The prior encoder `TextEncoder768`: content + pitch → `(m_p, logs_p)`.
#[derive(Module, Debug)]
pub struct TextEncoder<B: Backend> {
    emb_phone: Linear<B>,
    emb_pitch: Embedding<B>,
    encoder: Encoder<B>,
    proj: Conv1d<B>,
    hidden_channels: usize,
    out_channels: usize,
}

impl<B: Backend> TextEncoder<B> {
    /// Build from the synthesizer config.
    pub fn new(cfg: &SynthesizerConfig, device: &B::Device) -> Self {
        Self {
            emb_phone: LinearConfig::new(768, cfg.hidden_channels).init(device),
            emb_pitch: EmbeddingConfig::new(256, cfg.hidden_channels).init(device),
            encoder: Encoder::new(&cfg.encoder(), device),
            proj: Conv1dConfig::new(cfg.hidden_channels, cfg.inter_channels * 2, 1).init(device),
            hidden_channels: cfg.hidden_channels,
            out_channels: cfg.inter_channels,
        }
    }

    /// `phone`: `[batch, time, 768]`, `pitch`: `[batch, time]` (coarse pitch ids).
    /// Returns `(m, logs)`, each `[batch, inter_channels, time]`.
    pub fn forward(
        &self,
        phone: Tensor<B, 3>,
        pitch: Tensor<B, 2, burn::tensor::Int>,
    ) -> (Tensor<B, 3>, Tensor<B, 3>) {
        let x = self.emb_phone.forward(phone) + self.emb_pitch.forward(pitch);
        let x = x * (self.hidden_channels as f64).sqrt();
        let x = leaky_relu(x, 0.1);
        let x = x.swap_dims(1, 2); // [b, hidden, t]
        let x = self.encoder.forward(x);
        let stats = self.proj.forward(x);
        let m = stats.clone().narrow(1, 0, self.out_channels);
        let logs = stats.narrow(1, self.out_channels, self.out_channels);
        (m, logs)
    }
}
