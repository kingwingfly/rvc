//! The audio encoder: log-mel spectrogram → 1500 frames of acoustic features.
//!
//! Runs once per 30 s window regardless of how much text comes out, which is why
//! large-v3-turbo keeps all 32 layers here and cuts the decoder to 4.

use burn::module::Module;
use burn::nn::PaddingConfig1d;
use burn::nn::conv::{Conv1d, Conv1dConfig};
use burn::nn::{Embedding, EmbeddingConfig, LayerNorm, LayerNormConfig, Linear, LinearConfig};
use burn::tensor::Tensor;
use burn::tensor::activation::gelu;
use burn::tensor::backend::Backend;

use crate::attention::Attention;
use crate::config::WhisperConfig;

#[derive(Module, Debug)]
pub struct EncoderLayer<B: Backend> {
    self_attn: Attention<B>,
    self_attn_layer_norm: LayerNorm<B>,
    fc1: Linear<B>,
    fc2: Linear<B>,
    final_layer_norm: LayerNorm<B>,
}

impl<B: Backend> EncoderLayer<B> {
    fn new(cfg: &WhisperConfig, device: &B::Device) -> Self {
        Self {
            self_attn: Attention::new(cfg.d_model, cfg.encoder_attention_heads, device),
            self_attn_layer_norm: LayerNormConfig::new(cfg.d_model).init(device),
            fc1: LinearConfig::new(cfg.d_model, cfg.encoder_ffn_dim).init(device),
            fc2: LinearConfig::new(cfg.encoder_ffn_dim, cfg.d_model).init(device),
            final_layer_norm: LayerNormConfig::new(cfg.d_model).init(device),
        }
    }

    /// Pre-norm residual block: normalise, transform, add.
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let h = self
            .self_attn
            .forward(self.self_attn_layer_norm.forward(x.clone()), None);
        let x = x + h;
        let h = self.fc2.forward(gelu(
            self.fc1.forward(self.final_layer_norm.forward(x.clone())),
        ));
        x + h
    }
}

#[derive(Module, Debug)]
pub struct AudioEncoder<B: Backend> {
    conv1: Conv1d<B>,
    conv2: Conv1d<B>,
    embed_positions: Embedding<B>,
    layers: Vec<EncoderLayer<B>>,
    layer_norm: LayerNorm<B>,
}

impl<B: Backend> AudioEncoder<B> {
    pub fn new(cfg: &WhisperConfig, device: &B::Device) -> Self {
        Self {
            conv1: Conv1dConfig::new(cfg.num_mel_bins, cfg.d_model, 3)
                .with_padding(PaddingConfig1d::Explicit(1, 1))
                .init(device),
            // Stride 2: 3000 mel frames (30 s at 100 fps) become 1500.
            conv2: Conv1dConfig::new(cfg.d_model, cfg.d_model, 3)
                .with_stride(2)
                .with_padding(PaddingConfig1d::Explicit(1, 1))
                .init(device),
            // Sinusoidal, but stored in the checkpoint as a plain table rather
            // than recomputed — so it is loaded, not derived.
            embed_positions: EmbeddingConfig::new(cfg.max_source_positions, cfg.d_model)
                .init(device),
            layers: (0..cfg.encoder_layers)
                .map(|_| EncoderLayer::new(cfg, device))
                .collect(),
            layer_norm: LayerNormConfig::new(cfg.d_model).init(device),
        }
    }

    /// `mel`: `[batch, n_mels, frames]` → `[batch, frames / 2, d_model]`.
    pub fn forward(&self, mel: Tensor<B, 3>) -> Tensor<B, 3> {
        let x = gelu(self.conv1.forward(mel));
        let x = gelu(self.conv2.forward(x));
        let x = x.swap_dims(1, 2); // [batch, frames, d_model]

        let [_, frames, _] = x.dims();
        let pos = self.embed_positions.weight.val();
        let [_, d_model] = pos.dims();
        let mut x = x + pos.slice([0..frames, 0..d_model]).unsqueeze::<3>();

        for layer in &self.layers {
            x = layer.forward(x);
        }
        self.layer_norm.forward(x)
    }
}
