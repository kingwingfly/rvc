//! The prior / text encoder (`TextEncoder768` + `attentions.Encoder`).
//!
//! Faithful port of the RVC reference. The relative-position attention follows
//! `MultiHeadAttention` exactly so weights load unchanged. Sequence masking is
//! omitted: inference and ONNX export run one clip at a time (full length), and
//! batched training pads to equal lengths — add masks there if batches mix
//! lengths.

use burn::module::{Module, Param};
use burn::nn::conv::{Conv1d, Conv1dConfig};
use burn::nn::{Embedding, EmbeddingConfig, Linear, LinearConfig, PaddingConfig1d};
use burn::tensor::activation::{relu, softmax};
use burn::tensor::backend::Backend;
use burn::tensor::{Distribution, Tensor};

use crate::config::SynthesizerConfig;
use crate::nn::{RvcLayerNorm, leaky_relu};

/// Multi-head self-attention with relative position embeddings.
#[derive(Module, Debug)]
pub struct MultiHeadAttention<B: Backend> {
    conv_q: Conv1d<B>,
    conv_k: Conv1d<B>,
    conv_v: Conv1d<B>,
    conv_o: Conv1d<B>,
    emb_rel_k: Param<Tensor<B, 3>>,
    emb_rel_v: Param<Tensor<B, 3>>,
    n_heads: usize,
    k_channels: usize,
    window_size: usize,
}

impl<B: Backend> MultiHeadAttention<B> {
    fn new(channels: usize, n_heads: usize, window_size: usize, device: &B::Device) -> Self {
        let k_channels = channels / n_heads;
        let conv = || Conv1dConfig::new(channels, channels, 1).init(device);
        let rel_stddev = (k_channels as f64).powf(-0.5);
        let rel = || {
            Param::from_tensor(Tensor::random(
                [1, window_size * 2 + 1, k_channels],
                Distribution::Normal(0.0, rel_stddev),
                device,
            ))
        };
        Self {
            conv_q: conv(),
            conv_k: conv(),
            conv_v: conv(),
            conv_o: conv(),
            emb_rel_k: rel(),
            emb_rel_v: rel(),
            n_heads,
            k_channels,
            window_size,
        }
    }

    /// `x`: `[batch, channels, time]` self-attention → `[batch, channels, time]`.
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let q = self.conv_q.forward(x.clone());
        let k = self.conv_k.forward(x.clone());
        let v = self.conv_v.forward(x);
        let out = self.attention(q, k, v);
        self.conv_o.forward(out)
    }

    fn attention(&self, q: Tensor<B, 3>, k: Tensor<B, 3>, v: Tensor<B, 3>) -> Tensor<B, 3> {
        let [b, d, t] = q.dims();
        let (h, dk) = (self.n_heads, self.k_channels);
        // [b, d, t] -> [b, h, t, dk]
        let reshape = |x: Tensor<B, 3>| x.reshape([b, h, dk, t]).swap_dims(2, 3);
        let q = reshape(q);
        let k = reshape(k);
        let v = reshape(v);

        let scale = (dk as f64).sqrt();
        let q_scaled = q / scale;
        // content scores: [b, h, t, t]
        let mut scores = q_scaled.clone().matmul(k.swap_dims(2, 3));
        // relative-key scores
        let key_rel = self.get_relative_embeddings(self.emb_rel_k.val(), t); // [1, 2t-1, dk]
        let rel_logits = q_scaled.matmul(key_rel.swap_dims(1, 2).unsqueeze::<4>()); // [b,h,t,2t-1]
        scores = scores + Self::rel_to_abs(rel_logits);

        let p_attn = softmax(scores, 3); // [b, h, t, t]
        let mut output = p_attn.clone().matmul(v); // [b, h, t, dk]
        let rel_weights = Self::abs_to_rel(p_attn); // [b, h, t, 2t-1]
        let value_rel = self.get_relative_embeddings(self.emb_rel_v.val(), t); // [1, 2t-1, dk]
        output = output + rel_weights.matmul(value_rel.unsqueeze::<4>()); // [b,h,t,dk]

        output.swap_dims(2, 3).reshape([b, d, t])
    }

    /// Slice/pad the `[1, 2*window+1, dk]` table to `[1, 2*length-1, dk]`.
    fn get_relative_embeddings(&self, emb: Tensor<B, 3>, length: usize) -> Tensor<B, 3> {
        let w = self.window_size;
        let pad = (length as isize - (w as isize + 1)).max(0) as usize;
        let slice_start = ((w as isize + 1) - length as isize).max(0) as usize;
        let slice_end = slice_start + 2 * length - 1;
        let emb = if pad > 0 { pad_dim1(emb, pad, pad) } else { emb };
        let [_, n, dk] = emb.dims();
        let _ = n;
        emb.slice([0..1, slice_start..slice_end, 0..dk])
    }

    /// `[b, h, l, 2l-1]` relative → `[b, h, l, l]` absolute indexing.
    fn rel_to_abs(x: Tensor<B, 4>) -> Tensor<B, 4> {
        let [b, h, l, _] = x.dims();
        let x = pad_last(x, 0, 1); // [b,h,l,2l]
        let x = x.reshape([b, h, l * 2 * l]);
        let x = pad_last(x, 0, l - 1); // [b,h, 2l*l + l-1]
        x.reshape([b, h, l + 1, 2 * l - 1])
            .slice([0..b, 0..h, 0..l, (l - 1)..(2 * l - 1)])
    }

    /// `[b, h, l, l]` absolute → `[b, h, l, 2l-1]` relative indexing.
    fn abs_to_rel(x: Tensor<B, 4>) -> Tensor<B, 4> {
        let [b, h, l, _] = x.dims();
        let x = pad_last(x, 0, l - 1); // [b,h,l,2l-1]
        let x = x.reshape([b, h, l * l + l * (l - 1)]);
        let x = pad_last(x, l, 0); // prepend l zeros
        x.reshape([b, h, l, 2 * l]).slice([0..b, 0..h, 0..l, 1..(2 * l)])
    }
}

/// Zero-pad the last dim of a rank-3 or rank-4 tensor by `(left, right)`.
fn pad_last<B: Backend, const D: usize>(x: Tensor<B, D>, left: usize, right: usize) -> Tensor<B, D> {
    if left == 0 && right == 0 {
        return x;
    }
    let mut parts = Vec::new();
    let mut dims = x.dims();
    let device = x.device();
    if left > 0 {
        dims[D - 1] = left;
        parts.push(Tensor::zeros(dims, &device));
    }
    parts.push(x.clone());
    if right > 0 {
        let mut dims = x.dims();
        dims[D - 1] = right;
        parts.push(Tensor::zeros(dims, &device));
    }
    Tensor::cat(parts, D - 1)
}

/// Zero-pad dim 1 of a `[1, n, dk]` tensor by `(left, right)`.
fn pad_dim1<B: Backend>(x: Tensor<B, 3>, left: usize, right: usize) -> Tensor<B, 3> {
    let [a, _, c] = x.dims();
    let device = x.device();
    let l = Tensor::zeros([a, left, c], &device);
    let r = Tensor::zeros([a, right, c], &device);
    Tensor::cat(vec![l, x, r], 1)
}

/// Position-wise feed-forward (`FFN`): conv → relu → conv, "same" padded.
#[derive(Module, Debug)]
pub struct Ffn<B: Backend> {
    conv_1: Conv1d<B>,
    conv_2: Conv1d<B>,
}

impl<B: Backend> Ffn<B> {
    fn new(channels: usize, filter_channels: usize, kernel: usize, device: &B::Device) -> Self {
        // RVC's "same" padding: (left, right) = ((k-1)/2, k/2).
        let pad = PaddingConfig1d::Explicit((kernel - 1) / 2, kernel / 2);
        Self {
            conv_1: Conv1dConfig::new(channels, filter_channels, kernel)
                .with_padding(pad.clone())
                .init(device),
            conv_2: Conv1dConfig::new(filter_channels, channels, kernel)
                .with_padding(pad)
                .init(device),
        }
    }

    /// `x`: `[batch, channels, time]`.
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let x = relu(self.conv_1.forward(x));
        self.conv_2.forward(x)
    }
}

/// One transformer block: attention + FFN with pre-residual LayerNorms.
#[derive(Module, Debug)]
struct EncoderLayer<B: Backend> {
    attn: MultiHeadAttention<B>,
    norm_1: RvcLayerNorm<B>,
    ffn: Ffn<B>,
    norm_2: RvcLayerNorm<B>,
}

/// The transformer stack (`attentions.Encoder`).
#[derive(Module, Debug)]
pub struct Encoder<B: Backend> {
    layers: Vec<EncoderLayer<B>>,
}

impl<B: Backend> Encoder<B> {
    fn new(cfg: &SynthesizerConfig, device: &B::Device) -> Self {
        let layers = (0..cfg.n_layers)
            .map(|_| EncoderLayer {
                attn: MultiHeadAttention::new(
                    cfg.hidden_channels,
                    cfg.n_heads,
                    cfg.window_size,
                    device,
                ),
                norm_1: RvcLayerNorm::new(cfg.hidden_channels, device),
                ffn: Ffn::new(cfg.hidden_channels, cfg.filter_channels, cfg.kernel_size, device),
                norm_2: RvcLayerNorm::new(cfg.hidden_channels, device),
            })
            .collect();
        Self { layers }
    }

    /// `x`: `[batch, hidden, time]`.
    pub fn forward(&self, mut x: Tensor<B, 3>) -> Tensor<B, 3> {
        for layer in &self.layers {
            let y = layer.attn.forward(x.clone());
            x = layer.norm_1.forward(x + y);
            let y = layer.ffn.forward(x.clone());
            x = layer.norm_2.forward(x + y);
        }
        x
    }
}

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
            encoder: Encoder::new(cfg, device),
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
