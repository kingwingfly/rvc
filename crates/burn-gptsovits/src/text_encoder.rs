//! `enc_p` — semantic tokens and phonemes to the prior the flow inverts.
//!
//! Three attention stacks and a cross-attention between them. The semantic side
//! and the text side are encoded separately, the MRTE lets the semantic side
//! attend over the text, and a third stack runs over the result.
//!
//! The stacks are `burn-vits`' `Encoder` unchanged. The MRTE is the one
//! mechanism here that is GPT-SoVITS's own.

use burn::module::Module;
use burn::nn::conv::{Conv1d, Conv1dConfig};
use burn::nn::{Embedding, EmbeddingConfig};
use burn::tensor::activation::softmax;
use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor};
use burn_vits::{Encoder, EncoderConfig};

/// The shape of `enc_p`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TextEncoderConfig {
    /// Phoneme vocabulary — 732 for v2, and `text-kit`'s table must agree.
    pub n_symbols: usize,
    /// SSL feature width in — cnhubert's 768.
    pub ssl_dim: usize,
    pub hidden_channels: usize,
    pub filter_channels: usize,
    pub n_heads: usize,
    /// Depth of the *text* stack. The semantic and output stacks are half this,
    /// which is upstream's `n_layers // 2`.
    pub n_layers: usize,
    pub kernel_size: usize,
    /// Latent width out. The projection emits twice this — a mean and a scale.
    pub out_channels: usize,
    /// Width the MRTE cross-attention works in.
    pub mrte_hidden: usize,
    pub mrte_heads: usize,
}

impl Default for TextEncoderConfig {
    fn default() -> Self {
        Self {
            n_symbols: 732,
            ssl_dim: 768,
            hidden_channels: 192,
            filter_channels: 768,
            n_heads: 2,
            n_layers: 6,
            kernel_size: 3,
            out_channels: 192,
            mrte_hidden: 512,
            mrte_heads: 4,
        }
    }
}

impl TextEncoderConfig {
    fn stack(&self, layers: usize) -> EncoderConfig {
        EncoderConfig {
            hidden_channels: self.hidden_channels,
            filter_channels: self.filter_channels,
            n_heads: self.n_heads,
            n_layers: layers,
            kernel_size: self.kernel_size,
            window_size: 4,
        }
    }
}

/// The multi-reference timbre encoder's cross-attention.
///
/// Its own module rather than `burn-vits`' because it has **no
/// relative-position embeddings** — the checkpoint has `conv_q/k/v/o` and no
/// `emb_rel_k`/`emb_rel_v`. Relative positions would make no sense here anyway:
/// query and key live in different sequences, so the distance between two of
/// their indices means nothing.
#[derive(Module, Debug)]
pub struct CrossAttention<B: Backend> {
    conv_q: Conv1d<B>,
    conv_k: Conv1d<B>,
    conv_v: Conv1d<B>,
    conv_o: Conv1d<B>,
    n_heads: usize,
}

impl<B: Backend> CrossAttention<B> {
    fn new(channels: usize, n_heads: usize, device: &B::Device) -> Self {
        let conv = || Conv1dConfig::new(channels, channels, 1).init(device);
        Self {
            conv_q: conv(),
            conv_k: conv(),
            conv_v: conv(),
            conv_o: conv(),
            n_heads,
        }
    }

    /// Query from `x`, keys and values from `context`; both
    /// `[batch, channels, time]`, with independent lengths.
    fn forward(&self, x: Tensor<B, 3>, context: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch, channels, t_q] = x.dims();
        let t_kv = context.dims()[2];
        let d_head = channels / self.n_heads;

        // [batch, channels, time] -> [batch, head, time, d_head]
        let heads = |t: Tensor<B, 3>, time: usize| {
            t.reshape([batch, self.n_heads, d_head, time])
                .swap_dims(2, 3)
        };
        let q = heads(self.conv_q.forward(x), t_q) / (d_head as f64).sqrt();
        let k = heads(self.conv_k.forward(context.clone()), t_kv);
        let v = heads(self.conv_v.forward(context), t_kv);

        let out = softmax(q.matmul(k.swap_dims(2, 3)), 3)
            .matmul(v)
            .swap_dims(2, 3)
            .reshape([batch, channels, t_q]);
        self.conv_o.forward(out)
    }
}

/// The multi-reference timbre encoder.
///
/// Widens both sides to its own working width, lets the semantic sequence attend
/// over the text, adds the speaker vector, and narrows back. The residual is on
/// the semantic side, so text conditions the result without replacing it.
#[derive(Module, Debug)]
pub struct Mrte<B: Backend> {
    cross_attention: CrossAttention<B>,
    c_pre: Conv1d<B>,
    text_pre: Conv1d<B>,
    c_post: Conv1d<B>,
}

impl<B: Backend> Mrte<B> {
    fn new(cfg: &TextEncoderConfig, device: &B::Device) -> Self {
        let hidden = cfg.mrte_hidden;
        Self {
            cross_attention: CrossAttention::new(hidden, cfg.mrte_heads, device),
            c_pre: Conv1dConfig::new(cfg.hidden_channels, hidden, 1).init(device),
            text_pre: Conv1dConfig::new(cfg.hidden_channels, hidden, 1).init(device),
            c_post: Conv1dConfig::new(hidden, cfg.hidden_channels, 1).init(device),
        }
    }

    /// `ssl` and `text`: `[batch, hidden_channels, time]` with independent
    /// lengths; `g`: `[batch, mrte_hidden, 1]`, broadcast across time.
    fn forward(&self, ssl: Tensor<B, 3>, text: Tensor<B, 3>, g: Tensor<B, 3>) -> Tensor<B, 3> {
        let ssl = self.c_pre.forward(ssl);
        let text = self.text_pre.forward(text);
        let x = self.cross_attention.forward(ssl.clone(), text) + ssl + g;
        self.c_post.forward(x)
    }
}

/// `enc_p`.
#[derive(Module, Debug)]
pub struct TextEncoder<B: Backend> {
    ssl_proj: Conv1d<B>,
    encoder_ssl: Encoder<B>,
    text_embedding: Embedding<B>,
    encoder_text: Encoder<B>,
    mrte: Mrte<B>,
    encoder2: Encoder<B>,
    proj: Conv1d<B>,
    out_channels: usize,
}

impl<B: Backend> TextEncoder<B> {
    pub fn new(cfg: &TextEncoderConfig, device: &B::Device) -> Self {
        Self {
            ssl_proj: Conv1dConfig::new(cfg.ssl_dim, cfg.hidden_channels, 1).init(device),
            encoder_ssl: Encoder::new(&cfg.stack(cfg.n_layers / 2), device),
            text_embedding: EmbeddingConfig::new(cfg.n_symbols, cfg.hidden_channels).init(device),
            encoder_text: Encoder::new(&cfg.stack(cfg.n_layers), device),
            mrte: Mrte::new(cfg, device),
            encoder2: Encoder::new(&cfg.stack(cfg.n_layers / 2), device),
            proj: Conv1dConfig::new(cfg.hidden_channels, cfg.out_channels * 2, 1).init(device),
            out_channels: cfg.out_channels,
        }
    }

    /// `ssl`: `[batch, ssl_dim, frames]` — quantised semantic features.
    /// `text`: `[batch, phones]` — phoneme ids from `text-kit`.
    /// `g`: `[batch, mrte_hidden, 1]` — the speaker vector.
    ///
    /// Returns `(m, logs)`, the prior the flow is inverted into.
    pub fn forward(
        &self,
        ssl: Tensor<B, 3>,
        text: Tensor<B, 2, Int>,
        g: Tensor<B, 3>,
    ) -> (Tensor<B, 3>, Tensor<B, 3>) {
        let y = self.encoder_ssl.forward(self.ssl_proj.forward(ssl));

        // The embedding yields [batch, phones, hidden]; every stack here works
        // channel-first.
        let text = self.text_embedding.forward(text).swap_dims(1, 2);
        let text = self.encoder_text.forward(text);

        let y = self.mrte.forward(y, text, g);
        let y = self.encoder2.forward(y);

        let stats = self.proj.forward(y);
        let [batch, _, time] = stats.dims();
        let m = stats
            .clone()
            .slice([0..batch, 0..self.out_channels, 0..time]);
        let logs = stats.slice([0..batch, self.out_channels..self.out_channels * 2, 0..time]);
        (m, logs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::TensorData;

    type B = burn_ndarray::NdArray;

    #[test]
    fn the_prior_follows_the_semantic_sequence_not_the_text() {
        // The residual in the MRTE is on the semantic side, so the output is one
        // vector per semantic frame however many phonemes came in. Getting this
        // backwards would make the decoder produce audio the length of the
        // transcript rather than of the utterance.
        let cfg = TextEncoderConfig::default();
        let device = Default::default();
        let enc = TextEncoder::<B>::new(&cfg, &device);

        let frames = 12;
        let phones = 5;
        let ssl = Tensor::zeros([1, cfg.ssl_dim, frames], &device);
        let text = Tensor::<B, 2, Int>::from_data(
            TensorData::new(vec![1i32, 2, 3, 4, 5], [1, phones]),
            &device,
        );
        let g = Tensor::zeros([1, cfg.mrte_hidden, 1], &device);

        let (m, logs) = enc.forward(ssl, text, g);
        assert_eq!(m.dims(), [1, cfg.out_channels, frames]);
        assert_eq!(logs.dims(), [1, cfg.out_channels, frames]);
    }

    #[test]
    fn cross_attention_accepts_differing_sequence_lengths() {
        // The whole point of it: the semantic and text sequences have no reason
        // to be the same length, and self-attention would require they were.
        let device = Default::default();
        let attn = CrossAttention::<B>::new(16, 4, &device);
        let x = Tensor::zeros([1, 16, 7], &device);
        let ctx = Tensor::zeros([1, 16, 3], &device);
        assert_eq!(attn.forward(x, ctx).dims(), [1, 16, 7]);
    }
}
