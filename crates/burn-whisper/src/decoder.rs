//! The text decoder: tokens so far + encoder output → logits for the next token.
//!
//! Autoregressive, so it runs once per generated token and the keys and values
//! of everything before are worth keeping. [`DecodeState`] holds those; the
//! module itself stays stateless.

use burn::module::Module;
use burn::nn::{Embedding, EmbeddingConfig, LayerNorm, LayerNormConfig, Linear, LinearConfig};
use burn::tensor::activation::gelu;
use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor};

use crate::attention::{Attention, KvCache, causal_mask};
use crate::config::WhisperConfig;

/// Everything one in-progress decode needs to remember between steps.
///
/// One entry per layer, each holding the self-attention cache (grows by a token
/// per step) and the cross-attention cache (computed once from the encoder
/// output and then constant). `offset` is how many tokens have been fed, which
/// is where the next positional embedding is read from.
#[derive(Debug, Clone)]
pub struct DecodeState<B: Backend> {
    self_attn: Vec<Option<KvCache<B>>>,
    cross_attn: Vec<Option<KvCache<B>>>,
    offset: usize,
}

impl<B: Backend> DecodeState<B> {
    pub fn new(layers: usize) -> Self {
        Self {
            self_attn: vec![None; layers],
            cross_attn: vec![None; layers],
            offset: 0,
        }
    }

    /// Tokens consumed so far.
    pub fn offset(&self) -> usize {
        self.offset
    }
}

#[derive(Module, Debug)]
pub struct DecoderLayer<B: Backend> {
    self_attn: Attention<B>,
    self_attn_layer_norm: LayerNorm<B>,
    encoder_attn: Attention<B>,
    encoder_attn_layer_norm: LayerNorm<B>,
    fc1: Linear<B>,
    fc2: Linear<B>,
    final_layer_norm: LayerNorm<B>,
}

impl<B: Backend> DecoderLayer<B> {
    fn new(cfg: &WhisperConfig, device: &B::Device) -> Self {
        Self {
            self_attn: Attention::new(cfg.d_model, cfg.decoder_attention_heads, device),
            self_attn_layer_norm: LayerNormConfig::new(cfg.d_model).init(device),
            encoder_attn: Attention::new(cfg.d_model, cfg.decoder_attention_heads, device),
            encoder_attn_layer_norm: LayerNormConfig::new(cfg.d_model).init(device),
            fc1: LinearConfig::new(cfg.d_model, cfg.decoder_ffn_dim).init(device),
            fc2: LinearConfig::new(cfg.decoder_ffn_dim, cfg.d_model).init(device),
            final_layer_norm: LayerNormConfig::new(cfg.d_model).init(device),
        }
    }

    fn forward(
        &self,
        x: Tensor<B, 3>,
        xa: Tensor<B, 3>,
        self_cache: &mut Option<KvCache<B>>,
        cross_cache: &mut Option<KvCache<B>>,
    ) -> Tensor<B, 3> {
        let [_, n_q, _] = x.dims();
        let n_kv = n_q + self_cache.as_ref().map_or(0, |c| c.k.dims()[1]);

        let h = self.self_attn.forward_cached(
            self.self_attn_layer_norm.forward(x.clone()),
            Some(causal_mask(n_q, n_kv, &x.device())),
            self_cache,
        );
        let x = x + h;

        let h = self.encoder_attn.forward_cross(
            self.encoder_attn_layer_norm.forward(x.clone()),
            xa,
            cross_cache,
        );
        let x = x + h;

        let h = self.fc2.forward(gelu(
            self.fc1.forward(self.final_layer_norm.forward(x.clone())),
        ));
        x + h
    }
}

#[derive(Module, Debug)]
pub struct TextDecoder<B: Backend> {
    embed_tokens: Embedding<B>,
    embed_positions: Embedding<B>,
    layers: Vec<DecoderLayer<B>>,
    layer_norm: LayerNorm<B>,
}

impl<B: Backend> TextDecoder<B> {
    pub fn new(cfg: &WhisperConfig, device: &B::Device) -> Self {
        Self {
            embed_tokens: EmbeddingConfig::new(cfg.vocab_size, cfg.d_model).init(device),
            embed_positions: EmbeddingConfig::new(cfg.max_target_positions, cfg.d_model)
                .init(device),
            layers: (0..cfg.decoder_layers)
                .map(|_| DecoderLayer::new(cfg, device))
                .collect(),
            layer_norm: LayerNormConfig::new(cfg.d_model).init(device),
        }
    }

    /// One decoding step: `tokens` `[batch, seq]` against encoder output `xa`,
    /// yielding logits `[batch, seq, vocab]` for the positions just fed.
    ///
    /// `state` is advanced, so the caller passes only the *new* tokens on each
    /// call — the whole prompt first, then one token at a time.
    pub fn forward(
        &self,
        tokens: Tensor<B, 2, Int>,
        xa: Tensor<B, 3>,
        state: &mut DecodeState<B>,
    ) -> Tensor<B, 3> {
        let [_, seq] = tokens.dims();

        let pos = self.embed_positions.weight.val();
        let [_, d_model] = pos.dims();
        let mut x = self.embed_tokens.forward(tokens)
            + pos
                .slice([state.offset..state.offset + seq, 0..d_model])
                .unsqueeze::<3>();

        for (i, layer) in self.layers.iter().enumerate() {
            x = layer.forward(
                x,
                xa.clone(),
                &mut state.self_attn[i],
                &mut state.cross_attn[i],
            );
        }
        state.offset += seq;

        // The output projection is the token embedding transposed — tied, and so
        // absent from the checkpoint.
        let x = self.layer_norm.forward(x);
        // `val()` hands out a clone while the parameter stays live in the
        // module, and on LibTorch a `swap_dims` view claims a storage handle
        // burn-tch believes is exclusive — so this is one in-place op away from
        // scribbling on the embedding table. Its only consumer is `matmul`,
        // which allocates its output and mutates neither operand; `tch_aliasing`
        // in `burn-rmvpe` pins that, and CLAUDE.md's **`swap_dims` on LibTorch
        // returns a view burn-tch forgets the provenance of** is the entry.
        let vocab = self.embed_tokens.weight.val().swap_dims(0, 1);
        x.matmul(vocab.unsqueeze::<3>())
    }

    /// A fresh [`DecodeState`] sized for this decoder.
    pub fn state(&self) -> DecodeState<B> {
        DecodeState::new(self.layers.len())
    }
}
