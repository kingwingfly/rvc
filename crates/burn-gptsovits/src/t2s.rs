//! `s1` — the text-to-semantic transformer.
//!
//! A decoder-only transformer that predicts the semantic tokens `s2` renders.
//! Its input is one sequence built from three pieces: the phonemes, their BERT
//! prosody features, and whatever audio tokens have been generated so far. It
//! runs once per token, so this is where synthesis spends its time.
//!
//! Two details that load fine when wrong and are therefore spelled out at the
//! point they matter: the layers are **post**-norm (`norm_first=False`, and no
//! final norm — the checkpoint has no `h.norm`), and the attention projections
//! are **fused** into one `in_proj_weight` of `3 * d_model` rows.

use burn::module::{Module, Param};
use burn::nn::{Embedding, EmbeddingConfig, LayerNorm, LayerNormConfig, Linear, LinearConfig};
use burn::tensor::activation::{relu, softmax};
use burn::tensor::backend::Backend;
use burn::tensor::{Bool, Int, Tensor, TensorData};

/// The shape of one `s1` checkpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct T2sConfig {
    pub model_dim: usize,
    pub n_head: usize,
    pub n_layer: usize,
    pub ffn_dim: usize,
    /// Phoneme vocabulary — `text-kit`'s 732.
    pub phoneme_vocab_size: usize,
    /// Semantic vocabulary: the 1024 codebook entries plus one end-of-sequence.
    pub vocab_size: usize,
    /// Width of the prosody features `bert_proj` narrows.
    pub bert_dim: usize,
}

impl Default for T2sConfig {
    fn default() -> Self {
        Self {
            model_dim: 512,
            n_head: 16,
            n_layer: 24,
            ffn_dim: 2048,
            phoneme_vocab_size: 732,
            vocab_size: 1025,
            bert_dim: 1024,
        }
    }
}

impl T2sConfig {
    /// The end-of-sequence token — the last of the semantic vocabulary, and the
    /// only id that is not a codebook entry.
    pub fn eos(&self) -> u32 {
        self.vocab_size as u32 - 1
    }
}

/// Sinusoidal positions with one learnable scalar.
///
/// The table itself is fixed; `alpha` decides how loudly it speaks. It is a
/// single trained parameter, which is why this is a module rather than a
/// function.
#[derive(Module, Debug)]
pub struct SinePosition<B: Backend> {
    alpha: Param<Tensor<B, 1>>,
    dim: usize,
}

impl<B: Backend> SinePosition<B> {
    fn new(dim: usize, device: &B::Device) -> Self {
        Self {
            alpha: Param::from_tensor(Tensor::ones([1], device)),
            dim,
        }
    }

    /// Add positions `offset..offset + seq` to `x` (`[batch, seq, dim]`).
    ///
    /// `offset` is what makes incremental decoding work: a token generated
    /// tenth must be told it is tenth, not first.
    fn forward(&self, x: Tensor<B, 3>, offset: usize) -> Tensor<B, 3> {
        let [_, seq, _] = x.dims();
        let device = x.device();

        let mut table = vec![0.0f32; seq * self.dim];
        for (i, row) in table.chunks_mut(self.dim).enumerate() {
            let pos = (offset + i) as f32;
            for k in 0..self.dim / 2 {
                let inv = (-(10_000f32.ln()) * (2 * k) as f32 / self.dim as f32).exp();
                row[2 * k] = (pos * inv).sin();
                row[2 * k + 1] = (pos * inv).cos();
            }
        }
        let pe: Tensor<B, 3> =
            Tensor::from_data(TensorData::new(table, [1, seq, self.dim]), &device);
        x + pe * self.alpha.val().unsqueeze::<3>()
    }
}

/// Cached keys and values for one layer.
#[derive(Debug, Clone)]
pub struct LayerCache<B: Backend> {
    k: Tensor<B, 3>,
    v: Tensor<B, 3>,
}

/// An in-progress generation.
#[derive(Debug, Clone)]
pub struct T2sState<B: Backend> {
    layers: Vec<Option<LayerCache<B>>>,
    offset: usize,
}

impl<B: Backend> T2sState<B> {
    /// Tokens consumed so far — also where the next position index comes from.
    pub fn offset(&self) -> usize {
        self.offset
    }
}

/// Self-attention with the three projections fused, as `nn.MultiheadAttention`
/// stores them.
#[derive(Module, Debug)]
pub struct FusedAttention<B: Backend> {
    /// `[3 * d_model, d_model]` — query, key and value stacked in that order.
    in_proj_weight: Param<Tensor<B, 2>>,
    in_proj_bias: Param<Tensor<B, 1>>,
    out_proj: Linear<B>,
    n_head: usize,
}

impl<B: Backend> FusedAttention<B> {
    fn new(cfg: &T2sConfig, device: &B::Device) -> Self {
        let d = cfg.model_dim;
        Self {
            in_proj_weight: Param::from_tensor(Tensor::zeros([3 * d, d], device)),
            in_proj_bias: Param::from_tensor(Tensor::zeros([3 * d], device)),
            out_proj: LinearConfig::new(d, d).init(device),
            n_head: cfg.n_head,
        }
    }

    /// `x`: `[batch, seq, d_model]`. `cache` carries earlier keys and values;
    /// `mask` blocks attention where true.
    fn forward(
        &self,
        x: Tensor<B, 3>,
        mask: Option<Tensor<B, 2, Bool>>,
        cache: &mut Option<LayerCache<B>>,
    ) -> Tensor<B, 3> {
        let [batch, seq, d_model] = x.dims();
        let d_head = d_model / self.n_head;

        // One matmul for all three projections, then split. `in_proj_weight` is
        // `[out, in]` as PyTorch stores it, so it is transposed here.
        let qkv = x.matmul(
            self.in_proj_weight
                .val()
                .transpose()
                .unsqueeze::<3>()
                .repeat_dim(0, batch),
        ) + self.in_proj_bias.val().unsqueeze::<3>();
        let part = |i: usize| {
            qkv.clone()
                .slice([0..batch, 0..seq, i * d_model..(i + 1) * d_model])
        };
        let (q, k, v) = (part(0), part(1), part(2));

        let (k, v) = match cache.take() {
            Some(prev) => (
                Tensor::cat(vec![prev.k, k], 1),
                Tensor::cat(vec![prev.v, v], 1),
            ),
            None => (k, v),
        };
        *cache = Some(LayerCache {
            k: k.clone(),
            v: v.clone(),
        });
        let n_kv = k.dims()[1];

        let heads = |t: Tensor<B, 3>, len: usize| {
            t.reshape([batch, len, self.n_head, d_head]).swap_dims(1, 2)
        };
        let q = heads(q, seq) / (d_head as f64).sqrt();
        let mut scores = q.matmul(heads(k, n_kv).swap_dims(2, 3));
        if let Some(mask) = mask {
            scores = scores.mask_fill(mask.unsqueeze::<4>(), f32::NEG_INFINITY);
        }
        let out = softmax(scores, 3)
            .matmul(heads(v, n_kv))
            .swap_dims(1, 2)
            .reshape([batch, seq, d_model]);
        self.out_proj.forward(out)
    }
}

/// One post-norm block.
#[derive(Module, Debug)]
pub struct T2sLayer<B: Backend> {
    self_attn: FusedAttention<B>,
    linear1: Linear<B>,
    linear2: Linear<B>,
    norm1: LayerNorm<B>,
    norm2: LayerNorm<B>,
}

impl<B: Backend> T2sLayer<B> {
    fn new(cfg: &T2sConfig, device: &B::Device) -> Self {
        Self {
            self_attn: FusedAttention::new(cfg, device),
            linear1: LinearConfig::new(cfg.model_dim, cfg.ffn_dim).init(device),
            linear2: LinearConfig::new(cfg.ffn_dim, cfg.model_dim).init(device),
            norm1: LayerNormConfig::new(cfg.model_dim).init(device),
            norm2: LayerNormConfig::new(cfg.model_dim).init(device),
        }
    }

    fn forward(
        &self,
        x: Tensor<B, 3>,
        mask: Option<Tensor<B, 2, Bool>>,
        cache: &mut Option<LayerCache<B>>,
    ) -> Tensor<B, 3> {
        // Post-norm: normalise *after* the residual add. The pre-norm arrangement
        // is one line away and loads identically.
        let h = self.self_attn.forward(x.clone(), mask, cache);
        let x = self.norm1.forward(x + h);
        let h = self.linear2.forward(relu(self.linear1.forward(x.clone())));
        self.norm2.forward(x + h)
    }
}

/// A list of layers, named `h` because that is what the checkpoint calls it.
#[derive(Module, Debug)]
pub struct T2sStack<B: Backend> {
    layers: Vec<T2sLayer<B>>,
}

/// The text-to-semantic model.
#[derive(Module, Debug)]
pub struct T2s<B: Backend> {
    bert_proj: Linear<B>,
    ar_text_embedding: TokenEmbedding<B>,
    ar_text_position: SinePosition<B>,
    ar_audio_embedding: TokenEmbedding<B>,
    ar_audio_position: SinePosition<B>,
    h: T2sStack<B>,
    ar_predict_layer: Linear<B>,
    n_layer: usize,
}

/// An embedding under the extra `word_embeddings` level the checkpoint has.
#[derive(Module, Debug)]
pub struct TokenEmbedding<B: Backend> {
    word_embeddings: Embedding<B>,
}

impl<B: Backend> T2s<B> {
    pub fn new(cfg: &T2sConfig, device: &B::Device) -> Self {
        let embedding = |vocab: usize| TokenEmbedding {
            word_embeddings: EmbeddingConfig::new(vocab, cfg.model_dim).init(device),
        };
        Self {
            bert_proj: LinearConfig::new(cfg.bert_dim, cfg.model_dim).init(device),
            ar_text_embedding: embedding(cfg.phoneme_vocab_size),
            ar_text_position: SinePosition::new(cfg.model_dim, device),
            ar_audio_embedding: embedding(cfg.vocab_size),
            ar_audio_position: SinePosition::new(cfg.model_dim, device),
            h: T2sStack {
                layers: (0..cfg.n_layer)
                    .map(|_| T2sLayer::new(cfg, device))
                    .collect(),
            },
            ar_predict_layer: LinearConfig::new(cfg.model_dim, cfg.vocab_size)
                .with_bias(false)
                .init(device),
            n_layer: cfg.n_layer,
        }
    }

    /// A fresh generation state.
    pub fn state(&self) -> T2sState<B> {
        T2sState {
            layers: vec![None; self.n_layer],
            offset: 0,
        }
    }

    /// The prompt: phonemes plus their prosody, embedded and positioned.
    ///
    /// `phones`: `[batch, n]`, `bert`: `[batch, n, bert_dim]` — one feature per
    /// phoneme, which is what `text-kit`'s `word2ph` expansion produces. The two
    /// are **added**, not concatenated: prosody colours each phoneme rather than
    /// extending the sequence.
    pub fn embed_text(&self, phones: Tensor<B, 2, Int>, bert: Tensor<B, 3>) -> Tensor<B, 3> {
        let x =
            self.ar_text_embedding.word_embeddings.forward(phones) + self.bert_proj.forward(bert);
        self.ar_text_position.forward(x, 0)
    }

    /// Embed semantic tokens, positioned from `offset`.
    pub fn embed_audio(&self, tokens: Tensor<B, 2, Int>, offset: usize) -> Tensor<B, 3> {
        let x = self.ar_audio_embedding.word_embeddings.forward(tokens);
        self.ar_audio_position.forward(x, offset)
    }

    /// Run the stack over `x` and return logits for its final position.
    ///
    /// `x` is already embedded — the caller decides whether it is text, audio or
    /// the two concatenated, because the prompt is all three.
    pub fn forward(&self, x: Tensor<B, 3>, state: &mut T2sState<B>) -> Tensor<B, 2> {
        let [batch, seq, _] = x.dims();
        let n_kv = seq + state.offset;
        let mask = causal_mask::<B>(seq, n_kv, &x.device());

        let mut h = x;
        for (i, layer) in self.h.layers.iter().enumerate() {
            h = layer.forward(h, Some(mask.clone()), &mut state.layers[i]);
        }
        state.offset += seq;

        let width = h.dims()[2];
        let last = h.slice([0..batch, seq - 1..seq, 0..width]);
        self.ar_predict_layer.forward(last).squeeze_dim::<2>(1)
    }

    /// Load an `s1*.ckpt`.
    ///
    /// The state dict is under `"weight"` and everything sits below a `model.`
    /// prefix that carries no structure.
    pub fn load_pytorch(
        &mut self,
        path: impl AsRef<std::path::Path>,
    ) -> Result<burn_store::ApplyResult, Box<dyn std::error::Error>> {
        burn_kit::store::load_pytorch_into::<B, _>(
            self,
            path.as_ref(),
            Some("weight"),
            &[(r"^model\.", "")],
        )
    }
}

/// `true` where attention must be blocked — the future, given `n_kv - n_q`
/// cached positions.
///
/// [`Tensor::tril_mask`], not `triu_mask`: Burn names its masks for the triangle
/// they *keep*. The same trap that made the Whisper decoder emit NaN.
fn causal_mask<B: Backend>(n_q: usize, n_kv: usize, device: &B::Device) -> Tensor<B, 2, Bool> {
    Tensor::tril_mask([n_q, n_kv], (n_kv - n_q) as i64, device)
}

#[cfg(test)]
mod tests {
    use super::*;

    type B = burn_ndarray::NdArray;

    fn tiny() -> T2sConfig {
        T2sConfig {
            model_dim: 32,
            n_head: 4,
            n_layer: 2,
            ffn_dim: 64,
            phoneme_vocab_size: 40,
            vocab_size: 20,
            bert_dim: 16,
        }
    }

    #[test]
    fn eos_is_the_last_id_and_not_a_codebook_entry() {
        // Generation stops on it, and it is the only semantic id `s2` cannot
        // decode — off by one and every utterance ends on a real token instead.
        let cfg = T2sConfig::default();
        assert_eq!(cfg.eos(), 1024);
        assert_eq!(cfg.vocab_size, 1025);
    }

    #[test]
    fn incremental_generation_matches_one_shot() {
        // The test that earns its keep, as it did for Whisper. Feeding a prompt
        // whole and feeding it token by token must agree at the final position —
        // true only if the causal mask and the positional offset are both right
        // under a growing cache. Both are invisible to weight coverage.
        let cfg = tiny();
        let device = Default::default();
        let model = T2s::<B>::new(&cfg, &device);

        let ids = [3i32, 7, 1, 9, 4];
        let all: Tensor<B, 2, Int> = Tensor::from_data(TensorData::from([ids]), &device);

        let mut state = model.state();
        let one_shot = model.forward(model.embed_audio(all, 0), &mut state);

        let mut state = model.state();
        let mut stepwise = None;
        for (i, id) in ids.iter().enumerate() {
            let token: Tensor<B, 2, Int> = Tensor::from_data(TensorData::from([[*id]]), &device);
            let x = model.embed_audio(token, i);
            stepwise = Some(model.forward(x, &mut state));
        }
        assert_eq!(state.offset(), ids.len());

        let a: Vec<f32> = one_shot.into_data().to_vec().unwrap();
        let b: Vec<f32> = stepwise.unwrap().into_data().to_vec().unwrap();
        assert!(
            a.iter().all(|v| v.is_finite()),
            "one-shot logits not finite"
        );
        assert!(
            b.iter().all(|v| v.is_finite()),
            "stepwise logits not finite"
        );
        for (x, y) in a.iter().zip(&b) {
            assert!((x - y).abs() < 1e-4, "one-shot {x} vs stepwise {y}");
        }
    }

    #[test]
    fn prosody_colours_the_phonemes_without_lengthening_them() {
        // BERT features are added to the phoneme embeddings, so the prompt is as
        // long as the phoneme sequence. Concatenating instead would double it and
        // silently misalign every position.
        let cfg = tiny();
        let device = Default::default();
        let model = T2s::<B>::new(&cfg, &device);

        let phones: Tensor<B, 2, Int> =
            Tensor::from_data(TensorData::from([[1i32, 2, 3]]), &device);
        let bert = Tensor::<B, 3>::zeros([1, 3, cfg.bert_dim], &device);
        assert_eq!(model.embed_text(phones, bert).dims(), [1, 3, cfg.model_dim]);
    }
}
