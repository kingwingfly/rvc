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
        // The scale goes on `q` while it is still a slice of `qkv`, and that
        // ordering is the fix rather than a preference. On LibTorch `swap_dims`
        // returns torch's view of somebody else's buffer stamped with a fresh
        // `Storage::Owned`, so `can_mut()` says yes and the next in-place
        // capable op writes *through* the view — while `qkv`, `cache.k` and
        // `cache.v` are all still live on that buffer. Dividing after `heads`
        // therefore scaled `qkv`'s first `d_model` columns in place; it was
        // harmless only because nothing reads that region again, which is a
        // property of this layout and not of this code. `slice` keeps
        // `Storage::View`, whose `can_mut()` is false, so the division here is
        // out of place and the transposed view that follows aliases a tensor
        // nobody else holds. The arithmetic is untouched: a scalar divide is
        // elementwise, so it commutes exactly with reshape and transpose. See
        // CLAUDE.md, **`swap_dims` on LibTorch returns a view burn-tch forgets
        // the provenance of**, and `tch_aliasing` below.
        let q = heads(q / (d_head as f64).sqrt(), seq);
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
    vocab_size: usize,
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
            vocab_size: cfg.vocab_size,
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

    /// The opening pass, over the phonemes and the prompt's tokens together.
    ///
    /// The mask is **not** one causal triangle across the join. Text attends
    /// within itself in both directions — it is given, not predicted — while
    /// audio attends over all the text and causally over itself. A plain causal
    /// mask would stop each phoneme from seeing the ones after it, which is a
    /// different and much weaker conditioning that nothing would report.
    pub fn forward_prompt(
        &self,
        text: Tensor<B, 3>,
        audio: Tensor<B, 3>,
        state: &mut T2sState<B>,
    ) -> Tensor<B, 2> {
        let n_text = text.dims()[1];
        let n_audio = audio.dims()[1];
        let x = Tensor::cat(vec![text, audio], 1);
        let mask = prompt_mask::<B>(n_text, n_audio, &x.device());
        self.run(x, Some(mask), state)
    }

    /// Logits for **every** audio position — the training forward.
    ///
    /// Inference wants only the last position, because it generates one token at
    /// a time. Training wants all of them at once: each predicts the next token,
    /// so one pass over a clip supplies as many examples as it has tokens. The
    /// text positions are dropped; nothing predicts a phoneme.
    ///
    /// Returns `[n_audio, vocab]`, for one clip at a time.
    pub fn forward_prompt_all(&self, text: Tensor<B, 3>, audio: Tensor<B, 3>) -> Tensor<B, 2> {
        let n_text = text.dims()[1];
        let n_audio = audio.dims()[1];
        let x = Tensor::cat(vec![text, audio], 1);
        let mask = prompt_mask::<B>(n_text, n_audio, &x.device());

        let mut h = x;
        let mut state = self.state();
        for (i, layer) in self.h.layers.iter().enumerate() {
            h = layer.forward(h, Some(mask.clone()), &mut state.layers[i]);
        }

        let width = h.dims()[2];
        let audio_only = h.slice([0..1, n_text..n_text + n_audio, 0..width]);
        self.ar_predict_layer
            .forward(audio_only)
            .reshape([n_audio, self.vocab_size])
    }

    /// Run the stack over `x` and return logits for its final position.
    ///
    /// `x` is already embedded, and every position it carries is new — the cache
    /// holds everything before. Used one token at a time after
    /// [`T2s::forward_prompt`].
    pub fn forward(&self, x: Tensor<B, 3>, state: &mut T2sState<B>) -> Tensor<B, 2> {
        let [_, seq, _] = x.dims();
        let n_kv = seq + state.offset;
        let mask = causal_mask::<B>(seq, n_kv, &x.device());
        self.run(x, Some(mask), state)
    }

    fn run(
        &self,
        x: Tensor<B, 3>,
        mask: Option<Tensor<B, 2, Bool>>,
        state: &mut T2sState<B>,
    ) -> Tensor<B, 2> {
        let [batch, seq, _] = x.dims();
        let mut h = x;
        for (i, layer) in self.h.layers.iter().enumerate() {
            h = layer.forward(h, mask.clone(), &mut state.layers[i]);
        }
        state.offset += seq;

        let width = h.dims()[2];
        let last = h.slice([0..batch, seq - 1..seq, 0..width]);
        self.ar_predict_layer.forward(last).squeeze_dim::<2>(1)
    }

    /// Load weights, dispatching on extension: a `.ckpt` from upstream, or a
    /// `.safetensors` this toolkit fine-tuned. Both are `s1`; only the container
    /// differs.
    pub fn load_weights(
        &mut self,
        path: impl AsRef<std::path::Path>,
    ) -> Result<burn_store::ApplyResult, Box<dyn std::error::Error>> {
        let path = path.as_ref();
        match path.extension().and_then(|e| e.to_str()) {
            Some("safetensors") => burn_kit::store::load_burn_safetensors_into::<B, _>(self, path),
            _ => self.load_pytorch(path),
        }
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

/// The prompt's block mask: text sees all text, audio sees all text and its own
/// past. `true` blocks.
fn prompt_mask<B: Backend>(
    n_text: usize,
    n_audio: usize,
    device: &B::Device,
) -> Tensor<B, 2, Bool> {
    let n = n_text + n_audio;
    let mut flags = vec![0i32; n * n];
    for row in 0..n {
        for col in 0..n {
            let blocked = if row < n_text {
                // A phoneme may not look ahead into the audio.
                col >= n_text
            } else {
                // An audio position sees every phoneme, and audio only up to now.
                col >= n_text && col > row
            };
            flags[row * n + col] = blocked as i32;
        }
    }
    Tensor::<B, 2, Int>::from_data(TensorData::new(flags, [n, n]), device).bool()
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

/// The hazard that only exists on LibTorch, so it takes LibTorch to see it.
///
/// `cargo test -p burn-gptsovits --features tch` — a separate invocation on
/// purpose, the same shape `burn-mdx` uses: the default-feature test run cannot
/// reach the backend that has the defect, so a test gated any other way would
/// silently never run.
#[cfg(all(test, feature = "tch"))]
mod tch_aliasing {
    use super::*;
    use burn::module::{Module, ModuleMapper};

    type Tch = burn::backend::LibTorch<f32>;
    type Nd = burn_ndarray::NdArray;

    /// Small enough for `ndarray` to run the same pass, and still wide enough
    /// that `n_head` divides `model_dim` into more than one head — a single
    /// head makes the transposed view degenerate and proves nothing.
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

    /// A fixed LCG, so the two backends get byte-identical weights without a
    /// record round-trip. `Distribution` is not required to draw the same
    /// numbers on two backends, and a test that assumed it would fail for a
    /// reason that is not the one it is for.
    struct Lcg(u64);

    impl Lcg {
        fn take(&mut self, n: usize) -> Vec<f32> {
            (0..n)
                .map(|_| {
                    self.0 = self
                        .0
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    ((self.0 >> 40) as f32 / 8388608.0) - 1.0
                })
                .collect()
        }
    }

    /// Replaces every parameter with that sequence, in module-tree order.
    ///
    /// Initialised weights would not do: `in_proj_weight` and `in_proj_bias`
    /// are zeros, and a zero projection makes every attention output zero —
    /// which passes any comparison between two backends while saying nothing.
    struct Fill(Lcg);

    impl<B: Backend> ModuleMapper<B> for Fill {
        fn map_float<const D: usize>(&mut self, param: Param<Tensor<B, D>>) -> Param<Tensor<B, D>> {
            let (device, shape) = (param.val().device(), param.val().shape());
            let data = TensorData::new(self.0.take(shape.num_elements()), shape.clone());
            param.map(move |_| Tensor::from_data(data.clone(), &device))
        }
    }

    /// Runs a prompt and then two cached steps, and returns the final logits.
    ///
    /// The cached path is the one that matters here: `FusedAttention` splits
    /// one `qkv` into three slices and hands two of them to the cache, so
    /// every step leaves three live handles on one buffer for the next
    /// in-place-capable op to find.
    fn logits<B: Backend>(cfg: &T2sConfig, device: &B::Device) -> Vec<f32> {
        let model = T2s::<B>::new(cfg, device).map(&mut Fill(Lcg(1)));
        let mut state = model.state();

        let phones: Tensor<B, 2, Int> =
            Tensor::from_data(TensorData::from([[5i32, 11, 2, 30]]), device);
        let bert = Tensor::<B, 3>::from_data(
            TensorData::new(Lcg(7).take(4 * cfg.bert_dim), [1, 4, cfg.bert_dim]),
            device,
        );
        let prompt: Tensor<B, 2, Int> = Tensor::from_data(TensorData::from([[3i32, 9]]), device);

        let text = model.embed_text(phones, bert);
        let audio = model.embed_audio(prompt, 0);
        let mut out = model.forward_prompt(text, audio, &mut state);
        for (i, id) in [4i32, 12].iter().enumerate() {
            let token: Tensor<B, 2, Int> = Tensor::from_data(TensorData::from([[*id]]), device);
            out = model.forward(model.embed_audio(token, 2 + i), &mut state);
        }
        out.into_data().to_vec().unwrap()
    }

    /// LibTorch and `ndarray` are two independent implementations of the same
    /// arithmetic, so on one set of weights and one input they have to agree.
    ///
    /// That is the reading that catches an aliased view whatever it corrupts:
    /// `ndarray` refcounts its buffers correctly, so it computes what the code
    /// says while LibTorch computes what the code plus the defect say. The
    /// scale on `q` used to be applied *after* the transpose, which wrote the
    /// scaled queries back into `qkv`'s first `d_model` columns — harmless only
    /// because nothing reads that region again, so this test passed before the
    /// fix as well and is here to keep it passing when the layout moves.
    #[test]
    fn libtorch_agrees_with_ndarray_over_a_growing_cache() {
        let cfg = tiny();
        let want = logits::<Nd>(&cfg, &Default::default());
        let got = logits::<Tch>(&cfg, &Default::default());

        assert!(
            got.iter().all(|v| v.is_finite()),
            "LibTorch logits must be finite"
        );
        let worst = want
            .iter()
            .zip(&got)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let scale = want.iter().fold(0.0f32, |a, b| a.max(b.abs()));
        assert!(
            worst < 1e-4 * scale.max(1.0),
            "the two backends differ by {worst} on logits peaking at {scale}"
        );
    }

    /// The defect itself, in eight lines, because the fix above is otherwise
    /// indistinguishable from a stylistic preference.
    ///
    /// `slice` yields a `Storage::View`, so an in-place-capable op on it is
    /// forced out of place — that is what makes scaling *before* the transpose
    /// safe. `swap_dims` stamps a fresh `Storage::Owned` on the same borrowed
    /// buffer, so the identical op afterwards writes straight through into the
    /// source. If this test ever fails, burn-tch has fixed the defect and the
    /// ordering in `FusedAttention::forward` is free again — read it as news,
    /// not as a regression.
    #[test]
    fn a_transposed_slice_writes_through_into_its_source_and_a_plain_slice_does_not() {
        let device = Default::default();
        let base = Tensor::<Tch, 3>::ones([1, 2, 4], &device);

        let view = base.clone().slice([0..1, 0..2, 0..2]);
        let _ = view.reshape([1, 2, 2, 1]).swap_dims(1, 2) * 3.0;
        let after: Vec<f32> = base.clone().into_data().to_vec().unwrap();
        assert_eq!(
            after,
            vec![3.0, 3.0, 1.0, 1.0, 3.0, 3.0, 1.0, 1.0],
            "a transposed slice no longer aliases its source"
        );

        let base = Tensor::<Tch, 3>::ones([1, 2, 4], &device);
        let _ = base.clone().slice([0..1, 0..2, 0..2]) * 3.0;
        let after: Vec<f32> = base.into_data().to_vec().unwrap();
        assert_eq!(
            after,
            vec![1.0; 8],
            "a plain slice must never write through"
        );
    }
}
