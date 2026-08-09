//! The U-ViT diffusion transformer: content + timbre in, a mel out.
//!
//! This is the only generative network in Seed-VC and 255 of the checkpoint's
//! 302 tensors. Everything else — Whisper, the length regulator, the timbre
//! encoder, BigVGAN — either prepares its conditioning or renders its output.
//! What it computes is a **velocity field**: given a partly-noised mel and a flow
//! time `t`, it predicts the direction that mel should move in, and the sampler
//! integrates that over a handful of Euler steps.
//!
//! # Three shapes carry the architecture
//!
//! Reading any of them as its more common cousin gives a port that loads at 100%
//! and produces plausible-looking rubbish, which is the failure mode this repo
//! has already shipped once:
//!
//! - **`project_layer [1024, 512]` on every norm** is *adaptive* layer
//!   normalisation. The conditioning vector — here the flow-time embedding, and
//!   nothing else — is projected to `2 × 512` and split into a scale and a shift
//!   applied around the norm. A plain norm would drop the model's only sense of
//!   where along the trajectory it is. What sits inside is **RMSNorm**, not
//!   LayerNorm: one `weight`, no bias, no mean subtraction.
//! - **`w1`, `w2` and `w3`, with `w1` and `w3` both `[1536, 512]`**, is a gated
//!   feed-forward, `w2(silu(w1 x) ⊙ w3 x)` — SwiGLU. A two-matrix MLP would
//!   consume the same tensors in the same order and simply compute something
//!   else.
//! - **`skip_in_linear [512, 1024]` per layer plus `skip_linear [512, 592]` at
//!   the top** are two different skip schemes. The per-layer one is the **U-ViT**
//!   connection: layers 0–5 emit their output and layers 7–12 concatenate one
//!   back on in reverse order, so 7 receives 5's and 12 receives 0's — layer 6 is
//!   the bottom of the U and does neither. The top-level one is the **long** skip,
//!   concatenating the *input mel* (80 channels, hence 592 = 512 + 80) onto the
//!   transformer's output.
//!
//! # There is no causal mask, and adding one is the classic mistake here
//!
//! `is_causal: false`. The transformer sees the whole utterance at once and its
//! attention is unmasked — this is a denoiser, not a language model. Burn's
//! `triu_mask`/`tril_mask` are named for the triangle they *keep*, and a wrong
//! mask feeds a full row of `-inf` into softmax, which is `NaN` rather than an
//! error and reaches every output. The tests below assert finiteness separately
//! for that reason.
//!
//! Upstream does build one mask, from `x_lens`, but it is a **padding** mask over
//! keys and is all ones for a single clip. Inference only ever passes one clip —
//! the classifier-free-guidance pair is the same clip twice — so it is not
//! modelled. **A padded batched trainer would have to add it back.**
//!
//! # Attention is written here rather than reused
//!
//! [`burn_vits::MultiHeadAttention`] is VITS's, and disagrees on three counts: it
//! always carries `emb_rel_k`/`emb_rel_v` relative-position embeddings this
//! checkpoint has no tensors for, it keeps `conv_q`/`conv_k`/`conv_v` as separate
//! projections where this is one fused `wqkv [1536, 512]`, and it has no rotary
//! embedding. Bending it would change a module RVC and GPT-SoVITS both depend on
//! to suit a third caller that agrees with neither.
//!
//! # What the checkpoint carries and the model never runs
//!
//! Four tensors are allocated by upstream's `__init__` and never reached by its
//! `forward`. They are held as parameters anyway — dropping them would report
//! four false `unused` and hide a real one in the noise:
//!
//! - **`x_embedder`** (weight-normed, 80 → 512). `DiT.forward` never calls it;
//!   the mel enters through `cond_x_merge_linear` instead.
//! - **`cond_embedder`** (1024 × 512), the discrete-content codebook. Upstream
//!   pins `cond_in_module = self.cond_projection` with the discrete branch
//!   commented out, so `content_type: 'discrete'` in the config is a claim the
//!   code overrides — the continuous projection is what runs.
//! - **`content_mask_embedder`** (1 × 512). Constructed, never referenced;
//!   classifier-free guidance zeroes the conditioning rather than substituting a
//!   learned token.
//! - **`f0_embedder`** (512 × 512). `f0_condition: false`, and the released source
//!   has no `f0_embedder` at all — it is a fossil of a variant that did.
//!
//! `input_pos` is the fifth and differs in kind: a registered buffer holding
//! `arange(8192)`, i.e. the positions themselves. Rotary tables are recomputed
//! from the sequence length so its values are never read, but it is held so the
//! coverage count stays honest and so the trained `block_size` stays visible.
//!
//! # Provenance
//!
//! Ported from `modules/diffusion_transformer.py` of Seed-VC
//! (<https://github.com/Plachtaa/seed-vc>, GPL-3.0), read as a reference and
//! never run or vendored. That file in turn carries Meta's `gpt-fast`
//! transformer, which is where the fused `wqkv`, the rotary embedding and the
//! `w1`/`w2`/`w3` naming come from.

use std::error::Error;
use std::path::Path;

use burn::module::{Module, Param, ParamId};
use burn::nn::conv::{Conv1d, Conv1dConfig};
use burn::nn::{Embedding, EmbeddingConfig, Linear, LinearConfig};
use burn::tensor::activation::{silu, softmax};
use burn::tensor::backend::Backend;
use burn::tensor::{Distribution, Int, Tensor, TensorData};
use burn_store::ApplyResult;

use crate::config::SeedVcConfig;
use crate::wavenet::WaveNet;

/// Width of the sinusoidal timestep code before the MLP sees it.
///
/// Upstream's `TimestepEmbedder(hidden_size, frequency_embedding_size=256)`
/// default, pinned independently by the checkpoint's `mlp.0.weight` being
/// `[512, 256]`.
const TIME_FREQ_DIM: usize = 256;

/// Entries in the discrete-content codebook — upstream's `DiT.content_codebook_size`.
///
/// **Not [`SeedVcConfig::codebook_size`]**, which is the *length regulator's*
/// 2048. Two codebooks, two sizes, and neither is indexed on this preset.
const CONTENT_CODEBOOK: usize = 1024;

/// Quantisation bins the absent F0 conditioning would have used
/// (`n_f0_bins: 512`).
const F0_BINS: usize = 512;

/// `RMSNorm`'s epsilon — upstream's `ModelArgs.norm_eps`.
const NORM_EPS: f64 = 1e-5;

/// The output head's affine-free LayerNorm epsilon. Deliberately not
/// [`NORM_EPS`]: upstream writes `nn.LayerNorm(…, eps=1e-6)` there and RMSNorm's
/// 1e-5 everywhere else.
const FINAL_NORM_EPS: f64 = 1e-6;

/// Base of the rotary embedding's geometric frequency ladder, and of the
/// timestep code's — upstream reuses 10000 for both.
const FREQ_BASE: f64 = 10_000.0;

/// A linear layer under `torch.nn.utils.weight_norm`, which stores a direction
/// and a magnitude in place of a weight.
///
/// [`burn_vits::WeightNormConv1d`] is the convolutional counterpart; there is no
/// linear one there because RVC and GPT-SoVITS weight-normalise only
/// convolutions. **`weight_v` keeps PyTorch's `[out, in]` layout**: the
/// transposing adapter fires on Burn's own `Linear` and nothing else, so a
/// hand-rolled holder like this is handed the checkpoint's orientation unchanged.
#[derive(Module, Debug)]
struct WeightNormLinear<B: Backend> {
    weight_g: Param<Tensor<B, 2>>,
    weight_v: Param<Tensor<B, 2>>,
    bias: Param<Tensor<B, 1>>,
}

impl<B: Backend> WeightNormLinear<B> {
    fn new(d_in: usize, d_out: usize, device: &B::Device) -> Self {
        // **`weight_v` must not start at zero.** The forward pass divides by its
        // per-row norm, so a zero direction is `0 · (g / 0)` — `NaN`, in every
        // output, before a single weight is loaded. Weight norm's own
        // initialisation sets `g` from a random `v` for exactly this reason, and
        // it is what `burn_vits::WeightNormConv1d` does; copying the zero-init of
        // an ordinary bias here is the mistake that is invisible until an
        // untrained module is run.
        let v = Tensor::random([d_out, d_in], Distribution::Normal(0.0, 0.02), device);
        let g = v.clone().powf_scalar(2.0).sum_dim(1).sqrt();
        Self {
            weight_g: Param::from_tensor(g),
            weight_v: Param::from_tensor(v),
            bias: Param::from_tensor(Tensor::zeros([d_out], device)),
        }
    }

    /// `[batch, frames, d_in]` → `[batch, frames, d_out]`.
    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let v = self.weight_v.val();
        // weight_norm's default `dim=0`: one norm per output unit, over the input
        // features.
        let norm = v.clone().powf_scalar(2.0).sum_dim(1).sqrt();
        let weight = (v * (self.weight_g.val() / norm)).transpose();
        let [d_in, d_out] = weight.dims();
        x.matmul(weight.reshape([1, d_in, d_out])) + self.bias.val().reshape([1, 1, d_out])
    }
}

/// RMS normalisation with a learned scale and no bias.
///
/// Written here rather than taken from `burn::nn::RmsNorm` for one reason: Burn
/// names the scale `gamma`, and the PyTorch adapter renames `weight` → `gamma`
/// for BatchNorm, LayerNorm and GroupNorm only. Reusing Burn's type would mean a
/// regex remap reaching 27 call sites to buy ten lines.
#[derive(Module, Debug)]
struct RmsNorm<B: Backend> {
    weight: Param<Tensor<B, 1>>,
}

impl<B: Backend> RmsNorm<B> {
    fn new(dim: usize, device: &B::Device) -> Self {
        Self {
            weight: Param::from_tensor(Tensor::ones([dim], device)),
        }
    }

    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let dim = x.dims()[2];
        let scale = x
            .clone()
            .powf_scalar(2.0)
            .mean_dim(2)
            .add_scalar(NORM_EPS)
            .sqrt()
            .recip();
        x * scale * self.weight.val().reshape([1, 1, dim])
    }
}

/// Adaptive layer normalisation: `scale ⊙ rms(x) + shift`, both read off the
/// conditioning vector.
///
/// The conditioning is the flow-time embedding broadcast over every frame, so
/// **this is how the transformer knows where along the trajectory it is** —
/// nothing else inside a block depends on `t`.
#[derive(Module, Debug)]
struct AdaLayerNorm<B: Backend> {
    project_layer: Linear<B>,
    norm: RmsNorm<B>,
}

impl<B: Backend> AdaLayerNorm<B> {
    fn new(dim: usize, device: &B::Device) -> Self {
        Self {
            project_layer: LinearConfig::new(dim, 2 * dim).init(device),
            norm: RmsNorm::new(dim, device),
        }
    }

    /// `x`: `[batch, frames, dim]`, `c`: `[batch, 1, dim]` — broadcast over frames.
    fn forward(&self, x: Tensor<B, 3>, c: Tensor<B, 3>) -> Tensor<B, 3> {
        let dim = x.dims()[2];
        let proj = self.project_layer.forward(c);
        // Scale first, shift second. The output head splits its own modulation
        // the other way round, because upstream writes the two chunks in opposite
        // orders in the two places.
        let scale = proj.clone().narrow(2, 0, dim);
        let shift = proj.narrow(2, dim, dim);
        self.norm.forward(x) * scale + shift
    }
}

/// Rotary position tables for one sequence length, as `(cos, sin)` of shape
/// `[1, frames, 1, head_dim / 2]`.
///
/// Recomputed per forward rather than cached: the transformer runs a handful of
/// times per clip, and a cache keyed on length is state that can go stale against
/// the device.
fn rope_tables<B: Backend>(
    frames: usize,
    head_dim: usize,
    device: &B::Device,
) -> (Tensor<B, 4>, Tensor<B, 4>) {
    let half = head_dim / 2;
    let mut cos = Vec::with_capacity(frames * half);
    let mut sin = Vec::with_capacity(frames * half);
    for pos in 0..frames {
        for i in 0..half {
            let theta = pos as f64 * FREQ_BASE.powf(-2.0 * i as f64 / head_dim as f64);
            cos.push(theta.cos() as f32);
            sin.push(theta.sin() as f32);
        }
    }
    let shape = [1, frames, 1, half];
    (
        Tensor::from_data(TensorData::new(cos, shape), device),
        Tensor::from_data(TensorData::new(sin, shape), device),
    )
}

/// Rotate `x` (`[batch, frames, heads, head_dim]`) along its feature pairs.
///
/// **The pairs are adjacent, not half-and-half.** `gpt-fast` reshapes the last
/// dimension to `(head_dim/2, 2)`, so it rotates `(x[0], x[1])`, `(x[2], x[3])`
/// and so on, where the split-halves convention of Llama-style code rotates
/// `(x[i], x[i + head_dim/2])`. Both are called "RoPE", they are not
/// interchangeable, and choosing wrongly costs nothing at load time and
/// everything at inference.
fn apply_rope<B: Backend>(x: Tensor<B, 4>, cos: Tensor<B, 4>, sin: Tensor<B, 4>) -> Tensor<B, 4> {
    let [b, t, h, d] = x.dims();
    let half = d / 2;
    let pairs = x.reshape([b, t, h, half, 2]);
    let x0 = pairs.clone().narrow(4, 0, 1);
    let x1 = pairs.narrow(4, 1, 1);
    let cos = cos.reshape([1, t, 1, half, 1]);
    let sin = sin.reshape([1, t, 1, half, 1]);
    let out0 = x0.clone() * cos.clone() - x1.clone() * sin.clone();
    let out1 = x1 * cos + x0 * sin;
    Tensor::cat(vec![out0, out1], 4).reshape([b, t, h, d])
}

/// Unmasked multi-head self-attention with a fused qkv projection and rotary
/// positions.
#[derive(Module, Debug)]
struct Attention<B: Backend> {
    wqkv: Linear<B>,
    wo: Linear<B>,
    heads: usize,
    head_dim: usize,
}

impl<B: Backend> Attention<B> {
    fn new(dim: usize, heads: usize, device: &B::Device) -> Self {
        // `total_head_dim = (n_head + 2 · n_local_heads) · head_dim`, and this
        // preset has no grouped-query attention (`n_local_heads == n_head`), so
        // it comes to a plain 3 × dim.
        Self {
            wqkv: LinearConfig::new(dim, 3 * dim)
                .with_bias(false)
                .init(device),
            wo: LinearConfig::new(dim, dim).with_bias(false).init(device),
            heads,
            head_dim: dim / heads,
        }
    }

    fn forward(&self, x: Tensor<B, 3>, cos: Tensor<B, 4>, sin: Tensor<B, 4>) -> Tensor<B, 3> {
        let [b, t, dim] = x.dims();
        let qkv = self.wqkv.forward(x);
        let split = |i: usize| {
            qkv.clone()
                .narrow(2, i * dim, dim)
                .reshape([b, t, self.heads, self.head_dim])
        };

        let q = apply_rope(split(0), cos.clone(), sin.clone()).swap_dims(1, 2);
        let k = apply_rope(split(1), cos, sin).swap_dims(1, 2);
        let v = split(2).swap_dims(1, 2);

        let scores = q
            .matmul(k.swap_dims(2, 3))
            .div_scalar((self.head_dim as f32).sqrt());
        let y = softmax(scores, 3)
            .matmul(v)
            .swap_dims(1, 2)
            .reshape([b, t, dim]);
        self.wo.forward(y)
    }
}

/// The gated feed-forward, `w2(silu(w1 x) ⊙ w3 x)`.
#[derive(Module, Debug)]
struct FeedForward<B: Backend> {
    w1: Linear<B>,
    w2: Linear<B>,
    w3: Linear<B>,
}

impl<B: Backend> FeedForward<B> {
    fn new(dim: usize, hidden: usize, device: &B::Device) -> Self {
        let no_bias = |d_in, d_out| LinearConfig::new(d_in, d_out).with_bias(false).init(device);
        Self {
            w1: no_bias(dim, hidden),
            w2: no_bias(hidden, dim),
            w3: no_bias(dim, hidden),
        }
    }

    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        self.w2
            .forward(silu(self.w1.forward(x.clone())) * self.w3.forward(x))
    }
}

/// One pre-norm transformer block, plus the U-ViT skip inlet.
#[derive(Module, Debug)]
struct TransformerBlock<B: Backend> {
    attention: Attention<B>,
    feed_forward: FeedForward<B>,
    ffn_norm: AdaLayerNorm<B>,
    attention_norm: AdaLayerNorm<B>,
    /// Present on **every** block, including the seven that never receive a skip:
    /// upstream builds it from a config flag rather than from the block's index.
    /// The dead ones are in the checkpoint all the same.
    skip_in_linear: Linear<B>,
}

impl<B: Backend> TransformerBlock<B> {
    fn new(dim: usize, heads: usize, ffn_hidden: usize, device: &B::Device) -> Self {
        Self {
            attention: Attention::new(dim, heads, device),
            feed_forward: FeedForward::new(dim, ffn_hidden, device),
            ffn_norm: AdaLayerNorm::new(dim, device),
            attention_norm: AdaLayerNorm::new(dim, device),
            skip_in_linear: LinearConfig::new(2 * dim, dim).init(device),
        }
    }

    fn forward(
        &self,
        x: Tensor<B, 3>,
        c: Tensor<B, 3>,
        skip_in: Option<Tensor<B, 3>>,
        cos: Tensor<B, 4>,
        sin: Tensor<B, 4>,
    ) -> Tensor<B, 3> {
        let x = match skip_in {
            Some(skip) => self.skip_in_linear.forward(Tensor::cat(vec![x, skip], 2)),
            None => x,
        };
        let attended =
            self.attention
                .forward(self.attention_norm.forward(x.clone(), c.clone()), cos, sin);
        let h = x + attended;
        let fed = self
            .feed_forward
            .forward(self.ffn_norm.forward(h.clone(), c));
        h + fed
    }
}

/// The stack, with the U-ViT long skips wired across it.
#[derive(Module, Debug)]
struct Transformer<B: Backend> {
    layers: Vec<TransformerBlock<B>>,
    norm: AdaLayerNorm<B>,
}

impl<B: Backend> Transformer<B> {
    fn new(cfg: &SeedVcConfig, device: &B::Device) -> Self {
        // `ModelArgs.intermediate_size` when unset: `find_multiple(2/3 · 4 · dim,
        // 256)`, which is 1536 at dim 512 — exactly what `w1 [1536, 512]` records.
        let ffn_hidden = (8 * cfg.hidden_dim / 3).div_ceil(256) * 256;
        Self {
            layers: (0..cfg.depth)
                .map(|_| TransformerBlock::new(cfg.hidden_dim, cfg.heads, ffn_hidden, device))
                .collect(),
            norm: AdaLayerNorm::new(cfg.hidden_dim, device),
        }
    }

    fn forward(&self, x: Tensor<B, 3>, c: Tensor<B, 3>) -> Tensor<B, 3> {
        let frames = x.dims()[1];
        let head_dim = self.layers[0].attention.head_dim;
        let (cos, sin) = rope_tables::<B>(frames, head_dim, &x.device());

        // `i < depth/2` emits and `i > depth/2` receives, popping from the end —
        // so the last emitter feeds the first receiver and the pairing is
        // symmetric about the middle layer, which does neither. Upstream's depth
        // of 13 is odd, which is what makes the two lists the same length; an
        // even depth would leave one emitted tensor unclaimed, and upstream would
        // not error either.
        let half = self.layers.len() / 2;
        let mut skips: Vec<Tensor<B, 3>> = Vec::with_capacity(half);
        let mut x = x;
        for (i, layer) in self.layers.iter().enumerate() {
            let skip = if i > half { skips.pop() } else { None };
            x = layer.forward(x, c.clone(), skip, cos.clone(), sin.clone());
            if i < half {
                skips.push(x.clone());
            }
        }
        self.norm.forward(x, c)
    }
}

/// Sinusoidal flow-time code through a two-layer MLP.
#[derive(Module, Debug)]
struct TimestepEmbedder<B: Backend> {
    /// Upstream's `nn.Sequential(Linear, SiLU, Linear)`, so its checkpoint keys
    /// are `mlp.0` and `mlp.2` — the gap is the activation, which carries no
    /// tensors. [`Dit::load_pytorch`] closes it.
    mlp: Vec<Linear<B>>,
}

impl<B: Backend> TimestepEmbedder<B> {
    fn new(dim: usize, device: &B::Device) -> Self {
        Self {
            mlp: vec![
                LinearConfig::new(TIME_FREQ_DIM, dim).init(device),
                LinearConfig::new(dim, dim).init(device),
            ],
        }
    }

    /// `t`: `[batch]` in `[0, 1]` → `[batch, dim]`.
    fn forward(&self, t: Tensor<B, 1>) -> Tensor<B, 2> {
        let batch = t.dims()[0];
        let half = TIME_FREQ_DIM / 2;
        // `scale · exp(-ln(10000) · i / half)` with `scale = 1000`: the flow time
        // is in [0, 1] where a diffusion step index would be in [0, 1000), so
        // upstream rescales the argument rather than re-deriving the ladder.
        let freqs: Vec<f32> = (0..half)
            .map(|i| (1000.0 * FREQ_BASE.powf(-(i as f64) / half as f64)) as f32)
            .collect();
        let freqs = Tensor::<B, 1>::from_data(TensorData::new(freqs, [half]), &t.device());

        let args = t.reshape([batch, 1]) * freqs.reshape([1, half]);
        // `cat([cos, sin])`, in that order — reversing it is silent.
        let code = Tensor::cat(vec![args.clone().cos(), args.sin()], 1);
        self.mlp[1].forward(silu(self.mlp[0].forward(code)))
    }
}

/// The DiT output head: modulate by the flow time, then project.
#[derive(Module, Debug)]
struct FinalLayer<B: Backend> {
    linear: WeightNormLinear<B>,
    /// `nn.Sequential(SiLU, Linear)`, hence the checkpoint's
    /// `adaLN_modulation.1`. [`Dit::load_pytorch`] drops the index.
    ada_ln_modulation: Linear<B>,
}

impl<B: Backend> FinalLayer<B> {
    fn new(dim: usize, device: &B::Device) -> Self {
        Self {
            linear: WeightNormLinear::new(dim, dim, device),
            ada_ln_modulation: LinearConfig::new(dim, 2 * dim).init(device),
        }
    }

    /// `x`: `[batch, frames, dim]`, `c`: `[batch, dim]`.
    fn forward(&self, x: Tensor<B, 3>, c: Tensor<B, 2>) -> Tensor<B, 3> {
        let dim = x.dims()[2];
        let modulation = self.ada_ln_modulation.forward(silu(c));
        // **Shift is the first half here, scale the second** — the reverse of
        // [`AdaLayerNorm`]. Swapping them is a silent quality loss, not a crash.
        let shift = modulation.clone().narrow(1, 0, dim).unsqueeze_dim::<3>(1);
        let scale = modulation.narrow(1, dim, dim).unsqueeze_dim::<3>(1);

        // `nn.LayerNorm(elementwise_affine=False)`. PyTorch's variance here is
        // biased (divided by N), which is why it is spelled out rather than taken
        // from a helper that might not be.
        let mean = x.clone().mean_dim(2);
        let centred = x - mean;
        let var = centred.clone().powf_scalar(2.0).mean_dim(2);
        let normed = centred / var.add_scalar(FINAL_NORM_EPS).sqrt();

        self.linear.forward(normed * scale.add_scalar(1.0) + shift)
    }
}

/// Seed-VC's diffusion transformer.
#[derive(Module, Debug)]
pub struct Dit<B: Backend> {
    cond_projection: Linear<B>,
    cond_x_merge_linear: Linear<B>,
    t_embedder: TimestepEmbedder<B>,
    /// A **second** timestep embedder, feeding the WaveNet tail while
    /// [`Self::t_embedder`] feeds the transformer and the output head. Same
    /// architecture, separate weights: the two halves of the network are
    /// conditioned independently, and reusing one embedding for both would load
    /// fine and quietly halve the conditioning capacity.
    t_embedder2: TimestepEmbedder<B>,
    transformer: Transformer<B>,
    skip_linear: Linear<B>,
    conv1: Linear<B>,
    wavenet: WaveNet<B>,
    res_projection: Linear<B>,
    final_layer: FinalLayer<B>,
    conv2: Conv1d<B>,

    // Allocated by upstream, never reached by its forward pass — see the module
    // docs. Held so that the coverage count means something.
    x_embedder: WeightNormLinear<B>,
    cond_embedder: Embedding<B>,
    content_mask_embedder: Embedding<B>,
    f0_embedder: Embedding<B>,
    input_pos: Param<Tensor<B, 1, Int>>,

    block_size: usize,
}

impl<B: Backend> Dit<B> {
    pub fn new(cfg: &SeedVcConfig, device: &B::Device) -> Self {
        let dim = cfg.hidden_dim;
        // 80 (noisy mel) + 80 (prompt mel) + 512 (content) + 192 (timbre) = 864,
        // which is what `cond_x_merge_linear [512, 864]` records. **The whole of
        // the model's conditioning enters through this one projection**, which is
        // why classifier-free guidance can be done by zeroing inputs.
        let merged = 2 * cfg.n_mels + dim + cfg.style_dim;

        Self {
            cond_projection: LinearConfig::new(dim, dim).init(device),
            cond_x_merge_linear: LinearConfig::new(merged, dim).init(device),
            t_embedder: TimestepEmbedder::new(dim, device),
            t_embedder2: TimestepEmbedder::new(dim, device),
            transformer: Transformer::new(cfg, device),
            skip_linear: LinearConfig::new(dim + cfg.n_mels, dim).init(device),
            conv1: LinearConfig::new(dim, dim).init(device),
            wavenet: WaveNet::new(cfg, device),
            res_projection: LinearConfig::new(dim, dim).init(device),
            final_layer: FinalLayer::new(dim, device),
            conv2: Conv1dConfig::new(dim, cfg.n_mels, 1).init(device),

            x_embedder: WeightNormLinear::new(cfg.n_mels, dim, device),
            cond_embedder: EmbeddingConfig::new(CONTENT_CODEBOOK, dim).init(device),
            content_mask_embedder: EmbeddingConfig::new(1, dim).init(device),
            f0_embedder: EmbeddingConfig::new(F0_BINS, dim).init(device),
            // `Param::from_tensor` is float-only, and this buffer is `i64`.
            input_pos: Param::initialized(
                ParamId::new(),
                Tensor::arange(0..cfg.block_size as i64, device),
            ),

            block_size: cfg.block_size,
        }
    }

    /// Predict the flow velocity at time `t`.
    ///
    /// - `x`: `[batch, n_mels, frames]` — the current, partly-noised mel.
    /// - `prompt_x`: `[batch, n_mels, frames]` — the reference mel in the leading
    ///   frames and zero after. Shaped like `x`, **not** like the reference: the
    ///   sampler writes the reference into a zero tensor of `x`'s length.
    /// - `t`: `[batch]` — flow time in `[0, 1]`, one per batch element.
    /// - `style`: `[batch, style_dim]` — the timbre vector, broadcast over frames.
    /// - `cond`: `[batch, frames, hidden_dim]` — the length regulator's output,
    ///   already at the mel frame rate and channels-last.
    ///
    /// Returns `[batch, n_mels, frames]`.
    ///
    /// Classifier-free guidance is the caller's business and is done by **input**
    /// rather than by a flag: stack the batch twice, and zero `prompt_x`, `style`
    /// and `cond` in the second half. That is why there is no `mask_content`
    /// argument — upstream's `forward` has one and its inference path never sets
    /// it.
    ///
    /// **Batch entries are independent**, which is what makes that legal: nothing
    /// here reduces over dimension 0, so stacking the guided and unguided inputs
    /// into one call of twice the batch cannot blend them. The tests pin it by
    /// running a row alone and against its pair.
    ///
    /// The sampler's trait adds an `x_lens: Tensor<B, 1, Int>` between `prompt_x`
    /// and `t`; the forwarding impl drops it, because the only thing upstream
    /// builds from it is the padding mask this port does not model.
    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        prompt_x: Tensor<B, 3>,
        t: Tensor<B, 1>,
        style: Tensor<B, 2>,
        cond: Tensor<B, 3>,
    ) -> Tensor<B, 3> {
        let frames = x.dims()[2];
        assert!(
            frames <= self.block_size,
            "{frames} frames exceeds the trained block_size of {}",
            self.block_size
        );

        let t1 = self.t_embedder.forward(t.clone());
        let x = x.swap_dims(1, 2);
        let x_in = Tensor::cat(
            vec![
                x.clone(),
                prompt_x.swap_dims(1, 2),
                self.cond_projection.forward(cond),
                style.unsqueeze_dim::<3>(1).repeat_dim(1, frames),
            ],
            2,
        );
        let x_in = self.cond_x_merge_linear.forward(x_in);

        let x_res = self.transformer.forward(x_in, t1.clone().unsqueeze_dim(1));
        let x_res = self.skip_linear.forward(Tensor::cat(vec![x_res, x], 2));

        let t2 = self.t_embedder2.forward(t).unsqueeze_dim::<3>(2);
        let h = self
            .wavenet
            .forward(self.conv1.forward(x_res.clone()).swap_dims(1, 2), t2);
        // The long residual: the WaveNet refines the transformer's output rather
        // than replacing it.
        let h = h.swap_dims(1, 2) + self.res_projection.forward(x_res);

        self.conv2
            .forward(self.final_layer.forward(h, t1).swap_dims(1, 2))
    }

    /// Load this module's slice of the Seed-VC checkpoint.
    ///
    /// Four remaps, each closing a gap between a PyTorch container and a Burn
    /// field rather than renaming anything:
    ///
    /// - the `net.cfm.module.estimator.` prefix, which is where the flow-matching
    ///   wrapper keeps its one submodule;
    /// - `conv.conv` → nothing, collapsing `SConv1d` → `NormConv1d` → `Conv1d`.
    ///   Both wrappers are parameterless under weight norm (`NormConv1d.norm` is
    ///   an `Identity`), so mirroring them would mean two empty structs;
    /// - `mlp.2` → `mlp.1`, closing the index the `nn.Sequential`'s activation
    ///   occupies;
    /// - `adaLN_modulation.1` → `ada_ln_modulation`, likewise, and into a
    ///   snake-case field name.
    pub fn load_pytorch(&mut self, path: impl AsRef<Path>) -> Result<ApplyResult, Box<dyn Error>> {
        let mut remaps = vec![
            (r"^net\.cfm\.module\.estimator\.", ""),
            (r"\.conv\.conv\.", "."),
            (r"\.mlp\.2\.", ".mlp.1."),
            (
                r"^final_layer\.adaLN_modulation\.1\.",
                "final_layer.ada_ln_modulation.",
            ),
        ];
        // Last, so the prefix strip and `conv.conv` flattening above have
        // already run — these match a suffix on whatever path those produced.
        // This is also WaveNet's only loader: its weight-normalised convolutions
        // arrive as part of the estimator subtree.
        remaps.extend(crate::WEIGHT_NORM_REMAPS);
        burn_kit::store::load_pytorch_into::<B, _>(self, path.as_ref(), None, &remaps)
    }
}

/// What lets [`crate::flow::Sampler`] drive this transformer.
///
/// Pure forwarding, with `x_lens` dropped — the only thing upstream derives from
/// it is the key-padding mask this port does not model, and it is all ones for
/// the one batch shape inference ever builds (the guidance pair is the same clip
/// twice). **A padded batched trainer would have to reinstate it here**, not in
/// the sampler, which is why the argument stays on the trait.
impl<B: Backend> crate::flow::Estimator<B> for Dit<B> {
    fn velocity(
        &self,
        x: Tensor<B, 3>,
        prompt_x: Tensor<B, 3>,
        _x_lens: Tensor<B, 1, Int>,
        t: Tensor<B, 1>,
        style: Tensor<B, 2>,
        cond: Tensor<B, 3>,
    ) -> Tensor<B, 3> {
        self.forward(x, prompt_x, t, style, cond)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    type B = burn_ndarray::NdArray;

    /// A small stand-in for the released preset: same shapes, a depth and width a
    /// CPU test can afford. The depth stays **odd**, because that is what makes
    /// the U-ViT's emit and receive lists the same length.
    fn tiny() -> SeedVcConfig {
        SeedVcConfig {
            hidden_dim: 32,
            depth: 5,
            heads: 4,
            wavenet_layers: 2,
            ..SeedVcConfig::uvit_whisper_small_wavenet()
        }
    }

    #[test]
    fn a_mel_and_a_timbre_vector_predict_a_mel() {
        let cfg = tiny();
        let device = Default::default();
        let dit = Dit::<B>::new(&cfg, &device);
        let frames = 20;

        let out = dit.forward(
            Tensor::zeros([1, cfg.n_mels, frames], &device),
            Tensor::zeros([1, cfg.n_mels, frames], &device),
            Tensor::from_floats([0.3], &device),
            Tensor::zeros([1, cfg.style_dim], &device),
            Tensor::zeros([1, frames, cfg.hidden_dim], &device),
        );

        assert_eq!(out.dims(), [1, cfg.n_mels, frames]);
        // Asserted separately from any value check: Burn's approximate
        // comparisons treat NaN as equal to NaN, and a NaN out of a mis-built
        // mask reaches every output without erroring.
        assert!(!out.contains_nan().into_scalar());
    }

    #[test]
    fn the_classifier_free_guidance_pair_is_one_batched_call() {
        // The property the sampler depends on: it stacks the conditioned and
        // unconditional inputs into a single call of twice the batch, so a row's
        // output must be what that row alone would have produced. Anything that
        // reduced over the batch — a norm taken across dimension 0, say — would
        // silently blend the two branches and turn guidance into a smear.
        let cfg = tiny();
        let device = Default::default();
        let dit = Dit::<B>::new(&cfg, &device);
        let frames = 16;
        let normal = Distribution::Normal(0.0, 1.0);

        let x = Tensor::<B, 3>::random([2, cfg.n_mels, frames], normal, &device);
        // Row 0 conditioned, row 1 with everything the guidance path zeroes.
        let prompt = Tensor::cat(
            vec![
                Tensor::<B, 3>::random([1, cfg.n_mels, frames], normal, &device),
                Tensor::zeros([1, cfg.n_mels, frames], &device),
            ],
            0,
        );
        let style = Tensor::cat(
            vec![
                Tensor::<B, 2>::random([1, cfg.style_dim], normal, &device),
                Tensor::zeros([1, cfg.style_dim], &device),
            ],
            0,
        );
        let cond = Tensor::cat(
            vec![
                Tensor::<B, 3>::random([1, frames, cfg.hidden_dim], normal, &device),
                Tensor::zeros([1, frames, cfg.hidden_dim], &device),
            ],
            0,
        );
        let t = Tensor::from_floats([0.5, 0.5], &device);

        let paired = dit.forward(
            x.clone(),
            prompt.clone(),
            t.clone(),
            style.clone(),
            cond.clone(),
        );
        assert_eq!(paired.dims(), [2, cfg.n_mels, frames]);
        assert!(!paired.clone().contains_nan().into_scalar());

        let row = |i: usize| {
            dit.forward(
                x.clone().narrow(0, i, 1),
                prompt.clone().narrow(0, i, 1),
                t.clone().narrow(0, i, 1),
                style.clone().narrow(0, i, 1),
                cond.clone().narrow(0, i, 1),
            )
        };
        let rows = paired.chunk(2, 0);
        for (i, batched) in rows.iter().enumerate() {
            let drift = (batched.clone() - row(i)).abs().max().into_scalar();
            assert!(drift < 1e-5, "row {i} drifted by {drift} when batched");
        }
        let spread = (rows[0].clone() - rows[1].clone())
            .abs()
            .max()
            .into_scalar();
        assert!(spread > 0.0, "the guided and unguided rows must differ");
    }

    #[test]
    fn rotary_pairs_are_adjacent_not_half_and_half() {
        // Position 0 rotates by nothing; position 1 rotates the first pair by one
        // radian, since the lowest rung of the ladder is 10000^0 = 1. Pinning
        // that pins the convention, which is the part no shape reveals.
        let device = Default::default();
        let (cos, sin) = rope_tables::<B>(2, 4, &device);
        let x = Tensor::<B, 1>::from_floats([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0], &device)
            .reshape([1, 2, 1, 4]);
        let out: Vec<f32> = apply_rope(x, cos, sin).into_data().to_vec().unwrap();

        assert_eq!(&out[..4], &[1.0, 0.0, 1.0, 0.0]);
        assert!((out[4] - 1.0f32.cos()).abs() < 1e-6, "{}", out[4]);
        assert!((out[5] - 1.0f32.sin()).abs() < 1e-6, "{}", out[5]);
    }
}
