//! cnhubert — the SSL encoder GPT-SoVITS takes its semantic tokens from.
//!
//! A stock `transformers` `HubertModel` (`TencentGameMate/chinese-hubert-base`),
//! so the module tree mirrors that `state_dict` and the checkpoint loads
//! unchanged. Three stages: a stack of strided convolutions that turns 16 kHz
//! audio into 50 Hz frames, a projection to the model width, and a post-norm
//! transformer.
//!
//! Used on both sides of the pipeline. The quantiser downstream turns these
//! features into the discrete tokens the T2S model predicts, and the same
//! features describe the reference audio at synthesis time.

use burn::module::{Module, Param};
use burn::nn::conv::{Conv1d, Conv1dConfig};
use burn::nn::{
    GroupNorm, GroupNormConfig, LayerNorm, LayerNormConfig, Linear, LinearConfig, PaddingConfig1d,
};
use burn::tensor::Tensor;
use burn::tensor::activation::{gelu, softmax};
use burn::tensor::backend::Backend;

/// The shape of one HuBERT checkpoint, named as `config.json` names it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HubertConfig {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    /// Channels out of each convolution in the feature extractor.
    pub conv_dim: Vec<usize>,
    pub conv_kernel: Vec<usize>,
    pub conv_stride: Vec<usize>,
    /// Kernel of the positional convolution in the encoder.
    pub num_conv_pos_embeddings: usize,
    pub num_conv_pos_embedding_groups: usize,
}

impl HubertConfig {
    /// `chinese-hubert-base` — 95M parameters, 12 layers, 768 wide.
    pub fn chinese_base() -> Self {
        Self {
            hidden_size: 768,
            num_hidden_layers: 12,
            num_attention_heads: 12,
            intermediate_size: 3072,
            conv_dim: vec![512; 7],
            conv_kernel: vec![10, 3, 3, 3, 3, 2, 2],
            conv_stride: vec![5, 2, 2, 2, 2, 2, 2],
            num_conv_pos_embeddings: 128,
            num_conv_pos_embedding_groups: 16,
        }
    }

    /// Audio samples per output frame — the product of every stride.
    ///
    /// 320 here, i.e. 50 frames a second at 16 kHz, which is the rate the
    /// semantic tokens and everything downstream of them run at.
    pub fn samples_per_frame(&self) -> usize {
        self.conv_stride.iter().product()
    }
}

/// A weight-normalised convolution whose norm is taken over the **last** axis.
///
/// `burn-vits` has one that normalises over output channels (`weight_g` is
/// `[out, 1, 1]`), which is what the VITS decoders use. HuBERT's positional
/// convolution is normalised the other way — `weight_g` is `[1, 1, kernel]` — so
/// the two are not interchangeable, and reusing the wrong one loads a checkpoint
/// successfully with silently mis-scaled weights.
#[derive(Module, Debug)]
pub struct PosConv<B: Backend> {
    weight_g: Param<Tensor<B, 3>>,
    weight_v: Param<Tensor<B, 3>>,
    bias: Param<Tensor<B, 1>>,
    groups: usize,
    padding: usize,
}

impl<B: Backend> PosConv<B> {
    fn new(cfg: &HubertConfig, device: &B::Device) -> Self {
        let (channels, kernel) = (cfg.hidden_size, cfg.num_conv_pos_embeddings);
        let groups = cfg.num_conv_pos_embedding_groups;
        Self {
            weight_g: Param::from_tensor(Tensor::ones([1, 1, kernel], device)),
            weight_v: Param::from_tensor(Tensor::zeros(
                [channels, channels / groups, kernel],
                device,
            )),
            bias: Param::from_tensor(Tensor::zeros([channels], device)),
            groups,
            padding: kernel / 2,
        }
    }

    /// `x`: `[batch, channels, time]`.
    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        // w = g * v / ||v||, the norm taken over every axis but the last.
        let v = self.weight_v.val();
        let norm = v
            .clone()
            .powi_scalar(2)
            .sum_dim(0)
            .sum_dim(1)
            .sqrt()
            .clamp_min(1e-12);
        let weight = v * self.weight_g.val().div(norm);

        let options = burn::tensor::ops::ConvOptions::new([1], [self.padding], [1], self.groups);
        let y = burn::tensor::module::conv1d(x, weight, Some(self.bias.val()), options);

        // An even kernel with `kernel / 2` padding leaves one extra frame on the
        // right; the reference drops it rather than padding asymmetrically.
        let [batch, channels, time] = y.dims();
        gelu(y.slice([0..batch, 0..channels, 0..time - 1]))
    }
}

/// One convolution of the feature extractor.
///
/// Only the first carries a norm: `feat_extract_norm: "group"` means group
/// normalisation on layer 0 alone, with one group per channel.
#[derive(Module, Debug)]
pub struct ConvLayer<B: Backend> {
    conv: Conv1d<B>,
    layer_norm: Option<GroupNorm<B>>,
}

impl<B: Backend> ConvLayer<B> {
    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let x = self.conv.forward(x);
        let x = match &self.layer_norm {
            Some(norm) => norm.forward(x),
            None => x,
        };
        gelu(x)
    }
}

/// The convolutional front-end: raw 16 kHz audio to 50 Hz frames.
#[derive(Module, Debug)]
pub struct FeatureExtractor<B: Backend> {
    conv_layers: Vec<ConvLayer<B>>,
}

impl<B: Backend> FeatureExtractor<B> {
    fn new(cfg: &HubertConfig, device: &B::Device) -> Self {
        let conv_layers = (0..cfg.conv_dim.len())
            .map(|i| ConvLayer {
                conv: Conv1dConfig::new(
                    if i == 0 { 1 } else { cfg.conv_dim[i - 1] },
                    cfg.conv_dim[i],
                    cfg.conv_kernel[i],
                )
                .with_stride(cfg.conv_stride[i])
                .with_padding(PaddingConfig1d::Explicit(0, 0))
                .with_bias(false)
                .init(device),
                layer_norm: (i == 0)
                    .then(|| GroupNormConfig::new(cfg.conv_dim[0], cfg.conv_dim[0]).init(device)),
            })
            .collect();
        Self { conv_layers }
    }

    /// `wav`: `[batch, samples]` → `[batch, channels, frames]`.
    pub fn forward(&self, wav: Tensor<B, 2>) -> Tensor<B, 3> {
        let [batch, samples] = wav.dims();
        let mut x = wav.reshape([batch, 1, samples]);
        for layer in &self.conv_layers {
            x = layer.forward(x);
        }
        x
    }
}

/// LayerNorm then a linear projection to the transformer's width.
#[derive(Module, Debug)]
pub struct FeatureProjection<B: Backend> {
    layer_norm: LayerNorm<B>,
    projection: Linear<B>,
}

impl<B: Backend> FeatureProjection<B> {
    fn new(cfg: &HubertConfig, device: &B::Device) -> Self {
        let width = *cfg.conv_dim.last().unwrap_or(&512);
        Self {
            layer_norm: LayerNormConfig::new(width).init(device),
            projection: LinearConfig::new(width, cfg.hidden_size).init(device),
        }
    }

    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        self.projection.forward(self.layer_norm.forward(x))
    }
}

/// Multi-head self-attention, named as `transformers` names it: `attention` with
/// `{q,k,v,out}_proj`, all four with a bias.
#[derive(Module, Debug)]
pub struct Attention<B: Backend> {
    q_proj: Linear<B>,
    k_proj: Linear<B>,
    v_proj: Linear<B>,
    out_proj: Linear<B>,
    n_head: usize,
}

impl<B: Backend> Attention<B> {
    fn new(cfg: &HubertConfig, device: &B::Device) -> Self {
        let d = cfg.hidden_size;
        let linear = || LinearConfig::new(d, d).init(device);
        Self {
            q_proj: linear(),
            k_proj: linear(),
            v_proj: linear(),
            out_proj: linear(),
            n_head: cfg.num_attention_heads,
        }
    }

    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch, time, d_model] = x.dims();
        let d_head = d_model / self.n_head;
        let heads = |t: Tensor<B, 3>| {
            t.reshape([batch, time, self.n_head, d_head])
                .swap_dims(1, 2)
        };

        let q = heads(self.q_proj.forward(x.clone())) * (d_head as f64).powf(-0.5);
        let k = heads(self.k_proj.forward(x.clone()));
        let v = heads(self.v_proj.forward(x));

        let out = softmax(q.matmul(k.swap_dims(2, 3)), 3)
            .matmul(v)
            .swap_dims(1, 2)
            .reshape([batch, time, d_model]);
        self.out_proj.forward(out)
    }
}

/// The position-wise feed-forward block.
#[derive(Module, Debug)]
pub struct FeedForward<B: Backend> {
    intermediate_dense: Linear<B>,
    output_dense: Linear<B>,
}

impl<B: Backend> FeedForward<B> {
    fn new(cfg: &HubertConfig, device: &B::Device) -> Self {
        Self {
            intermediate_dense: LinearConfig::new(cfg.hidden_size, cfg.intermediate_size)
                .init(device),
            output_dense: LinearConfig::new(cfg.intermediate_size, cfg.hidden_size).init(device),
        }
    }

    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        self.output_dense
            .forward(gelu(self.intermediate_dense.forward(x)))
    }
}

/// One transformer block.
///
/// **Post-norm**, because this checkpoint sets `do_stable_layer_norm: false`:
/// normalisation comes *after* each residual addition, not before it.
/// `transformers` implements the pre-norm variant under a near-identical name,
/// and it is a plausible-looking wrong answer here — the weights load either way.
#[derive(Module, Debug)]
pub struct EncoderLayer<B: Backend> {
    attention: Attention<B>,
    layer_norm: LayerNorm<B>,
    feed_forward: FeedForward<B>,
    final_layer_norm: LayerNorm<B>,
}

impl<B: Backend> EncoderLayer<B> {
    fn new(cfg: &HubertConfig, device: &B::Device) -> Self {
        Self {
            attention: Attention::new(cfg, device),
            layer_norm: LayerNormConfig::new(cfg.hidden_size).init(device),
            feed_forward: FeedForward::new(cfg, device),
            final_layer_norm: LayerNormConfig::new(cfg.hidden_size).init(device),
        }
    }

    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let x = self
            .layer_norm
            .forward(x.clone() + self.attention.forward(x));
        self.final_layer_norm
            .forward(x.clone() + self.feed_forward.forward(x))
    }
}

/// The transformer, with its convolutional position embedding.
#[derive(Module, Debug)]
pub struct Encoder<B: Backend> {
    pos_conv_embed: PosConv<B>,
    layer_norm: LayerNorm<B>,
    layers: Vec<EncoderLayer<B>>,
}

impl<B: Backend> Encoder<B> {
    fn new(cfg: &HubertConfig, device: &B::Device) -> Self {
        Self {
            pos_conv_embed: PosConv::new(cfg, device),
            layer_norm: LayerNormConfig::new(cfg.hidden_size).init(device),
            layers: (0..cfg.num_hidden_layers)
                .map(|_| EncoderLayer::new(cfg, device))
                .collect(),
        }
    }

    /// `x`: `[batch, time, hidden]`. Returns the input embedding followed by every
    /// layer's output — `hidden_states` in `transformers` terms, because callers
    /// want a specific intermediate layer rather than the last one.
    fn forward(&self, x: Tensor<B, 3>) -> Vec<Tensor<B, 3>> {
        // Position information is added, not concatenated: a grouped convolution
        // over time, then GELU.
        let pos = self.pos_conv_embed.forward(x.clone().swap_dims(1, 2));
        let mut x = self.layer_norm.forward(x + pos.swap_dims(1, 2));

        let mut states = Vec::with_capacity(self.layers.len() + 1);
        states.push(x.clone());
        for layer in &self.layers {
            x = layer.forward(x);
            states.push(x.clone());
        }
        states
    }
}

/// cnhubert.
#[derive(Module, Debug)]
pub struct Hubert<B: Backend> {
    feature_extractor: FeatureExtractor<B>,
    feature_projection: FeatureProjection<B>,
    encoder: Encoder<B>,
}

impl<B: Backend> Hubert<B> {
    pub fn new(cfg: &HubertConfig, device: &B::Device) -> Self {
        Self {
            feature_extractor: FeatureExtractor::new(cfg, device),
            feature_projection: FeatureProjection::new(cfg, device),
            encoder: Encoder::new(cfg, device),
        }
    }

    /// Features for 16 kHz mono audio: `[batch, samples]` → `[batch, frames, hidden]`.
    ///
    /// This is `last_hidden_state`, which is what GPT-SoVITS's `CNHubert.forward`
    /// returns and what the quantiser consumes.
    pub fn forward(&self, wav: Tensor<B, 2>) -> Tensor<B, 3> {
        self.hidden_states(wav).pop().expect("at least one state")
    }

    /// Every layer's output, the projected input first.
    ///
    /// Exposed because the useful representation is not always the last one.
    pub fn hidden_states(&self, wav: Tensor<B, 2>) -> Vec<Tensor<B, 3>> {
        let frames = self.feature_extractor.forward(wav);
        let x = self.feature_projection.forward(frames.swap_dims(1, 2));
        self.encoder.forward(x)
    }

    /// Load a Hugging Face `pytorch_model.bin`.
    ///
    /// The state dict is at the root, with no wrapping key. One remap: upstream
    /// wraps the positional convolution's parameters in a `conv` submodule, and
    /// flattening it here keeps [`PosConv`] a single struct rather than adding a
    /// level that exists only to carry a name.
    ///
    /// Two entries are legitimately unused. `masked_spec_embed` is SpecAugment's
    /// mask token, which only exists while the SSL model itself is trained. And
    /// every LayerNorm's `weight`/`bias` is *reported* unused while having
    /// applied — `burn-store` consumes them under Burn's `gamma`/`beta` names and
    /// still counts the originals as unconsumed.
    pub fn load_pytorch(
        &mut self,
        path: impl AsRef<std::path::Path>,
    ) -> Result<burn_store::ApplyResult, Box<dyn std::error::Error>> {
        let remaps = [(
            r"^encoder\.pos_conv_embed\.conv\.",
            "encoder.pos_conv_embed.",
        )];
        burn_kit::store::load_pytorch_into::<B, _>(self, path.as_ref(), None, &remaps)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    type B = burn_ndarray::NdArray;

    #[test]
    fn the_convolutions_downsample_by_exactly_320() {
        // 16 kHz in, 50 Hz out. Every rate downstream is derived from this — the
        // semantic tokens, the T2S sequence length — so a wrong stride list
        // misaligns audio against text everywhere at once.
        let cfg = HubertConfig::chinese_base();
        assert_eq!(cfg.samples_per_frame(), 320);

        let device = Default::default();
        let model = Hubert::<B>::new(&cfg, &device);
        let [batch, frames, hidden] = model.forward(Tensor::zeros([1, 16_000], &device)).dims();
        assert_eq!(batch, 1);
        assert_eq!(hidden, cfg.hidden_size);
        // The convolutions are unpadded, so the count floor-divides to a little
        // under 50 rather than exactly it.
        assert!((45..=50).contains(&frames), "{frames} frames for 1 s");
    }

    #[test]
    fn every_layer_reports_a_hidden_state() {
        // The embedding has to be counted as well as each layer's output, because
        // callers index this from the end — GPT-SoVITS reads the third-from-last
        // of its BERT, and that arithmetic is only right if the count includes it.
        let cfg = HubertConfig::chinese_base();
        let device = Default::default();
        let model = Hubert::<B>::new(&cfg, &device);
        let states = model.hidden_states(Tensor::zeros([1, 16_000], &device));
        assert_eq!(states.len(), cfg.num_hidden_layers + 1);
    }
}
