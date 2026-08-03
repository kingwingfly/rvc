//! CAM++ — **the speaker embedding Seed-VC actually conditions on.**
//!
//! A reference clip in, one 192-dim timbre vector out. This is the network
//! `inference.py` builds as `CAMPPlus(feat_dim=80, embedding_size=192)`, and its
//! output is the `style2` the diffusion transformer is conditioned on — so
//! without it there is no zero-shot conversion at all.
//!
//! It is **not** [`crate::style_encoder`], which is an 18-tensor subtree of the
//! Seed-VC checkpoint that upstream's `build_model` never assembles. That module
//! documents why it is a fossil; this one is its live replacement.
//!
//! # The weights live in somebody else's release
//!
//! `campplus_cn_common.bin` (28 MB, 937 tensors) from Hugging Face
//! `funasr/campplus` — the repository `inference.py` names verbatim. It is a
//! 3D-Speaker model, and the architecture below is a port of the copy Seed-VC
//! vendors under `modules/campplus/`, which carries 3D-Speaker's Apache-2.0
//! header. Apache-2.0 into GPL-3.0 is the compatible direction, so this file
//! does not widen the workspace's licence beyond what Seed-VC already imposed.
//!
//! # What it consumes is not this crate's mel, and that is the trap
//!
//! **The input is a Kaldi filterbank at 16 kHz — 80 bins, 25 ms window, 10 ms
//! hop, `dither=0`, mean-normalised over time — not the 22.05 kHz 80-band mel
//! the transformer predicts and the vocoder consumes.** Both are "an 80-band
//! mel" by shape, so the wrong one loads, runs, and produces a plausible vector
//! from a distribution the network was never trained on. Upstream computes it as
//!
//! ```text
//! feat = kaldi.fbank(wave_16k, num_mel_bins=80, dither=0, sample_frequency=16000)
//! feat = feat - feat.mean(dim=0, keepdim=True)
//! ```
//!
//! and the subtraction is part of the contract, not a nicety: the front end has
//! no input normalisation of its own.
//!
//! [`CamPPlus::forward`] therefore takes `[batch, frames, bins]`, matching
//! upstream's signature rather than the `[batch, bins, frames]` the rest of this
//! crate passes around. Keeping the transpose where upstream has it — the first
//! line of `CAMPPlus.forward` — is what makes the two readable side by side.
//!
//! # Shape of the network
//!
//! - **FCM**, a 2-D convolutional ResNet front end that treats the filterbank as
//!   an image and downsamples *frequency* by 8 while leaving time alone, then
//!   folds the 32 channels and 10 remaining bins into one 320-channel sequence.
//! - **`xvector`**: a strided TDNN layer (which is the only thing that halves the
//!   frame rate, 100 Hz to 50 Hz), then three **CAM dense TDNN blocks** of 12, 24
//!   and 16 layers. Dense means every layer's output is *concatenated* onto its
//!   input, so a block's width grows by `growth_rate` per layer and a
//!   `TransitLayer` halves it again afterwards.
//! - **Context-aware masking** is the "CAM" and the one piece worth reading
//!   twice: each layer computes a local convolution and multiplies it by a gate
//!   derived from two pooled summaries — the whole utterance's mean, plus a
//!   100-frame segment average broadcast back over time. So every frame is
//!   scaled by what its neighbourhood and the utterance as a whole look like,
//!   which is how a speaker-level network suppresses content-level detail.
//! - **Statistics pooling** — mean and standard deviation over time, concatenated
//!   — is what makes the reference length irrelevant, exactly as the average pool
//!   does in [`crate::style_encoder`].
//! - A final `1024 → 192` pointwise layer and a non-affine batch norm.
//!
//! # Inference only
//!
//! [`Norm`] always normalises by the stored running statistics, where PyTorch's
//! `BatchNorm` switches on a `training` flag. Upstream calls `.eval()` before
//! ever running this model and there is nothing here to fine-tune — the speaker
//! encoder is frozen in every Seed-VC path — so the batch-statistics branch would
//! be dead code whose only effect could be to make an embedding depend on what
//! else was in the batch.
//!
//! # What is established and what is not
//!
//! The layout is not guessed: upstream's source is available and the 937 tensors
//! account for it exactly. Two things are worth naming as unverified:
//!
//! - **No numerical diff against upstream exists.** `examples/load` checks that
//!   the embedding is finite, repeatable, and moves when the *spectral tilt* of
//!   its input moves — which catches a network that ignores its input, the
//!   failure mode that looks healthiest, but not a subtly mis-scaled one.
//! - **`eps = 1e-5`** is PyTorch's `BatchNorm` default rather than anything the
//!   checkpoint records. Upstream never passes an `eps`, so this follows from
//!   reading the constructor, but no tensor pins it.

use std::error::Error;
use std::path::Path;

use burn::module::{Module, Param};
use burn::nn::conv::{Conv1d, Conv1dConfig, Conv2d, Conv2dConfig};
use burn::nn::{PaddingConfig1d, PaddingConfig2d};
use burn::tensor::Tensor;
use burn::tensor::activation::{relu, sigmoid};
use burn::tensor::backend::Backend;

/// PyTorch's `BatchNorm` default, which upstream never overrides.
const EPS: f64 = 1e-5;

/// Frames per segment in the context-aware mask's segment pooling.
///
/// Upstream's `seg_len`, and at the 50 Hz this runs at it is a two-second
/// window — long enough to average over a phrase rather than a phoneme.
const SEG_LEN: usize = 100;

/// The shape of the network. The defaults are `campplus_cn_common.bin`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CamPPlusConfig {
    /// Filterbank bins consumed, 80.
    pub feat_dim: usize,
    /// Width of the timbre vector produced, 192 — which is what makes it drop
    /// into `SeedVcConfig::style_dim`.
    pub embedding_size: usize,
    /// Channels the FCM front end works at.
    pub m_channels: usize,
    /// Width the first TDNN layer projects to, and the first dense block's input.
    pub init_channels: usize,
    /// Channels each dense layer *adds* to its block's running width.
    pub growth_rate: usize,
    /// Bottleneck multiplier: each dense layer squeezes to `bn_size * growth_rate`
    /// before its convolution.
    pub bn_size: usize,
    /// `(layers, kernel, dilation)` per dense block.
    pub blocks: [(usize, usize, usize); 3],
}

impl Default for CamPPlusConfig {
    fn default() -> Self {
        Self {
            feat_dim: 80,
            embedding_size: 192,
            m_channels: 32,
            init_channels: 128,
            growth_rate: 32,
            bn_size: 4,
            blocks: [(12, 3, 1), (24, 3, 2), (16, 3, 2)],
        }
    }
}

impl CamPPlusConfig {
    /// Channels the [`Fcm`] front end emits — three stride-2 stages take the
    /// bins down by 8, and what is left is flattened onto the channel axis.
    ///
    /// 320 on the released shape, which is the first TDNN layer's input width.
    pub fn fcm_channels(&self) -> usize {
        self.m_channels * (self.feat_dim / 8)
    }
}

/// `BatchNorm` in evaluation mode, with PyTorch's own parameter names.
///
/// Written here rather than reached for from `burn::nn` for two reasons that
/// both come down to the checkpoint. Burn spells the affine pair `gamma`/`beta`
/// and keeps the running statistics in a `RunningState`, so loading needs an
/// adapter to rename them; these fields are `weight`/`bias`/`running_mean`/
/// `running_var`, which is what the file says, and nothing has to be translated.
/// And the affine pair is **optional** — upstream's `batchnorm_` config string
/// builds `affine=False`, which the final layer uses and which no `burn::nn`
/// norm can express, so it would report two parameters missing for ever.
///
/// `num_batches_tracked` has no inference role and is deliberately not a field;
/// it is the one key per norm that `load_pytorch` reports unused.
#[derive(Module, Debug)]
pub struct Norm<B: Backend> {
    weight: Option<Param<Tensor<B, 1>>>,
    bias: Option<Param<Tensor<B, 1>>>,
    running_mean: Param<Tensor<B, 1>>,
    running_var: Param<Tensor<B, 1>>,
}

impl<B: Backend> Norm<B> {
    fn new(channels: usize, affine: bool, device: &B::Device) -> Self {
        let param = || Param::from_tensor(Tensor::zeros([channels], device));
        Self {
            weight: affine.then(|| Param::from_tensor(Tensor::ones([channels], device))),
            bias: affine.then(param),
            running_mean: param(),
            running_var: Param::from_tensor(Tensor::ones([channels], device)),
        }
    }

    /// Normalise over the channel axis, whatever the rank — the same weights
    /// serve `BatchNorm1d` on `[batch, channels, time]` and `BatchNorm2d` on
    /// `[batch, channels, bins, time]`, because in both the channel is axis 1.
    fn forward<const D: usize>(&self, x: Tensor<B, D>) -> Tensor<B, D> {
        let mut shape = [1usize; D];
        shape[1] = self.running_mean.dims()[0];
        let over_channels = |t: Tensor<B, 1>| t.reshape(shape);

        let centred = x - over_channels(self.running_mean.val());
        let scaled = centred / over_channels(self.running_var.val().add_scalar(EPS).sqrt());
        let scaled = match &self.weight {
            Some(w) => scaled * over_channels(w.val()),
            None => scaled,
        };
        match &self.bias {
            Some(b) => scaled + over_channels(b.val()),
            None => scaled,
        }
    }
}

/// A 3×3 residual block over `[batch, channels, bins, time]`.
///
/// The stride is `(2, 1)`: frequency is halved, time is untouched. That
/// asymmetry is the whole point of the front end — it is compressing *what the
/// spectrum looks like* while leaving the frame rate for the TDNN stack to
/// handle.
#[derive(Module, Debug)]
pub struct ResBlock<B: Backend> {
    conv1: Conv2d<B>,
    bn1: Norm<B>,
    conv2: Conv2d<B>,
    bn2: Norm<B>,
    /// Present only where the stride makes the input and output shapes differ.
    shortcut: Option<Shortcut<B>>,
}

/// The projection on a strided [`ResBlock`]'s skip path.
///
/// Its own struct because upstream builds it as an `nn.Sequential`, so the
/// checkpoint spells it `shortcut.0` and `shortcut.1` — see
/// [`CamPPlus::load_pytorch`] for how those become `conv` and `norm`.
#[derive(Module, Debug)]
pub struct Shortcut<B: Backend> {
    conv: Conv2d<B>,
    norm: Norm<B>,
}

impl<B: Backend> ResBlock<B> {
    fn new(in_channels: usize, channels: usize, stride: usize, device: &B::Device) -> Self {
        let conv3 = |input: usize, stride: usize| {
            Conv2dConfig::new([input, channels], [3, 3])
                .with_stride([stride, 1])
                .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
                .with_bias(false)
                .init(device)
        };
        Self {
            conv1: conv3(in_channels, stride),
            bn1: Norm::new(channels, true, device),
            conv2: conv3(channels, 1),
            bn2: Norm::new(channels, true, device),
            shortcut: (stride != 1 || in_channels != channels).then(|| Shortcut {
                conv: Conv2dConfig::new([in_channels, channels], [1, 1])
                    .with_stride([stride, 1])
                    .with_bias(false)
                    .init(device),
                norm: Norm::new(channels, true, device),
            }),
        }
    }

    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let out = relu(self.bn1.forward(self.conv1.forward(x.clone())));
        let out = self.bn2.forward(self.conv2.forward(out));
        let skip = match &self.shortcut {
            Some(s) => s.norm.forward(s.conv.forward(x)),
            None => x,
        };
        relu(out + skip)
    }
}

/// The 2-D front end: an image of the filterbank in, a 320-channel sequence out.
#[derive(Module, Debug)]
pub struct Fcm<B: Backend> {
    conv1: Conv2d<B>,
    bn1: Norm<B>,
    layer1: Vec<ResBlock<B>>,
    layer2: Vec<ResBlock<B>>,
    conv2: Conv2d<B>,
    bn2: Norm<B>,
}

impl<B: Backend> Fcm<B> {
    fn new(cfg: &CamPPlusConfig, device: &B::Device) -> Self {
        let m = cfg.m_channels;
        let stage = || {
            vec![
                ResBlock::new(m, m, 2, device),
                ResBlock::new(m, m, 1, device),
            ]
        };
        Self {
            conv1: Conv2dConfig::new([1, m], [3, 3])
                .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
                .with_bias(false)
                .init(device),
            bn1: Norm::new(m, true, device),
            layer1: stage(),
            layer2: stage(),
            conv2: Conv2dConfig::new([m, m], [3, 3])
                .with_stride([2, 1])
                .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
                .with_bias(false)
                .init(device),
            bn2: Norm::new(m, true, device),
        }
    }

    /// `[batch, bins, frames]` → `[batch, channels * bins / 8, frames]`.
    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch, _, frames] = x.dims();
        let x = x.unsqueeze_dim(1);
        let mut out = relu(self.bn1.forward(self.conv1.forward(x)));
        for block in self.layer1.iter().chain(&self.layer2) {
            out = block.forward(out);
        }
        let out = relu(self.bn2.forward(self.conv2.forward(out)));

        let [_, channels, bins, _] = out.dims();
        out.reshape([batch, channels * bins, frames])
    }
}

/// A convolution followed by batch norm and a ReLU — upstream's `TDNNLayer`
/// and, with a 1×1 kernel and the order reversed, its `TransitLayer`.
#[derive(Module, Debug)]
pub struct TdnnLayer<B: Backend> {
    linear: Conv1d<B>,
    nonlinear: Norm<B>,
}

impl<B: Backend> TdnnLayer<B> {
    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        relu(self.nonlinear.forward(self.linear.forward(x)))
    }
}

/// Batch norm and a ReLU followed by a 1×1 convolution that halves the width.
///
/// The mirror image of [`TdnnLayer`], and the field names are upstream's, which
/// is why this is a separate type rather than a flag on that one: a dense block
/// grows by concatenation, so something has to bring the width back down before
/// the next block, and it does it *after* activating rather than before.
#[derive(Module, Debug)]
pub struct TransitLayer<B: Backend> {
    nonlinear: Norm<B>,
    linear: Conv1d<B>,
}

impl<B: Backend> TransitLayer<B> {
    fn new(in_channels: usize, channels: usize, device: &B::Device) -> Self {
        Self {
            nonlinear: Norm::new(in_channels, true, device),
            linear: Conv1dConfig::new(in_channels, channels, 1)
                .with_bias(false)
                .init(device),
        }
    }

    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        self.linear.forward(relu(self.nonlinear.forward(x)))
    }
}

/// Context-aware masking: a local convolution gated by pooled context.
#[derive(Module, Debug)]
pub struct CamLayer<B: Backend> {
    linear_local: Conv1d<B>,
    linear1: Conv1d<B>,
    linear2: Conv1d<B>,
}

impl<B: Backend> CamLayer<B> {
    fn new(
        bn_channels: usize,
        channels: usize,
        kernel: usize,
        dilation: usize,
        device: &B::Device,
    ) -> Self {
        let padding = (kernel - 1) / 2 * dilation;
        Self {
            linear_local: Conv1dConfig::new(bn_channels, channels, kernel)
                .with_dilation(dilation)
                .with_padding(PaddingConfig1d::Explicit(padding, padding))
                .with_bias(false)
                .init(device),
            // `reduction=2`, and upstream leaves these two with their default bias.
            linear1: Conv1dConfig::new(bn_channels, bn_channels / 2, 1).init(device),
            linear2: Conv1dConfig::new(bn_channels / 2, channels, 1).init(device),
        }
    }

    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let local = self.linear_local.forward(x.clone());
        // Two scales of context: the whole utterance, and a 2 s neighbourhood.
        let context = x.clone().mean_dim(2) + Self::segment_pool(x);
        let gate = sigmoid(self.linear2.forward(relu(self.linear1.forward(context))));
        local * gate
    }

    /// Average each `SEG_LEN`-frame segment and broadcast it back over the
    /// segment's own frames.
    ///
    /// Upstream reaches this through `avg_pool1d(..., ceil_mode=True)` followed
    /// by an expand and a truncation. Averaging each slice directly is the same
    /// arithmetic and says what it means: with `ceil_mode` and no padding, the
    /// final short window is divided by the frames actually in it, which is
    /// exactly a mean over the slice.
    fn segment_pool(x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch, channels, frames] = x.dims();
        let segments = (0..frames)
            .step_by(SEG_LEN)
            .map(|start| {
                let end = (start + SEG_LEN).min(frames);
                x.clone()
                    .slice([0..batch, 0..channels, start..end])
                    .mean_dim(2)
                    .repeat_dim(2, end - start)
            })
            .collect();
        Tensor::cat(segments, 2)
    }
}

/// One layer of a dense block: bottleneck, then a context-aware convolution.
#[derive(Module, Debug)]
pub struct CamDenseTdnnLayer<B: Backend> {
    nonlinear1: Norm<B>,
    linear1: Conv1d<B>,
    nonlinear2: Norm<B>,
    cam_layer: CamLayer<B>,
}

impl<B: Backend> CamDenseTdnnLayer<B> {
    fn new(
        in_channels: usize,
        channels: usize,
        bn_channels: usize,
        kernel: usize,
        dilation: usize,
        device: &B::Device,
    ) -> Self {
        Self {
            nonlinear1: Norm::new(in_channels, true, device),
            linear1: Conv1dConfig::new(in_channels, bn_channels, 1)
                .with_bias(false)
                .init(device),
            nonlinear2: Norm::new(bn_channels, true, device),
            cam_layer: CamLayer::new(bn_channels, channels, kernel, dilation, device),
        }
    }

    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let x = self.linear1.forward(relu(self.nonlinear1.forward(x)));
        self.cam_layer.forward(relu(self.nonlinear2.forward(x)))
    }
}

/// Everything after the front end, up to and including the embedding.
///
/// Named for the `nn.Sequential` upstream calls `xvector`, because that is the
/// prefix the checkpoint uses — including `xvector.dense`, which upstream has
/// since moved out of the sequential and carries a compatibility remap for. The
/// released file predates the move, and mirroring the file is what keeps the
/// loader free of a remap that only exists to undo somebody else's refactor.
#[derive(Module, Debug)]
pub struct XVector<B: Backend> {
    tdnn: TdnnLayer<B>,
    block1: Vec<CamDenseTdnnLayer<B>>,
    transit1: TransitLayer<B>,
    block2: Vec<CamDenseTdnnLayer<B>>,
    transit2: TransitLayer<B>,
    block3: Vec<CamDenseTdnnLayer<B>>,
    transit3: TransitLayer<B>,
    out_nonlinear: Norm<B>,
    dense: DenseLayer<B>,
}

/// The final `1024 → 192` projection, with a non-affine batch norm after it.
#[derive(Module, Debug)]
pub struct DenseLayer<B: Backend> {
    linear: Conv1d<B>,
    nonlinear: Norm<B>,
}

impl<B: Backend> XVector<B> {
    fn new(cfg: &CamPPlusConfig, device: &B::Device) -> Self {
        let mut channels = cfg.init_channels;
        let mut block = |(layers, kernel, dilation): (usize, usize, usize)| {
            let dense: Vec<_> = (0..layers)
                .map(|i| {
                    CamDenseTdnnLayer::new(
                        channels + i * cfg.growth_rate,
                        cfg.growth_rate,
                        cfg.bn_size * cfg.growth_rate,
                        kernel,
                        dilation,
                        device,
                    )
                })
                .collect();
            channels += layers * cfg.growth_rate;
            let transit = TransitLayer::new(channels, channels / 2, device);
            channels /= 2;
            (dense, transit)
        };
        let (block1, transit1) = block(cfg.blocks[0]);
        let (block2, transit2) = block(cfg.blocks[1]);
        let (block3, transit3) = block(cfg.blocks[2]);

        Self {
            tdnn: TdnnLayer {
                linear: Conv1dConfig::new(cfg.fcm_channels(), cfg.init_channels, 5)
                    .with_stride(2)
                    .with_padding(PaddingConfig1d::Explicit(2, 2))
                    .with_bias(false)
                    .init(device),
                nonlinear: Norm::new(cfg.init_channels, true, device),
            },
            block1,
            transit1,
            block2,
            transit2,
            block3,
            transit3,
            out_nonlinear: Norm::new(channels, true, device),
            dense: DenseLayer {
                // Statistics pooling concatenates a mean and a deviation, so the
                // projection sees twice the channels the stack ends on.
                linear: Conv1dConfig::new(channels * 2, cfg.embedding_size, 1)
                    .with_bias(false)
                    .init(device),
                nonlinear: Norm::new(cfg.embedding_size, false, device),
            },
        }
    }

    /// `[batch, channels, frames]` → `[batch, embedding, 1]`, the trailing axis
    /// being the one-frame sequence the pointwise projection works on.
    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let mut h = self.tdnn.forward(x);
        for (block, transit) in [
            (&self.block1, &self.transit1),
            (&self.block2, &self.transit2),
            (&self.block3, &self.transit3),
        ] {
            for layer in block {
                // Dense connectivity: every layer's output is kept *beside* its
                // input rather than replacing it, which is where the width goes.
                h = Tensor::cat(vec![h.clone(), layer.forward(h)], 1);
            }
            h = transit.forward(h);
        }
        let h = relu(self.out_nonlinear.forward(h));

        // Statistics pooling. The deviation is Bessel-corrected, matching
        // `torch.std(unbiased=True)`; `burn`'s `var` is the corrected one and
        // `var_bias` is not, which is the wrong way round to guess.
        let stats = Tensor::cat(vec![h.clone().mean_dim(2), h.var(2).sqrt()], 1);
        self.dense
            .nonlinear
            .forward(self.dense.linear.forward(stats))
    }
}

/// The speaker encoder.
#[derive(Module, Debug)]
pub struct CamPPlus<B: Backend> {
    head: Fcm<B>,
    xvector: XVector<B>,
}

impl<B: Backend> CamPPlus<B> {
    pub fn new(cfg: &CamPPlusConfig, device: &B::Device) -> Self {
        Self {
            head: Fcm::new(cfg, device),
            xvector: XVector::new(cfg, device),
        }
    }

    /// `features`: a Kaldi filterbank, `[batch, frames, bins]` → `[batch, embedding]`.
    ///
    /// **Frames before bins**, which is upstream's order and the opposite of
    /// [`crate::style_encoder::StyleEncoder::forward`] — see the module docs, and
    /// note that both are 80 wide, so the two transpositions of the same clip
    /// differ only in which axis is which and neither will fail loudly.
    ///
    /// Upstream also accepts per-item lengths, for a padded batch; nothing here
    /// needs it, because a reference is one clip and statistics pooled over
    /// padding would be wrong rather than merely different.
    pub fn forward(&self, features: Tensor<B, 3>) -> Tensor<B, 2> {
        let x = self.head.forward(features.swap_dims(1, 2));
        self.xvector.forward(x).squeeze_dims(&[2])
    }

    /// Load `campplus_cn_common.bin`.
    ///
    /// Three remaps, all of them undoing an `nn.Sequential`'s positional or
    /// hard-coded child names — no layout differs:
    ///
    /// - `get_nonlinear` builds a `Sequential` whose norm is registered as
    ///   `batchnorm`, so every norm sits one level deeper than its field here.
    /// - a strided residual block's projection is a bare `Sequential`, hence
    ///   `shortcut.0`/`shortcut.1` for its convolution and norm.
    /// - a dense block is an `nn.ModuleList` whose children are *named*
    ///   `tdnnd1`…`tdnndN`, one-based, where a Burn `Vec` numbers from zero.
    ///   A regex cannot subtract one, so the pairs are generated.
    ///
    /// The one key per norm that goes unclaimed is `num_batches_tracked`, a
    /// training counter with no inference role — 122 of them, and the only
    /// expected entry in `unused`.
    pub fn load_pytorch(
        &mut self,
        path: impl AsRef<Path>,
    ) -> Result<burn_store::ApplyResult, Box<dyn Error>> {
        // Read off the tree that was built rather than off the default config,
        // so a model constructed with deeper blocks still generates enough pairs.
        let longest = self
            .xvector
            .block1
            .len()
            .max(self.xvector.block2.len())
            .max(self.xvector.block3.len());
        let dense: Vec<(String, String)> = (0..longest)
            .map(|i| (format!(r"\.tdnnd{}\.", i + 1), format!(".{i}.")))
            .collect();

        let mut remaps = vec![
            (r"\.batchnorm\.", "."),
            (r"\.shortcut\.0\.", ".shortcut.conv."),
            (r"\.shortcut\.1\.", ".shortcut.norm."),
        ];
        remaps.extend(dense.iter().map(|(from, to)| (from.as_str(), to.as_str())));

        burn_kit::store::load_pytorch_into::<B, _>(self, path.as_ref(), None, &remaps)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Distribution, Int};

    type B = burn_ndarray::NdArray;

    /// A reference clip is 1–30 s, so the embedding has to be the same shape
    /// whatever arrives — that is the premise of zero-shot conversion, and the
    /// statistics pool is the only thing enforcing it.
    #[test]
    fn any_length_of_reference_gives_one_embedding() {
        let cfg = CamPPlusConfig::default();
        let device = Default::default();
        let model = CamPPlus::<B>::new(&cfg, &device);
        // Past `SEG_LEN` as well as short of it: segment pooling has a partial
        // final window in one case and only a partial window in the other.
        for frames in [40, 260] {
            let feats = Tensor::zeros([1, frames, cfg.feat_dim], &device);
            assert_eq!(model.forward(feats).dims(), [1, cfg.embedding_size]);
        }
    }

    /// A timbre encoder that ignores its input fails in the way that looks
    /// healthiest: every clip converts and every output is the same voice.
    ///
    /// The two inputs differ in **spectral tilt**, not merely in their samples.
    /// Two draws of white noise are the same signal to a speaker encoder and
    /// would agree closely however the arithmetic were wired, so a comparison
    /// between them proves nothing.
    #[test]
    fn tilt_moves_the_embedding_and_the_same_clip_repeats() {
        let cfg = CamPPlusConfig::default();
        let device = Default::default();
        let model = CamPPlus::<B>::new(&cfg, &device);
        let normal = Distribution::Normal(0.0, 1.0);
        let a = Tensor::<B, 3>::random([1, 120, cfg.feat_dim], normal, &device);
        let tilt = Tensor::<B, 1, Int>::arange(0..cfg.feat_dim as i64, &device)
            .float()
            .reshape([1, 1, cfg.feat_dim]);
        let b = Tensor::<B, 3>::random([1, 120, cfg.feat_dim], normal, &device) * tilt;

        let embed = |x| -> Vec<f32> { model.forward(x).into_data().to_vec().unwrap() };
        let (va, va_again, vb) = (embed(a.clone()), embed(a), embed(b));

        // Asserted separately because `assert_approx_eq` compares NaN to NaN
        // without complaining, and a division by a zero variance would produce
        // exactly that.
        assert!(va.iter().all(|x| x.is_finite()));
        assert_eq!(va, va_again);
        assert!(
            va.iter().zip(&vb).any(|(x, y)| (x - y).abs() > 1e-6),
            "two references of different spectral tilt gave the same embedding"
        );
    }

    /// Frequency is downsampled by 8 and time is left alone — the asymmetry the
    /// front end exists for. A symmetric stride would still typecheck and would
    /// halve the frame rate three times over before the TDNN stack saw it.
    #[test]
    fn the_front_end_downsamples_frequency_only() {
        let cfg = CamPPlusConfig::default();
        let device = Default::default();
        let fcm = Fcm::<B>::new(&cfg, &device);
        let x = Tensor::random(
            [1, cfg.feat_dim, 37],
            Distribution::Normal(0.0, 1.0),
            &device,
        );
        assert_eq!(fcm.forward(x).dims(), [1, cfg.fcm_channels(), 37]);
    }

    /// Segment pooling has to give back exactly the frames it was handed, with
    /// each segment holding its own mean — an off-by-one in the truncation is
    /// invisible in a shape assertion once the lengths happen to line up.
    #[test]
    fn segment_pooling_averages_within_each_segment() {
        let device = Default::default();
        let frames = SEG_LEN + 3;
        let x = Tensor::<B, 1, Int>::arange(0..frames as i64, &device)
            .float()
            .reshape([1, 1, frames]);
        let pooled: Vec<f32> = CamLayer::<B>::segment_pool(x).into_data().to_vec().unwrap();

        assert_eq!(pooled.len(), frames);
        // 0..99 averages to 49.5; the three-frame tail to 101.
        assert!(pooled[..SEG_LEN].iter().all(|x| (x - 49.5).abs() < 1e-4));
        assert!(pooled[SEG_LEN..].iter().all(|x| (x - 101.0).abs() < 1e-4));
    }
}
