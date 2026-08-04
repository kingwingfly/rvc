//! The deep U-net: five encoder levels, four intermediate blocks, five decoder
//! levels, all built from one residual convolution block.
//!
//! The tensor is `[batch, channels, time, bins]` throughout — **time is the
//! height and the 128 mel bins are the width**, because [`crate::Rmvpe::forward`]
//! transposes the mel before entering. Every pooling and every transposed
//! convolution therefore halves or doubles *both* of them, which is where the
//! multiple-of-32 requirement on the frame count comes from.
//!
//! # Two irregularities in the checkpoint
//!
//! - **`ConvBlockRes`'s 1×1 shortcut exists only when the block changes channel
//!   count.** Upstream creates the attribute conditionally and probes it with
//!   `hasattr`, so `rmvpe.pt` simply has no `shortcut.*` key for the other 45 of
//!   the 56 blocks. Modelling it as an `Option` is what makes that a load with
//!   nothing missing rather than 90 absent parameters.
//! - **Everything upstream builds as an `nn.Sequential` is positional**, so the
//!   file spells a block's two convolutions `conv.0` and `conv.3` and its two
//!   norms `conv.1` and `conv.4` — a Rust field cannot be called `0`, so those
//!   are the renames [`crate::Rmvpe::load_pytorch`] applies.

use burn::module::{Module, Param};
use burn::nn::PaddingConfig2d;
use burn::nn::conv::{Conv2d, Conv2dConfig, ConvTranspose2d, ConvTranspose2dConfig};
use burn::nn::pool::{AvgPool2d, AvgPool2dConfig};
use burn::tensor::Tensor;
use burn::tensor::activation::relu;
use burn::tensor::backend::Backend;

/// PyTorch's `BatchNorm` default. Upstream passes `momentum=0.01` and never an
/// `eps`, and momentum governs how the running statistics are *updated*, so it
/// has no effect on the inference this crate does — it would matter only if
/// someone later fine-tuned RMVPE here, and Burn's `momentum` is the
/// complement of PyTorch's under some conventions, so that is the moment to
/// check rather than to guess.
const EPS: f64 = 1e-5;

/// `BatchNorm2d` in evaluation mode, with PyTorch's own parameter names.
///
/// Written out rather than reached for from `burn::nn` because Burn spells the
/// affine pair `gamma`/`beta` and keeps the statistics in a `RunningState`,
/// which loading would have to translate. These four fields are what the file
/// says. `num_batches_tracked` is deliberately absent: it counts training
/// batches and has no inference role, so it is the one key per norm that a load
/// reports unused — 118 of them.
#[derive(Module, Debug)]
pub struct Norm<B: Backend> {
    weight: Param<Tensor<B, 1>>,
    bias: Param<Tensor<B, 1>>,
    running_mean: Param<Tensor<B, 1>>,
    running_var: Param<Tensor<B, 1>>,
}

impl<B: Backend> Norm<B> {
    fn new(channels: usize, device: &B::Device) -> Self {
        Self {
            weight: Param::from_tensor(Tensor::ones([channels], device)),
            bias: Param::from_tensor(Tensor::zeros([channels], device)),
            running_mean: Param::from_tensor(Tensor::zeros([channels], device)),
            running_var: Param::from_tensor(Tensor::ones([channels], device)),
        }
    }

    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let channels = self.running_mean.dims()[0];
        let over_channels = |t: Tensor<B, 1>| t.reshape([1, channels, 1, 1]);
        let centred = x - over_channels(self.running_mean.val());
        let scaled = centred / over_channels(self.running_var.val().add_scalar(EPS).sqrt());
        scaled * over_channels(self.weight.val()) + over_channels(self.bias.val())
    }
}

/// Two 3×3 convolutions with a residual connection.
///
/// The residual is the identity where the channel count is unchanged and a 1×1
/// convolution where it is not — see the module docs for why that is an
/// `Option` and not a always-present layer.
#[derive(Module, Debug)]
pub struct ConvBlockRes<B: Backend> {
    conv1: Conv2d<B>,
    bn1: Norm<B>,
    conv2: Conv2d<B>,
    bn2: Norm<B>,
    shortcut: Option<Conv2d<B>>,
}

impl<B: Backend> ConvBlockRes<B> {
    fn new(in_channels: usize, out_channels: usize, device: &B::Device) -> Self {
        let conv3 = |input: usize| {
            Conv2dConfig::new([input, out_channels], [3, 3])
                .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
                .with_bias(false)
                .init(device)
        };
        Self {
            conv1: conv3(in_channels),
            bn1: Norm::new(out_channels, device),
            conv2: conv3(out_channels),
            bn2: Norm::new(out_channels, device),
            // The 1×1 shortcut *does* carry a bias — upstream leaves
            // `nn.Conv2d`'s default alone here, unlike the 3×3 pair.
            shortcut: (in_channels != out_channels)
                .then(|| Conv2dConfig::new([in_channels, out_channels], [1, 1]).init(device)),
        }
    }

    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let out = relu(self.bn1.forward(self.conv1.forward(x.clone())));
        let out = relu(self.bn2.forward(self.conv2.forward(out)));
        match &self.shortcut {
            Some(s) => out + s.forward(x),
            None => out + x,
        }
    }
}

/// `n_blocks` residual blocks, optionally followed by 2×2 average pooling.
///
/// The two return values are not the same tensor at different scales: the
/// *pre-pool* one is what the decoder concatenates on the way back up, and the
/// pooled one is what the next level consumes. Upstream returns them as a pair
/// for exactly that reason, and the pooling is skipped entirely in the
/// intermediate stack, which is what [`Self::pool`] being an `Option` encodes.
#[derive(Module, Debug)]
pub struct ResEncoderBlock<B: Backend> {
    conv: Vec<ConvBlockRes<B>>,
    pool: Option<AvgPool2d>,
}

impl<B: Backend> ResEncoderBlock<B> {
    fn new(
        in_channels: usize,
        out_channels: usize,
        n_blocks: usize,
        pooled: bool,
        device: &B::Device,
    ) -> Self {
        let conv = (0..n_blocks)
            .map(|i| {
                let input = if i == 0 { in_channels } else { out_channels };
                ConvBlockRes::new(input, out_channels, device)
            })
            .collect();
        Self {
            conv,
            pool: pooled.then(|| AvgPool2dConfig::new([2, 2]).init()),
        }
    }

    /// Returns `(pre-pool, pooled)`; the second is the first when there is no
    /// pooling stage.
    fn forward(&self, x: Tensor<B, 4>) -> (Tensor<B, 4>, Tensor<B, 4>) {
        let mut x = x;
        for block in &self.conv {
            x = block.forward(x);
        }
        match &self.pool {
            Some(pool) => (x.clone(), pool.forward(x)),
            None => (x.clone(), x),
        }
    }
}

/// The contracting half: one input norm and five pooling levels.
#[derive(Module, Debug)]
pub struct Encoder<B: Backend> {
    bn: Norm<B>,
    layers: Vec<ResEncoderBlock<B>>,
}

impl<B: Backend> Encoder<B> {
    fn new(cfg: &crate::RmvpeConfig, device: &B::Device) -> Self {
        let mut in_channels = cfg.in_channels;
        let mut out_channels = cfg.en_out_channels;
        let layers = (0..cfg.en_de_layers)
            .map(|_| {
                let layer =
                    ResEncoderBlock::new(in_channels, out_channels, cfg.n_blocks, true, device);
                in_channels = out_channels;
                out_channels *= 2;
                layer
            })
            .collect();
        Self {
            bn: Norm::new(cfg.in_channels, device),
            layers,
        }
    }

    fn forward(&self, x: Tensor<B, 4>) -> (Tensor<B, 4>, Vec<Tensor<B, 4>>) {
        let mut x = self.bn.forward(x);
        let mut skips = Vec::with_capacity(self.layers.len());
        for layer in &self.layers {
            let (skip, pooled) = layer.forward(x);
            skips.push(skip);
            x = pooled;
        }
        (x, skips)
    }
}

/// The bottleneck: four residual stacks at full width and no resampling.
#[derive(Module, Debug)]
pub struct Intermediate<B: Backend> {
    layers: Vec<ResEncoderBlock<B>>,
}

impl<B: Backend> Intermediate<B> {
    fn new(
        in_channels: usize,
        out_channels: usize,
        cfg: &crate::RmvpeConfig,
        device: &B::Device,
    ) -> Self {
        let layers = (0..cfg.inter_layers)
            .map(|i| {
                let input = if i == 0 { in_channels } else { out_channels };
                ResEncoderBlock::new(input, out_channels, cfg.n_blocks, false, device)
            })
            .collect();
        Self { layers }
    }

    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let mut x = x;
        for layer in &self.layers {
            // No pooling here, so the two halves of the pair are the same tensor.
            x = layer.forward(x).1;
        }
        x
    }
}

/// One expanding level: transposed convolution, concatenate the skip, then the
/// same residual stack the encoder uses.
#[derive(Module, Debug)]
pub struct ResDecoderBlock<B: Backend> {
    conv1: ConvTranspose2d<B>,
    bn: Norm<B>,
    conv2: Vec<ConvBlockRes<B>>,
}

impl<B: Backend> ResDecoderBlock<B> {
    fn new(in_channels: usize, out_channels: usize, n_blocks: usize, device: &B::Device) -> Self {
        let conv2 = (0..n_blocks)
            .map(|i| {
                // The first block eats the concatenation, hence the doubling.
                let input = if i == 0 {
                    out_channels * 2
                } else {
                    out_channels
                };
                ConvBlockRes::new(input, out_channels, device)
            })
            .collect();
        Self {
            // `output_padding = 1` with a 3×3 kernel, stride 2 and padding 1
            // makes the output exactly twice the input on both axes, which is
            // what lets the skip concatenate without a resize.
            conv1: ConvTranspose2dConfig::new([in_channels, out_channels], [3, 3])
                .with_stride([2, 2])
                .with_padding([1, 1])
                .with_padding_out([1, 1])
                .with_bias(false)
                .init(device),
            bn: Norm::new(out_channels, device),
            conv2,
        }
    }

    fn forward(&self, x: Tensor<B, 4>, skip: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = relu(self.bn.forward(self.conv1.forward(x)));
        let mut x = Tensor::cat(vec![x, skip], 1);
        for block in &self.conv2 {
            x = block.forward(x);
        }
        x
    }
}

/// The expanding half, consuming the encoder's skips in reverse.
#[derive(Module, Debug)]
pub struct Decoder<B: Backend> {
    layers: Vec<ResDecoderBlock<B>>,
}

impl<B: Backend> Decoder<B> {
    fn new(in_channels: usize, cfg: &crate::RmvpeConfig, device: &B::Device) -> Self {
        let mut in_channels = in_channels;
        let layers = (0..cfg.en_de_layers)
            .map(|_| {
                let out_channels = in_channels / 2;
                let layer = ResDecoderBlock::new(in_channels, out_channels, cfg.n_blocks, device);
                in_channels = out_channels;
                layer
            })
            .collect();
        Self { layers }
    }

    fn forward(&self, x: Tensor<B, 4>, skips: Vec<Tensor<B, 4>>) -> Tensor<B, 4> {
        let mut x = x;
        for (layer, skip) in self.layers.iter().zip(skips.into_iter().rev()) {
            x = layer.forward(x, skip);
        }
        x
    }
}

/// Encoder, bottleneck and decoder as one module.
#[derive(Module, Debug)]
pub struct DeepUnet<B: Backend> {
    encoder: Encoder<B>,
    intermediate: Intermediate<B>,
    decoder: Decoder<B>,
}

impl<B: Backend> DeepUnet<B> {
    pub(crate) fn new(cfg: &crate::RmvpeConfig, device: &B::Device) -> Self {
        // What the encoder's loop leaves behind after its last doubling: 512 on
        // the released shape. The bottleneck widens *to* it from half of it,
        // which is the encoder's final output width.
        let widest = cfg.en_out_channels * (1 << cfg.en_de_layers);
        Self {
            encoder: Encoder::new(cfg, device),
            intermediate: Intermediate::new(widest / 2, widest, cfg, device),
            decoder: Decoder::new(widest, cfg, device),
        }
    }

    pub(crate) fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let (x, skips) = self.encoder.forward(x);
        let x = self.intermediate.forward(x);
        self.decoder.forward(x, skips)
    }
}
