//! The blocks MDX23C is built from: instance normalisation, the TFC-TDF
//! residual block, and the two resampling stages.
//!
//! The tensor is `[batch, channels, height, width]` throughout, but **which
//! axis is which changes once**: [`crate::TfcTdfNet::forward`] transposes the
//! last two axes before the encoder and back after the decoder, so inside every
//! block here the height is *time* and the width is *frequency*. That is what
//! lets [`Tdf`]'s `Linear` act on the frequency axis — Burn's `Linear`, like
//! PyTorch's, only ever touches the last one — and it is the single detail that
//! makes a transposed port load at 100% and separate nothing.

use burn::module::{Module, Param};
use burn::nn::conv::{Conv2d, Conv2dConfig, ConvTranspose2d, ConvTranspose2dConfig};
use burn::nn::{Linear, LinearConfig, PaddingConfig2d};
use burn::tensor::Tensor;
use burn::tensor::activation::gelu;
use burn::tensor::backend::Backend;

/// `torch.nn.InstanceNorm2d`'s default.
const EPS: f64 = 1e-5;

/// `nn.InstanceNorm2d(c, affine=True)`, written out rather than reached for.
///
/// **This is not a `BatchNorm` in evaluation mode, and the checkpoint is the
/// only thing that says so.** The config calls it `InstanceNorm`, and the
/// tensors agree: every norm in `MDX23C-8KFFT-InstVoc_HQ.ckpt` carries
/// `weight` and `bias` and *no* `running_mean`/`running_var`. Modelling it with
/// running statistics — the shape `burn_rmvpe::unet::Norm` has, which is the
/// obvious thing to copy — would load with two parameters missing per norm out
/// of 152 norms, and normalising by a stale running mean instead of the
/// sample's own is silent: the output stays finite and the separation just gets
/// worse.
///
/// Burn's own `nn::InstanceNorm` spells the affine pair `gamma`/`beta` and
/// wraps them in `Option`, so loading it would lean on the adapter's norm
/// renaming. These two field names *are* the checkpoint's, which is one less
/// thing to be right about. Same reasoning as `burn_rmvpe::unet::Norm`.
#[derive(Module, Debug)]
pub struct Norm<B: Backend> {
    weight: Param<Tensor<B, 1>>,
    bias: Param<Tensor<B, 1>>,
}

impl<B: Backend> Norm<B> {
    fn new(channels: usize, device: &B::Device) -> Self {
        Self {
            weight: Param::from_tensor(Tensor::ones([channels], device)),
            bias: Param::from_tensor(Tensor::zeros([channels], device)),
        }
    }

    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let [_, channels, _, _] = x.dims();
        let over_channels = |t: Tensor<B, 1>| t.reshape([1, channels, 1, 1]);
        // Per sample and per channel, over both spatial axes — that is the only
        // thing separating instance from batch normalisation once the running
        // statistics are gone.
        let mean = x.clone().mean_dim(3).mean_dim(2);
        let centred = x - mean;
        // Biased variance, as every normalisation layer uses.
        let var = centred.clone().powi_scalar(2).mean_dim(3).mean_dim(2);
        let normed = centred / var.add_scalar(EPS).sqrt();
        normed * over_channels(self.weight.val()) + over_channels(self.bias.val())
    }
}

/// Time-frequency convolution: `norm → GELU → 3×3 conv`.
///
/// Upstream builds this as `nn.Sequential(norm(c), act, nn.Conv2d(...))`, so
/// the checkpoint spells the two parameterised children `.0` and `.2` — index
/// `1` is the activation, which holds nothing. Those are the renames
/// [`crate::TfcTdfNet::load_pytorch`] applies.
///
/// Note the order: normalisation and activation come **before** the
/// convolution, not after. A port that reads `conv → norm → act` off habit
/// loads perfectly and computes a different function.
#[derive(Module, Debug)]
pub struct Tfc<B: Backend> {
    norm: Norm<B>,
    conv: Conv2d<B>,
}

impl<B: Backend> Tfc<B> {
    fn new(in_channels: usize, out_channels: usize, device: &B::Device) -> Self {
        Self {
            norm: Norm::new(in_channels, device),
            conv: Conv2dConfig::new([in_channels, out_channels], [3, 3])
                .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
                .with_bias(false)
                .init(device),
        }
    }

    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        self.conv.forward(gelu(self.norm.forward(x)))
    }
}

/// Time-*distributed* fully-connected: a bottleneck across the frequency axis.
///
/// `norm → GELU → Linear(f → f/bn) → norm → GELU → Linear(f/bn → f)`, which the
/// checkpoint numbers `.0`, `.2`, `.3`, `.5`. Both linears are bias-free.
///
/// The frequency axis is the **last** one here only because
/// [`crate::TfcTdfNet::forward`] transposed it there; see the module docs.
#[derive(Module, Debug)]
pub struct Tdf<B: Backend> {
    norm1: Norm<B>,
    down: Linear<B>,
    norm2: Norm<B>,
    up: Linear<B>,
}

impl<B: Backend> Tdf<B> {
    fn new(channels: usize, bins: usize, bottleneck: usize, device: &B::Device) -> Self {
        let narrow = bins / bottleneck;
        Self {
            norm1: Norm::new(channels, device),
            down: LinearConfig::new(bins, narrow)
                .with_bias(false)
                .init(device),
            norm2: Norm::new(channels, device),
            up: LinearConfig::new(narrow, bins)
                .with_bias(false)
                .init(device),
        }
    }

    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = self.down.forward(gelu(self.norm1.forward(x)));
        self.up.forward(gelu(self.norm2.forward(x)))
    }
}

/// One TFC-TDF residual block.
///
/// `shortcut` is a 1×1 convolution and is present on **every** block, including
/// the many where it maps a channel count onto itself — upstream builds it
/// unconditionally, unlike RMVPE's `ConvBlockRes`, whose `hasattr` probe is why
/// *that* one is an `Option`. The checkpoint confirms it: there is a
/// `shortcut.weight` for all 110 blocks, `[128, 128, 1, 1]` where the widths
/// match and `[640, 1280, 1, 1]` where the decoder has just concatenated a
/// skip. Making it optional here would leave 55 parameters unloaded.
#[derive(Module, Debug)]
pub struct TfcTdfBlock<B: Backend> {
    tfc1: Tfc<B>,
    tdf: Tdf<B>,
    tfc2: Tfc<B>,
    shortcut: Conv2d<B>,
}

impl<B: Backend> TfcTdfBlock<B> {
    fn new(
        in_channels: usize,
        channels: usize,
        bins: usize,
        bottleneck: usize,
        device: &B::Device,
    ) -> Self {
        Self {
            tfc1: Tfc::new(in_channels, channels, device),
            tdf: Tdf::new(channels, bins, bottleneck, device),
            tfc2: Tfc::new(channels, channels, device),
            shortcut: Conv2dConfig::new([in_channels, channels], [1, 1])
                .with_bias(false)
                .init(device),
        }
    }

    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let skip = self.shortcut.forward(x.clone());
        let x = self.tfc1.forward(x);
        // The TDF is a *residual on the TFC's output*, not a second path from
        // the block input. Reading it the other way is a plausible misreading
        // of upstream's three consecutive statements and changes what the
        // bottleneck sees.
        let x = x.clone() + self.tdf.forward(x);
        self.tfc2.forward(x) + skip
    }
}

/// `num_blocks_per_scale` of them in sequence — upstream's `TFC_TDF`.
#[derive(Module, Debug)]
pub struct TfcTdf<B: Backend> {
    blocks: Vec<TfcTdfBlock<B>>,
}

impl<B: Backend> TfcTdf<B> {
    pub(crate) fn new(
        in_channels: usize,
        channels: usize,
        blocks: usize,
        bins: usize,
        bottleneck: usize,
        device: &B::Device,
    ) -> Self {
        Self {
            blocks: (0..blocks)
                .map(|i| {
                    let input = if i == 0 { in_channels } else { channels };
                    TfcTdfBlock::new(input, channels, bins, bottleneck, device)
                })
                .collect(),
        }
    }

    pub(crate) fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        self.blocks.iter().fold(x, |acc, block| block.forward(acc))
    }
}

/// `norm → GELU → strided conv`, halving both axes.
///
/// The checkpoint spells the two children `conv.0` and `conv.2`, because
/// upstream wraps the whole thing in a module whose only field is a
/// `nn.Sequential` called `conv`. Flattening that to `norm`/`conv` is the one
/// remap here that removes a level rather than renaming one.
#[derive(Module, Debug)]
pub struct Downscale<B: Backend> {
    norm: Norm<B>,
    conv: Conv2d<B>,
}

impl<B: Backend> Downscale<B> {
    fn new(in_channels: usize, out_channels: usize, scale: [usize; 2], device: &B::Device) -> Self {
        Self {
            norm: Norm::new(in_channels, device),
            conv: Conv2dConfig::new([in_channels, out_channels], scale)
                .with_stride(scale)
                .with_padding(PaddingConfig2d::Valid)
                .with_bias(false)
                .init(device),
        }
    }

    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        self.conv.forward(gelu(self.norm.forward(x)))
    }
}

/// `norm → GELU → transposed conv`, doubling both axes. The mirror of
/// [`Downscale`], down to the `conv.0`/`conv.2` naming.
#[derive(Module, Debug)]
pub struct Upscale<B: Backend> {
    norm: Norm<B>,
    conv: ConvTranspose2d<B>,
}

impl<B: Backend> Upscale<B> {
    fn new(in_channels: usize, out_channels: usize, scale: [usize; 2], device: &B::Device) -> Self {
        Self {
            norm: Norm::new(in_channels, device),
            conv: ConvTranspose2dConfig::new([in_channels, out_channels], scale)
                .with_stride(scale)
                .with_bias(false)
                .init(device),
        }
    }

    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        self.conv.forward(gelu(self.norm.forward(x)))
    }
}

/// One contracting level: the residual stack, then the halving.
#[derive(Module, Debug)]
pub struct EncoderBlock<B: Backend> {
    tfc_tdf: TfcTdf<B>,
    downscale: Downscale<B>,
}

impl<B: Backend> EncoderBlock<B> {
    pub(crate) fn new(
        channels: usize,
        growth: usize,
        blocks: usize,
        bins: usize,
        bottleneck: usize,
        scale: [usize; 2],
        device: &B::Device,
    ) -> Self {
        Self {
            tfc_tdf: TfcTdf::new(channels, channels, blocks, bins, bottleneck, device),
            downscale: Downscale::new(channels, channels + growth, scale, device),
        }
    }

    /// Returns `(skip, downscaled)`. The skip is the stack's output *before*
    /// the halving, which is what the matching decoder level concatenates.
    pub(crate) fn forward(&self, x: Tensor<B, 4>) -> (Tensor<B, 4>, Tensor<B, 4>) {
        let x = self.tfc_tdf.forward(x);
        (x.clone(), self.downscale.forward(x))
    }
}

/// One expanding level: the doubling, the skip concatenation, then the stack.
#[derive(Module, Debug)]
pub struct DecoderBlock<B: Backend> {
    upscale: Upscale<B>,
    tfc_tdf: TfcTdf<B>,
}

impl<B: Backend> DecoderBlock<B> {
    pub(crate) fn new(
        channels: usize,
        growth: usize,
        blocks: usize,
        bins: usize,
        bottleneck: usize,
        scale: [usize; 2],
        device: &B::Device,
    ) -> Self {
        let narrower = channels - growth;
        Self {
            upscale: Upscale::new(channels, narrower, scale, device),
            // Twice `narrower` in, because the skip has just been concatenated.
            tfc_tdf: TfcTdf::new(2 * narrower, narrower, blocks, bins, bottleneck, device),
        }
    }

    pub(crate) fn forward(&self, x: Tensor<B, 4>, skip: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = self.upscale.forward(x);
        self.tfc_tdf.forward(Tensor::cat(vec![x, skip], 1))
    }
}

/// The head: `1×1 conv → GELU → 1×1 conv`, spelled `final_conv.0` and
/// `final_conv.2`.
///
/// The first convolution's input is the network's output *concatenated with the
/// mixture spectrum*, which is where its `128 + 16 = 144` input channels come
/// from; the second emits `stems × dim_c`.
#[derive(Module, Debug)]
pub struct FinalConv<B: Backend> {
    conv1: Conv2d<B>,
    conv2: Conv2d<B>,
}

impl<B: Backend> FinalConv<B> {
    pub(crate) fn new(
        in_channels: usize,
        channels: usize,
        out_channels: usize,
        device: &B::Device,
    ) -> Self {
        let point = |input: usize, output: usize| {
            Conv2dConfig::new([input, output], [1, 1])
                .with_bias(false)
                .init(device)
        };
        Self {
            conv1: point(in_channels, channels),
            conv2: point(channels, out_channels),
        }
    }

    pub(crate) fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        self.conv2.forward(gelu(self.conv1.forward(x)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::Distribution;

    type B = burn_ndarray::NdArray;

    /// [`Norm`] against Burn's own `InstanceNorm`, which routes through
    /// `group_norm` with one group per channel.
    ///
    /// Written out here rather than used directly, because Burn spells the
    /// affine pair `gamma`/`beta` and wraps it in `Option` while the checkpoint
    /// says `weight`/`bias` — but that is a naming choice, and the *arithmetic*
    /// had better be identical. This is the test for the trap the struct's doc
    /// describes: if this were a batch norm in evaluation mode it would
    /// normalise by a running mean and diverge here immediately, and every
    /// shape check and coverage count would still pass.
    #[test]
    fn the_norm_is_instance_normalisation() {
        let device = Default::default();
        let channels = 5;
        let mine = Norm::<B>::new(channels, &device);
        let theirs = burn::nn::InstanceNormConfig::new(channels).init::<B>(&device);

        let x = Tensor::<B, 4>::random(
            [2, channels, 7, 11],
            Distribution::Normal(0.0, 3.0),
            &device,
        );
        let (a, b): (Vec<f32>, Vec<f32>) = (
            mine.forward(x.clone()).into_data().to_vec().unwrap(),
            theirs.forward(x).into_data().to_vec().unwrap(),
        );
        let diff = a
            .iter()
            .zip(&b)
            .map(|(p, q)| (p - q).abs())
            .fold(0.0f32, f32::max);
        assert!(
            a.iter().all(|v| v.is_finite()),
            "normalisation must be finite"
        );
        assert!(diff < 1e-4, "instance normalisation differs by {diff}");
    }

    /// The config says `act: gelu`, which is `nn.GELU()` — the **erf** form,
    /// not the tanh approximation. Burn spells them `gelu` and
    /// `gelu_approximate`, and picking the wrong one is a small, finite,
    /// content-independent error that no coverage count and no finiteness
    /// assertion can see. Pinned against erf values computed by hand.
    #[test]
    fn the_activation_is_exact_gelu_not_the_tanh_approximation() {
        let device = Default::default();
        let x = Tensor::<B, 1>::from_floats([-2.0, -1.0, 0.0, 1.0, 2.0], &device);
        let got: Vec<f32> = gelu(x).into_data().to_vec().unwrap();
        // 0.5 * x * (1 + erf(x / sqrt(2)))
        let expected = [-0.045_500_3, -0.158_655_3, 0.0, 0.841_344_7, 1.954_499_7];
        for (g, e) in got.iter().zip(&expected) {
            assert!((g - e).abs() < 1e-5, "gelu gave {g}, expected {e}");
        }
        // The tanh approximation gives 0.841192 at x = 1; the tolerance above is
        // tight enough to reject it, which is the whole point of the test.
        assert!((got[3] - 0.841_192).abs() > 1e-5);
    }
}
