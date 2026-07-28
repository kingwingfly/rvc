//! Weight-normalised convolutions.
//!
//! RVC keeps `torch.nn.utils.weight_norm` parametrisation in its checkpoints:
//! each conv stores `weight_g` (`[out,1,1]`) and `weight_v` (`[out,in,k]`) instead
//! of a single `weight`. We mirror that exactly — holding `g` and `v` as params
//! and recomputing `w = g * v / ‖v‖` per forward — so the reference weights load
//! unchanged and training keeps the same parametrisation.

use burn::module::{Module, Param};
use burn::tensor::backend::Backend;
use burn::tensor::module::{conv_transpose1d, conv1d, conv2d};
use burn::tensor::ops::{ConvOptions, ConvTransposeOptions};
use burn::tensor::{Distribution, Tensor};

/// L2 norm of `v` over the non-output dims, keepdim → `[out,1,1]`.
fn norm_except_dim0<B: Backend>(v: Tensor<B, 3>) -> Tensor<B, 3> {
    v.powf_scalar(2.0).sum_dim(2).sum_dim(1).sqrt()
}

/// L2 norm of a rank-4 `v` over dims 1..4, keepdim → `[out,1,1,1]`.
fn norm_except_dim0_4<B: Backend>(v: Tensor<B, 4>) -> Tensor<B, 4> {
    v.powf_scalar(2.0).sum_dim(3).sum_dim(2).sum_dim(1).sqrt()
}

/// Weight-normalised 1-D convolution (`torch.nn.utils.weight_norm(Conv1d)`).
#[derive(Module, Debug)]
pub struct WeightNormConv1d<B: Backend> {
    weight_g: Param<Tensor<B, 3>>,
    weight_v: Param<Tensor<B, 3>>,
    bias: Param<Tensor<B, 1>>,
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
}

impl<B: Backend> WeightNormConv1d<B> {
    /// `in_ch → out_ch`, `kernel`-wide, with the given stride/padding/dilation.
    pub fn new(
        in_ch: usize,
        out_ch: usize,
        kernel: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        device: &B::Device,
    ) -> Self {
        Self::new_grouped(in_ch, out_ch, kernel, stride, padding, dilation, 1, device)
    }

    /// As [`Self::new`] but with grouped convolution.
    #[allow(clippy::too_many_arguments)]
    pub fn new_grouped(
        in_ch: usize,
        out_ch: usize,
        kernel: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        groups: usize,
        device: &B::Device,
    ) -> Self {
        let v = Tensor::random(
            [out_ch, in_ch / groups, kernel],
            Distribution::Normal(0.0, 0.02),
            device,
        );
        let g = norm_except_dim0(v.clone());
        Self {
            weight_g: Param::from_tensor(g),
            weight_v: Param::from_tensor(v),
            bias: Param::from_tensor(Tensor::zeros([out_ch], device)),
            stride,
            padding,
            dilation,
            groups,
        }
    }

    fn weight(&self) -> Tensor<B, 3> {
        let v = self.weight_v.val();
        let g = self.weight_g.val();
        let norm = norm_except_dim0(v.clone());
        v * (g / norm)
    }

    /// `x`: `[batch, in_ch, time]` → `[batch, out_ch, time']`.
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        conv1d(
            x,
            self.weight(),
            Some(self.bias.val()),
            ConvOptions::new([self.stride], [self.padding], [self.dilation], self.groups),
        )
    }
}

/// Weight-normalised 2-D convolution (`torch.nn.utils.weight_norm(Conv2d)`),
/// used by the period discriminators.
#[derive(Module, Debug)]
pub struct WeightNormConv2d<B: Backend> {
    weight_g: Param<Tensor<B, 4>>,
    weight_v: Param<Tensor<B, 4>>,
    bias: Param<Tensor<B, 1>>,
    stride: [usize; 2],
    padding: [usize; 2],
}

impl<B: Backend> WeightNormConv2d<B> {
    /// `in_ch → out_ch` with `kernel`=(h,w), stride=(h,w), padding=(h,w).
    pub fn new(
        in_ch: usize,
        out_ch: usize,
        kernel: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        device: &B::Device,
    ) -> Self {
        let v = Tensor::random(
            [out_ch, in_ch, kernel[0], kernel[1]],
            Distribution::Normal(0.0, 0.02),
            device,
        );
        let g = norm_except_dim0_4(v.clone());
        Self {
            weight_g: Param::from_tensor(g),
            weight_v: Param::from_tensor(v),
            bias: Param::from_tensor(Tensor::zeros([out_ch], device)),
            stride,
            padding,
        }
    }

    fn weight(&self) -> Tensor<B, 4> {
        let v = self.weight_v.val();
        let g = self.weight_g.val();
        let norm = norm_except_dim0_4(v.clone());
        v * (g / norm)
    }

    /// `x`: `[batch, in_ch, h, w]` → `[batch, out_ch, h', w']`.
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        conv2d(
            x,
            self.weight(),
            Some(self.bias.val()),
            ConvOptions::new(self.stride, self.padding, [1, 1], 1),
        )
    }
}

/// Weight-normalised 1-D transposed convolution (`weight_norm(ConvTranspose1d)`).
///
/// The transposed-conv weight is `[in, out, k]`, so weight-norm's dim-0 norm is
/// still over the trailing dims → `[in,1,1]`.
#[derive(Module, Debug)]
pub struct WeightNormConvTranspose1d<B: Backend> {
    weight_g: Param<Tensor<B, 3>>,
    weight_v: Param<Tensor<B, 3>>,
    bias: Param<Tensor<B, 1>>,
    stride: usize,
    padding: usize,
}

impl<B: Backend> WeightNormConvTranspose1d<B> {
    /// `in_ch → out_ch`, `kernel`-wide, with stride and (symmetric) padding.
    pub fn new(
        in_ch: usize,
        out_ch: usize,
        kernel: usize,
        stride: usize,
        padding: usize,
        device: &B::Device,
    ) -> Self {
        let v = Tensor::random(
            [in_ch, out_ch, kernel],
            Distribution::Normal(0.0, 0.02),
            device,
        );
        let g = norm_except_dim0(v.clone());
        Self {
            weight_g: Param::from_tensor(g),
            weight_v: Param::from_tensor(v),
            bias: Param::from_tensor(Tensor::zeros([out_ch], device)),
            stride,
            padding,
        }
    }

    fn weight(&self) -> Tensor<B, 3> {
        let v = self.weight_v.val();
        let g = self.weight_g.val();
        let norm = norm_except_dim0(v.clone());
        v * (g / norm)
    }

    /// `x`: `[batch, in_ch, time]` → `[batch, out_ch, time·stride]`.
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        conv_transpose1d(
            x,
            self.weight(),
            Some(self.bias.val()),
            ConvTransposeOptions::new([self.stride], [self.padding], [0], [1], 1),
        )
    }
}
