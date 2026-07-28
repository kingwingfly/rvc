//! `modules.ResBlock1` — the HiFiGAN residual block.
//!
//! Shared because every VITS-lineage decoder stacks these; what differs between
//! them is the surrounding upsample chain and its conditioning, not this.

use burn::module::Module;
use burn::tensor::Tensor;
use burn::tensor::backend::Backend;

use crate::nn::leaky_relu;
use crate::weightnorm::WeightNormConv1d;

/// Leaky-ReLU slope the VITS decoders use throughout.
pub const LRELU_SLOPE: f64 = 0.1;

/// Padding that keeps a dilated odd-kernel convolution length-preserving.
pub fn get_padding(kernel: usize, dilation: usize) -> usize {
    dilation * (kernel - 1) / 2
}

/// HiFi-GAN ResBlock type 1: two dilated conv stacks with residual adds.
#[derive(Module, Debug)]
pub struct ResBlock1<B: Backend> {
    convs1: Vec<WeightNormConv1d<B>>,
    convs2: Vec<WeightNormConv1d<B>>,
}

impl<B: Backend> ResBlock1<B> {
    pub fn new(channels: usize, kernel: usize, dilations: &[usize], device: &B::Device) -> Self {
        let convs1 = dilations
            .iter()
            .map(|&d| {
                WeightNormConv1d::new(
                    channels,
                    channels,
                    kernel,
                    1,
                    get_padding(kernel, d),
                    d,
                    device,
                )
            })
            .collect();
        let convs2 = dilations
            .iter()
            .map(|_| {
                WeightNormConv1d::new(
                    channels,
                    channels,
                    kernel,
                    1,
                    get_padding(kernel, 1),
                    1,
                    device,
                )
            })
            .collect();
        Self { convs1, convs2 }
    }

    /// `x`: `[batch, channels, time]`.
    pub fn forward(&self, mut x: Tensor<B, 3>) -> Tensor<B, 3> {
        for (c1, c2) in self.convs1.iter().zip(self.convs2.iter()) {
            let xt = leaky_relu(x.clone(), LRELU_SLOPE);
            let xt = c1.forward(xt);
            let xt = leaky_relu(xt, LRELU_SLOPE);
            let xt = c2.forward(xt);
            x = x + xt;
        }
        x
    }
}
