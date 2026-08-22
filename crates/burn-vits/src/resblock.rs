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

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    use burn::tensor::TensorData;

    type B = NdArray;

    const LEN: usize = 64;
    const IMPULSE: usize = 32;

    /// Scalar reference, transcribed from `commons.get_padding`:
    /// `int((kernel_size * dilation - dilation) / 2)`.
    #[test]
    fn get_padding_matches_the_reference() {
        // RVC's decoder uses kernels 3/7/11 and dilations 1/3/5.
        assert_eq!(get_padding(3, 1), 1);
        assert_eq!(get_padding(3, 3), 3);
        assert_eq!(get_padding(3, 5), 5);
        assert_eq!(get_padding(7, 1), 3);
        assert_eq!(get_padding(7, 3), 9);
        assert_eq!(get_padding(11, 1), 5);
        assert_eq!(get_padding(11, 5), 25);
    }

    /// Where an impulse reaches after the block, as an index range.
    ///
    /// Every conv here is bias-free at init and a leaky ReLU maps zero to zero,
    /// so anything outside the receptive field is **exactly** zero rather than
    /// merely small — the returned bounds are found by scanning for `!= 0.0`.
    fn support(kernel: usize, dilations: &[usize]) -> (usize, usize) {
        let device = Default::default();
        let block = ResBlock1::<B>::new(1, kernel, dilations, &device);
        let mut samples = vec![0.0f32; LEN];
        samples[IMPULSE] = 1.0;
        let x = Tensor::from_data(TensorData::new(samples, [1, 1, LEN]), &device);
        let out = block.forward(x);
        assert_eq!(out.dims(), [1, 1, LEN], "the block must preserve its shape");
        let y: Vec<f32> = out.into_data().to_vec().unwrap();
        let first = y.iter().position(|v| *v != 0.0).expect("some output");
        let last = y.iter().rposition(|v| *v != 0.0).unwrap();
        (first, last)
    }

    /// Exactly-zero invariant, and the one that pins the dilation schedule.
    ///
    /// Each `(c1, c2)` pair spreads an impulse by `d·(k-1)/2` through the
    /// dilated conv and by `(k-1)/2` through the second, which upstream fixes
    /// at dilation 1 — a radius of `(d + 1)·(k-1)/2` per pair, and the pairs
    /// compose, so the whole block reaches the sum. The numbers below are that
    /// arithmetic done by hand: `(3, [1])` is 2, `(3, [5])` is 6, `(7, [3])` is
    /// 12, and RVC's `(3, [1, 3, 5])` is `2 + 4 + 6 = 12`.
    ///
    /// Reach: dropping the dilation from `convs1` collapses `(3, [5])` to 2 and
    /// the schedule to 6; giving `convs2` the dilation too would take them to
    /// 10 and 18. Both fail here, and neither changes a shape, a coverage count
    /// or a finiteness check.
    #[test]
    fn an_impulse_reaches_exactly_the_dilated_receptive_field() {
        for (kernel, dilations, radius) in [
            (3, &[1][..], 2),
            (3, &[5][..], 6),
            (7, &[3][..], 12),
            (3, &[1, 3, 5][..], 12),
            (11, &[1][..], 10),
        ] {
            assert_eq!(
                support(kernel, dilations),
                (IMPULSE - radius, IMPULSE + radius),
                "kernel {kernel}, dilations {dilations:?}"
            );
        }
    }

    /// The block is length-preserving for every kernel/dilation RVC and
    /// GPT-SoVITS actually configure, which is what `get_padding` buys and what
    /// the decoders' `Tensor::cat` over parallel resblocks depends on.
    #[test]
    fn the_decoder_schedules_preserve_length() {
        let device = Default::default();
        for kernel in [3, 7, 11] {
            let block = ResBlock1::<B>::new(2, kernel, &[1, 3, 5], &device);
            let x = Tensor::zeros([1, 2, LEN], &device);
            assert_eq!(block.forward(x).dims(), [1, 2, LEN], "kernel {kernel}");
        }
    }
}
