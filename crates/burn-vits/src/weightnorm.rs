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

#[cfg(test)]
impl<B: Backend> WeightNormConv1d<B> {
    /// Test-only: make [`Self::weight`] reconstruct exactly `w`.
    ///
    /// Storing `v = w` and `g = ‖w‖` gives `w · (‖w‖ / ‖w‖)`, which is `w`
    /// scaled by exactly one. It exists so [`crate::Wn`]'s tests can pin the
    /// gate and the conditioning offsets against hand-written arithmetic
    /// instead of against whatever the random init drew — the layers there are
    /// private to this module, so a caller could not do it.
    ///
    /// **A row of zeros makes the reconstruction `0/0`.** Every caller passes
    /// non-zero rows for that reason, which is also why the weights below are
    /// small integers rather than a convenient zero.
    pub(crate) fn set_weight(&mut self, w: Tensor<B, 3>) {
        self.weight_g = Param::from_tensor(norm_except_dim0(w.clone()));
        self.weight_v = Param::from_tensor(w);
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

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    use burn::tensor::TensorData;

    type B = NdArray;

    fn t<const D: usize>(data: &[f32], shape: [usize; D]) -> Tensor<B, D> {
        Tensor::from_data(TensorData::new(data.to_vec(), shape), &Default::default())
    }

    fn values<const D: usize>(x: Tensor<B, D>) -> Vec<f32> {
        x.into_data().to_vec().unwrap()
    }

    /// Scalar reference, and the one that pins the *axis*.
    ///
    /// `weight_norm(Conv1d)` defaults to `dim=0`, so the magnitude is one
    /// scalar per **output** channel and each row of `v` is normalised on its
    /// own. Row 0 is `[3, 4]` with norm 5 and `g = 10`, giving `[6, 8]`; row 1
    /// is `[0, 1]` with norm 1 and `g = 5`, giving `[0, 5]`. Every number is
    /// exact in binary, so this is an equality rather than a tolerance.
    ///
    /// Reach: taking the norm over the whole tensor instead — `√26 ≈ 5.0990` —
    /// would give row 0 as `[5.883, 7.844]`, and normalising per *element*
    /// would give `[10, 10]`. Neither is close to `[6, 8]`.
    #[test]
    fn the_1d_norm_is_per_output_channel() {
        let mut conv = WeightNormConv1d::<B>::new(1, 2, 2, 1, 0, 1, &Default::default());
        conv.weight_v = Param::from_tensor(t(&[3.0, 4.0, 0.0, 1.0], [2, 1, 2]));
        conv.weight_g = Param::from_tensor(t(&[10.0, 5.0], [2, 1, 1]));
        assert_eq!(values(conv.weight()), [6.0, 8.0, 0.0, 5.0]);
    }

    /// The same reconstruction seen through `forward`, so the test above is
    /// pinning the weight the convolution actually uses rather than a method
    /// nothing calls. A length-2 kernel over a length-2 input with no padding
    /// is one dot product per output channel: `6·1 + 8·2 = 22` and
    /// `0·1 + 5·2 = 10`. The bias is zero at init and left alone.
    #[test]
    fn forward_convolves_with_the_reconstructed_weight() {
        let mut conv = WeightNormConv1d::<B>::new(1, 2, 2, 1, 0, 1, &Default::default());
        conv.weight_v = Param::from_tensor(t(&[3.0, 4.0, 0.0, 1.0], [2, 1, 2]));
        conv.weight_g = Param::from_tensor(t(&[10.0, 5.0], [2, 1, 1]));
        assert_eq!(
            values(conv.forward(t(&[1.0, 2.0], [1, 1, 2]))),
            [22.0, 10.0]
        );
    }

    /// The two on-disk spellings pair `original0`→`weight_g` and
    /// `original1`→`weight_v`, and this pins that the pairing carries meaning
    /// rather than being a convention either way round.
    ///
    /// `torch.nn.utils.weight_norm` writes `weight_g`/`weight_v`;
    /// `torch.nn.utils.parametrizations.weight_norm`, which supersedes it,
    /// writes `parametrizations.weight.original0`/`original1` for the same two
    /// tensors in the same order. The rename belongs to whichever loader reads
    /// such a file — `burn-hubert` and `burn-seedvc` both carry it, and neither
    /// RVC nor GPT-SoVITS publishes a checkpoint in the new spelling, since
    /// both import the legacy `weight_norm` in every module this crate mirrors.
    /// What *this* file owns is the other half: that the two fields are not
    /// interchangeable, so a loader pairing `original0` with `weight_v` would
    /// be wrong rather than merely differently named.
    ///
    /// Read the swapped answer, because it is the reason a shape check is not
    /// the safety net it looks like. `g` is `[out, 1, 1]`, so read as `v` it
    /// broadcasts and its own per-row norm is `|g_i|` — the division cancels it
    /// outright and the weight comes back as **±`v`, with the magnitude
    /// discarded entirely**. Here that is `[3, 4, 0, 1]` where the right
    /// pairing gives `[6, 8, 0, 5]`: a convolution scaled by the wrong constant
    /// per output channel, loading at full coverage and running fine.
    #[test]
    fn the_magnitude_and_the_direction_do_not_commute() {
        let device = Default::default();
        let g = t(&[10.0, 5.0], [2, 1, 1]);
        let v = t(&[3.0, 4.0, 0.0, 1.0], [2, 1, 2]);

        let mut right = WeightNormConv1d::<B>::new(1, 2, 2, 1, 0, 1, &device);
        right.weight_g = Param::from_tensor(g.clone());
        right.weight_v = Param::from_tensor(v.clone());

        // The same two tensors read in the opposite order. `g` broadcasts into
        // `[2, 1, 2]`, so this is a load that succeeds rather than a shape error
        // — and a `[out, 1, 1]` `g` broadcasts against *any* `[out, in, k]`, so
        // that is the general case rather than an artefact of these dimensions.
        let mut swapped = WeightNormConv1d::<B>::new(1, 2, 2, 1, 0, 1, &device);
        swapped.weight_g = Param::from_tensor(v.clone());
        swapped.weight_v = Param::from_tensor(g);

        assert_eq!(values(right.weight()), [6.0, 8.0, 0.0, 5.0]);
        assert_eq!(values(swapped.weight()), values(v));
    }

    /// Invariant: a freshly initialised conv sets `g = ‖v‖`, so it reconstructs
    /// `v` itself — `v · (‖v‖/‖v‖)`, a scale by exactly one. Asserted as an
    /// equality because the two norms are the same computation on the same
    /// bits; a `g` computed over a different axis would not divide out.
    #[test]
    fn a_fresh_conv_reconstructs_its_own_direction() {
        let conv = WeightNormConv1d::<B>::new(3, 4, 5, 1, 2, 1, &Default::default());
        assert_eq!(values(conv.weight()), values(conv.weight_v.val()));
    }

    /// Scalar reference for the counterintuitive one.
    ///
    /// A `ConvTranspose1d` weight is `[in, out, k]`, and upstream applies plain
    /// `weight_norm(ConvTranspose1d(...))` — default `dim=0` — so the magnitude
    /// is per **input** channel, not per output channel. That reads like a
    /// transposition bug and is not: it is what the published `f0G48k.pth` and
    /// `s2G2333k.pth` carry, which is why `burn-rvc`'s load example reports
    /// 560/0/0 with this shape.
    ///
    /// Values as above: `[3, 4]` with `g = 10` → `[6, 8]`, `[0, 1]` with
    /// `g = 5` → `[0, 5]`, here indexed by input channel.
    #[test]
    fn the_transposed_norm_is_per_input_channel() {
        let mut conv = WeightNormConvTranspose1d::<B>::new(2, 1, 2, 1, 0, &Default::default());
        conv.weight_v = Param::from_tensor(t(&[3.0, 4.0, 0.0, 1.0], [2, 1, 2]));
        conv.weight_g = Param::from_tensor(t(&[10.0, 5.0], [2, 1, 1]));
        assert_eq!(values(conv.weight()), [6.0, 8.0, 0.0, 5.0]);
    }

    /// Scalar reference over all three trailing dims of a rank-4 weight.
    ///
    /// Output channel 0 is `[3, 4, 12, 0]` across `(in, h, w)`, whose norm is
    /// `√169 = 13`; with `g = 26` it doubles to `[6, 8, 24, 0]`. Channel 1 is
    /// `[0, 0, 0, 1]`, norm 1, `g = 3` → `[0, 0, 0, 3]`. Summing over only two
    /// of the three dims would leave a `[out, in, 1, 1]` magnitude and a
    /// different answer in every element but the zeros.
    #[test]
    fn the_2d_norm_covers_every_trailing_dim() {
        let mut conv =
            WeightNormConv2d::<B>::new(2, 2, [1, 2], [1, 1], [0, 0], &Default::default());
        conv.weight_v =
            Param::from_tensor(t(&[3.0, 4.0, 12.0, 0.0, 0.0, 0.0, 0.0, 1.0], [2, 2, 1, 2]));
        conv.weight_g = Param::from_tensor(t(&[26.0, 3.0], [2, 1, 1, 1]));
        assert_eq!(
            values(conv.weight()),
            [6.0, 8.0, 24.0, 0.0, 0.0, 0.0, 0.0, 3.0]
        );
    }
}
