//! The WaveNet residual stack (`modules.WN`), shared by `enc_q` and the flow.

use burn::module::Module;
use burn::tensor::Tensor;
use burn::tensor::activation::{sigmoid, tanh};
use burn::tensor::backend::Backend;

use crate::weightnorm::WeightNormConv1d;

/// Gated WaveNet stack with speaker conditioning.
#[derive(Module, Debug)]
pub struct Wn<B: Backend> {
    cond_layer: WeightNormConv1d<B>,
    in_layers: Vec<WeightNormConv1d<B>>,
    res_skip_layers: Vec<WeightNormConv1d<B>>,
    hidden_channels: usize,
    n_layers: usize,
}

impl<B: Backend> Wn<B> {
    /// `hidden_channels` wide, `n_layers` deep, dilation `dilation_rate**i`.
    pub fn new(
        hidden_channels: usize,
        kernel_size: usize,
        dilation_rate: usize,
        n_layers: usize,
        gin_channels: usize,
        device: &B::Device,
    ) -> Self {
        let h = hidden_channels;
        let cond_layer = WeightNormConv1d::new(gin_channels, 2 * h * n_layers, 1, 1, 0, 1, device);
        let mut in_layers = Vec::with_capacity(n_layers);
        let mut res_skip_layers = Vec::with_capacity(n_layers);
        for i in 0..n_layers {
            let dilation = dilation_rate.pow(i as u32);
            let padding = (kernel_size * dilation - dilation) / 2;
            in_layers.push(WeightNormConv1d::new(
                h,
                2 * h,
                kernel_size,
                1,
                padding,
                dilation,
                device,
            ));
            let res_skip_ch = if i < n_layers - 1 { 2 * h } else { h };
            res_skip_layers.push(WeightNormConv1d::new(h, res_skip_ch, 1, 1, 0, 1, device));
        }
        Self {
            cond_layer,
            in_layers,
            res_skip_layers,
            hidden_channels: h,
            n_layers,
        }
    }

    /// `x`, `g` (speaker cond): `[batch, hidden, time]` / `[batch, gin, 1]`.
    pub fn forward(&self, x: Tensor<B, 3>, g: Tensor<B, 3>) -> Tensor<B, 3> {
        let h = self.hidden_channels;
        let g = self.cond_layer.forward(g); // [b, 2h*n_layers, t]
        let mut x = x;
        let mut output = x.zeros_like();
        for i in 0..self.n_layers {
            let x_in = self.in_layers[i].forward(x.clone());
            let g_l = g.clone().narrow(1, i * 2 * h, 2 * h);
            let in_act = x_in + g_l;
            let acts = tanh(in_act.clone().narrow(1, 0, h)) * sigmoid(in_act.narrow(1, h, h));
            let res_skip = self.res_skip_layers[i].forward(acts);
            if i < self.n_layers - 1 {
                x = x + res_skip.clone().narrow(1, 0, h);
                output = output + res_skip.narrow(1, h, h);
            } else {
                output = output + res_skip;
            }
        }
        output
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

    fn values(x: Tensor<B, 3>) -> Vec<f32> {
        x.into_data().to_vec().unwrap()
    }

    /// `tanh(a)·sigmoid(b)`, written from `commons`'s
    /// `fused_add_tanh_sigmoid_multiply` rather than from this module.
    fn gate(a: f64, b: f64) -> f64 {
        a.tanh() * (1.0 / (1.0 + (-b).exp()))
    }

    /// Exactly-zero invariant, the shape `burn-mdx` uses: every path from input
    /// to output passes through convolutions whose bias is zero at init, and
    /// `tanh(0) = 0` kills the gate, so silence in must be silence out with no
    /// rounding to hide behind. A stray constant anywhere in the stack — a
    /// residual added before its gate, a skip taken from the wrong slice —
    /// shows up as a non-zero here.
    #[test]
    fn zero_in_is_exactly_zero_out() {
        let device = Default::default();
        let wn = Wn::<B>::new(4, 3, 2, 3, 2, &device);
        let out = wn.forward(
            Tensor::zeros([1, 4, 9], &device),
            Tensor::zeros([1, 2, 1], &device),
        );
        assert_eq!(out.abs().max().into_scalar(), 0.0);
    }

    /// Scalar reference for the gated activation.
    ///
    /// One layer, one hidden channel and a 1-wide kernel, with the input conv
    /// set to `[1, 2]` so the two halves of its output are `x` and `2x`, and
    /// the res/skip conv set to 1 so the layer's output *is* the gate. The
    /// conditioning is zero, so `in_act` is the input conv's output alone.
    ///
    /// Reach: the asymmetry is deliberate. Taking `tanh` of the *second* half
    /// and `sigmoid` of the first gives `tanh(2x)·sigmoid(x)`, which differs at
    /// every `x != 0` — where `[1, 1]` would have made the two indistinguishable.
    #[test]
    fn the_gate_is_tanh_of_the_first_half_times_sigmoid_of_the_second() {
        let device = Default::default();
        let mut wn = Wn::<B>::new(1, 1, 1, 1, 1, &device);
        wn.in_layers[0].set_weight(t(&[1.0, 2.0], [2, 1, 1]));
        wn.res_skip_layers[0].set_weight(t(&[1.0], [1, 1, 1]));

        let xs = [0.5f64, -0.75, 0.25];
        let x = t(&xs.map(|v| v as f32), [1, 1, 3]);
        let got = values(wn.forward(x, Tensor::zeros([1, 1, 1], &device)));
        for (g, x) in got.iter().zip(xs) {
            let want = gate(x, 2.0 * x);
            assert!((*g as f64 - want).abs() < 1e-6, "{g} vs {want}");
        }
    }

    /// Scalar reference for the res/skip split.
    ///
    /// A non-final layer's `res_skip` conv emits `2h` channels: the first `h`
    /// are added back into `x` and the second `h` accumulate into the output.
    /// Two layers, one hidden channel, and coefficients 3 and 5 chosen distinct
    /// so that swapping the halves is visible — it would feed `5·acts0` into
    /// the residual and `3·acts0` into the skip, changing both terms.
    ///
    /// The final layer emits only `h` channels and contributes all of them to
    /// the output, which is the other half of what this pins.
    #[test]
    fn the_residual_and_skip_halves_do_not_swap() {
        let device = Default::default();
        let mut wn = Wn::<B>::new(1, 1, 1, 2, 1, &device);
        wn.in_layers[0].set_weight(t(&[1.0, 2.0], [2, 1, 1]));
        wn.in_layers[1].set_weight(t(&[1.0, 2.0], [2, 1, 1]));
        wn.res_skip_layers[0].set_weight(t(&[3.0, 5.0], [2, 1, 1]));
        wn.res_skip_layers[1].set_weight(t(&[7.0], [1, 1, 1]));

        let x0 = 0.3f64;
        let got = values(wn.forward(
            t(&[x0 as f32], [1, 1, 1]),
            Tensor::zeros([1, 1, 1], &device),
        ));

        let acts0 = gate(x0, 2.0 * x0);
        let x1 = x0 + 3.0 * acts0;
        let want = 5.0 * acts0 + 7.0 * gate(x1, 2.0 * x1);
        assert!((got[0] as f64 - want).abs() < 1e-6, "{} vs {want}", got[0]);
    }

    /// Scalar reference for the conditioning offsets.
    ///
    /// `cond_layer` emits `2·h·n_layers` channels in one convolution and layer
    /// `i` reads the slice at `i·2h`. Feeding zeros as `x` makes every input
    /// conv contribute only its zero bias, so `in_act` *is* the conditioning
    /// slice and the four cond channels `[1, 2, 3, 4]` are directly readable:
    /// layer 0 must gate on `(1, 2)` and layer 1 on `(3, 4)`.
    ///
    /// Reach: an offset of `i·h` would hand layer 1 `(2, 3)` and a fixed offset
    /// of 0 would hand it `(1, 2)`. Both load at full coverage, run finite, and
    /// condition the flow on the wrong speaker channels.
    #[test]
    fn each_layer_reads_its_own_slice_of_the_conditioning() {
        let device = Default::default();
        let mut wn = Wn::<B>::new(1, 1, 1, 2, 1, &device);
        wn.cond_layer
            .set_weight(t(&[1.0, 2.0, 3.0, 4.0], [4, 1, 1]));
        wn.in_layers[0].set_weight(t(&[1.0, 2.0], [2, 1, 1]));
        wn.in_layers[1].set_weight(t(&[1.0, 2.0], [2, 1, 1]));
        wn.res_skip_layers[0].set_weight(t(&[3.0, 5.0], [2, 1, 1]));
        wn.res_skip_layers[1].set_weight(t(&[7.0], [1, 1, 1]));

        let got = values(wn.forward(Tensor::zeros([1, 1, 1], &device), t(&[1.0], [1, 1, 1])));

        let acts0 = gate(1.0, 2.0);
        let x1 = 3.0 * acts0;
        let want = 5.0 * acts0 + 7.0 * gate(x1 + 3.0, 2.0 * x1 + 4.0);
        assert!((got[0] as f64 - want).abs() < 1e-6, "{} vs {want}", got[0]);
    }
}
