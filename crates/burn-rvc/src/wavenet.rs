//! The WaveNet residual stack (`modules.WN`), shared by `enc_q` and the flow.

use burn::module::Module;
use burn::tensor::activation::{sigmoid, tanh};
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

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
        let cond_layer =
            WeightNormConv1d::new(gin_channels, 2 * h * n_layers, 1, 1, 0, 1, device);
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
        Self { cond_layer, in_layers, res_skip_layers, hidden_channels: h, n_layers }
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
