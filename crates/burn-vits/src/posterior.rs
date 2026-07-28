//! The posterior encoder (`enc_q`). Training-only: maps the linear spectrogram
//! to the latent distribution `(m, logs)` used by the KL term.

use burn::module::Module;
use burn::nn::conv::{Conv1d, Conv1dConfig};
use burn::tensor::Tensor;
use burn::tensor::backend::Backend;

use crate::wavenet::Wn;

const KERNEL: usize = 5;
const DILATION_RATE: usize = 1;
const N_LAYERS: usize = 16;

/// Posterior encoder over the linear spectrogram.
#[derive(Module, Debug)]
pub struct PosteriorEncoder<B: Backend> {
    pre: Conv1d<B>,
    enc: Wn<B>,
    proj: Conv1d<B>,
    out_channels: usize,
}

impl<B: Backend> PosteriorEncoder<B> {
    /// `spec_channels → out_channels` latent, via a `hidden_channels`-wide WN.
    pub fn new(
        spec_channels: usize,
        out_channels: usize,
        hidden_channels: usize,
        gin_channels: usize,
        device: &B::Device,
    ) -> Self {
        Self {
            pre: Conv1dConfig::new(spec_channels, hidden_channels, 1).init(device),
            enc: Wn::new(
                hidden_channels,
                KERNEL,
                DILATION_RATE,
                N_LAYERS,
                gin_channels,
                device,
            ),
            proj: Conv1dConfig::new(hidden_channels, out_channels * 2, 1).init(device),
            out_channels,
        }
    }

    /// `x`: `[batch, spec_channels, time]`, `g`: `[batch, gin, 1]`.
    /// Returns `(m, logs)`, each `[batch, out_channels, time]` (training samples
    /// `z = m + eps·exp(logs)`).
    pub fn forward(&self, x: Tensor<B, 3>, g: Tensor<B, 3>) -> (Tensor<B, 3>, Tensor<B, 3>) {
        let x = self.pre.forward(x);
        let x = self.enc.forward(x, g);
        let stats = self.proj.forward(x);
        let m = stats.clone().narrow(1, 0, self.out_channels);
        let logs = stats.narrow(1, self.out_channels, self.out_channels);
        (m, logs)
    }
}
