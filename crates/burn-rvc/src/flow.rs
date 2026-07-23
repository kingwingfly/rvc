//! The normalizing flow (`ResidualCouplingBlock` + `ResidualCouplingLayer`).
//!
//! All coupling layers are `mean_only` (additive), so `logs = 0`: forward adds
//! the predicted mean, reverse subtracts it. Flips (channel reversal) between
//! layers carry no parameters.

use burn::module::Module;
use burn::nn::conv::{Conv1d, Conv1dConfig};
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

use crate::wavenet::Wn;

const FLOW_KERNEL: usize = 5;
const FLOW_DILATION_RATE: usize = 1;
const FLOW_N_LAYERS: usize = 3;

/// One additive coupling layer.
#[derive(Module, Debug)]
pub struct ResidualCouplingLayer<B: Backend> {
    pre: Conv1d<B>,
    enc: Wn<B>,
    post: Conv1d<B>,
    half_channels: usize,
}

impl<B: Backend> ResidualCouplingLayer<B> {
    fn new(channels: usize, hidden_channels: usize, gin_channels: usize, device: &B::Device) -> Self {
        let half = channels / 2;
        Self {
            pre: Conv1dConfig::new(half, hidden_channels, 1).init(device),
            enc: Wn::new(hidden_channels, FLOW_KERNEL, FLOW_DILATION_RATE, FLOW_N_LAYERS, gin_channels, device),
            post: Conv1dConfig::new(hidden_channels, half, 1).init(device),
            half_channels: half,
        }
    }

    fn mean(&self, x0: Tensor<B, 3>, g: Tensor<B, 3>) -> Tensor<B, 3> {
        let h = self.pre.forward(x0);
        let h = self.enc.forward(h, g);
        self.post.forward(h)
    }

    /// Forward transform: `x1 += mean(x0)`.
    pub fn forward(&self, x: Tensor<B, 3>, g: Tensor<B, 3>) -> Tensor<B, 3> {
        let x0 = x.clone().narrow(1, 0, self.half_channels);
        let x1 = x.narrow(1, self.half_channels, self.half_channels);
        let m = self.mean(x0.clone(), g);
        Tensor::cat(vec![x0, x1 + m], 1)
    }

    /// Inverse transform: `x1 -= mean(x0)`.
    pub fn reverse(&self, x: Tensor<B, 3>, g: Tensor<B, 3>) -> Tensor<B, 3> {
        let x0 = x.clone().narrow(1, 0, self.half_channels);
        let x1 = x.narrow(1, self.half_channels, self.half_channels);
        let m = self.mean(x0.clone(), g);
        Tensor::cat(vec![x0, x1 - m], 1)
    }
}

/// Stack of coupling layers with channel flips between them.
#[derive(Module, Debug)]
pub struct ResidualCouplingBlock<B: Backend> {
    flows: Vec<ResidualCouplingLayer<B>>,
}

impl<B: Backend> ResidualCouplingBlock<B> {
    /// `n_flows` coupling layers (RVC uses 4).
    pub fn new(
        channels: usize,
        hidden_channels: usize,
        gin_channels: usize,
        n_flows: usize,
        device: &B::Device,
    ) -> Self {
        let flows = (0..n_flows)
            .map(|_| ResidualCouplingLayer::new(channels, hidden_channels, gin_channels, device))
            .collect();
        Self { flows }
    }

    /// Prior → latent (training direction).
    pub fn forward(&self, mut x: Tensor<B, 3>, g: Tensor<B, 3>) -> Tensor<B, 3> {
        for flow in &self.flows {
            x = flow.forward(x, g.clone());
            x = x.flip([1]);
        }
        x
    }

    /// Latent → prior (inference direction).
    pub fn reverse(&self, mut x: Tensor<B, 3>, g: Tensor<B, 3>) -> Tensor<B, 3> {
        for flow in self.flows.iter().rev() {
            x = x.flip([1]);
            x = flow.reverse(x, g.clone());
        }
        x
    }
}
