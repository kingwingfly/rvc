//! The normalizing flow (`ResidualCouplingBlock` + `ResidualCouplingLayer`).
//!
//! All coupling layers are `mean_only` (additive), so `logs = 0`: forward adds
//! the predicted mean, reverse subtracts it. Flips (channel reversal) between
//! layers carry no parameters.
//!
//! **`post` is not zero-initialised here, and upstream's is.** Both references
//! end `ResidualCouplingLayer.__init__` with `self.post.weight.data.zero_()`
//! and the same for the bias (RVC `infer/module/modules.py`, GPT-SoVITS
//! `GPT_SoVITS/module/modules.py`), so every coupling starts as the identity
//! and the flow contributes nothing until it learns to; this port leaves
//! Burn's `Conv1dConfig` default in place instead. It reaches nothing that
//! loads weights — warm-start, resume and inference all overwrite `post`
//! before the first forward pass — so the difference is confined to training
//! from scratch, which is the path `rvc train` already warns about. Recorded
//! rather than changed: nothing here has been measured either way, and both
//! trainers' from-scratch mel figures would need re-baselining.

use burn::module::Module;
use burn::nn::conv::{Conv1d, Conv1dConfig};
use burn::tensor::Tensor;
use burn::tensor::backend::Backend;

use crate::wavenet::Wn;

const FLOW_KERNEL: usize = 5;
const FLOW_DILATION_RATE: usize = 1;

/// One additive coupling layer.
#[derive(Module, Debug)]
pub struct ResidualCouplingLayer<B: Backend> {
    pre: Conv1d<B>,
    enc: Wn<B>,
    post: Conv1d<B>,
    half_channels: usize,
}

impl<B: Backend> ResidualCouplingLayer<B> {
    /// `n_layers` is the depth of the coupling's WaveNet — 3 in RVC, 4 in
    /// GPT-SoVITS. A parameter rather than a constant because that is the only
    /// thing that differs between them here.
    pub fn new(
        channels: usize,
        hidden_channels: usize,
        gin_channels: usize,
        n_layers: usize,
        device: &B::Device,
    ) -> Self {
        let half = channels / 2;
        Self {
            pre: Conv1dConfig::new(half, hidden_channels, 1).init(device),
            enc: Wn::new(
                hidden_channels,
                FLOW_KERNEL,
                FLOW_DILATION_RATE,
                n_layers,
                gin_channels,
                device,
            ),
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
    /// `n_flows` coupling layers, each `n_layers` deep. Both RVC and GPT-SoVITS
    /// use four flows; they differ in the depth.
    pub fn new(
        channels: usize,
        hidden_channels: usize,
        gin_channels: usize,
        n_flows: usize,
        n_layers: usize,
        device: &B::Device,
    ) -> Self {
        let flows = (0..n_flows)
            .map(|_| {
                ResidualCouplingLayer::new(
                    channels,
                    hidden_channels,
                    gin_channels,
                    n_layers,
                    device,
                )
            })
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

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    use burn::tensor::Distribution;

    type B = NdArray;

    const CHANNELS: usize = 4;
    const HIDDEN: usize = 8;
    const GIN: usize = 2;
    const FLOWS: usize = 3;
    const DEPTH: usize = 2;
    const FRAMES: usize = 7;

    fn block() -> (ResidualCouplingBlock<B>, Tensor<B, 3>, Tensor<B, 3>) {
        let device = Default::default();
        let net = ResidualCouplingBlock::new(CHANNELS, HIDDEN, GIN, FLOWS, DEPTH, &device);
        let x = Tensor::random(
            [1, CHANNELS, FRAMES],
            Distribution::Normal(0.0, 1.0),
            &device,
        );
        let g = Tensor::random([1, GIN, 1], Distribution::Normal(0.0, 1.0), &device);
        (net, x, g)
    }

    fn max_abs(x: Tensor<B, 3>) -> f64 {
        x.abs().max().into_scalar() as f64
    }

    /// Exactly-zero invariant: an additive coupling touches only the second
    /// half of the channels, and the first half is the *same tensor* on the way
    /// out — not a recomputation of it. So the difference is bit-for-bit zero,
    /// and a `narrow` whose offset or length slipped would show up here rather
    /// than as a slightly wrong conversion.
    #[test]
    fn a_coupling_layer_leaves_the_first_half_untouched() {
        let device = Default::default();
        let layer = ResidualCouplingLayer::<B>::new(CHANNELS, HIDDEN, GIN, DEPTH, &device);
        let x = Tensor::random(
            [1, CHANNELS, FRAMES],
            Distribution::Normal(0.0, 1.0),
            &device,
        );
        let g = Tensor::random([1, GIN, 1], Distribution::Normal(0.0, 1.0), &device);
        let half = CHANNELS / 2;
        let out = layer.forward(x.clone(), g);
        let d = out.narrow(1, 0, half) - x.narrow(1, 0, half);
        assert_eq!(max_abs(d), 0.0);
    }

    /// Round trip: the flow's whole job is to be invertible, and
    /// `Synthesizer::forward_train` runs `forward` where inference runs
    /// `reverse`, so a mis-indexed coupling or a flip on the wrong side of the
    /// loop is a training/inference mismatch nothing else observes.
    ///
    /// The first assertion is what keeps the second one meaningful: with
    /// `post` zero-initialised — which is what upstream does and what this port
    /// does not, see the module docs — every coupling would be the identity and
    /// the round trip would prove nothing at all. So the test refuses to pass
    /// on a flow that did not move its input.
    #[test]
    fn the_flow_round_trips() {
        let (net, x, g) = block();
        let z = net.forward(x.clone(), g.clone());
        assert!(
            max_abs(z.clone() - x.clone()) > 1e-3,
            "the flow is the identity here, so the round trip proves nothing"
        );
        assert!(max_abs(net.reverse(z, g) - x) < 1e-5);
    }

    /// The injected hazard for the test above, and the reason it has reach.
    ///
    /// `reverse` flips *before* each coupling and walks the stack backwards,
    /// mirroring `forward`, which flips after. Dropping the flips — the natural
    /// simplification, since they carry no parameters — leaves an inverse that
    /// is still shaped right, still finite, and no longer an inverse. This
    /// reproduces that here rather than in the shipped code.
    #[test]
    fn reversing_without_the_flips_does_not_recover_the_input() {
        let (net, x, g) = block();
        let mut z = net.forward(x.clone(), g.clone());
        for flow in net.flows.iter().rev() {
            z = flow.reverse(z, g.clone());
        }
        assert!(max_abs(z - x) > 1e-3);
    }
}
