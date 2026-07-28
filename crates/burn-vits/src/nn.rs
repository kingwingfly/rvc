//! Small shared building blocks.

use burn::module::{Module, Param};
use burn::tensor::Tensor;
use burn::tensor::backend::Backend;

/// Leaky ReLU with the given negative slope (RVC uses 0.1).
pub fn leaky_relu<B: Backend, const D: usize>(x: Tensor<B, D>, slope: f64) -> Tensor<B, D> {
    let neg = x.clone().mul_scalar(slope);
    x.clone().mask_where(x.lower_elem(0.0), neg)
}

/// RVC's channel-wise LayerNorm (`modules.LayerNorm`): normalises over the
/// channel axis of a `[batch, channels, time]` tensor, with learnable
/// `gamma`/`beta` per channel.
#[derive(Module, Debug)]
pub struct VitsLayerNorm<B: Backend> {
    pub gamma: Param<Tensor<B, 1>>,
    pub beta: Param<Tensor<B, 1>>,
    eps: f64,
}

impl<B: Backend> VitsLayerNorm<B> {
    /// Create with `channels` and the reference eps (1e-5).
    pub fn new(channels: usize, device: &B::Device) -> Self {
        Self {
            gamma: Param::from_tensor(Tensor::ones([channels], device)),
            beta: Param::from_tensor(Tensor::zeros([channels], device)),
            eps: 1e-5,
        }
    }

    /// `x`: `[batch, channels, time]` → normalised over `channels`.
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [_, c, _] = x.dims();
        let mean = x.clone().mean_dim(1);
        let centered = x - mean;
        let var = centered.clone().powf_scalar(2.0).mean_dim(1);
        let normed = centered / (var + self.eps).sqrt();
        let gamma = self.gamma.val().reshape([1, c, 1]);
        let beta = self.beta.val().reshape([1, c, 1]);
        normed * gamma + beta
    }
}
