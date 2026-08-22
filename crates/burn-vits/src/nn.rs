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

#[cfg(test)]
mod tests {
    use super::*;

    type B = burn::backend::NdArray;

    fn vec1(t: Tensor<B, 1>) -> Vec<f32> {
        t.into_data().to_vec().unwrap()
    }
    fn vec3(t: Tensor<B, 3>) -> Vec<f32> {
        t.into_data().to_vec().unwrap()
    }
    fn worst(got: &[f32], want: &[f32]) -> f32 {
        assert_eq!(got.len(), want.len());
        assert!(got.iter().all(|v| v.is_finite()), "output must be finite");
        got.iter()
            .zip(want)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max)
    }

    /// Scalar reference, written as a literal table rather than as the formula:
    /// the negative branch is `slope * x` and the positive one is `x` untouched,
    /// with zero on the *positive* side (`x < 0`, not `x <= 0`). A `mask_where`
    /// whose condition is inverted scales the positives instead, which this
    /// table sees and a symmetric input would not.
    #[test]
    fn leaky_relu_matches_a_hand_written_table() {
        let device = Default::default();
        let x = Tensor::<B, 1>::from_floats([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0], &device);
        let got = vec1(leaky_relu(x, 0.1));
        let want = [-0.2, -0.1, -0.05, 0.0, 0.5, 1.0, 2.0];
        assert!(worst(&got, &want) < 1e-7, "{got:?}");
    }

    /// The channel-wise LayerNorm against arithmetic done by hand.
    ///
    /// The two time columns are `[1, 3, 5]` and `[2, 6, 10]` — the second is
    /// exactly twice the first, so normalising **over channels** must give both
    /// columns the *same* values, while normalising over time (the same shapes,
    /// the same finiteness) could not. Biased variance, matching
    /// `F.layer_norm`'s, and the reference eps of 1e-5.
    #[test]
    fn layer_norm_matches_hand_computed_statistics() {
        let device = Default::default();
        let mut ln = VitsLayerNorm::<B>::new(3, &device);
        // Per channel, so gamma and beta are per *row* of the table below.
        ln.gamma = Param::from_tensor(Tensor::from_floats([1.0, 2.0, 3.0], &device));
        ln.beta = Param::from_tensor(Tensor::from_floats([10.0, 20.0, 30.0], &device));

        // [batch = 1, channels = 3, time = 2], channel-major.
        let x = Tensor::<B, 1>::from_floats([1.0, 2.0, 3.0, 6.0, 5.0, 10.0], &device)
            .reshape([1, 3, 2]);
        let got = vec3(ln.forward(x));

        // Column 0: mean 3, biased var ((-2)^2 + 0 + 2^2)/3 = 8/3.
        // Column 1: mean 6, biased var ((-4)^2 + 0 + 4^2)/3 = 32/3.
        let s0 = (8.0f64 / 3.0 + 1e-5).sqrt();
        let s1 = (32.0f64 / 3.0 + 1e-5).sqrt();
        let n = [[-2.0 / s0, -4.0 / s1], [0.0, 0.0], [2.0 / s0, 4.0 / s1]];
        let (gamma, beta) = ([1.0, 2.0, 3.0], [10.0, 20.0, 30.0]);
        let want: Vec<f32> = (0..3)
            .flat_map(|c| (0..2).map(move |t| (n[c][t] * gamma[c] + beta[c]) as f32))
            .collect();
        assert!(worst(&got, &want) < 1e-5, "{got:?} vs {want:?}");

        // The two columns are one scalar multiple apart, so a per-column
        // normalisation cancels it: state that separately from the table, since
        // it is the property a wrong `mean_dim` axis breaks.
        for c in 0..3 {
            assert!((got[c * 2] - got[c * 2 + 1]).abs() < 1e-5);
        }
    }
}
