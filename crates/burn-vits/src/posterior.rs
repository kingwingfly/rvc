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

#[cfg(test)]
mod tests {
    use super::*;
    use burn::module::Param;
    use burn::tensor::Distribution;

    type B = burn_ndarray::NdArray;
    type Dev = burn_ndarray::NdArrayDevice;

    const HIDDEN: usize = 8;
    const GIN: usize = 4;

    fn flat(t: Tensor<B, 3>) -> Vec<f32> {
        t.into_data().to_vec().unwrap()
    }

    fn encoder(spec: usize, out: usize, device: &Dev) -> PosteriorEncoder<B> {
        PosteriorEncoder::new(spec, out, HIDDEN, GIN, device)
    }

    /// The projection emits `2 * out_channels` and the first half is the mean.
    ///
    /// Scalar reference by construction: `proj`'s weight is zeroed and its bias
    /// set to `0, 1, 2, …`, so channel `c` of the statistics tensor is the
    /// constant `c` whatever the WaveNet stack in front of it did. The mean must
    /// then be `0..out` and the log-scale `out..2*out` — swap the two `narrow`
    /// offsets and both halves are still the right shape, still finite, and
    /// exchanged.
    #[test]
    fn the_statistics_split_is_mean_first_then_log_scale() {
        let device = Default::default();
        let (out, t) = (3usize, 4usize);
        let mut pe = encoder(6, out, &device);
        pe.proj.weight = Param::from_tensor(Tensor::zeros([out * 2, HIDDEN, 1], &device));
        let ramp: Vec<f32> = (0..out * 2).map(|i| i as f32).collect();
        pe.proj.bias = Some(Param::from_tensor(Tensor::<B, 1>::from_floats(
            ramp.as_slice(),
            &device,
        )));

        let x = Tensor::<B, 3>::random([1, 6, t], Distribution::Normal(0.0, 1.0), &device);
        let g = Tensor::<B, 3>::random([1, GIN, 1], Distribution::Normal(0.0, 1.0), &device);
        let (m, logs) = pe.forward(x, g);

        let want = |base: usize| -> Vec<f32> {
            (0..out).flat_map(|c| (0..t).map(move |_| (base + c) as f32)).collect()
        };
        let (m, logs) = (flat(m), flat(logs));
        assert!(m.iter().chain(&logs).all(|v| v.is_finite()));
        assert_eq!(m, want(0));
        assert_eq!(logs, want(out));
    }

    /// The encoder keeps the frame rate and narrows the width: a `[b, spec, t]`
    /// spectrogram becomes two `[b, out, t]` tensors, with `t` untouched at
    /// every length. A stride or a pad anywhere in the stack shows up here and
    /// nowhere else, since the KL term consumes both halves elementwise.
    #[test]
    fn the_frame_rate_survives_and_the_width_narrows() {
        let device = Default::default();
        let (spec, out) = (17usize, 5usize);
        let pe = encoder(spec, out, &device);
        for t in [1usize, 7, 32] {
            let x = Tensor::<B, 3>::random([2, spec, t], Distribution::Normal(0.0, 1.0), &device);
            let g = Tensor::<B, 3>::random([2, GIN, 1], Distribution::Normal(0.0, 1.0), &device);
            let (m, logs) = pe.forward(x, g);
            assert_eq!(m.dims(), [2, out, t], "length {t}");
            assert_eq!(logs.dims(), [2, out, t], "length {t}");
            assert!(flat(m).iter().all(|v| v.is_finite()), "length {t}");
            assert!(flat(logs).iter().all(|v| v.is_finite()), "length {t}");
        }
    }

    /// The speaker conditioning has to *reach* the output. `g` enters through
    /// one conv inside the WaveNet stack and is added to every layer's
    /// activation, so dropping it entirely — a plausible edit, since the tensor
    /// is threaded through by hand — changes no shape and breaks no test that
    /// only looks at one call. Two different `g` on one `x` must disagree.
    #[test]
    fn the_speaker_conditioning_changes_the_output() {
        let device = Default::default();
        let (spec, out, t) = (9usize, 4usize, 6usize);
        let pe = encoder(spec, out, &device);
        let x = Tensor::<B, 3>::random([1, spec, t], Distribution::Normal(0.0, 1.0), &device);
        let run = |g: Tensor<B, 3>| flat(pe.forward(x.clone(), g).0);
        let a = run(Tensor::zeros([1, GIN, 1], &device));
        let b = run(Tensor::ones([1, GIN, 1], &device));
        let spread = a
            .iter()
            .zip(&b)
            .map(|(p, q)| (p - q).abs())
            .fold(0.0f32, f32::max);
        assert!(spread > 1e-6, "the conditioning never reaches the mean");
    }
}
