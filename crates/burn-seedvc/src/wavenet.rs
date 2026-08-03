//! The WaveNet block the diffusion transformer ends with.
//!
//! Eight gated, weight-normalised convolutions over the transformer's output,
//! conditioned on a second embedding of the flow time. It is the same
//! `WN`/`fused_add_tanh_sigmoid_multiply` shape VITS uses — and which
//! [`burn_vits::Wn`] already implements — but **it is not the same module**, for
//! two reasons that each change the arithmetic:
//!
//! - VITS conditions once, adding a single `g` broadcast over every layer. This
//!   one slices a per-layer window out of `cond_layer`'s `2 · hidden · n_layers`
//!   channels, which is what the checkpoint's `[8192, 512, 1]` records
//!   (`2 × 512 × 8`).
//! - The convolutions are upstream's `SConv1d`, which **ignores the `padding`
//!   argument it is handed and reflect-pads instead** — see [`reflect_pad`].
//!
//! Reusing [`burn_vits::Wn`] would therefore have meant bending a module that
//! RVC and GPT-SoVITS both depend on to fit a third caller agreeing with
//! neither. What *is* reused is [`burn_vits::WeightNormConv1d`], which mirrors
//! `torch.nn.utils.weight_norm`'s `weight_g`/`weight_v` pair exactly and needs no
//! change at all.
//!
//! # What is deliberately absent
//!
//! Upstream multiplies by `x_mask` in the residual and again at the output, and
//! drops out the gate at `p_dropout: 0.2`. Neither is modelled: dropout is a
//! training-time term, and the mask is all ones for a single clip — the only
//! batch shape inference builds, the classifier-free-guidance pair being the same
//! clip twice. **A padded batched trainer would have to add the mask back**, or
//! short samples would leak conditioning past their own end.
//!
//! # Provenance
//!
//! Ported from `modules/wavenet.py` and `modules/encodec.py` of Seed-VC
//! (<https://github.com/Plachtaa/seed-vc>, GPL-3.0), read as a reference and
//! never run or vendored.

use burn::module::Module;
use burn::tensor::Tensor;
use burn::tensor::activation::{sigmoid, tanh};
use burn::tensor::backend::Backend;
use burn_vits::WeightNormConv1d;

use crate::config::SeedVcConfig;

/// Pad `x` by reflection, `pad` frames at each end — what upstream's `SConv1d`
/// does instead of the zero padding its call site appears to ask for.
///
/// `WN` builds each layer as `SConv1d(…, padding=(k·d − d)/2, …)`, but
/// `SConv1d.__init__` never forwards `padding` to the `nn.Conv1d` it wraps: the
/// argument disappears into `**kwargs` and the convolution is built unpadded.
/// `SConv1d.forward` then pads the *input* by `k_eff − stride`, split as evenly
/// as an odd total allows, in `reflect` mode. **Reading the call site rather
/// than the wrapper is how this port would have acquired zero padding**, which
/// is not an error anywhere — it simply pulls the first and last `pad` frames of
/// every layer towards silence, eight times over.
///
/// Upstream additionally zero-extends a signal shorter than the padding before
/// reflecting. That path is not modelled: at the mel frame rate two frames is
/// 23 ms, and asserting beats quietly computing something else.
fn reflect_pad<B: Backend>(x: Tensor<B, 3>, pad: usize) -> Tensor<B, 3> {
    if pad == 0 {
        return x;
    }
    let [b, c, frames] = x.dims();
    assert!(
        frames > pad,
        "reflect padding of {pad} needs more than {pad} frames, got {frames}"
    );
    // Reflection excludes the edge sample itself: for pad = 2 the left side is
    // `x[2], x[1]` and the right side is `x[T-3], x[T-4]`.
    let left = x.clone().slice([0..b, 0..c, 1..pad + 1]).flip([2]);
    let right = x
        .clone()
        .slice([0..b, 0..c, (frames - 1 - pad)..(frames - 1)])
        .flip([2]);
    Tensor::cat(vec![left, x, right], 2)
}

/// Seed-VC's `WN`: gated convolutions with per-layer conditioning.
#[derive(Module, Debug)]
pub struct WaveNet<B: Backend> {
    in_layers: Vec<WeightNormConv1d<B>>,
    res_skip_layers: Vec<WeightNormConv1d<B>>,
    cond_layer: WeightNormConv1d<B>,
    hidden: usize,
    pad: usize,
}

impl<B: Backend> WaveNet<B> {
    /// Build the block from the preset's `wavenet_*` dimensions.
    ///
    /// **`dilation_rate` is 1 on this preset**, so `dilation_rate.pow(i)` is 1 at
    /// every layer and the receptive field grows linearly rather than
    /// exponentially — worth knowing before assuming a WaveNet-shaped module has
    /// WaveNet's usual reach. Nothing else here would need changing for a
    /// dilating preset except the per-layer padding, so the constructor refuses
    /// one rather than computing the wrong padding for it.
    pub fn new(cfg: &SeedVcConfig, device: &B::Device) -> Self {
        let hidden = cfg.hidden_dim;
        let n = cfg.wavenet_layers;
        let k = cfg.wavenet_kernel;
        assert_eq!(
            cfg.wavenet_dilation, 1,
            "only dilation_rate 1 is ported; a dilating preset needs per-layer padding"
        );

        Self {
            in_layers: (0..n)
                .map(|_| WeightNormConv1d::new(hidden, 2 * hidden, k, 1, 0, 1, device))
                .collect(),
            // The last layer feeds only the skip sum, so it needs no residual
            // half — which is why `res_skip_layers.7` is `[512, …]` where the
            // other seven are `[1024, …]`.
            res_skip_layers: (0..n)
                .map(|i| {
                    let out = if i + 1 < n { 2 * hidden } else { hidden };
                    WeightNormConv1d::new(hidden, out, 1, 1, 0, 1, device)
                })
                .collect(),
            cond_layer: WeightNormConv1d::new(hidden, 2 * hidden * n, 1, 1, 0, 1, device),
            hidden,
            pad: (k - 1) / 2,
        }
    }

    /// `x`: `[batch, hidden, frames]`, `g`: `[batch, hidden, 1]` — the flow-time
    /// embedding. Returns `[batch, hidden, frames]`.
    ///
    /// `g` is projected once and then read a window at a time, so each layer sees
    /// a different 1024-channel slice of the same projection.
    pub fn forward(&self, x: Tensor<B, 3>, g: Tensor<B, 3>) -> Tensor<B, 3> {
        let g = self.cond_layer.forward(g);
        let n = self.in_layers.len();

        let mut x = x;
        let mut output = Tensor::zeros_like(&x);
        for i in 0..n {
            let acts = self.in_layers[i].forward(reflect_pad(x.clone(), self.pad))
                + g.clone().narrow(1, i * 2 * self.hidden, 2 * self.hidden);
            // `fused_add_tanh_sigmoid_multiply`: the first half is the value, the
            // second the gate.
            let acts = tanh(acts.clone().narrow(1, 0, self.hidden))
                * sigmoid(acts.narrow(1, self.hidden, self.hidden));

            let res_skip = self.res_skip_layers[i].forward(acts);
            if i + 1 < n {
                x = x + res_skip.clone().narrow(1, 0, self.hidden);
                output = output + res_skip.narrow(1, self.hidden, self.hidden);
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

    type B = burn_ndarray::NdArray;

    #[test]
    fn reflection_mirrors_the_edges_without_repeating_them() {
        // The one piece of arithmetic here a reader would get wrong from the call
        // site alone, so it is pinned rather than inferred.
        let device = Default::default();
        let x = Tensor::<B, 1>::from_floats([1.0, 2.0, 3.0, 4.0, 5.0], &device).reshape([1, 1, 5]);
        let padded: Vec<f32> = reflect_pad(x, 2).into_data().to_vec().unwrap();
        assert_eq!(padded, vec![3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0]);
    }

    #[test]
    fn the_block_is_length_preserving_and_finite() {
        let cfg = SeedVcConfig::uvit_whisper_small_wavenet();
        let device = Default::default();
        let net = WaveNet::<B>::new(&cfg, &device);

        let x = Tensor::zeros([1, cfg.hidden_dim, 24], &device);
        let g = Tensor::zeros([1, cfg.hidden_dim, 1], &device);
        let out = net.forward(x, g);
        assert_eq!(out.dims(), [1, cfg.hidden_dim, 24]);
        // Separate from any value check: Burn's approximate comparisons treat NaN
        // as equal to NaN, so a NaN would pass one silently.
        assert!(!out.contains_nan().into_scalar());
    }
}
