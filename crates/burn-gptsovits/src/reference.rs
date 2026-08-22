//! `ref_enc` — the reference encoder that supplies the speaker vector.
//!
//! A MelStyleEncoder: two pointwise layers, two gated convolutions, one round of
//! self-attention, then an average over time. The result is a single vector that
//! conditions the flow and the decoder, so it is where a synthesised voice gets
//! its timbre from.
//!
//! Nothing here is shared with RVC, which takes its speaker vector from a
//! lookup table of trained ids rather than from a reference recording — that
//! difference is the whole reason GPT-SoVITS clones from a few seconds of audio.

use burn::module::Module;
use burn::nn::conv::{Conv1d, Conv1dConfig};
use burn::nn::{Linear, LinearConfig, PaddingConfig1d};
use burn::tensor::Tensor;
use burn::tensor::activation::{sigmoid, softmax, softplus, tanh};
use burn::tensor::backend::Backend;

/// The shape of the reference encoder.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReferenceConfig {
    /// Width of the reference features. 704 for v2 — a hardcoded constant
    /// upstream, not the mel count it looks like.
    pub in_dim: usize,
    pub hidden: usize,
    /// Width of the speaker vector produced — `gin_channels`.
    pub out_dim: usize,
    pub kernel_size: usize,
    pub n_head: usize,
}

impl Default for ReferenceConfig {
    fn default() -> Self {
        Self {
            in_dim: 704,
            hidden: 128,
            out_dim: 512,
            kernel_size: 5,
            n_head: 2,
        }
    }
}

/// `x * tanh(softplus(x))` — the activation upstream uses between the pointwise
/// layers.
fn mish<B: Backend, const D: usize>(x: Tensor<B, D>) -> Tensor<B, D> {
    x.clone() * tanh(softplus(x, 1.0))
}

/// `LinearNorm` — a plain `Linear` under a wrapper that exists upstream only to
/// carry an optional spectral norm, which these checkpoints do not use. Kept as
/// a struct because the extra `fc` level is in the parameter names.
#[derive(Module, Debug)]
pub struct LinearNorm<B: Backend> {
    fc: Linear<B>,
}

impl<B: Backend> LinearNorm<B> {
    fn new(input: usize, output: usize, device: &B::Device) -> Self {
        Self {
            fc: LinearConfig::new(input, output).init(device),
        }
    }
}

/// A gated convolution with a residual — `Conv1dGLU`.
///
/// The convolution produces twice the channels; half is the signal and half,
/// through a sigmoid, is the gate that decides how much of it passes.
#[derive(Module, Debug)]
pub struct Conv1dGlu<B: Backend> {
    conv1: ConvNorm<B>,
    channels: usize,
}

/// `ConvNorm` — same story as [`LinearNorm`]: a wrapper whose only lasting
/// effect is the extra `conv` level in the parameter names.
#[derive(Module, Debug)]
pub struct ConvNorm<B: Backend> {
    conv: Conv1d<B>,
}

impl<B: Backend> Conv1dGlu<B> {
    fn new(channels: usize, kernel: usize, device: &B::Device) -> Self {
        Self {
            conv1: ConvNorm {
                conv: Conv1dConfig::new(channels, channels * 2, kernel)
                    .with_padding(PaddingConfig1d::Explicit(kernel / 2, kernel / 2))
                    .init(device),
            },
            channels,
        }
    }

    /// `x`: `[batch, channels, time]`.
    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch, _, time] = x.dims();
        let y = self.conv1.conv.forward(x.clone());
        let signal = y.clone().slice([0..batch, 0..self.channels, 0..time]);
        let gate = y.slice([0..batch, self.channels..self.channels * 2, 0..time]);
        x + signal * sigmoid(gate)
    }
}

/// Self-attention over the reference frames.
///
/// Its own module rather than `burn-vits`' because two things differ and both
/// are visible in the checkpoint: the projections are `Linear` named
/// `w_qs`/`w_ks`/`w_vs`/`fc` rather than convolutions, and there are no
/// relative-position embeddings.
///
/// It also scales by **√d_model, not √d_head** — upstream passes
/// `temperature=d_model ** 0.5`. With two heads over 128 channels that is √128
/// where the conventional choice is √64, so the conventional one is wrong here
/// by a factor of √2 and would load perfectly well.
#[derive(Module, Debug)]
pub struct StyleAttention<B: Backend> {
    w_qs: Linear<B>,
    w_ks: Linear<B>,
    w_vs: Linear<B>,
    fc: Linear<B>,
    n_head: usize,
}

impl<B: Backend> StyleAttention<B> {
    fn new(d_model: usize, n_head: usize, device: &B::Device) -> Self {
        let linear = || LinearConfig::new(d_model, d_model).init(device);
        Self {
            w_qs: linear(),
            w_ks: linear(),
            w_vs: linear(),
            fc: linear(),
            n_head,
        }
    }

    /// `x`: `[batch, time, d_model]`.
    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch, time, d_model] = x.dims();
        let d_head = d_model / self.n_head;
        let heads = |t: Tensor<B, 3>| {
            t.reshape([batch, time, self.n_head, d_head])
                .swap_dims(1, 2)
        };

        let q = heads(self.w_qs.forward(x.clone()));
        let k = heads(self.w_ks.forward(x.clone()));
        let v = heads(self.w_vs.forward(x.clone()));

        let scores = q.matmul(k.swap_dims(2, 3)) / (d_model as f64).sqrt();
        let out = softmax(scores, 3)
            .matmul(v)
            .swap_dims(1, 2)
            .reshape([batch, time, d_model]);
        self.fc.forward(out) + x
    }
}

/// The reference encoder.
#[derive(Module, Debug)]
pub struct ReferenceEncoder<B: Backend> {
    spectral: Vec<LinearNorm<B>>,
    temporal: Vec<Conv1dGlu<B>>,
    slf_attn: StyleAttention<B>,
    fc: LinearNorm<B>,
}

impl<B: Backend> ReferenceEncoder<B> {
    pub fn new(cfg: &ReferenceConfig, device: &B::Device) -> Self {
        Self {
            spectral: vec![
                LinearNorm::new(cfg.in_dim, cfg.hidden, device),
                LinearNorm::new(cfg.hidden, cfg.hidden, device),
            ],
            temporal: vec![
                Conv1dGlu::new(cfg.hidden, cfg.kernel_size, device),
                Conv1dGlu::new(cfg.hidden, cfg.kernel_size, device),
            ],
            slf_attn: StyleAttention::new(cfg.hidden, cfg.n_head, device),
            fc: LinearNorm::new(cfg.hidden, cfg.out_dim, device),
        }
    }

    /// `x`: `[batch, in_dim, frames]` → `[batch, out_dim, 1]`.
    ///
    /// The trailing axis of length one is deliberate: everything downstream
    /// broadcasts this vector across time, so it is shaped like a
    /// one-frame sequence rather than a plain vector.
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let mut h = x.swap_dims(1, 2);
        for layer in &self.spectral {
            h = mish(layer.fc.forward(h));
        }

        // This is the one place in the crate where the write burn-tch forgets
        // about actually happens: `Conv1dGlu::forward` ends in
        // `x + signal * sigmoid(gate)`, and on LibTorch that add mutates this
        // transposed view in place, straight through into the spectral stack's
        // output (CLAUDE.md, **`swap_dims` on LibTorch returns a view burn-tch
        // forgets the provenance of**). It is benign because that output is
        // moved in here and no other handle to it survives, so the write lands
        // in a buffer nothing else reads — not because the view is inert.
        let mut h = h.swap_dims(1, 2);
        for layer in &self.temporal {
            h = layer.forward(h);
        }

        let h = self.slf_attn.forward(h.swap_dims(1, 2));
        let h = self.fc.fc.forward(h);

        // Average over time — the speaker is a property of the whole reference,
        // not of any frame in it.
        h.mean_dim(1).swap_dims(1, 2)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    type B = burn_ndarray::NdArray;

    #[test]
    fn any_length_of_reference_gives_one_speaker_vector() {
        // The point of the average pool: three seconds of reference audio and
        // thirty both condition the model with the same shape, so cloning does
        // not depend on how much was supplied.
        let cfg = ReferenceConfig::default();
        let device = Default::default();
        let enc = ReferenceEncoder::<B>::new(&cfg, &device);
        for frames in [4, 50] {
            let x = Tensor::zeros([1, cfg.in_dim, frames], &device);
            assert_eq!(enc.forward(x).dims(), [1, cfg.out_dim, 1]);
        }
    }

    #[test]
    fn the_gated_convolution_keeps_its_width() {
        // It emits twice the channels and halves them again through the gate; a
        // slice off by one would silently mix signal with gate.
        let device = Default::default();
        let glu = Conv1dGlu::<B>::new(8, 5, &device);
        let x = Tensor::random(
            [1, 8, 10],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        let out = glu.forward(x);
        assert_eq!(out.dims(), [1, 8, 10]);
        let v: Vec<f32> = out.into_data().to_vec().unwrap();
        assert!(v.iter().all(|x| x.is_finite()));
    }
}

/// The backend that has the aliasing defect, against one that does not.
///
/// `t2s.rs` carries the same cross-backend check for `FusedAttention`; this is
/// the crate's other place where a `swap_dims` view is mutated **in place** —
/// see the comment on the transpose in [`ReferenceEncoder::forward`], which is
/// what the write there lands on. It duplicates that file's fill harness rather
/// than sharing it, because a `cfg(test)` module is private to its own file and
/// reaching across would make one test module part of the crate's internal
/// surface.
#[cfg(all(test, feature = "tch"))]
mod tch_aliasing {
    use super::*;
    use burn::module::{ModuleMapper, Param};
    use burn::tensor::TensorData;

    type Tch = burn::backend::LibTorch<f32>;
    type Nd = burn_ndarray::NdArray;

    /// Small, and still wide enough that `n_head` divides `hidden` into more
    /// than one head — a single head makes the transposed view degenerate.
    fn tiny() -> ReferenceConfig {
        ReferenceConfig {
            in_dim: 24,
            hidden: 16,
            out_dim: 8,
            kernel_size: 5,
            n_head: 2,
        }
    }

    /// A fixed LCG, so the two backends get byte-identical weights without a
    /// record round-trip. `Distribution` is not required to draw the same
    /// numbers on two backends, and a test that assumed it would fail for a
    /// reason that is not the one it is for.
    struct Lcg(u64);

    impl Lcg {
        fn take(&mut self, n: usize) -> Vec<f32> {
            (0..n)
                .map(|_| {
                    self.0 = self
                        .0
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    ((self.0 >> 40) as f32 / 8388608.0) - 1.0
                })
                .collect()
        }
    }

    /// Replaces every parameter with that sequence, in module-tree order.
    ///
    /// Initialised weights would not do: the point is to make every stage of the
    /// encoder carry signal, so a corrupted intermediate has somewhere to show.
    struct Fill(Lcg);

    impl<B: Backend> ModuleMapper<B> for Fill {
        fn map_float<const D: usize>(&mut self, param: Param<Tensor<B, D>>) -> Param<Tensor<B, D>> {
            let (device, shape) = (param.val().device(), param.val().shape());
            let data = TensorData::new(self.0.take(shape.num_elements()), shape.clone());
            param.map(move |_| Tensor::from_data(data.clone(), &device))
        }
    }

    fn speaker<B: Backend>(cfg: &ReferenceConfig, device: &B::Device) -> Vec<f32> {
        let enc = ReferenceEncoder::<B>::new(cfg, device).map(&mut Fill(Lcg(1)));
        let frames = 12;
        let x = Tensor::<B, 3>::from_data(
            TensorData::new(Lcg(7).take(cfg.in_dim * frames), [1, cfg.in_dim, frames]),
            device,
        );
        enc.forward(x).into_data().to_vec().unwrap()
    }

    /// LibTorch and `ndarray` are two independent implementations of the same
    /// arithmetic, so on one set of weights and one input they have to agree.
    ///
    /// That is the reading that catches an aliased view whatever it corrupts:
    /// `ndarray` refcounts its buffers correctly, so it computes what the code
    /// says while LibTorch computes what the code plus the defect say. The
    /// residual add inside `Conv1dGlu` writes *through* the transpose in
    /// `ReferenceEncoder::forward`; today the buffer it reaches is owned by that
    /// expression alone, and this is what reports it if a second handle on the
    /// spectral stack's output ever arrives.
    #[test]
    fn libtorch_agrees_with_ndarray_through_the_gated_convolutions() {
        let cfg = tiny();
        let want = speaker::<Nd>(&cfg, &Default::default());
        let got = speaker::<Tch>(&cfg, &Default::default());

        assert!(
            got.iter().all(|v| v.is_finite()),
            "LibTorch speaker vector must be finite"
        );
        assert_eq!(want.len(), got.len(), "both backends give one vector");
        let worst = want
            .iter()
            .zip(&got)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let scale = want.iter().fold(0.0f32, |a, b| a.max(b.abs()));
        assert!(
            worst < 1e-4 * scale.max(1.0),
            "the two backends differ by {worst} on a vector peaking at {scale}"
        );
    }
}
