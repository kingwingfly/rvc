//! `net.style_encoder` — a mel in, one 192-dim timbre vector out.
//!
//! Two pointwise convolutions, two gated convolutions, one round of
//! self-attention, then an average over time. Averaging is what makes the
//! reference length irrelevant: 1 s and 30 s both condition the model with the
//! same shape, which is the whole premise of zero-shot conversion.
//!
//! It consumes an **80-band mel**, not a waveform and not the fbank a speaker
//! verifier would want — the first convolution is `[512, 80, 1]`, so it reads
//! the same representation the diffusion transformer predicts.
//!
//! # This is not the network upstream runs, and that matters
//!
//! **Seed-VC's released inference path takes its timbre vector from CAMPPlus,
//! loaded from a separate `campplus_cn_common.bin`, and never constructs these
//! 18 tensors at all.** `modules/commons.py::build_model` builds only `cfm` and
//! `length_regulator`; `load_checkpoint` iterates over what `build_model`
//! returned, so `net.style_encoder.*` is read past. `inference.py` instead
//! builds `CAMPPlus(feat_dim=80, embedding_size=192)`, loads that separate file,
//! and passes its output as the `style2` the transformer conditions on. The
//! preset's own config says as much: `style_encoder.campplus_path`.
//!
//! So these weights are a fossil of the training-time model, published in the
//! checkpoint because the trainer saved every branch of its module dict. They
//! are ported here because they are what the checkpoint contains and therefore
//! the only thing weight coverage can verify — **but wiring inference to this
//! module would produce a timbre vector Seed-VC was never conditioned on.**
//! Reaching parity with upstream needs CAMPPlus, which is a port of its own.
//!
//! # What is guessed, and how wrong it can be
//!
//! The class was never released: it is in no commit of the public repository,
//! and `build_model` has never referenced it. The layout below is read off the
//! tensor shapes and off the lineage the names give away — StyleSpeech's
//! `MelStyleEncoder` (`spectral`/`temporal`/`slf_attn`/`fc`, which
//! `burn-gptsovits`'s `reference.rs` ports in its original `Linear` flavour),
//! rewritten with `Conv1d` throughout so it can stay `[batch, channels, time]`,
//! and with VITS' `attentions.MultiHeadAttention` in place of the original
//! scaled-dot-product block — `conv_q`/`conv_k`/`conv_v`/`conv_o` are that
//! class's field names verbatim.
//!
//! Three things the checkpoint cannot settle, all of which load perfectly either
//! way, and none of which anything downstream would notice:
//!
//! - **the head count.** No shape depends on it, since there are no
//!   relative-position embeddings to size. [`StyleEncoderConfig::n_heads`] is 2,
//!   inherited from StyleSpeech's `style_head=2`, and is a guess.
//! - **the attention scale.** VITS divides by `√k_channels`; StyleSpeech divides
//!   by `√d_model`. This follows VITS, because the projection names say the VITS
//!   block was the one pasted in. With 2 heads over 512 channels the two differ
//!   by a factor of 4.
//! - **the residual around the attention.** StyleSpeech adds one inside its
//!   block; VITS' has none, and adds it in the caller. Kept, since the caller
//!   here *is* StyleSpeech's.
//!
//! **None of this is verified numerically**, and it cannot be from this
//! checkpoint alone — there is no reference implementation to diff against and
//! no downstream consumer whose output would degrade visibly. Treat the forward
//! pass as untested and the coverage number as covering the *layout* only.
//!
//! # Why the attention is written here
//!
//! `burn_vits::MultiHeadAttention` is the same class, but it always carries
//! `emb_rel_k`/`emb_rel_v` — upstream's `window_size` is mandatory there,
//! because both models that crate serves set it. This checkpoint has neither
//! tensor, so reusing it would report two missing parameters for ever and add a
//! relative-position term the trained weights never saw. Widening `burn-vits`
//! to make the window optional would change a crate two shipped engines depend
//! on, for a subtree that is not even on this engine's inference path.

use std::error::Error;
use std::path::Path;

use burn::module::Module;
use burn::nn::PaddingConfig1d;
use burn::nn::conv::{Conv1d, Conv1dConfig};
use burn::tensor::Tensor;
use burn::tensor::activation::{sigmoid, softmax, softplus, tanh};
use burn::tensor::backend::Backend;

/// The shape of the style encoder.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StyleEncoderConfig {
    /// Mel bands it consumes — `spectral.0.weight`'s middle axis.
    pub n_mels: usize,
    /// Working width, 512 throughout.
    pub hidden: usize,
    /// Width of the timbre vector produced — `style_encoder.dim`, 192.
    pub out_dim: usize,
    /// Gated-convolution kernel, 5.
    pub kernel_size: usize,
    /// Attention heads. **A guess** — see the module docs; no tensor's shape
    /// depends on it.
    pub n_heads: usize,
}

impl Default for StyleEncoderConfig {
    /// The released checkpoint's dimensions.
    fn default() -> Self {
        Self {
            n_mels: 80,
            hidden: 512,
            out_dim: 192,
            kernel_size: 5,
            n_heads: 2,
        }
    }
}

/// `x * tanh(softplus(x))` — Mish, the activation between the pointwise layers.
fn mish<B: Backend>(x: Tensor<B, 3>) -> Tensor<B, 3> {
    x.clone() * tanh(softplus(x, 1.0))
}

/// A gated convolution with a residual — `Conv1dGLU`.
///
/// The convolution emits twice the channels; half is the signal and half,
/// through a sigmoid, is the gate deciding how much of it passes. Padding keeps
/// the length, so the residual lines up.
#[derive(Module, Debug)]
pub struct Conv1dGlu<B: Backend> {
    conv1: Conv1d<B>,
    channels: usize,
}

impl<B: Backend> Conv1dGlu<B> {
    fn new(channels: usize, kernel: usize, device: &B::Device) -> Self {
        Self {
            conv1: Conv1dConfig::new(channels, channels * 2, kernel)
                .with_padding(PaddingConfig1d::Explicit(kernel / 2, kernel / 2))
                .init(device),
            channels,
        }
    }

    /// `x`: `[batch, channels, time]`.
    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch, _, time] = x.dims();
        let y = self.conv1.forward(x.clone());
        let signal = y.clone().slice([0..batch, 0..self.channels, 0..time]);
        let gate = y.slice([0..batch, self.channels..self.channels * 2, 0..time]);
        x + signal * sigmoid(gate)
    }
}

/// Self-attention over the reference frames, channels-first.
///
/// VITS' `attentions.MultiHeadAttention` with `window_size=None`: four
/// pointwise convolutions and no relative-position term.
#[derive(Module, Debug)]
pub struct StyleAttention<B: Backend> {
    conv_q: Conv1d<B>,
    conv_k: Conv1d<B>,
    conv_v: Conv1d<B>,
    conv_o: Conv1d<B>,
    n_heads: usize,
    k_channels: usize,
}

impl<B: Backend> StyleAttention<B> {
    fn new(channels: usize, n_heads: usize, device: &B::Device) -> Self {
        let conv = || Conv1dConfig::new(channels, channels, 1).init(device);
        Self {
            conv_q: conv(),
            conv_k: conv(),
            conv_v: conv(),
            conv_o: conv(),
            n_heads,
            k_channels: channels / n_heads,
        }
    }

    /// `x`: `[batch, channels, time]` → `[batch, channels, time]`.
    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch, channels, time] = x.dims();
        let (heads, k_channels) = (self.n_heads, self.k_channels);
        // [batch, channels, time] -> [batch, heads, time, k_channels]
        let split = |t: Tensor<B, 3>| t.reshape([batch, heads, k_channels, time]).swap_dims(2, 3);

        let q = split(self.conv_q.forward(x.clone())) / (k_channels as f64).sqrt();
        let k = split(self.conv_k.forward(x.clone()));
        let v = split(self.conv_v.forward(x));

        let attn = softmax(q.matmul(k.swap_dims(2, 3)), 3);
        let out = attn
            .matmul(v)
            .swap_dims(2, 3)
            .reshape([batch, channels, time]);
        self.conv_o.forward(out)
    }
}

/// The style encoder.
#[derive(Module, Debug)]
pub struct StyleEncoder<B: Backend> {
    spectral: Vec<Conv1d<B>>,
    temporal: Vec<Conv1dGlu<B>>,
    slf_attn: StyleAttention<B>,
    fc: Conv1d<B>,
}

impl<B: Backend> StyleEncoder<B> {
    pub fn new(cfg: &StyleEncoderConfig, device: &B::Device) -> Self {
        let pointwise = |input, output| Conv1dConfig::new(input, output, 1).init(device);
        Self {
            spectral: vec![
                pointwise(cfg.n_mels, cfg.hidden),
                pointwise(cfg.hidden, cfg.hidden),
            ],
            temporal: vec![
                Conv1dGlu::new(cfg.hidden, cfg.kernel_size, device),
                Conv1dGlu::new(cfg.hidden, cfg.kernel_size, device),
            ],
            slf_attn: StyleAttention::new(cfg.hidden, cfg.n_heads, device),
            fc: pointwise(cfg.hidden, cfg.out_dim),
        }
    }

    /// `mel`: `[batch, n_mels, frames]` → `[batch, out_dim, 1]`.
    ///
    /// The trailing axis of length one is deliberate: the transformer broadcasts
    /// this vector across time, so it is shaped like a one-frame sequence rather
    /// than a plain vector.
    ///
    /// Dropout sits between every stage upstream and is absent here — this is an
    /// inference port, where it is the identity.
    pub fn forward(&self, mel: Tensor<B, 3>) -> Tensor<B, 3> {
        let mut h = mel;
        for layer in &self.spectral {
            h = mish(layer.forward(h));
        }
        for layer in &self.temporal {
            h = layer.forward(h);
        }
        let h = h.clone() + self.slf_attn.forward(h);
        let h = self.fc.forward(h);

        // Average over time — the speaker is a property of the whole reference,
        // not of any frame in it.
        h.mean_dim(2)
    }

    /// Load `net.style_encoder.*` out of the Seed-VC checkpoint.
    ///
    /// Two remaps, and both are name-shape rather than layout changes:
    /// `net.style_encoder.module.` is the trainer's own nesting (a `Munch` of
    /// nets, each wrapped in `nn.DataParallel`), and `spectral.3` is the second
    /// convolution's index inside an `nn.Sequential` that also holds the two
    /// activations and the two dropouts — a `Vec` in Burn numbers only the
    /// parameterised entries, so it is `spectral.1` here.
    ///
    /// Everything is a convolution, so `PyTorchToBurnAdapter`'s `Linear`
    /// transposition never fires and there is nothing here that could be
    /// silently transposed.
    pub fn load_pytorch(
        &mut self,
        path: impl AsRef<Path>,
    ) -> Result<burn_store::ApplyResult, Box<dyn Error>> {
        let remaps = [
            (r"^net\.style_encoder\.module\.", ""),
            (r"^spectral\.3\.", "spectral.1."),
        ];
        burn_kit::store::load_pytorch_into::<B, _>(self, path.as_ref(), None, &remaps)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::Distribution;

    type B = burn_ndarray::NdArray;

    /// The point of the average pool: any amount of reference audio conditions
    /// the model with the same shape, so cloning does not depend on how much was
    /// supplied.
    #[test]
    fn any_length_of_reference_gives_one_timbre_vector() {
        let cfg = StyleEncoderConfig::default();
        let device = Default::default();
        let enc = StyleEncoder::<B>::new(&cfg, &device);
        for frames in [4, 37] {
            let mel = Tensor::zeros([1, cfg.n_mels, frames], &device);
            assert_eq!(enc.forward(mel).dims(), [1, cfg.out_dim, 1]);
        }
    }

    /// A timbre encoder that ignores its input is the failure mode that looks
    /// healthy: the pipeline runs, every clip converts, and every output is the
    /// same voice. So two different mels must disagree and one mel must agree
    /// with itself.
    #[test]
    fn different_references_give_different_vectors_and_the_same_one_repeats() {
        let cfg = StyleEncoderConfig::default();
        let device = Default::default();
        let enc = StyleEncoder::<B>::new(&cfg, &device);
        let normal = Distribution::Normal(0.0, 1.0);
        let a = Tensor::<B, 3>::random([1, cfg.n_mels, 20], normal, &device);
        let b = Tensor::<B, 3>::random([1, cfg.n_mels, 20], normal, &device);

        let va: Vec<f32> = enc.forward(a.clone()).into_data().to_vec().unwrap();
        let va_again: Vec<f32> = enc.forward(a).into_data().to_vec().unwrap();
        let vb: Vec<f32> = enc.forward(b).into_data().to_vec().unwrap();

        assert!(va.iter().all(|x| x.is_finite()));
        assert_eq!(va, va_again);
        assert!(
            va.iter().zip(&vb).any(|(x, y)| (x - y).abs() > 1e-6),
            "two unrelated references produced the same timbre vector"
        );
    }

    /// The gated convolution emits twice the channels and halves them again
    /// through the gate; a slice off by one would silently mix signal with gate
    /// and still typecheck.
    #[test]
    fn the_gated_convolution_keeps_its_width() {
        let device = Default::default();
        let glu = Conv1dGlu::<B>::new(8, 5, &device);
        let x = Tensor::random([1, 8, 10], Distribution::Normal(0.0, 1.0), &device);
        let out = glu.forward(x);
        assert_eq!(out.dims(), [1, 8, 10]);
        let v: Vec<f32> = out.into_data().to_vec().unwrap();
        assert!(v.iter().all(|x| x.is_finite()));
    }

    /// Softmax over the wrong axis is the classic attention bug that neither
    /// shapes nor weight coverage catch: the rows of the attention matrix must
    /// sum to one over *keys*, so a constant value sequence must come back
    /// unchanged whatever the queries are.
    #[test]
    fn attention_averages_over_keys() {
        let device = Default::default();
        let mut attn = StyleAttention::<B>::new(4, 2, &device);
        // Make `conv_v` and `conv_o` the identity so the output is exactly the
        // attention-weighted average of the input.
        let eye =
            || burn::module::Param::from_tensor(Tensor::<B, 2>::eye(4, &device).reshape([4, 4, 1]));
        attn.conv_v.weight = eye();
        attn.conv_o.weight = eye();
        attn.conv_v.bias = None;
        attn.conv_o.bias = None;

        let value = Tensor::<B, 1>::from_floats([1.0, -2.0, 3.0, -4.0], &device)
            .reshape([1, 4, 1])
            .repeat_dim(2, 6);
        let out: Vec<f32> = attn.forward(value).into_data().to_vec().unwrap();
        for (i, x) in out.iter().enumerate() {
            let expected = [1.0, -2.0, 3.0, -4.0][i / 6];
            assert!((x - expected).abs() < 1e-5, "{i}: {x} != {expected}");
        }
    }
}
