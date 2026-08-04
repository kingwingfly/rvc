//! The RMVPE pitch-estimation network in [Burn](https://burn.dev).
//!
//! RMVPE — *Robust Model for Vocal Pitch Estimation* — is what tells RVC which
//! note a frame of the source is on, and preserving that contour is the whole
//! reason a converted voice keeps its expression. Until this crate existed it
//! ran only on ONNX Runtime, which made ORT a hard requirement for every `rvc`
//! invocation; the port is an **addition**, so `rmvpe.onnx` stays a supported
//! deployment target and gains company rather than being replaced.
//!
//! `[batch, 128, frames]` log-mel in, `[batch, frames, 360]` cents-bin salience
//! out. **That is the whole of this crate's job.** The mel front end and the
//! salience → Hz decode are already pure Rust and live in `rvc-core`
//! (`mel.rs`, `dsp::rmvpe_decode`), backend-agnostic and shared by both
//! runtimes, so duplicating either here would give the two backends two chances
//! to disagree about something neither of them computes.
//!
//! # The frame count is padded here, not by the caller
//!
//! The U-net halves time five times, so a forward pass needs a multiple of 32
//! frames. [`Rmvpe::forward`] pads to that itself and trims the salience back,
//! because the requirement is a property of *this network's shape* — a caller
//! that has to know about it is a caller that can get it wrong, and the ONNX
//! path in `rvc-core::f0` had to encode the same arithmetic by hand.
//!
//! The padding is **zeros**, which is upstream's `F.pad(..., mode="constant")`.
//! `rvc-core::f0` reflect-pads instead; both trim afterwards, so the two can
//! differ only over the last ≤ 31 frames and only by whatever leaks in through
//! the convolution stack's receptive field. This crate follows upstream because
//! upstream is what the published weights were run with.
//!
//! # Provenance and what is checked
//!
//! Ported from RVC-Project `2.3.260718`, `infer/rmvpe.py`, read as a reference
//! and never run. The instantiation is upstream's `E2E(4, 1, (2, 2))`, and
//! `rmvpe.pt` is a bare `torch.load` state dict — no wrapper key and no `model`
//! nesting, unlike the generator checkpoints in the same repository.
//!
//! `examples/load` reports coverage, which says the module *tree* is right and
//! nothing about the arithmetic. The second check is in [`gru`], where the risk
//! actually is: a bidirectional GRU with the gates in the wrong order or the two
//! biases fused loads at 100% and predicts confident nonsense.

pub mod gru;
pub mod unet;

use std::error::Error;
use std::path::Path;

use burn::module::Module;
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::{Linear, LinearConfig, PaddingConfig2d};
use burn::tensor::Tensor;
use burn::tensor::activation::sigmoid;
use burn::tensor::backend::Backend;

use gru::BiGru;
use unet::DeepUnet;

/// The shape of the network. The defaults are `rmvpe.pt`.
///
/// The 2×2 kernel of upstream's `E2E(4, 1, (2, 2))` is not a field: it is the
/// pooling *and* the transposed-convolution stride, and the decoder's
/// `output_padding` is chosen for that stride specifically, so a different value
/// is a different port rather than a different configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RmvpeConfig {
    /// Residual blocks per encoder, bottleneck and decoder level — upstream's
    /// `n_blocks`, 4.
    pub n_blocks: usize,
    /// Encoder levels, and equally decoder levels. 5, which is where the
    /// multiple-of-32 frame alignment comes from.
    pub en_de_layers: usize,
    /// Bottleneck stacks, 4.
    pub inter_layers: usize,
    /// Channels entering the U-net. 1 — a log-mel is a single-channel image.
    pub in_channels: usize,
    /// Channels the first encoder level emits, doubling per level. 16.
    pub en_out_channels: usize,
    /// Mel bins consumed, 128. Also the U-net's width, so it must divide by 32.
    pub n_mels: usize,
    /// Cents bins emitted, 360 — 20 cents apart from 1997.38, which is what
    /// `rvc_core::dsp::rmvpe_decode` inverts.
    pub n_class: usize,
    /// Hidden width of each GRU direction, 256; the `Linear` above it therefore
    /// sees 512.
    pub gru_hidden: usize,
}

impl Default for RmvpeConfig {
    fn default() -> Self {
        Self {
            n_blocks: 4,
            en_de_layers: 5,
            inter_layers: 4,
            in_channels: 1,
            en_out_channels: 16,
            n_mels: 128,
            n_class: 360,
            gru_hidden: 256,
        }
    }
}

impl RmvpeConfig {
    /// Frames the network's time axis has to be a multiple of: one factor of two
    /// per encoder level.
    pub fn frame_alignment(&self) -> usize {
        1 << self.en_de_layers
    }
}

/// RMVPE, upstream's `E2E`.
#[derive(Module, Debug)]
pub struct Rmvpe<B: Backend> {
    unet: DeepUnet<B>,
    cnn: Conv2d<B>,
    /// Upstream's `fc.0.gru`; the `nn.Sequential` it sits in is flattened away
    /// at load time.
    gru: BiGru<B>,
    /// Upstream's `fc.1`. The `Dropout(0.25)` between it and the sigmoid is not
    /// modelled — this crate only ever runs in evaluation mode, where dropout is
    /// the identity.
    fc: Linear<B>,
    align: usize,
}

impl<B: Backend> Rmvpe<B> {
    pub fn new(cfg: &RmvpeConfig, device: &B::Device) -> Self {
        Self {
            unet: DeepUnet::new(cfg, device),
            cnn: Conv2dConfig::new([cfg.en_out_channels, 3], [3, 3])
                .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
                .init(device),
            gru: BiGru::new(3 * cfg.n_mels, cfg.gru_hidden, device),
            fc: LinearConfig::new(2 * cfg.gru_hidden, cfg.n_class).init(device),
            align: cfg.frame_alignment(),
        }
    }

    /// `[batch, n_mels, frames]` log-mel → `[batch, frames, 360]` salience in
    /// `(0, 1)`.
    ///
    /// The frame count is padded to a multiple of 32 and the result trimmed back,
    /// so any number of frames is accepted — see the crate docs for why that
    /// belongs here.
    pub fn forward(&self, mel: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch, n_mels, frames] = mel.dims();
        let padded = frames.div_ceil(self.align) * self.align;
        let mel = if padded == frames {
            mel
        } else {
            let zeros = Tensor::zeros([batch, n_mels, padded - frames], &mel.device());
            Tensor::cat(vec![mel, zeros], 2)
        };

        // `[batch, 1, frames, bins]`: time becomes the height and the mel bins
        // the width, which is the orientation every block below assumes.
        let x = mel.swap_dims(1, 2).reshape([batch, 1, padded, n_mels]);
        let x = self.cnn.forward(self.unet.forward(x));
        // `[batch, frames, 3, bins]` flattened to `[batch, frames, 3 * bins]` —
        // channel-major, which is what `transpose(1, 2).flatten(-2)` gives.
        let x = x.swap_dims(1, 2).reshape([batch, padded, 3 * n_mels]);
        let x = sigmoid(self.fc.forward(self.gru.forward(x)));
        x.narrow(1, 0, frames)
    }

    /// Load `rmvpe.pt` (Hugging Face `lj1995/VoiceConversionWebUI`).
    ///
    /// A bare state dict — no `top_level_key`, where the RVC generator
    /// checkpoints in the same repository wrap theirs under `"model"`.
    ///
    /// Every remap undoes an `nn.Sequential`'s positional child names, and none
    /// of them changes a layout:
    ///
    /// - a `ConvBlockRes` is a six-element `Sequential`, so its convolutions are
    ///   `conv.0`/`conv.3` and its norms `conv.1`/`conv.4`. These patterns are
    ///   anchored on the parameter name so that the *outer* `conv.<i>` — the
    ///   `ModuleList` index, which Burn's `Vec` numbers identically — is left
    ///   alone.
    /// - a decoder level's transposed convolution and norm are `conv1.0` and
    ///   `conv1.1`.
    /// - the head is `fc.0.gru.*` and `fc.1.*`.
    ///
    /// Expected on `rmvpe.pt`: **623 applied, 0 missing, 118 unused**, the
    /// unused being one `num_batches_tracked` per norm — a training counter with
    /// no inference role, deliberately not a field.
    pub fn load_pytorch(
        &mut self,
        path: impl AsRef<Path>,
    ) -> Result<burn_store::ApplyResult, Box<dyn Error>> {
        let remaps = [
            (r"\.conv\.0\.(weight|bias)$", ".conv1.$1"),
            (
                r"\.conv\.1\.(weight|bias|running_mean|running_var)$",
                ".bn1.$1",
            ),
            (r"\.conv\.3\.(weight|bias)$", ".conv2.$1"),
            (
                r"\.conv\.4\.(weight|bias|running_mean|running_var)$",
                ".bn2.$1",
            ),
            (r"\.conv1\.0\.", ".conv1."),
            (r"\.conv1\.1\.", ".bn."),
            (r"^fc\.0\.gru\.", "gru."),
            (r"^fc\.1\.", "fc."),
        ];
        burn_kit::store::load_pytorch_into::<B, _>(self, path.as_ref(), None, &remaps)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::Distribution;

    type B = burn_ndarray::NdArray;

    /// A small enough network to run on CPU in a test, keeping every structural
    /// property that matters: five halvings of both axes, a bottleneck, and a
    /// decoder that has to line its skips up exactly.
    fn tiny() -> RmvpeConfig {
        RmvpeConfig {
            n_blocks: 1,
            inter_layers: 1,
            en_out_channels: 2,
            ..RmvpeConfig::default()
        }
    }

    /// The U-net's shape arithmetic is the thing that fails loudly, so pin it —
    /// including the two cases the padding exists for.
    #[test]
    fn any_frame_count_survives_the_round_trip() {
        let cfg = tiny();
        let device = Default::default();
        let model = Rmvpe::<B>::new(&cfg, &device);
        // Exactly aligned, one short of a multiple, and one over.
        for frames in [32usize, 31, 33] {
            let mel = Tensor::<B, 3>::random(
                [1, cfg.n_mels, frames],
                Distribution::Normal(0.0, 1.0),
                &device,
            );
            let out = model.forward(mel);
            assert_eq!(out.dims(), [1, frames, cfg.n_class]);
            let v: Vec<f32> = out.into_data().to_vec().unwrap();
            // Asserted separately because Burn's `assert_approx_eq` compares NaN
            // to NaN without complaint, so a NaN would otherwise pass silently.
            assert!(v.iter().all(|x| x.is_finite()), "salience must be finite");
            assert!(
                v.iter().all(|x| (0.0..=1.0).contains(x)),
                "a sigmoid cannot leave (0, 1)"
            );
        }
    }

    /// The internal padding must be indistinguishable from the caller having
    /// supplied those zero frames — otherwise the alignment step is doing
    /// something of its own, and a clip's pitch would depend on how close its
    /// length happened to be to a multiple of 32.
    ///
    /// Note what this deliberately does *not* claim: that padding leaves the
    /// earlier frames alone. It cannot, and neither can upstream — the reverse
    /// half of the GRU carries the end of the sequence back to its start, so
    /// every frame's salience depends on every later frame.
    #[test]
    fn the_internal_padding_is_the_zeros_it_says_it_is() {
        let cfg = tiny();
        let device = Default::default();
        let model = Rmvpe::<B>::new(&cfg, &device);
        let short =
            Tensor::<B, 3>::random([1, cfg.n_mels, 50], Distribution::Normal(0.0, 1.0), &device);
        let explicit = Tensor::cat(
            vec![short.clone(), Tensor::zeros([1, cfg.n_mels, 14], &device)],
            2,
        );

        let padded_here: Vec<f32> = model.forward(short).into_data().to_vec().unwrap();
        let padded_by_hand: Vec<f32> = model.forward(explicit).into_data().to_vec().unwrap();

        let diff = padded_here
            .iter()
            .zip(&padded_by_hand)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            diff < 1e-5,
            "alignment padding changed the answer by {diff}"
        );
    }
}
