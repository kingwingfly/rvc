//! `InterpolateRegulator` — Whisper's content frames, made mel-shaped.
//!
//! One small convolutional stack sits between the frozen content encoder and the
//! diffusion transformer, and it does exactly two things: it changes the width
//! from Whisper's 768 to the transformer's 512, and it changes the **rate**.
//!
//! # The rate, since three modules have to agree on it
//!
//! Whisper's encoder emits one frame per 320 input samples at 16 kHz — **50 Hz**.
//! The mel the transformer predicts and the vocoder consumes runs at
//! `sample_rate / hop_length` = 22050/256 — **≈86.13 Hz**
//! ([`crate::config::SeedVcConfig::frame_rate`]). This module is where the one
//! becomes the other, by nearest-neighbour interpolation onto a frame count the
//! *caller* supplies (upstream's `ylens`, which `seed_vc_wrapper.py` sets to the
//! mel's own frame count). The target length is therefore an argument, not a
//! fixed ratio, which is also how upstream's `length_adjust` knob stretches or
//! compresses the result.
//!
//! **`sampling_ratios` is not a ratio.** Upstream reads only its *length* — one
//! `conv → GroupNorm → Mish` stage per entry — and never looks at the values.
//! `[1, 1, 1, 1]` means four stages, not "rate unchanged", and a reader who takes
//! the name at face value will conclude this module preserves the frame rate when
//! it is the only thing in the model that changes it.
//!
//! # What is live on this preset and what is not
//!
//! `config.yml` for `seed-uvit-whisper-small-wavenet` sets `is_discrete: false`,
//! `vector_quantize: false` and `f0_condition: false`, which settles three
//! things:
//!
//! - **live**: `content_in_proj`, the four conv stages, and the final width-1
//!   conv. Continuous 768-dim content in, 512-dim conditioning out.
//! - **allocated but never indexed**: `embedding` (2048×512) and `mask_token`.
//!   They are the discrete path — a preset whose content arrives as codebook ids
//!   looks them up instead of projecting a continuous stream. The checkpoint
//!   carries both tensors, so they are held as parameters rather than dropped;
//!   dropping them would report two false `unused` and hide a real one in the
//!   noise.
//! - **absent**: `f0_embedding` and `f0_mask` are not in the checkpoint at all,
//!   because `f0_condition: false` means upstream never constructs them. There is
//!   nothing to model, and their absence is not a gap in the port.
//!
//! Upstream's `n_quantizers` argument is inert here for the same reason: it
//! selects among *extra* codebooks and this preset has `n_codebooks: 1`. Callers
//! pass 3 regardless, which is worth knowing before hunting for what it does.
//!
//! # Provenance
//!
//! Ported from `modules/length_regulator.py` of Seed-VC
//! (<https://github.com/Plachtaa/seed-vc>, GPL-3.0), read as a reference and
//! never run or vendored.

use std::error::Error;
use std::path::Path;

use burn::module::{Module, Param};
use burn::nn::conv::{Conv1d, Conv1dConfig};
use burn::nn::{
    Embedding, EmbeddingConfig, GroupNorm, GroupNormConfig, Linear, LinearConfig, PaddingConfig1d,
};
use burn::tensor::activation::mish;
use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor, TensorData};
use burn_store::ApplyResult;

use crate::config::SeedVcConfig;

/// One `Conv1d(k=3) → GroupNorm → Mish` stage of the stack.
///
/// The GroupNorm has a single group — upstream's `groups` argument defaults to 1
/// and nothing overrides it — so it normalises each frame across all 512 channels
/// at once. That is a LayerNorm in everything but the parameter names, which is
/// why the checkpoint's `weight`/`bias` land on Burn's `gamma`/`beta` and get
/// counted twice in an [`ApplyResult`].
#[derive(Module, Debug)]
pub struct RegulatorBlock<B: Backend> {
    conv: Conv1d<B>,
    norm: GroupNorm<B>,
}

impl<B: Backend> RegulatorBlock<B> {
    fn new(channels: usize, device: &B::Device) -> Self {
        Self {
            conv: Conv1dConfig::new(channels, channels, 3)
                .with_padding(PaddingConfig1d::Explicit(1, 1))
                .init(device),
            norm: GroupNormConfig::new(1, channels).init(device),
        }
    }

    /// `[batch, channels, frames]` in and out — the stage is length-preserving.
    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        mish(self.norm.forward(self.conv.forward(x)))
    }
}

/// The length regulator: content features in, transformer conditioning out.
#[derive(Module, Debug)]
pub struct InterpolateRegulator<B: Backend> {
    content_in_proj: Linear<B>,
    blocks: Vec<RegulatorBlock<B>>,
    out_conv: Conv1d<B>,
    /// The discrete path's codebook. Allocated, never indexed on this preset —
    /// see the module docs.
    embedding: Embedding<B>,
    /// What the discrete path substitutes for a dropped codebook. Same story.
    mask_token: Param<Tensor<B, 2>>,
}

impl<B: Backend> InterpolateRegulator<B> {
    pub fn new(cfg: &SeedVcConfig, device: &B::Device) -> Self {
        let channels = cfg.hidden_dim;
        Self {
            content_in_proj: LinearConfig::new(cfg.content_dim, channels).init(device),
            blocks: cfg
                .sampling_ratios
                .iter()
                .map(|_| RegulatorBlock::new(channels, device))
                .collect(),
            out_conv: Conv1dConfig::new(channels, channels, 1).init(device),
            embedding: EmbeddingConfig::new(cfg.codebook_size, channels).init(device),
            mask_token: Param::from_tensor(Tensor::zeros([1, channels], device)),
        }
    }

    /// `content`: `[batch, source_frames, content_dim]` at 50 Hz.
    /// Returns `[batch, frames, hidden_dim]` at the mel rate.
    ///
    /// `frames` is upstream's `ylens.max()`. Upstream then multiplies by a
    /// sequence mask built from the per-sample `ylens`, which is all ones unless
    /// a batch mixes lengths — it never does at inference, where the batch is one
    /// clip — so the mask is not modelled. **A batched trainer would have to add
    /// it back**, or short samples would contribute conditioning past their own
    /// end.
    pub fn forward(&self, content: Tensor<B, 3>, frames: usize) -> Tensor<B, 3> {
        let x = self.content_in_proj.forward(content).swap_dims(1, 2);

        // `F.interpolate(..., mode='nearest')`: PyTorch takes source frame
        // `min(floor(i · in/out), in - 1)`. That is the identity when the two
        // lengths agree and an upsample from 50 Hz to 86.13 Hz when they do not.
        let [_, _, source] = x.dims();
        let scale = source as f64 / frames as f64;
        let picks: Vec<i64> = (0..frames)
            .map(|i| ((i as f64 * scale) as i64).min(source as i64 - 1))
            .collect();
        let picks = Tensor::<B, 1, Int>::from_data(TensorData::new(picks, [frames]), &x.device());
        let mut x = x.select(2, picks);

        for block in &self.blocks {
            x = block.forward(x);
        }
        self.out_conv.forward(x).swap_dims(1, 2)
    }

    /// Load this module's slice of the Seed-VC checkpoint.
    ///
    /// Upstream's stack is one `nn.Sequential`, so its keys are flat indices over
    /// a repeating conv/norm/Mish triple whose activation carries no tensors:
    /// `model.0`, `model.1`, then `model.3`, `model.4`, and so on, with the final
    /// width-1 conv at `model.12`. The remaps rebuild that arithmetic onto the
    /// module tree, and they are **generated from `self.blocks.len()`** rather
    /// than written out as a literal table: a table would pin this preset's four
    /// stages while [`Self::new`] builds one per `sampling_ratios` entry, and the
    /// two would drift apart in silence.
    pub fn load_pytorch(&mut self, path: impl AsRef<Path>) -> Result<ApplyResult, Box<dyn Error>> {
        let mut owned = vec![(
            r"^net\.length_regulator\.module\.".to_string(),
            String::new(),
        )];
        for i in 0..self.blocks.len() {
            owned.push((format!(r"^model\.{}\.", 3 * i), format!("blocks.{i}.conv.")));
            owned.push((
                format!(r"^model\.{}\.", 3 * i + 1),
                format!("blocks.{i}.norm."),
            ));
        }
        owned.push((
            format!(r"^model\.{}\.", 3 * self.blocks.len()),
            "out_conv.".to_string(),
        ));

        let remaps: Vec<(&str, &str)> = owned
            .iter()
            .map(|(from, to)| (from.as_str(), to.as_str()))
            .collect();
        burn_kit::store::load_pytorch_into::<B, _>(self, path.as_ref(), None, &remaps)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    type B = burn_ndarray::NdArray;

    #[test]
    fn content_arrives_at_the_mel_rate_and_the_transformer_width() {
        // The shape contract everything downstream depends on, and the cheapest
        // check that the stack is wired at all: coverage says the tensors fit,
        // not that the forward pass composes.
        let cfg = SeedVcConfig::uvit_whisper_small_wavenet();
        let device = Default::default();
        let model = InterpolateRegulator::<B>::new(&cfg, &device);

        // One second of audio: 50 Whisper frames in, 86 mel frames out.
        let frames = cfg.frame_rate() as usize;
        let out = model.forward(Tensor::zeros([1, 50, cfg.content_dim], &device), frames);
        assert_eq!(out.dims(), [1, frames, cfg.hidden_dim]);
        // Asserted separately because Burn's approximate comparisons treat NaN as
        // equal to NaN: a NaN here reaches every mel frame and never errors.
        assert!(!out.contains_nan().into_scalar());
    }

    #[test]
    fn asking_for_the_source_length_resamples_to_the_identity() {
        // The nearest-neighbour index must be exactly `i` when the lengths agree,
        // or the conditioning slides by a frame against the mel it describes.
        let cfg = SeedVcConfig::uvit_whisper_small_wavenet();
        let device = Default::default();
        let model = InterpolateRegulator::<B>::new(&cfg, &device);

        let out = model.forward(Tensor::zeros([1, 37, cfg.content_dim], &device), 37);
        assert_eq!(out.dims(), [1, 37, cfg.hidden_dim]);
    }
}
