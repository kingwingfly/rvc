//! ContentVec — RVC's content encoder, on top of [`burn_hubert::Hubert`].
//!
//! ContentVec is a HuBERT checkpoint, so **this module owns no network of its
//! own**: the architecture lives in `burn-hubert`, shared with GPT-SoVITS's
//! cnhubert, and what is RVC's alone is the readout and the published config.
//! Until now those frames could only come from ONNX Runtime; this is an
//! **addition**, so `vec-768-layer-12.onnx` stays a supported deployment target
//! and gains company rather than being replaced.
//!
//! # What the published config settles
//!
//! The weights are `lj1995/VoiceConversionWebUI`'s `hubert_base/` — the same
//! first-party repo RMVPE and the pretrained bases come from — and its
//! `config.json` is [`HubertConfig::chinese_base`] field for field: 12 layers,
//! 768 wide, 3072 intermediate, seven convolutions of kernel
//! `[10,3,3,3,3,2,2]` over strides `[5,2,2,2,2,2,2]`, and a 128-wide positional
//! convolution in 16 groups. Three of its entries decide things that would
//! otherwise be guesses, and all three agree with what `burn-hubert` already
//! does:
//!
//! - `do_stable_layer_norm: false` — **post-norm**, which is the variant the
//!   port implements. The pre-norm one loads the same weights and computes
//!   something else.
//! - `feat_extract_norm: "group"` — group normalisation on convolution 0 alone.
//! - `preprocessor_config.json`'s `do_normalize: false` — **the waveform is fed
//!   raw**. There is deliberately no normalisation step here; adding the
//!   zero-mean/unit-variance one that other wav2vec-family checkpoints want
//!   would pass weight coverage and shift every feature.
//!
//! # Which layer, and the tensors that go unused
//!
//! **RVC v2 reads the final (12th) encoder layer directly** — upstream says so
//! in `infer/hubert.py`, where v1 instead takes layer 9 and pushes it through
//! `final_proj`. So this wraps [`Hubert::forward`], the last-layer shorthand,
//! and the checkpoint's `final_proj.weight`/`final_proj.bias` are **expected to
//! report as unused**: they are v1's readout, present in the file and read by
//! nothing here. There is no field for them, because a field nobody uses is
//! indistinguishable from one somebody forgot to wire up.
//!
//! `masked_spec_embed` is unused for the same kind of reason — it is
//! SpecAugment's mask token, which exists only while the SSL model itself is
//! trained — and every LayerNorm's `weight`/`bias` is *reported* unused while
//! having applied, because `burn-store` consumes them under Burn's
//! `gamma`/`beta` names.

use std::error::Error;
use std::path::Path;

use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, TensorData};
use burn_hubert::{Hubert, HubertConfig};
use burn_kit::ApplyResult;

/// The one filename a HuBERT checkpoint fixes across a Hub snapshot, a git
/// clone and a hand-made copy alike, so [`ContentVec::load`] can accept the
/// directory `hub_kit::fetch_contentvec` returns as readily as the file itself.
const WEIGHTS: &str = "pytorch_model.bin";

/// RVC's content encoder: mono 16 kHz audio in, 50 Hz `[T][768]` frames out.
pub struct ContentVec<B: Backend> {
    model: Hubert<B>,
    device: B::Device,
}

impl<B: Backend> ContentVec<B> {
    /// Load ContentVec from a `hubert_base/` directory or from the weights file
    /// inside one.
    ///
    /// A directory is resolved by joining [`WEIGHTS`]. The sibling
    /// `config.json` is deliberately **not** parsed: its contents are
    /// [`HubertConfig::chinese_base`] and are pinned here as a constant, so a
    /// checkpoint whose shape disagrees fails as a shape mismatch on load
    /// rather than silently building a different network. It is fetched
    /// alongside the weights all the same, so that a cached directory is
    /// identifiable as this checkpoint by eye.
    pub fn load(path: &Path, device: &B::Device) -> Result<(Self, ApplyResult), Box<dyn Error>> {
        let weights = if path.is_dir() {
            path.join(WEIGHTS)
        } else {
            path.to_path_buf()
        };

        let mut model = Hubert::new(&HubertConfig::chinese_base(), device);
        let applied = model.load_pytorch(&weights)?;
        Ok((
            Self {
                model,
                device: device.clone(),
            },
            applied,
        ))
    }

    /// Content features for a mono **16 kHz** buffer: time-major `[T][768]` at
    /// 50 Hz, one frame per 320 samples.
    ///
    /// `&self` because nothing is carried between clips — the ONNX encoder's
    /// `&mut self` is `ort::Session::run`'s requirement, not the model's.
    pub fn extract(&self, wav16k: &[f32]) -> Vec<Vec<f32>> {
        let wav = Tensor::<B, 2>::from_data(
            TensorData::new(wav16k.to_vec(), [1, wav16k.len()]),
            &self.device,
        );
        let frames = self.model.forward(wav);
        let [_, time, hidden] = frames.dims();

        let flat: Vec<f32> = frames
            .into_data()
            .into_vec()
            .expect("HuBERT features are f32");
        let mut out = Vec::with_capacity(time);
        for frame in flat.chunks_exact(hidden) {
            out.push(frame.to_vec());
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    type B = burn_ndarray::NdArray;

    /// The shape contract `rvc-core` codes against: time-major, 768 wide, and at
    /// the same 50 Hz rate the ONNX encoder produces, so the ×2 upsample onto
    /// the 100 Hz F0 grid lands identically whichever runtime ran.
    #[test]
    fn extract_is_time_major_at_fifty_hertz() {
        let device = Default::default();
        let encoder = ContentVec::<B> {
            model: Hubert::new(&HubertConfig::chinese_base(), &device),
            device,
        };

        let width = HubertConfig::chinese_base().hidden_size;
        let frames = encoder.extract(&vec![0.0; 16_000]);
        assert!(frames.iter().all(|f| f.len() == width));
        // Unpadded convolutions floor the count to a little under 50 for 1 s.
        assert!((45..=50).contains(&frames.len()), "{} frames", frames.len());
    }
}
