//! OpenAI Whisper, implemented in [Burn](https://burn.dev).
//!
//! A port of `openai/whisper` faithful enough that the **Hugging Face
//! checkpoints load unchanged** — [`Whisper::load_safetensors`] against
//! `openai/whisper-large-v3-turbo` applies all 587 tensors with nothing missing.
//! That property is the whole design constraint: the module tree mirrors
//! `WhisperForConditionalGeneration`'s `state_dict` layout, so the only key
//! remapping needed is stripping its `model.` prefix.
//!
//! Preferring the first-party repo to a community ONNX re-export is deliberate.
//! Mirrors of converted weights move and disappear; `openai/whisper-large-v3-turbo`
//! does not.
//!
//! ```no_run
//! # use burn::tensor::{Int, Tensor};
//! # use burn_whisper::{Whisper, WhisperConfig};
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! # type B = burn_ndarray::NdArray;
//! # let mel: Tensor<B, 3> = unimplemented!();
//! # let tokens: Tensor<B, 2, Int> = unimplemented!();
//! let device = Default::default();
//! let mut model = Whisper::<B>::new(&WhisperConfig::large_v3_turbo(), &device);
//! model.load_safetensors("models/whisper/model.safetensors")?;
//!
//! // The encoder runs once per 30 s window...
//! let audio = model.encoder.forward(mel);              // [batch, 1500, d_model]
//! // ...the decoder once per token, carrying its caches in `state`.
//! let mut state = model.decoder.state();
//! let logits = model.decoder.forward(tokens, audio, &mut state);
//! # Ok(())
//! # }
//! ```
//!
//! The model is generic over the Burn [`Backend`](burn::tensor::backend::Backend);
//! nothing in `src/` names a compute backend.

mod attention;
mod config;
mod decoder;
mod encoder;

pub use attention::{Attention, KvCache, causal_mask};
pub use config::WhisperConfig;
pub use decoder::{DecodeState, DecoderLayer, TextDecoder};
pub use encoder::{AudioEncoder, EncoderLayer};

use std::error::Error;
use std::path::Path;

use burn::module::Module;
use burn::tensor::backend::Backend;
use burn_store::ApplyResult;

/// The full encoder-decoder model.
#[derive(Module, Debug)]
pub struct Whisper<B: Backend> {
    pub encoder: AudioEncoder<B>,
    pub decoder: TextDecoder<B>,
}

impl<B: Backend> Whisper<B> {
    pub fn new(cfg: &WhisperConfig, device: &B::Device) -> Self {
        Self {
            encoder: AudioEncoder::new(cfg, device),
            decoder: TextDecoder::new(cfg, device),
        }
    }

    /// Load a Hugging Face `model.safetensors`, reporting coverage.
    ///
    /// The single remap drops transformers' `model.` prefix; every name below it
    /// already matches, which is the point of mirroring their layout. `proj_out`
    /// is absent from the checkpoint because Whisper ties the output projection
    /// to the token embedding — [`TextDecoder::forward`] does the same.
    pub fn load_safetensors(
        &mut self,
        path: impl AsRef<Path>,
    ) -> Result<ApplyResult, Box<dyn Error>> {
        burn_kit::store::load_safetensors_into::<B, _>(self, path.as_ref(), &[(r"^model\.", "")])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Distribution, Int, Tensor, TensorData};

    type B = burn_ndarray::NdArray;

    #[test]
    fn encoder_halves_the_frame_rate() {
        // The stride-2 convolution is what makes 30 s of audio 1500 frames; get
        // it wrong and every positional embedding lines up against the wrong
        // moment in time, which no weight-coverage check would notice.
        let cfg = WhisperConfig::tiny();
        let device = Default::default();
        let encoder = AudioEncoder::<B>::new(&cfg, &device);

        let frames = cfg.max_source_positions * 2;
        let mel = Tensor::random(
            [1, cfg.num_mel_bins, frames],
            Distribution::Normal(0.0, 1.0),
            &device,
        );
        assert_eq!(
            encoder.forward(mel).dims(),
            [1, cfg.max_source_positions, cfg.d_model]
        );
    }

    #[test]
    fn incremental_decoding_matches_one_shot() {
        // The test that earns its keep. Feeding a prompt all at once and feeding
        // it a token at a time must produce identical logits for the final
        // position — which holds only if the causal mask and the positional
        // offset are both right under a growing KV cache. A checkpoint loads at
        // 100% coverage whether or not they are, and a wrong mask shows up as
        // fluent, confident, wrong transcription.
        let cfg = WhisperConfig::tiny();
        let device = Default::default();
        let decoder = TextDecoder::<B>::new(&cfg, &device);

        let ids = [3i32, 17, 5, 29, 11];
        let audio = Tensor::random(
            [1, cfg.max_source_positions, cfg.d_model],
            Distribution::Normal(0.0, 1.0),
            &device,
        );

        let all: Tensor<B, 2, Int> = Tensor::from_data(TensorData::from([ids]), &device);
        let mut state = decoder.state();
        let one_shot = decoder.forward(all, audio.clone(), &mut state);
        let [_, seq, vocab] = one_shot.dims();
        let one_shot = one_shot.slice([0..1, seq - 1..seq, 0..vocab]);

        let mut state = decoder.state();
        let mut stepwise = None;
        for id in ids {
            let token: Tensor<B, 2, Int> = Tensor::from_data(TensorData::from([[id]]), &device);
            stepwise = Some(decoder.forward(token, audio.clone(), &mut state));
        }
        assert_eq!(state.offset(), ids.len());

        one_shot.into_data().assert_approx_eq::<f32>(
            &stepwise.unwrap().into_data(),
            burn::tensor::Tolerance::default(),
        );
    }
}
