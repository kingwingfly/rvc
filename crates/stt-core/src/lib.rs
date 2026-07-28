//! Speech recognition for the `voice` toolkit — Whisper, end to end.
//!
//! Audio in as mono `f32` at 16 kHz, [`Segment`]s out. The pipeline is
//! segmentation → log-mel → encoder → greedy decode, and the first step matters
//! more than its size suggests: Whisper is a language model conditioned on
//! audio, and handed a long quiet stretch it will invent fluent sentences to
//! fill it. Feeding it speech-shaped pieces is the standard defence, and the
//! toolkit already has a slicer tuned not to cut soft or breathy passages.
//!
//! ```no_run
//! # use stt_core::{Transcriber, TranscribeOptions};
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! # type B = burn_ndarray::NdArray;
//! # let audio: Vec<f32> = vec![];
//! let mut stt = Transcriber::<B>::load("models/whisper".as_ref(), &Default::default())?;
//! for segment in stt.transcribe(&audio, &TranscribeOptions::default())? {
//!     println!("[{:.2} -> {:.2}] {}", segment.start, segment.end, segment.text);
//! }
//! # Ok(())
//! # }
//! ```

mod decode;
mod error;
mod mel;
mod tokenizer;

pub use decode::DecodeOptions;
pub use error::{Result, SttError};
pub use mel::{HOP, SAMPLE_RATE, WINDOW_FRAMES, WINDOW_SAMPLES};
pub use tokenizer::{Tokens, Vocabulary};

use std::path::Path;

use audio_kit::slice::{SliceOptions, slice};
use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, TensorData};
use burn_whisper::{Whisper, WhisperConfig};

/// One transcribed stretch of speech.
#[derive(Debug, Clone)]
pub struct Segment {
    /// Seconds from the start of the input.
    pub start: f32,
    pub end: f32,
    pub text: String,
    /// ISO code the model used, detected or forced.
    pub language: String,
}

/// How to cut the input up and what to ask the model for.
#[derive(Debug, Clone)]
pub struct TranscribeOptions {
    /// Where to cut. Defaults match `voice rvc preprocess`: energy is used only
    /// to find long silent gaps, never to gate quiet-but-present sound.
    pub slice: SliceOptions,
    pub decode: DecodeOptions,
}

impl Default for TranscribeOptions {
    fn default() -> Self {
        Self {
            slice: SliceOptions {
                // A segment must fit one encoder window, so unlike the training
                // slicer this one has a hard ceiling rather than 0 ("never split").
                max_clip: (WINDOW_SAMPLES / SAMPLE_RATE as usize) as f32,
                ..SliceOptions::default()
            },
            decode: DecodeOptions::default(),
        }
    }
}

/// A loaded Whisper model, ready to transcribe.
pub struct Transcriber<B: Backend> {
    model: Whisper<B>,
    mel: mel::LogMel,
    tokens: Tokens,
    vocab: Vocabulary,
    device: B::Device,
}

impl<B: Backend> Transcriber<B> {
    /// Load from a directory holding a Hugging Face Whisper repo:
    /// `model.safetensors`, `config.json`, `generation_config.json` and
    /// `tokenizer.json`.
    pub fn load(dir: &Path, device: &B::Device) -> Result<Self> {
        let cfg = read_config(&dir.join("config.json"))?;
        let mut model = Whisper::<B>::new(&cfg, device);
        let applied = model
            .load_safetensors(dir.join("model.safetensors"))
            .map_err(|e| SttError::Weights(e.to_string()))?;
        if !applied.missing.is_empty() {
            return Err(SttError::Weights(format!(
                "{} parameters had no tensor in the checkpoint (first: {})",
                applied.missing.len(),
                applied.missing[0].0
            )));
        }

        Ok(Self {
            mel: mel::LogMel::new(cfg.num_mel_bins),
            model,
            tokens: Tokens::load(&dir.join("generation_config.json"))?,
            vocab: Vocabulary::load(&dir.join("tokenizer.json"))?,
            device: device.clone(),
        })
    }

    /// Transcribe mono 16 kHz audio.
    pub fn transcribe(&self, audio: &[f32], opts: &TranscribeOptions) -> Result<Vec<Segment>> {
        let spans = slice(audio, SAMPLE_RATE, &opts.slice);
        let mut out = Vec::with_capacity(spans.len());
        for (start, end) in spans {
            let text = self.window(&audio[start..end], &opts.decode)?;
            if text.text.trim().is_empty() {
                continue;
            }
            out.push(Segment {
                start: start as f32 / SAMPLE_RATE as f32,
                end: end as f32 / SAMPLE_RATE as f32,
                ..text
            });
        }
        Ok(out)
    }

    /// Transcribe one span, which must fit a single 30 s encoder window.
    fn window(&self, audio: &[f32], opts: &DecodeOptions) -> Result<Segment> {
        let (frames, data) = self.mel.compute(audio);
        let mel: Tensor<B, 3> = Tensor::from_data(
            TensorData::new(data, [1, self.mel.bins(), frames]),
            &self.device,
        );

        let encoded = self.model.encoder.forward(mel);
        let decoded = decode::greedy(&self.model, encoded, &self.tokens, opts, &self.device)?;
        if decoded.truncated {
            tracing::warn!(
                "a segment hit the {}-token cap and was cut off — raise --max-tokens, \
                 or split it with a shorter --max-clip",
                opts.max_tokens
            );
        }

        let language = self
            .tokens
            .languages
            .iter()
            .find(|(_, id)| **id == decoded.language)
            .map(|(code, _)| code.clone())
            .unwrap_or_default();

        Ok(Segment {
            start: 0.0,
            end: 0.0,
            text: self.vocab.decode(&decoded.tokens)?.trim().to_string(),
            language,
        })
    }
}

/// Read the model dimensions from a Hugging Face `config.json`.
///
/// Whisper's sizes are all in that file, so a new checkpoint is a download
/// rather than a code change — including the 80-vs-128 mel split, which the
/// front-end has to agree with.
fn read_config(path: &Path) -> Result<WhisperConfig> {
    let raw = std::fs::read_to_string(path)?;
    let json: serde_json::Value = serde_json::from_str(&raw)?;
    let get = |key: &str| -> Result<usize> {
        json.get(key)
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .ok_or_else(|| SttError::Config(format!("config.json has no `{key}`")))
    };
    Ok(WhisperConfig {
        num_mel_bins: get("num_mel_bins")?,
        max_source_positions: get("max_source_positions")?,
        max_target_positions: get("max_target_positions")?,
        d_model: get("d_model")?,
        encoder_attention_heads: get("encoder_attention_heads")?,
        encoder_layers: get("encoder_layers")?,
        encoder_ffn_dim: get("encoder_ffn_dim")?,
        decoder_attention_heads: get("decoder_attention_heads")?,
        decoder_layers: get("decoder_layers")?,
        decoder_ffn_dim: get("decoder_ffn_dim")?,
        vocab_size: get("vocab_size")?,
    })
}

/// A loaded transcriber with its compute backend erased, so `voice-cli` never
/// names a Burn type and `--backend` stays a run-time choice.
///
/// The same shape as `rvc_core::Generator`, and for the same reason.
pub trait Transcribe: Send {
    fn transcribe(&self, audio: &[f32], opts: &TranscribeOptions) -> Result<Vec<Segment>>;
}

impl<B: Backend> Transcribe for Transcriber<B> {
    fn transcribe(&self, audio: &[f32], opts: &TranscribeOptions) -> Result<Vec<Segment>> {
        Transcriber::transcribe(self, audio, opts)
    }
}

/// Load a transcriber onto the CubeCL/CUDA backend.
#[cfg(feature = "cuda")]
pub fn cuda_transcriber(dir: &Path, device: burn_kit::DeviceSpec) -> Result<Box<dyn Transcribe>> {
    let device = burn_kit::cuda_device(device)?;
    let model = burn_kit::guard_init("cuda", || {
        Transcriber::<burn::backend::Cuda>::load(dir, &device)
    })??;
    Ok(Box::new(model))
}

/// Load a transcriber onto LibTorch — CUDA, MPS, Vulkan or CPU.
#[cfg(feature = "tch")]
pub fn libtorch_transcriber(
    dir: &Path,
    device: burn_kit::DeviceSpec,
) -> Result<Box<dyn Transcribe>> {
    let device = burn_kit::libtorch_device(device)?;
    let model = burn_kit::guard_init("tch", || {
        Transcriber::<burn::backend::LibTorch<f32>>::load(dir, &device)
    })??;
    Ok(Box::new(model))
}

/// Load a transcriber onto WebGPU.
#[cfg(feature = "wgpu")]
pub fn wgpu_transcriber(dir: &Path, device: burn_kit::DeviceSpec) -> Result<Box<dyn Transcribe>> {
    let device = burn_kit::wgpu_device(device)?;
    let model = burn_kit::guard_init("wgpu", || {
        Transcriber::<burn::backend::Wgpu>::load(dir, &device)
    })??;
    Ok(Box::new(model))
}
