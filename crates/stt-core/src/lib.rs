//! Speech recognition for the `voice` toolkit — Whisper, end to end.
//!
//! Audio in as mono `f32` at 16 kHz, [`Segment`]s out. The pipeline is
//! segmentation → log-mel → encoder → greedy decode, and the first step matters
//! more than its size suggests: Whisper is a language model conditioned on
//! audio, and handed a long quiet stretch it will invent fluent sentences to
//! fill it. Feeding it speech-shaped pieces is the standard defence, and the
//! toolkit already has a slicer tuned not to cut soft or breathy passages.
//!
//! Two runtimes sit behind one [`Transcriber`]: the native Burn port, and ONNX
//! Runtime. They are picked by constructor and are otherwise indistinguishable —
//! the decode loop deals in token ids and `f32` logits, which is the most either
//! has to agree on.
//!
//! Nothing here needs the whole recording. Each span is encoded and decoded from
//! scratch — both engines reset their key/value cache and encoder output on
//! [`Transcriber::segment`] — so a caller holding a [`Slicer`] can transcribe
//! and print each clip the moment its audio has arrived. [`Transcriber::transcribe`]
//! is that same loop over a slicer fed in one go, so the batch and streaming
//! paths cannot drift apart.
//!
//! ```no_run
//! # use stt_core::{Transcriber, TranscribeOptions};
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! # let audio: Vec<f32> = vec![];
//! # #[cfg(feature = "tch")] {
//! let mut stt = Transcriber::libtorch("models/whisper".as_ref(), Default::default())?;
//! for segment in stt.transcribe(&audio, &TranscribeOptions::default())? {
//!     println!("[{:.2} -> {:.2}] {}", segment.start, segment.end, segment.text);
//! }
//! # }
//! # Ok(())
//! # }
//! ```

mod burn_engine;
mod decode;
mod engine;
mod error;
mod mel;
#[cfg(feature = "onnx")]
mod onnx_engine;
mod tokenizer;

pub use decode::DecodeOptions;
pub use error::{Result, SttError};
pub use mel::{HOP, SAMPLE_RATE, WINDOW_FRAMES, WINDOW_SAMPLES, WINDOW_SECONDS};
pub use tokenizer::{Tokens, Vocabulary};

use std::path::Path;

use audio_kit::slice::{SliceOptions, Slicer};
use burn_whisper::WhisperConfig;
use engine::Engine;

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
    /// Where to cut. Defaults match the corpus slicer's: energy is used only to
    /// find long silent gaps, never to gate quiet-but-present sound.
    pub slice: SliceOptions,
    pub decode: DecodeOptions,
}

impl Default for TranscribeOptions {
    fn default() -> Self {
        Self {
            slice: SliceOptions {
                // A segment must fit one encoder window, so unlike the training
                // slicer this one has a hard ceiling rather than 0 ("never split").
                max_clip: WINDOW_SECONDS,
                ..SliceOptions::default()
            },
            decode: DecodeOptions::default(),
        }
    }
}

/// A loaded Whisper model, ready to transcribe.
///
/// Not generic over a compute backend — the backend lives behind the engine
/// trait instead, so picking one is a constructor call and callers never name a
/// Burn type. `&mut self` throughout, because a decode carries key/value caches.
pub struct Transcriber {
    engine: Box<dyn Engine>,
    mel: mel::LogMel,
    tokens: Tokens,
    vocab: Vocabulary,
}

impl Transcriber {
    /// Pair a loaded engine with the JSON that shipped beside the weights.
    fn assemble(engine: Box<dyn Engine>, dir: &Path) -> Result<Self> {
        Ok(Self {
            mel: mel::LogMel::new(engine.mel_bins()),
            engine,
            tokens: Tokens::load(&dir.join("generation_config.json"))?,
            vocab: Vocabulary::load(&dir.join("tokenizer.json"))?,
        })
    }

    /// Native Burn on the CubeCL/CUDA backend.
    #[cfg(feature = "cuda")]
    pub fn cuda(dir: &Path, device: burn_kit::DeviceSpec) -> Result<Self> {
        let device = burn_kit::cuda_device(device)?;
        let cfg = read_config(&dir.join("config.json"))?;
        let engine = burn_kit::guard_init("cuda", || {
            burn_engine::BurnEngine::<burn::backend::Cuda>::load(
                &cfg,
                &dir.join("model.safetensors"),
                &device,
            )
        })??;
        Self::assemble(Box::new(engine), dir)
    }

    /// Native Burn on LibTorch — CUDA, MPS, Vulkan or CPU.
    #[cfg(feature = "tch")]
    pub fn libtorch(dir: &Path, device: burn_kit::DeviceSpec) -> Result<Self> {
        let device = burn_kit::libtorch_device(device)?;
        let cfg = read_config(&dir.join("config.json"))?;
        let engine = burn_kit::guard_init("tch", || {
            burn_engine::BurnEngine::<burn::backend::LibTorch<f32>>::load(
                &cfg,
                &dir.join("model.safetensors"),
                &device,
            )
        })??;
        Self::assemble(Box::new(engine), dir)
    }

    /// Native Burn on WebGPU.
    #[cfg(feature = "wgpu")]
    pub fn wgpu(dir: &Path, device: burn_kit::DeviceSpec) -> Result<Self> {
        let device = burn_kit::wgpu_device(device)?;
        let cfg = read_config(&dir.join("config.json"))?;
        let engine = burn_kit::guard_init("wgpu", || {
            burn_engine::BurnEngine::<burn::backend::Wgpu>::load(
                &cfg,
                &dir.join("model.safetensors"),
                &device,
            )
        })??;
        Self::assemble(Box::new(engine), dir)
    }

    /// ONNX Runtime, from an `optimum`-style two-graph export.
    ///
    /// Takes no device: `ort` picks its own execution provider — CUDA if the
    /// runtime was built with it, else CPU.
    #[cfg(feature = "onnx")]
    pub fn onnx(dir: &Path) -> Result<Self> {
        let cfg = read_config(&dir.join("config.json"))?;
        let engine = onnx_engine::OnnxEngine::load(
            dir,
            cfg.num_mel_bins,
            cfg.decoder_layers,
            cfg.decoder_attention_heads,
            cfg.d_model,
        )?;
        Self::assemble(Box::new(engine), dir)
    }

    /// Transcribe mono 16 kHz audio.
    ///
    /// Held here for callers that already have the whole recording; a filter
    /// should drive [`Slicer`] and [`Transcriber::segment`] itself, which is
    /// what this is.
    pub fn transcribe(&mut self, audio: &[f32], opts: &TranscribeOptions) -> Result<Vec<Segment>> {
        let mut slicer = Slicer::new(SAMPLE_RATE, &opts.slice);
        let mut out = Vec::new();
        let clips = slicer.push(audio).into_iter().chain(slicer.finish());
        for clip in clips {
            let start = clip.start as f32 / SAMPLE_RATE as f32;
            out.extend(self.segment(&clip.samples, start, &opts.decode)?);
        }
        Ok(out)
    }

    /// Transcribe one clip already cut out of the input.
    ///
    /// `start` is the clip's offset in seconds from the beginning of the
    /// recording, so the timings come back absolute however the audio was cut
    /// up. `None` when the model produced nothing but whitespace, which is what
    /// a clip of breath or room tone gives.
    pub fn segment(
        &mut self,
        clip: &[f32],
        start: f32,
        opts: &DecodeOptions,
    ) -> Result<Option<Segment>> {
        let segment = self.window(clip, opts)?;
        if segment.text.trim().is_empty() {
            return Ok(None);
        }
        Ok(Some(Segment {
            start,
            end: start + clip.len() as f32 / SAMPLE_RATE as f32,
            ..segment
        }))
    }

    /// Transcribe one span, which must fit a single 30 s encoder window.
    fn window(&mut self, audio: &[f32], opts: &DecodeOptions) -> Result<Segment> {
        let (frames, data) = self.mel.compute(audio);
        self.engine.encode(&data, frames)?;

        let decoded = decode::greedy(self.engine.as_mut(), &self.tokens, opts)?;
        if decoded.truncated {
            tracing::warn!(
                "a segment hit the {}-token cap and was cut off — raise the token \
                 cap, or split it with a shorter maximum clip length",
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
