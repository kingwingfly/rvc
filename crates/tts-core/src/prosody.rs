//! Per-phoneme prosody features from a Chinese BERT.
//!
//! GPT-SoVITS conditions its text-to-semantic model on features from
//! `chinese-roberta-wwm-ext-large`, taken from the **third-from-last** hidden
//! layer — not the last. Each *character*'s feature is then repeated as many
//! times as that character produced phonemes, which is what `text-kit`'s
//! `word2ph` is for.
//!
//! # Why this one is ONNX and not a Burn port
//!
//! It is frozen: GPT-SoVITS trains `s1` and `s2` and never touches this. And it
//! is small work in the wrong place — one sentence is a few dozen tokens, so the
//! run is dominated by kernel-launch overhead rather than arithmetic, and a Burn
//! port would not be meaningfully faster. The autoregressive loop downstream,
//! which runs hundreds of steps per utterance, is where speed is worth chasing.
//!
//! [`ProsodyEncoder`] is a trait anyway, so a Burn implementation can be added
//! without reshaping anything — the same arrangement `stt-core` uses. Two things
//! would justify writing one: dropping ONNX Runtime from the TTS path entirely,
//! and reading `hfl/chinese-roberta-wwm-ext-large`'s own weights rather than
//! trusting a third-party conversion of them.

use std::path::Path;

use crate::error::{Result, TtsError};

/// Features for one utterance, one row per phoneme.
#[derive(Debug, Clone)]
pub struct ProsodyFeatures {
    /// Width of the encoder — 1024 for `chinese-roberta-wwm-ext-large`.
    pub hidden: usize,
    /// Rows, one per phoneme.
    pub phones: usize,
    /// Phone-major: `data[phone * hidden + i]`.
    pub data: Vec<f32>,
}

impl ProsodyFeatures {
    /// All-zero features, which is what the model is given for languages it has
    /// no prosody encoder for.
    ///
    /// Not a fallback dressed up as a value: GPT-SoVITS itself feeds zeros for
    /// English and Japanese, because the encoder is Chinese-only.
    pub fn zeros(hidden: usize, phones: usize) -> Self {
        Self {
            hidden,
            phones,
            data: vec![0.0; hidden * phones],
        }
    }
}

/// Something that turns text into per-phoneme prosody features.
pub trait ProsodyEncoder: Send {
    /// Encoder width, so callers can build zeros of the right shape.
    fn hidden(&self) -> usize;

    /// Features for `text`, expanded through `word2ph`.
    ///
    /// `word2ph[i]` is how many phonemes character `i` of `text` produced, so it
    /// must have exactly one entry per character.
    fn encode(&mut self, text: &str, word2ph: &[usize]) -> Result<ProsodyFeatures>;
}

/// The ONNX implementation.
#[cfg(feature = "onnx")]
pub struct OnnxProsody {
    /// Never dropped — see [`crate::onnx_engine`], which owns four of these for
    /// the same reason.
    session: std::mem::ManuallyDrop<ort::session::Session>,
    tokenizer: tokenizers::Tokenizer,
    hidden: usize,
}

#[cfg(feature = "onnx")]
impl OnnxProsody {
    /// Load from a directory holding `model.onnx` and `tokenizer.json`.
    ///
    /// The export must be one that emits the third-from-last hidden layer —
    /// a stock `optimum` export gives `last_hidden_state`, which is a different
    /// representation and would sound wrong rather than fail.
    pub fn load(dir: &Path, hidden: usize) -> Result<Self> {
        use ort::execution_providers::{CPUExecutionProvider, CUDAExecutionProvider};

        let model = dir.join("model.onnx");
        if !model.exists() {
            return Err(TtsError::Weights(format!(
                "{} not found — expected an ONNX export of \
                 chinese-roberta-wwm-ext-large that emits the third-from-last \
                 hidden layer",
                model.display()
            )));
        }
        let session = ort::session::Session::builder()
            .map_err(onnx)?
            .with_execution_providers([
                CUDAExecutionProvider::default().build(),
                CPUExecutionProvider::default().build(),
            ])
            .map_err(|e| TtsError::Weights(e.to_string()))?
            .commit_from_file(&model)
            .map_err(onnx)?;

        let tokenizer = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json"))
            .map_err(|e| TtsError::Vocabulary(format!("failed to read tokenizer.json: {e}")))?;

        Ok(Self {
            session: std::mem::ManuallyDrop::new(session),
            tokenizer,
            hidden,
        })
    }
}

#[cfg(feature = "onnx")]
impl ProsodyEncoder for OnnxProsody {
    fn hidden(&self) -> usize {
        self.hidden
    }

    fn encode(&mut self, text: &str, word2ph: &[usize]) -> Result<ProsodyFeatures> {
        use ort::value::Tensor;

        let encoded = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| TtsError::Vocabulary(format!("failed to tokenize: {e}")))?;
        let ids: Vec<i64> = encoded.get_ids().iter().map(|&i| i as i64).collect();
        let n = ids.len() as i64;

        let outputs = self
            .session
            .run(ort::inputs![
                "input_ids" => Tensor::from_array((vec![1, n], ids)).map_err(onnx)?,
                "token_type_ids" =>
                    Tensor::from_array((vec![1, n], vec![0i64; n as usize])).map_err(onnx)?,
                "attention_mask" =>
                    Tensor::from_array((vec![1, n], vec![1i64; n as usize])).map_err(onnx)?,
            ])
            .map_err(onnx)?;
        let (shape, data) = outputs["output"]
            .try_extract_tensor::<f32>()
            .map_err(onnx)?;

        // `[tokens, hidden]`, tokens including [CLS] and [SEP]. The graph keeps
        // them; the reference strips them here and so do we.
        let hidden = *shape.last().unwrap_or(&0) as usize;
        let tokens = data.len() / hidden.max(1);
        let characters = tokens.saturating_sub(2);
        if characters != word2ph.len() {
            return Err(TtsError::Misaligned {
                characters,
                word2ph: word2ph.len(),
            });
        }

        // One row per phoneme: character i's feature, repeated word2ph[i] times.
        let mut out = Vec::with_capacity(word2ph.iter().sum::<usize>() * hidden);
        for (i, &count) in word2ph.iter().enumerate() {
            let row = &data[(i + 1) * hidden..(i + 2) * hidden];
            for _ in 0..count {
                out.extend_from_slice(row);
            }
        }

        Ok(ProsodyFeatures {
            hidden,
            phones: word2ph.iter().sum(),
            data: out,
        })
    }
}

#[cfg(feature = "onnx")]
fn onnx(e: ort::Error) -> TtsError {
    TtsError::Weights(format!("onnxruntime: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zeros_are_the_right_shape_for_a_language_without_an_encoder() {
        // GPT-SoVITS feeds zeros for English and Japanese, so this is a real
        // input rather than an error path — it has to be shaped like the real
        // thing or the model sees a different sequence length than the phonemes.
        let f = ProsodyFeatures::zeros(1024, 7);
        assert_eq!(f.data.len(), 1024 * 7);
        assert_eq!(f.phones, 7);
    }
}
