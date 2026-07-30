//! Corpus → training clips.
//!
//! A clip is an audio file beside a transcript, and preparing it means running
//! the two frozen encoders over it: cnhubert and the quantiser turn the audio
//! into semantic tokens, and the prosody BERT turns the transcript into a
//! feature per phoneme. Both are expensive and neither changes during training,
//! so the whole corpus is prepared once, up front.
//!
//! That is also why fine-tuning wants `stt`: a corpus is audio, and this needs
//! audio *with transcripts*.

use std::path::{Path, PathBuf};

use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, TensorData};
use burn_gptsovits::{Hubert, HubertConfig, Quantizer, QuantizerConfig};
use text_kit::Language;
use tts_core::{ANALYSIS_SR, ProsodyEncoder};

use crate::error::{Result, TrainError};

/// Waveform samples per latent frame at the synthesis rate — the decoder's
/// upsample product (`[10, 8, 2, 2, 2]`), which is also the spectrogram hop, so
/// one spectrogram frame is exactly one latent frame.
pub const SAMPLES_PER_FRAME: usize = 640;

/// One prepared example.
#[derive(Debug, Clone)]
pub struct Clip {
    /// Where it came from, for error messages.
    pub source: PathBuf,
    /// Phoneme ids from the transcript.
    pub phones: Vec<u32>,
    /// Prosody, one row of `bert_dim` per phoneme, phone-major.
    pub bert: Vec<f32>,
    pub bert_dim: usize,
    /// Semantic tokens the audio quantised to, at 25 Hz.
    pub tokens: Vec<u32>,
    /// Ground-truth waveform at [`tts_core::OUTPUT_SR`], trimmed to exactly
    /// `frames() * SAMPLES_PER_FRAME` samples.
    ///
    /// Empty unless preparation was asked for it: only `s2` compares generated
    /// audio against the original, and decoding the corpus a second time is
    /// wasted work for an `s1`-only run.
    pub audio: Vec<f32>,
}

impl Clip {
    /// Roughly how long the clip is, from its token count.
    pub fn seconds(&self) -> f32 {
        self.tokens.len() as f32 / 25.0
    }

    /// Latent frames the clip is worth: one per 50 Hz frame, so two per token.
    ///
    /// `enc_p` upsamples the 25 Hz codes to 50 Hz and `enc_q` reads a 50 Hz
    /// spectrogram, so this is the length both sides of the KL term must agree
    /// on. Bounded by the waveform as well when there is one, since ffmpeg's
    /// output length and the token count are rounded independently and the
    /// shorter is the only one both can honour.
    ///
    /// Always even. `enc_p` reaches its length by repeating each 25 Hz code
    /// twice, so it can only ever produce an even count; an odd `frames` would
    /// leave the prior one frame shorter than the posterior and the KL term
    /// would silently compare two different time bases.
    pub fn frames(&self) -> usize {
        let from_tokens = self.tokens.len() * 2;
        let frames = match self.audio.is_empty() {
            true => from_tokens,
            false => from_tokens.min(self.audio.len() / SAMPLES_PER_FRAME),
        };
        frames & !1
    }
}

/// Find `<stem>.wav` / `<stem>.txt` pairs under `dir`.
///
/// A transcript with no audio, or audio with no transcript, is skipped with a
/// warning rather than failing the run — a corpus assembled by hand usually has
/// a few of both, and stopping on the first would be tedious.
pub fn pairs(dir: &Path) -> Result<Vec<(PathBuf, PathBuf)>> {
    let mut out = Vec::new();
    let entries = std::fs::read_dir(dir)
        .map_err(|e| TrainError::Corpus(format!("cannot read {}: {e}", dir.display())))?;
    for entry in entries.flatten() {
        let audio = entry.path();
        let is_audio = audio
            .extension()
            .and_then(|e| e.to_str())
            .is_some_and(|e| matches!(e, "wav" | "mp3" | "flac" | "m4a" | "ogg" | "opus"));
        if !is_audio {
            continue;
        }
        let text = audio.with_extension("txt");
        if text.exists() {
            out.push((audio, text));
        } else {
            tracing::warn!("no transcript beside {}; skipped", audio.display());
        }
    }
    out.sort();
    if out.is_empty() {
        return Err(TrainError::Corpus(format!(
            "no <name>.wav + <name>.txt pairs under {} — \
             `stt` can produce the transcripts",
            dir.display()
        )));
    }
    Ok(out)
}

/// Run the frozen encoders over a corpus.
///
/// `prosody` is optional and its absence is not fatal: zeros cost expressiveness
/// rather than correctness, and a fine-tune without them still adapts.
///
/// `waveforms` additionally decodes each clip at the synthesis rate, which `s2`
/// needs as ground truth and `s1` never looks at. Preparation is the expensive
/// part of a fine-tune — cnhubert, the quantiser and the prosody BERT over every
/// clip — so a `both` run does it once with this on rather than twice.
pub fn prepare<B: Backend>(
    pairs: &[(PathBuf, PathBuf)],
    hubert: &Hubert<B>,
    quantizer: &Quantizer<B>,
    prosody: &mut Option<Box<dyn ProsodyEncoder>>,
    language: Language,
    waveforms: bool,
    device: &B::Device,
) -> Result<Vec<Clip>> {
    let bert_dim = prosody.as_ref().map_or(1024, |p| p.hidden());
    let mut clips = Vec::with_capacity(pairs.len());

    for (audio_path, text_path) in pairs {
        let text = std::fs::read_to_string(text_path)
            .map_err(|e| TrainError::Corpus(format!("{}: {e}", text_path.display())))?;
        let text = text.trim();
        if text.is_empty() {
            tracing::warn!("{} is empty; skipped", text_path.display());
            continue;
        }

        let phonemes = text_kit::phonemize_mixed(text, language)?;
        if phonemes.phones.is_empty() {
            tracing::warn!("{} produced no phonemes; skipped", text_path.display());
            continue;
        }

        let audio = crate::audio::read(audio_path, ANALYSIS_SR)?;
        if audio.len() < ANALYSIS_SR as usize / 4 {
            tracing::warn!(
                "{} is under a quarter second; skipped",
                audio_path.display()
            );
            continue;
        }

        let wav: Tensor<B, 2> =
            Tensor::from_data(TensorData::new(audio.clone(), [1, audio.len()]), device);
        let ssl = hubert.forward(wav).swap_dims(1, 2);
        let tokens = crate::ids(quantizer.encode(ssl))?;

        let bert = match (prosody.as_mut(), &phonemes.word2ph) {
            (Some(encoder), Some(word2ph)) => encoder.encode(&phonemes.normalized, word2ph)?.data,
            _ => vec![0.0; bert_dim * phonemes.phones.len()],
        };

        // Trimmed to whole latent frames so the spectrogram, the codes and the
        // waveform all end together — `Spectral::linear` yields exactly
        // `len / hop` frames, so a ragged tail would put `enc_q` and `enc_p` one
        // frame apart and the KL term would compare misaligned sequences.
        let waveform = match waveforms {
            false => Vec::new(),
            true => {
                let mut wav = crate::audio::read(audio_path, tts_core::OUTPUT_SR)?;
                let frames = (tokens.len() * 2).min(wav.len() / SAMPLES_PER_FRAME);
                wav.truncate(frames * SAMPLES_PER_FRAME);
                wav
            }
        };

        clips.push(Clip {
            source: audio_path.clone(),
            phones: phonemes.ids().iter().map(|&i| i as u32).collect(),
            bert,
            bert_dim,
            tokens,
            audio: waveform,
        });
    }

    if clips.is_empty() {
        return Err(TrainError::Corpus(
            "every clip was skipped — nothing to train on".into(),
        ));
    }
    Ok(clips)
}

/// Build a `Hubert` and a `Quantizer` for preparation.
///
/// They are frozen, so they exist only long enough to prepare the corpus and are
/// dropped before training starts — which matters on a small card, where they
/// would otherwise sit beside the model being trained.
pub fn encoders<B: Backend>(
    hubert_path: &Path,
    s2_path: &Path,
    device: &B::Device,
) -> Result<(Hubert<B>, Quantizer<B>)> {
    let mut hubert = Hubert::<B>::new(&HubertConfig::chinese_base(), device);
    hubert
        .load_pytorch(hubert_path)
        .map_err(|e| TrainError::Weights(format!("cnhubert: {e}")))?;

    let mut quantizer = Quantizer::<B>::new(&QuantizerConfig::default(), 1, device);
    quantizer
        .load_pytorch(s2_path)
        .map_err(|e| TrainError::Weights(format!("quantiser: {e}")))?;

    Ok((hubert, quantizer))
}
