//! The BPE vocabulary and Whisper's control tokens.
//!
//! Every id here is read from the model repo's own `generation_config.json`
//! rather than written down. They move between Whisper releases — large-v3 added
//! Cantonese and pushed all 100 language tokens up by one — and a hardcoded
//! table would decode fluent nonsense on the wrong checkpoint instead of failing.

use std::collections::HashMap;
use std::path::Path;

use crate::error::{Result, SttError};

/// Whisper's control tokens, resolved for one checkpoint.
pub struct Tokens {
    /// `<|startoftranscript|>` — always the first token fed to the decoder.
    pub sot: u32,
    /// `<|endoftext|>` — generation stops here.
    pub eot: u32,
    /// `<|transcribe|>`: transcribe in the source language.
    pub transcribe: u32,
    /// `<|translate|>`: translate to English. Whisper's only translation
    /// direction, which is why `voice translate` will be a separate model.
    pub translate: u32,
    /// `<|notimestamps|>` — suppresses timestamp tokens in the output.
    pub no_timestamps: u32,
    /// Language tag by ISO code (`"en"`, `"zh"`, `"ja"`).
    pub languages: HashMap<String, u32>,
    /// Tokens never worth emitting (formatting marks, most control tokens).
    pub suppress: Vec<u32>,
    /// Tokens suppressed only at the first generated position — a leading space
    /// or an immediate end-of-text.
    pub suppress_at_start: Vec<u32>,
}

impl Tokens {
    /// Read the control tokens from a `generation_config.json`.
    pub fn load(path: &Path) -> Result<Self> {
        let raw = std::fs::read_to_string(path)?;
        let cfg: serde_json::Value = serde_json::from_str(&raw)?;

        let id = |key: &str| -> Result<u32> {
            cfg.get(key)
                .and_then(|v| v.as_u64())
                .map(|v| v as u32)
                .ok_or_else(|| SttError::Config(format!("generation_config.json has no `{key}`")))
        };
        let ids = |key: &str| -> Vec<u32> {
            cfg.get(key)
                .and_then(|v| v.as_array())
                .map(|a| {
                    a.iter()
                        .filter_map(|v| v.as_u64())
                        .map(|v| v as u32)
                        .collect()
                })
                .unwrap_or_default()
        };

        let task = cfg.get("task_to_id").and_then(|v| v.as_object());
        let task_id = |name: &str| -> Result<u32> {
            task.and_then(|t| t.get(name))
                .and_then(|v| v.as_u64())
                .map(|v| v as u32)
                .ok_or_else(|| SttError::Config(format!("no `{name}` in task_to_id")))
        };

        // `<|en|>` in the config, `en` here.
        let languages = cfg
            .get("lang_to_id")
            .and_then(|v| v.as_object())
            .map(|m| {
                m.iter()
                    .filter_map(|(k, v)| {
                        let code = k.trim_start_matches("<|").trim_end_matches("|>");
                        Some((code.to_string(), v.as_u64()? as u32))
                    })
                    .collect()
            })
            .unwrap_or_default();

        Ok(Self {
            sot: id("decoder_start_token_id")?,
            eot: id("eos_token_id")?,
            transcribe: task_id("transcribe")?,
            translate: task_id("translate")?,
            no_timestamps: id("no_timestamps_token_id")?,
            languages,
            suppress: ids("suppress_tokens"),
            suppress_at_start: ids("begin_suppress_tokens"),
        })
    }

    /// The id for an ISO language code, or an error naming what is available.
    pub fn language(&self, code: &str) -> Result<u32> {
        self.languages.get(code).copied().ok_or_else(|| {
            let mut known: Vec<&str> = self.languages.keys().map(String::as_str).collect();
            known.sort_unstable();
            SttError::Config(format!(
                "unknown language `{code}` (this model knows {} codes: {} …)",
                known.len(),
                known
                    .iter()
                    .take(12)
                    .cloned()
                    .collect::<Vec<_>>()
                    .join(", ")
            ))
        })
    }
}

/// The BPE vocabulary, wrapping the `tokenizer.json` the model repo ships.
pub struct Vocabulary(tokenizers::Tokenizer);

impl Vocabulary {
    pub fn load(path: &Path) -> Result<Self> {
        tokenizers::Tokenizer::from_file(path)
            .map(Self)
            .map_err(|e| SttError::Config(format!("failed to read tokenizer.json: {e}")))
    }

    /// Decode ids to text, dropping control tokens.
    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        self.0
            .decode(ids, true)
            .map_err(|e| SttError::Config(format!("failed to decode tokens: {e}")))
    }
}
