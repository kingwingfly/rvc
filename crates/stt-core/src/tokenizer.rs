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

#[cfg(test)]
mod tests {
    use super::*;

    /// A scratch directory, removed when the guard drops.
    ///
    /// Both types here load from a *path*, because both files ship beside the
    /// weights — so a test that bypassed the filesystem would not exercise the
    /// half of these functions that can actually fail.
    struct Scratch(std::path::PathBuf);

    impl Scratch {
        fn new() -> Self {
            use std::sync::atomic::{AtomicUsize, Ordering};
            static N: AtomicUsize = AtomicUsize::new(0);
            let dir = std::env::temp_dir().join(format!(
                "stt-core-tokenizer-{}-{}",
                std::process::id(),
                N.fetch_add(1, Ordering::Relaxed)
            ));
            std::fs::create_dir_all(&dir).expect("scratch dir");
            Self(dir)
        }

        fn write(&self, name: &str, body: &str) -> std::path::PathBuf {
            let path = self.0.join(name);
            std::fs::write(&path, body).expect("write fixture");
            path
        }
    }

    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    /// `expect_err` wants `Debug` on the success type, which neither of these
    /// two has and neither needs.
    fn failure<T>(result: Result<T>, why: &str) -> String {
        match result {
            Ok(_) => panic!("{why}"),
            Err(e) => e.to_string(),
        }
    }

    /// Whisper-shaped but with **synthetic** ids: every one is distinct, so a
    /// field read from the wrong key cannot coincidentally match. Real ids are
    /// deliberately not used — they move between releases, which is the whole
    /// reason this file reads them rather than writing them down.
    const CONFIG: &str = r#"{
      "decoder_start_token_id": 50,
      "eos_token_id": 51,
      "no_timestamps_token_id": 60,
      "task_to_id": { "transcribe": 55, "translate": 56 },
      "lang_to_id": { "<|en|>": 40, "<|zh|>": 41, "<|ja|>": 42 },
      "suppress_tokens": [1, 2, 7],
      "begin_suppress_tokens": [3, 51]
    }"#;

    #[test]
    fn every_control_token_comes_from_its_own_key() {
        let dir = Scratch::new();
        let tokens = Tokens::load(&dir.write("generation_config.json", CONFIG)).expect("load");

        assert_eq!(tokens.sot, 50, "sot is `decoder_start_token_id`");
        assert_eq!(tokens.eot, 51, "eot is `eos_token_id`");
        assert_eq!(tokens.transcribe, 55, "from task_to_id, not a scalar key");
        assert_eq!(tokens.translate, 56);
        assert_eq!(tokens.no_timestamps, 60);
        assert_eq!(tokens.suppress, vec![1, 2, 7]);
        assert_eq!(
            tokens.suppress_at_start,
            vec![3, 51],
            "`begin_suppress_tokens`"
        );
    }

    #[test]
    fn language_tags_lose_their_angle_brackets() {
        let dir = Scratch::new();
        let tokens = Tokens::load(&dir.write("generation_config.json", CONFIG)).expect("load");

        // The config spells them `<|en|>`; every caller spells them `en`.
        assert_eq!(tokens.language("en").expect("en"), 40);
        assert_eq!(tokens.language("zh").expect("zh"), 41);
        assert_eq!(tokens.language("ja").expect("ja"), 42);
        assert_eq!(tokens.languages.len(), 3);
        assert!(
            !tokens.languages.contains_key("<|en|>"),
            "brackets are stripped"
        );
    }

    #[test]
    fn an_unknown_language_says_how_many_this_model_knows() {
        let dir = Scratch::new();
        let tokens = Tokens::load(&dir.write("generation_config.json", CONFIG)).expect("load");

        let message = tokens.language("xx").expect_err("no such code").to_string();
        assert!(message.contains("`xx`"), "{message}");
        assert!(
            message.contains("3 codes"),
            "the count is the model's: {message}"
        );
    }

    #[test]
    fn a_missing_required_key_names_the_key() {
        let dir = Scratch::new();
        let without_eos = CONFIG.replace("\"eos_token_id\": 51,", "");
        let message = failure(
            Tokens::load(&dir.write("generation_config.json", &without_eos)),
            "eos_token_id is required",
        );
        assert!(message.contains("eos_token_id"), "{message}");

        let without_task = CONFIG.replace("\"transcribe\": 55, ", "");
        let message = failure(
            Tokens::load(&dir.write("no-task.json", &without_task)),
            "transcribe is required",
        );
        assert!(message.contains("transcribe"), "{message}");
        assert!(message.contains("task_to_id"), "{message}");
    }

    #[test]
    fn absent_suppression_lists_are_empty_rather_than_an_error() {
        let dir = Scratch::new();
        const BARE: &str = r#"{
          "decoder_start_token_id": 50,
          "eos_token_id": 51,
          "no_timestamps_token_id": 60,
          "task_to_id": { "transcribe": 55, "translate": 56 },
          "lang_to_id": { "<|en|>": 40 }
        }"#;
        let tokens = Tokens::load(&dir.write("generation_config.json", BARE)).expect("load");

        assert!(tokens.suppress.is_empty());
        assert!(tokens.suppress_at_start.is_empty());
    }

    /// The asymmetry `decode.rs`'s `detect_language` has to cope with: every
    /// scalar id is required, but `lang_to_id` is optional and defaults to an
    /// empty map. See `a_model_with_no_language_tokens_is_refused_rather_than`
    /// … in `decode.rs` for what that costs downstream.
    #[test]
    fn a_config_with_no_lang_to_id_still_loads_with_no_languages() {
        let dir = Scratch::new();
        let no_languages = CONFIG.replace(
            "\"lang_to_id\": { \"<|en|>\": 40, \"<|zh|>\": 41, \"<|ja|>\": 42 },",
            "",
        );
        let tokens =
            Tokens::load(&dir.write("generation_config.json", &no_languages)).expect("load");

        assert!(tokens.languages.is_empty(), "silently empty, not an error");
        assert!(tokens.language("en").is_err());
    }

    /// A miniature Whisper-shaped vocabulary: four ordinary tokens, then a run
    /// of control tokens with **one non-special token interleaved among them**.
    ///
    /// That interleaving is the point. It makes the fixture distinguish the
    /// implementation from the plausible wrong one — "skip every id at or above
    /// the first control token" — which no contiguous fixture can do.
    ///
    /// One thing to know before editing it: `tokenizers` hands the added tokens
    /// ids running on from the **model** vocabulary's size, not the ids written
    /// beside them here. Add a word to `vocab` without moving `added_tokens`
    /// down and every control token silently shifts by one, which reads exactly
    /// like the off-by-one the tests below are looking for.
    const VOCAB: &str = r#"{
      "version": "1.0",
      "truncation": null,
      "padding": null,
      "added_tokens": [
        {"id": 4, "content": "<|startoftranscript|>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
        {"id": 5, "content": "<|en|>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
        {"id": 6, "content": "<|nonspecial|>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": false},
        {"id": 7, "content": "<|notimestamps|>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
        {"id": 8, "content": "<|endoftext|>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}
      ],
      "normalizer": null,
      "pre_tokenizer": {"type": "Whitespace"},
      "post_processor": null,
      "decoder": null,
      "model": {
        "type": "WordLevel",
        "unk_token": "unknown",
        "vocab": {"hello": 0, "world": 1, "unknown": 2, "boundary": 3}
      }
    }"#;

    /// Decoding is the inverse of encoding for text the vocabulary covers.
    ///
    /// `Vocabulary` is decode-only — nothing in this crate encodes text, since
    /// Whisper is fed audio — so the encode half goes through the wrapped
    /// tokenizer directly. What that pins is ours all the same: that dropping
    /// control tokens does not also drop ordinary ones, and that the file this
    /// type loads is the one the ids are addressed against.
    #[test]
    fn an_encoding_decodes_back_to_the_text_it_came_from() {
        let dir = Scratch::new();
        let vocab = Vocabulary::load(&dir.write("tokenizer.json", VOCAB)).expect("load");

        for text in ["hello", "world", "hello world", "boundary world hello"] {
            let ids = vocab
                .0
                .encode(text, false)
                .expect("encode")
                .get_ids()
                .to_vec();
            assert!(!ids.is_empty(), "{text:?} is covered by the vocabulary");
            assert_eq!(vocab.decode(&ids).expect("decode"), text);
        }
    }

    #[test]
    fn ordinary_tokens_decode_to_their_text() {
        let dir = Scratch::new();
        let vocab = Vocabulary::load(&dir.write("tokenizer.json", VOCAB)).expect("load");

        assert_eq!(vocab.decode(&[0, 1]).expect("decode"), "hello world");
    }

    /// The whole of what `Vocabulary` adds over the `tokenizers` crate is the
    /// `true` in `decode(ids, true)`. This is that argument, stated as a test:
    /// a prompt's worth of control tokens must contribute **nothing** to the
    /// transcript.
    #[test]
    fn control_tokens_contribute_no_text() {
        let dir = Scratch::new();
        let vocab = Vocabulary::load(&dir.write("tokenizer.json", VOCAB)).expect("load");

        // Exactly the shape `decode.rs` builds: start, language, notimestamps,
        // content, end.
        let text = vocab.decode(&[4, 5, 7, 3, 8]).expect("decode");
        assert_eq!(text, "boundary");
        assert!(!text.contains("<|"), "no control token leaked: {text:?}");
    }

    /// The off-by-one this whole fixture exists for.
    ///
    /// Id 3 is the last ordinary token and id 4 the first control token, so a
    /// threshold that is off by one in either direction changes the answer:
    /// too low swallows `boundary`, too high leaks `<|startoftranscript|>`.
    /// Id 6 then pins that the rule is the `special` flag rather than any
    /// threshold at all — it *looks* like a control token, sits between two
    /// real ones, and must survive.
    #[test]
    fn the_special_flag_decides_and_not_an_id_threshold() {
        let dir = Scratch::new();
        let vocab = Vocabulary::load(&dir.write("tokenizer.json", VOCAB)).expect("load");

        assert_eq!(
            vocab.decode(&[3]).expect("decode"),
            "boundary",
            "id 3 is kept"
        );
        assert_eq!(vocab.decode(&[4]).expect("decode"), "", "id 4 is dropped");
        assert_eq!(
            vocab.decode(&[6]).expect("decode"),
            "<|nonspecial|>",
            "an id above the first control token, but not flagged special"
        );
        assert_eq!(
            vocab.decode(&[3, 4, 5, 6, 7, 8]).expect("decode"),
            "boundary <|nonspecial|>"
        );
    }

    #[test]
    fn an_unreadable_vocabulary_names_the_file() {
        let dir = Scratch::new();
        let message = failure(Vocabulary::load(&dir.0.join("absent.json")), "no such file");
        assert!(message.contains("tokenizer.json"), "{message}");
    }
}
