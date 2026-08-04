//! Grapheme-to-phoneme for the TTS engine — pure Rust, no model, no tensors.
//!
//! Turns text into the phoneme symbols GPT-SoVITS was trained on, and into the
//! embedding indices that address its phoneme table. Historically this is where
//! non-Python TTS ports die: the neural part is arithmetic, but the front-end is
//! a pile of language-specific rules that each upstream keeps in Python.
//!
//! ```
//! # use text_kit::{Language, phonemize, phonemize_mixed};
//! let out = phonemize("你好", Language::Zh, None).unwrap();
//! assert_eq!(out.phones, ["n", "i2", "h", "ao3"]);
//!
//! // Mixed text is the normal case here, and each run gets its own front-end.
//! let out = phonemize_mixed("你好world", Language::Zh, None).unwrap();
//! assert_eq!(out.phones[4..], ["W", "ER1", "L", "D"]);
//! ```
//!
//! Three front-ends, and they are not symmetric. Mandarin is a pipeline of
//! rules (jieba, pinyin, tone sandhi, the opencpop table) and reports a phoneme
//! count per character. English is a 126k-entry dictionary with a cascade of
//! fallbacks behind it, and reports no per-character count at all — see
//! [`Phonemes::word2ph`]. Japanese is an OpenJTalk analyser whose accent output
//! this crate turns into prosody symbols, and it is the one that needs something
//! from the caller: a [`JapaneseDict`], because its dictionary is 28.7 MB and a
//! download does not belong in a `build.rs`.
//!
//! Testable without a GPU, without weights and without a network — and, for the
//! Japanese rules, without the dictionary either — which is why it is built
//! before the network it feeds.

mod chinese;
mod english;
mod japanese;
mod normalize;
mod segment;
pub mod symbols;

pub use japanese::JapaneseDict;
pub use segment::{Language, Run, split};
pub use symbols::{SYMBOLS, UNKNOWN, id, to_ids};

/// Something the front-end cannot say.
#[derive(Debug, thiserror::Error)]
pub enum TextError {
    /// Japanese was asked for with no dictionary to ask.
    ///
    /// Deliberately an error rather than a fallback to another front-end:
    /// phonemizing Japanese as English produces confident nonsense, and silence
    /// about it is worse than refusing.
    #[error(
        "Japanese needs a NAIST-JDic dictionary and none was given — \
         open one with `JapaneseDict::open` and pass it to `phonemize`"
    )]
    NoJapaneseDictionary,

    /// The Japanese dictionary would not load, or would not analyse a line.
    ///
    /// A `jpreprocess` dictionary is version-locked to the release that built
    /// it, so a directory from a different one is the commonest cause.
    #[error("the Japanese dictionary at `{}` could not be used: {reason}", path.display())]
    JapaneseDictionary {
        path: std::path::PathBuf,
        reason: String,
    },
}

pub type Result<T> = std::result::Result<T, TextError>;

/// Phonemes for one stretch of text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Phonemes {
    /// Phoneme symbols, in order. Members of [`SYMBOLS`].
    pub phones: Vec<String>,
    /// How many phonemes each character of [`Phonemes::normalized`] produced.
    ///
    /// `Some` for Chinese, where the T2S model uses it to spread per-character
    /// BERT features across phonemes; `None` for languages whose front-end has
    /// no per-character alignment to give. Sums to `phones.len()`.
    ///
    /// English is always `None`, as upstream returns it: normalization rewrites
    /// whole spans ("$3.50" becomes five words), so no character of the input
    /// owns a run of phonemes. Consumers feed the prosody encoder zeros
    /// instead, which is why English is intelligible but flatter than Chinese.
    pub word2ph: Option<Vec<usize>>,
    /// The text the phonemes were actually derived from.
    pub normalized: String,
}

impl Phonemes {
    /// Embedding indices for [`Phonemes::phones`].
    pub fn ids(&self) -> Vec<usize> {
        to_ids(&self.phones)
    }
}

/// Phonemize text known to be in one language.
pub fn phonemize(text: &str, language: Language, ja: Option<&JapaneseDict>) -> Result<Phonemes> {
    match language {
        Language::Zh => Ok(chinese::phonemize(text)),
        Language::En => Ok(english::phonemize(text)),
        // `ja` is threaded rather than opened here because the dictionary is
        // 28.7 MB: a caller that never sees Japanese must never pay for it, and
        // only the caller knows whether it will.
        Language::Ja => japanese::phonemize(text, ja),
    }
}

/// Phonemize code-switched text, choosing a front-end per run.
///
/// The corpus this toolkit targets is mixed Chinese and English, so this rather
/// than [`phonemize`] is the normal entry point. `default` decides runs that
/// carry no script information.
pub fn phonemize_mixed(
    text: &str,
    default: Language,
    ja: Option<&JapaneseDict>,
) -> Result<Phonemes> {
    let mut phones = Vec::new();
    let mut word2ph: Option<Vec<usize>> = Some(Vec::new());
    let mut normalized = String::new();

    for run in split(text, default) {
        let part = phonemize(&run.text, run.language, ja)?;
        phones.extend(part.phones);
        normalized.push_str(&part.normalized);
        // One run without per-character counts makes the whole result's counts
        // unusable — they would no longer line up with the text.
        match (&mut word2ph, part.word2ph) {
            (Some(acc), Some(part)) => acc.extend(part),
            _ => word2ph = None,
        }
    }

    Ok(Phonemes {
        phones,
        word2ph,
        normalized,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_unwritten_front_end_refuses_rather_than_guessing() {
        // Running Japanese through the English path would produce fluent-sounding
        // wrong audio, which is far harder to notice than an error.
        assert!(phonemize("こんにちは", Language::Ja, None).is_err());
    }

    #[test]
    fn ids_address_the_symbol_table() {
        let out = phonemize("你好", Language::Zh, None).unwrap();
        let ids = out.ids();
        assert_eq!(ids.len(), out.phones.len());
        assert!(ids.iter().all(|&i| i < SYMBOLS.len()));
    }

    #[test]
    fn a_code_switched_sentence_uses_both_front_ends() {
        // The corpus this toolkit targets mixes the two constantly. Sending the
        // whole line to one front-end drops the other language's characters
        // silently, which reads as the model swallowing a word.
        let out = phonemize_mixed("你好world", Language::Zh, None).unwrap();
        assert_eq!(out.phones[..4], ["n", "i2", "h", "ao3"]);
        assert_eq!(out.phones[4..], ["W", "ER1", "L", "D"]);
        // One run without per-character counts makes the whole result's counts
        // unusable, so a mixed line reports none at all.
        assert_eq!(out.word2ph, None);
    }

    #[test]
    fn every_phoneme_of_a_mixed_line_is_in_the_models_vocabulary() {
        for text in ["我用 ChatGPT 写了 3 行代码。", "今天 the weather is 很好!"] {
            let out = phonemize_mixed(text, Language::Zh, None).unwrap();
            assert!(!out.phones.is_empty(), "{text} produced nothing");
            for p in &out.phones {
                assert_ne!(
                    symbols::id(p),
                    symbols::id(UNKNOWN),
                    "{text}: `{p}` is not a known symbol"
                );
            }
        }
    }
}
