//! Grapheme-to-phoneme for the TTS engine — pure Rust, no model, no tensors.
//!
//! Turns text into the phoneme symbols GPT-SoVITS was trained on, and into the
//! embedding indices that address its phoneme table. Historically this is where
//! non-Python TTS ports die: the neural part is arithmetic, but the front-end is
//! a pile of language-specific rules that each upstream keeps in Python.
//!
//! ```
//! # use text_kit::{Language, phonemize};
//! let out = phonemize("你好", Language::Zh).unwrap();
//! assert_eq!(out.phones, ["n", "i2", "h", "ao3"]);
//! ```
//!
//! Testable without a GPU, without weights and without a network, which is why
//! it is built before the network it feeds.

mod chinese;
mod english;
mod normalize;
mod segment;
pub mod symbols;

pub use segment::{Language, Run, split};
pub use symbols::{SYMBOLS, UNKNOWN, id, to_ids};

/// Something the front-end cannot say.
#[derive(Debug, thiserror::Error)]
pub enum TextError {
    /// A language whose front-end is not written yet. Deliberately an error
    /// rather than a fallback: phonemizing Japanese as English produces confident
    /// nonsense, and silence about it is worse than refusing.
    #[error(
        "no grapheme-to-phoneme front-end for {0} yet — \
         Chinese and English are implemented"
    )]
    Unsupported(&'static str),
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
pub fn phonemize(text: &str, language: Language) -> Result<Phonemes> {
    match language {
        Language::Zh => Ok(chinese::phonemize(text)),
        Language::En => Ok(english::phonemize(text)),
        Language::Ja => Err(TextError::Unsupported("Japanese")),
    }
}

/// Phonemize code-switched text, choosing a front-end per run.
///
/// The corpus this toolkit targets is mixed Chinese and English, so this rather
/// than [`phonemize`] is the normal entry point. `default` decides runs that
/// carry no script information.
pub fn phonemize_mixed(text: &str, default: Language) -> Result<Phonemes> {
    let mut phones = Vec::new();
    let mut word2ph: Option<Vec<usize>> = Some(Vec::new());
    let mut normalized = String::new();

    for run in split(text, default) {
        let part = phonemize(&run.text, run.language)?;
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
        assert!(phonemize("こんにちは", Language::Ja).is_err());
    }

    #[test]
    fn ids_address_the_symbol_table() {
        let out = phonemize("你好", Language::Zh).unwrap();
        let ids = out.ids();
        assert_eq!(ids.len(), out.phones.len());
        assert!(ids.iter().all(|&i| i < SYMBOLS.len()));
    }
}
