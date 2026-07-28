//! Error type for speech synthesis.

/// Errors from loading a GPT-SoVITS model or synthesising with it.
#[derive(Debug, thiserror::Error)]
pub enum TtsError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("{0}")]
    Text(#[from] text_kit::TextError),
    /// Weights could not be read, or did not fit the graph.
    #[error("failed to load weights: {0}")]
    Weights(String),
    /// A tokenizer or vocabulary problem.
    #[error("{0}")]
    Vocabulary(String),
    /// The per-character phoneme counts did not line up with the text.
    ///
    /// Its own variant because it is the one alignment the whole prosody path
    /// depends on, and a silent mismatch shifts features against phonemes rather
    /// than failing.
    #[error("{characters} characters of text but {word2ph} per-character phoneme counts")]
    Misaligned { characters: usize, word2ph: usize },
    #[error("device error: {0}")]
    Device(String),
}

impl From<burn_kit::Error> for TtsError {
    fn from(e: burn_kit::Error) -> Self {
        Self::Device(e.to_string())
    }
}

pub type Result<T> = std::result::Result<T, TtsError>;
