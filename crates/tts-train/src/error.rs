//! Error type for fine-tuning.

/// Errors from preparing a corpus or running a training loop.
#[derive(Debug, thiserror::Error)]
pub enum TrainError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    /// Something about the corpus itself — missing transcripts, empty files.
    #[error("{0}")]
    Corpus(String),
    #[error("{0}")]
    Text(#[from] text_kit::TextError),
    #[error("{0}")]
    Tts(#[from] tts_core::TtsError),
    #[error("failed to load weights: {0}")]
    Weights(String),
    #[error("device error: {0}")]
    Device(String),
}

impl From<burn_kit::Error> for TrainError {
    fn from(e: burn_kit::Error) -> Self {
        Self::Device(e.to_string())
    }
}

pub type Result<T> = std::result::Result<T, TrainError>;
