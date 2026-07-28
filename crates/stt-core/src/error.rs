//! Error type for speech recognition.

/// Errors from loading a Whisper checkpoint or transcribing with it.
#[derive(Debug, thiserror::Error)]
pub enum SttError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("malformed model config: {0}")]
    Json(#[from] serde_json::Error),
    /// A config file parsed but did not say something required.
    #[error("{0}")]
    Config(String),
    /// Weights could not be read or did not fit the module tree.
    #[error("failed to load whisper weights: {0}")]
    Weights(String),
    #[error("device error: {0}")]
    Device(String),
}

impl From<burn_kit::Error> for SttError {
    fn from(e: burn_kit::Error) -> Self {
        Self::Device(e.to_string())
    }
}

pub type Result<T> = std::result::Result<T, SttError>;
