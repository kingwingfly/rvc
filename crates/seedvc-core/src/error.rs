//! The engine's error type.

/// Anything that can go wrong loading or running the converter.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// A checkpoint could not be read, or did not fit the module tree.
    #[error("failed to load {what}: {why}")]
    Load { what: &'static str, why: String },
    /// A backend or device was named that this build cannot provide.
    #[error("{0}")]
    Device(String),
    /// The reference clip is unusable — too short to specify a speaker.
    #[error("{0}")]
    Reference(String),
    /// A conversion was asked for that the model's fixed windows cannot express:
    /// a chunk past the content encoder's 30 s, a conditioning window past the
    /// transformer's `block_size`, or a noise buffer of the wrong width. Every
    /// one of them is a chunking decision, so the message names the arithmetic
    /// rather than the symptom.
    #[error("{0}")]
    Input(String),
    #[error(transparent)]
    Audio(#[from] audio_kit::AudioError),
    #[error(transparent)]
    Io(#[from] std::io::Error),
}

pub type Result<T> = std::result::Result<T, Error>;

/// `burn-kit` reports a failed device or a caught backend abort; flattening it to
/// a string keeps an `anyhow` chain from printing the same sentence twice, the
/// same reason `rvc-core` does it.
impl From<burn_kit::Error> for Error {
    fn from(e: burn_kit::Error) -> Self {
        Self::Device(e.to_string())
    }
}
