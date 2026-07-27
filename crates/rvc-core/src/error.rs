//! Error type for the voice-conversion crate.

/// Errors produced while loading models or running RVC inference.
#[derive(Debug, thiserror::Error)]
pub enum VcError {
    /// An error from the ONNX Runtime (`ort`) — session build, run, extract.
    #[error("onnxruntime error: {0}")]
    Ort(#[from] ort::Error),
    /// Audio decode/encode error bubbled up from `voice-audio`.
    #[error("audio error: {0}")]
    Audio(#[from] voice_audio::AudioError),
    /// A model produced an output whose shape we could not interpret.
    #[error("unexpected tensor shape {got:?} for {what} (expected axis of size {expected})")]
    Shape {
        what: &'static str,
        expected: usize,
        got: Vec<i64>,
    },
    /// The model output tensor could not be located by name.
    #[error("model has no output named `{0}`")]
    MissingOutput(String),
    /// Model file (or a required component) was not found.
    #[error("model file not found: {0}")]
    NotFound(std::path::PathBuf),
    /// An error from the native Burn generator backend (weight load / inference).
    #[cfg(feature = "burn")]
    #[error("burn backend error: {0}")]
    Burn(String),
    /// The requested compute device is unavailable, or the chosen backend cannot
    /// drive it. Always a message rather than a panic: asking for hardware you
    /// don't have is a user mistake, not a bug.
    #[error("device error: {0}")]
    Device(String),
}

/// Flatten `burn-kit`'s device errors to text rather than nesting them.
///
/// A `#[from]` field would also become this variant's `source()`, and the two
/// print the same sentence — so an `anyhow` chain would say it twice.
impl From<burn_kit::Error> for VcError {
    fn from(e: burn_kit::Error) -> Self {
        Self::Device(e.to_string())
    }
}

/// Convenience alias used throughout the crate.
pub type Result<T> = std::result::Result<T, VcError>;
