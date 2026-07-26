//! Error type for the audio crate.

/// Errors produced while decoding, resampling or encoding audio.
#[derive(Debug, thiserror::Error)]
pub enum AudioError {
    /// An error bubbled up from the underlying ffmpeg bindings.
    #[error("ffmpeg error: {0}")]
    Ffmpeg(#[from] ffmpeg_next::Error),
    /// A std I/O error (raw PCM read/write, file creation, ...).
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    /// The input container held no decodable audio stream.
    #[error("no audio stream found in input")]
    NoAudioStream,
    /// The blocking decode worker disappeared before finishing.
    #[error("decode worker terminated unexpectedly")]
    WorkerGone,
    /// The decoded sample format is not one we convert.
    #[error("unsupported sample format: {0:?}")]
    UnsupportedFormat(ffmpeg_next::format::Sample),
    /// A required libavfilter filter is missing from this ffmpeg build.
    #[error("ffmpeg filter unavailable: {0}")]
    FilterUnavailable(&'static str),
}

/// Convenience alias used throughout the crate.
pub type Result<T> = std::result::Result<T, AudioError>;
