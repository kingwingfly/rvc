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
    ///
    /// Only ever the graph's **endpoints** — `abuffer` and `abuffersink` —
    /// because those are found by name before the graph is built. A filter
    /// named inside a chain description is resolved by libavfilter's own
    /// parser, so its absence arrives as [`AudioError::Ffmpeg`] instead. Do
    /// not match on this variant to detect a missing chain filter: it will not
    /// fire, which is a test that steps around nothing while looking like it
    /// steps around something.
    #[error("ffmpeg filter unavailable: {0}")]
    FilterUnavailable(&'static str),
    /// A filter chain negotiated an output sample rate other than the one the
    /// graph was declared at.
    ///
    /// [`AudioFilter`](crate::AudioFilter) pins its sink's *format* and
    /// *channel layout* but cannot pin its rate, so a chain holding a
    /// resampler — `aresample`, or `loudnorm`, which negotiates 192 kHz of its
    /// own accord — hands back samples at a rate the caller has no way to
    /// learn. Counting those as if they were the declared rate is silent and
    /// total: the audio is right and every duration computed from it is wrong
    /// by the ratio.
    #[error(
        "filter chain renegotiated the sample rate to {got} Hz, but the graph is declared at \
         {declared} Hz; a chain that resamples cannot be driven through this type"
    )]
    RateRenegotiated {
        /// The rate the graph was built for.
        declared: u32,
        /// The rate a frame actually came out at.
        got: u32,
    },
    /// A [`StereoSamples`](crate::StereoSamples) arrived with its two channels
    /// at different lengths, which its own invariant forbids.
    #[error("stereo chunk channels differ in length: left {left}, right {right}")]
    ChannelLengthMismatch {
        /// Samples in the left channel.
        left: usize,
        /// Samples in the right channel.
        right: usize,
    },
}

/// Convenience alias used throughout the crate.
pub type Result<T> = std::result::Result<T, AudioError>;
