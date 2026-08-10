//! Audio I/O for the rvc voice-conversion toolkit.
//!
//! Everything crosses module boundaries as a [`futures::Stream`] of chunks, so
//! decode → convert → encode wires together as one async pipeline that works
//! equally for batch files and realtime stdin/stdout piping. That chunk is
//! **mono `f32`** ([`Samples`]) everywhere an engine can see it: mono is the
//! default and the crate-boundary type, and every engine in this toolkit reads
//! only that.
//!
//! - [`decode`] turns mp3/wav/... files into a resampled mono `f32` stream.
//! - [`pcm`] turns raw `f32le` stdin/stdout into/out of that same stream shape,
//!   and rate-adapts a chunk when the two ends of a pipe disagree.
//! - [`encode`] writes a stream to a float WAV file.
//! - [`filter`] applies a streaming libavfilter chain (e.g. de-noise) in-process.
//! - [`denoise`] is the one such chain more than one caller wants — `anlmdn`
//!   de-hiss, run by voice conversion on its output and by corpus preparation
//!   on a whole recording, so it belongs to neither of them.
//!
//! # The one stereo path, and why it stops where it does
//!
//! [`StereoSamples`] is the single exception to the mono rule, and it exists
//! for one consumer: a source-separation model of the MDX-Net family is
//! **stereo-native**. It eats the complex STFT of two channels as real planes,
//! and a centre-panned vocal is precisely what the stereo field lets it find —
//! so handing such a model a duplicated mono channel throws away the cue that
//! makes separation work. Nothing else in the toolkit wants two channels.
//!
//! The path is therefore deliberately short: [`decode::decode_path_stereo`]
//! produces it and [`encode::write_wav_stereo`] consumes it, and **no other
//! module speaks it**. In particular [`filter`] stays mono — its graph pins
//! `channel_layout=mono` at both the source and the sink — because separation
//! runs a model rather than a libavfilter chain and never passes through one.
//! Widening `AudioFilter` would be a second, unrelated piece of work; it is not
//! a loose end left by this one.

pub mod decode;
pub mod denoise;
pub mod encode;
mod error;
pub mod filter;
pub mod pcm;
pub mod slice;

pub use error::{AudioError, Result};

/// A chunk of mono `f32` PCM samples. This is the unit that flows through every
/// stream in the toolkit.
pub type Samples = Vec<f32>;

/// A chunk of two-channel `f32` PCM, held **planar**: one [`Samples`] per
/// channel rather than one interleaved `[L, R, L, R, …]` buffer.
///
/// ffmpeg hands frames over in whichever layout the codec produced, and a WAV
/// file's `data` chunk is interleaved by definition, so exactly one
/// de-interleave has to happen somewhere. It happens at the two ends —
/// [`decode::decode_path_stereo`] splits on the way in and
/// [`encode::write_wav_stereo`] weaves on the way out — so that everything in
/// between holds the layout its consumer wants: an STFT transforms each channel
/// separately, and would otherwise stride over an interleaved buffer once per
/// frame for the whole length of the signal, where a decode pays for the split
/// once per file.
///
/// Named fields cost nothing and make a left/right swap unspellable. The same
/// mistake in an interleaved `Vec<f32>` is an off-by-one that nobody hears
/// until a separation is quietly worse than it should be.
///
/// Both channels always hold the same number of samples; every producer in this
/// crate fills them in lockstep, and [`frames`](Self::frames) is that count.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct StereoSamples {
    /// The left channel.
    pub left: Samples,
    /// The right channel, sample-aligned with [`left`](Self::left).
    pub right: Samples,
}

impl StereoSamples {
    /// The number of *frames* — one sample from each channel — in this chunk.
    pub fn frames(&self) -> usize {
        debug_assert_eq!(
            self.left.len(),
            self.right.len(),
            "StereoSamples channels must stay the same length"
        );
        self.left.len()
    }

    /// Whether this chunk carries no audio at all.
    pub fn is_empty(&self) -> bool {
        self.left.is_empty() && self.right.is_empty()
    }
}

pub use decode::{
    DecodeOptions, decode_path, decode_path_stereo, decode_paths, decode_paths_stereo,
};
pub use denoise::{DenoiseParams, Denoiser};
pub use encode::{write_wav, write_wav_file, write_wav_stereo, write_wav_stereo_file};
pub use filter::AudioFilter;
pub use pcm::{read_f32le, resample_linear, write_f32le, write_f32le_chunk};
pub use slice::{Clip, SliceOptions, Slicer, slice};
