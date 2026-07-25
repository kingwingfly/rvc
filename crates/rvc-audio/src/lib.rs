//! Audio I/O for the rvc voice-conversion toolkit.
//!
//! Everything is mono `f32` PCM and everything crosses module boundaries as a
//! [`futures::Stream`] of [`Samples`] chunks, so decode → convert → encode
//! wires together as one async pipeline that works equally for batch files and
//! realtime stdin/stdout piping.
//!
//! - [`decode`] turns mp3/wav/... files into a resampled mono `f32` stream.
//! - [`pcm`] turns raw `f32le` stdin/stdout into/out of that same stream shape.
//! - [`encode`] writes a stream to a float WAV file.

pub mod decode;
pub mod encode;
mod error;
pub mod pcm;
pub mod slice;

pub use error::{AudioError, Result};

/// A chunk of mono `f32` PCM samples. This is the unit that flows through every
/// stream in the toolkit.
pub type Samples = Vec<f32>;

pub use decode::{DecodeOptions, decode_path, decode_paths};
pub use encode::{write_wav, write_wav_file};
pub use pcm::{read_f32le, write_f32le};
pub use slice::{SliceOptions, slice};
