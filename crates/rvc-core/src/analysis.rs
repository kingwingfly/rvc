//! The seam between feature extraction and whichever runtime computes it.
//!
//! ContentVec and RMVPE were ONNX-only for as long as `rvc` had one runtime for
//! them. They now have a Burn path as well, so the two models need the same
//! run-time choice the generator has had since `Generator` was introduced —
//! **an addition, never a replacement**: ONNX Runtime stays a supported target
//! and is still what `auto` picks for an `.onnx` model.
//!
//! These are trait *objects* rather than a type parameter on
//! [`FeatureExtractor`](crate::FeatureExtractor), and that is deliberate. The
//! values crossing this boundary are already `Vec<Vec<f32>>` and `Vec<f32>` —
//! there is no tensor to keep generic over — so a `Box<dyn _>` costs one
//! indirection per clip and keeps `FeatureExtractor` non-generic. That is the
//! shape `stt-core`'s `Engine` uses to keep `Transcriber` non-generic, for the
//! same reason: the backend becomes a constructor call instead of a type that
//! infects every caller.
//!
//! Neither trait mentions a backend, a device or a file format, so this module
//! pulls in nothing. Deciding *which* implementation to build is the binary's
//! job — `stt-cli` and `tts-cli` already build their `Engine` themselves, and
//! `rvc-cli` builds these the same way via
//! [`FeatureExtractor::from_parts`](crate::FeatureExtractor::from_parts).

use crate::error::Result;

/// Per-frame content vectors from a mono 16 kHz waveform.
///
/// The output is time-major `[T][768]` at **50 Hz** — one frame per 320 input
/// samples. Callers upsample ×2 onto the 100 Hz F0 grid; that is not this
/// trait's job, because it is the same arithmetic whatever computed the frames.
///
/// `&mut self` because an ONNX session's `run` takes `&mut`, not because there
/// is state worth carrying between clips.
pub trait ContentEncoder: Send {
    /// Extract content features from a mono **16 kHz** `f32` buffer.
    fn extract(&mut self, wav16k: &[f32]) -> Result<Vec<Vec<f32>>>;
}

/// A fundamental-frequency contour in Hz at 100 Hz, from a mono 16 kHz waveform.
///
/// **Zero means unvoiced, and stays zero.** Every consumer in this toolkit
/// depends on that: `dsp::f0_to_coarse` maps 0 to bin 1 and `shift_pitch`
/// leaves zeros alone, so the generator's noise branch — the thing that renders
/// breath and unvoiced phonation — fires exactly on those frames. Upstream RVC
/// 2.3 began interpolating F0 across unvoiced frames, which silences that
/// branch entirely; `CLAUDE.md`'s "What the 2.3 audit found" records why this
/// toolkit deliberately did not follow. An implementation that interpolates
/// would satisfy the type and break the model.
pub trait PitchEstimator: Send {
    /// Estimate the F0 (Hz) contour from a mono **16 kHz** `f32` buffer.
    fn extract(&mut self, wav16k: &[f32]) -> Result<Vec<f32>>;
}
