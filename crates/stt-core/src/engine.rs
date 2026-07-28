//! What the decode loop needs from a Whisper implementation.
//!
//! Two implementations sit behind this: the native Burn port
//! ([`burn_engine`](crate::burn_engine)) and ONNX Runtime
//! ([`onnx_engine`](crate::onnx_engine)). The loop above them never learns which
//! — it deals in token ids and `f32` logits, which is all either can agree on.
//!
//! Encoded audio and decoder key/value caches stay *inside* the engine. They are
//! backend-specific tensors with no useful common type, and hoisting them would
//! mean converting them to host memory every step for nothing.

use crate::Result;

/// One loaded Whisper model, mid-transcription.
pub trait Engine: Send {
    /// Mel bands this model expects — 80 through large-v2, 128 from large-v3.
    /// The front-end has to agree, so it is read from the model, not assumed.
    fn mel_bins(&self) -> usize;

    /// Encode one window of log-mel (`[n_mels, frames]`, mel-major) and begin a
    /// fresh decode against it.
    fn encode(&mut self, mel: &[f32], frames: usize) -> Result<()>;

    /// Feed the next tokens and return the logits for the final position.
    ///
    /// Called with the whole prompt first, then one token at a time; the engine
    /// carries the key/value cache between calls.
    fn step(&mut self, tokens: &[u32]) -> Result<Vec<f32>>;

    /// Drop the decode state, keeping the encoded audio.
    ///
    /// Language detection probes the model with a single token and then throws
    /// that away before the real decode starts.
    fn restart(&mut self);
}
