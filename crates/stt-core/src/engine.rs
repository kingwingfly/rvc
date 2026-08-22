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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::SttError;

    /// The contract stated as code: an engine that enforces on itself
    /// everything the decode loop is entitled to assume.
    ///
    /// Both real implementations already behave this way — `burn_engine` and
    /// `onnx_engine` each return `"step before encode"` from the same place —
    /// so this is the shared floor rather than a third convention.
    struct Contract {
        mel_bins: usize,
        vocab: usize,
        /// Frames of the window currently encoded, if any.
        window: Option<usize>,
        /// Whether `restart` is allowed to forget the window. Wrong, and here
        /// to show what the loop above would hit if an engine did it.
        restart_forgets: bool,
    }

    impl Contract {
        fn new() -> Self {
            Self {
                mel_bins: 80,
                vocab: 16,
                window: None,
                restart_forgets: false,
            }
        }
    }

    impl Engine for Contract {
        fn mel_bins(&self) -> usize {
            self.mel_bins
        }

        fn encode(&mut self, mel: &[f32], frames: usize) -> Result<()> {
            // `[n_mels, frames]`, mel-major — the layout both engines build a
            // `[1, n_mels, frames]` tensor from without transposing.
            assert_eq!(mel.len(), self.mel_bins * frames, "mel-major, no padding");
            self.window = Some(frames);
            self.restart();
            Ok(())
        }

        fn step(&mut self, tokens: &[u32]) -> Result<Vec<f32>> {
            if self.window.is_none() {
                return Err(SttError::Config("step before encode".into()));
            }
            assert!(!tokens.is_empty(), "a step always feeds at least one token");
            Ok(vec![0.0; self.vocab])
        }

        fn restart(&mut self) {
            if self.restart_forgets {
                self.window = None;
            }
        }
    }

    /// `Transcriber` holds a `Box<dyn Engine>` and is driven from wherever the
    /// caller's audio arrives, so the trait object has to be both.
    #[test]
    fn the_trait_object_is_object_safe_and_send() {
        fn assert_send<T: Send + ?Sized>() {}
        assert_send::<dyn Engine>();

        let boxed: Box<dyn Engine> = Box::new(Contract::new());
        assert_eq!(boxed.mel_bins(), 80);
    }

    #[test]
    fn a_step_before_encode_is_an_error_rather_than_a_panic() {
        let mut engine = Contract::new();
        let message = engine.step(&[0]).expect_err("no window").to_string();
        assert!(message.contains("step before encode"), "{message}");
    }

    /// The half of `restart` that is easy to get wrong, and that `decode.rs`
    /// depends on: it drops the *decode* state and keeps the *encoded audio*.
    ///
    /// `greedy` restarts twice per window — once before language detection and
    /// once before the real prompt — and never re-encodes. An engine that
    /// cleared its encoder output in `restart` would therefore fail on the
    /// prompt, which is what the second half of this shows.
    #[test]
    fn restart_drops_the_decode_state_and_keeps_the_window() {
        let mut engine = Contract::new();
        engine.encode(&vec![0.0; 80 * 30], 30).expect("encode");

        engine.restart();
        engine.restart();
        let logits = engine.step(&[1, 2, 3]).expect("the window survived");
        assert_eq!(logits.len(), 16);

        let mut wrong = Contract::new();
        wrong.restart_forgets = true;
        wrong.encode(&vec![0.0; 80 * 30], 30).expect("encode");
        wrong.restart();
        wrong
            .step(&[1])
            .expect_err("an engine that forgets breaks the loop");
    }

    /// What a caller may read out of `step`: one score per vocabulary entry,
    /// indexed by token id, the same width every call, and finite.
    ///
    /// Finiteness is asserted separately and on purpose. The decode loop is a
    /// plain argmax over these, so it neither produces nor detects a `NaN` —
    /// `burn-whisper` once shipped a reversed causal mask that made a decode
    /// step's only position `-inf`, and `-inf` through the attention softmax is
    /// `NaN`, which propagates to every logit without erroring anywhere.
    #[test]
    fn a_step_returns_one_finite_score_per_vocabulary_entry() {
        let mut engine = Contract::new();
        engine.encode(&vec![0.0; 80 * 30], 30).expect("encode");

        let prompt = engine.step(&[0, 1, 2, 3]).expect("prompt");
        let next = engine.step(&[4]).expect("one more position");

        assert_eq!(prompt.len(), 16, "the vocabulary, not the prompt length");
        assert_eq!(next.len(), prompt.len(), "the width does not move per step");
        assert!(prompt.iter().all(|v| v.is_finite()));
        assert!(next.iter().all(|v| v.is_finite()));
    }

    /// The front-end is built once from this (`LogMel::new(engine.mel_bins())`),
    /// so it may not depend on decode state.
    #[test]
    fn mel_bins_does_not_move_with_the_decode() {
        let mut engine = Contract::new();
        let before = engine.mel_bins();
        engine.encode(&vec![0.0; 80 * 30], 30).expect("encode");
        engine.step(&[0]).expect("step");
        engine.restart();

        assert_eq!(engine.mel_bins(), before);
    }
}
