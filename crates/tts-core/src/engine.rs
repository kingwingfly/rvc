//! What synthesis needs from a GPT-SoVITS implementation.
//!
//! Two implementations sit behind this: the native Burn port
//! ([`burn_engine`](crate::burn_engine)) and ONNX Runtime
//! ([`onnx_engine`](crate::onnx_engine)). [`Synthesizer`](crate::Synthesizer)
//! above them never learns which — it deals in token ids and `f32`, which is all
//! either can agree on, and that is what keeps it non-generic where a
//! `Synthesizer<B>` would force every caller to name a Burn type.
//!
//! The same arrangement `stt-core` uses, with one difference worth naming. There
//! the encoded audio has to stay *inside* the engine because it is a large
//! backend-specific tensor; here the two things the reference analysis produces
//! — a few hundred token ids and a 512-float speaker vector — are small enough
//! to hand back as plain `Vec`s, so a [`Reference`](crate::Reference) is a value
//! rather than a handle and can outlive the engine that made it. The `s1`
//! key/value cache is the part that stays inside, for the same reason it does in
//! `stt-core`.

use crate::Result;

/// One loaded GPT-SoVITS, mid-utterance.
pub trait Engine: Send {
    /// Analyse a reference clip: `[samples]` mono `f32` at
    /// [`ANALYSIS_SR`](crate::ANALYSIS_SR).
    ///
    /// Returns its semantic tokens — the prompt `s1` continues from — and the
    /// speaker vector `s2` is conditioned on.
    fn analyse(&mut self, audio: &[f32]) -> Result<(Vec<u32>, Vec<f32>)>;

    /// Begin a generation: the whole phoneme sequence and the reference's tokens
    /// in one pass, returning the logits for the position after them.
    ///
    /// `bert` is phone-major, `phones.len() * hidden` long. Discards whatever
    /// generation was in progress.
    fn s1_prompt(
        &mut self,
        phones: &[u32],
        bert: &[f32],
        hidden: usize,
        prompt: &[u32],
    ) -> Result<Vec<f32>>;

    /// Feed one generated token back and return the next logits.
    ///
    /// `position` counts **audio** tokens alone: the two positional encodings are
    /// applied separately and only then concatenated, so a token generated tenth
    /// is the tenth of the audio sequence, not of the whole prompt.
    fn s1_step(&mut self, token: u32, position: usize) -> Result<Vec<f32>>;

    /// Render semantic tokens to `[samples]` at [`OUTPUT_SR`](crate::OUTPUT_SR).
    ///
    /// `noise` is `[192, 2 * codes.len()]` row-major and already scaled by
    /// `noise_scale`: the prior is sampled from a distribution the caller draws,
    /// not one the engine draws, so the two runtimes can be compared on the same
    /// input. Whether that mattered is settled — it is how the ONNX path was
    /// checked against the Burn one.
    fn s2(
        &mut self,
        codes: &[u32],
        text: &[u32],
        speaker: &[f32],
        noise: &[f32],
    ) -> Result<Vec<f32>>;
}

/// Latent width of `s2`'s prior — how many rows of `noise` [`Engine::s2`] wants.
pub const PRIOR_CHANNELS: usize = 192;

/// The id generation stops on: the last of the semantic vocabulary, and the only
/// one that is not a codebook entry.
pub const EOS: u32 = 1024;
