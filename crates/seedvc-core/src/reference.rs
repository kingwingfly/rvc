//! Turning a reference clip into what conditions the transformer.
//!
//! This is the whole of a speaker's specification, and it is the reason there is
//! no `train` subcommand anywhere in this crate: a few seconds of somebody
//! talking is what `rvc` and `tts` each spend hours of GPU time to learn.
//!
//! # Two decodes of one file, and they are not redundant
//!
//! [`Model::analyse`] wants the clip at **both** rates — 16 kHz for Whisper and
//! CAMPPlus, 22.05 kHz for the mel — because two of the three things a reference
//! contributes are computed at one rate and the third at the other. Decoding
//! twice is what upstream effectively does, and it is the better of the two
//! options: resampling one to the other would put a linear interpolator in front
//! of a network whose weights were fitted behind a polyphase filter.
//! `audio-kit` has [`resample_linear`](audio_kit::resample_linear) for the pipe,
//! where there is no file to decode a second time; a batch run has the file.
//!
//! # 25 s, and it is a cap rather than a target
//!
//! Upstream trims to `ref_audio[:sr * 25]` and this does too. The timbre vector
//! is a pooled average, so past a few seconds more audio buys very little; what
//! it does buy is a **shorter source chunk**, because the reference's mel and the
//! source's share one 30 s context window — see [`crate::convert`], where that
//! arithmetic lives. A 25 s reference leaves under 5 s per chunk.

use std::path::Path;

use audio_kit::DecodeOptions;
use futures::StreamExt;

use crate::error::{Error, Result};
use crate::model::{CONTENT_SR, Model, Reference};

/// Longest stretch of a reference clip that is read, in seconds — upstream's
/// `ref_audio[:sr * 25]`.
pub const REFERENCE_SECONDS: usize = 25;

/// Decode a reference recording at both rates and analyse it.
///
/// Anything ffmpeg opens will do, at any rate and channel count; what comes back
/// is the [`Reference`] every conversion against this speaker is conditioned on,
/// so analysing once and converting many times is the intended shape.
pub async fn analyse(model: &dyn Model, path: impl AsRef<Path>) -> Result<Reference> {
    let path = path.as_ref();
    let content = decode(path, CONTENT_SR).await?;
    let mel = decode(path, model.config().sample_rate).await?;
    analyse_pcm(model, &content, &mel)
}

/// Analyse a reference already decoded at both rates.
///
/// `content` is mono `f32` at [`CONTENT_SR`] and `mel` is the **same clip** at
/// [`SeedVcConfig::sample_rate`](burn_seedvc::SeedVcConfig::sample_rate). Both
/// are trimmed to [`REFERENCE_SECONDS`] here rather than by the caller, so the
/// cap cannot be applied to one and forgotten on the other — the two lengths
/// disagreeing is not an error the model can detect, since neither tells it
/// anything about the other.
pub fn analyse_pcm(model: &dyn Model, content: &[f32], mel: &[f32]) -> Result<Reference> {
    let cfg = model.config();
    let content = trim(content, CONTENT_SR);
    let mel = trim(mel, cfg.sample_rate);

    // The clip has to survive the trim as *audio*, and this failure is worth
    // separating from the model's own checks: those speak in mel frames and
    // Kaldi windows, which is the right language once a clip is inside but not
    // the right one for a path ffmpeg decoded to nothing.
    if content.is_empty() || mel.is_empty() {
        return Err(Error::Reference(
            "the reference clip decoded to no audio at all".into(),
        ));
    }
    model.analyse(content, mel)
}

/// Upstream's cap, applied at whichever rate the samples are at.
fn trim(pcm: &[f32], sample_rate: u32) -> &[f32] {
    &pcm[..pcm.len().min(REFERENCE_SECONDS * sample_rate as usize)]
}

/// Decode one file to mono `f32` at `sample_rate`.
async fn decode(path: &Path, sample_rate: u32) -> Result<Vec<f32>> {
    let mut stream = Box::pin(audio_kit::decode_path(
        path.to_path_buf(),
        DecodeOptions::new(sample_rate),
    ));
    let mut pcm = Vec::new();
    while let Some(chunk) = stream.next().await {
        pcm.extend_from_slice(&chunk?);
    }
    Ok(pcm)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The cap is in *seconds*, so it has to be a different sample count at each
    /// of the two rates — trimming both to one length is the mistake this
    /// function exists to make impossible, and it would leave the mel describing
    /// 25 s of a clip whose content vector described 34 s.
    #[test]
    fn the_cap_is_a_duration_not_a_length() {
        let long = vec![0.0f32; 40 * 22_050];
        assert_eq!(trim(&long, CONTENT_SR).len(), 25 * 16_000);
        assert_eq!(trim(&long, 22_050).len(), 25 * 22_050);

        // Under the cap nothing is touched.
        let short = vec![0.0f32; 1234];
        assert_eq!(trim(&short, CONTENT_SR).len(), 1234);
    }
}
