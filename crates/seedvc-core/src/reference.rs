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
//! # 25 s is the default cap, and it is the dial worth turning
//!
//! Upstream trims to `ref_audio[:sr * 25]` and [`REFERENCE_SECONDS`] is that
//! number. It is a **default rather than a constant**, because it is the most
//! consequential length in the engine: the timbre vector is a pooled average, so
//! past a few seconds more audio buys very little, while what it does buy is a
//! **shorter source chunk** — the reference's mel and the source's share one
//! 30 s context window, see [`crate::convert`], where that arithmetic lives. At
//! 25 s a chunk carries under 5 s of source; at 5 s it carries nearly 25. A
//! caller handing a long clip to the default therefore pays five times the
//! chunks and five times the seams for a timbre vector that saturated seconds
//! in, and until this was a parameter the only escape was trimming the file.
//!
//! The cap can only ever *shorten* what is read, so [`MAX_REFERENCE_SECONDS`] is
//! where raising it stops meaning anything: the content encoder refuses a clip
//! past its own window rather than truncating it, so a longer cap reads no more
//! audio and only moves where the refusal comes from.

use std::path::Path;

use audio_kit::DecodeOptions;
use burn_seedvc::content::WINDOW_SAMPLES;
use futures::StreamExt;

use crate::error::{Error, Result};
use crate::model::{CONTENT_SR, Model, Reference};

/// Longest stretch of a reference clip that is read by default, in seconds —
/// upstream's `ref_audio[:sr * 25]`, and the value every `analyse` caller passes
/// unless a user says otherwise.
pub const REFERENCE_SECONDS: f32 = 25.0;

/// The longest cap that means anything — Whisper's own window, in seconds.
///
/// Derived from [`WINDOW_SAMPLES`] rather than written as 30, because it is the
/// same 30 s three times over in this engine (the content encoder's window, the
/// transformer's context, the mel prefix's share of it) and only one of them is
/// this bound. [`Model::analyse`] refuses a clip past it rather than truncating
/// one, so a cap above this cannot read more audio — it can only defer that
/// refusal until after a gigabyte of weights has loaded.
pub const MAX_REFERENCE_SECONDS: f32 = WINDOW_SAMPLES as f32 / CONTENT_SR as f32;

/// Decode a reference recording at both rates and analyse it.
///
/// `seconds` caps how much of the clip is read; [`REFERENCE_SECONDS`] is
/// upstream's own value and what a caller with no opinion passes.
///
/// Anything ffmpeg opens will do, at any rate and channel count; what comes back
/// is the [`Reference`] every conversion against this speaker is conditioned on,
/// so analysing once and converting many times is the intended shape.
pub async fn analyse(
    model: &dyn Model,
    path: impl AsRef<Path>,
    seconds: f32,
) -> Result<Reference> {
    // Checked before the two decodes rather than inside `analyse_pcm`, so a cap
    // that cannot be used costs a message and not two passes of ffmpeg.
    checked(seconds)?;
    let path = path.as_ref();
    let content = decode(path, CONTENT_SR).await?;
    let mel = decode(path, model.config().sample_rate).await?;
    analyse_pcm(model, &content, &mel, seconds)
}

/// Analyse a reference already decoded at both rates.
///
/// `content` is mono `f32` at [`CONTENT_SR`] and `mel` is the **same clip** at
/// [`SeedVcConfig::sample_rate`](burn_seedvc::SeedVcConfig::sample_rate). Both
/// are trimmed to `seconds` here rather than by the caller, so the cap cannot be
/// applied to one and forgotten on the other — the two lengths disagreeing is
/// not an error the model can detect, since neither tells it anything about the
/// other. That is also why the cap is one argument and not two: it is a
/// *duration*, and the sample count it becomes differs at each rate.
pub fn analyse_pcm(
    model: &dyn Model,
    content: &[f32],
    mel: &[f32],
    seconds: f32,
) -> Result<Reference> {
    checked(seconds)?;
    let cfg = model.config();
    let content = trim(content, CONTENT_SR, seconds);
    let mel = trim(mel, cfg.sample_rate, seconds);

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

/// Reject a cap the trim cannot use, in terms of what it is for.
///
/// The upper end is deliberately *not* checked here: a cap above
/// [`MAX_REFERENCE_SECONDS`] is only a problem when the clip is actually that
/// long, and [`Model::analyse`] names that case exactly — with the clip's own
/// duration in it, which this cannot see. A caller that can refuse earlier
/// should, and the binary does.
fn checked(seconds: f32) -> Result<()> {
    if !(seconds.is_finite() && seconds > 0.0) {
        return Err(Error::Reference(format!(
            "the reference cap is {seconds} s — it is the longest stretch of the clip that is \
             read, so it has to be a positive, finite number of seconds"
        )));
    }
    Ok(())
}

/// The cap, applied at whichever rate the samples are at.
fn trim(pcm: &[f32], sample_rate: u32, seconds: f32) -> &[f32] {
    // `round`, not a truncating cast, for the reason `rvc`'s geometry flags give
    // for theirs: a value a user typed is rarely exact in `f32`, and truncation
    // would land a sample short of the duration they asked for. It changes
    // nothing at the default — 25.0 times either rate is exact — which is what
    // keeps an unspecified cap producing byte-for-byte what it always did.
    let cap = (seconds * sample_rate as f32).round() as usize;
    &pcm[..pcm.len().min(cap)]
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
        assert_eq!(
            trim(&long, CONTENT_SR, REFERENCE_SECONDS).len(),
            25 * 16_000
        );
        assert_eq!(trim(&long, 22_050, REFERENCE_SECONDS).len(), 25 * 22_050);

        // Under the cap nothing is touched.
        let short = vec![0.0f32; 1234];
        assert_eq!(trim(&short, CONTENT_SR, REFERENCE_SECONDS).len(), 1234);
    }

    /// The default did not move when it became a parameter.
    ///
    /// A construction rather than a measurement: the arithmetic is the same
    /// multiply it always was and 25.0 is exact at both rates, so an explicit
    /// 25 has to give byte-identical slices to the constant. This is what pins
    /// that, and it is the property the whole flag rests on.
    #[test]
    fn an_explicit_default_is_the_default() {
        let long = vec![0.0f32; 40 * 22_050];
        for rate in [CONTENT_SR, 22_050] {
            assert_eq!(
                trim(&long, rate, 25.0).len(),
                trim(&long, rate, REFERENCE_SECONDS).len(),
                "{rate} Hz",
            );
        }
    }

    /// A shorter cap reads less of the clip at *both* rates, which is the whole
    /// point: the mel prefix shrinks with the content vector, so the source's
    /// share of the shared window grows by the same duration.
    #[test]
    fn a_shorter_cap_reads_less_at_both_rates() {
        let long = vec![0.0f32; 40 * 22_050];
        assert_eq!(trim(&long, CONTENT_SR, 5.0).len(), 5 * 16_000);
        assert_eq!(trim(&long, 22_050, 5.0).len(), 5 * 22_050);

        // Fractional too, since the flag is in seconds and nothing rounds it to
        // whole ones — 7.5 s is 165 375 samples at 22.05 kHz and not 165 374.
        assert_eq!(trim(&long, 22_050, 7.5).len(), 165_375);
    }

    /// A cap that is not a positive number of seconds is refused in terms of
    /// what the cap is, rather than surfacing later as a clip that "decoded to
    /// no audio at all" — which is what a zero or a `NaN` would otherwise look
    /// like once the trim had emptied both slices.
    #[test]
    fn a_cap_that_is_not_a_duration_is_refused() {
        for seconds in [0.0, -1.0, f32::NAN, f32::INFINITY] {
            let err = checked(seconds)
                .expect_err("{seconds} is not a duration")
                .to_string();
            assert!(err.contains("positive, finite"), "{err}");
        }
        assert!(checked(REFERENCE_SECONDS).is_ok());
    }

    /// The upper bound is the content encoder's window and is derived from it,
    /// so it cannot drift from the check that actually enforces it.
    #[test]
    fn the_ceiling_is_whispers_own_window() {
        assert_eq!(MAX_REFERENCE_SECONDS, 30.0);
        assert!(REFERENCE_SECONDS < MAX_REFERENCE_SECONDS);
    }
}
