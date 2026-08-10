//! Strip steady background hiss from a recording.
//!
//! One file in, one `<base>.wav` out — the same [`audio_kit::Denoiser`] the
//! voice-conversion filter runs on its output, pointed at a corpus instead.
//! Which is the reason it is worth having as a stage: hiss that survives into
//! the training clips is hiss the model learns to reproduce, and removing it
//! once beforehand is cheaper and better than removing it from every
//! conversion afterwards.
//!
//! `anlmdn` averages each short patch of audio with self-similar patches found
//! nearby in time, so steady hiss averages away while ever-changing breath
//! texture — which is spectrally indistinguishable from hiss, and is exactly
//! what this toolkit exists to preserve — has few close matches and survives.
//! A spectral-subtraction de-noiser keyed on level and frequency does not have
//! that property.

use std::path::Path;

use anyhow::{Context, Result};
use audio_kit::{DenoiseParams, Denoiser, write_wav_file};
use futures::stream;

use crate::InputFile;

/// What to de-hiss, and how hard.
#[derive(Debug, Clone, Copy)]
pub struct DenoiseOptions {
    /// Sample rate to decode at and write out. Hiss is broadband, so a stage
    /// that resampled on the way through would change what it is removing.
    pub sr: u32,
    /// The `anlmdn` knobs.
    pub params: DenoiseParams,
}

/// What one file produced.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DenoiseReport {
    /// Duration decoded from the input.
    pub in_secs: f64,
    /// Duration written out. Not necessarily equal to `in_secs`: the filter has
    /// a look-ahead, and the drained tail is what makes up the difference.
    pub out_secs: f64,
}

/// De-hiss one file into the output directory.
///
/// The whole file goes through one filter graph, and the graph is dropped
/// afterwards, so `anlmdn`'s research window never spans two recordings.
pub async fn file(
    input: &InputFile,
    opts: &DenoiseOptions,
    output_dir: &Path,
) -> Result<DenoiseReport> {
    let samples = crate::decode_mono(&input.path, opts.sr).await?;
    let in_secs = samples.len() as f64 / opts.sr as f64;

    let mut denoiser = Denoiser::new(opts.sr, opts.params);
    let mut out = denoiser.process(&samples);
    out.extend(denoiser.flush());
    let out_secs = out.len() as f64 / opts.sr as f64;

    let out_path = output_dir.join(format!("{}.wav", input.base));
    let out_stream = stream::iter([Ok::<_, audio_kit::AudioError>(out)]);
    write_wav_file(&out_path, opts.sr, out_stream)
        .await
        .with_context(|| format!("writing {}", out_path.display()))?;

    Ok(DenoiseReport { in_secs, out_secs })
}
