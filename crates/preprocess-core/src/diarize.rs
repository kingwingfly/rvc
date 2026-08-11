//! Keep only the parts of a recording spoken by one target voice.
//!
//! Source separation cannot do this. A vocals stem holds *every* voice in the
//! mixture — the streamer talking and the singer on the background track alike
//! — because separation asks "is this a voice", not "whose". Telling them apart
//! needs speaker identity, which is a different model: [`crate::embed`], over
//! `burn-campplus`.
//!
//! So the two stages compose, and the order is the whole recipe for a stream
//! recorded over somebody else's music:
//!
//! ```sh
//! preprocess separate  raw/     -o vocals/   # music bed off
//! preprocess diarize   vocals/  -o mine/  --reference streamer.wav
//! preprocess clip      mine/    -o dataset/  # per-utterance clips
//! ```
//!
//! # Target-speaker extraction, not clustering
//!
//! A reference clip of the voice to keep is required, and that is the mode
//! worth having first: a user who wants their own speech out of a stream *has*
//! a clean sample of it, and matching against one known voice is far more
//! robust than discovering how many speakers a recording holds. Blind
//! clustering is the follow-up, and its absence is a gap rather than a
//! statement — unlike `seedvc`'s missing `train`.
//!
//! # The output is window-quantised, and it has to be said out loud
//!
//! The recording is cut into fixed windows, each embedded whole and scored by
//! cosine against the reference; runs of windows that clear the threshold become
//! the written segments. **A window that spans a speaker change belongs to
//! whoever dominates it**, so a boundary lands on a hop rather than on the
//! sample where one voice stopped. Two consequences worth expecting: a retained
//! segment can carry a fraction of a second of the other voice at each end, and
//! a one-word interjection shorter than a window will not be removed on its own.
//!
//! The window length is the trade. Too short and the embedding is noisy —
//! CAM++'s context-aware mask pools over 100 of its own 50 Hz frames, so below
//! about 2 s its two context scales collapse into one. Too long and a speaker
//! change hides inside a window. [`DiarizeOptions::window`] is therefore a knob
//! with a measured default rather than a constant.
//!
//! # What the defaults were measured on
//!
//! `examples/timbre` is the harness, and the numbers below are what it printed
//! for one 60 s excerpt of a stream with a song playing under it, separated
//! first: the **vocals** stem stands in for the target speaker's material and
//! the **instrumental** stem — the same seconds with him removed, leaving the
//! music and the singer — for what has to be rejected. Reference: 14 s of the
//! same speaker from a different part of the same recording, separated the same
//! way.
//!
//! | window | target p5 / p25 / median | reject median / p75 / p95 |
//! |---|---|---|
//! | 1.5 s | 0.215 / 0.456 / 0.562 | 0.362 / 0.449 / 0.609 |
//! | **3 s** | 0.374 / 0.568 / 0.628 | 0.411 / 0.464 / 0.674 |
//! | 5 s | 0.488 / 0.625 / 0.660 | 0.430 / 0.510 / 0.722 |
//!
//! **At 1.5 s the two distributions do not separate at all** — the target's
//! lower quartile sits *below* the rejected material's upper one — which is the
//! 2 s pooling window showing up as a number. 5 s separates marginally better
//! than 3 s and quantises every boundary five seconds wide, so 3 s is the
//! default. At 3 s a threshold of 0.55 keeps **80%** of the target's windows and
//! rejects **86%** of the music-and-singer ones.
//!
//! Two honest limits on that, both of which bound it from above. The reference
//! and the material share a session, so the same microphone and the same room —
//! and CAM++'s known failure is keying on the channel rather than the speaker,
//! which flatters a same-session measurement. And the rejected material is a
//! *music bed with a sung voice in it* rather than a second person talking,
//! because no clean recording of a second speaker was available; it is what this
//! stage has to reject in the case it was built for, not a speaker-verification
//! error rate.
//!
//! # There is no level gate, and that is a measurement rather than an oversight
//!
//! A window with no voice in it embeds to something arbitrary, so the obvious
//! guard is an RMS floor. It is not here because the threshold already covers
//! it: over that excerpt the near-silent windows scored **0.20–0.37** against
//! the reference, far below any threshold that keeps the target, and a −40 dBFS
//! floor removed 3 of 58 windows that were being dropped anyway. A floor would
//! be a second knob agreeing with the first. `examples/timbre` keeps one, since
//! *there* it is what separates "nobody spoke" from "the wrong person did".
//!
//! # Per-window filterbank, and never per file
//!
//! Each window is embedded from **its own** filterbank, which is what makes a
//! window's vector comparable to the reference's: the front end subtracts the
//! per-clip mean, so a mean taken over a whole recording of two speakers and a
//! music bed would normalise every window against a distribution none of them
//! has. Computing one filterbank for the file and slicing frames out of it is
//! roughly fifty times cheaper and silently wrong, which is exactly why it is
//! named here — it is what an optimiser reaches for.

use std::path::Path;

use anyhow::{Context, Result};
use audio_kit::write_wav_file;
use futures::stream;

use crate::InputFile;
use crate::embed::{ANALYSIS_SR, SpeakerEmbedder, cosine};

/// How much of a reference clip is read.
///
/// CAM++ pools statistics over time, so a longer reference buys a steadily
/// smaller improvement while costing memory linear in its length — and a user
/// pointing `--reference` at a whole recording by mistake should get an
/// embedding rather than an allocation failure. Documented in the flag's help,
/// because silently ignoring most of a file is the kind of thing that has to be
/// visible where the file is named.
pub const REFERENCE_SECONDS: f32 = 30.0;

/// What to keep, and what to write.
#[derive(Debug, Clone, Copy)]
pub struct DiarizeOptions {
    /// Sample rate of the written audio. Independent of [`ANALYSIS_SR`]: the
    /// speaker model decides nothing about what lands on disk.
    pub sr: u32,
    /// Analysis window, seconds.
    pub window: f32,
    /// Step between consecutive windows, seconds. Below `window` the windows
    /// overlap, which is what keeps a boundary from being a whole window wide.
    pub hop: f32,
    /// Cosine against the reference a window must clear to be kept.
    pub threshold: f32,
    /// Drop any retained run shorter than this, seconds.
    pub min_segment: f32,
}

/// What one file produced.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DiarizeReport {
    /// Segments written.
    pub segments: usize,
    /// Duration decoded from the input.
    pub in_secs: f64,
    /// Duration written out — the rest was somebody else, or nobody.
    pub kept_secs: f64,
    /// Windows scored.
    pub windows: usize,
    /// Windows that cleared the threshold, before short runs were dropped.
    pub kept_windows: usize,
    /// Mean cosine over every window, kept or not. Printed rather than acted
    /// on: it is what tells a user whose output came out empty whether the
    /// threshold was slightly too high or the reference was the wrong person.
    pub mean_score: f32,
}

/// Embed the reference clip: the voice to keep.
///
/// Read whole (up to [`REFERENCE_SECONDS`]) rather than windowed, so the
/// statistics pool sees as much of the speaker as it is given.
pub async fn reference(embedder: &dyn SpeakerEmbedder, path: &Path) -> Result<Vec<f32>> {
    let mut samples = crate::decode_mono(path, ANALYSIS_SR).await?;
    samples.truncate((REFERENCE_SECONDS * ANALYSIS_SR as f32) as usize);
    anyhow::ensure!(
        samples.len() >= embedder.min_samples(),
        "the reference {} is {:.3} s of audio, too short to embed — give a recording of at \
         least {:.2} s of the voice to keep",
        path.display(),
        samples.len() as f64 / ANALYSIS_SR as f64,
        embedder.min_samples() as f64 / ANALYSIS_SR as f64,
    );
    let mut embedded = embedder.embed(&[&samples])?;
    Ok(embedded.remove(0))
}

/// Score one file's windows against the reference and write what matches.
///
/// The file is decoded **twice**: once at [`ANALYSIS_SR`] for the speaker model,
/// once at `opts.sr` for what is written. Resampling one into the other in
/// process would be a second resampler beside ffmpeg's, and segment boundaries
/// derived from 16 kHz sample indices round differently from the ones the output
/// needs — so boundaries are carried in **seconds** and converted once, here.
pub async fn file(
    input: &InputFile,
    opts: &DiarizeOptions,
    embedder: &dyn SpeakerEmbedder,
    target: &[f32],
    output_dir: &Path,
) -> Result<DiarizeReport> {
    let analysis = crate::decode_mono(&input.path, ANALYSIS_SR).await?;
    let in_secs = analysis.len() as f64 / ANALYSIS_SR as f64;

    let win = (opts.window * ANALYSIS_SR as f32) as usize;
    let hop = ((opts.hop * ANALYSIS_SR as f32) as usize).max(1);
    // A trailing partial window is dropped rather than zero-padded: the padding
    // would land inside the per-clip mean the front end subtracts, so a padded
    // window is not the same measurement as a full one.
    let windows: Vec<&[f32]> = (0..)
        .map(|i| i * hop)
        .take_while(|start| start + win <= analysis.len())
        .map(|start| &analysis[start..start + win])
        .collect();
    anyhow::ensure!(
        !windows.is_empty(),
        "{} is {:.2} s, shorter than one --window of {:.2} s — nothing to score",
        input.path.display(),
        in_secs,
        opts.window,
    );

    let scores: Vec<f32> = embedder
        .embed(&windows)?
        .iter()
        .map(|e| cosine(e, target))
        .collect();
    let mean_score = scores.iter().sum::<f32>() / scores.len() as f32;
    let kept_windows = scores.iter().filter(|s| **s >= opts.threshold).count();
    let ranges = segments(&scores, opts, in_secs);

    // The second decode is skipped when nothing matched, which is not a
    // micro-optimisation: a recording of the wrong speaker is the case a user
    // runs a whole corpus through, and paying a full decode per file to write
    // nothing is how "it kept nothing" becomes "it also took an hour".
    let audio = match ranges.is_empty() {
        true => Vec::new(),
        false => crate::decode_mono(&input.path, opts.sr).await?,
    };
    let mut kept_secs = 0.0f64;
    for (i, (start, end)) in ranges.iter().enumerate() {
        let sample = |t: f64| ((t * opts.sr as f64) as usize).min(audio.len());
        let (from, to) = (sample(*start), sample(*end));
        if from >= to {
            continue;
        }
        kept_secs += (to - from) as f64 / opts.sr as f64;
        let out_path = output_dir.join(format!("{}_{i:03}.wav", input.base));
        let out_stream = stream::iter([Ok::<_, audio_kit::AudioError>(audio[from..to].to_vec())]);
        write_wav_file(&out_path, opts.sr, out_stream)
            .await
            .with_context(|| format!("writing {}", out_path.display()))?;
    }

    Ok(DiarizeReport {
        segments: ranges.len(),
        in_secs,
        kept_secs,
        windows: scores.len(),
        kept_windows,
        mean_score,
    })
}

/// Turn per-window scores into the second ranges to write.
///
/// A run of consecutive windows over the threshold becomes one range, from the
/// first window's start to the last window's *end* — so an overlapping hop
/// widens a boundary outward rather than trimming it, and a kept segment is
/// never shorter than the audio the model actually approved of.
///
/// **So seconds kept overstates windows kept**, by up to `window - hop` per
/// segment: a lone window over the threshold at the default 3 s/1 s writes 3 s.
/// That is not a rounding error to trim away — the model did approve of all
/// three seconds — but it is why [`DiarizeReport`] carries the window counts
/// beside the durations, and why the window count is the honest denominator
/// when asking how much of a recording matched.
///
/// Separated from [`file`] because it is the only part of this stage that can be
/// tested without a checkpoint, and because a rounding error here is a
/// misplaced cut rather than a crash.
fn segments(scores: &[f32], opts: &DiarizeOptions, total_secs: f64) -> Vec<(f64, f64)> {
    let (window, hop) = (opts.window as f64, opts.hop as f64);
    let mut out: Vec<(f64, f64)> = Vec::new();
    let mut prev: Option<usize> = None;
    for (i, _) in scores
        .iter()
        .enumerate()
        .filter(|(_, s)| **s >= opts.threshold)
    {
        let (start, end) = (
            (i as f64 * hop).min(total_secs),
            (i as f64 * hop + window).min(total_secs),
        );
        match out.last_mut() {
            // Consecutive kept windows are one segment. The test is on the
            // **index**, not on whether the times overlap: with a window wider
            // than the hop every window overlaps its neighbour's span, so a
            // time-based test merges straight across a window that was
            // rejected — and a rejected window is exactly where the other
            // speaker was. Merging there hands their audio back in the kept
            // output, which is the whole thing this stage exists to prevent.
            Some(last) if prev == Some(i - 1) => last.1 = end.max(last.1),
            _ => out.push((start, end)),
        }
        prev = Some(i);
    }
    out.retain(|(start, end)| end - start >= opts.min_segment as f64);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn opts(threshold: f32, min_segment: f32) -> DiarizeOptions {
        DiarizeOptions {
            sr: 48_000,
            window: 2.0,
            hop: 1.0,
            threshold,
            min_segment,
        }
    }

    /// The arithmetic the whole stage's output is built out of: a run of kept
    /// windows spans from the first one's start to the last one's end, and the
    /// overlap between windows must not open a gap between two runs that touch.
    #[test]
    fn a_run_of_windows_becomes_one_segment_covering_all_of_them() {
        // Windows at 0, 1, 2, 3, 4 s, each 2 s long; the last one ends at 6 s.
        let scores = [0.9, 0.9, 0.1, 0.9, 0.9];
        let got = segments(&scores, &opts(0.5, 0.0), 6.0);
        assert_eq!(got, vec![(0.0, 3.0), (3.0, 6.0)]);

        // A single kept window in the middle is exactly one window wide.
        let got = segments(&[0.1, 0.9, 0.1], &opts(0.5, 0.0), 6.0);
        assert_eq!(got, vec![(1.0, 3.0)]);
    }

    /// Nothing may be reported past the end of the file: the last window ends
    /// `window` seconds after its start, which is beyond a recording whose tail
    /// did not fill a whole window.
    #[test]
    fn segments_are_clamped_to_the_recording() {
        let got = segments(&[0.9, 0.9], &opts(0.5, 0.0), 2.5);
        assert_eq!(got, vec![(0.0, 2.5)]);
    }

    /// `--min-segment` drops a fragment rather than shrinking it, and it is
    /// applied *after* merging — otherwise two adjacent windows of one second
    /// each would be discarded before they had a chance to become two seconds.
    #[test]
    fn short_runs_are_dropped_after_merging_and_not_before() {
        let scores = [0.9, 0.1, 0.1, 0.9, 0.9];
        // Merged: (0, 2) and (3, 6). A 2.5 s floor keeps only the second.
        assert_eq!(segments(&scores, &opts(0.5, 2.5), 6.0), vec![(3.0, 6.0)]);
        assert_eq!(segments(&scores, &opts(0.5, 2.0), 6.0).len(), 2);
    }

    /// A file nobody's voice matches writes nothing, rather than one empty
    /// segment or the whole recording.
    #[test]
    fn nothing_over_the_threshold_writes_nothing() {
        assert!(segments(&[0.1, 0.2, 0.3], &opts(0.5, 0.0), 6.0).is_empty());
    }
}
