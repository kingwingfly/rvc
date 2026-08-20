//! Strip the silence off the head and tail of a recording.
//!
//! One file in, one `<base>.wav` out, and **never a split**: everything between
//! the first voiced sample and the last is kept exactly as it was, pauses
//! included. That is the difference from [`crate::clip`], which cuts the same
//! recording into one file per sentence — the two share every line of silence
//! detection and differ only in what they do with the gaps in the middle.
//!
//! Which is why this reuses [`audio_kit::slice`] rather than looking for quiet
//! itself. A second silence detector would be a second set of answers to
//! "where does this sentence end", and the hysteresis that keeps a soft breathy
//! tail inside a clip is the whole reason the first one is written the way it
//! is.

use std::path::Path;

use anyhow::{Context, Result};
use audio_kit::{SliceOptions, write_wav_file};
use futures::stream;

use crate::InputFile;

/// What to trim, and what counts as silence.
#[derive(Debug, Clone, Copy)]
pub struct TrimOptions {
    /// Sample rate to decode at and write out.
    pub sr: u32,
    /// Energy floor in dBFS, and how much bordering quiet to keep at each edge.
    /// The gap knobs are set from the file itself — see [`file`].
    pub silence_db: f32,
    /// Edge-pad by up to this many seconds of the silence being removed.
    pub pad: f32,
    /// Measure the floor from the recording instead of using `silence_db`.
    pub measure_floor: bool,
}

/// What one file produced.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TrimReport {
    /// Duration decoded from the input.
    pub in_secs: f64,
    /// Duration written out.
    pub out_secs: f64,
    /// Seconds removed from the head, and from the tail.
    pub head_secs: f64,
    pub tail_secs: f64,
    /// The floor actually used, in dBFS — worth reporting because
    /// [`TrimOptions::measure_floor`] means it is not necessarily the one asked
    /// for.
    pub silence_db: f32,
    /// Whether the recording had no audible content at all, in which case it is
    /// written through unchanged rather than emptied.
    pub silent: bool,
}

/// Trim one file into the output directory.
pub async fn file(input: &InputFile, opts: &TrimOptions, output_dir: &Path) -> Result<TrimReport> {
    let samples = crate::decode_mono(&input.path, opts.sr).await?;
    let in_secs = samples.len() as f64 / opts.sr as f64;

    let mut slice = SliceOptions {
        silence_db: opts.silence_db,
        // Longer than the recording, so no interior gap can ever be a cut
        // point: this stage strips the edges and must not split. A finite
        // number rather than an infinity because the slicer does arithmetic on
        // this (`max(min_silence, 2 * pad)`, converted to a frame count), and
        // an infinity saturates rather than computing.
        min_silence: in_secs as f32 + 1.0,
        // Nothing to discard and nothing to cap: there is exactly one segment
        // by construction, and dropping it for being short would delete the
        // file.
        min_clip: 0.0,
        max_clip: 0.0,
        pad: opts.pad,
    };
    if opts.measure_floor {
        slice = slice.with_measured_floor(&samples, opts.sr);
    }

    let segments = audio_kit::slice(&samples, opts.sr, &slice);
    // A recording with nothing above the floor has no head and no tail to tell
    // apart, so it passes through whole. Writing the empty result instead would
    // silently delete a file whose floor was simply set too high.
    let (start, end) = match (segments.first(), segments.last()) {
        (Some(first), Some(last)) => (first.0, last.1),
        _ => (0, samples.len()),
    };
    let silent = segments.is_empty();

    let head_secs = start as f64 / opts.sr as f64;
    let tail_secs = (samples.len() - end) as f64 / opts.sr as f64;
    let out = samples[start..end].to_vec();
    let out_secs = out.len() as f64 / opts.sr as f64;

    let out_path = output_dir.join(format!("{}.wav", input.base));
    let out_stream = stream::iter([Ok::<_, audio_kit::AudioError>(out)]);
    write_wav_file(&out_path, opts.sr, out_stream)
        .await
        .with_context(|| format!("writing {}", out_path.display()))?;

    Ok(TrimReport {
        in_secs,
        out_secs,
        head_secs,
        tail_secs,
        silence_db: slice.silence_db,
        silent,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const SR: u32 = 48_000;

    fn voiced(secs: f32) -> Vec<f32> {
        let n = (secs * SR as f32) as usize;
        (0..n)
            .map(|i| (0.5 * (std::f64::consts::TAU * 220.0 * i as f64 / SR as f64).sin()) as f32)
            .collect()
    }

    fn silence(secs: f32) -> Vec<f32> {
        vec![0.0; (secs * SR as f32) as usize]
    }

    fn scratch(name: &str) -> std::path::PathBuf {
        let dir =
            std::env::temp_dir().join(format!("preprocess-core-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("create scratch dir");
        dir
    }

    async fn write_tone(path: &Path, samples: Vec<f32>) {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).expect("create parent");
        }
        let s = stream::iter([Ok::<_, audio_kit::AudioError>(samples)]);
        write_wav_file(path, SR, s).await.expect("write tone");
    }

    fn options() -> TrimOptions {
        TrimOptions {
            sr: SR,
            silence_db: -40.0,
            pad: 0.15,
            measure_floor: false,
        }
    }

    /// The defining property, and the one that separates this stage from
    /// `clip`: a second of dead air *between* two sentences is interior and
    /// stays, while the same second at either end goes.
    #[tokio::test]
    async fn the_edges_go_and_the_middle_survives_whole() {
        let dir = scratch("trim-middle");
        let mut sig = silence(2.0);
        sig.extend(voiced(1.0));
        sig.extend(silence(1.0));
        sig.extend(voiced(1.0));
        sig.extend(silence(3.0));
        write_tone(&dir.join("take.wav"), sig).await;

        let out = dir.join("out");
        std::fs::create_dir_all(&out).expect("create out");
        let files = crate::plan(std::slice::from_ref(&dir), &out).expect("plan");
        let report = file(&files[0], &options(), &out).await.expect("trim");

        assert!(!report.silent);
        assert!((report.in_secs - 8.0).abs() < 0.05, "{report:?}");
        // 1 s + the interior 1 s + 1 s, plus 0.15 s of pad kept at each end.
        assert!((report.out_secs - 3.3).abs() < 0.05, "{report:?}");
        assert!((report.head_secs - 1.85).abs() < 0.05, "{report:?}");
        assert!((report.tail_secs - 2.85).abs() < 0.05, "{report:?}");
    }

    /// A file with nothing above the floor must come out whole. Emptying it
    /// would be the same observable event as deleting it, and the likeliest
    /// cause is a `--silence-db` set above the recording rather than a file
    /// with no content.
    #[tokio::test]
    async fn a_recording_below_the_floor_passes_through_whole() {
        let dir = scratch("trim-silent");
        write_tone(&dir.join("quiet.wav"), silence(2.0)).await;

        let out = dir.join("out");
        std::fs::create_dir_all(&out).expect("create out");
        let files = crate::plan(std::slice::from_ref(&dir), &out).expect("plan");
        let report = file(&files[0], &options(), &out).await.expect("trim");

        assert!(report.silent, "{report:?}");
        assert_eq!(report.out_secs, report.in_secs, "{report:?}");
        assert_eq!(report.head_secs, 0.0);
        assert_eq!(report.tail_secs, 0.0);
    }

    /// The measured floor is opt-in, and this is what it buys: room tone
    /// *above* the fixed -40 dBFS floor, which a fixed floor calls speech and
    /// therefore refuses to trim at all.
    #[tokio::test]
    async fn a_measured_floor_trims_a_recording_a_fixed_one_cannot() {
        let dir = scratch("trim-measured");
        // Room tone at ~-28 dBFS: comfortably above the fixed floor, so the
        // fixed pass sees no silence anywhere.
        let tone: Vec<f32> = (0..(SR as usize * 2))
            .map(|i| (0.04 * (std::f64::consts::TAU * 90.0 * i as f64 / SR as f64).sin()) as f32)
            .collect();
        let mut sig = tone.clone();
        sig.extend(voiced(2.0));
        sig.extend(tone);
        write_tone(&dir.join("roomy.wav"), sig).await;

        let out = dir.join("out");
        std::fs::create_dir_all(&out).expect("create out");
        let files = crate::plan(std::slice::from_ref(&dir), &out).expect("plan");

        let fixed = file(&files[0], &options(), &out).await.expect("trim");
        assert_eq!(fixed.head_secs, 0.0, "a fixed floor sees no silence here");

        let measured = TrimOptions {
            measure_floor: true,
            ..options()
        };
        let report = file(&files[0], &measured, &out).await.expect("trim");
        assert!(report.silence_db > -40.0, "floor not measured: {report:?}");
        assert!(report.head_secs > 1.5, "{report:?}");
        assert!(report.tail_secs > 1.5, "{report:?}");
    }

    /// A sine of `hz` at half scale, at whatever rate the fixture runs at.
    fn tone_at(rate: u32, secs: f32, hz: f64) -> Vec<f32> {
        let n = (secs * rate as f32) as usize;
        (0..n)
            .map(|i| (0.5 * (std::f64::consts::TAU * hz * i as f64 / f64::from(rate)).sin()) as f32)
            .collect()
    }

    fn peak(samples: &[f32]) -> f32 {
        samples.iter().fold(0.0f32, |m, x| m.max(x.abs()))
    }

    /// The three tests above assert *durations*, and a boundary a hop out — or
    /// a head and a tail read the wrong way round — gives the same durations on
    /// a symmetric fixture. This one decodes what was written and compares it
    /// sample for sample against the input at the offset the report names,
    /// which is the only reading that tells "3.3 s of the right audio" from
    /// "3.3 s".
    #[tokio::test]
    async fn the_written_file_is_the_input_between_the_head_and_tail_it_reports() {
        let dir = scratch("trim-exact");
        // Asymmetric at both ends and two different tones inside, so a slice
        // taken from the wrong end shows up in the samples as well as in the
        // arithmetic.
        let mut sig = silence(1.0);
        sig.extend(voiced(1.0));
        sig.extend(silence(0.5));
        sig.extend(tone_at(SR, 2.0, 660.0));
        sig.extend(silence(3.0));
        write_tone(&dir.join("take.wav"), sig).await;

        let out = dir.join("out");
        std::fs::create_dir_all(&out).expect("create out");
        let files = crate::plan(std::slice::from_ref(&dir), &out).expect("plan");
        let report = file(&files[0], &options(), &out).await.expect("trim");

        let input = crate::decode_mono(&files[0].path, SR)
            .await
            .expect("decode input");
        let written = crate::decode_mono(&out.join("take.wav"), SR)
            .await
            .expect("decode output");

        let head = (report.head_secs * f64::from(SR)).round() as usize;
        let tail = (report.tail_secs * f64::from(SR)).round() as usize;
        assert_eq!(
            head + written.len() + tail,
            input.len(),
            "the report must account for every sample: {report:?}"
        );
        assert_eq!(
            written,
            input[head..input.len() - tail],
            "the file is not the interior the report describes"
        );

        // ...and it is the *right* interior: everything discarded is under the
        // floor, and the loudest sample in the recording is inside what stayed.
        assert!(
            peak(&input[..head]) < 0.01,
            "the head that went was not silence: {}",
            peak(&input[..head])
        );
        assert!(peak(&input[input.len() - tail..]) < 0.01);
        assert!((peak(&written) - peak(&input)).abs() < 1e-9);
    }

    /// The stage strips edges and never splits, and this is the case that
    /// would break a stage that reached for a plausible constant instead: a
    /// twenty-second pause is longer than any `--min-silence` a user would
    /// name, and it still has to survive into the output.
    ///
    /// It is guarded twice over, which is worth knowing before changing
    /// either half: [`file`] sets `min_silence` past the recording's own
    /// length so the slicer cannot cut in the middle at all, *and* it writes
    /// `first.0 .. last.1`, so an interior cut would be spanned even if one
    /// happened. The second guard holds only while `min_clip` is 0 — with a
    /// floor under a segment's length, a split could drop the piece that
    /// carries `last.1` and the tail would be trimmed into the speech.
    #[tokio::test]
    async fn an_interior_gap_longer_than_any_sentence_survives() {
        // 16 kHz, written and read at its own rate so nothing resamples, and a
        // fixture this long stays a few megabytes rather than tens.
        const RATE: u32 = 16_000;
        let dir = scratch("trim-long-gap");
        let mut sig = vec![0.0f32; 2 * RATE as usize];
        sig.extend(tone_at(RATE, 30.0, 220.0));
        sig.extend(std::iter::repeat_n(0.0, 20 * RATE as usize));
        sig.extend(tone_at(RATE, 30.0, 440.0));
        sig.extend(std::iter::repeat_n(0.0, 2 * RATE as usize));

        let path = dir.join("long.wav");
        let s = stream::iter([Ok::<_, audio_kit::AudioError>(sig)]);
        write_wav_file(&path, RATE, s).await.expect("write fixture");

        let out = dir.join("out");
        std::fs::create_dir_all(&out).expect("create out");
        let files = crate::plan(std::slice::from_ref(&dir), &out).expect("plan");
        let opts = TrimOptions {
            sr: RATE,
            ..options()
        };
        let report = file(&files[0], &opts, &out).await.expect("trim");

        assert!(!report.silent, "{report:?}");
        assert!((report.in_secs - 84.0).abs() < 0.05, "{report:?}");
        // 30 + 20 + 30, plus 0.15 s of pad kept at each end.
        assert!((report.out_secs - 80.3).abs() < 0.05, "{report:?}");
        assert!((report.head_secs - 1.85).abs() < 0.05, "{report:?}");
        assert!((report.tail_secs - 1.85).abs() < 0.05, "{report:?}");

        // The written file still has the pause in it: a stage that split here
        // would have written the first sentence alone, or the two with the gap
        // squeezed out, and either shows up as missing silence.
        let written = crate::decode_mono(&out.join("long.wav"), RATE)
            .await
            .expect("decode output");
        let quiet = written.iter().filter(|x| x.abs() < 0.01).count();
        assert!(
            quiet >= 19 * RATE as usize,
            "only {:.2} s of the 20 s pause survived",
            quiet as f64 / f64::from(RATE)
        );
    }
}
