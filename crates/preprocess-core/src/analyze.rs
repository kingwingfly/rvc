//! Report what a recording *is*, so the other stages can be driven from
//! evidence instead of from guesses.
//!
//! Every other stage here has at least one knob whose right value depends on
//! the recording — `clip`'s `--silence-db` above all — and nothing showed the
//! user what the recording was. This stage is that missing half: it writes no
//! audio, loads no model, and reports the properties the other stages' defaults
//! can only guess at.
//!
//! # It measures nothing of its own
//!
//! The noise floor is [`audio_kit::noise_floor`], the derived threshold is
//! [`audio_kit::NoiseFloor::silence_db`], and the clip counts come from running
//! [`audio_kit::slice`] rather than from a model of it. That is deliberate on
//! both counts. A second definition of "the noise floor" is exactly what was
//! removed when that measurement was hoisted into `audio-kit`, and a stage that
//! *predicted* `clip`'s output instead of running its slicer would drift from
//! it the first time either changed — while still printing numbers that looked
//! like the truth.
//!
//! # Two floors, because a suggestion has to be checked
//!
//! [`FileReport`] holds the slicer's verdict twice: at the floor the user asked
//! for ([`SliceOutcome::silence_db`] = `--silence-db`), and at the one measured
//! from the recording itself. Reporting only the second would make the
//! suggestion an assertion; running the slicer at it is what turns it into a
//! measurement, and it is the only way to notice the case that matters most —
//! **a recording with no dead air at any floor**. Speech over a continuous
//! music bed is that shape: the floor sits under the music, the slicer finds
//! nothing quiet, and no `--silence-db` can rescue it because the quiet is not
//! there to find. [`FileReport::continuous`] is that condition, and a suggested
//! floor is withheld when it holds everywhere ([`Summary::suggested_silence_db`]),
//! because a floor that cannot work is worse than no floor at all. What to do
//! about it is `separate`, which this crate also has.
//!
//! # There is no loudness (LUFS) here, and that is a decision
//!
//! ffmpeg's `loudnorm` computes exactly what a corpus wants to know, but it
//! reports it through `av_log` rather than through the samples, so reading it
//! from [`audio_kit::AudioFilter`] would mean installing a **global** ffmpeg log
//! callback and parsing stderr. Its `ebur128` sibling attaches the same numbers
//! as frame metadata, which `AudioFilter` discards and does not expose. Either
//! route is a change to a shared crate for one column, so neither was taken, and
//! hand-rolling BS.1770 here would be a third definition of loudness in a
//! toolkit that already has [`NoiseFloor`](audio_kit::NoiseFloor). Peak, floor
//! and SNR answer "is this corpus usable" without it.

use std::path::{Path, PathBuf};

use anyhow::Result;
use audio_kit::{NoiseFloor, SliceOptions};

use crate::InputFile;

/// Full scale, near enough: a sample at or above this counts as clipped.
///
/// Not `1.0`. 16-bit full scale decodes to 32767/32768 = 0.99997, so an exact
/// test would report nothing at all on the commonest source of clipping there
/// is.
const CLIP_LEVEL: f32 = 0.999;

/// Digital silence, for a peak that is exactly zero. The same floor
/// `audio_kit`'s own signal-to-floor ratio clamps at, so a silent file reports
/// a large negative number rather than `-inf` through every formatter
/// downstream.
const SILENT_DB: f32 = -120.0;

/// How long a recording has to be before "no dead air in it" says anything.
///
/// Under this, a gapless file is just a short one — and the whole *output* of
/// `clip` is gapless files of a few seconds, so a corpus that has already been
/// sliced would otherwise report every one of its clips as pathological. See
/// [`FileReport::continuous`] for the measured numbers.
const CONTINUOUS_SECS: f64 = 15.0;

/// Below this fraction of dead air, at the best of the two floors tried, a
/// recording has nothing for a silence-based slicer to cut on.
const CONTINUOUS_SILENCE_RATIO: f64 = 0.02;

/// A file with at least this many samples at full scale is worth naming.
///
/// ~2 ms at 48 kHz. One clipped sample is a rounding artefact of whatever wrote
/// the file; two milliseconds of them is a recording made too hot, which is
/// the thing a user can still act on.
const CLIPPED_SAMPLES_WORTH_SAYING: usize = 100;

/// Upper bounds of the clip-length buckets, in seconds.
///
/// The first is `--min-clip`'s default, so the first bucket is what the slicer
/// would discard outright; the last is where a "clip" has stopped being an
/// utterance and is a stretch of unsplit recording.
pub const DURATION_BUCKETS: [f64; 6] = [1.0, 2.0, 4.0, 8.0, 16.0, 30.0];

/// What to measure, and what to measure it against.
#[derive(Debug, Clone, Copy)]
pub struct AnalyzeOptions {
    /// Sample rate to decode at. Nothing is written, so this only decides the
    /// grid everything is measured on.
    pub sr: u32,
    /// The slicer settings to report against — `clip`'s own, so the clip counts
    /// and the histogram are what that stage would actually produce.
    pub slice: SliceOptions,
}

/// Clip lengths in [`DURATION_BUCKETS`], plus a final "everything longer" bin.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Histogram {
    pub counts: [usize; DURATION_BUCKETS.len() + 1],
}

impl Histogram {
    fn add(&mut self, secs: f64) {
        let i = DURATION_BUCKETS
            .iter()
            .position(|&b| secs < b)
            .unwrap_or(DURATION_BUCKETS.len());
        self.counts[i] += 1;
    }

    /// Fold another file's clips into this one.
    pub fn merge(&mut self, other: &Self) {
        for (a, b) in self.counts.iter_mut().zip(other.counts) {
            *a += b;
        }
    }

    /// How bucket `i` reads: `<1s`, `1-2s`, … `>=30s`.
    pub fn label(i: usize) -> String {
        match i {
            0 => format!("<{}s", DURATION_BUCKETS[0]),
            i if i < DURATION_BUCKETS.len() => {
                format!("{}-{}s", DURATION_BUCKETS[i - 1], DURATION_BUCKETS[i])
            }
            _ => format!(">={}s", DURATION_BUCKETS[DURATION_BUCKETS.len() - 1]),
        }
    }
}

/// What the slicer does to one recording at one floor.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SliceOutcome {
    /// The floor this was cut at, in dBFS.
    pub silence_db: f32,
    /// Clips `clip` would write.
    pub clips: usize,
    /// Seconds inside those clips.
    pub kept_secs: f64,
    /// Fraction of the recording that is *not* inside a clip: between-sentence
    /// dead air, plus any fragment `--min-clip` discarded.
    pub silence_ratio: f64,
    /// Their lengths.
    pub durations: Histogram,
}

/// What one file measured.
#[derive(Debug, Clone)]
pub struct FileReport {
    pub path: PathBuf,
    pub secs: f64,
    /// The recording's own quiet and loud percentiles. `None` when there is too
    /// little audio to take them over (under ~80 ms).
    pub floor: Option<NoiseFloor>,
    /// Loudest sample, in dBFS.
    pub peak_db: f32,
    /// Samples at or above [`CLIP_LEVEL`].
    ///
    /// Counted on the decoded stream at [`AnalyzeOptions::sr`], so a file that
    /// was resampled on the way in can report a few more than were written:
    /// intersample peaks in the source become real ones at the new rate. That
    /// makes a handful of samples uninformative and a thousand of them real.
    pub clipped: usize,
    /// What `clip` would produce at the requested floor.
    pub at_requested: SliceOutcome,
    /// …and at the floor measured from this recording. `None` when nothing
    /// could be measured, or when the measurement lands on the requested floor
    /// anyway.
    pub at_measured: Option<SliceOutcome>,
}

impl FileReport {
    /// The floor this recording's own frames suggest, if it had enough to
    /// measure over.
    pub fn suggested_silence_db(&self) -> Option<f32> {
        self.floor.map(|f| f.silence_db())
    }

    /// The most dead air any floor tried found, as a fraction.
    pub fn best_silence_ratio(&self) -> f64 {
        self.at_measured
            .map(|m| m.silence_ratio.max(self.at_requested.silence_ratio))
            .unwrap_or(self.at_requested.silence_ratio)
    }

    /// Whether this is a recording a silence-based slicer cannot cut: long
    /// enough to hold several sentences, and holding no dead air at either
    /// floor.
    ///
    /// The two constants are what separate that from an already-sliced corpus,
    /// which is gapless for a good reason. Measured on this repository's own
    /// `dataset/` (92 clips `clip` wrote) against a 60 s excerpt of speech over
    /// music: see the module docs of `preprocess-cli`'s `analyze` command for
    /// the numbers both sides landed at.
    ///
    /// It says nothing about the *cause*. Absence of silence is measurable;
    /// whether it is music, a fan, or somebody who never pauses is not, which
    /// is why the advisory this drives is worded as a question rather than a
    /// diagnosis.
    pub fn continuous(&self) -> bool {
        self.secs >= CONTINUOUS_SECS && self.best_silence_ratio() < CONTINUOUS_SILENCE_RATIO
    }

    /// Whether this file's clipping is worth naming rather than counting.
    pub fn clipping_worth_saying(&self) -> bool {
        self.clipped >= CLIPPED_SAMPLES_WORTH_SAYING
    }
}

/// A median and the spread around it, over one column of a corpus.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Stat {
    pub median: f32,
    pub min: f32,
    pub max: f32,
}

impl Stat {
    /// `None` for an empty column — a corpus where nothing could be measured.
    fn of(mut values: Vec<f32>) -> Option<Self> {
        if values.is_empty() {
            return None;
        }
        values.sort_by(f32::total_cmp);
        Some(Self {
            median: values[values.len() / 2],
            min: values[0],
            max: values[values.len() - 1],
        })
    }
}

/// One floor's verdict over the whole corpus.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Totals {
    pub clips: usize,
    pub kept_secs: f64,
    pub silence_ratio: f64,
    pub durations: Histogram,
}

/// What the corpus is, which is the question a user actually asks.
#[derive(Debug, Clone)]
pub struct Summary {
    pub files: usize,
    pub total_secs: f64,
    pub floor_db: Option<Stat>,
    pub signal_db: Option<Stat>,
    pub snr_db: Option<Stat>,
    /// Loudest sample anywhere in the corpus, in dBFS.
    pub peak_db: f32,
    pub clipped: usize,
    /// Files carrying enough clipping to be worth naming.
    pub clipped_files: Vec<PathBuf>,
    pub at_requested: Totals,
    /// At each file's *own* measured floor, so this is not one cut but the best
    /// each recording could do.
    pub at_measured: Totals,
    /// The floor to pass to `clip`, or `None` when no floor would help —
    /// see [`FileReport::continuous`].
    pub suggested_silence_db: Option<f32>,
    /// Files with no dead air at any floor.
    pub continuous: Vec<PathBuf>,
}

impl Summary {
    /// Fold per-file reports into the corpus answer.
    ///
    /// The suggested floor is the **lowest** of the per-file suggestions rather
    /// than their middle, and that is the same bias `audio_kit`'s margin was
    /// swept for: a floor that is too low keeps some dead air, which the next
    /// stage's `--min-clip` and padding absorb, while a floor that is too high
    /// eats the soft tails this toolkit exists to preserve. The spread is
    /// printed beside it so a corpus that disagrees with itself is visible
    /// rather than averaged away.
    pub fn of(reports: &[FileReport]) -> Self {
        let mut at_requested = Totals::default();
        let mut at_measured = Totals::default();
        let mut peak_db = SILENT_DB;
        let mut clipped = 0usize;
        let mut clipped_files = Vec::new();
        let mut continuous = Vec::new();
        let mut total_secs = 0.0f64;
        let (mut floors, mut signals, mut snrs, mut suggestions) =
            (Vec::new(), Vec::new(), Vec::new(), Vec::new());

        for r in reports {
            total_secs += r.secs;
            peak_db = peak_db.max(r.peak_db);
            clipped += r.clipped;
            if r.clipping_worth_saying() {
                clipped_files.push(r.path.clone());
            }
            if r.continuous() {
                continuous.push(r.path.clone());
            }
            if let Some(f) = r.floor {
                floors.push(f.floor_db);
                signals.push(f.signal_db);
                snrs.push(f.snr_db());
                suggestions.push(f.silence_db());
            }
            at_requested.clips += r.at_requested.clips;
            at_requested.kept_secs += r.at_requested.kept_secs;
            at_requested.durations.merge(&r.at_requested.durations);
            let measured = r.at_measured.as_ref().unwrap_or(&r.at_requested);
            at_measured.clips += measured.clips;
            at_measured.kept_secs += measured.kept_secs;
            at_measured.durations.merge(&measured.durations);
        }

        let ratio = |kept: f64| {
            if total_secs > 0.0 {
                (1.0 - kept / total_secs).clamp(0.0, 1.0)
            } else {
                0.0
            }
        };
        at_requested.silence_ratio = ratio(at_requested.kept_secs);
        at_measured.silence_ratio = ratio(at_measured.kept_secs);

        // Withheld only when *every* analysed file is one no floor can help:
        // one bad recording among twenty must not cost the other nineteen their
        // number, and twenty bad ones must not be answered with a threshold.
        let hopeless = !reports.is_empty() && continuous.len() == reports.len();
        let suggested_silence_db = if hopeless {
            None
        } else {
            suggestions
                .iter()
                .copied()
                .fold(None, |acc: Option<f32>, v| Some(acc.map_or(v, |a| a.min(v))))
        };

        Self {
            files: reports.len(),
            total_secs,
            floor_db: Stat::of(floors),
            signal_db: Stat::of(signals),
            snr_db: Stat::of(snrs),
            peak_db,
            clipped,
            clipped_files,
            at_requested,
            at_measured,
            suggested_silence_db,
            continuous,
        }
    }
}

/// Measure one decoded recording.
///
/// Separate from [`file`] because everything here is arithmetic over samples:
/// it is what the unit tests drive, on a machine with no ffmpeg and no corpus.
pub fn measure(path: &Path, samples: &[f32], opts: &AnalyzeOptions) -> FileReport {
    let secs = samples.len() as f64 / opts.sr.max(1) as f64;
    let peak = samples.iter().fold(0.0f32, |m, x| m.max(x.abs()));
    let peak_db = if peak > 0.0 {
        (20.0 * peak.log10()).max(SILENT_DB)
    } else {
        SILENT_DB
    };
    let clipped = samples.iter().filter(|x| x.abs() >= CLIP_LEVEL).count();

    let at_requested = cut(samples, secs, opts, opts.slice.silence_db);
    let floor = audio_kit::noise_floor(samples, opts.sr);
    // A measurement that lands on the requested floor would re-slice the whole
    // recording to print the same three numbers again.
    let at_measured = floor
        .map(|f| f.silence_db())
        .filter(|db| (db - opts.slice.silence_db).abs() > 0.05)
        .map(|db| cut(samples, secs, opts, db));

    FileReport {
        path: path.to_path_buf(),
        secs,
        floor,
        peak_db,
        clipped,
        at_requested,
        at_measured,
    }
}

/// Run the real slicer at `silence_db` and describe what came out.
fn cut(samples: &[f32], secs: f64, opts: &AnalyzeOptions, silence_db: f32) -> SliceOutcome {
    let slice_opts = SliceOptions {
        silence_db,
        ..opts.slice
    };
    let segments = audio_kit::slice(samples, opts.sr, &slice_opts);
    let mut durations = Histogram::default();
    let mut kept_secs = 0.0f64;
    for (start, end) in &segments {
        let d = (end - start) as f64 / opts.sr.max(1) as f64;
        kept_secs += d;
        durations.add(d);
    }
    SliceOutcome {
        silence_db,
        clips: segments.len(),
        kept_secs,
        silence_ratio: if secs > 0.0 {
            (1.0 - kept_secs / secs).clamp(0.0, 1.0)
        } else {
            0.0
        },
        durations,
    }
}

/// Decode one file and measure it.
///
/// No output directory, unlike every other stage: this one writes nothing, so
/// there is no directory for it to create and none for a re-run to re-ingest.
pub async fn file(input: &InputFile, opts: &AnalyzeOptions) -> Result<FileReport> {
    let samples = crate::decode_mono(&input.path, opts.sr).await?;
    Ok(measure(&input.path, &samples, opts))
}

#[cfg(test)]
mod tests {
    use super::*;

    const SR: u32 = 48_000;

    fn opts() -> AnalyzeOptions {
        AnalyzeOptions {
            sr: SR,
            slice: SliceOptions::default(),
        }
    }

    /// A 220 Hz tone at `amp`, which the slicer reads as speech.
    fn voiced(secs: f32, amp: f32) -> Vec<f32> {
        let n = (secs * SR as f32) as usize;
        (0..n)
            .map(|i| amp * (std::f64::consts::TAU * 220.0 * i as f64 / SR as f64).sin() as f32)
            .collect()
    }

    /// Room tone: deterministic low-level noise, so the floor is a real
    /// distribution rather than digital silence.
    fn room(secs: f32, amp: f32) -> Vec<f32> {
        let n = (secs * SR as f32) as usize;
        let mut s: u64 = 0x1234_5678;
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                amp * ((s >> 40) as f32 / (1u32 << 23) as f32 - 1.0)
            })
            .collect()
    }

    /// Three sentences over room tone: the shape every corpus recording has,
    /// and the one the report has to describe correctly.
    fn three_sentences() -> Vec<f32> {
        let mut sig = room(1.0, 0.002);
        for _ in 0..3 {
            sig.extend(voiced(2.0, 0.4));
            sig.extend(room(1.0, 0.002));
        }
        sig
    }

    #[test]
    fn the_report_describes_a_recording_with_gaps() {
        let sig = three_sentences();
        let r = measure(Path::new("a.wav"), &sig, &opts());

        assert!((r.secs - 10.0).abs() < 0.01, "{:?}", r.secs);
        // Three sentences, three clips, and the gaps are dead air.
        assert_eq!(r.at_requested.clips, 3);
        assert!(
            r.at_requested.silence_ratio > 0.2,
            "{:?}",
            r.at_requested.silence_ratio
        );
        // 0.4 full scale is -8 dBFS, and nothing is near full scale.
        assert!((r.peak_db + 7.96).abs() < 0.2, "{}", r.peak_db);
        assert_eq!(r.clipped, 0);
        // Two seconds a sentence, so every clip lands in the 2-4 s bucket
        // (padding puts it just over 2).
        assert_eq!(r.at_requested.durations.counts[2], 3);
        // A recording with gaps in it is not the pathological shape, however
        // long it is.
        assert!(!r.continuous());

        let floor = r.floor.expect("10 s is plenty to measure");
        assert!(floor.snr_db() > 20.0, "{floor:?}");
        assert!(
            r.suggested_silence_db().expect("measured") < -20.0,
            "{:?}",
            r.suggested_silence_db()
        );
    }

    /// The finding this stage exists to report: a continuous bed leaves no
    /// silence, so no floor can rescue it and the suggestion must be withheld.
    #[test]
    fn a_continuous_bed_is_reported_as_one_no_floor_can_fix() {
        // Speech over a bed loud enough that the gaps between sentences are not
        // quiet — which is what a backing track does to a recording.
        let mut sig = three_sentences();
        let bed = voiced(sig.len() as f32 / SR as f32, 0.1);
        for (s, b) in sig.iter_mut().zip(bed) {
            *s += b;
        }
        let r = measure(Path::new("mix.wav"), &sig, &opts());

        assert!(r.best_silence_ratio() < CONTINUOUS_SILENCE_RATIO, "{r:?}");
        assert!(r.continuous(), "{r:?}");
        assert_eq!(Summary::of(&[r]).suggested_silence_db, None);
    }

    /// …but an already-sliced corpus is gapless for a good reason, and must not
    /// be reported as the same thing.
    #[test]
    fn short_gapless_clips_are_not_the_pathological_shape() {
        let r = measure(Path::new("clip_000.wav"), &voiced(3.0, 0.4), &opts());
        assert!(r.best_silence_ratio() < CONTINUOUS_SILENCE_RATIO);
        assert!(!r.continuous(), "an already-sliced clip is not a bed: {r:?}");

        let s = Summary::of(&[r]);
        assert!(s.continuous.is_empty());
        assert!(s.suggested_silence_db.is_some());
    }

    #[test]
    fn clipping_is_counted_at_full_scale_not_above_it() {
        // 16-bit full scale, which never reaches 1.0 exactly.
        let mut sig = voiced(1.0, 0.4);
        sig.extend([32767.0 / 32768.0; 200]);
        let r = measure(Path::new("hot.wav"), &sig, &opts());
        assert_eq!(r.clipped, 200);
        assert!(r.clipping_worth_saying());
    }

    #[test]
    fn a_silent_file_reports_a_floor_rather_than_negative_infinity() {
        let r = measure(Path::new("silent.wav"), &vec![0.0; SR as usize], &opts());
        assert_eq!(r.peak_db, SILENT_DB);
        assert!(r.peak_db.is_finite());
        assert_eq!(r.at_requested.clips, 0);
    }

    #[test]
    fn a_file_too_short_to_measure_still_reports_what_it_can() {
        // Under `noise_floor`'s ~80 ms minimum: no percentiles, so no
        // suggestion — and no second slicing pass to compare against.
        let r = measure(Path::new("blip.wav"), &voiced(0.02, 0.4), &opts());
        assert!(r.floor.is_none());
        assert_eq!(r.suggested_silence_db(), None);
        assert!(r.at_measured.is_none());
        assert!(r.peak_db.is_finite());
    }

    #[test]
    fn the_corpus_suggestion_is_the_lowest_of_its_files() {
        // Two recordings with different floors: the corpus answer keeps the
        // softest one's, because a floor that is too high eats soft tails.
        let quiet = measure(Path::new("q.wav"), &three_sentences(), &opts());
        let mut loud_sig = room(1.0, 0.02);
        loud_sig.extend(voiced(2.0, 0.4));
        loud_sig.extend(room(1.0, 0.02));
        let loud = measure(Path::new("l.wav"), &loud_sig, &opts());

        let want = quiet
            .suggested_silence_db()
            .unwrap()
            .min(loud.suggested_silence_db().unwrap());
        let s = Summary::of(&[quiet, loud]);
        assert_eq!(s.suggested_silence_db, Some(want));
        assert_eq!(s.files, 2);
        assert!(s.total_secs > 13.0);
        // The corpus' own spread, not one file's.
        assert!(s.floor_db.unwrap().min < s.floor_db.unwrap().max);
    }

    #[test]
    fn histogram_buckets_cover_every_length() {
        let mut h = Histogram::default();
        for d in [0.5, 1.0, 1.9, 3.0, 7.0, 15.0, 29.0, 30.0, 120.0] {
            h.add(d);
        }
        assert_eq!(h.counts, [1, 2, 1, 1, 1, 1, 2]);
        assert_eq!(Histogram::label(0), "<1s");
        assert_eq!(Histogram::label(2), "2-4s");
        assert_eq!(Histogram::label(6), ">=30s");

        let mut other = Histogram::default();
        other.add(0.1);
        h.merge(&other);
        assert_eq!(h.counts[0], 2);
    }
}
