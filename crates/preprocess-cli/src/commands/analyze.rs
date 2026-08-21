//! `preprocess analyze` — report what a corpus is, and what to set the other
//! stages to.
//!
//! The one stage here that runs no model and writes no file. What it costs is a
//! decode; what it buys is that `clip`'s `--silence-db`, `denoise`'s strength
//! and the decision to run `separate` at all stop being guesses. The
//! measurements themselves are [`preprocess_core::analyze`]'s, which are
//! `audio-kit`'s — this module only drives the batch and prints.
//!
//! # Two output shapes, one of which must stay parseable
//!
//! `--format json` puts **one** document on stdout and nothing else: skips and
//! advisories go to stderr, exactly as `clip`'s skips already do. That is not
//! tidiness — a warning line on stdout turns the document into something no
//! parser accepts, and the failure appears in the consumer rather than here.
//!
//! # What the real material says, which is not what this stage was built expecting
//!
//! Measured with this binary on every recording the repository has, `clips /
//! dead air` given at the requested `--silence-db -40` and then at the floor
//! derived from the recording:
//!
//! | | length | floor | SNR | at -40 dB | at the measured floor |
//! |---|---|---|---|---|---|
//! | `dataset/` (92 clips a previous `clip` run wrote) | 1207 s | -47.2 | 13.0 | 72 / 4% | 285 / 15% |
//! | the source stream, whole | 1191 s | -40.3 | 14.2 | 64 / 4% | 196 / 15% |
//! | `mix_60s.wav`, speech over music | 60 s | -40.3 | 15.1 | 5 / 3% | 14 / 19% |
//! | the same after `separate` | 60 s | -53.6 | 26.4 | 12 / 34% | 13 / 18% |
//! | `at_180s.wav`, the densest bed on hand | 30 s | -35.4 | 7.9 | **1 / 0%** | 9 / 35% |
//!
//! **Not one of them is flagged
//! [`continuous`](preprocess_core::analyze::FileReport::continuous)**, and that
//! is the finding rather than a gap in the sample. The condition asks for no
//! dead air at *either* floor, and on real speech-over-music the derived floor
//! always finds some — even `at_180s.wav`, which at the default floor is a
//! single unsliceable 30 s block, opens into nine clips once the floor is read
//! off the recording. So `continuous` stays the tail case it describes, pinned
//! by a synthetic fixture in [`preprocess_core::analyze`] and by nothing else.
//! **Do not calibrate its two constants against these numbers** — none of them
//! is near the boundary, so any threshold that "fixed" one would be fitting
//! noise.
//!
//! What the table does demonstrate is the half of this stage that fires on
//! everything: the *requested* floor is wrong for this material by a factor of
//! three to nine in clip count, in the direction that matters — 5 clips out of
//! 60 s of speech is sentences merged into paragraphs, which is what a trainer
//! then samples 0.48 s windows out of. The suggestion is what closes that, and
//! it is worth running before `clip` on anything not recorded in a quiet room.

use std::{fmt::Write, path::Path};

use anyhow::Result;
use preprocess_core::analyze::{FileReport, Histogram, Stat, Summary};

use crate::args::{AnalyzeArgs, Format};

pub async fn run(args: AnalyzeArgs) -> Result<()> {
    args.verify()?;
    // Not `prepare`: that creates the output directory, and a stage that writes
    // nothing must not leave one behind. `plan` takes the directory to exclude
    // so a re-run never re-ingests its own output — there is none here, and an
    // empty path canonicalises to nothing, which excludes nothing.
    let files = preprocess_core::plan(&args.input, Path::new(""))?;
    anyhow::ensure!(
        !files.is_empty(),
        "no audio files found in the given input paths: {}",
        args.input
            .iter()
            .map(|p| p.display().to_string())
            .collect::<Vec<_>>()
            .join(", ")
    );
    let opts = args.options();

    let mut reports = Vec::with_capacity(files.len());
    let mut failed = 0usize;
    for f in &files {
        // One undecodable file must not abort the whole batch, exactly as in
        // `clip` — and stderr, so `--format json` stays one document.
        match preprocess_core::analyze::file(f, &opts).await {
            Ok(r) => reports.push(r),
            Err(e) => {
                eprintln!("skipping {}: {e:#}", f.path.display());
                failed += 1;
            }
        }
    }
    anyhow::ensure!(
        !reports.is_empty(),
        "none of the {} input files could be decoded",
        files.len()
    );

    let summary = Summary::of(&reports);
    match args.format {
        Format::Text => print!(
            "{}",
            text(
                &summary,
                &reports,
                failed,
                args.per_file,
                args.slice.silence_db
            )
        ),
        Format::Json => println!("{}", json(&summary, &reports, failed)),
    }
    Ok(())
}

/// dBFS, or `n/a` for a file with nothing to measure — never `-inf` or `NaN`,
/// which read as a defect in this stage rather than as a short recording.
fn db(v: Option<f32>) -> String {
    match v {
        Some(v) => format!("{v:.1} dBFS"),
        None => "n/a".to_string(),
    }
}

fn stat(s: Option<Stat>, unit: &str) -> String {
    match s {
        Some(s) => format!(
            "{:.1} {unit} (spread {:.1} .. {:.1})",
            s.median, s.min, s.max
        ),
        None => "n/a".to_string(),
    }
}

fn histogram(h: &Histogram) -> String {
    h.counts
        .iter()
        .enumerate()
        .map(|(i, n)| format!("{} {n}", Histogram::label(i)))
        .collect::<Vec<_>>()
        .join(" | ")
}

/// The report as text, returned rather than printed.
///
/// A `String` rather than a series of `println!`s because that is what makes
/// the numbers checkable at all: the text form and the JSON form carry the same
/// figures, and only a test that can read both can say so. It takes the two
/// flags it reads rather than the whole [`AnalyzeArgs`] for the same reason — a
/// formatter that needs a parsed command line to be exercised is one nobody
/// exercises.
fn text(
    summary: &Summary,
    reports: &[FileReport],
    failed: usize,
    per_file: bool,
    requested_db: f32,
) -> String {
    let mut out = String::new();
    if per_file {
        for r in reports {
            let measured = r.at_measured.as_ref().unwrap_or(&r.at_requested);
            let _ = writeln!(
                out,
                "{}: {:.1}s  floor {}  snr {}  peak {:.1} dBFS{}  {} clips / {:.0}% silence at {:.1} dB, \
                 {} clips / {:.0}% silence at {:.1} dB{}",
                r.path.display(),
                r.secs,
                db(r.floor.map(|f| f.floor_db)),
                match r.floor {
                    Some(f) => format!("{:.1} dB", f.snr_db()),
                    None => "n/a".to_string(),
                },
                r.peak_db,
                if r.clipped > 0 {
                    format!(" ({} clipped)", r.clipped)
                } else {
                    String::new()
                },
                r.at_requested.clips,
                r.at_requested.silence_ratio * 100.0,
                r.at_requested.silence_db,
                measured.clips,
                measured.silence_ratio * 100.0,
                measured.silence_db,
                if r.continuous() {
                    "  [no dead air]"
                } else {
                    ""
                },
            );
        }
        let _ = writeln!(out);
    }

    let skipped = if failed > 0 {
        format!(" ({failed} skipped)")
    } else {
        String::new()
    };
    let _ = writeln!(
        out,
        "corpus: {} files{}, {:.1}s total",
        summary.files, skipped, summary.total_secs
    );
    let _ = writeln!(out, "  noise floor    {}", stat(summary.floor_db, "dBFS"));
    let _ = writeln!(out, "  speech level   {}", stat(summary.signal_db, "dBFS"));
    let _ = writeln!(out, "  snr            {}", stat(summary.snr_db, "dB"));
    // The two counts answer different questions and the line has to say so:
    // `clipped` is every sample at full scale anywhere in the corpus, while
    // `clipped_files` holds only the files with *enough* of them to be worth
    // naming (`CLIPPED_SAMPLES_WORTH_SAYING`). Written as "N clipped samples in
    // M files" they read as one tally, and the honest reading of this repo's own
    // `dataset/` — 7 stray samples, no file near the threshold — came out as the
    // self-contradicting "7 clipped samples in 0 files".
    let _ = match summary.clipped_files.len() {
        0 => writeln!(
            out,
            "  peak           {:.1} dBFS (loudest sample), {} samples at full scale \
             (no file has enough to matter)",
            summary.peak_db, summary.clipped,
        ),
        n => writeln!(
            out,
            "  peak           {:.1} dBFS (loudest sample), {} samples at full scale, \
             {n} file(s) clipped enough to hear",
            summary.peak_db, summary.clipped,
        ),
    };
    let _ = writeln!(
        out,
        "  at --silence-db {:.1}:  {} clips, {:.0}% dead air",
        requested_db,
        summary.at_requested.clips,
        summary.at_requested.silence_ratio * 100.0,
    );
    let _ = writeln!(
        out,
        "  at the measured floor: {} clips, {:.0}% dead air",
        summary.at_measured.clips,
        summary.at_measured.silence_ratio * 100.0,
    );
    let _ = writeln!(
        out,
        "  clip lengths   {}",
        histogram(&summary.at_measured.durations)
    );

    // Advisories last, because they are what a reader acts on — and every one
    // of them names the files it is about, so a corpus of hundreds does not
    // have to be re-run per file to find them.
    let _ = writeln!(out);
    let _ = match summary.suggested_silence_db {
        Some(db) => writeln!(out, "suggested: clip --silence-db {db:.1}"),
        None => writeln!(
            out,
            "suggested: nothing. No floor finds dead air in this material, so no \
             --silence-db would help."
        ),
    };
    if !summary.continuous.is_empty() {
        let _ = writeln!(
            out,
            "warning: {} of {} files hold no dead air at any floor: {}. `clip` cuts on \
             silence, so it cannot split these into sentences. If there is music or \
             another continuous bed under the speech, run `separate` first — a lower \
             --silence-db cannot reach quiet that is not there.",
            summary.continuous.len(),
            summary.files,
            named(&summary.continuous),
        );
    }
    if !summary.clipped_files.is_empty() {
        let _ = writeln!(
            out,
            "warning: {} files are clipped: {}. That distortion is in the recording, \
             so nothing downstream can remove it — it will be trained on.",
            summary.clipped_files.len(),
            named(&summary.clipped_files),
        );
    }
    out
}

/// The first few offenders, and how many more there are. A warning that lists
/// two hundred paths is one nobody reads.
fn named(paths: &[std::path::PathBuf]) -> String {
    const SHOWN: usize = 3;
    let head = paths
        .iter()
        .take(SHOWN)
        .map(|p| p.display().to_string())
        .collect::<Vec<_>>()
        .join(", ");
    if paths.len() > SHOWN {
        format!("{head} and {} more", paths.len() - SHOWN)
    } else {
        head
    }
}

/// The whole report as one JSON document.
///
/// Built with `serde_json` rather than by hand: this is a nested document with
/// two float-heavy tables in it, where `stt`'s hand-rolled escaping covers one
/// string field. Nothing derives `Serialize` — the stage's own types stay free
/// of serde, exactly as they stay free of clap.
fn json(summary: &Summary, reports: &[FileReport], failed: usize) -> String {
    let outcome = |o: &preprocess_core::analyze::SliceOutcome| {
        serde_json::json!({
            "silence_db": o.silence_db,
            "clips": o.clips,
            "kept_secs": o.kept_secs,
            "silence_ratio": o.silence_ratio,
            "durations": durations_json(&o.durations),
        })
    };
    let stat = |s: Option<Stat>| match s {
        Some(s) => serde_json::json!({"median": s.median, "min": s.min, "max": s.max}),
        None => serde_json::Value::Null,
    };
    let files: Vec<_> = reports
        .iter()
        .map(|r| {
            serde_json::json!({
                "path": r.path.display().to_string(),
                "secs": r.secs,
                "floor_db": r.floor.map(|f| f.floor_db),
                "signal_db": r.floor.map(|f| f.signal_db),
                "snr_db": r.floor.map(|f| f.snr_db()),
                "peak_db": r.peak_db,
                "clipped": r.clipped,
                "suggested_silence_db": r.suggested_silence_db(),
                "continuous": r.continuous(),
                "at_requested": outcome(&r.at_requested),
                "at_measured": r.at_measured.as_ref().map(outcome),
            })
        })
        .collect();

    let doc = serde_json::json!({
        "summary": {
            "files": summary.files,
            "skipped": failed,
            "total_secs": summary.total_secs,
            "floor_db": stat(summary.floor_db),
            "signal_db": stat(summary.signal_db),
            "snr_db": stat(summary.snr_db),
            "peak_db": summary.peak_db,
            "clipped": summary.clipped,
            "clipped_files": paths_json(&summary.clipped_files),
            "at_requested": {
                "clips": summary.at_requested.clips,
                "kept_secs": summary.at_requested.kept_secs,
                "silence_ratio": summary.at_requested.silence_ratio,
                "durations": durations_json(&summary.at_requested.durations),
            },
            "at_measured": {
                "clips": summary.at_measured.clips,
                "kept_secs": summary.at_measured.kept_secs,
                "silence_ratio": summary.at_measured.silence_ratio,
                "durations": durations_json(&summary.at_measured.durations),
            },
            "suggested_silence_db": summary.suggested_silence_db,
            "continuous": paths_json(&summary.continuous),
        },
        "files": files,
    });
    serde_json::to_string_pretty(&doc).expect("a report of numbers and paths serialises")
}

/// Buckets as `{"<1s": 3, …}` — labelled, because a bare array of seven counts
/// is a document that cannot be read without this source file open.
fn durations_json(h: &Histogram) -> serde_json::Value {
    let mut map = serde_json::Map::new();
    for (i, n) in h.counts.iter().enumerate() {
        map.insert(Histogram::label(i), serde_json::json!(n));
    }
    serde_json::Value::Object(map)
}

fn paths_json(paths: &[std::path::PathBuf]) -> serde_json::Value {
    serde_json::json!(
        paths
            .iter()
            .map(|p| p.display().to_string())
            .collect::<Vec<_>>()
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use audio_kit::NoiseFloor;
    use preprocess_core::analyze::SliceOutcome;
    use std::path::PathBuf;

    /// `--silence-db`'s default, which is what a report is rendered against
    /// unless a test says otherwise.
    const REQUESTED_DB: f32 = -40.0;

    fn outcome(silence_db: f32, clips: usize, kept_secs: f64, secs: f64) -> SliceOutcome {
        let mut durations = Histogram::default();
        for _ in 0..clips {
            durations.add(kept_secs / clips.max(1) as f64);
        }
        SliceOutcome {
            silence_db,
            clips,
            kept_secs,
            silence_ratio: (1.0 - kept_secs / secs).clamp(0.0, 1.0),
            durations,
        }
    }

    /// A recording with gaps: three clips out of ten seconds, and a floor its
    /// own frames suggest.
    fn with_gaps(path: &str) -> FileReport {
        FileReport {
            path: PathBuf::from(path),
            secs: 10.0,
            floor: Some(NoiseFloor {
                floor_db: -52.0,
                signal_db: -18.0,
            }),
            peak_db: -6.0,
            clipped: 0,
            at_requested: outcome(REQUESTED_DB, 3, 7.0, 10.0),
            at_measured: Some(outcome(-46.5, 4, 8.0, 10.0)),
        }
    }

    /// Long enough to be judged, and gapless at both floors: the shape the
    /// advisory exists for.
    fn continuous(path: &str) -> FileReport {
        FileReport {
            path: PathBuf::from(path),
            secs: 60.0,
            floor: Some(NoiseFloor {
                floor_db: -40.0,
                signal_db: -22.0,
            }),
            peak_db: -1.0,
            clipped: 0,
            at_requested: outcome(REQUESTED_DB, 1, 60.0, 60.0),
            at_measured: Some(outcome(-37.0, 1, 60.0, 60.0)),
        }
    }

    fn parse(doc: &str) -> serde_json::Value {
        serde_json::from_str(doc).expect("`--format json` puts one parseable document on stdout")
    }

    /// The line a `.starts_with` finds, so a test pins one number rather than
    /// the whole blob and survives a reworded sentence around it.
    fn line<'a>(rendered: &'a str, prefix: &str) -> &'a str {
        rendered
            .lines()
            .find(|l| l.starts_with(prefix))
            .unwrap_or_else(|| panic!("no line starting {prefix:?} in:\n{rendered}"))
    }

    /// Every field of [`Summary`] has to appear, because a formatter that drops
    /// one is invisible: the document still parses and the consumer reads a
    /// report with a hole in it. Pinned as the whole key set rather than field
    /// by field, so *adding* a field to the struct without emitting it fails
    /// here too.
    #[test]
    fn the_json_document_parses_and_names_every_summary_field() {
        let reports = vec![with_gaps("a.wav"), continuous("b.wav")];
        let summary = Summary::of(&reports);
        let doc = parse(&json(&summary, &reports, 1));

        let mut keys: Vec<_> = doc["summary"]
            .as_object()
            .expect("an object")
            .keys()
            .map(String::as_str)
            .collect();
        keys.sort_unstable();
        assert_eq!(
            keys,
            [
                "at_measured",
                "at_requested",
                "clipped",
                "clipped_files",
                "continuous",
                "files",
                "floor_db",
                "peak_db",
                "signal_db",
                "skipped",
                "snr_db",
                "suggested_silence_db",
                "total_secs",
            ],
            "the twelve `Summary` fields plus the skip count this module adds"
        );

        let s = &doc["summary"];
        assert_eq!(s["files"], 2);
        assert_eq!(s["skipped"], 1);
        assert_eq!(s["clipped"], 0);
        assert_eq!(s["total_secs"], 70.0);
        assert_eq!(s["at_requested"]["clips"], 4);
        assert_eq!(s["at_requested"]["kept_secs"], 67.0);
        assert_eq!(s["at_measured"]["clips"], 5);
        assert_eq!(s["continuous"], serde_json::json!(["b.wav"]));
        assert_eq!(s["clipped_files"], serde_json::json!([]));
        // The buckets are labelled rather than positional, which is the whole
        // reason `durations_json` exists.
        assert_eq!(s["at_measured"]["durations"]["2-4s"], 4);

        // …and one file per report, carrying what only a per-file view holds.
        assert_eq!(doc["files"].as_array().expect("an array").len(), 2);
        assert_eq!(doc["files"][0]["path"], "a.wav");
        assert_eq!(doc["files"][0]["continuous"], false);
        assert_eq!(doc["files"][1]["continuous"], true);
        assert_eq!(doc["files"][0]["at_measured"]["silence_db"], -46.5);
    }

    /// The two forms are two renderings of one struct, and nothing but this
    /// says so: a stale field in either is a report that disagrees with itself
    /// and parses fine.
    #[test]
    fn the_text_and_json_forms_report_the_same_numbers() {
        let reports = vec![with_gaps("a.wav"), with_gaps("b.wav")];
        let summary = Summary::of(&reports);
        let rendered = text(&summary, &reports, 1, false, REQUESTED_DB);
        let s = parse(&json(&summary, &reports, 1))["summary"].clone();

        assert_eq!(
            line(&rendered, "corpus:"),
            format!(
                "corpus: {} files ({} skipped), {:.1}s total",
                s["files"].as_u64().unwrap(),
                s["skipped"].as_u64().unwrap(),
                s["total_secs"].as_f64().unwrap()
            )
        );
        assert_eq!(
            line(&rendered, "  at --silence-db"),
            format!(
                "  at --silence-db {:.1}:  {} clips, {:.0}% dead air",
                REQUESTED_DB,
                s["at_requested"]["clips"].as_u64().unwrap(),
                s["at_requested"]["silence_ratio"].as_f64().unwrap() * 100.0
            )
        );
        assert_eq!(
            line(&rendered, "  at the measured floor:"),
            format!(
                "  at the measured floor: {} clips, {:.0}% dead air",
                s["at_measured"]["clips"].as_u64().unwrap(),
                s["at_measured"]["silence_ratio"].as_f64().unwrap() * 100.0
            )
        );
        assert_eq!(
            line(&rendered, "suggested:"),
            format!(
                "suggested: clip --silence-db {:.1}",
                s["suggested_silence_db"].as_f64().unwrap()
            )
        );
        assert_eq!(
            line(&rendered, "  noise floor"),
            format!(
                "  noise floor    {:.1} dBFS (spread {:.1} .. {:.1})",
                s["floor_db"]["median"].as_f64().unwrap(),
                s["floor_db"]["min"].as_f64().unwrap(),
                s["floor_db"]["max"].as_f64().unwrap()
            )
        );
        // The two clipping counts are one line and mean different things, which
        // is the sentence that had to be rewritten once already.
        assert!(
            line(&rendered, "  peak").ends_with("(no file has enough to matter)"),
            "{}",
            line(&rendered, "  peak")
        );
        // Nothing is skipped silently: a run that dropped a file says so.
        assert!(rendered.contains("(1 skipped)"));
    }

    /// The advisory is the stage's most consequential output, and it has to
    /// name the files rather than only count them.
    #[test]
    fn a_corpus_of_continuous_files_withholds_the_suggestion_in_both_forms() {
        let reports = vec![continuous("mix_a.wav"), continuous("mix_b.wav")];
        let summary = Summary::of(&reports);
        let rendered = text(&summary, &reports, 0, false, REQUESTED_DB);

        assert_eq!(
            line(&rendered, "suggested:"),
            "suggested: nothing. No floor finds dead air in this material, so no \
             --silence-db would help."
        );
        let warning = line(&rendered, "warning:");
        assert!(
            warning.contains("2 of 2 files hold no dead air"),
            "{warning}"
        );
        assert!(warning.contains("mix_a.wav, mix_b.wav"), "{warning}");
        assert!(warning.contains("run `separate` first"), "{warning}");

        let s = parse(&json(&summary, &reports, 0))["summary"].clone();
        assert_eq!(s["suggested_silence_db"], serde_json::Value::Null);
        assert_eq!(
            s["continuous"],
            serde_json::json!(["mix_a.wav", "mix_b.wav"])
        );
    }

    /// `--per-file` is off by default, and the line it adds falls back to the
    /// requested outcome for a file with no measured floor — the same fallback
    /// [`Summary::of`] makes, which is why they are checked against each other
    /// rather than against a literal.
    #[test]
    fn the_per_file_lines_appear_only_when_asked() {
        let mut short = with_gaps("short.wav");
        short.floor = None;
        short.at_measured = None;
        short.clipped = 4321;
        let reports = vec![short];
        let summary = Summary::of(&reports);

        assert!(!text(&summary, &reports, 0, false, REQUESTED_DB).contains("short.wav: "));

        let rendered = text(&summary, &reports, 0, true, REQUESTED_DB);
        let per_file = line(&rendered, "short.wav: ");
        assert!(per_file.contains("floor n/a"), "{per_file}");
        assert!(per_file.contains("snr n/a"), "{per_file}");
        assert!(per_file.contains("(4321 clipped)"), "{per_file}");
        // With no measured floor, both halves of the line are the requested cut.
        assert!(
            per_file
                .contains("3 clips / 30% silence at -40.0 dB, 3 clips / 30% silence at -40.0 dB"),
            "{per_file}"
        );
        // A file with no floor contributes no percentile, so the corpus columns
        // are `n/a` rather than a number invented from one file.
        assert_eq!(line(&rendered, "  snr  "), "  snr            n/a");
        assert!(rendered.contains("4321 samples at full scale, 1 file(s) clipped enough to hear"));
    }
}
