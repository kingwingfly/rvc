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
//! # The two constants that decide "no floor can help"
//!
//! [`preprocess_core::analyze::FileReport::continuous`] needs a length and a
//! silence ratio, and the pair has to separate a continuous bed from an
//! already-sliced corpus, which is gapless for a good reason. Measured on this
//! repository's own material:
//!
//! - `dataset/` — 92 clips a previous `clip` run wrote, 0.9-903 s each, SNR
//!   32-46 dB. Silence ratio at the measured floor: 0.00-0.13. The 90 short
//!   ones are gapless *and* under 15 s, and the two long ones (300 s and 903 s)
//!   are unsliced recordings that do hold dead air, at ratios of 0.27 and 0.13.
//!   Nothing here is flagged.
//! - `mix_60s.wav` — 60 s of speech over music, SNR 15.6 dB, silence ratio 0.00
//!   at every floor tried. Flagged.
//! - `mix_60s.vocals.wav` — the same 60 s after `separate`, SNR 30.4 dB,
//!   silence ratio 0.29. Not flagged, which is the contrast that says the
//!   advisory points at something a user can act on.

use std::path::Path;

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
        Format::Text => print_text(&summary, &reports, failed, &args),
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

fn print_text(summary: &Summary, reports: &[FileReport], failed: usize, args: &AnalyzeArgs) {
    if args.per_file {
        for r in reports {
            let measured = r.at_measured.as_ref().unwrap_or(&r.at_requested);
            println!(
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
        println!();
    }

    let skipped = if failed > 0 {
        format!(" ({failed} skipped)")
    } else {
        String::new()
    };
    println!(
        "corpus: {} files{}, {:.1}s total",
        summary.files, skipped, summary.total_secs
    );
    println!("  noise floor    {}", stat(summary.floor_db, "dBFS"));
    println!("  speech level   {}", stat(summary.signal_db, "dBFS"));
    println!("  snr            {}", stat(summary.snr_db, "dB"));
    println!(
        "  peak           {:.1} dBFS (loudest sample), {} clipped samples in {} files",
        summary.peak_db,
        summary.clipped,
        summary.clipped_files.len(),
    );
    println!(
        "  at --silence-db {:.1}:  {} clips, {:.0}% dead air",
        args.slice.silence_db,
        summary.at_requested.clips,
        summary.at_requested.silence_ratio * 100.0,
    );
    println!(
        "  at the measured floor: {} clips, {:.0}% dead air",
        summary.at_measured.clips,
        summary.at_measured.silence_ratio * 100.0,
    );
    println!(
        "  clip lengths   {}",
        histogram(&summary.at_measured.durations)
    );

    // Advisories last, because they are what a reader acts on — and every one
    // of them names the files it is about, so a corpus of hundreds does not
    // have to be re-run per file to find them.
    println!();
    match summary.suggested_silence_db {
        Some(db) => println!("suggested: clip --silence-db {db:.1}"),
        None => println!(
            "suggested: nothing. No floor finds dead air in this material, so no \
             --silence-db would help."
        ),
    }
    if !summary.continuous.is_empty() {
        println!(
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
        println!(
            "warning: {} files are clipped: {}. That distortion is in the recording, \
             so nothing downstream can remove it — it will be trained on.",
            summary.clipped_files.len(),
            named(&summary.clipped_files),
        );
    }
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
