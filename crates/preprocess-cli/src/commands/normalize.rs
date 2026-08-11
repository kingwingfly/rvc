//! `preprocess normalize` — match every recording to one level.
//!
//! A constant gain and nothing else. The peak target is what `clip
//! --normalize` applied all along, now reachable; the LUFS target measures
//! integrated loudness instead, which is what makes two takes recorded on
//! different days sit at the same *perceived* level rather than the same
//! sample maximum.

use anyhow::Result;

use crate::args::NormalizeArgs;
use crate::commands::prepare;

pub async fn run(args: NormalizeArgs) -> Result<()> {
    args.verify()?;
    let files = prepare(&args.io.input, &args.output_dir).await?;
    let opts = args.options();

    let mut total_secs = 0.0f64;
    let mut failed = 0usize;
    // Counted rather than warned per file: on a LUFS run a quiet take can need
    // enough gain to push its loudest sample past full scale, and a batch where
    // that happened to half the corpus is a different message from one file.
    let mut clipped = 0usize;

    for f in &files {
        // One undecodable file must not abort the whole batch, exactly as in
        // `clip`.
        let report = match preprocess_core::normalize::file(f, &opts, &args.output_dir).await {
            Ok(r) => r,
            Err(e) => {
                eprintln!("skipping {}: {e:#}", f.path.display());
                failed += 1;
                continue;
            }
        };
        total_secs += report.in_secs;
        if report.peak_after > 1.0 {
            clipped += 1;
        }
        // The loudness pair is only measured on a LUFS run, so it is only
        // printed on one — a peak run that reported `None -> None` would read
        // as a measurement that failed rather than one nobody asked for.
        match report.lufs {
            Some((before, after)) => println!(
                "{}: {:.1}s  {:+.1} dB  peak {:.3} -> {:.3}  lufs {:.1} -> {:.1}  -> {}.wav",
                f.path.display(),
                report.in_secs,
                report.gain_db,
                report.peak_before,
                report.peak_after,
                before,
                after,
                f.base,
            ),
            None => println!(
                "{}: {:.1}s  {:+.1} dB  peak {:.3} -> {:.3}  -> {}.wav",
                f.path.display(),
                report.in_secs,
                report.gain_db,
                report.peak_before,
                report.peak_after,
                f.base,
            ),
        }
    }

    let skipped = if failed > 0 {
        format!(" ({failed} skipped)")
    } else {
        String::new()
    };
    println!(
        "total: {} files{}  ({:.1}s)  -> {}",
        files.len() - failed,
        skipped,
        total_secs,
        args.output_dir.display(),
    );
    if clipped > 0 {
        // Not an error: the gain asked for was applied, and reporting it is what
        // this stage promises instead of silently limiting. But a corpus that
        // clips is one somebody wants to know about before training on it.
        eprintln!(
            "note: {clipped} of {} files peak above full scale after the gain — lower --lufs, or \
             use --peak, if the clipping matters",
            files.len() - failed,
        );
    }
    Ok(())
}
