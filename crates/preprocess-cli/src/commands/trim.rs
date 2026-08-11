//! `preprocess trim` — drop the silence at each end, and nothing in the middle.
//!
//! `clip` finds these boundaries already and then throws the whole-file case
//! away, because its job is to cut a recording into sentences. A take that is
//! *already* one utterance wants the same measurement and one file back, which
//! is the only difference between the two stages.

use anyhow::Result;

use crate::args::TrimArgs;
use crate::commands::prepare;

pub async fn run(args: TrimArgs) -> Result<()> {
    args.verify()?;
    let files = prepare(&args.io.input, &args.output_dir).await?;
    let opts = args.options();

    let mut in_total = 0.0f64;
    let mut out_total = 0.0f64;
    let mut failed = 0usize;
    // A file with nothing above the floor is passed through whole rather than
    // emptied, which is the right call and also the one a user is most likely
    // to want to hear about: it usually means the floor is wrong for that take.
    let mut silent = 0usize;

    for f in &files {
        // One undecodable file must not abort the whole batch, exactly as in
        // `clip`.
        let report = match preprocess_core::trim::file(f, &opts, &args.output_dir).await {
            Ok(r) => r,
            Err(e) => {
                eprintln!("skipping {}: {e:#}", f.path.display());
                failed += 1;
                continue;
            }
        };
        in_total += report.in_secs;
        out_total += report.out_secs;
        if report.silent {
            silent += 1;
            println!(
                "{}: {:.1}s  nothing above {:.1} dBFS, written through unchanged  -> {}.wav",
                f.path.display(),
                report.in_secs,
                report.silence_db,
                f.base,
            );
            continue;
        }
        // The floor is printed because `--measure-floor` means it is not
        // necessarily the one that was asked for.
        println!(
            "{}: {:.1}s -> {:.1}s  (head {:.2}s, tail {:.2}s, floor {:.1} dBFS)  -> {}.wav",
            f.path.display(),
            report.in_secs,
            report.out_secs,
            report.head_secs,
            report.tail_secs,
            report.silence_db,
            f.base,
        );
    }

    let skipped = if failed > 0 {
        format!(" ({failed} skipped)")
    } else {
        String::new()
    };
    println!(
        "total: {} files{}  ({:.1}s -> {:.1}s)  -> {}",
        files.len() - failed,
        skipped,
        in_total,
        out_total,
        args.output_dir.display(),
    );
    if silent > 0 {
        eprintln!(
            "note: {silent} file(s) had nothing above the floor and were copied whole — lower \
             --silence-db, or pass --measure-floor to read it off each recording"
        );
    }
    Ok(())
}
