//! `preprocess clip` — slice recordings into clean per-utterance clips.
//!
//! This is the stage a trainer needs run first. Training draws short random
//! windows uniformly across each corpus file, so a raw recording full of
//! between-sentence dead-air puts most of those windows in silence and collapses
//! the generator to silence with them. Slicing first removes the gaps and
//! nothing else — the softest passages are content, and energy is used only to
//! find where nobody is speaking.

use anyhow::Result;

use crate::args::ClipArgs;
use crate::commands::prepare;

pub async fn run(args: ClipArgs) -> Result<()> {
    args.verify()?;
    let files = prepare(&args.io.input, &args.output_dir).await?;
    let opts = args.options();

    let mut total_clips = 0usize;
    let mut total_in_secs = 0.0f64;
    let mut total_kept_secs = 0.0f64;
    let mut failed = 0usize;

    for f in &files {
        // One undecodable file must not abort the whole batch: a corpus of
        // hundreds routinely has one truncated download in it.
        let report = match preprocess_core::clip::file(f, &opts, &args.output_dir).await {
            Ok(r) => r,
            Err(e) => {
                eprintln!("skipping {}: {e:#}", f.path.display());
                failed += 1;
                continue;
            }
        };
        total_clips += report.clips;
        total_in_secs += report.in_secs;
        total_kept_secs += report.kept_secs;
        println!(
            "{}: {} clips  ({:.1}s in -> {:.1}s kept)",
            f.path.display(),
            report.clips,
            report.in_secs,
            report.kept_secs,
        );
    }

    let skipped = if failed > 0 {
        format!(" ({failed} skipped)")
    } else {
        String::new()
    };
    println!(
        "total: {} clips from {} files{}  ({:.1}s in -> {:.1}s kept)  -> {}",
        total_clips,
        files.len() - failed,
        skipped,
        total_in_secs,
        total_kept_secs,
        args.output_dir.display(),
    );
    Ok(())
}
