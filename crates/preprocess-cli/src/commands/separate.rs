//! `preprocess separate` — split recordings into a voice stem and a music stem.
//!
//! The stage a corpus recorded over a backing track needs run **first**, before
//! `clip`: slicing cuts on silence, and a continuous bed means there is none to
//! cut on. What that is worth, and what it is not, is in
//! [`preprocess_core::separate`]'s own docs — they are the same measurements
//! `--help` quotes, kept in one place.
//!
//! The model is loaded **once** for the whole batch. It is 448 MB and takes
//! seconds to reach a GPU, so a per-file load would dominate a corpus of short
//! recordings; the stage itself holds no state between files.

use anyhow::{Context, Result};

use crate::args::SeparateArgs;
use crate::commands::prepare;

pub async fn run(args: SeparateArgs) -> Result<()> {
    args.verify()?;
    args.download.install()?;
    let files = prepare(&args.input, &args.output_dir).await?;
    let opts = args.options();

    // After `prepare`, so a mistyped input path costs nothing, and after
    // `verify`, so a backend this build cannot run never reaches the fetch.
    let weights = match &args.model {
        Some(path) => path.clone(),
        None => hub_kit::fetch_mdx23c(&args.cache_dir)
            .await
            .context("fetching the separation model")?,
    };
    let separator = preprocess_core::backend::load(&weights, args.backend, args.device)?;

    let mut total_in_secs = 0.0f64;
    let mut total_passes = 0usize;
    let mut written = 0usize;
    let mut failed = 0usize;

    for f in &files {
        // One undecodable file must not abort the whole batch, exactly as in
        // `clip` — and here the model behind it took seconds to load.
        let report =
            match preprocess_core::separate::file(f, &opts, separator.as_ref(), &args.output_dir)
                .await
            {
                Ok(r) => r,
                Err(e) => {
                    eprintln!("skipping {}: {e:#}", f.path.display());
                    failed += 1;
                    continue;
                }
            };
        total_in_secs += report.in_secs;
        total_passes += report.passes;
        written += report.stems.len();
        let stems = report
            .stems
            .iter()
            .map(|s| {
                let name = s.path.file_name().unwrap_or(s.path.as_os_str());
                format!("{} rms {:.4}", name.to_string_lossy(), s.rms)
            })
            .collect::<Vec<_>>()
            .join(", ");
        // The gain is worth a column rather than a log line: the model is not
        // scale-invariant, so a recording that arrives far from full scale is
        // separated at a level it was not trained on, and this is where that
        // shows. It is undone before anything is written, so the RMS beside it
        // is at the input's own level.
        println!(
            "{}: {:.1}s in {} passes, input x{:.2} -> {}",
            f.path.display(),
            report.in_secs,
            report.passes,
            report.gain,
            stems,
        );
    }

    let skipped = if failed > 0 {
        format!(" ({failed} skipped)")
    } else {
        String::new()
    };
    println!(
        "total: {written} stems from {} files{}  ({total_in_secs:.1}s in {total_passes} passes)  -> {}",
        files.len() - failed,
        skipped,
        args.output_dir.display(),
    );
    Ok(())
}
