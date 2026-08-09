//! `preprocess denoise` — remove steady background hiss from recordings.
//!
//! Worth doing before slicing rather than after: `anlmdn` looks for
//! self-similar patches *nearby in time*, and a clip boundary is where the
//! neighbours it would have averaged over stop existing. The same reasoning
//! says the whole recording goes through one filter graph, which is what
//! [`preprocess_core::denoise::file`] does.

use anyhow::Result;

use crate::args::DenoiseArgs;
use crate::commands::prepare;

pub async fn run(args: DenoiseArgs) -> Result<()> {
    args.verify()?;
    let files = prepare(&args.io.input, &args.output_dir).await?;
    let opts = args.options();

    let mut total_secs = 0.0f64;
    let mut failed = 0usize;

    for f in &files {
        // One undecodable file must not abort the whole batch, exactly as in
        // `clip`.
        let report = match preprocess_core::denoise::file(f, &opts, &args.output_dir).await {
            Ok(r) => r,
            Err(e) => {
                eprintln!("skipping {}: {e:#}", f.path.display());
                failed += 1;
                continue;
            }
        };
        total_secs += report.out_secs;
        println!(
            "{}: {:.1}s -> {}.wav",
            f.path.display(),
            report.out_secs,
            f.base,
        );
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
    Ok(())
}
