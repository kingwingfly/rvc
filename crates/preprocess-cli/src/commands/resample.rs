//! `preprocess resample` — one rate, one container, stated rather than implied.
//!
//! Every other stage decodes to whatever it needs and writes what it produced,
//! so a corpus assembled from several sources arrives at a trainer as several
//! rates. This is the stage that says which one, and it is the only one whose
//! whole job is the conversion the others do on the way past.

use anyhow::Result;

use crate::args::ResampleArgs;
use crate::commands::prepare;

pub async fn run(args: ResampleArgs) -> Result<()> {
    args.verify()?;
    let files = prepare(&args.io.input, &args.output_dir).await?;
    let opts = args.options();

    let mut total_secs = 0.0f64;
    let mut failed = 0usize;

    for f in &files {
        // One undecodable file must not abort the whole batch, exactly as in
        // `clip`.
        let report = match preprocess_core::resample::file(f, &opts, &args.output_dir).await {
            Ok(r) => r,
            Err(e) => {
                eprintln!("skipping {}: {e:#}", f.path.display());
                failed += 1;
                continue;
            }
        };
        total_secs += report.out_secs;
        // The channel count is reported per file because it is the half of this
        // conversion a user is most likely not to have thought about: `separate`
        // writes stereo on purpose, and folding that to mono here is a decision,
        // not a detail.
        let channels = match report.channels {
            preprocess_core::resample::Channels::Mono => "mono",
            preprocess_core::resample::Channels::Stereo => "stereo",
        };
        println!(
            "{}: {:.1}s  {} Hz {channels}  -> {}.wav",
            f.path.display(),
            report.out_secs,
            opts.sr,
            f.base,
        );
    }

    let skipped = if failed > 0 {
        format!(" ({failed} skipped)")
    } else {
        String::new()
    };
    println!(
        "total: {} files{}  ({:.1}s at {} Hz)  -> {}",
        files.len() - failed,
        skipped,
        total_secs,
        opts.sr,
        args.output_dir.display(),
    );
    Ok(())
}
