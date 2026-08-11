//! `preprocess diarize` — keep only the parts one voice is speaking.
//!
//! **This is the stage source separation cannot do.** A vocals stem holds every
//! voice in the mixture, because separation asks "is this a voice" rather than
//! "whose" — so a stream with a song playing behind it comes out of `separate`
//! as the streamer *and* the singer. Telling them apart needs speaker identity,
//! which is what CAM++ supplies here.
//!
//! Which makes the ordering the whole recipe, and it is spelled out in the
//! subcommand's own `--help` rather than only here:
//!
//! ```sh
//! preprocess separate raw/    -o vocals/ --stem vocals
//! preprocess diarize  vocals/ -o mine/   --reference streamer.wav
//! preprocess clip     mine/   -o dataset/
//! ```

use anyhow::{Context, Result};

use crate::args::DiarizeArgs;
use crate::commands::prepare;

pub async fn run(args: DiarizeArgs) -> Result<()> {
    // `verify` is where `--backend onnx` is refused, so a request nothing could
    // run costs no fetch and no decode.
    args.verify()?;
    args.download.install()?;
    let files = prepare(&args.io.input, &args.output_dir).await?;
    let opts = args.options();

    // After `prepare`, so a mistyped input path costs nothing, and after
    // `verify`, so a backend this build cannot run never reaches the fetch —
    // `separate`'s ordering, for the same two reasons.
    let weights = match &args.model {
        Some(path) => path.clone(),
        None => hub_kit::fetch_campplus(&args.cache_dir)
            .await
            .context("fetching the speaker-embedding model")?,
    };
    let embedder = preprocess_core::embed::load(&weights, args.backend, args.device)?;
    // The reference is embedded once, before any input is opened: it is the
    // whole specification of what to keep, so a bad one should fail before a
    // corpus has been read rather than after.
    let target = preprocess_core::diarize::reference(embedder.as_ref(), &args.reference)
        .await
        .with_context(|| format!("embedding the reference {}", args.reference.display()))?;

    let mut total_segments = 0usize;
    let mut total_in_secs = 0.0f64;
    let mut total_kept_secs = 0.0f64;
    let mut failed = 0usize;

    for f in &files {
        // One undecodable file must not abort the whole batch, exactly as in
        // `clip` — and here a file shorter than one window is the same case.
        let report = match preprocess_core::diarize::file(
            f,
            &opts,
            embedder.as_ref(),
            &target,
            &args.output_dir,
        )
        .await
        {
            Ok(r) => r,
            Err(e) => {
                eprintln!("skipping {}: {e:#}", f.path.display());
                failed += 1;
                continue;
            }
        };
        total_segments += report.segments;
        total_in_secs += report.in_secs;
        total_kept_secs += report.kept_secs;
        println!(
            "{}: {} segments  ({:.1}s in -> {:.1}s kept, {}/{} windows, mean cosine {:.3})",
            f.path.display(),
            report.segments,
            report.in_secs,
            report.kept_secs,
            report.kept_windows,
            report.windows,
            report.mean_score,
        );
    }

    let skipped = if failed > 0 {
        format!(" ({failed} skipped)")
    } else {
        String::new()
    };
    println!(
        "total: {} segments from {} files{}  ({:.1}s in -> {:.1}s kept)  -> {}",
        total_segments,
        files.len() - failed,
        skipped,
        total_in_secs,
        total_kept_secs,
        args.output_dir.display(),
    );
    // An empty output is the failure a user is most likely to hit and least
    // likely to diagnose, so it names the cause in the order they turned out to
    // occur rather than leaving an empty directory to interpret. A reference
    // recorded elsewhere comes first because it is the largest effect measured:
    // one speaker across two sessions scores 0.36-0.78 where the same session
    // gives 0.87-0.90, which is a wider spread than the one the threshold sits
    // in.
    if total_segments == 0 && failed < files.len() {
        eprintln!(
            "nothing matched the reference. Most likely {} is from a different \
             recording — the embedding keys partly on the microphone and the \
             room, so take the reference from the audio being filtered. \
             Otherwise it is the wrong voice, or --threshold ({}) is above every \
             window's score: the per-file mean cosine above is what to compare \
             it against.",
            args.reference.display(),
            args.threshold,
        );
    }
    Ok(())
}
