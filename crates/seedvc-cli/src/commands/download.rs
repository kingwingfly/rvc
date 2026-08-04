//! `seedvc download` — prefetch what a conversion fetches on its first run.

use anyhow::{Context, Result};

use crate::args::DownloadArgs;

pub async fn run(args: DownloadArgs) -> Result<()> {
    let paths = hub_kit::fetch_seedvc(&args.cache_dir)
        .await
        .context("failed to fetch the Seed-VC models")?;

    // Where they landed is the point of the command, so it goes to stdout. Four
    // lines rather than the one bundle directory `tts download` prints, because
    // these come from four separate repos and no directory holds them all — and
    // each line is what the matching override flag takes, which is what makes a
    // hand-assembled set checkable against a fetched one.
    //
    // There is nothing optional to skip and no warm-start base to leave out: a
    // conversion opens every one of these, and Seed-VC is zero-shot, so nothing
    // it downloads is ever a training input.
    println!("checkpoint: {}", paths.checkpoint.display());
    println!("campplus:   {}", paths.campplus.display());
    println!("bigvgan:    {}", paths.bigvgan.display());
    println!("content:    {}", paths.whisper.display());
    Ok(())
}
