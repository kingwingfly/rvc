//! `tts download` — prefetch the weights synthesis would otherwise fetch on its
//! first run.

use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Args;

/// What a default `tts` run would fetch on demand, fetched up front instead.
///
/// The `s2` discriminator is deliberately not among them: only a fine-tune ever
/// opens one, so `train` fetches it and a synthesis-only user never pays the
/// 94 MB. Prosody is here because a default run does download it — leaving it
/// out would mean the first synthesis still went to the network.
#[derive(Debug, Args)]
pub struct DownloadArgs {
    /// Directory the downloaded models are cached in. Shared by every engine
    /// unless `$TTS_CACHE_DIR` (or `$VOICE_CACHE_DIR`) says otherwise.
    #[arg(long, default_value_os_t = hub_kit::cache_dir_for("TTS_CACHE_DIR"))]
    pub cache_dir: PathBuf,
    /// Skip the prosody encoder (~1.3 GB). Synthesis still runs without it,
    /// flatter on Chinese and unchanged on English, which is fed zeros anyway.
    #[arg(long)]
    pub no_prosody: bool,
}

pub async fn run(args: DownloadArgs) -> Result<()> {
    let dir = hub_kit::fetch_gptsovits(&args.cache_dir)
        .await
        .context("failed to fetch the GPT-SoVITS models")?;
    let paths = hub_kit::gptsovits_paths(&dir)?;

    // Where they landed is the point of the command, so it goes to stdout: the
    // bundle directory is what `--models` takes, and printing the three inside
    // it is how a hand-assembled directory gets checked against a fetched one.
    println!("models:  {}", dir.display());
    println!("  hubert: {}", paths.hubert.display());
    println!("  s1:     {}", paths.s1.display());
    println!("  s2:     {}", paths.s2.display());

    if !args.no_prosody {
        // An error rather than the warning synthesis gives: this command was
        // asked to fetch, so a half-fetched cache must not look like success.
        // The bundle above is already on disk, so a retry costs nothing.
        let prosody = hub_kit::fetch_prosody_bert(None, &args.cache_dir)
            .await
            .context("failed to fetch the prosody encoder (skip it with --no-prosody)")?;
        println!("prosody: {}", prosody.display());
    }
    Ok(())
}
