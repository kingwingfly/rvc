//! `seedvc download` — prefetch what a conversion fetches on its first run.
//!
//! Filled in by the unit that owns this file.

use anyhow::Result;

use crate::args::DownloadArgs;

pub async fn run(_args: DownloadArgs) -> Result<()> {
    anyhow::bail!("asset download is not wired up yet")
}
