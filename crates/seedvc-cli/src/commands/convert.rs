//! `seedvc convert` — batch files in, WAV files out.
//!
//! Filled in by the unit that owns this file.

use anyhow::Result;

use crate::args::ConvertArgs;

pub async fn run(_args: Box<ConvertArgs>) -> Result<()> {
    anyhow::bail!("batch conversion is not wired up yet")
}
