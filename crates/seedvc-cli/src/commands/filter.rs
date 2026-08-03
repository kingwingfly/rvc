//! The bare invocation — a Unix filter: raw f32le PCM stdin -> stdout.
//!
//! Filled in by the unit that owns this file.

use anyhow::Result;

use crate::args::FilterArgs;

pub async fn run(_args: FilterArgs) -> Result<()> {
    anyhow::bail!("the streaming filter is not wired up yet")
}
