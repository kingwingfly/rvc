//! `seedvc` as a library, so the `voice` binary can host the same commands
//! without duplicating a single flag definition.
//!
//! [`args::SeedVcCli`] is the whole command tree: the `seedvc` binary flattens it
//! at its top level, `voice` nests it (`voice seedvc`). Both parse the same
//! argument structs and dispatch through [`run`].
//!
//! There is no `train` here and there will not be one — Seed-VC is zero-shot, so
//! a reference clip does the job a fine-tune does in the other engines.

pub mod args;
pub mod commands;

use anyhow::Result;

pub use args::{SeedVcCli, SeedVcCommand};
pub use cli_kit::Backend;

/// Run one conversion invocation.
///
/// Not generic over the hosting binary: it used to take one so the `completions`
/// arm could build that binary's command tree, and `completions` now belongs to
/// the binary rather than to the engine.
pub async fn run(cli: SeedVcCli) -> Result<()> {
    match cli.command {
        // Converting is the whole engine, so it is the bare invocation;
        // everything else is a subcommand beside it.
        None => commands::filter::run(cli.filter).await,
        Some(SeedVcCommand::Convert(a)) => commands::convert::run(a).await,
        Some(SeedVcCommand::Download(a)) => commands::download::run(a).await,
    }
}

/// Initialise tracing.
///
/// Everything goes to stderr, because the bare invocation's stdout carries PCM.
/// Unlike `rvc` and `tts` there is no dashboard to route around — this engine
/// has no training loop.
pub fn init_logging() {
    cli_kit::init_logging(None);
}
