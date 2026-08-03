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
use clap::CommandFactory;

pub use args::{SeedVcCli, SeedVcCommand};
pub use cli_kit::Backend;

/// Run one conversion invocation.
///
/// `C` is the binary's own top-level command tree — the one `completions` should
/// describe, which is `seedvc` for the standalone tool and `voice` when nested.
/// It stays a type parameter so nothing but the `completions` arm pays for
/// building it.
pub async fn run<C: CommandFactory>(cli: SeedVcCli) -> Result<()> {
    match cli.command {
        // Converting is the whole engine, so it is the bare invocation;
        // everything else is a subcommand beside it.
        None => commands::filter::run(cli.filter).await,
        Some(SeedVcCommand::Convert(a)) => commands::convert::run(a).await,
        Some(SeedVcCommand::Download(a)) => commands::download::run(a).await,
        Some(SeedVcCommand::Completions(a)) => cli_kit::completions(a, C::command()),
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
