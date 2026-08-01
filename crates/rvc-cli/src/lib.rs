//! `rvc` as a library, so the `voice` binary can host the same voice-conversion
//! commands without duplicating a single flag definition.
//!
//! [`args::RvcCli`] is the whole command tree: the `rvc` binary flattens it at
//! its top level, `voice` nests it (`voice rvc convert`). Both parse the same
//! argument structs and dispatch through [`run`].

pub mod args;
pub mod commands;

use anyhow::Result;
use clap::CommandFactory;

use args::{RvcCli, RvcCommand, TrainArgs};

/// Run one voice-conversion invocation.
///
/// `C` is the binary's own top-level command tree — the one `completions`
/// should describe, which is `rvc` for the standalone tool and `voice` when
/// nested. It stays a type parameter so nothing but the `completions` arm pays
/// for building it.
pub async fn run<C: CommandFactory>(cli: RvcCli) -> Result<()> {
    match cli.command {
        // Converting a stream is the whole engine, so it is the bare
        // invocation; everything else is a subcommand beside it.
        None => commands::filter::run(cli.filter).await,
        Some(RvcCommand::Convert(a)) => commands::convert::run(a).await,
        Some(RvcCommand::Train(a)) => commands::train::run(*a).await,
        Some(RvcCommand::Preprocess(a)) => preprocess_kit::run(a).await,
        Some(RvcCommand::Models(a)) => commands::models::run(a).await,
        Some(RvcCommand::Completions(a)) => cli_kit::completions(a, C::command()),
    }
}

/// Initialise tracing for a voice-conversion command.
///
/// Everything goes to stderr, except `train` with its dashboard up: that one
/// gets `./train.log` — the working directory, so a run never scatters files
/// into wherever `-o` points. `cli_kit` owns the rest of the decision.
pub fn init_logging(train: Option<&TrainArgs>) {
    cli_kit::init_logging(
        train
            .filter(|a| !a.no_tui)
            .map(|_| std::path::PathBuf::from("train.log")),
    );
}
