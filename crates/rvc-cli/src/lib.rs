//! `rvc` as a library, so the `voice` binary can host the same voice-conversion
//! commands without duplicating a single flag definition.
//!
//! [`args::RvcCli`] is the whole command tree: the `rvc` binary flattens it at
//! its top level, `voice` nests it (`voice rvc convert`). Both parse the same
//! argument structs and dispatch through [`run`].

pub mod args;
pub mod commands;

use anyhow::Result;

use args::{RvcCli, RvcCommand, TrainArgs};

/// Run one voice-conversion invocation.
///
/// Not generic over the hosting binary: it used to take one so the `completions`
/// arm could build that binary's command tree, and `completions` now belongs to
/// the binary rather than to the engine.
pub async fn run(cli: RvcCli) -> Result<()> {
    match cli.command {
        // Converting a stream is the whole engine, so it is the bare
        // invocation; everything else is a subcommand beside it.
        None => commands::filter::run(cli.filter).await,
        Some(RvcCommand::Convert(a)) => commands::convert::run(a).await,
        Some(RvcCommand::Train(a)) => commands::train::run(*a).await,
        Some(RvcCommand::Download(a)) => commands::download::run(a).await,
    }
}

/// Initialise tracing for a voice-conversion command.
///
/// Everything goes to stderr, except `train` with its dashboard up: that one
/// gets `train.log` beside the weights the run writes, since the log is part of
/// what the run produced and belongs with the rest of it rather than in
/// whichever directory the run happened to be started from. `cli_kit` owns the
/// rest of the decision.
pub fn init_logging(train: Option<&TrainArgs>) {
    cli_kit::init_logging(
        train
            .filter(|a| !a.no_tui)
            .map(|a| cli_kit::log_beside(&a.out)),
    );
}
