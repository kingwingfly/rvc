//! `rvc` as a library, so the `voice` binary can host the same voice-conversion
//! subcommands without duplicating a single flag definition.
//!
//! `rvc` flattens [`args::RvcCommand`] at its top level (`rvc convert`); `voice`
//! nests it (`voice rvc convert`). Both parse the same argument structs and
//! dispatch through [`run_rvc`].

pub mod args;
pub mod commands;

use anyhow::Result;

use args::{RvcCommand, TrainArgs};

/// Run one voice-conversion subcommand.
pub async fn run_rvc(cmd: RvcCommand) -> Result<()> {
    match cmd {
        RvcCommand::Convert(a) => commands::convert::run(a).await,
        RvcCommand::Serve(a) => commands::serve::run(a).await,
        RvcCommand::Train(a) => commands::train::run(a).await,
        RvcCommand::Preprocess(a) => commands::preprocess::run(a).await,
    }
}

/// Initialise tracing for a voice-conversion command.
///
/// Everything goes to stderr, except `train` with its dashboard up: that one
/// gets `{work_dir}/train.log`, which is the only path this engine has to
/// choose — `cli_kit` owns the rest of the decision.
pub fn init_logging(train: Option<&TrainArgs>) {
    cli_kit::init_logging(
        train
            .filter(|a| !a.no_tui)
            .map(|a| a.work_dir.join("train.log")),
    );
}
