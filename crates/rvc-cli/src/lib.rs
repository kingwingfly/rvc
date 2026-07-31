//! `rvc` as a library, so the `voice` binary can host the same voice-conversion
//! subcommands without duplicating a single flag definition.
//!
//! `rvc` flattens [`args::RvcCommand`] at its top level (`rvc convert`); `voice`
//! nests it (`voice rvc convert`). Both parse the same argument structs and
//! dispatch through [`run_rvc`].

pub mod args;
pub mod commands;

use std::io::IsTerminal;

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
/// Everything goes to stderr, except `train` with its dashboard up: the TUI owns
/// the terminal, so logs are redirected to `./train.log` instead — the working
/// directory, so a run never scatters files into wherever `-o` points.
pub fn init_logging(train: Option<&TrainArgs>) {
    let log_file = train
        .filter(|a| !a.no_tui && std::io::stdout().is_terminal())
        .map(|_| std::path::PathBuf::from("train.log"));
    if cli_kit::init_logging(log_file.as_deref()) {
        eprintln!(
            "training dashboard active — logs: {}",
            log_file.expect("a path was used").display()
        );
    }
}
