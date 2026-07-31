//! `rvc` as a library, so the `voice` binary can host the same voice-conversion
//! commands without duplicating a single flag definition.
//!
//! [`args::RvcCli`] is the whole command tree: the `rvc` binary flattens it at
//! its top level, `voice` nests it (`voice rvc convert`). Both parse the same
//! argument structs and dispatch through [`run`].

pub mod args;
pub mod commands;

use std::io::IsTerminal;

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
        Some(RvcCommand::Preprocess(a)) => commands::preprocess::run(a).await,
        Some(RvcCommand::Models(a)) => commands::models::run(a).await,
        Some(RvcCommand::Completions(a)) => cli_kit::completions(a, C::command()),
    }
}

/// Initialise tracing for a voice-conversion command.
///
/// Everything goes to stderr, except `train` with its dashboard up: the TUI owns
/// the terminal, so logs are redirected to `{work_dir}/train.log` instead.
pub fn init_logging(train: Option<&TrainArgs>) {
    let log_file = train
        .filter(|a| !a.no_tui && std::io::stdout().is_terminal())
        .map(|a| a.work_dir.join("train.log"));
    if cli_kit::init_logging(log_file.as_deref()) {
        eprintln!(
            "training dashboard active — logs: {}",
            log_file.expect("a path was used").display()
        );
    }
}
