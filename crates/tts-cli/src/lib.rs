//! `tts` as a library, so the `voice` binary can host the same commands without
//! duplicating a single flag definition.
//!
//! [`args::TtsCli`] is the whole command tree: the `tts` binary flattens it at
//! its top level, `voice` nests it (`voice tts`, `voice tts train`). Both parse
//! the same argument structs and dispatch through [`run`].

pub mod args;
pub mod backend;
pub mod train;

use anyhow::Result;
use clap::CommandFactory;

pub use args::{TtsArgs, TtsCli, TtsCommand};
pub use cli_kit::Backend;
pub use train::{Stage, TrainArgs};

/// Run one synthesis invocation.
///
/// `C` is the binary's own top-level command tree — the one `completions`
/// should describe, which is `tts` for the standalone tool and `voice` when
/// nested. It stays a type parameter so nothing but the `completions` arm pays
/// for building it.
pub async fn run<C: CommandFactory>(cli: TtsCli) -> Result<()> {
    match cli.command {
        // Synthesising is the whole engine, so it is the bare invocation;
        // everything else is a subcommand beside it.
        None => args::synthesize(cli.synth).await,
        Some(TtsCommand::Train(a)) => train::run(*a).await,
        Some(TtsCommand::Completions(a)) => cli_kit::completions(a, C::command()),
    }
}

/// Start logging, sending a dashboard run's output to a file.
///
/// The TUI owns the terminal for the length of a fine-tune, so tracing on stderr
/// would scribble over it. Mirrors `rvc-cli`'s helper, and lives here rather
/// than in either binary so `tts` and `voice` cannot disagree about it.
pub fn init_logging(train: Option<&TrainArgs>) {
    use std::io::IsTerminal;

    let log_file = train
        .filter(|a| !a.no_tui && std::io::stdout().is_terminal())
        // The working directory, not `-o`'s: a run writes its weights where it
        // was told to and its scratch where it was started, so an output
        // directory holds models and nothing else.
        .map(|_| std::path::PathBuf::from("train.log"));
    if cli_kit::init_logging(log_file.as_deref()) {
        eprintln!(
            "training dashboard active — logs: {}",
            log_file.expect("a path was used").display()
        );
    }
}
