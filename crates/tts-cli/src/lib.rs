//! `tts` as a library, so the `voice` binary can host the same commands without
//! duplicating a single flag definition.
//!
//! [`args::TtsCli`] is the whole command tree: the `tts` binary flattens it at
//! its top level, `voice` nests it (`voice tts`, `voice tts train`). Both parse
//! the same argument structs and dispatch through [`run`].

pub mod args;
pub mod backend;
pub mod convert;
pub mod download;
pub mod train;

use anyhow::Result;

pub use args::{TtsArgs, TtsCli, TtsCommand};
pub use cli_kit::Backend;
pub use convert::ConvertArgs;
pub use train::{Stage, TrainArgs};

/// Run one synthesis invocation.
///
/// Not generic over the hosting binary: it used to take one so the `completions`
/// arm could build that binary's command tree, and `completions` now belongs to
/// the binary rather than to the engine.
pub async fn run(cli: TtsCli) -> Result<()> {
    match cli.command {
        // Synthesising is the whole engine, so it is the bare invocation;
        // everything else is a subcommand beside it.
        None => args::synthesize(cli.synth).await,
        Some(TtsCommand::Convert(a)) => convert::run(*a).await,
        Some(TtsCommand::Train(a)) => train::run(*a).await,
        Some(TtsCommand::Preprocess(a)) => preprocess_kit::run(a).await,
        Some(TtsCommand::Download(a)) => download::run(a).await,
    }
}

/// Start logging, sending a dashboard run's output to a file.
///
/// Lives here rather than in either binary so `tts` and `voice` cannot disagree
/// about it. All this engine decides is the path; `cli_kit::init_logging` knows
/// when a TUI is about to take the terminal and what to tell the user.
pub fn init_logging(train: Option<&TrainArgs>) {
    // Beside the weights, not in the working directory: the log is one of the
    // things the run produced, and `--stage both` writes two checkpoint families
    // that share one log.
    cli_kit::init_logging(
        train
            .filter(|a| !a.no_tui)
            .map(|a| cli_kit::log_beside(&a.out)),
    );
}
