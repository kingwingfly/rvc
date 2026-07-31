//! `tts` as a library, so the `voice` binary can host the same subcommand
//! without duplicating a single flag definition.

pub mod args;
pub mod backend;
pub mod train;

pub use args::{TtsArgs, run};
pub use backend::TtsBackend;
pub use train::{Stage, TrainArgs};

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
