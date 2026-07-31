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
/// Lives here rather than in either binary so `tts` and `voice` cannot disagree
/// about it. All this engine decides is the path; `cli_kit::init_logging` knows
/// when a TUI is about to take the terminal and what to tell the user.
pub fn init_logging(train: Option<&TrainArgs>) {
    cli_kit::init_logging(train.filter(|a| !a.no_tui).map(|a| {
        // Beside the weights, which is the one directory the user already named.
        a.out
            .parent()
            .unwrap_or(std::path::Path::new(""))
            .join("train.log")
    }));
}
