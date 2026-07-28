//! `tts` as a library, so the `voice` binary can host the same subcommand
//! without duplicating a single flag definition.

pub mod args;
pub mod backend;
pub mod train;

pub use args::{TtsArgs, run};
pub use backend::TtsBackend;
pub use train::TrainArgs;
