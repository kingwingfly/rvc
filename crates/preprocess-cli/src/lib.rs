//! `preprocess` as a library, so the `voice` binary can host the same corpus
//! commands without duplicating a single flag definition.
//!
//! [`args::PreprocessCli`] is the whole command tree: the `preprocess` binary
//! flattens it at its top level, `voice` nests it (`voice preprocess clip`).
//! Both parse the same argument structs and dispatch through [`run`].
//!
//! # There is no bare invocation, and that is not the streaming rule being
//! broken
//!
//! Every other binary here reads a pipe when given no subcommand, and the rule
//! that put it there is emphatic: streaming is the *primary* mode of a Unix
//! filter, `rvc serve` was promoted to the bare invocation for exactly that
//! reason, and a future engine with a streaming mode gets the same treatment.
//!
//! Read literally, that rule would have this binary grow one too. It must not,
//! and the difference is what the name denotes. `rvc`, `stt`, `tts` and
//! `seedvc` each name **an engine** — one transformation, so "the engine on a
//! pipe" is a complete description of what a bare invocation does.
//! `preprocess` names **a phase**, and which stage of it to run is precisely
//! what the subcommand chooses. A bare `preprocess` would have to pick one
//! silently, and whichever it picked would be wrong for everybody who wanted a
//! different one. That is not a missing feature to be filled in later: adding
//! more stages makes it *worse*, and the whole reason this crate exists is that
//! more stages are coming.
//!
//! So the test to apply before adding one is not "does this binary stream" but
//! "**is there one thing a bare invocation would mean**". If a stage here ever
//! wants to be a filter, it gets it as a property of that stage — `preprocess
//! denoise` reading stdin when given no files, say — and the top level stays a
//! chooser.

pub mod args;
pub mod commands;

use anyhow::Result;

pub use args::{PreprocessCli, PreprocessCommand};

/// Run one corpus-preparation invocation.
///
/// Not generic over the hosting binary: `completions` belongs to the binary
/// rather than to the stages, so nothing here needs to know which executable is
/// hosting it.
pub async fn run(cli: PreprocessCli) -> Result<()> {
    match cli.command {
        PreprocessCommand::Clip(a) => commands::clip::run(a).await,
        PreprocessCommand::Denoise(a) => commands::denoise::run(a).await,
    }
}

/// Initialise tracing.
///
/// Everything goes to stderr, and there is nothing to route around: no stage
/// here writes data to stdout, and none of them raises a dashboard. Its own
/// function all the same, so `preprocess` and `voice` cannot disagree about it
/// the moment one of those stops being true.
pub fn init_logging() {
    cli_kit::init_logging(None);
}
