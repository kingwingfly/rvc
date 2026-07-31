//! `stt` as a library, so the `voice` binary can host the same commands without
//! duplicating a single flag definition.
//!
//! [`args::SttCli`] is the whole command tree: the `stt` binary flattens it at
//! its top level, `voice` nests it (`voice stt`). Both parse the same argument
//! structs and dispatch through [`run`].

pub mod args;
pub mod backend;

use anyhow::Result;
use clap::CommandFactory;

pub use args::{Format, SttArgs, SttCli, SttCommand};
pub use cli_kit::Backend;

/// Run one recognition invocation.
///
/// `C` is the binary's own top-level command tree — the one `completions`
/// should describe, which is `stt` for the standalone tool and `voice` when
/// nested. It stays a type parameter so nothing but the `completions` arm pays
/// for building it.
pub async fn run<C: CommandFactory>(cli: SttCli) -> Result<()> {
    match cli.command {
        // Transcribing is the whole engine, so it is the bare invocation rather
        // than a subcommand; `completions` is the only thing beside it.
        None => args::transcribe(cli.transcribe).await,
        Some(SttCommand::Completions(a)) => cli_kit::completions(a, C::command()),
    }
}
