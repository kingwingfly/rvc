//! `stt` as a library, so the `voice` binary can host the same commands without
//! duplicating a single flag definition.
//!
//! [`args::SttCli`] is the whole command tree: the `stt` binary flattens it at
//! its top level, `voice` nests it (`voice stt`). Both parse the same argument
//! structs and dispatch through [`run`].

pub mod args;
pub mod backend;
pub mod convert;

use anyhow::Result;

pub use args::{Format, SttArgs, SttCli, SttCommand};
pub use cli_kit::Backend;
pub use convert::ConvertArgs;

/// Run one recognition invocation.
///
/// Not generic over the hosting binary: it used to take one so the `completions`
/// arm could build that binary's command tree, and `completions` now belongs to
/// the binary rather than to the engine.
pub async fn run(cli: SttCli) -> Result<()> {
    match cli.command {
        // Transcribing is the whole engine, so it is the bare invocation;
        // everything else is a subcommand beside it.
        None => args::transcribe(cli.transcribe).await,
        Some(SttCommand::Convert(a)) => convert::run(a).await,
        Some(SttCommand::Download(a)) => args::download(a).await,
    }
}
