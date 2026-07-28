//! `stt` — speech recognition on its own.
//!
//! ```sh
//! ffmpeg -i take.mp3 -f f32le -ar 16000 -ac 1 - | stt --format jsonl
//! ```
//!
//! Install this if recognition is all you need; `voice` hosts the same
//! subcommand as `voice stt`, from the same code, alongside the rest of the
//! toolkit. Logs go to **stderr** so stdout carries only transcripts.

use anyhow::Result;
use clap::{CommandFactory, Parser, Subcommand};
use cli_kit::CompletionsArgs;
use stt_cli::SttArgs;

/// stt — speech recognition: f32le mono PCM @16 kHz on stdin, text on stdout.
#[derive(Debug, Parser)]
#[command(name = "stt", version, about)]
struct Cli {
    #[command(flatten)]
    transcribe: SttArgs,
    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Print a shell completion script (bash, zsh, fish, powershell, elvish).
    Completions(CompletionsArgs),
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    cli_kit::init_logging(None);
    match cli.command {
        // Transcribing is the whole tool, so it is the bare invocation rather
        // than a subcommand; `completions` is the only thing beside it.
        None => stt_cli::run(cli.transcribe).await,
        Some(Command::Completions(a)) => cli_kit::completions(a, Cli::command()),
    }
}
