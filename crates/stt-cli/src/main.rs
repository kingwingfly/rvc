//! `stt` — speech recognition on its own.
//!
//! ```sh
//! ffmpeg -i take.mp3 -f f32le -ar 16000 -ac 1 - | stt --format jsonl
//! ```
//!
//! Install this if recognition is all you need; `voice` hosts the same command
//! tree as `voice stt`, from the same code, alongside the rest of the toolkit.
//! Logs go to **stderr** so stdout carries only transcripts.

use anyhow::Result;
use clap::{CommandFactory, Parser, Subcommand};
use cli_kit::CompletionsArgs;
use stt_cli::args::SttArgs;
use stt_cli::{SttCli, SttCommand};

/// stt — speech recognition: f32le mono PCM @16 kHz on stdin, text on stdout.
#[derive(Debug, Parser)]
#[command(name = "stt", version, about)]
struct Cli {
    #[command(flatten)]
    transcribe: SttArgs,
    #[command(subcommand)]
    command: Option<Command>,
}

/// The engine's own subcommands, plus the one that belongs to this binary.
///
/// `completions` is here rather than in [`SttCommand`] because a completion
/// script describes an executable: under `voice` the executable is `voice`, and
/// a nested `voice stt completions` could only emit `voice`'s script while
/// appearing to offer `stt`'s.
#[derive(Debug, Subcommand)]
enum Command {
    #[command(flatten)]
    Engine(SttCommand),
    /// Print a shell completion script (bash, zsh, fish, powershell, elvish).
    Completions(CompletionsArgs),
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    cli_kit::init_logging(None);
    let command = match cli.command {
        Some(Command::Completions(a)) => return cli_kit::completions(a, Cli::command()),
        Some(Command::Engine(c)) => Some(c),
        None => None,
    };
    stt_cli::run(SttCli {
        transcribe: cli.transcribe,
        command,
    })
    .await
}
