//! `tts` — speech synthesis on its own.
//!
//! ```sh
//! echo "你好世界" | tts --reference clip.wav > out.f32le
//! ```
//!
//! Install this if synthesis is all you need; `voice` hosts the same command
//! tree as `voice tts`, from the same code. Logs go to **stderr** so stdout
//! carries only audio.

use anyhow::Result;
use clap::{CommandFactory, Parser, Subcommand};
use cli_kit::CompletionsArgs;
use tts_cli::args::TtsArgs;
use tts_cli::{TtsCli, TtsCommand};

/// tts — speech synthesis: text on stdin, f32le mono PCM on stdout.
#[derive(Debug, Parser)]
#[command(name = "tts", version, about)]
struct Cli {
    #[command(flatten)]
    synth: TtsArgs,
    #[command(subcommand)]
    command: Option<Command>,
}

/// The engine's own subcommands, plus the one that belongs to this binary.
///
/// `completions` is here rather than in [`TtsCommand`] because a completion
/// script describes an executable: under `voice` the executable is `voice`, and
/// a nested `voice tts completions` could only emit `voice`'s script while
/// appearing to offer `tts`'s.
#[derive(Debug, Subcommand)]
enum Command {
    #[command(flatten)]
    Engine(TtsCommand),
    /// Print a shell completion script (bash, zsh, fish, powershell, elvish).
    Completions(CompletionsArgs),
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    tts_cli::init_logging(match &cli.command {
        Some(Command::Engine(TtsCommand::Train(a))) => Some(a.as_ref()),
        _ => None,
    });
    let command = match cli.command {
        Some(Command::Completions(a)) => return cli_kit::completions(a, Cli::command()),
        Some(Command::Engine(c)) => Some(c),
        None => None,
    };
    tts_cli::run(TtsCli {
        synth: cli.synth,
        command,
    })
    .await
}
