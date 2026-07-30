//! `tts` — speech synthesis on its own.
//!
//! ```sh
//! echo "你好世界" | tts --reference clip.wav > out.f32le
//! ```
//!
//! Install this if synthesis is all you need; `voice` hosts the same subcommand
//! as `voice tts`, from the same code. Logs go to **stderr** so stdout carries
//! only audio.

use anyhow::Result;
use clap::{CommandFactory, Parser, Subcommand};
use cli_kit::CompletionsArgs;
use tts_cli::TtsArgs;

/// tts — speech synthesis: text on stdin, f32le mono PCM on stdout.
#[derive(Debug, Parser)]
#[command(name = "tts", version, about)]
struct Cli {
    #[command(flatten)]
    synth: TtsArgs,
    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Fine-tune GPT-SoVITS on a corpus of audio with transcripts.
    /// Boxed because it carries every knob of two training loops, and an enum is
    /// as large as its biggest variant.
    Train(Box<tts_cli::TrainArgs>),
    /// Print a shell completion script (bash, zsh, fish, powershell, elvish).
    Completions(CompletionsArgs),
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    tts_cli::init_logging(match &cli.command {
        Some(Command::Train(a)) => Some(a),
        _ => None,
    });
    match cli.command {
        None => tts_cli::run(cli.synth).await,
        Some(Command::Train(a)) => tts_cli::train::run(*a).await,
        Some(Command::Completions(a)) => cli_kit::completions(a, Cli::command()),
    }
}
