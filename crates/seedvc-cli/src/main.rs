//! `seedvc` — zero-shot voice conversion as a Unix filter.
//!
//! Raw f32le mono PCM at 16 kHz on stdin, converted PCM on stdout, logs on
//! stderr. The voice comes from a reference recording rather than from a trained
//! model, so `-r` is the only thing that decides who the output sounds like:
//!
//! ```sh
//! ffmpeg -i take.mp3 -f f32le -ar 16000 -ac 1 - \
//!   | seedvc -r target-voice.wav \
//!   | ffplay -f f32le -ar 22050 -ac 1 -
//! ```
//!
//! `voice seedvc …` hosts the same command tree, from this crate as a library.

use anyhow::Result;
use clap::{CommandFactory, Parser, Subcommand};
use seedvc_cli::args::{CompletionsArgs, FilterArgs, SeedVcCli, SeedVcCommand};

/// seedvc — convert a voice into the voice of a reference clip, with no training.
#[derive(Debug, Parser)]
#[command(name = "seedvc", version, about)]
struct Cli {
    #[command(flatten)]
    filter: FilterArgs,
    #[command(subcommand)]
    command: Option<Command>,
}

/// The engine's own subcommands, plus the one that belongs to this binary.
///
/// `completions` is here rather than in [`SeedVcCommand`] because a completion
/// script describes an executable: under `voice` the executable is `voice`, and
/// a nested `voice seedvc completions` could only emit `voice`'s script while
/// appearing to offer `seedvc`'s.
#[derive(Debug, Subcommand)]
enum Command {
    #[command(flatten)]
    Engine(SeedVcCommand),
    /// Print a shell completion script (bash, zsh, fish, powershell, elvish).
    Completions(CompletionsArgs),
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    seedvc_cli::init_logging();
    let command = match cli.command {
        Some(Command::Completions(a)) => return cli_kit::completions(a, Cli::command()),
        Some(Command::Engine(c)) => Some(c),
        None => None,
    };
    seedvc_cli::run(SeedVcCli {
        filter: cli.filter,
        command,
    })
    .await
}
