//! `rvc` — CLI for RVC voice conversion.
//!
//! The bare invocation is the Unix filter: raw f32le mono PCM at 16 kHz on
//! stdin, converted PCM on stdout. Beside it sit the subcommands — `convert`
//! (batch files to WAV), `train`, `preprocess`, `models` and `completions`.
//!
//! This is the voice-conversion tool on its own. The `voice` binary hosts the
//! same command tree under `voice rvc …`, alongside `stt` and `tts`; install
//! whichever matches what you need.
//!
//! Logs go to **stderr** so the filter's stdout carries only PCM — except during
//! `train` with the TUI dashboard, where they go to `train.log` beside the
//! weights so they don't corrupt the display.

use anyhow::Result;
use clap::{CommandFactory, Parser, Subcommand};
use rvc_cli::args::{CompletionsArgs, FilterArgs, RvcCli, RvcCommand};

/// rvc — RVC voice conversion: f32le mono PCM @16 kHz on stdin, converted PCM
/// on stdout.
#[derive(Debug, Parser)]
#[command(name = "rvc", version, about)]
struct Cli {
    #[command(flatten)]
    filter: FilterArgs,
    #[command(subcommand)]
    command: Option<Command>,
}

/// The engine's own subcommands, plus the one that belongs to this binary.
///
/// `completions` is here rather than in [`RvcCommand`] because a completion
/// script describes an executable: under `voice` the executable is `voice`, and
/// a nested `voice rvc completions` could only emit `voice`'s script while
/// appearing to offer `rvc`'s.
#[derive(Debug, Subcommand)]
enum Command {
    #[command(flatten)]
    Engine(RvcCommand),
    /// Print a shell completion script (bash, zsh, fish, powershell, elvish).
    Completions(CompletionsArgs),
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    rvc_cli::init_logging(match &cli.command {
        Some(Command::Engine(RvcCommand::Train(a))) => Some(a.as_ref()),
        _ => None,
    });
    let command = match cli.command {
        Some(Command::Completions(a)) => return cli_kit::completions(a, Cli::command()),
        Some(Command::Engine(c)) => Some(c),
        None => None,
    };
    rvc_cli::run(RvcCli {
        filter: cli.filter,
        command,
    })
    .await
}
