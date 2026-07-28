//! `voice` — the full speech toolkit: recognition, synthesis and voice
//! conversion, each one a Unix filter so they compose in a pipe.
//!
//! ```sh
//! ffmpeg -i take.mp3 -f f32le -ar 16000 -ac 1 - \
//!   | voice rvc serve -m voice.safetensors --model-sr 48000 \
//!   | ffplay -f f32le -ar 48000 -ac 1 -
//! ```
//!
//! `voice rvc …` hosts exactly the subcommands of the standalone `rvc` binary,
//! from the same code — install `rvc` on its own if voice conversion is all you
//! need, and `voice` if you want the rest of the toolkit with it.
//!
//! Logs go to **stderr** so a filter's stdout carries only data.

mod args;

use anyhow::Result;
use clap::{CommandFactory, Parser};
use rvc_cli::args::RvcCommand;
use rvc_cli::commands;

use args::{Cli, Command};

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    rvc_cli::init_logging(match &cli.command {
        Command::Rvc { command } => match command.as_ref() {
            RvcCommand::Train(a) => Some(a),
            _ => None,
        },
        _ => None,
    });
    match cli.command {
        Command::Rvc { command } => rvc_cli::run_rvc(*command).await,
        Command::Stt(a) => stt_cli::run(*a).await,
        Command::Tts(a) => tts_cli::run(*a).await,
        Command::Models(a) => commands::models::run(a).await,
        Command::Completions(a) => cli_kit::completions(a, Cli::command()),
    }
}
