//! `rvc` — CLI for RVC voice conversion.
//!
//! Subcommands:
//! - `convert` — batch-convert mp3 files to the target timbre (WAV out).
//! - `serve`   — realtime Unix filter: raw f32le PCM stdin -> stdout.
//! - `models`  — prefetch the shared ONNX assets from Hugging Face.
//! - `train`   — train an RVC generator natively in Rust (burn).
//!
//! This is the voice-conversion tool on its own. The `voice` binary hosts these
//! same subcommands under `voice rvc …`, alongside `stt` and `tts`; install
//! whichever matches what you need.
//!
//! Logs go to **stderr** so `serve`'s stdout carries only PCM — except during
//! `train` with the TUI dashboard, where they go to `./train.log` so they don't
//! corrupt the display.

use anyhow::Result;
use clap::{CommandFactory, Parser};
use rvc_cli::args::{Cli, Command, RvcCommand};
use rvc_cli::commands;

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    rvc_cli::init_logging(match &cli.command {
        Command::Rvc(c) => match c.as_ref() {
            RvcCommand::Train(a) => Some(a),
            _ => None,
        },
        _ => None,
    });
    match cli.command {
        Command::Rvc(c) => rvc_cli::run_rvc(*c).await,
        Command::Models(a) => commands::models::run(a).await,
        Command::Completions(a) => cli_kit::completions(a, Cli::command()),
    }
}
