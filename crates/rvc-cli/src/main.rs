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
//! `train` with the TUI dashboard, where they go to `{work_dir}/train.log` so
//! they don't corrupt the display.

use anyhow::Result;
use clap::Parser;
use rvc_cli::args::{RvcCli, RvcCommand};

/// rvc — RVC voice conversion: f32le mono PCM @16 kHz on stdin, converted PCM
/// on stdout.
#[derive(Debug, Parser)]
#[command(name = "rvc", version, about)]
struct Cli {
    #[command(flatten)]
    rvc: RvcCli,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    rvc_cli::init_logging(match &cli.rvc.command {
        Some(RvcCommand::Train(a)) => Some(a.as_ref()),
        _ => None,
    });
    rvc_cli::run::<Cli>(cli.rvc).await
}
