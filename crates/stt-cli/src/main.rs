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
use clap::Parser;
use stt_cli::SttCli;

/// stt — speech recognition: f32le mono PCM @16 kHz on stdin, text on stdout.
#[derive(Debug, Parser)]
#[command(name = "stt", version, about)]
struct Cli {
    #[command(flatten)]
    stt: SttCli,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    cli_kit::init_logging(None);
    stt_cli::run::<Cli>(cli.stt).await
}
