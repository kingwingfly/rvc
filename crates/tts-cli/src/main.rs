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
use clap::Parser;
use tts_cli::{TtsCli, TtsCommand};

/// tts — speech synthesis: text on stdin, f32le mono PCM on stdout.
#[derive(Debug, Parser)]
#[command(name = "tts", version, about)]
struct Cli {
    #[command(flatten)]
    tts: TtsCli,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    tts_cli::init_logging(match &cli.tts.command {
        Some(TtsCommand::Train(a)) => Some(a.as_ref()),
        _ => None,
    });
    tts_cli::run::<Cli>(cli.tts).await
}
