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
use clap::Parser;
use seedvc_cli::SeedVcCli;

/// seedvc — convert a voice into the voice of a reference clip, with no training.
#[derive(Debug, Parser)]
#[command(name = "seedvc", version, about)]
struct Cli {
    #[command(flatten)]
    seedvc: SeedVcCli,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    seedvc_cli::init_logging();
    seedvc_cli::run::<Cli>(cli.seedvc).await
}
