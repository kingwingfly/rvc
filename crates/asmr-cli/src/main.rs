//! `asmr` — CLI for RVC voice conversion.
//!
//! Subcommands:
//! - `convert` — batch-convert mp3 files to the target timbre (WAV out).
//! - `serve`   — realtime Unix filter: raw f32le PCM stdin -> stdout.
//! - `models`  — prefetch the shared ONNX assets from Hugging Face.
//! - `train`   — train an RVC generator natively in Rust (burn).
//!
//! Logs go to **stderr** so `serve`'s stdout carries only PCM.

mod args;
mod commands;

use anyhow::Result;
use clap::Parser;

use args::{Cli, Command};

#[tokio::main]
async fn main() -> Result<()> {
    // Logs to stderr; keep stdout clean for piped PCM.
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let cli = Cli::parse();
    match cli.command {
        Command::Convert(a) => commands::convert::run(a).await,
        Command::Serve(a) => commands::serve::run(a).await,
        Command::Models(a) => commands::models::run(a).await,
        Command::Train(a) => commands::train::run(a).await,
    }
}
