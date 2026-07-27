//! `rvc` — CLI for RVC voice conversion.
//!
//! Subcommands:
//! - `convert` — batch-convert mp3 files to the target timbre (WAV out).
//! - `serve`   — realtime Unix filter: raw f32le PCM stdin -> stdout.
//! - `models`  — prefetch the shared ONNX assets from Hugging Face.
//! - `train`   — train an RVC generator natively in Rust (burn).
//!
//! Logs go to **stderr** so `serve`'s stdout carries only PCM — except during
//! `train` with the TUI dashboard, where they go to `{work_dir}/train.log` so
//! they don't corrupt the display.

mod args;
mod commands;

use std::io::IsTerminal;

use anyhow::Result;
use clap::Parser;
use tracing_subscriber::EnvFilter;

use args::{Cli, Command};

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    init_logging(&cli.command);
    match cli.command {
        Command::Convert(a) => commands::convert::run(a).await,
        Command::Serve(a) => commands::serve::run(a).await,
        Command::Models(a) => commands::models::run(a).await,
        Command::Train(a) => commands::train::run(a).await,
        Command::Preprocess(a) => commands::preprocess::run(a).await,
        Command::Completions(a) => commands::completions::run(a).await,
    }
}

/// Initialise tracing. Normally logs to stderr (keeping stdout clean for piped
/// PCM). For `train` with the dashboard (a TTY and no `--no-tui`), the TUI owns
/// the terminal, so logs are redirected to `{work_dir}/train.log` instead.
/// `mm:ss` since start. `tracing_subscriber`'s own uptime timer prints
/// nanoseconds, which is 12 columns of noise in front of every line.
struct Elapsed(std::time::Instant);

impl tracing_subscriber::fmt::time::FormatTime for Elapsed {
    fn format_time(&self, w: &mut tracing_subscriber::fmt::format::Writer<'_>) -> std::fmt::Result {
        let s = self.0.elapsed().as_secs();
        write!(w, "{:02}:{:02}", s / 60, s % 60)
    }
}

fn init_logging(command: &Command) {
    // App logs at `info`, but drop `ort`'s chatty INFO (per-tensor allocation /
    // static-memory-planning spam) to `warn`. Override the whole thing with
    // `RUST_LOG`, e.g. `RUST_LOG=ort=info`.
    let filter =
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info,ort=warn"));

    if let Command::Train(a) = command {
        let tui = !a.no_tui && std::io::stdout().is_terminal();
        if tui {
            let _ = std::fs::create_dir_all(&a.work_dir);
            let path = a.work_dir.join("train.log");
            if let Ok(file) = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&path)
            {
                tracing_subscriber::fmt()
                    .with_env_filter(filter)
                    .with_ansi(false)
                    .with_writer(move || file.try_clone().expect("clone log file handle"))
                    .init();
                eprintln!("training dashboard active — logs: {}", path.display());
                return;
            }
        }
    }

    // Compact on stderr: this is read by a person watching a run, not grepped for
    // module paths, and elapsed time answers "how long has this been going" better
    // than a wall-clock date that never changes mid-run.
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_env_filter(filter)
        .with_target(false)
        .with_timer(Elapsed(std::time::Instant::now()))
        .init();
}
