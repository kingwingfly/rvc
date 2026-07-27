//! `rvc` as a library, so the `voice` binary can host the same voice-conversion
//! subcommands without duplicating a single flag definition.
//!
//! `rvc` flattens [`args::RvcCommand`] at its top level (`rvc convert`); `voice`
//! nests it (`voice rvc convert`). Both parse the same argument structs and
//! dispatch through [`run_rvc`].

pub mod args;
pub mod commands;

use std::io::IsTerminal;

use anyhow::Result;
use tracing_subscriber::EnvFilter;

use args::{RvcCommand, TrainArgs};

/// Run one voice-conversion subcommand.
pub async fn run_rvc(cmd: RvcCommand) -> Result<()> {
    match cmd {
        RvcCommand::Convert(a) => commands::convert::run(a).await,
        RvcCommand::Serve(a) => commands::serve::run(a).await,
        RvcCommand::Train(a) => commands::train::run(a).await,
        RvcCommand::Preprocess(a) => commands::preprocess::run(a).await,
    }
}

/// `mm:ss` since start. `tracing_subscriber`'s own uptime timer prints
/// nanoseconds, which is 12 columns of noise in front of every line.
struct Elapsed(std::time::Instant);

impl tracing_subscriber::fmt::time::FormatTime for Elapsed {
    fn format_time(&self, w: &mut tracing_subscriber::fmt::format::Writer<'_>) -> std::fmt::Result {
        let s = self.0.elapsed().as_secs();
        write!(w, "{:02}:{:02}", s / 60, s % 60)
    }
}

/// Initialise tracing. Normally logs to stderr (keeping stdout clean for piped
/// PCM). Pass the [`TrainArgs`] when the command being run is `train`: with the
/// dashboard up (a TTY and no `--no-tui`) the TUI owns the terminal, so logs go
/// to `{work_dir}/train.log` instead.
pub fn init_logging(train: Option<&TrainArgs>) {
    // App logs at `info`, but drop `ort`'s chatty INFO (per-tensor allocation /
    // static-memory-planning spam) to `warn`. Override the whole thing with
    // `RUST_LOG`, e.g. `RUST_LOG=ort=info`.
    let filter =
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info,ort=warn"));

    if let Some(a) = train {
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
