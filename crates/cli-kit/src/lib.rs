//! The plumbing every binary in the toolkit needs and none of them should own.
//!
//! `rvc`, `stt` and `voice` each ship separately, so anything shared between
//! them has to live somewhere that pulls in no engine — otherwise installing the
//! recognition tool drags in voice conversion for the sake of a log formatter.

mod backend;

use std::io::IsTerminal;
use std::path::{Path, PathBuf};

use anyhow::Result;
use clap::Args;
use clap_complete::Shell;

pub use backend::Backend;

/// `mm:ss` since start. `tracing_subscriber`'s own uptime timer prints
/// nanoseconds, which is 12 columns of noise in front of every line.
struct Elapsed(std::time::Instant);

impl tracing_subscriber::fmt::time::FormatTime for Elapsed {
    fn format_time(&self, w: &mut tracing_subscriber::fmt::format::Writer<'_>) -> std::fmt::Result {
        let s = self.0.elapsed().as_secs();
        write!(w, "{:02}:{:02}", s / 60, s % 60)
    }
}

/// Initialise tracing.
///
/// Logs go to **stderr**, because every subcommand here is a filter whose stdout
/// carries data. A training run is the exception: its dashboard owns the
/// terminal and stderr would scribble over it, so a caller about to raise one
/// passes the file the logs should go to instead, and the user is told where
/// they went. Where that file lives is the caller's decision — each engine has
/// its own layout — and passing one when stdout is *not* a terminal is not an
/// error: no dashboard comes up, so nothing needs redirecting.
pub fn init_logging(log_file: Option<PathBuf>) {
    if let Some(path) = log_file.filter(|_| std::io::stdout().is_terminal()) {
        // Only announce the file once it is known to be the one in use:
        // `init_tracing` falls back to stderr if it cannot be opened.
        if init_tracing(Some(&path)) {
            eprintln!("training dashboard active — logs: {}", path.display());
        }
    } else {
        init_tracing(None);
    }
}

/// Install the subscriber, returning whether `log_file` was the one it got.
fn init_tracing(log_file: Option<&Path>) -> bool {
    use tracing_subscriber::EnvFilter;

    // App logs at `info`, but drop `ort`'s chatty INFO (per-tensor allocation /
    // static-memory-planning spam) to `warn`. Override the whole thing with
    // `RUST_LOG`, e.g. `RUST_LOG=ort=info`.
    let filter =
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info,ort=warn"));

    if let Some(path) = log_file {
        if let Some(dir) = path.parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        if let Ok(file) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
        {
            tracing_subscriber::fmt()
                .with_env_filter(filter)
                .with_ansi(false)
                .with_writer(move || file.try_clone().expect("clone log file handle"))
                .init();
            return true;
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
    false
}

#[derive(Debug, Args)]
pub struct CompletionsArgs {
    /// Shell to generate the completion script for.
    #[arg(value_enum)]
    pub shell: Shell,
}

/// Print a shell completion script to stdout.
///
/// `clap_complete` derives it from the same command tree the parser uses, so
/// completions never drift from the actual flags. The caller passes its own
/// tree, which is how one implementation serves three binaries, and the binary
/// name is read off that tree so the script and the thing it completes cannot
/// disagree.
pub fn completions(args: CompletionsArgs, mut cmd: clap::Command) -> Result<()> {
    let bin = cmd.get_name().to_string();
    clap_complete::generate(args.shell, &mut cmd, bin, &mut std::io::stdout());
    Ok(())
}

/// Parse `--device` in clap, so a typo is a usage error rather than a late
/// failure after the model has already been loaded.
pub fn parse_device(s: &str) -> std::result::Result<burn_kit::DeviceSpec, String> {
    s.parse()
}

/// The early-stop flag every training subcommand hands its trainer.
///
/// Ctrl-C flips it and the loop saves what it has rather than dying with the
/// run's work unwritten — which is why this is a flag polled between steps and
/// not a process exit. With a dashboard up Ctrl-C is captured as a key instead,
/// so `q` is how a TUI run stops.
pub fn stop_on_ctrl_c() -> std::sync::Arc<std::sync::atomic::AtomicBool> {
    use std::sync::atomic::{AtomicBool, Ordering};

    let stop = std::sync::Arc::new(AtomicBool::new(false));
    tokio::spawn({
        let stop = stop.clone();
        async move {
            if tokio::signal::ctrl_c().await.is_ok() {
                stop.store(true, Ordering::Relaxed);
            }
        }
    });
    stop
}

/// Whether a training run should raise the TUI dashboard.
///
/// The dashboard owns the terminal, so this must agree with [`init_logging`]'s
/// decision to route logs to a file: a run that shows the dashboard *and* logs
/// to stderr scribbles over its own display.
pub fn use_tui(no_tui: bool) -> bool {
    !no_tui && std::io::IsTerminal::is_terminal(&std::io::stdout())
}
