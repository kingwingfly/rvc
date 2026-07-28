//! The plumbing every binary in the toolkit needs and none of them should own.
//!
//! `rvc`, `stt` and `voice` each ship separately, so anything shared between
//! them has to live somewhere that pulls in no engine — otherwise installing the
//! recognition tool drags in voice conversion for the sake of a log formatter.

use std::path::Path;

use anyhow::Result;
use clap::Args;
use clap_complete::Shell;

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
/// carries data. `log_file` redirects them to a file instead — which `rvc train`
/// needs, since its dashboard owns the terminal. Returns whether the file was
/// used, so the caller can say where the logs went.
pub fn init_logging(log_file: Option<&Path>) -> bool {
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
