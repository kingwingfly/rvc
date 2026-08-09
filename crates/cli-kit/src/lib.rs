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

/// Where a training run's log goes, given its `-o`.
///
/// `-o` names a *stem* (`models/voice` -> `models/voice.safetensors`), so the
/// output directory is its parent — `models/train.log`, beside the checkpoint
/// family and the `checkpoint/` best family. A stem with no directory part
/// (`-o voice`) writes its weights into the current directory, and the log
/// follows them there.
pub fn log_beside(out: &Path) -> PathBuf {
    out.parent().unwrap_or(Path::new("")).join("train.log")
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

/// The two knobs on how patient a download is, shared by every engine's
/// `download` subcommand.
///
/// Here rather than in each engine for the reason `Backend` is: four copies of
/// a flag are four spellings waiting to diverge, and `--backend` had already
/// done exactly that before it moved. It is also here rather than in `hub-kit`,
/// which owns the policy this produces, because `hub-kit` must stay free of
/// `clap` — it is the crate an engine's *core* depends on, and a download
/// policy is not an argument parser.
///
/// The spellings: `--download-timeout` keeps its prefix because "timeout" alone
/// would be read as a limit on the run, and this one bounds a transfer that has
/// gone quiet. `--retries` needs no prefix — inside a command whose whole job
/// is fetching, there is nothing else it could be counting.
#[derive(Debug, Clone, Copy, Args)]
pub struct DownloadOpts {
    /// Seconds a download may go without arriving before it is abandoned and
    /// started again. Not a limit on how long a download may take: the largest
    /// asset here is well over a gigabyte, and any limit generous enough for
    /// that on a slow link would be far too long to notice a stall.
    #[arg(long, value_name = "SECONDS", default_value_t = hub_kit::Retry::default().stall.as_secs())]
    pub download_timeout: u64,
    /// How many times a download that stalled or failed transiently is started
    /// again. A missing file is not retried at all: it would fail identically
    /// every time and only report it later.
    #[arg(long, value_name = "N", default_value_t = hub_kit::Retry::default().retries)]
    pub retries: u32,
}

impl DownloadOpts {
    /// Reject a pair that would fetch nothing.
    pub fn verify(&self) -> Result<()> {
        // Zero is not "no timeout" here, it is a window that has already
        // expired: every attempt would be abandoned before its first byte, so
        // the command would spend its whole retry budget and report a stall on
        // a link that is working. There is deliberately no spelling for "wait
        // forever" — that is the behaviour this flag exists to remove.
        anyhow::ensure!(
            self.download_timeout > 0,
            "--download-timeout must be at least 1 second: 0 abandons every \
             download before its first byte arrives"
        );
        Ok(())
    }

    /// Check the pair and make it this process's download policy.
    ///
    /// The one call a subcommand makes, so the check cannot be left out of a
    /// command that installs; [`Self::verify`] stays public because it is the
    /// house-shaped half and a caller that only validates has it.
    pub fn install(&self) -> Result<()> {
        self.verify()?;
        hub_kit::Retry {
            stall: std::time::Duration::from_secs(self.download_timeout),
            retries: self.retries,
        }
        .install();
        Ok(())
    }
}

/// How hard to de-hiss, shared by every command that runs the stage.
///
/// Here for the reason [`DownloadOpts`] and [`Backend`] are: two commands now
/// drive [`audio_kit::Denoiser`] — voice conversion as an optional stage on its
/// output, corpus preparation as a stage of its own — and three float flags
/// copied into two crates are two spellings and two sets of defaults waiting to
/// diverge. The `research > patch` check especially: it is not cosmetic, and a
/// copy that lost it would build a filter graph libavfilter refuses.
///
/// **Only the tuning lives here, not the on/off switch.** Voice conversion
/// needs one (`--denoise`), because de-hiss is an extra stage on a pipeline
/// that otherwise does not run it; corpus preparation does not, because there
/// the stage *is* the subcommand and a flag turning it off would leave a
/// command that copies files. So the gate stays with the caller that has one.
#[derive(Debug, Clone, Copy, Args)]
pub struct DenoiseOpts {
    /// De-hiss strength: raise to remove more hiss, lower if soft/breathy
    /// texture starts to smear.
    #[arg(long = "denoise-strength", default_value_t = 0.008)]
    pub denoise_strength: f32,
    /// `anlmdn` patch duration (seconds): the unit compared for
    /// self-similarity; smaller keeps finer detail.
    #[arg(long = "denoise-patch", default_value_t = 0.002)]
    pub denoise_patch: f32,
    /// `anlmdn` research window (seconds): how far in time it looks for similar
    /// patches. Must exceed the patch, and sets the de-hiss latency.
    #[arg(long = "denoise-research", default_value_t = 0.006)]
    pub denoise_research: f32,
}

impl DenoiseOpts {
    /// Reject a de-hiss configuration `anlmdn` would refuse or misbehave on.
    ///
    /// A caller whose de-hiss is optional checks this only when it is on: the
    /// flags have defaults, so checking them unconditionally would reject a run
    /// that never denoises anything.
    pub fn verify(&self) -> Result<()> {
        anyhow::ensure!(
            self.denoise_strength > 0.0,
            "--denoise-strength must be positive: zero leaves the audio unchanged"
        );
        anyhow::ensure!(self.denoise_patch > 0.0, "--denoise-patch must be positive");
        // Not cosmetic: `anlmdn` searches a window for patches to compare, so a
        // window no larger than the patch leaves it nothing to average over.
        anyhow::ensure!(
            self.denoise_research > self.denoise_patch,
            "--denoise-research ({}) must exceed --denoise-patch ({}): the research \
             window is where similar patches are looked for",
            self.denoise_research,
            self.denoise_patch
        );
        Ok(())
    }

    /// The [`audio_kit::DenoiseParams`] these flags describe.
    pub fn params(&self) -> audio_kit::DenoiseParams {
        audio_kit::DenoiseParams {
            strength: self.denoise_strength,
            patch_secs: self.denoise_patch,
            research_secs: self.denoise_research,
        }
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn log_goes_beside_the_weights() {
        assert_eq!(
            log_beside(Path::new("models/voice")),
            Path::new("models/train.log")
        );
        // `-o` may be given with the suffix it writes; the parent is the same.
        assert_eq!(
            log_beside(Path::new("out/run1/voice.safetensors")),
            Path::new("out/run1/train.log")
        );
        // A bare stem writes its weights into the current directory.
        assert_eq!(log_beside(Path::new("voice")), Path::new("train.log"));
    }
}
