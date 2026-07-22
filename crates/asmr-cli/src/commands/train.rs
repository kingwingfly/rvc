//! `asmr train` — thin wrapper that drives the Python/uv RVC training pipeline.
//!
//! Training is a one-time, offline step. We shell out to `uv run` in the
//! `training/` project so the whole heavyweight Python/torch stack stays out of
//! the shipped binary. stdio is inherited so the user sees training progress.

use anyhow::{bail, Context, Result};
use tokio::process::Command;

use crate::args::TrainArgs;

pub async fn run(args: TrainArgs) -> Result<()> {
    if !args.project.join("pyproject.toml").exists() {
        bail!(
            "training project not found at {} (expected a uv pyproject.toml)",
            args.project.display()
        );
    }

    // uv run --project <dir> python -m asmr_train --out <out> -- <data...> [extra]
    let mut cmd = Command::new("uv");
    cmd.arg("run")
        .arg("--project")
        .arg(&args.project)
        .args(["python", "-m", "asmr_train"])
        .arg("--out")
        .arg(&args.out);

    for path in &args.data {
        cmd.arg(path);
    }
    if !args.extra.is_empty() {
        cmd.arg("--").args(&args.extra);
    }

    tracing::info!("launching training via uv (project: {})", args.project.display());
    let status = cmd
        .status()
        .await
        .context("failed to launch `uv` (is it installed and on PATH?)")?;

    if !status.success() {
        bail!("training exited with status {status}");
    }
    tracing::info!("training complete; generator written to {}", args.out.display());
    Ok(())
}
