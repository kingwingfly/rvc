//! `asmr tts` — GPT-SoVITS text-to-speech via a Python/uv sidecar.
//!
//! GPT-SoVITS' ONNX export is only partial (text frontend, BERT, SSL), so TTS
//! runs through the `asmr_tts` sidecar in the `training/` uv project rather than
//! pure Rust. The Rust side keeps the same ergonomic CLI and streams the
//! sidecar's stdio through.

use anyhow::{bail, Context, Result};
use tokio::process::Command;

use crate::args::{SidecarOpts, TtsArgs, TtsCommand, TtsFinetuneArgs, TtsSpeakArgs};

pub async fn run(args: TtsArgs) -> Result<()> {
    match args.command {
        TtsCommand::Speak(a) => speak(a).await,
        TtsCommand::Finetune(a) => finetune(a).await,
    }
}

/// Base `uv run --project <project> python -m asmr_tts <sub>` command.
fn sidecar_cmd(sidecar: &SidecarOpts, subcommand: &str) -> Result<Command> {
    let pyproject = sidecar.project.join("pyproject.toml");
    if !pyproject.exists() {
        bail!(
            "sidecar project not found at {} (expected a uv pyproject.toml)",
            sidecar.project.display()
        );
    }
    let mut cmd = Command::new("uv");
    cmd.arg("run")
        .arg("--project")
        .arg(&sidecar.project)
        .args(["python", "-m", "asmr_tts", subcommand]);
    Ok(cmd)
}

fn append_extra(cmd: &mut Command, extra: &[String]) {
    if !extra.is_empty() {
        cmd.arg("--").args(extra);
    }
}

async fn run_sidecar(mut cmd: Command, what: &str) -> Result<()> {
    tracing::info!("running GPT-SoVITS sidecar: {what}");
    let status = cmd
        .status()
        .await
        .context("failed to launch `uv` (is it installed and on PATH?)")?;
    if !status.success() {
        bail!("tts {what} exited with status {status}");
    }
    Ok(())
}

async fn speak(a: TtsSpeakArgs) -> Result<()> {
    let mut cmd = sidecar_cmd(&a.sidecar, "speak")?;
    cmd.arg("--text")
        .arg(&a.text)
        .arg("--ref")
        .arg(&a.ref_audio)
        .arg("--out")
        .arg(&a.out)
        .arg("--lang")
        .arg(&a.lang);
    if let Some(model) = &a.model {
        cmd.arg("--model").arg(model);
    }
    append_extra(&mut cmd, &a.sidecar.extra);
    run_sidecar(cmd, "speak").await?;
    tracing::info!("wrote {}", a.out.display());
    Ok(())
}

async fn finetune(a: TtsFinetuneArgs) -> Result<()> {
    let mut cmd = sidecar_cmd(&a.sidecar, "finetune")?;
    cmd.arg("--out").arg(&a.out);
    for path in &a.data {
        cmd.arg(path);
    }
    append_extra(&mut cmd, &a.sidecar.extra);
    run_sidecar(cmd, "finetune").await?;
    tracing::info!("fine-tuned model written to {}", a.out.display());
    Ok(())
}
