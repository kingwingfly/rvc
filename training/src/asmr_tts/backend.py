"""Integration seam with the GPT-SoVITS repository.

GPT-SoVITS exposes inference/training through repo scripts and an API server that
change between versions. This module keeps those invocations in one typed place;
adapt the script paths/flags here to match your GPT-SoVITS checkout. Each call
fails loudly with the exact path it expected, so a version mismatch is obvious.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from .config import FinetuneConfig, SpeakConfig, resolve_repo


def _run(repo: Path, script: str, args: list[str]) -> None:
    script_path = repo / script
    if not script_path.exists():
        raise FileNotFoundError(
            f"expected GPT-SoVITS script not found: {script_path}. "
            "Adjust asmr_tts/backend.py to match your GPT-SoVITS version."
        )
    subprocess.run(["python", str(script_path), *args], cwd=repo, check=True)


def synthesize(cfg: SpeakConfig) -> Path:
    """Run inference and return the path to the produced WAV.

    Integration point: wire this to your GPT-SoVITS inference entrypoint. The
    canonical CLI takes the target text, a reference clip (for timbre/prosody),
    the fine-tuned GPT + SoVITS weights, and a language hint.
    """
    repo = resolve_repo(cfg.repo)
    cfg.out.parent.mkdir(parents=True, exist_ok=True)

    args = [
        "--text", cfg.text,
        "--ref_audio", str(cfg.ref_audio),
        "--language", cfg.lang,
        "--output", str(cfg.out),
    ]
    if cfg.model is not None:
        args += ["--model_dir", str(cfg.model)]
    args += list(cfg.extra)

    # `inference_cli.py` is the conventional entrypoint; adapt as needed.
    _run(repo, "GPT_SoVITS/inference_cli.py", args)

    if not cfg.out.exists():
        raise RuntimeError(f"synthesis did not produce {cfg.out}")
    return cfg.out


def finetune(cfg: FinetuneConfig, dataset_dir: Path) -> Path:
    """Few-shot fine-tune and return the output model directory.

    Integration point: GPT-SoVITS fine-tuning is a short pipeline (ASR/labeling ->
    formatting -> GPT + SoVITS training). Wire the stages to your checkout.
    """
    repo = resolve_repo(cfg.repo)
    cfg.out.mkdir(parents=True, exist_ok=True)

    args = [
        "--data", str(dataset_dir),
        "--exp_name", cfg.out.name,
        "--epochs", str(cfg.epochs),
        *cfg.extra,
    ]
    _run(repo, "GPT_SoVITS/finetune_cli.py", args)
    return cfg.out
