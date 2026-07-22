"""Typed configuration for the GPT-SoVITS sidecar."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def resolve_repo(explicit: Path | None) -> Path:
    """Locate the GPT-SoVITS checkout from an explicit path or `$GPT_SOVITS_REPO`."""
    repo = explicit
    if repo is None:
        env = os.environ.get("GPT_SOVITS_REPO")
        if env:
            repo = Path(env)
    if repo is None:
        raise RuntimeError(
            "GPT-SoVITS repo not configured. Pass --repo or set $GPT_SOVITS_REPO to a checkout of "
            "https://github.com/RVC-Boss/GPT-SoVITS"
        )
    if not repo.exists():
        raise FileNotFoundError(f"GPT-SoVITS repo does not exist: {repo}")
    return repo


@dataclass(frozen=True, slots=True)
class SpeakConfig:
    """Inputs for a single synthesis request."""

    text: str
    ref_audio: Path
    out: Path
    lang: str = "auto"
    model: Path | None = None
    repo: Path | None = None
    extra: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class FinetuneConfig:
    """Inputs for a few-shot fine-tune run."""

    data: tuple[Path, ...]
    out: Path
    repo: Path | None = None
    epochs: int = 8
    extra: tuple[str, ...] = field(default_factory=tuple)
