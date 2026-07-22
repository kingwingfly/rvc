"""Stage a corpus (vector of files) into a directory of mono WAVs for fine-tuning.

GPT-SoVITS does its own slicing/ASR, so we only need to decode inputs to a common
mono WAV format. ffmpeg handles arbitrary input containers.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

_DEFAULT_SR = 32_000


def _require_ffmpeg() -> str:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg not found on PATH")
    return ffmpeg


def stage_dataset(data: tuple[Path, ...], work_dir: Path, sample_rate: int = _DEFAULT_SR) -> Path:
    """Decode each input to a mono WAV under `work_dir/dataset`, returning that dir."""
    ffmpeg = _require_ffmpeg()
    dataset = work_dir / "dataset"
    if dataset.exists():
        shutil.rmtree(dataset)
    dataset.mkdir(parents=True, exist_ok=True)

    for i, src in enumerate(data):
        if not src.exists():
            raise FileNotFoundError(src)
        dst = dataset / f"{i:03d}_{src.stem}.wav"
        subprocess.run(
            [
                ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
                "-i", str(src), "-ac", "1", "-ar", str(sample_rate),
                "-f", "wav", str(dst),
            ],
            check=True,
        )
    return dataset
