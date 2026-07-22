"""Corpus preprocessing: decode arbitrary inputs to 16 kHz mono WAV segments.

This stage is pure and fully typed; it shells out to ffmpeg for robust decoding
of mp3/aac/etc. and slices the result into fixed-length segments suitable for
RVC training.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from .config import ANALYSIS_SR, TrainConfig


def _require_ffmpeg() -> str:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg not found on PATH")
    return ffmpeg


def _decode_to_wav(ffmpeg: str, src: Path, dst: Path, sample_rate: int) -> None:
    """Decode `src` to mono `sample_rate` PCM WAV at `dst`."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(src),
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-f",
        "wav",
        str(dst),
    ]
    subprocess.run(cmd, check=True)


def _segment(ffmpeg: str, src: Path, out_dir: Path, seconds: float, prefix: str) -> list[Path]:
    """Split `src` WAV into `seconds`-long segments under `out_dir`."""
    out_dir.mkdir(parents=True, exist_ok=True)
    pattern = str(out_dir / f"{prefix}_%05d.wav")
    cmd = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(src),
        "-f",
        "segment",
        "-segment_time",
        str(seconds),
        "-c",
        "copy",
        pattern,
    ]
    subprocess.run(cmd, check=True)
    return sorted(out_dir.glob(f"{prefix}_*.wav"))


def prepare_dataset(cfg: TrainConfig) -> list[Path]:
    """Decode and segment every input file into `cfg.dataset_dir`.

    Returns the list of produced segment paths.
    """
    ffmpeg = _require_ffmpeg()
    dataset_dir = cfg.dataset_dir
    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    tmp = cfg.work_dir / "tmp"
    tmp.mkdir(parents=True, exist_ok=True)

    segments: list[Path] = []
    for i, src in enumerate(cfg.data):
        if not src.exists():
            raise FileNotFoundError(src)
        decoded = tmp / f"src_{i:03d}.wav"
        _decode_to_wav(ffmpeg, src, decoded, ANALYSIS_SR)
        segments.extend(_segment(ffmpeg, decoded, dataset_dir, cfg.segment_seconds, f"seg{i:03d}"))

    if not segments:
        raise RuntimeError("preprocessing produced no segments")
    return segments
