"""Typed configuration for the training pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# The generator I/O contract shared with the Rust `asmr-vc` crate. Keep these in
# sync with `crates/asmr-vc/src/config.rs::GeneratorIo`.
GENERATOR_INPUT_NAMES: tuple[str, ...] = ("phone", "phone_lengths", "pitch", "pitchf", "ds", "rnd")
GENERATOR_OUTPUT_NAME: str = "audio"

# ContentVec v2 feature width and analysis rate (fixed by the pretrained models).
CONTENT_DIM: int = 768
ANALYSIS_SR: int = 16_000


@dataclass(frozen=True, slots=True)
class TrainConfig:
    """Everything needed to run a training job end to end."""

    data: tuple[Path, ...]
    """Input corpus: one or more audio files of the target voice."""

    out: Path
    """Destination path for the exported generator ONNX."""

    work_dir: Path
    """Scratch directory for the dataset, features and checkpoints."""

    sample_rate: int = 48_000
    """Generator output sample rate (40000 or 48000)."""

    epochs: int = 200
    batch_size: int = 8
    speaker_id: int = 0
    f0_method: str = "rmvpe"
    seed: int = 1234

    segment_seconds: float = 3.0
    """Length of each preprocessed training segment."""

    rvc_repo: Path | None = None
    """Checkout of the RVC-Project repo; falls back to $RVC_REPO."""

    extra: tuple[str, ...] = field(default_factory=tuple)
    """Passthrough arguments for the RVC training entrypoint."""

    @property
    def dataset_dir(self) -> Path:
        return self.work_dir / "dataset"

    @property
    def checkpoint_path(self) -> Path:
        return self.work_dir / "checkpoints" / "generator.pth"
