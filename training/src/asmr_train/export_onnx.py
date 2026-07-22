"""Export a trained RVC generator checkpoint to ONNX.

The exported graph's input/output names and dynamic axes match the contract the
Rust `asmr-vc` crate consumes (`crates/asmr-vc/src/config.rs`):

    inputs : phone [1,T,768] f32, phone_lengths [1] i64,
             pitch [1,T] i64, pitchf [1,T] f32, ds [1] i64, rnd [1,192,T] f32
    output : audio [1,1,L] f32

The generator architecture (`SynthesizerTrnMs768NSFsid`) lives in the RVC-Project
repo; this module locates it, loads the checkpoint, and drives torch.onnx.export.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch

from .config import (
    CONTENT_DIM,
    GENERATOR_INPUT_NAMES,
    GENERATOR_OUTPUT_NAME,
    TrainConfig,
)


def _load_rvc_synthesizer(cfg: TrainConfig, checkpoint: dict[str, Any]) -> torch.nn.Module:
    """Import the RVC synthesizer class and build it from the checkpoint config."""
    repo = cfg.rvc_repo
    if repo is None:
        raise RuntimeError(
            "RVC repo path is required for ONNX export (set --rvc-repo or $RVC_REPO)"
        )
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))

    try:
        # Path within the RVC-Project repository.
        from infer.lib.infer_pack.models import (  # type: ignore[import-not-found]
            SynthesizerTrnMs768NSFsid,
        )
    except ImportError as exc:  # pragma: no cover - integration seam
        raise RuntimeError(
            f"could not import RVC synthesizer from {repo}; check the RVC checkout"
        ) from exc

    model_config = checkpoint["config"]
    model = SynthesizerTrnMs768NSFsid(*model_config, is_half=False)
    model.load_state_dict(checkpoint["weight"], strict=False)
    model.eval()
    return model


def _dummy_inputs(sample_frames: int, rnd_dim: int) -> tuple[torch.Tensor, ...]:
    """Build correctly-shaped dummy tensors to trace the export."""
    phone = torch.rand(1, sample_frames, CONTENT_DIM, dtype=torch.float32)
    phone_lengths = torch.tensor([sample_frames], dtype=torch.int64)
    pitch = torch.randint(1, 255, (1, sample_frames), dtype=torch.int64)
    pitchf = torch.rand(1, sample_frames, dtype=torch.float32) * 200.0
    ds = torch.tensor([0], dtype=torch.int64)
    rnd = torch.randn(1, rnd_dim, sample_frames, dtype=torch.float32)
    return phone, phone_lengths, pitch, pitchf, ds, rnd


def export_generator(cfg: TrainConfig, checkpoint_path: Path, rnd_dim: int = 192) -> Path:
    """Export the trained generator at `checkpoint_path` to `cfg.out` (ONNX)."""
    checkpoint: dict[str, Any] = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = _load_rvc_synthesizer(cfg, checkpoint)

    inputs = _dummy_inputs(sample_frames=100, rnd_dim=rnd_dim)
    dynamic_axes: dict[str, dict[int, str]] = {
        "phone": {1: "t"},
        "pitch": {1: "t"},
        "pitchf": {1: "t"},
        "rnd": {2: "t"},
        GENERATOR_OUTPUT_NAME: {2: "l"},
    }

    cfg.out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        inputs,
        str(cfg.out),
        input_names=list(GENERATOR_INPUT_NAMES),
        output_names=[GENERATOR_OUTPUT_NAME],
        dynamic_axes=dynamic_axes,
        opset_version=17,
        do_constant_folding=True,
    )
    return cfg.out
