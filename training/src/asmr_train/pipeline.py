"""End-to-end training orchestration: preprocess -> train -> export ONNX."""

from __future__ import annotations

import logging
from pathlib import Path

from .config import TrainConfig
from .export_onnx import export_generator
from .preprocess import prepare_dataset
from .rvc_backend import train_generator

logger = logging.getLogger("asmr_train")


def run_pipeline(cfg: TrainConfig) -> Path:
    """Run the full pipeline and return the path to the exported ONNX generator."""
    cfg.work_dir.mkdir(parents=True, exist_ok=True)

    logger.info("preprocessing %d input file(s) -> %s", len(cfg.data), cfg.dataset_dir)
    segments = prepare_dataset(cfg)
    logger.info("prepared %d training segments", len(segments))

    logger.info("training RVC generator (%d epochs, sr=%d)", cfg.epochs, cfg.sample_rate)
    checkpoint = train_generator(cfg, cfg.dataset_dir)
    logger.info("trained checkpoint: %s", checkpoint)

    logger.info("exporting generator -> %s", cfg.out)
    onnx_path = export_generator(cfg, checkpoint)
    logger.info("done: %s", onnx_path)
    return onnx_path
