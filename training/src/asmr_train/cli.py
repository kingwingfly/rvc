"""Command-line entrypoint for the training pipeline.

Invoked by the Rust CLI as `uv run python -m asmr_train --out <onnx> <data...>`.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .config import TrainConfig
from .pipeline import run_pipeline


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="asmr_train", description="Train an RVC voice model and export ONNX"
    )
    parser.add_argument("data", type=Path, nargs="+", help="corpus audio of the target voice")
    parser.add_argument("--out", type=Path, required=True, help="output path for generator ONNX")
    parser.add_argument(
        "--work-dir", type=Path, default=Path("training/.work"), help="scratch directory"
    )
    parser.add_argument("--sample-rate", type=int, default=48_000, choices=(40_000, 48_000))
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--speaker-id", type=int, default=0)
    parser.add_argument("--segment-seconds", type=float, default=3.0)
    parser.add_argument(
        "--rvc-repo", type=Path, default=None, help="path to a RVC-Project checkout"
    )
    parser.add_argument("--verbose", action="store_true")
    # Anything after `--` is passed through to the RVC training entrypoint.
    parser.add_argument("extra", nargs=argparse.REMAINDER, help="passthrough args after --")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    ns = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if ns.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Strip a leading "--" left by argparse.REMAINDER.
    extra: list[str] = list(ns.extra)
    if extra and extra[0] == "--":
        extra = extra[1:]

    cfg = TrainConfig(
        data=tuple(ns.data),
        out=ns.out,
        work_dir=ns.work_dir,
        sample_rate=ns.sample_rate,
        epochs=ns.epochs,
        batch_size=ns.batch_size,
        speaker_id=ns.speaker_id,
        segment_seconds=ns.segment_seconds,
        rvc_repo=ns.rvc_repo,
        extra=tuple(extra),
    )

    run_pipeline(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
