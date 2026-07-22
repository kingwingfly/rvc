"""Command-line entrypoint for the GPT-SoVITS sidecar.

Invoked by the Rust CLI as
`uv run python -m asmr_tts {speak,finetune} ...`.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from . import backend
from .config import FinetuneConfig, SpeakConfig
from .stage import stage_dataset

logger = logging.getLogger("asmr_tts")


def _strip_dashes(extra: list[str]) -> tuple[str, ...]:
    if extra and extra[0] == "--":
        extra = extra[1:]
    return tuple(extra)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="asmr_tts", description="GPT-SoVITS TTS sidecar")
    parser.add_argument("--repo", type=Path, default=None, help="GPT-SoVITS checkout path")
    sub = parser.add_subparsers(dest="command", required=True)

    speak = sub.add_parser("speak", help="synthesize speech from text")
    speak.add_argument("--text", required=True)
    speak.add_argument("--ref", dest="ref_audio", type=Path, required=True)
    speak.add_argument("--out", type=Path, required=True)
    speak.add_argument("--lang", default="auto")
    speak.add_argument("--model", type=Path, default=None)
    speak.add_argument("extra", nargs=argparse.REMAINDER)

    ft = sub.add_parser("finetune", help="few-shot fine-tune on a target voice")
    ft.add_argument("data", type=Path, nargs="+")
    ft.add_argument("--out", type=Path, required=True)
    ft.add_argument("--epochs", type=int, default=8)
    ft.add_argument("--work-dir", type=Path, default=Path("training/.work_tts"))
    ft.add_argument("extra", nargs=argparse.REMAINDER)
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    ns = _build_parser().parse_args(argv)

    if ns.command == "speak":
        cfg = SpeakConfig(
            text=ns.text,
            ref_audio=ns.ref_audio,
            out=ns.out,
            lang=ns.lang,
            model=ns.model,
            repo=ns.repo,
            extra=_strip_dashes(list(ns.extra)),
        )
        out = backend.synthesize(cfg)
        logger.info("synthesized %s", out)
        return 0

    if ns.command == "finetune":
        dataset = stage_dataset(tuple(ns.data), ns.work_dir)
        cfg = FinetuneConfig(
            data=tuple(ns.data),
            out=ns.out,
            repo=ns.repo,
            epochs=ns.epochs,
            extra=_strip_dashes(list(ns.extra)),
        )
        out = backend.finetune(cfg, dataset)
        logger.info("fine-tuned model at %s", out)
        return 0

    raise SystemExit(2)


if __name__ == "__main__":
    raise SystemExit(main())
