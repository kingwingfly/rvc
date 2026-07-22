"""Integration seam with the reference RVC-Project training code.

RVC training is a multi-stage pipeline (preprocess -> F0 extraction -> content
feature extraction -> generator training) implemented by scripts in the
RVC-Project repository. Those scripts are version-sensitive, so this module keeps
the invocation in one typed place: adapt the paths/flags here to match your RVC
checkout rather than scattering subprocess calls across the pipeline.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from .config import TrainConfig


def resolve_repo(cfg: TrainConfig) -> Path:
    """Locate the RVC-Project checkout from config or the `$RVC_REPO` env var."""
    repo = cfg.rvc_repo
    if repo is None:
        env = os.environ.get("RVC_REPO")
        if env:
            repo = Path(env)
    if repo is None:
        raise RuntimeError(
            "RVC repo not configured. Pass --rvc-repo or set $RVC_REPO to a checkout of "
            "https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI"
        )
    if not repo.exists():
        raise FileNotFoundError(f"RVC repo does not exist: {repo}")
    return repo


def _run(repo: Path, script: str, args: list[str]) -> None:
    """Run a RVC script with the repo as CWD, raising if it is missing."""
    script_path = repo / script
    if not script_path.exists():
        raise FileNotFoundError(
            f"expected RVC script not found: {script_path}. "
            "Adjust rvc_backend.py to match your RVC version."
        )
    subprocess.run(["python", str(script_path), *args], cwd=repo, check=True)


def train_generator(cfg: TrainConfig, dataset_dir: Path) -> Path:
    """Drive the RVC training pipeline and return the generator checkpoint path.

    This wires the standard RVC stages. The exact script names/flags below are the
    integration point to adapt to your RVC checkout; each `_run` call fails loudly
    with the path it expected so mismatches are obvious.
    """
    repo = resolve_repo(cfg)
    exp = cfg.work_dir / "rvc_exp"
    exp.mkdir(parents=True, exist_ok=True)

    # 1. Preprocess dataset (resample/normalize into the experiment dir).
    _run(repo, "infer/modules/train/preprocess.py",
         [str(dataset_dir), str(cfg.sample_rate), "4", str(exp), "False", "3.7"])

    # 2. Extract F0 with the configured method.
    _run(repo, "infer/modules/train/extract/extract_f0_rmvpe.py",
         ["1", "0", "0", str(exp), "True"])

    # 3. Extract ContentVec features.
    _run(repo, "infer/modules/train/extract_feature_print.py",
         ["cuda:0", "1", "0", str(exp), "v2", "True"])

    # 4. Train the generator. Adapt flags (epochs, batch, pretrained G/D) to taste.
    _run(repo, "infer/modules/train/train.py",
         ["-e", exp.name, "-sr", str(cfg.sample_rate), "-f0", "1",
          "-bs", str(cfg.batch_size), "-te", str(cfg.epochs),
          "-se", "5", "-l", "0", "-c", "0", "-sw", "0", "-v", "v2", *cfg.extra])

    # RVC writes generator checkpoints as G_<step>.pth in the experiment logs.
    candidates = sorted(exp.glob("G_*.pth")) or sorted((repo / "logs" / exp.name).glob("G_*.pth"))
    if not candidates:
        raise RuntimeError(
            f"no generator checkpoint (G_*.pth) found under {exp}; check training output"
        )
    checkpoint = candidates[-1]
    cfg.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    return checkpoint
