# asmr-train

RVC training + ONNX export for the ASMR voice toolkit. Runs **once, offline**;
nothing here ships to end users. Managed by **uv**; fully type-annotated
(`mypy --strict` / pyright strict, ruff-linted).

## Setup

```sh
uv sync            # create the venv and install deps (torch, onnx, ...)
```

The reference RVC training code is **not** a pip dependency (it is
version-sensitive). Provide a checkout and point `$RVC_REPO` (or `--rvc-repo`) at
it:

```sh
git clone https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI
export RVC_REPO="$PWD/Retrieval-based-Voice-Conversion-WebUI"
```

## Run

Usually invoked through the Rust CLI (`asmr train ...`), which calls:

```sh
uv run python -m asmr_train --out ../models/voice.onnx  clip1.mp3 clip2.mp3
```

Options: `--sample-rate {40000,48000}`, `--epochs`, `--batch-size`,
`--segment-seconds`, `--rvc-repo`, and passthrough args after `--`.

## Pipeline stages (`pipeline.py`)

1. `preprocess.py` — ffmpeg-decode each input to 16 kHz mono and slice into
   segments (`config.TrainConfig.dataset_dir`).
2. `rvc_backend.py` — drive the RVC stages (preprocess → F0 → ContentVec
   features → train). **This is the integration seam**: the script paths/flags
   there target a standard RVC checkout and fail loudly with the exact path they
   expected, so adapt them to your RVC version.
3. `export_onnx.py` — load the trained generator checkpoint and export ONNX with
   the input/output contract the Rust `asmr-vc` crate consumes
   (`phone, phone_lengths, pitch, pitchf, ds, rnd → audio`). Kept in sync with
   `crates/asmr-vc/src/config.rs`.

## Type-check / lint

```sh
uv run mypy
uvx ruff check src
```
