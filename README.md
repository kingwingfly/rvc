# asmr — ASMR voice toolkit

Retimbre your Chinese/English ASMR library into a voice you like. **Voice
conversion (VC)** runs as pure-Rust ONNX inference (RVC v2) with a streaming
Unix-filter CLI, and training is moving to a native Rust (burn) pipeline — no
Python in the toolkit except the single weight → ONNX conversion step.

Why the expressive vocalizations survive: RVC swaps only **timbre**, driving the
generator from the source's *content features* + *F0 pitch*. Breathy cries,
moans and gasps live in those streams, so they are preserved by construction.

## Architecture

| crate | role |
|-------|------|
| `asmr-audio` | ffmpeg (8.1) mp3/wav decode, resample, WAV/raw-PCM I/O — all as `futures::Stream` of mono `f32` |
| `asmr-vc` | RVC pipeline over `ort`: ContentVec + RMVPE + trained generator; `Stream`-in → `Stream`-out `Converter`; reusable `FeatureExtractor` |
| `asmr-hub` | auto-download ContentVec/RMVPE ONNX from Hugging Face |
| `asmr-train` | native (Rust/burn) RVC generator training — see `crates/asmr-train/ARCHITECTURE.md` |
| `asmr-cli` | the `asmr` binary (clap) |

## Prerequisites

- **ONNX Runtime**: this tool never bundles or downloads it. Point
  `ORT_DYLIB_PATH` at your own build (GPU-enabled for CUDA):
  ```sh
  export ORT_DYLIB_PATH=/path/to/libonnxruntime.so
  # e.g. a system install: export ORT_DYLIB_PATH=/usr/lib/libonnxruntime.so
  ```
  The CUDA execution provider is tried first, then CPU. A 6 GB GPU is enough for
  inference. (Built against the ORT 1.24 API; newer runtimes work too.)
- **ffmpeg 8.1** dev libraries (for `asmr-audio`) and the `ffmpeg` binary (for
  piping raw PCM in the realtime example).

## Build

```sh
cargo build --release
```

## Use

### Convert `a.mp3` into a trained voice's timbre

The target timbre lives in a **trained generator**, so this is two stages: train
a generator on the *target voice* once, then convert *source* clips through it.
The corpus is passed as a vector of paths — give it as much clean audio of the
target voice as you have.

```sh
export ORT_DYLIB_PATH=/usr/lib/libonnxruntime.so

# 1. Train a generator on the target voice (once, offline).
asmr train --out models/piner.onnx --model-sr 48000  /tmp/piner.mp3

# 2. Convert a.mp3 through it -> out/a.wav in that timbre.
asmr convert -m models/piner.onnx --model-sr 48000 -o out/  /tmp/a.mp3
```

If the source and target sit in different pitch ranges, add e.g. `-t 2` (up) or
`-t -3` (down) to step 2.

### 1. Fine-tune a generator (native Rust/Burn, GPU)

Training runs natively in Rust (Burn on wgpu/Vulkan) — no Python. Warm-start
from the public pretrained bases (`f0G48k.pth`/`f0D48k.pth`) for good results on
a small corpus:

```sh
asmr train --out models/voice --model-sr 48000 \
  --pretrained-g f0G48k.pth --pretrained-d f0D48k.pth \
  -b 4 -e 200  clip1.mp3 clip2.mp3 clip3.mp3
```

This writes `models/voice.safetensors`. The shared ContentVec + RMVPE ONNX assets
(training features + inference) download automatically, or prefetch them:

```sh
asmr models download
```

Notes: only 48 kHz is supported today; a 6 GB GPU handles batch ≈4. Losses
(`g`/`d`/`mel`) are logged as it trains.

### 2. Batch-convert files

```sh
# Burn generator (native) — pass the trained .safetensors
asmr convert -m models/voice.safetensors --model-sr 48000 -o out/  input1.mp3 input2.mp3

# ONNX Runtime generator — pass a .onnx (see the ONNX export step)
asmr convert -m models/voice.onnx --model-sr 48000 -o out/  input1.mp3 input2.mp3
```

Writes `out/input1.wav`, `out/input2.wav` in the target timbre. `--backend`
(`auto`/`burn`/`onnx`) picks the generator; `auto` chooses by file extension
(`.onnx` → ONNX Runtime, else Burn). The Burn generator runs on the GPU (wgpu).

### 3. Realtime — a Unix filter (raw f32le PCM stdin → stdout)

Input is mono **f32le @ 16 kHz** (the RVC analysis rate); output is mono f32le at
the model's sample rate. ffmpeg handles capture/playback on either end:

```sh
# play a file through the converter
ffmpeg -i in.mp3 -f f32le -ar 16000 -ac 1 - \
  | asmr serve -m models/voice.onnx --model-sr 48000 \
  | ffplay -f f32le -ar 48000 -ac 1 -

# live mic → converted → speakers
ffmpeg -f alsa -i default -f f32le -ar 16000 -ac 1 - \
  | asmr serve -m models/voice.onnx --model-sr 48000 \
  | ffplay -f f32le -ar 48000 -ac 1 -
```

Logs go to stderr, so stdout carries only PCM.

## Status

- **Native training works** (Rust/Burn, GPU): the full RVC v2 generator +
  MultiPeriod discriminator are ported to Burn (`crates/burn-rvc`), warm-start
  from the public pretrained bases, and fine-tune adversarially (mel-L1 + KL +
  feature-matching + LSGAN) on wgpu. Verified end-to-end on a real clip —
  losses decrease and the saved `.safetensors` round-trips through
  `asmr convert`.
- **Inference works on both backends**: the native Burn generator (GPU) and
  ONNX Runtime. Workspace is clippy-clean.

- **ONNX export works**: `export/` is a small standalone uv/python script that converts
  a Burn `.safetensors` to ONNX (clean-room torch, no RVC repo); verified by
  running the result through `asmr convert --backend onnx`. See
  [`export/README.md`](export/README.md).

## Roadmap

- 40 kHz training; streaming (`serve`) on the Burn backend.
- Index/retrieval blend + `protect` for even tighter timbre match.
- TTS (text → voice) — deferred.
