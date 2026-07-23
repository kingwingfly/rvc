# rvc — RVC voice conversion toolkit

Retimbre any voice recording into a voice you like. **Voice conversion (VC)**
runs as pure-Rust inference (RVC v2) — native Burn (GPU) or ONNX Runtime — with a
streaming Unix-filter CLI, and training is a native Rust (burn) pipeline: no
Python in the toolkit except the single weight → ONNX conversion step.

Why the expressive vocalizations survive: RVC swaps only **timbre**, driving the
generator from the source's *content features* + *F0 pitch*. Breathy cries,
moans and gasps live in those streams, so they are preserved by construction.

## Architecture

| crate | role |
|-------|------|
| `rvc-audio` | ffmpeg (8.1) mp3/wav decode, resample, WAV/raw-PCM I/O — all as `futures::Stream` of mono `f32` |
| `rvc-core` | the voice-conversion pipeline: ContentVec + RMVPE feature extraction, and **both** generator backends (ONNX Runtime via `ort`, and native Burn behind the `burn` feature) behind one `Generator` trait; `Stream`-in → `Stream`-out `Converter`; reusable `FeatureExtractor` |
| `burn-rvc` | the RVC v2 network itself, a standalone Burn port of `SynthesizerTrnMs768NSFsid` (no app deps — like `burn_dinov3`) |
| `rvc-hub` | auto-download ContentVec/RMVPE ONNX from Hugging Face |
| `rvc-train` | native (Rust/burn) RVC generator training — see `crates/rvc-train/ARCHITECTURE.md` |
| `rvc-cli` | the `rvc` binary (clap) |

**On the names** (`burn-rvc` vs `rvc-core`): they're deliberately different.
`burn-rvc` is named after the *model* — **RVC** (Retrieval-based Voice Conversion)
v2 — and is a self-contained network crate, so it reads like other Burn model
crates (`burn_dinov3`). `rvc-core` is the app's **voice-conversion** pipeline (the
`vc` function: feature extraction + backends + streaming), not tied to one model.

**One conversion path, two runtimes.** Everything downstream of the generator is
shared: the same `FeatureExtractor` (ContentVec + RMVPE), the same coarse-pitch /
upsample / pitch-shift DSP, and the same streaming `Converter` (block / overlap /
crossfade) drive either backend through the `Generator` trait. `convert` and
`serve` both take `--backend auto|burn|onnx` (`auto` picks by the `-m`
extension: `.onnx` → ONNX Runtime, else Burn).

## Prerequisites

- **ONNX Runtime**: this tool never bundles or downloads it. Point
  `ORT_DYLIB_PATH` at your own build (GPU-enabled for CUDA):
  ```sh
  export ORT_DYLIB_PATH=/path/to/libonnxruntime.so
  # e.g. a system install: export ORT_DYLIB_PATH=/usr/lib/libonnxruntime.so
  ```
  The CUDA execution provider is tried first, then CPU. A 6 GB GPU is enough for
  inference. (Built against the ORT 1.24 API; newer runtimes work too.)
- **ffmpeg 8.1** dev libraries (for `rvc-audio`) and the `ffmpeg` binary (for
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

# 1. Train a generator on the target voice (once, offline) -> models/piner.safetensors.
rvc train --out models/piner --model-sr 48000  /tmp/piner.mp3

# 2. Convert a.mp3 through it -> out/a.wav in that timbre (native Burn backend).
rvc convert -m models/piner.safetensors --model-sr 48000 -o out/  /tmp/a.mp3
```

If the source and target sit in different pitch ranges, add e.g. `-t 2` (up) or
`-t -3` (down) to step 2.

### 1. Fine-tune a generator (native Rust/Burn, GPU)

Training runs natively in Rust (Burn on wgpu/Vulkan) — no Python. Warm-start
from the public pretrained bases (`f0G48k.pth`/`f0D48k.pth`) for good results on
a small corpus. Get the bases from Hugging Face `lj1995/VoiceConversionWebUI`
(`assets/pretrained_v2/`) and point `--pretrained-g/-d` at them:

```sh
rvc train --out models/voice --model-sr 48000 \
  --pretrained-g models/pretrained/f0G48k.pth --pretrained-d models/pretrained/f0D48k.pth \
  clip1.mp3 clip2.mp3 clip3.mp3
```

This writes the generator to `models/voice.safetensors` and, next to it, the
discriminator checkpoint `models/voice.disc.safetensors` (the sidecar that makes
`--continue` below resume the adversary too). The shared ContentVec + RMVPE ONNX
assets (training features + inference) download automatically, or prefetch them:

```sh
rvc models download
```

**Dashboard & early stop.** On a terminal, training shows Burn's live TUI
dashboard (loss plots + progress); logs go to `{work-dir}/train.log` so they
don't corrupt it. Press `q` to stop early — the model is saved. Without a TTY
(or with `--no-tui`), it logs `g`/`d`/`mel` to stderr and **Ctrl-C** stops and
saves. Watch **`mel`** for quality — it should fall and plateau; `g`/`d` are
adversarial and just stay balanced.

**Resume a run (`--continue`).** Stopped early and want to keep going? Point
`--continue` (alias `--resume`) at the generator `.safetensors` from the earlier
run instead of a pretrained base — it loads the generator, and picks up the
discriminator from the `.disc.safetensors` sidecar automatically:

```sh
rvc train --out models/voice --model-sr 48000 \
  --continue models/voice.safetensors \
  clip1.mp3 clip2.mp3 clip3.mp3
```

`--continue` conflicts with `--pretrained-g` (the checkpoint replaces the base).
If the discriminator sidecar is missing it falls back to `--pretrained-d` (pass
it), else the discriminator starts fresh. Note that optimizer (AdamW) momentum
is not persisted across runs — negligible for short fine-tunes.

Notes: only 48 kHz is supported today; defaults are `-e 20` epochs and `-b 2`
(safe on a 6 GB RTX 2060). Fine-tuning a warm-started base on ~30 min of audio
converges in a few dozen epochs — lean on early stop rather than a big `-e`.

### 2. Batch-convert files

```sh
# Burn generator (native) — pass the trained .safetensors
rvc convert -m models/voice.safetensors --model-sr 48000 -o out/  input1.mp3 input2.mp3

# ONNX Runtime generator — pass a .onnx (see the ONNX export step)
rvc convert -m models/voice.onnx --model-sr 48000 -o out/  input1.mp3 input2.mp3
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
  | rvc serve -m models/voice.onnx --model-sr 48000 \
  | ffplay -f f32le -ar 48000 -ac 1 -

# live mic → converted → speakers
ffmpeg -f alsa -i default -f f32le -ar 16000 -ac 1 - \
  | rvc serve -m models/voice.onnx --model-sr 48000 \
  | ffplay -f f32le -ar 48000 -ac 1 -
```

`serve` takes the same `--backend` flag as `convert`; both runtimes stream
through the same `Converter`. Use ONNX Runtime for realtime — the Burn (wgpu)
generator works but is currently slower than realtime. Logs go to stderr, so
stdout carries only PCM.

## Status

- **Native training works** (Rust/Burn, GPU): the full RVC v2 generator +
  MultiPeriod discriminator are ported to Burn (`crates/burn-rvc`), warm-start
  from the public pretrained bases, and fine-tune adversarially (mel-L1 + KL +
  feature-matching + LSGAN) on wgpu. Verified end-to-end on a real clip —
  losses decrease and the saved `.safetensors` round-trips through
  `rvc convert`.
- **Inference works on both backends**, and both `convert` and `serve` run
  either the native Burn generator (GPU) or ONNX Runtime through one shared
  `Converter` (`--backend`). Workspace is clippy-clean.
- **Live training dashboard**: Burn's TUI shows loss plots + progress; `q` (or
  Ctrl-C without the TUI) stops early and saves.

- **ONNX export works**: `export/` is a small standalone uv/python script that converts
  a Burn `.safetensors` to ONNX (clean-room torch, no RVC repo); verified by
  running the result through `rvc convert --backend onnx`. See
  [`export/README.md`](export/README.md).

## Roadmap

- 40 kHz training.
- Faster Burn (wgpu) inference so `serve` is realtime on the native backend.
- Index/retrieval blend + `protect` for even tighter timbre match.
- TTS (text → voice) — deferred.
