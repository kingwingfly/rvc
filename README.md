# asmr — ASMR voice toolkit

Retimbre your Chinese/English ASMR library into a voice you like, and generate
speech from text in that voice. **Voice conversion (VC)** runs as pure-Rust ONNX
inference (RVC v2) with a streaming Unix-filter CLI; **TTS** (GPT-SoVITS) runs
through a typed Python/uv sidecar because its ONNX export is only partial.

Why the expressive vocalizations survive: RVC swaps only **timbre**, driving the
generator from the source's *content features* + *F0 pitch*. Breathy cries,
moans and gasps live in those streams, so they are preserved by construction.

## Architecture

| crate | role |
|-------|------|
| `asmr-audio` | ffmpeg (8.1) mp3/wav decode, resample, WAV/raw-PCM I/O — all as `futures::Stream` of mono `f32` |
| `asmr-vc` | RVC pipeline over `ort`: ContentVec + RMVPE + trained generator; `Stream`-in → `Stream`-out `Converter` |
| `asmr-hub` | auto-download ContentVec/RMVPE ONNX from Hugging Face |
| `asmr-cli` | the `asmr` binary (clap) |
| `training/` | Python + uv: RVC training (`asmr_train`) and GPT-SoVITS TTS sidecar (`asmr_tts`) |

## Prerequisites

- **ONNX Runtime**: this tool never bundles or downloads it. Point
  `ORT_DYLIB_PATH` at your own build (GPU-enabled for CUDA):
  ```sh
  export ORT_DYLIB_PATH=/path/to/libonnxruntime.so
  # e.g. a system install: export ORT_DYLIB_PATH=/usr/lib/libonnxruntime.so
  ```
  The CUDA execution provider is tried first, then CPU. A 6 GB GPU is enough for
  inference. (Built against the ORT 1.24 API; newer runtimes like 1.27 work too.)
- **ffmpeg 8.1** dev libraries (for `asmr-audio`) and the `ffmpeg` binary (for
  piping and for training preprocessing).

## Build

```sh
cargo build --release
```

## Use

### Worked example: convert `a.mp3` into `piner.mp3`'s timbre

The target timbre lives in a **trained model**, so this is two stages — train a
model on the *target voice* (`piner.mp3`) once, then convert the *source*
(`a.mp3`) through it. `piner.mp3` is the training corpus, not a runtime reference;
give it as much clean audio of that voice as you have.

```sh
export ORT_DYLIB_PATH=/usr/lib/libonnxruntime.so

# 1. Train a generator on piner's voice (once, offline; needs $RVC_REPO + uv).
asmr train --out models/piner.onnx --sample-rate 48000  /tmp/piner.mp3

# 2. Convert a.mp3 through it -> out/a.wav in piner's timbre.
asmr convert -m models/piner.onnx --model-sr 48000 -o out/  /tmp/a.mp3
```

If `a` and `piner` sit in different pitch ranges, add e.g. `-t 2` (up) or
`-t -3` (down) to step 2. For live playback instead of a file, pipe through
`asmr serve` (see step 3 below).

The rest of this section documents each command in detail.

### 1. Train a generator on your liked-voice corpus (once, offline)

Training runs in Python via uv (see [`training/README.md`](training/README.md)):

```sh
asmr train --out models/voice.onnx  clip1.mp3 clip2.mp3 clip3.mp3
# (accepts a whole vector of corpus files)
```

This produces `models/voice.onnx`. The shared ContentVec + RMVPE ONNX assets are
downloaded automatically on first conversion (or prefetch them):

```sh
asmr models download
```

### 2. Batch-convert files

```sh
asmr convert -m models/voice.onnx --model-sr 48000 -o out/  input1.mp3 input2.mp3
```

Writes `out/input1.wav`, `out/input2.wav` in the target timbre.

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

### 4. Text-to-speech (GPT-SoVITS)

TTS runs through the `asmr_tts` uv sidecar (point `$GPT_SOVITS_REPO` at a
[GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) checkout):

```sh
asmr tts finetune -o models/tts  clip1.mp3 clip2.mp3     # few-shot adapt the voice
asmr tts speak --text "你好，欢迎回来" --ref ref.mp3 -o out.wav --lang zh
```

## Status

- **Validated on real audio**: `asmr models download` fetches the shared assets,
  and the analysis frontend runs correctly end to end on real files — ContentVec
  (749 frames × 768) and RMVPE F0 (with the 128-mel input + UNet 32-alignment)
  were verified against the downloaded ONNX on a real clip. Full Rust workspace is
  clippy-clean with passing unit tests; Python packages lint clean.
- **Pending a trained model**: producing converted audio needs a trained
  `voice.onnx` (RVC training via `$RVC_REPO` + uv) — training is a heavy offline
  step. The generator's exact I/O (`rnd` input, opset) and the RVC/GPT-SoVITS
  training seams are configurable but only fully verified once a real generator
  exists. Tune `--transpose` (and later `protect`/index-retrieval) on real clips.

## Roadmap

- Index/retrieval blend + `protect` for even tighter timbre match.
