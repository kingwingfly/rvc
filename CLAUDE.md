# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`rvc` is a pure-Rust RVC v2 voice-conversion toolkit: it retimbres a source voice
into a trained target voice while preserving content + F0 pitch (so breathy/expressive
vocalizations survive by construction). Inference runs on **two interchangeable
generator backends** — ONNX Runtime (`ort`) and native Burn (GPU/wgpu) — and training
is native Rust/Burn. The only Python is a standalone `.safetensors → ONNX` exporter
under `export/`.

## Build / run / verify

```sh
# ONNX Runtime is never bundled — point at your own build (GPU-enabled for CUDA):
export ORT_DYLIB_PATH=/usr/lib/libonnxruntime.so

cargo build --release           # default: ort backend only
cargo clippy --workspace        # workspace is kept clippy-clean
cargo fmt

# The native Burn generator backend is behind rvc-core's `burn` feature; rvc-cli
# enables it, so building/running the CLI includes it automatically.
cargo run -p rvc-cli -- convert -m models/voice.safetensors --model-sr 48000 -o out/ in.mp3

# Correctness of the Burn port is checked by loading REAL pretrained weights
# (there are no unit tests for the network) — reports applied/missing/unused:
cargo run -p burn-rvc --example load -- <path/to/f0G48k.pth>
```

Requires **ffmpeg 8.1** dev libraries (and the `ffmpeg` binary for the realtime
`serve` example). ContentVec + RMVPE ONNX assets auto-download from Hugging Face
(`rvc models download` to prefetch). Only 48 kHz is supported today.

## Architecture

Data crosses every crate boundary as **mono `f32`**, and the whole conversion path
is `futures::Stream`-in → `futures::Stream`-out, which is why `rvc serve` is a plain
Unix filter (raw f32le PCM stdin→stdout) and batch `convert` is a thin wrapper.

| crate | role |
|-------|------|
| `rvc-audio` | ffmpeg decode/resample + WAV/raw-PCM I/O, all as `futures::Stream<f32>` |
| `rvc-core` | the voice-conversion pipeline: `FeatureExtractor` (ContentVec + RMVPE), coarse-pitch/upsample/pitch-shift DSP, streaming `Converter`, and **both** generator backends behind one `Generator` trait |
| `burn-rvc` | the RVC v2 network itself (standalone Burn port of `SynthesizerTrnMs768NSFsid` + `MultiPeriodDiscriminator`); no app deps |
| `rvc-train` | native Rust/Burn adversarial training loop (see `crates/rvc-train/ARCHITECTURE.md`) |
| `rvc-hub` | auto-download ContentVec/RMVPE ONNX from Hugging Face |
| `rvc-cli` | the `rvc` binary (clap): `convert`, `serve`, `models`, `train` |

### Two runtimes, one path (the key abstraction)
Everything downstream of the generator is shared: the same `FeatureExtractor`, the
same DSP, and the same streaming `Converter` drive **either** backend through the
`Generator` trait (`crates/rvc-core/src/backend.rs`). `convert` and `serve` both take
`--backend auto|burn|onnx`; `auto` picks by the `-m` extension (`.onnx` → ONNX
Runtime, else Burn). Use ONNX Runtime for realtime `serve` — the Burn (wgpu)
generator is currently slower than realtime.

### The `burn` feature
`rvc-core`'s Burn backend (`burn_backend.rs`, `pub BurnGenerator`) is behind the
optional `burn` feature (off by default) so consumers that only need the shared
`FeatureExtractor` (e.g. `rvc-train`) stay a lean `ort` crate. `rvc-cli` enables it.

### Naming: `burn-rvc` vs `rvc-core`
Deliberately different. `burn-rvc` is named after the **model** (RVC v2) and is a
self-contained network crate (reads like `burn_dinov3`). `rvc-core` is the app's
**voice-conversion pipeline** (feature extraction + backends + streaming), not tied
to one model.

### Weight-compatibility constraint (important when editing `burn-rvc`)
The Burn modules are kept **weight-compatible with RVC's PyTorch `state_dict`** so
they can warm-start from the public pretrained bases (`f0G48k.pth`/`f0D48k.pth`, HF
`lj1995/VoiceConversionWebUI`) — essential on a small (~1 h) corpus. The loader
(`crates/burn-rvc/src/store.rs`) remaps RVC's flat `attn_layers`/`norm_layers_*` lists
and the flow's even coupling indices onto the module tree and upcasts fp16→fp32.
Changing the module layout breaks warm-start and the ONNX exporter — keep names/
structure in step with the reference (RVC-Project tag `2.2.231006`,
`infer/lib/infer_pack/{models,attentions,modules}.py`).

### The Python boundary (`export/`)
The only Python: a standalone `uv` project that converts a Burn `.safetensors` to
ONNX. `rvc_infer.py` is a **clean-room** torch reimplementation mirroring the
`burn-rvc` layout (it does NOT depend on the RVC-Project repo). Native Burn inference
needs no export — this is only for ONNX Runtime / cross-framework deploy.

```sh
uv run --project export python export/export_onnx.py models/voice.safetensors models/voice.onnx
```

Exported graph contract (matches `rvc-core`):
`phone[1,T,768] f32, phone_lengths[1] i64, pitch[1,T] i64, pitchf[1,T] f32, ds[1] i64, rnd[1,192,T] f32 → audio[1,1,L] f32`.

## Training notes

Native Rust/Burn on wgpu/Vulkan; no `Learner` (the GAN loop doesn't fit it) — the
trainer drives Burn's `TuiMetricsRendererWrapper` directly (`crates/rvc-train/src/dashboard.rs`).
On a TTY a live dashboard shows `g`/`d`/`mel` losses; `q` stops early and saves. Off-TTY
(or `--no-tui`), it logs to stderr and Ctrl-C stops and saves. Losses match RVC exactly
(mel-L1 ×45, KL ×1, feature-matching ×2, LSGAN); the STFT front-end is n_fft=2048,
hop=480, 128 Slaney mels, center=False (`crates/rvc-train/src/spectral.rs`). Target GPU
is 6 GB (RTX 2060) → small batch. Warm-start from `--pretrained-g/-d` is strongly
recommended on a small corpus.
