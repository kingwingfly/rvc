# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`voice` is a pure-Rust speech toolkit whose engines compose over Unix pipes:
`voice -(stt)-> text -(translate)-> text -(tts)-> voice -(rvc)-> voice`. Only the
`rvc` stage exists today.

`rvc` is RVC v2 voice conversion: it retimbres a source voice into a trained
target voice while preserving content + F0 pitch (so breathy/expressive
vocalizations survive by construction). Inference runs on **three interchangeable
generator backends** — ONNX Runtime (`ort`), and native Burn on either LibTorch
(`burn-tch`, ~9x faster), CubeCL/CUDA or WebGPU — and training is native Rust/Burn
on any of the three.

### Three rules that are easy to break silently

**No Python.** Not for users, not for developers, not for setup. New models are
**ported to Burn** and load their original Hugging Face weights directly, training
loops included — never wrapped in a Python process, a `uv` project, or a
preprocessing script. Reading a cloned upstream repo as a porting reference is
fine; running or shipping it is not. This is not a style preference: driving the
RVC Python repo directly was tried and abandoned because it was unmaintainable
for the authors and unusable for anyone else. The single `.safetensors → ONNX`
exporter under `export/` is the last exception, gains no new responsibilities,
and should eventually go.

**One binary per engine, plus `voice`.** `rvc` and `stt` each stand alone and
pull in only what they use — installing `stt` costs no ONNX Runtime and none of
the RVC stack. `voice` is the *integration*: it depends on `rvc-cli` and
`stt-cli` **as libraries**, so an argument is defined exactly once and never
copied between binaries. Every `*-cli` crate is therefore a lib **and** a bin.
`rvc-cli::args::RvcCommand` is `#[command(flatten)]`ed by `rvc` and nested by
`voice`; `stt-cli::SttArgs` is `#[command(flatten)]`ed by `stt` and nested by
`voice`. User-facing strings in shared code must not name a binary ("train one
with the `train` subcommand", not "`rvc train`").

**No engine depends on another engine.** Voice conversion, recognition and
synthesis are siblings. Anything two of them need moves to a neutral crate first.

### Naming conventions
Three tiers, and the name says which tier a crate is in:

- **`*-kit`** — shared plumbing with no model and no engine knowledge, safe for
  anything to depend on: `burn-kit` (devices, checkpoints), `audio-kit` (ffmpeg
  I/O, the slicer), `hub-kit` (downloads), `cli-kit` (logging, completions).
- **`burn-*`** — one network each, named after the **model** (`burn-rvc` reads
  like `burn_dinov3`), holding no app dependencies and naming no compute backend.
- **`<engine>-core` / `<engine>-cli`** — one engine each, all the same shape:
  `rvc-core`+`rvc-train`+`rvc-cli`, `stt-core`+`stt-cli`, later `tts-core`+`tts-cli`.

**`voice-` is reserved for the top.** It marks the integration, so a crate that
an engine depends on must never be named `voice-*` — that is why the shared
crates are `*-kit`. `voice-cli` is the only `voice-*` crate.

## Build / run / verify

```sh
# Neither ONNX Runtime nor LibTorch is bundled or downloaded — point at your own:
export ORT_DYLIB_PATH=/usr/lib/libonnxruntime.so
export LIBTORCH=$PWD/libtorch   # must be exactly 2.9.0 (what tch 0.22 targets)

cargo build                     # all three binaries: `rvc`, `stt`, `voice`
cargo build --release -p rvc-cli   # just `rvc`, optimised

# `cargo build` (dev) is the right default even for running models: the profile
# gives dependencies `opt-level = 3` — that is where every tensor op lives — and
# leaves workspace crates cheap to recompile. Transcribing 24 s of audio takes
# 19 s in debug against 18 s in release, for a fraction of the build. Reach for
# `--release` to benchmark, not to test.
cargo check -p rvc-core  --features cuda,tch
cargo check -p rvc-train --features cuda,tch    # where the trait bounds bite
cargo clippy --workspace        # workspace is kept clippy-clean
cargo fmt

# `--backend auto|onnx|cuda|tch` (aliases: burn/burn-cuda, libtorch/burn-tch);
# `--device auto|cpu|cuda|cuda:N|mps|vulkan`. Both default to auto.
cargo run -p rvc-cli   --      convert -m models/voice.safetensors --model-sr 48000 -o out/ in.mp3
cargo run -p voice-cli -- rvc  convert -m models/voice.safetensors --model-sr 48000 -o out/ in.mp3

# No LibTorch on the machine? Drop it (then `--backend tch` errors cleanly):
cargo build --release --no-default-features --features cuda

# Correctness of the Burn port is checked by loading REAL pretrained weights
# (there are no unit tests for the network) — reports applied/missing/unused:
cargo run -p burn-rvc --example load -- <path/to/f0G48k.pth>       # 560/0, 165/0
cargo run -p burn-whisper --example load -- <path/to/model.safetensors>  # 587/0
```

Porting references are cloned under `/.reference` (gitignored) and **read, never
run** — `openai/whisper`, and RVC-Project tag `2.2.231006` for `burn-rvc`.

### Two runtimes for one model (`stt-core`)
`Engine` (`engine.rs`) is the whole boundary between the decode loop and a
runtime: token ids in, `f32` logits out. Encoded audio and KV caches stay inside
the engine, because they are backend-specific tensors with no useful common type.
That keeps `Transcriber` non-generic — the backend is a constructor call, not a
type parameter — and it buys the best correctness check available: **Burn and
ONNX Runtime produce byte-identical transcripts from the same weights**, which is
how the Burn port is validated against an independent implementation.

### Porting traps found the hard way
`Tensor::triu_mask`/`tril_mask` are named for the triangle they **keep**, not the
one they mask, so a causal mask is `tril_mask(shape, n_kv - n_q)`. Using
`triu_mask` reverses time, and when a decode step has one query and one key it
masks the only position there is — a full row of `-inf` into softmax is `NaN`,
not an error, and it propagates to every logit. Burn's `assert_approx_eq` also
compares `NaN` to `NaN` without complaint, so tests must assert finiteness
separately (`burn-whisper`'s do).

`burn_store::HalfPrecisionAdapter` reads as "widen fp16 on load" and is
**bidirectional** — it also *narrows* fp32 to fp16. `burn-kit`'s `Upcast` is the
one-way version and is what the safetensors loader uses. The bug is silent (the
model just loses mantissa) except on LibTorch, which rejects a conv whose bias
dtype stops matching its input. fp16 and fp32 checkpoints of the same model are
both common on the Hub, so test against both.

Requires **ffmpeg 8.1** dev libraries (and the `ffmpeg` binary for the realtime
`serve` example). ContentVec + RMVPE ONNX assets auto-download from Hugging Face
(`rvc models download` to prefetch). Only 48 kHz is supported today.

## Architecture

Data crosses every crate boundary as **mono `f32`**, and the whole conversion path
is `futures::Stream`-in → `futures::Stream`-out, which is why `rvc serve` is a plain
Unix filter (raw f32le PCM stdin→stdout) and batch `convert` is a thin wrapper.

| crate | role |
|-------|------|
| `burn-kit` | Burn plumbing with no model knowledge: `--device` resolution and checkpoint loading, shared by every network crate |
| `audio-kit` | ffmpeg decode/resample + WAV/raw-PCM I/O, all as `futures::Stream<f32>` |
| `rvc-core` | the voice-conversion pipeline: `FeatureExtractor` (ContentVec + RMVPE), coarse-pitch/upsample/pitch-shift DSP, streaming `Converter` (block/overlap with an **overlapping** crossfade — consecutive kept blocks share `xf_out` output samples so the blend adds, never deletes, audio), an optional post de-hiss stage (`denoise.rs`, `--denoise`), and **all three** generator backends (ort, Burn/LibTorch, Burn/CubeCL) behind one `Generator` trait |
| `burn-rvc` | the RVC v2 network itself (standalone Burn port of `SynthesizerTrnMs768NSFsid` + `MultiPeriodDiscriminator`); no app deps |
| `burn-whisper` | the Whisper network (standalone Burn port); mirrors HF's `state_dict` layout so `openai/whisper-large-v3-turbo` loads unchanged |
| `rvc-train` | native Rust/Burn adversarial training loop (see `crates/rvc-train/ARCHITECTURE.md`) |
| `hub-kit` | auto-download ContentVec/RMVPE ONNX from Hugging Face |
| `rvc-cli` | lib **and** the `rvc` binary (clap): `convert`, `serve`, `models`, `train`, `preprocess` |
| `stt-core` | speech recognition: Whisper log-mel front-end, BPE vocabulary, KV-cached greedy decode, segmentation via `audio-kit`'s slicer, and **two runtimes** (native Burn, ONNX Runtime) behind one `Engine` trait |
| `stt-cli` | lib **and** the `stt` binary |
| `cli-kit` | logging, shell completions and `--device` parsing, shared by all three binaries |
| `voice-cli` | the `voice` binary: `rvc-cli` and `stt-cli` nested as `voice rvc …` and `voice stt` |

### Three runtimes, one path (the key abstraction)
Everything downstream of the generator is shared: the same `FeatureExtractor`, the
same DSP, and the same streaming `Converter` drive **either** backend through the
`Generator` trait (`crates/rvc-core/src/backend.rs`). `BurnGenerator<B>` is generic
over the Burn compute backend and stores its device; the concrete `cuda_generator`
/ `libtorch_generator` constructors return `impl Generator`, so `rvc-cli` never
names a Burn type and the choice is purely run-time. `auto` picks by the `-m`
extension (`.onnx` → ONNX Runtime), then LibTorch-on-CUDA → CubeCL/CUDA →
LibTorch-on-CPU. Naming a backend or device explicitly is an error if it is
unavailable, never a fallback. The CubeCL/CUDA generator is still slower than
realtime for `serve`; `--backend tch` is ~9x faster per file on an RTX 2060 and is
what `auto` picks for `.safetensors` weights.

Device selection lives in `crates/burn-kit/src/device.rs` (`DeviceSpec`,
`cuda_device`, `libtorch_device`, `wgpu_device`, `guard_init`) — a crate that knows
about no model, so every engine and every subcommand resolves `auto` identically.
Two traps encoded there: `LibTorchDevice::default()` is **CPU** (unlike `CudaDevice::default()`), and
constructing `LibTorchDevice::Cuda` on a CPU-only LibTorch is a hard panic baked in
by `burn-tch`'s build script — so `libtorch_device` is the only place it is built.

### The `burn` / `cuda` / `tch` features
`rvc-core`'s Burn backend (`burn_backend.rs`, `pub BurnGenerator<B>`) is behind the
optional `burn` feature (off by default) so consumers that only need the shared
`FeatureExtractor` stay a lean `ort` crate. `burn` alone gives the *generic*
generator and no compute backend; `cuda` and `tch` each add one and can both be on
at once. `rvc-train` mirrors the same features. `rvc-cli` defaults to all three —
one binary, run-time choice. — and `crates/rvc-core/build.rs`
refuses a `tch` build with no `LIBTORCH`, because `burn-tch` hardcodes
`tch/download-libtorch` and cargo features are additive, so the silent fallback
would otherwise be a multi-GB download of a **CPU-only** LibTorch.
`crates/{rvc,stt,voice}-cli/build.rs` bake `$LIBTORCH/lib` into the binary as a
`RUNPATH`; without it a missing `libtorch.so` aborts in `ld.so` before `main`, on
every subcommand. They are deliberate duplicates — an rpath is per-executable.

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

Native Rust/Burn on a GPU; `trainer::run` is generic over `AutodiffBackend` and
`crates/rvc-train/src/lib.rs` picks the concrete one at run time from `--backend`.
All three train. One constraint shapes `burn-rvc`: burn 0.21's autodiff builds a
wrongly-shaped weight gradient for a *grouped, strided* `conv1d` whose padded length
isn't a multiple of the stride — CubeCL and WebGPU absorb it, LibTorch aborts.
`DiscriminatorS` is exactly that shape, so `DiscriminatorS::forward`
(`discriminator.rs`) reflect-pads its input to a length (`SCALE_ALIGN`) the whole
chain divides evenly. Don't remove it
without re-running `cargo run -p rvc-train --example convgrad --features tch,cuda,wgpu`. No `Learner` (the GAN loop doesn't fit it: `TrainStep::step`
takes `&self` and yields one `GradientsParams` for one optimizer, while a GAN needs
two models, two optimizers at different LRs, and D updated *between* the two
backward passes) — the
trainer drives Burn's `TuiMetricsRendererWrapper` directly (`crates/rvc-train/src/dashboard.rs`).
On a TTY a live dashboard shows `g`/`d`/`mel` losses; `q` stops early and saves. Off-TTY
(or `--no-tui`), it logs to stderr and Ctrl-C stops and saves. Losses match RVC exactly
(mel-L1 ×45, KL ×1, feature-matching ×2, LSGAN); the STFT front-end is n_fft=2048,
hop=480, 128 Slaney mels, center=False (`crates/rvc-train/src/spectral.rs`). Target GPU
is 6 GB (RTX 2060) → small batch. Warm-start from `--pretrained-g/-d` is strongly
recommended on a small corpus.

**Preprocess first (`rvc preprocess`).** `sample_batch` draws random 0.48 s
windows uniformly across each corpus file, so raw recordings full of
between-sentence dead-air collapse the generator to silence. `rvc preprocess
raw/*.mp3 -o clips/` then `rvc train clips/*.wav ...` slices the corpus into
clean per-sentence clips first — it removes between-sentence dead-air while
**preserving soft/breathy ASMR content** (energy is used only to find long
silent gaps, never to gate quiet-but-present sound). The shared slicer lives in
`crates/audio-kit/src/slice.rs` (`SliceOptions`, `slice`); the two tuning knobs
are `--silence-db` (energy floor; lower to keep the softest passages) and
`--min-silence` (how long a quiet gap must last to be a cut, so sentences are
never split). Training itself is unchanged — it just consumes the cleaned folder.
