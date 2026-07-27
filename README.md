# rvc — RVC voice conversion toolkit

Retimbre any voice recording into a voice you like. **Voice conversion (VC)**
runs as pure-Rust inference (RVC v2) — native Burn on LibTorch or CubeCL/CUDA, or
ONNX Runtime — with a streaming Unix-filter CLI, and training is a native Rust
(burn) pipeline: no Python in the toolkit except the single weight → ONNX
conversion step.

**New here?** [Requirements](#requirements) → [Install](#install) →
[Convert a file](#convert-amp3-into-a-trained-voices-timbre). Building on it?
[For developers](#for-developers).

## Requirements

| | needed by | how it's found |
|---|---|---|
| **ONNX Runtime** | everyone — it runs ContentVec + RMVPE on every path, including the native ones | dlopened at run time from `ORT_DYLIB_PATH` |
| **LibTorch** | optional — only `--backend tch` | linked at build time; found automatically at run time |
| **ffmpeg 8.1** | everyone (decode/resample) | system libraries |

Neither runtime is bundled or downloaded. Point at your own:

```sh
export ORT_DYLIB_PATH=/absolute/path/to/libonnxruntime.so   # required if not installed system-wide
```

ONNX Runtime is loaded dynamically, so this is a **run-time** variable — the same
binary works against any 1.24-compatible build, CPU or CUDA. The CUDA execution
provider is tried first, then CPU.

### LibTorch (optional)

Only needed for `--backend tch`, which is ~9x faster than the CUDA backend for
inference (see [Compute backends](#compute-backends--devices)). Skip it and
everything else still works.

Unlike ONNX Runtime, LibTorch is *linked*, so it must be present when the binary
is built. **Version must be 2.9.0** — that is what the `tch 0.22` bindings are
generated against, and a distro PyTorch is usually too new.

```sh
# libTorch 2.9.0 currently includes binary distributions with CUDA 12.6, 12.8 or 13.0 runtimes. 
wget https://download.pytorch.org/libtorch/cu126/libtorch-shared-with-deps-2.9.0%2Bcu126.zip
unzip libtorch-shared-with-deps-2.9.0+cu126.zip     # -> ./libtorch
export LIBTORCH=$PWD/libtorch
```

At run time the binary finds LibTorch by itself — **`LD_LIBRARY_PATH` is never
needed**. It searches in order:

1. the `LIBTORCH` it was built against,
2. `libtorch/` next to the binary,
3. `libtorch/` in the current working directory.

So keeping a `./libtorch` in your project directory is enough even after moving
the binary.

## Install

```sh
export LIBTORCH=$PWD/libtorch      # omit to build without the tch backend
cargo build --release              # -> target/release/rvc
```

Without LibTorch:

```sh
cargo build --release --no-default-features --features cuda
```

`--backend tch` then reports that it wasn't compiled in, and everything else is
unchanged.

## Compute backends & devices

One `rvc` binary carries all three generator runtimes; `--backend` chooses at run
time. `rvc train` takes the same flag, minus `onnx` (there is no ONNX training path).

| `--backend` | aliases | runtime | devices |
|---|---|---|---|
| `onnx` | | ONNX Runtime (`ort`) | CUDA EP, else CPU |
| `cuda` | `burn`, `burn-cuda` | native Burn, CubeCL/CUDA kernels | NVIDIA only |
| `tch` | `libtorch`, `burn-tch` | native Burn, LibTorch | CUDA, MPS, Vulkan, CPU |
| `wgpu` | `webgpu`, `burn-wgpu` | native Burn, WebGPU | any Vulkan/Metal/DX12 GPU |
| `auto` *(default)* | | `.onnx` weights → `onnx` (inference only), else the first of: LibTorch on a GPU, CubeCL/CUDA, WebGPU, LibTorch on CPU | |

`--device` picks *which* device inside the chosen backend: `auto` (default),
`cpu`, `gpu`, `gpu:N`, `mps`, `vulkan` (`cuda`/`cuda:N` are accepted spellings of
`gpu`). Multiple GPUs are addressed by index (`--device gpu:1`). `cuda` is the
only backend with no CPU device; LibTorch and WebGPU both have one.

`auto` picks the same way for `convert`, `serve` and `train`, and never picks a
backend it can tell won't run. It cannot always tell: with LibTorch linked its
device count settles the question, but in a build without it (`--features
cuda,wgpu`) nothing here can probe, and WebGPU is preferred because it also runs
on NVIDIA while CubeCL fails on anything else. Name a backend explicitly to
override.

Naming a backend or device that isn't available is an **error with a reason**,
never a silent fallback; only `auto` substitutes.

`wgpu` needs no vendor toolkit and runs on AMD, Intel and Apple GPUs, so it is
the portable fallback where neither CUDA nor LibTorch is available.

```sh
rvc convert -m models/voice.safetensors --backend tch  --device cuda:0 -o out/ in.mp3
rvc train clips/*.wav -o models/voice   --backend cuda --device cuda:0
```

### Inference speed (RTX 2060, 48 kHz, 8 files, 3 runs)

| backend | first file | warm median | 8 files total |
|---|---|---|---|
| `tch` (LibTorch) | 0.58 s | **0.24 s** | 6.0 s |
| `cuda` (CubeCL) | 3.76 s | 2.28 s | 23.8 s |

**LibTorch is ~9× faster per file in steady state.** The gap is not CubeCL
JIT warm-up: the first file is 6.5× faster too. That is why `auto` prefers it.
Timings are the CLI's own per-file log deltas, which start after the model is
loaded; the first file is listed separately because CubeCL compiles kernels on
first use.

### Training

All three backends train. The saved weights are identical in kind — a model
trained on one loads on any other.

```sh
rvc train clips/*.wav -o models/voice --backend tch     # or cuda, wgpu
```

One wrinkle worth knowing, because it shapes the code: burn 0.21's autodiff
produces a wrongly-shaped weight gradient for a *grouped, strided* `conv1d` whose
padded input length isn't a multiple of the stride. CubeCL and WebGPU absorb it;
LibTorch checks shapes strictly and aborts. The scale discriminator is exactly
that shape, so `burn-rvc` reflect-pads its input to a length the whole conv chain
divides evenly (`SCALE_ALIGN`). That costs under 1% of a training segment,
keeps every backend on the same arithmetic, and leaves weight shapes — so
pretrained warm-start — untouched.

```sh
# which backends survive which convolution shapes
cargo run -p rvc-train --example convgrad --features tch,cuda,wgpu
```

### Multi-GPU (training only)

`rvc train` takes a comma-separated device list — `--devices` reads better but is
the same flag as `--device`. The first entry is the master: it holds the weights,
both optimizer states and the EMA, and it is the only device that writes
checkpoints.

```sh
rvc train clips/*.wav -o models/voice --backend tch --devices gpu:0,gpu:1
```

Each device draws its own micro-batch and its gradients are copied back to the
master and averaged, so **`-b` is per device** and the effective batch is
`batch × grad-accum × devices`. That buys a steadier gradient, not a shorter
epoch: devices are dispatched in sequence, and replicas are re-copied from the
master every step because there is no all-reduce. Treat N>1 as a way to use
memory you already have, not as a speed-up.

Duplicates are rejected after resolution, so `--devices auto,gpu:0` is an error
rather than two replicas quietly sharing one card. Mixed device kinds are allowed
(`--devices gpu:0,cpu` works, if slowly). Inference is single-device by design —
`convert` and `serve` take one `--device`, because the streaming converter's
overlap-crossfade carries state from block to block; run one process per GPU for
batch throughput.

N=1 takes exactly the single-device path, with no clone and no copy. **N>1 has
only been exercised as GPU+CPU on one machine** — the two-GPU path is
untested.

## Usage

### Convert `a.mp3` into a trained voice's timbre

The target timbre lives in a **trained generator**, so this is two stages: train
a generator on the *target voice* once, then convert *source* clips through it.
The corpus is passed as a vector of paths — give it as much clean audio of the
target voice as you have.

```sh
export ORT_DYLIB_PATH=/usr/lib/libonnxruntime.so

# 1. Train a generator on the target voice (once, offline) -> models/personA.safetensors.
rvc train --out models/personA --model-sr 48000  /tmp/personA.mp3

# 2. Convert a.mp3 through it -> out/a.wav in that timbre (native Burn backend).
rvc convert -m models/personA.safetensors --model-sr 48000 -o out/ /tmp/a.mp3
```

If the source and target sit in different pitch ranges, add e.g. `-t 2` (up) or
`-t -3` (down) to step 2.

### 0. Preprocess the corpus (recommended)

`rvc train` draws random short windows uniformly across each corpus file, so
raw recordings full of between-sentence dead-air teach the generator to output
silence. `rvc preprocess` first slices the corpus into clean per-sentence clips
with that dead-air removed — **without** discarding quiet content (soft, breathy
ASMR passages are low-energy but wanted, so they are preserved by construction):

```sh
rvc preprocess raw/*.mp3 -o clips/        # or a directory: rvc preprocess raw/ -o clips/
rvc train clips/*.wav --out models/voice --model-sr 48000 \
  --pretrained-g models/pretrained/f0G48k.pth --pretrained-d models/pretrained/f0D48k.pth
```

Energy is used **only** to find long silent gaps between sentences: a gap is a
cut point only when it is both below the energy floor and lasts longer than
`--min-silence` (default 0.3 s), so a complete sentence is never split and short
internal pauses / soft tails stay inside the clip. Two knobs tune it:

- `--silence-db` (default `-40`) — the energy floor. **Lower** it (e.g. `-50`)
  to keep the very softest passages; raise it to strip more aggressively.
- `--min-silence` (default `0.3`) — how long a quiet gap must last to count as a
  between-sentence cut.

Other flags: `--min-clip` (drop clips shorter than, default 1 s), `--max-clip`
(cap clip length, `0` = never split), `--pad` (edge-pad each clip with bordering
quiet so onsets/tails aren't clipped, default 0.15 s), `--normalize` (peak-
normalize each clip), and `--model-sr` (match your training rate). The output
folder is inspectable — listen to a few clips before training.

### 1. Fine-tune a generator (native Rust/Burn, GPU)

Training runs natively in Rust (Burn on CUDA) — no Python. Warm-start
from the public pretrained bases (`f0G48k.pth`/`f0D48k.pth`) for good results on
a small corpus. Get the bases from Hugging Face `lj1995/VoiceConversionWebUI`
(`assets/pretrained_v2/`) and point `--pretrained-g/-d` at them:

```sh
rvc train --out models/voice --model-sr 48000 \
  --pretrained-g models/pretrained/f0G48k.pth --pretrained-d models/pretrained/f0D48k.pth \
  clip1.mp3 clip2.mp3 clip3.mp3
```

This writes the generator to `models/voice.safetensors` (the EMA weights — see
**Tuning**), the raw final weights to `models/voice.raw.safetensors`, and the
discriminator sidecar `models/voice.disc.safetensors` (which makes `--resume`
below resume the adversary too). Only the final model is written — no
periodic-checkpoint clutter. The shared ContentVec + RMVPE ONNX assets (training
features + inference) download automatically, or prefetch them:

```sh
rvc models download
```

**Dashboard & early stop.** On a terminal, training shows Burn's live TUI
dashboard; logs go to `{save-dir}/train/train.log`. It plots the `g`/`d`/`mel`
losses **and the learning rate**, and shows **CPU and GPU usage** (GPU memory,
utilisation, and power via NVML) as text. Press `q` to stop early — the model is
saved. Without a TTY (or with `--no-tui`), it logs `g`/`d`/`mel`/`lr` to stderr
and **Ctrl-C** stops and saves. Watch **`mel`** for quality — it should fall then
plateau; `g`/`d` are adversarial and just stay balanced. A `mel` that plateaus
*and oscillates* in the back half is normal for a GAN, but see **Tuning** below
if the audio is muffled or staticky.

**Resume a run (`--resume`).** Stopped early and want to keep going? Point
`--resume` (alias `--continue`) at the generator `.safetensors` from the earlier
run instead of a pretrained base — it loads the generator, and picks up the
discriminator from the `.disc.safetensors` sidecar automatically:

```sh
rvc train --out models/voice --model-sr 48000 \
  --resume models/voice.safetensors \
  clip1.mp3 clip2.mp3 clip3.mp3
```

You can pass either `models/voice.safetensors` (the EMA output) or its
`models/voice.raw.safetensors` twin — when the raw twin is present it's
**preferred automatically**. The raw (non-EMA) weights are the real last-step
generator that co-evolved with the saved discriminator, so `raw-G ↔ live-D` is
the faithful pairing to continue the adversarial game from; the EMA snapshot is a
smoothed average that never itself faced the discriminator. The discriminator is
saved once (the live one — it's training scaffolding, never deployed, so there's
no EMA variant), and both the EMA and raw generator paths resolve to that same
`.disc.safetensors` sidecar.

`--resume` conflicts with `--pretrained-g` (the checkpoint replaces the base).
If the discriminator sidecar is missing it falls back to `--pretrained-d` (pass
it), else the discriminator starts fresh. Note that optimizer (AdamW) momentum
is not persisted across runs — negligible for short fine-tunes.

**The best checkpoint (on by default).** A run that drifts in its last stretch
would otherwise lose its best model, so whenever the `mel` loss reaches a new low
the trainer snapshots the generator **and** its discriminator next to the output:

```sh
rvc train --out models/voice …  # -> models/checkpoint/voice.best.safetensors
                                #    models/checkpoint/voice.best.raw.safetensors
                                #    models/checkpoint/voice.best.disc.safetensors
                                #    models/checkpoint/voice.best.json  (its score)
```

That's the **same file set as the final output** — EMA weights, raw twin, and
discriminator sidecar — so a best checkpoint behaves exactly like one: deploy it
with `rvc convert -m models/checkpoint/voice.best.safetensors`, or `--resume`
from it and get *its* raw twin and *its* discriminator (never the final run's):

```sh
rvc train --out models/voice --resume models/checkpoint/voice.best.safetensors …
```

What's compared is the mean `mel` over a window of at most 50 steps, not a single
step, since the per-step loss is noisy enough that its minimum is mostly luck. The
score is kept in the `.best.json` sidecar and read back on the next run, so
resuming can only *improve* on the best you already have rather than overwrite it
with wherever the new run happens to start. Pass `--no-save-best` to skip it.

Notes: only 48 kHz is supported today; defaults are `-e 5` epochs and `-b 4`.
Batch size is **per device**, so a multi-device run multiplies it — lower `-b` if
a 6 GB card runs out of memory. One epoch is one pass over the corpus and can be slow
on the native Burn/CUDA trainer — lean on early stop (`q`/Ctrl-C, which saves)
rather than a big `-e`.

**Tuning (muffled / static / shaking plateau).** The loss magnitudes match RVC
exactly; these flags shape the training *dynamics* on a small corpus. The LR
schedule and EMA **auto-scale to the run length**, so they stay sensible at any
`-e`. Two are on by default:

- `--ema-frac` (default `0.1`, **on**) — the **saved model is an exponential
  moving average** of the generator weights over a window = this fraction of the
  run, averaging out the adversarial oscillation so it's cleaner than the raw
  final step. EMA is a *saved snapshot only* — it does **not** slow learning. The
  raw weights are always also written to `<out>.raw.safetensors`; `--ema-frac 0`
  saves only raws.
- `--lr-final` (default `0.1`, **on**) with `--lr` (default `1e-4`) — the LR
  decays exponentially from `--lr` to `--lr × --lr-final` over the whole run. A
  constant LR bounces around the minimum; decay lets the late-training oscillation
  settle. Set `--lr-final 1.0` to disable.
- `--grad-accum` (default `1`) — accumulate N micro-batches per optimizer step
  for an effective batch of `batch × N` at no extra VRAM (steadier gradients on a
  6 GB GPU; ~N× slower per epoch). Try `2`–`4`.
- `--d-lr-ratio` (default `1.0`) / `--d-interval` (default `1`) — rein in an
  over-eager discriminator if the output has **buzzy, high-frequency static**:
  set `--d-lr-ratio 0.5` or `--d-interval 2`.
- `--snr-weight` (default `0.0` = uniform) — bias sampling toward cleaner clips by
  `snr^alpha`, using each clip's **noise-floor SNR (not loudness)**, so soft/breathy
  ASMR passages are preserved.

If the static is a steady hiss rather than buzz, it may be baked into the corpus
(the generator faithfully reproduces recorded hiss); de-hiss the *training* audio
upstream rather than reaching for these knobs. See
[`crates/rvc-train/ARCHITECTURE.md`](crates/rvc-train/ARCHITECTURE.md) for the
implementation.

### 2. Export to ONNX (optional — for ONNX Runtime / cross-framework deploy)

Native Burn inference needs **no export**: `rvc convert`/`serve` run a trained
`.safetensors` directly on the GPU. Export only matters when you want to run the
generator under **ONNX Runtime** — notably realtime `serve`, which is faster on
ORT than the CubeCL/CUDA backend today — or to deploy the model in another
framework.

`export/` is the toolkit's **only Python**: a small, self-contained `uv` project
whose `rvc_infer.py` is a *clean-room* torch reimplementation mirroring the
`burn-rvc` network — it does **not** depend on the RVC-Project repo.
`export_onnx.py` loads a trained `.safetensors` and runs `torch.onnx.export`:

```sh
uv run --project export python export/export_onnx.py \
  models/voice.safetensors models/voice.onnx
```

Then point `convert`/`serve` at the `.onnx` — the ONNX Runtime backend is chosen
automatically by the extension (and needs `ORT_DYLIB_PATH`, see Prerequisites):

```sh
rvc convert -m models/voice.onnx --model-sr 48000 -o out/ input.mp3
```

The exported graph matches exactly what `rvc-core` feeds the generator:

```
phone[1,T,768] f32,  phone_lengths[1] i64,  pitch[1,T] i64,
pitchf[1,T] f32,     ds[1] i64,             rnd[1,192,T] f32   →   audio[1,1,L] f32
```

Setup and details: [`export/README.md`](export/README.md).

### 3. Batch-convert files

```sh
# Native Burn generator — pass the trained .safetensors
rvc convert -m models/voice.safetensors --model-sr 48000 -o out/  input1.mp3 input2.mp3

# ONNX Runtime generator — pass a .onnx (built by the Export step above)
rvc convert -m models/voice.onnx --model-sr 48000 -o out/  input1.mp3 input2.mp3

# Force a compute backend and device
rvc convert -m models/voice.safetensors --model-sr 48000 \
  --backend tch --device gpu:0 -o out/  input1.mp3
```

Writes `out/input1_voice.wav`, `out/input2_voice.wav` (named
`<input>_<model>.wav`) in the target timbre.

`--backend` picks the generator: `auto` (default), `onnx`, `cuda`, `tch` or
`wgpu` — see [Compute backends & devices](#compute-backends--devices) for the
aliases and what each needs. `auto` takes `.onnx` weights to ONNX Runtime and
anything else to the fastest native backend available.

Add `--denoise` to strip steady background **hiss** from the output — a
conservative spectral suppressor that removes the constant noise floor while
preserving soft/breathy content (breaths sit *above* the floor). For heavier
cleanup, run the WAV through ffmpeg's adaptive denoiser instead:
`ffmpeg -i out/input_voice.wav -af afftdn=nf=-25,highpass=f=60 clean.wav`.

### 4. Realtime — a Unix filter (raw f32le PCM stdin → stdout)

Input is mono **f32le @ 16 kHz** (the RVC analysis rate); output is mono f32le at
the model's sample rate. ffmpeg handles capture/playback on either end:

```sh
# play a file through the converter
ffmpeg -i in.mp3 -f f32le -ar 16000 -ac 1 - \
  | rvc serve -m models/voice.onnx --model-sr 48000 \
  | ffplay -f f32le -ar 48000 -

# live mic → converted → speakers
ffmpeg -f alsa -i default -f f32le -ar 16000 -ac 1 - \
  | rvc serve -m models/voice.onnx --model-sr 48000 \
  | ffplay -f f32le -ar 48000 -
```

`serve` takes the same `--backend` and `--device` flags as `convert`; every
runtime streams through the same `Converter`. For realtime, prefer `--backend
tch` or `onnx`: `cuda` (CubeCL) is ~9x slower per block and does not keep up.
Logs go to stderr, so stdout carries only PCM.

`serve` also takes `--denoise` (same conservative de-hiss as `convert`, ~21 ms
extra latency). Or denoise downstream with ffmpeg:

```sh
ffmpeg -i in.mp3 -f f32le -ar 16000 -ac 1 - \
  | rvc serve -m models/voice.onnx --model-sr 48000 \
  | ffmpeg -f f32le -ar 48000 -ac 1 -i - -af afftdn=nf=-25,highpass=f=60 \
      -f f32le -ar 48000 -ac 1 - \
  | ffplay -f f32le -ar 48000 -ac 1 -
```

### 5. Shell completions

`rvc completions <shell>` prints a completion script (generated from the actual
flags, so it never drifts) to stdout — `bash`, `zsh`, `fish`, `powershell`, or
`elvish`:

```sh
rvc completions zsh  > ~/.zfunc/_rvc
rvc completions bash > /etc/bash_completion.d/rvc
rvc completions fish > ~/.config/fish/completions/rvc.fish
```

## For developers

### Building from source

Same as Install, plus: **keep LibTorch at the project root as `./libtorch`.**
The build needs `LIBTORCH` set, and several crates' test binaries link it too
(cargo unifies features across the workspace), so the test suite needs the
library on the loader path:

```sh
export ORT_DYLIB_PATH=/usr/lib/libonnxruntime.so
export LIBTORCH=$PWD/libtorch

cargo build --release
cargo clippy --workspace --all-targets     # kept clean
cargo fmt

LD_LIBRARY_PATH=$PWD/libtorch/lib cargo test --workspace
```

`LD_LIBRARY_PATH` is a **test-only** requirement: cargo runs each test binary
with its own package directory as the working directory, so the relative search
path baked into `rvc` doesn't apply to them. The `rvc` binary itself never needs
it.

### Checking the network port

There are no unit tests for the network. Correctness is checked by loading the
real pretrained weights and reporting coverage — the whole generator must load
560/560 params with 0 missing:

```sh
cargo run -p burn-rvc --example load  -- models/pretrained/f0G48k.pth
cargo run -p burn-rvc --example infer -- models/pretrained/f0G48k.pth
```

Both take `--backend ndarray|cuda|tch` (default `ndarray`, needs no GPU); the
counts must come out identical on every backend.

### Backend conformance probe

`convgrad` checks which backends survive the convolution shapes the
discriminator uses — this is what identified the LibTorch training bug:

```sh
LD_LIBRARY_PATH=$PWD/libtorch/lib \
  cargo run -p rvc-train --example convgrad --features tch,cuda,wgpu
```

### Crate layout

| crate | role |
|-------|------|
| `rvc-audio` | ffmpeg (8.1) mp3/wav decode, resample, WAV/raw-PCM I/O — all as `futures::Stream` of mono `f32` |
| `rvc-core` | the voice-conversion pipeline: ContentVec + RMVPE feature extraction, and **every** generator backend (ONNX Runtime via `ort`; native Burn on LibTorch or CubeCL/CUDA, behind the `tch`/`cuda` features) behind one `Generator` trait; `Stream`-in → `Stream`-out `Converter`; reusable `FeatureExtractor` |
| `burn-rvc` | the RVC v2 network itself, a standalone Burn port of `SynthesizerTrnMs768NSFsid` |
| `rvc-hub` | auto-download ContentVec/RMVPE ONNX from Hugging Face |
| `rvc-train` | native (Rust/burn) RVC generator training — see `crates/rvc-train/ARCHITECTURE.md` |
| `rvc-cli` | the `rvc` binary (clap) |

## Status

- **Native training works** (Rust/Burn, GPU): the full RVC v2 generator +
  MultiPeriod discriminator are ported to Burn (`crates/burn-rvc`), warm-start
  from the public pretrained bases, and fine-tune adversarially (mel-L1 + KL +
  feature-matching + LSGAN) on cuda. Verified end-to-end on a real clip —
  losses decrease and the saved `.safetensors` round-trips through
  `rvc convert`.
- **Three compute backends, one binary**: CubeCL/CUDA, LibTorch and WebGPU, all
  for both inference and training, picked at run time with `--backend`. LibTorch
  is ~9× faster per file than CubeCL/CUDA on an RTX 2060 and is what `auto`
  chooses for `.safetensors` weights.
- **Live training dashboard**: Burn's TUI shows loss plots + progress; `q` (or
  Ctrl-C without the TUI) stops early and saves.
- **Data-parallel multi-device training** (`--devices gpu:0,gpu:1`): master-device
  gradient accumulation, N=1 unchanged. Verified as GPU+CPU; two GPUs untested.

- **ONNX export works**: `export/` is a small standalone uv/python script that converts
  a Burn `.safetensors` to ONNX (clean-room torch, no RVC repo); verified by
  running the result through `rvc convert --backend onnx`. See
  [`export/README.md`](export/README.md).

## Roadmap

- 40 kHz training.
- Faster CubeCL/CUDA inference so `serve` is realtime on that backend too
  (`--backend tch` is the faster native path today).
- Multi-GPU training that is actually *faster*: thread-per-device dispatch, and
  persistent replicas with all-reduce (`burn-collective`) instead of copying the
  master's weights out every step. The data-parallel path itself already works.
- Index/retrieval blend + `protect` for even tighter timbre match.
- TTS (text → voice) — deferred.
