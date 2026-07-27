# rvc — RVC voice conversion toolkit

Retimbre any voice recording into a voice you like. **Voice conversion (VC)**
runs as pure-Rust inference (RVC v2) — native Burn (GPU) or ONNX Runtime — with a
streaming Unix-filter CLI, and training is a native Rust (burn) pipeline: no
Python in the toolkit except the single weight → ONNX conversion step.

## Architecture

| crate | role |
|-------|------|
| `rvc-audio` | ffmpeg (8.1) mp3/wav decode, resample, WAV/raw-PCM I/O — all as `futures::Stream` of mono `f32` |
| `rvc-core` | the voice-conversion pipeline: ContentVec + RMVPE feature extraction, and **both** generator backends (ONNX Runtime via `ort`, and native Burn behind the `burn` feature) behind one `Generator` trait; `Stream`-in → `Stream`-out `Converter`; reusable `FeatureExtractor` |
| `burn-rvc` | the RVC v2 network itself, a standalone Burn port of `SynthesizerTrnMs768NSFsid` |
| `rvc-hub` | auto-download ContentVec/RMVPE ONNX from Hugging Face |
| `rvc-train` | native (Rust/burn) RVC generator training — see `crates/rvc-train/ARCHITECTURE.md` |
| `rvc-cli` | the `rvc` binary (clap) |

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
                                #    models/checkpoint/voice.best.disc.safetensors
```

The pair is a normal checkpoint — deploy it with `rvc convert -m
models/checkpoint/voice.best.safetensors`, or `--resume` from it (its sidecar is
the *matching* discriminator, not the final run's). What's compared is the mean
`mel` over a window sized to give ~20 evaluations across the run, not a single
step, since the per-step loss is noisy enough that its minimum is mostly luck;
what's written is the same thing the final save writes (the EMA when enabled).
Pass `--no-save-best` to skip it (saves ~20 extra weight writes over a run).

Notes: only 48 kHz is supported today; defaults are `-e 5` epochs and `-b 2`
(safe on a 6 GB RTX 2060). One epoch is one pass over the corpus and can be slow
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
  saves only those.
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
ORT than the Burn (CUDA) backend today — or to deploy the model in another
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
# Burn generator (native) — pass the trained .safetensors
rvc convert -m models/voice.safetensors --model-sr 48000 -o out/  input1.mp3 input2.mp3

# ONNX Runtime generator — pass a .onnx (built by the Export step above)
rvc convert -m models/voice.onnx --model-sr 48000 -o out/  input1.mp3 input2.mp3
```

Writes `out/input1_voice.wav`, `out/input2_voice.wav` (named
`<input>_<model>.wav`) in the target timbre. `--backend`
(`auto`/`burn`/`onnx`) picks the generator; `auto` chooses by file extension
(`.onnx` → ONNX Runtime, else Burn). The Burn generator runs on the GPU (CUDA).

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

`serve` takes the same `--backend` flag as `convert`; both runtimes stream
through the same `Converter`. Use ONNX Runtime for realtime — the Burn (CUDA)
generator works but is currently slower than onnx. Logs go to stderr, so
stdout carries only PCM.

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

## Status

- **Native training works** (Rust/Burn, GPU): the full RVC v2 generator +
  MultiPeriod discriminator are ported to Burn (`crates/burn-rvc`), warm-start
  from the public pretrained bases, and fine-tune adversarially (mel-L1 + KL +
  feature-matching + LSGAN) on cuda. Verified end-to-end on a real clip —
  losses decrease and the saved `.safetensors` round-trips through
  `rvc convert`.
- **Inference works on both backends**, and both `convert` and `serve` run
  either the native Burn generator (GPU) or ONNX Runtime through one shared
  `Converter` (`--backend`).
- **Live training dashboard**: Burn's TUI shows loss plots + progress; `q` (or
  Ctrl-C without the TUI) stops early and saves.

- **ONNX export works**: `export/` is a small standalone uv/python script that converts
  a Burn `.safetensors` to ONNX (clean-room torch, no RVC repo); verified by
  running the result through `rvc convert --backend onnx`. See
  [`export/README.md`](export/README.md).

## Roadmap

- 40 kHz training.
- Faster Burn (CUDA) inference so `serve` is realtime on the native backend.
- Index/retrieval blend + `protect` for even tighter timbre match.
- TTS (text → voice) — deferred.
