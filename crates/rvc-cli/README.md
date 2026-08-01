# rvc — RVC voice conversion

Retimbre a recording into a voice you trained. `rvc` is pure-Rust RVC v2: it
keeps the source's words and F0 pitch and replaces only the timbre, so breathy
and expressive passages survive by construction. Inference runs on ONNX Runtime
or native Burn; training runs on any Burn backend.

Runtimes, drivers and ffmpeg: [`docs/setup.md`](../../docs/setup.md). What the
network computes: [`docs/rvc-architecture.pdf`](../../docs/rvc-architecture.pdf).
Build it with `cargo build --release -p rvc-cli`. Every command below is also
`voice rvc …` — same code, same flags.

## Commands

| | |
|---|---|
| `rvc -m <weights> …` | **the bare invocation is the filter** — raw f32le mono PCM on stdin, converted PCM on stdout |
| `rvc convert <files…>` | batch-convert files to WAV |
| `rvc train <corpus…>` | fine-tune a generator on a target voice |
| `rvc preprocess <files…>` | slice a corpus into clean per-sentence clips |
| `rvc models download` | prefetch the shared ContentVec + RMVPE assets |
| `rvc completions <shell>` | completion script for bash, zsh, fish, powershell or elvish |

## Flags

Inference — the bare filter and `convert` take the same set:

| flag | default | |
|---|---|---|
| `-m`, `--model` | *required* | trained generator: `.safetensors` (Burn) or `.onnx` |
| `--model-sr` | `48000` | generator output rate; only 48 kHz is supported today |
| `-t`, `--transpose` | `0` | pitch shift in semitones, for a source and target in different ranges |
| `--backend` | `auto` | see [Backends](#backends) |
| `--device` | `auto` | `cpu`, `gpu`, `gpu:N`, `mps`, `vulkan` (`cuda`/`cuda:N` are accepted spellings of `gpu`) |
| `--speaker-id` | `0` | single-speaker models only have 0 |
| `--denoise` | off | conservative de-hiss: removes a steady noise floor and leaves breaths, which sit above it. Tune with `--denoise-strength` / `--denoise-patch` / `--denoise-research`; costs ~21 ms of latency |
| `--content`, `--rmvpe`, `--cache-dir` | auto-downloaded | the ContentVec / RMVPE assets and where they live — see [Where files land](#where-files-land) |
| `-o`, `--output-dir` | `.` | `convert` only; writes `<stem>_<model>.wav` |
| `--chunk` | `1600` | filter only: samples per stdin read, so it trades latency against overhead |

`train`:

| flag | default | |
|---|---|---|
| `-o`, `--out` | `models/voice` | writes `<out>.safetensors` (EMA), `<out>.raw.safetensors` and `<out>.disc.safetensors` |
| `-y` | off | **required to overwrite an existing output** — a trained voice is expensive and is never clobbered silently |
| `-e`, `--epochs` / `-b`, `--batch-size` | `5` / `4` | batch size is **per device**; lower it if a 6 GB card runs out of memory |
| `--lr` / `--lr-final` | `1e-4` / `0.1` | the LR decays exponentially to `lr × lr-final` over the run, because a constant one bounces around the minimum. `--lr-final 1.0` disables |
| `--ema-frac` | `0.1` | the saved model is an EMA over this fraction of the run, averaging out adversarial oscillation. `0` saves only the raw weights |
| `--grad-accum` | `1` | micro-batches per optimizer step: a steadier gradient at no extra VRAM, ~N× slower per epoch |
| `--d-lr-ratio` / `--d-interval` | `1.0` / `1` | rein in an over-eager discriminator when the output has buzzy high-frequency static |
| `--snr-weight` | `0.0` | bias sampling toward cleaner clips by `snr^alpha`, on noise-floor SNR rather than loudness so soft passages are not penalised |
| `--resume` | | continue from a generator `.safetensors`, discriminator sidecar included. Conflicts with `--pretrained-g` |
| `--pretrained-g`, `--pretrained-d` | auto-downloaded | warm-start bases; `--no-pretrained` starts from scratch and fetches nothing |
| `--no-save-best` / `--no-tui` | off | stop snapshotting the lowest-`mel` weights / log to stderr instead of the dashboard |
| `--device` | `auto` | comma-separated (alias `--devices`) for data-parallel training; the first is the master |

`preprocess`: `-o` (`dataset`), `--model-sr` (`48000`), `--silence-db` (`-40`),
`--min-silence` (`0.3`), `--min-clip` (`1.0`), `--max-clip` (`0` = never split a
long sentence), `--pad` (`0.15`), `--normalize`.

The two that matter are `--silence-db`, the energy floor — **lower** it (e.g.
`-50`) to keep the very softest passages, raise it to strip harder — and
`--min-silence`, how long a quiet gap must last to count as a sentence boundary,
so a shorter internal pause never splits a sentence. `--pad` keeps bordering
quiet at each edge so onsets and soft tails are not clipped. The output folder is
inspectable: listen to a few clips before committing a training run to them.

## Backends

One binary carries every runtime and `--backend` chooses at run time; the alias
set is identical in `rvc`, `stt`, `tts` and `voice`.

| `--backend` | aliases | runtime | devices |
|---|---|---|---|
| `auto` *(default)* | | `.onnx` weights → `onnx`, else the fastest native backend present | |
| `onnx` | | ONNX Runtime (`ort`) — inference only, there is no ONNX training path | CUDA EP, else CPU |
| `cuda` | `burn`, `burn-cuda` | native Burn, CubeCL/CUDA kernels | NVIDIA only |
| `tch` | `libtorch`, `burn-tch` | native Burn, LibTorch | CUDA, MPS, Vulkan, CPU |
| `wgpu` | `webgpu`, `burn-wgpu` | native Burn, WebGPU | any Vulkan/Metal/DX12 GPU |

**`tch` is ~9× faster per file than `cuda` on an RTX 2060** (0.24 s against
2.28 s warm median at 48 kHz), which is why `auto` prefers it and why realtime
streaming wants `tch` or `onnx` — CubeCL does not keep up. `wgpu` needs no
vendor toolkit, so it is the portable choice on AMD, Intel and Apple GPUs. All
three Burn backends train, and the weights they write are interchangeable.
Naming a backend or device that is unavailable is an **error with a reason**,
never a silent fallback; only `auto` substitutes.

## Train a voice

The timbre lives in a trained generator: train once on the *target* voice, then
convert *source* clips through it. Slice the corpus first — training draws random
short windows uniformly across each file, so raw recordings full of
between-sentence dead-air teach the generator to output silence. Energy finds
*only* the long silent gaps, and never gates sound that is quiet but present, so
soft breathy passages survive the slicing.

```sh
rvc preprocess raw/*.mp3 -o clips/     # or a directory: rvc preprocess raw/ -o clips/
rvc train clips/*.wav -o models/voice --backend tch

# continue a stopped run; -y because models/voice.safetensors already exists
rvc train clips/*.wav -o models/voice --resume models/voice.safetensors -y
```

Warm-start bases (`f0G48k.pth`, `f0D48k.pth`) download beside the run unless
`--no-pretrained` or `--resume` is given; on a small corpus they are what makes
the result usable. On a TTY a dashboard plots `g`/`d`/`mel` and the learning rate
and `q` stops early **and saves**; off a TTY it logs to stderr and Ctrl-C does
the same. Watch `mel` for quality — `g`/`d` are adversarial and only need to stay
balanced. Lowest-`mel` weights are snapshotted to `<out-dir>/checkpoint/`, scored
on a 50-step mean because one step's minimum is luck.

`--resume` prefers the `.raw.safetensors` twin automatically when it is present:
those are the weights that actually faced the saved discriminator, where the EMA
snapshot never did.

Comma-separate `--device` for data-parallel training. The first entry is the
master — it holds the weights, both optimizer states and the EMA, and it is the
only device that writes checkpoints. **`-b` is per device**, so the effective batch
is `batch × grad-accum × devices`. That buys a steadier gradient, not a shorter
epoch: devices are dispatched in sequence. Treat N>1 as a way to use memory you
already have. Inference stays single-device by design, because the streaming
converter's overlap-crossfade carries state from block to block; run one process
per GPU for batch throughput.

```sh
rvc train clips/*.wav -o models/voice --backend tch --devices gpu:0,gpu:1
```

## Convert and stream

```sh
rvc convert -m models/voice.safetensors --model-sr 48000 -o out/ in1.mp3 in2.mp3
rvc convert -m models/voice.safetensors --backend tch --device gpu:0 -o out/ in1.mp3
```

The bare invocation is the filter: mono **f32le at 16 kHz** in (the RVC analysis
rate), mono f32le at the model's rate out. Logs go to stderr, so stdout carries
only PCM.

```sh
# a file, or `-f alsa -i default` for a live microphone
ffmpeg -i in.mp3 -f f32le -ar 16000 -ac 1 - \
  | rvc -m models/voice.safetensors --model-sr 48000 --backend tch \
  | ffplay -f f32le -ar 48000 -
```

`--denoise` handles a steady noise floor on either path. If the result still
hisses it is usually baked into the corpus — the generator faithfully reproduces
recorded hiss — so de-hiss the *training* audio rather than reaching for a
stronger setting. For heavier cleanup after the fact, ffmpeg's adaptive denoiser:
`ffmpeg -i out/in1_voice.wav -af afftdn=nf=-25,highpass=f=60 clean.wav`.

## Export to ONNX (optional)

Native Burn inference needs no export — it runs a trained `.safetensors`
directly. Export only to deploy under ONNX Runtime or another framework; pass the
result wherever you passed the `.safetensors`, and `--backend auto` picks ORT
from the extension. Graph contract: [`export/README.md`](../../export/README.md).

```sh
uv run --project export python export/export_onnx.py models/voice.safetensors models/voice.onnx
```

## Where files land

Assets split by who reads them. **Inference assets** (ContentVec, RMVPE) are
shared across every run, so they live in a cache — `--cache-dir`, else
`RVC_CACHE_DIR`, else `VOICE_CACHE_DIR`, else `voice` under the XDG cache root,
which is `~/.cache` unless `$XDG_CACHE_HOME` names an absolute path. So the
usual answer is `~/.cache/voice`, and `-h` prints whichever it resolves to on
this machine. **Warm-start
bases** belong to one training run rather than to the machine, so they land in
`<out-dir>/pretrained/`, beside the model they produced. Everything else is
written under the directory you invoked from, the training log `./train.log`
included.

## Limits

Only 48 kHz is supported; 40 kHz training is not written yet. There is no
index/retrieval blend or `protect` for tighter timbre match. Multi-device
training works but is not *faster*, and **only GPU+CPU has been exercised, never
two GPUs**. The training loop is documented in
[`docs/training.md`](../../docs/training.md), which covers every trainer in the
toolkit.
