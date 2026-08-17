# Setup — shared by `rvc`, `stt`, `tts`, `seedvc`, `preprocess` and `voice`

All six binaries find their dependencies the same way, take the same
`--backend` and `--device` spellings, and cache downloaded models in the same
place. Driving them *live* — the sample rate at each end of a pipe, playback,
virtual microphones — is [`realtime.md`](realtime.md).

- [Requirements](#requirements)
- [ffmpeg](#ffmpeg)
- [ONNX Runtime](#onnx-runtime)
- [LibTorch](#libtorch)
- [Finding the libraries at run time](#finding-the-libraries-at-run-time)
- [Building](#building)
- [Backends and devices](#backends-and-devices)
- [Where models are stored](#where-models-are-stored)

## Requirements

| | needed by | how it is found |
|---|---|---|
| **ffmpeg 9.0** | everyone — decode, resample and PCM I/O | linked at build time; found automatically at run time |
| **ONNX Runtime** | any `--backend onnx` path, plus `tts`'s prosody encoder and `seedvc`'s six exported graphs | dlopened on first use from `ORT_DYLIB_PATH` |
| **LibTorch 2.9.0** | optional — only `--backend tch` | linked at build time; found automatically at run time |

**Neither machine-learning runtime is bundled or downloaded.**

## ffmpeg

The **9.0** development libraries, from your package manager. They are linked,
not dlopened, so they must be present at build time.

To build against your own instead, name it the way you name LibTorch:

```sh
export FFMPEG_DIR=$PWD/ffmpeg      # expects ffmpeg/lib and ffmpeg/include
```

An unpacked `./ffmpeg` at the project root that you have *not* named in it fails
the build saying so.

**One feature does not work on 9.0.1: de-hiss.** `rvc --denoise` and
`preprocess denoise` drive ffmpeg's `anlmdn`, which corrupts the heap on that
release and takes the process down with it — reproducible with the stock binary
and nothing of ours involved:

```sh
ffmpeg -cpuflags 0 -filter_threads 1 -f lavfi -i sine=d=1:r=48000 -af anlmdn -f null -
```

There is no flag that avoids it and no workaround on our side; leave `--denoise`
off until the system ffmpeg stops reproducing that. Every other stage — decode,
resample, WAV I/O, `afftdn`, `ebur128` — is unaffected.

## ONNX Runtime

Loaded **dynamically**, so this is a *run-time* variable and one
binary works against any 1.24-compatible build, CPU or CUDA. A run that never
touches ONNX Runtime never looks for it.

```sh
export ORT_DYLIB_PATH=/absolute/path/to/libonnxruntime.so   # if not installed system-wide
```

The CUDA execution provider is tried first, then CPU.

How much it matters depends on the engine: `stt` and `rvc` need it only where
something is actually running on it; for `tts` its absence downgrades the prosody
encoder to zeros with a warning rather than failing; `seedvc`'s ONNX path is six graphs written by `export/export_seedvc.py`, and
it needs `--onnx <dir>` to name them — different from the other three, whose
models are one file or one directory. Each engine's README says which applies.

**`rvc` used to be the exception and no longer is.** It runs three models — a
generator, ContentVec and RMVPE — and until ContentVec and RMVPE were ported to
Burn the last two went through ONNX Runtime on every path, native generator
included, which made this a hard requirement for the engine as a whole. Each of
the three now chooses its runtime separately (see
[Backends and devices](#backends-and-devices)). Note what that does *not* change:
`ort` is still a non-optional dependency of `rvc-core`, so there is no `onnx`
feature to drop and the crate always links it. What changes is when the library
is *opened* — it is dlopened on first use, and a run with all three models on
Burn has no first use. That last step follows from how the sessions are built
rather than from a measurement; treat it as untested until someone runs `rvc` on
a machine with no `libonnxruntime.so` at all.

## LibTorch

Optional, and only for `--backend tch`. Unlike ONNX Runtime it is *linked*, so
it must be present when the binary is **built**. **The version must be exactly
2.9.0** — that is what the `tch 0.22` bindings are generated against, and a
distribution's PyTorch is usually too new.

```sh
# LibTorch 2.9.0 ships binary distributions with CUDA 12.6, 12.8 or 13.0 runtimes.
wget https://download.pytorch.org/libtorch/cu126/libtorch-shared-with-deps-2.9.0%2Bcu126.zip
unzip libtorch-shared-with-deps-2.9.0+cu126.zip     # -> ./libtorch
export LIBTORCH=$PWD/libtorch
```

`LIBTORCH` is a **build-time** variable and nothing more — see
[Finding the libraries at run time](#finding-the-libraries-at-run-time).

## Finding the libraries at run time

ffmpeg and LibTorch are linked, so `ld.so` resolves them before `main` runs; no
variable of ours could be read in time. Each binary carries a relative `RUNPATH`
instead, and the loader searches:

1. `$LD_LIBRARY_PATH`,
2. `ffmpeg/lib` and `libtorch/lib` in the working directory,
3. the same two next to the binary, then one level up for a `bin/` layout,
4. `ld.so.cache` — the system packages.

**No path from the build machine is baked in**, so `ldd` never reports another
machine's directories — and running from a directory holding neither wants
`LD_LIBRARY_PATH`, `cargo test` included:

```sh
LD_LIBRARY_PATH=$PWD/libtorch/lib cargo test --workspace
```

## Building

```sh
export ORT_DYLIB_PATH=/usr/lib/libonnxruntime.so
export LIBTORCH=$PWD/libtorch        # omit to build without the tch backend

cargo build --release                    # all six binaries
cargo build --release -p rvc-cli         # just `rvc`
cargo build --release -p stt-cli         # just `stt`
cargo build --release -p tts-cli         # just `tts`
cargo build --release -p seedvc-cli      # just `seedvc`
cargo build --release -p preprocess-cli  # just `preprocess`
cargo build --release -p voice-cli       # just `voice`
```

The backend features are all on by default, so one binary carries every runtime
and `--backend` chooses at run time. Drop the ones you do not want:

```sh
# No LibTorch to install:
cargo build --release -p rvc-cli --no-default-features --features cuda,wgpu

# Burn only — no ONNX Runtime anywhere in the build:
cargo build --release -p stt-cli --no-default-features --features cuda,tch,wgpu
```

`stt-cli`, `tts-cli` and `seedvc-cli` have `cuda`, `tch`, `wgpu` and `onnx`;
`rvc-cli` has the first three only, because `rvc-core` depends on `ort`
unconditionally and so there is nothing to gate. `preprocess-cli` has the first
three too, for a different reason that is worth not misreading as an omission:
its two models are published as PyTorch checkpoints and nothing in this
workspace exports either, so `--backend onnx` is refused *with that reason*
rather than pointing at a feature flag that could not exist.
`voice-cli` re-declares the same four names and forwards each to the binaries
that have it, so one `--no-default-features --features cuda` line means the same
thing for every binary.

Naming a backend that was not compiled in is an error that says so, and nothing
else changes.

**A debug build runs models at full speed.** The `dev` profile gives
dependencies `opt-level = 3` — which is where every tensor operation lives —
while leaving workspace crates cheap to recompile. Transcribing 24 s of audio
takes 19 s in debug against 18 s in release. Reach for `--release` to benchmark,
not to test.

## Backends and devices

`--backend` chooses the *runtime*; `--device` chooses which device inside it.
Both default to `auto`, and **both take the same spellings on every binary**.

| `--backend` | aliases | runtime | devices |
|---|---|---|---|
| `onnx` | | ONNX Runtime (`ort`) | CUDA EP, else CPU |
| `cuda` | `burn`, `burn-cuda` | native Burn, CubeCL/CUDA kernels | NVIDIA only |
| `tch` | `libtorch`, `burn-tch` | native Burn, LibTorch | CUDA, MPS, Vulkan, CPU |
| `wgpu` | `webgpu`, `burn-wgpu` | native Burn, WebGPU | any Vulkan/Metal/DX12 GPU |
| `auto` *(default)* | | an ONNX export if one is there, else the first available of: LibTorch on a GPU, CubeCL/CUDA, LibTorch on CPU, WebGPU | |

LibTorch on CPU comes **before** WebGPU because it is the answer to a question
that was actually asked: LibTorch linked and reporting no GPU means this host has
none it can reach, whereas WebGPU panics outright when there is no adapter at
all. Slow beats broken. WebGPU is reached only when nothing was probeable, where
it is the safer guess than CubeCL — it runs on non-NVIDIA hardware too.

`--device` takes `auto`, `cpu`, `gpu`, `gpu:N`, `mps` or `vulkan`; `cuda` and
`cuda:N` are accepted spellings of `gpu` and `gpu:N`. Multiple GPUs are
addressed by index. `cuda` is the only backend with no CPU device; LibTorch and
WebGPU both have one.

**Naming a backend or device that is not available is an error with a reason,
never a silent fallback.** Only `auto` substitutes.

`seedvc` is the one engine whose ONNX path is a bundle rather than a single
file — it wants `--onnx <dir>` pointing at the six graphs
`export/export_seedvc.py` writes. `auto` resolves to ONNX Runtime when
`--onnx` is set, and by hardware otherwise. The two halves of that are both
errors rather than fallbacks: `--backend onnx` without `--onnx` is refused for
want of a directory, and `--onnx` beside `--backend cuda|tch|wgpu` is refused
because only ONNX Runtime can read a graph. `--onnx` on its own already selects
it.

Training is always Burn — ONNX Runtime has no training path at all — so a
`train` subcommand takes the same flag minus `onnx`.

### One flag per model, where an engine runs more than one

`--backend` names the runtime for the model an engine is *about*. An engine that
runs several can give each its own flag, and `rvc` is the one that does:
`--content-vec-backend` and `--rmvpe-backend` take exactly the spellings above,
default to whatever `--backend` resolved to, and are independent of it and of
each other — so a `.safetensors` generator with an ONNX F0 estimator, or the
reverse, is a real configuration rather than an accident.
`--content-vec-backend auto` means *inherit*, which is the same as leaving the
flag off.

**The one combination that is refused is an `.onnx` generator beside a Burn
feature model.** ONNX Runtime runs `rvc`'s three models as a single fused
pipeline built from all three graph paths at once, so there is nowhere to put a
feature model built elsewhere — `-m voice.onnx --rmvpe-backend tch` is an error
naming the flag, before anything is downloaded. Point `-m` at a `.safetensors`
generator and the two flags mix freely.

**A runtime and a weight format are not the same choice, and the second follows
from the first.** ONNX Runtime and Burn read different files, so picking a
backend picks which file gets downloaded: `rmvpe.onnx` against `rmvpe.pt`, and a
single ContentVec `.onnx` against a `hubert_base/` **directory** of weights,
config and preprocessor. That is why `rvc download` takes `--backend` too — it
has no model file to inspect, and prefetching the wrong format leaves the first
real run downloading anyway.

The defaults, the exceptions and the escape hatches for a mirror this does not
know about are in
[`crates/rvc-cli/README.md`](../crates/rvc-cli/README.md), which is where a
flag's behaviour belongs.

## Where models are stored

Two kinds of weight are downloaded, and they land in different places because
different things read them.

**Inference assets** — ContentVec, RMVPE, Whisper, the prosody encoder,
cnhubert, `s1*.ckpt`, `s2G*.pth`, Seed-VC's four networks, and the two
`preprocess` fetches (MDX23C for `separate`, CAM++ for `diarize`) — are shared
across runs and projects, so they go to a **cache**, resolved in this order:

1. `--cache-dir`,
2. `RVC_CACHE_DIR` / `STT_CACHE_DIR` / `TTS_CACHE_DIR` / `SEEDVC_CACHE_DIR`, per
   engine,
3. `VOICE_CACHE_DIR`, for all of them at once,
4. `voice` under the XDG cache root — `$XDG_CACHE_HOME` when that names an
   absolute path, otherwise `~/.cache/voice`.

So a machine that sets none of them caches in **`~/.cache/voice`**.

`-h` always tells you where a given machine will put them. **A shared asset never lands in an
output directory**, so pointing two runs at two output folders does not fetch
Whisper twice.

Both weight formats of ContentVec and RMVPE are inference assets, so both land in
that same cache and neither goes to `pretrained/`. Switching backends therefore
adds a download rather than replacing one, and the copies coexist.

Every engine will fetch what it needs on its first run, and each has a
`download` subcommand that does it up front instead — `rvc download`,
`stt download`, `tts download`, `seedvc download`. Each prints the paths it
filled, which are what that engine's own weight flags take, so this is also how a
machine that will be offline later is set up. There is deliberately no top-level
`voice download`: naming the engine is what says whose gigabytes are being spent.

**`preprocess` has no `download`, and that is the same rule rather than a gap.**
A `download` fetches what a default bare invocation would fetch on demand, and
`preprocess` has no bare invocation to have a default — six of its eight stages
run no model at all. `separate` (448 MB) and `diarize` (28 MB) fetch on their
first run like everything else; `--model` on either points at a copy you already
have.

**Training warm-start bases** — RVC's `f0G48k.pth`/`f0D48k.pth` and
GPT-SoVITS's `s2D*.pth` — are shared in the same way, so they go to
`pretrained/` inside that cache, flat under their upstream names so you can drop
in a copy you already have. They are fetched only when a fine-tune wants one;
`--no-pretrained` and `--resume` fetch nothing. `seedvc` has none at all: it is
zero-shot, so nothing it downloads is ever a training input.

**An output directory only ever holds what a run produced** — its weights, its
`checkpoint/` best family, and the dashboard's `train.log`.

There is no `--work-dir`: the working directory is the current directory, as it
is for any other Unix tool.
