# Setup — shared by `rvc`, `stt`, `tts` and `voice`

All four binaries find their dependencies the same way, take the same
`--backend` and `--device` spellings, and cache downloaded models in the same
place. That is written here once; each engine's README links to this page rather
than repeating it.

- [Requirements](#requirements)
- [ffmpeg](#ffmpeg)
- [ONNX Runtime](#onnx-runtime)
- [LibTorch](#libtorch)
- [Building](#building)
- [Backends and devices](#backends-and-devices)
- [Where models are stored](#where-models-are-stored)

## Requirements

| | needed by | how it is found |
|---|---|---|
| **ffmpeg 8.1** | everyone — decode, resample and PCM I/O | system libraries, linked at build time |
| **ONNX Runtime** | any `--backend onnx` path, plus `rvc`'s feature extraction and `tts`'s prosody encoder | dlopened on first use from `ORT_DYLIB_PATH` |
| **LibTorch 2.9.0** | optional — only `--backend tch` | linked at build time; found automatically at run time |

**Neither machine-learning runtime is bundled or downloaded.** You point at your
own, which is deliberate: a runtime that a distribution already ships should not
be duplicated per binary, and pinning one would make the choice of deployment
target ours rather than yours.

## ffmpeg

The **8.1** development libraries, from your package manager. They are linked,
not dlopened, so they must be present at build time. The `ffmpeg` binary itself
is only needed for the realtime examples, where it is what captures from a
microphone and plays to speakers on either end of a pipe.

## ONNX Runtime

Loaded **dynamically on first use**, so this is a *run-time* variable and one
binary works against any 1.24-compatible build, CPU or CUDA. A run that never
touches ONNX Runtime never looks for it.

```sh
export ORT_DYLIB_PATH=/absolute/path/to/libonnxruntime.so   # if not installed system-wide
```

The CUDA execution provider is tried first, then CPU.

How much it matters depends on the engine: `rvc` runs ContentVec and RMVPE
through it on *every* path, so it is required there; `stt` needs it only for
`--backend onnx`; for `tts` its absence downgrades the prosody encoder to zeros
with a warning rather than failing. Each engine's README says which applies.

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

At run time each binary finds LibTorch by itself — **`LD_LIBRARY_PATH` is never
needed.** Every `*-cli` crate's `build.rs` bakes `$LIBTORCH/lib` into its
executable as a `RUNPATH`; the loader then searches, in order:

1. the `LIBTORCH` (or `LIBTORCH_LIB`) the binary was built against, if that
   directory still exists,
2. `libtorch/` in the current working directory,
3. `libtorch/` next to the binary, then `../libtorch/` for a `bin/` layout,
4. whatever `ld.so.cache` already knows about.

So keeping a `./libtorch` in your project directory works even after the binary
moves. Without the `RUNPATH` a missing `libtorch.so` aborts inside `ld.so`
*before `main` runs* — on every invocation, including ones that never touch a
GPU, such as `completions`.

`LD_LIBRARY_PATH` **is** needed for `cargo test`: cargo runs each test binary
with its own package directory as the working directory, so the relative search
path baked into the installed binaries does not apply.

```sh
LD_LIBRARY_PATH=$PWD/libtorch/lib cargo test --workspace
```

## Building

```sh
export ORT_DYLIB_PATH=/usr/lib/libonnxruntime.so
export LIBTORCH=$PWD/libtorch        # omit to build without the tch backend

cargo build --release                # all four binaries
cargo build --release -p rvc-cli     # just `rvc`
cargo build --release -p stt-cli     # just `stt`
cargo build --release -p tts-cli     # just `tts`
cargo build --release -p voice-cli   # just `voice`
```

The backend features are all on by default, so one binary carries every runtime
and `--backend` chooses at run time. Drop the ones you do not want:

```sh
# No LibTorch to install:
cargo build --release -p rvc-cli --no-default-features --features cuda,wgpu

# Burn only — no ONNX Runtime anywhere in the build:
cargo build --release -p stt-cli --no-default-features --features cuda,tch,wgpu
```

`stt-cli` and `tts-cli` have `cuda`, `tch`, `wgpu` and `onnx`; `rvc-cli` has the
first three only, because ONNX Runtime is not optional there — ContentVec and
RMVPE run on it on every path, native backends included.

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
| `auto` *(default)* | | an ONNX export if one is there, else the first available of: LibTorch on a GPU, CubeCL/CUDA, WebGPU, LibTorch on CPU | |

`--device` takes `auto`, `cpu`, `gpu`, `gpu:N`, `mps` or `vulkan`; `cuda` and
`cuda:N` are accepted spellings of `gpu` and `gpu:N`. Multiple GPUs are
addressed by index. `cuda` is the only backend with no CPU device; LibTorch and
WebGPU both have one.

**Naming a backend or device that is not available is an error with a reason,
never a silent fallback.** Only `auto` substitutes.

Training is always Burn — ONNX Runtime has no training path at all — so a
`train` subcommand takes the same flag minus `onnx`.

## Where models are stored

Two kinds of weight are downloaded, and they land in different places because
different things read them.

**Inference assets** — ContentVec, RMVPE, Whisper, the prosody encoder,
cnhubert, `s1*.ckpt` and `s2G*.pth` — are shared across runs and projects, so
they go to a **cache**, resolved in this order:

1. `--cache-dir`,
2. `RVC_CACHE_DIR` / `STT_CACHE_DIR` / `TTS_CACHE_DIR`, per engine,
3. `VOICE_CACHE_DIR`, for all of them at once,
4. `$XDG_CACHE_HOME/voice`,
5. `~/.cache/voice`.

The resolved path is the printed default of `--cache-dir`, so `-h` always tells
you where a given machine will put them. **A shared asset never lands in an
output directory**, so pointing two runs at two output folders does not fetch
Whisper twice.

**Training warm-start bases** — RVC's `f0G48k.pth`/`f0D48k.pth` and
GPT-SoVITS's `s2D*.pth` — go to `pretrained/` *inside the run's output
directory* instead, because they belong to that experiment: they are read once
by one training run, and keeping them beside the checkpoints they produced is
what makes a run reproducible after the fact. `--no-pretrained` and `--resume`
fetch nothing.

There is no `--work-dir`: the working directory is the current directory, as it
is for any other Unix tool.
