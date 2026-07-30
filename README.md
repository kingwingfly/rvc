# voice — a Python-free speech toolkit in Rust

Recognition, synthesis and voice conversion, each one a Unix filter, so they
compose in a pipe:

```
voice -(stt)-> text -(translate)-> text -(tts)-> voice -(rvc)-> voice
```

Every stage reads raw mono `f32le` PCM or plain text on **stdin** and writes the
same on **stdout**, with logs on stderr — so any subset is a valid pipeline, and
anything else that speaks text or PCM drops into the middle of one.

**No Python.** Not to run it, not to install it, not to develop it. Models are
ported to [Burn](https://burn.dev) and load their original Hugging Face weights
directly; training is native Rust too. One exception stays on purpose: the
`.safetensors → ONNX` exporter under [`export/`](export/README.md). Burn can
*import* an ONNX graph but not emit one, so that script is the only route from a
model fine-tuned here to ONNX Runtime — and ONNX is a deployment target this
toolkit supports, not debt it is paying down. Nothing you run, install or train
touches it.

## One binary per engine, plus one that has them all

| | what it is | install it if |
|---|---|---|
| **`rvc`** | voice conversion — `convert`, `serve`, `train`, `preprocess` | retimbring recordings is all you need |
| **`stt`** | speech recognition — PCM in, text out | transcription is all you need |
| **`tts`** | speech synthesis — text in, PCM out | you want a voice to read text |
| **`voice`** | the whole toolkit: `voice rvc …`, `voice stt`, `voice tts` | you want them together |

Each engine stands alone and pulls in only what it uses — installing `stt` costs
you none of the RVC stack, and `--no-default-features` drops ONNX Runtime from it
too if you only ever want Burn. `voice` is an *integration*: it
depends on `rvc-cli`, `stt-cli` and `tts-cli` as libraries, so a flag cannot exist on one
spelling and not the other.

```sh
export ORT_DYLIB_PATH=/usr/lib/libonnxruntime.so   # onnxruntime is never bundled
export LIBTORCH=$PWD/libtorch                      # optional; must be 2.9.0
cargo build --release                              # all four binaries
cargo build --release -p rvc-cli                   # just `rvc`
cargo build --release -p stt-cli                   # just `stt`
cargo build --release -p tts-cli                   # just `tts`
```

A **debug build can run models at full speed** — the `dev` profile optimises
dependencies (`opt-level = 3`), where all the tensor math lives, while leaving
workspace crates cheap to recompile. Use `cargo build` for everything except
benchmarking; `--release` costs minutes of `lto` for a few percent.

Full setup, backend and device documentation lives in [the `rvc` tool's
README](crates/rvc-cli/README.md) — ONNX Runtime, LibTorch and ffmpeg are found
the same way for all four binaries, so it is written once there and the other
engine READMEs point at it.

## Engines

| | status | docs |
|---|---|---|
| **`rvc`** — voice conversion (RVC v2) | **works**, inference + native training | [crates/rvc-cli/README.md](crates/rvc-cli/README.md) |
| **`stt`** — speech recognition (Whisper large-v3-turbo) | **works**, Burn **or** ONNX Runtime | [crates/stt-cli/README.md](crates/stt-cli/README.md) |
| **`tts`** — speech synthesis (GPT-SoVITS v2) | **works**, inference + `s1` and `s2` fine-tuning | [crates/tts-cli/README.md](crates/tts-cli/README.md) |
| **`translate`** | not started; pipe to any external tool meanwhile | — |

**`stt`** reads raw f32le mono PCM at 16 kHz and writes text; `--format jsonl`
adds per-segment timings and the detected language, which is what a subtitle file
— or a TTS training manifest — needs.

```sh
ffmpeg -i take.mp3 -f f32le -ar 16000 -ac 1 - | stt      # or: voice stt
```

**`tts`** reads text and writes raw f32le mono PCM, cloning a voice from a few
seconds of reference audio. The reference's *transcript* is required beside its
audio, because `s1` generates by continuation and has to see what the reference
says:

```sh
echo "今天天气很好" \
  | tts --reference clip.wav --reference-text "这是一段示例录音" \
  | ffplay -f f32le -ar 32000 -ac 1 -
```

Each engine's README has the rest — every flag, which weights are fetched and
from where, the backends it accepts, and its fine-tuning loop.

### Backends

**`--backend` is a run-time choice on every engine, and ONNX Runtime is a
first-class target rather than a legacy one.** `rvc` and `stt` each run under
ONNX Runtime or under native Burn on LibTorch, CubeCL/CUDA or WebGPU; `tts` runs
on Burn today and is gaining ONNX Runtime graphs one model at a time.

Fine-tuning is always Burn. ONNX Runtime has no training path at all, which is
why any model this toolkit trains must be a Burn port whatever else it also runs
on — and why [`export/`](export/README.md) exists to carry the result back out to
ONNX.

The spellings are not quite uniform yet: `rvc` accepts `burn` as an alias for
`cuda`, `stt` wants `burn-cuda`. Each engine's README lists what it takes.

## Documentation

Usage lives with the tool and this file stays an index — one README per engine,
and `voice` inherits all three: [`rvc`](crates/rvc-cli/README.md),
[`stt`](crates/stt-cli/README.md), [`tts`](crates/tts-cli/README.md).

Under [`docs/`](docs/) there is one long-form architecture paper per network.
They are written for *reviewing and maintaining the port* — what each block
computes, what every loss term is for, why the training loop is shaped the way it
is — rather than for driving the CLI:

| paper | covers |
|---|---|
| [`docs/rvc-architecture.typ`](docs/rvc-architecture.typ) | RVC v2 — the VITS-derived synthesizer, the NSF source module, and the adversarial training objective |
| [`docs/gptsovits-architecture.typ`](docs/gptsovits-architecture.typ) | GPT-SoVITS v2 — cnhubert, the semantic quantiser, `s1` (text → semantic tokens) and `s2` (tokens → waveform) |
| [`docs/whisper-architecture.typ`](docs/whisper-architecture.typ) | Whisper large-v3-turbo — the log-mel front-end, the encoder/decoder stack, and KV-cached greedy decoding |

The convention is a [Typst](https://typst.app) source with its rendered PDF
committed beside it, so reading one needs no toolchain. Rebuild the PDF after an
edit and commit it with the source:

```sh
typst compile docs/rvc-architecture.typ
```

Notes that only matter if you are *changing* the code — weight-compatibility
constraints, backend genericity, and the traps that each cost a day to find —
are in [CLAUDE.md](CLAUDE.md) and
[`crates/rvc-train/ARCHITECTURE.md`](crates/rvc-train/ARCHITECTURE.md).

## Crate layout

Three tiers, and the names say which is which.

**Shared plumbing** — no model, no engine, safe for anything to depend on:

| crate | role |
|-------|------|
| `burn-kit` | device selection and checkpoint loading |
| `audio-kit` | ffmpeg decode/resample, WAV/raw-PCM I/O, the sentence slicer — all `futures::Stream<f32>` |
| `hub-kit` | model downloads from Hugging Face |
| `cli-kit` | logging, shell completions, `--device` parsing |
| `train-kit` | checkpoints, weight EMA, gradient accumulation, the live dashboard — generic over the module being trained |
| `text-kit` | grapheme-to-phoneme for TTS: language splitting by script, Mandarin g2p with a phrase dictionary, English g2p, GPT-SoVITS's phoneme table. No model, no tensors |

**Networks**, named after the model, holding no app dependencies:

| crate | role |
|-------|------|
| `burn-vits` | the VITS blocks RVC and GPT-SoVITS share — attention, WaveNet, flow, posterior encoder, ResBlock1, discriminators — plus the family's losses and its differentiable STFT |
| `burn-rvc` | what is RVC's alone: the NSF source module, the 768-dim content encoder, the synthesizer wiring |
| `burn-whisper` | the Whisper network; loads HF safetensors unchanged |
| `burn-gptsovits` | the GPT-SoVITS network: cnhubert, the quantiser, `s2` (SoVITS) and `s1` (T2S) |

**Engines**, one `-core` and one `-cli` apiece, all the same shape:

| crate | role |
|-------|------|
| `rvc-core` + `rvc-train` + `rvc-cli` | voice conversion, Burn **or** ONNX Runtime → binary `rvc` |
| `stt-core` + `stt-cli` | speech recognition, Burn **or** ONNX Runtime → binary `stt` |
| `tts-core` + `tts-train` + `tts-cli` | speech synthesis, Burn (ONNX Runtime landing model by model) → binary `tts` |
| `voice-cli` | the integration → binary `voice` |

Two rules keep it that way. **No engine depends on another engine** — anything
two of them need moves into the plumbing tier first. And **`voice-` is reserved
for the top**: a crate an engine depends on may not be named after the
integration, which is why the shared crates are `*-kit` rather than `voice-*`.

## Contributing

`cargo clippy --workspace` is kept clean and `cargo fmt` is enforced. See
[CLAUDE.md](CLAUDE.md) for the architectural constraints that are easy to break
silently — weight compatibility, backend genericity, and the rules above.
