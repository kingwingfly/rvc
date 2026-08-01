# voice — a Python-free speech toolkit in Rust

Recognition, synthesis and voice conversion, each one a Unix filter, so they
compose in a pipe:

```
voice -(stt)-> text -(translate)-> text -(tts)-> voice -(rvc)-> voice
```

Every engine reads raw mono `f32le` PCM or plain text on **stdin** and writes the
same on **stdout**, with logs on stderr — so any subset is a valid pipeline, and
anything else that speaks text or PCM drops into the middle of one.

**No Python.** Not to run it, not to install it, not to develop it. Models are
ported to [Burn](https://burn.dev) and load their original Hugging Face weights
directly; training is native Rust too. One exception stays on purpose: the
`.safetensors → ONNX` exporter under [`export/`](export/README.md), which is a
maintainer's tool that nothing you run, install or train invokes. The reasoning
is in [CLAUDE.md](CLAUDE.md).

## One binary per engine, plus one that has them all

**Running a binary with no subcommand *is* the filter.** Subcommands are for
everything that is not streaming — training, preprocessing, batch files, shell
completions — and every engine spells them the same way.

| | the bare invocation | subcommands |
|---|---|---|
| **`rvc`** | PCM in → retimbred PCM out | `convert`, `train`, `preprocess`, `models`, `completions` |
| **`stt`** | PCM in → text out | `completions` |
| **`tts`** | text in → PCM out | `train`, `completions` |
| **`voice`** | — | `voice rvc …`, `voice stt …`, `voice tts …`, all three unchanged |

Each engine stands alone and pulls in only what it uses — installing `stt` costs
you none of the RVC stack. `voice` is an *integration*: it depends on `rvc-cli`,
`stt-cli` and `tts-cli` as libraries, so a flag cannot exist on one spelling and
not the other, and `voice tts train` is the same code as `tts train`.

## Install

```sh
export LIBTORCH=$PWD/libtorch      # 2.9.0; omit to build without the tch backend
cargo build --release              # all four binaries
cargo build --release -p stt-cli   # or just one
```

ONNX Runtime, LibTorch and ffmpeg are found the same way by all four binaries,
as are `--backend`, `--device` and the model cache. That is written once:

**→ [`docs/setup.md`](docs/setup.md)**

## Engines

| | status | docs |
|---|---|---|
| **`rvc`** — voice conversion (RVC v2) | **works**, inference + native training | [crates/rvc-cli/README.md](crates/rvc-cli/README.md) |
| **`stt`** — speech recognition (Whisper large-v3-turbo) | **works**, Burn **or** ONNX Runtime | [crates/stt-cli/README.md](crates/stt-cli/README.md) |
| **`tts`** — speech synthesis (GPT-SoVITS v2) | **works**, inference + `s1` and `s2` fine-tuning | [crates/tts-cli/README.md](crates/tts-cli/README.md) |
| **`translate`** | not started; pipe to any external tool meanwhile | — |

Each engine's README has the rest — every flag, which weights are fetched and
from where, and its fine-tuning loop.

**`--backend` is a run-time choice on every engine, and ONNX Runtime is a
first-class target rather than a legacy one**: all three engines run under ONNX
Runtime or under native Burn on LibTorch, CubeCL/CUDA or WebGPU, in one binary,
with one set of spellings. Fine-tuning is always Burn — ONNX Runtime has no
training path at all, which is why [`export/`](export/README.md) exists to carry
a trained model back out to it.

## Documentation

Usage lives with the tool and this file stays an index. Beyond the three engine
READMEs and [`docs/setup.md`](docs/setup.md):

| | |
|---|---|
| [`docs/rvc-architecture.typ`](docs/rvc-architecture.typ) | RVC v2 — the VITS-derived synthesizer, the NSF source module, the adversarial objective |
| [`docs/gptsovits-architecture.typ`](docs/gptsovits-architecture.typ) | GPT-SoVITS v2 — cnhubert, the semantic quantiser, `s1` (text → tokens) and `s2` (tokens → waveform) |
| [`docs/whisper-architecture.typ`](docs/whisper-architecture.typ) | Whisper large-v3-turbo — the log-mel front-end, the encoder/decoder stack, KV-cached greedy decoding |
| [`docs/training.md`](docs/training.md) | every training loop in the toolkit: the shared objective, `train-kit`, warm-start, devices |
| [CLAUDE.md](CLAUDE.md) | why each decision was made and which trap it avoids — read before changing anything |

The architecture papers are written for *reviewing and maintaining a port*
rather than for driving the CLI. The convention is a [Typst](https://typst.app)
source with its rendered PDF committed beside it, so reading one needs no
toolchain; rebuild with `typst compile docs/<name>.typ` and commit both.

## Crate layout

Three tiers, and the names say which is which.

**Shared plumbing** — no model, no engine, safe for anything to depend on:

| crate | role |
|-------|------|
| `burn-kit` | device selection and checkpoint loading |
| `audio-kit` | ffmpeg decode/resample, WAV/raw-PCM I/O, the sentence slicer — all `futures::Stream<f32>` |
| `hub-kit` | model downloads from Hugging Face, and where they are cached |
| `cli-kit` | logging, shell completions, `--backend` and `--device` parsing |
| `train-kit` | checkpoints, weight EMA, gradient accumulation, the live dashboard — generic over the module being trained |
| `text-kit` | grapheme-to-phoneme for TTS: language splitting, Mandarin and English g2p, GPT-SoVITS's phoneme table. No model, no tensors |

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
| `rvc-core` + `rvc-train` + `rvc-cli` | voice conversion → binary `rvc` |
| `stt-core` + `stt-cli` | speech recognition → binary `stt` |
| `tts-core` + `tts-train` + `tts-cli` | speech synthesis → binary `tts` |
| `voice-cli` | the integration → binary `voice` |

Two rules keep it that way. **No engine depends on another engine** — anything
two of them need moves into the plumbing tier first. And **`voice-` is reserved
for the top**: a crate an engine depends on may not be named after the
integration, which is why the shared crates are `*-kit` rather than `voice-*`.

## Contributing

`cargo clippy --workspace` is kept clean and `cargo fmt` is enforced. Read
[CLAUDE.md](CLAUDE.md) first — it holds the constraints that are easy to break
silently, weight compatibility and backend genericity among them.
