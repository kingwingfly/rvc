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
directly; training is native Rust too. The one remaining exception, the optional
`.safetensors → ONNX` exporter under [`export/`](export/README.md), is on its way
out.

## Two binaries

| | what it is | install it if |
|---|---|---|
| **`rvc`** | voice conversion on its own — `convert`, `serve`, `train`, `preprocess` | retimbring recordings is all you need |
| **`voice`** | the whole toolkit; hosts the above as `voice rvc …` | you want recognition and synthesis too |

They share one implementation — `voice-cli` depends on `rvc-cli` as a library,
so a flag cannot exist on one and not the other.

```sh
export ORT_DYLIB_PATH=/usr/lib/libonnxruntime.so   # onnxruntime is never bundled
export LIBTORCH=$PWD/libtorch                      # optional; must be 2.9.0
cargo build --release                              # both binaries
cargo build --release -p rvc-cli                   # just `rvc`
```

Full setup, backend and device documentation lives in the `rvc` tool's README —
the requirements are the same for both binaries.

## Engines

| | status | docs |
|---|---|---|
| **`rvc`** — voice conversion (RVC v2) | **works**, inference + native training | [crates/rvc-cli/README.md](crates/rvc-cli/README.md) |
| **`stt`** — speech recognition (Whisper large-v3-turbo) | **works** — `voice stt` | below |
| **`tts`** — speech synthesis (GPT-SoVITS v2) | planned, inference + fine-tuning | — |
| **`translate`** | not started; pipe to any external tool meanwhile | — |

`rvc` runs on three interchangeable compute backends — ONNX Runtime, and native
Burn on either LibTorch or CubeCL/CUDA or WebGPU — chosen at run time with
`--backend`. New engines are Burn-only.

## `voice stt`

Raw f32le mono PCM at 16 kHz on stdin, text on stdout:

```sh
ffmpeg -i take.mp3 -f f32le -ar 16000 -ac 1 - | voice stt
```

`--format jsonl` adds per-segment timings and the detected language, which is
what a subtitle file — or a TTS training manifest — needs:

```json
{"start":2.300,"end":4.760,"language":"zh","text":"欺软怕硬的家伙算什么好汉"}
```

Weights come from `openai/whisper-large-v3-turbo`, downloaded on first use;
`--repo owner/name` picks a different one and every dimension is read from that
repo's own `config.json`, so a different size costs no code.

Input is **buffered, not streamed** — segmentation looks for silences across the
whole recording, and Whisper normalises each 30 s window against its own peak.
Segmentation matters more than its size suggests: Whisper is a language model
conditioned on audio, and handed a long quiet stretch it will invent fluent
sentences to fill it. `voice stt` reuses the sentence slicer from `rvc
preprocess`, which uses energy only to find silent gaps and never to gate
quiet-but-present sound, so soft and breathy speech survives the cut.
`--silence-db` and `--min-silence` tune where the cuts land.

## Crate layout

| crate | role |
|-------|------|
| `burn-kit` | Burn plumbing tied to no model: device selection, checkpoint loading |
| `voice-audio` | ffmpeg decode/resample + WAV/raw-PCM I/O, all as `futures::Stream<f32>` |
| `voice-hub` | auto-download model assets from Hugging Face |
| `rvc-core` | the voice-conversion pipeline: feature extraction, DSP, streaming `Converter`, all three generator backends |
| `burn-rvc` | the RVC v2 network itself (standalone Burn port); no app deps |
| `burn-whisper` | the Whisper network (standalone Burn port); loads HF safetensors unchanged |
| `voice-stt` | speech recognition: log-mel front-end, BPE vocabulary, greedy decode, segmentation |
| `rvc-train` | native Rust/Burn adversarial training — see [ARCHITECTURE.md](crates/rvc-train/ARCHITECTURE.md) |
| `rvc-cli` | the `rvc` binary, and the library behind `voice rvc …` |
| `voice-cli` | the `voice` binary |

Two conventions hold: network crates are named after the **model** (`burn-rvc`),
app crates after the **job** (`rvc-core`). And **no engine depends on another
engine** — anything two of them need moves to a neutral crate first.

## Contributing

`cargo clippy --workspace` is kept clean and `cargo fmt` is enforced. See
[CLAUDE.md](CLAUDE.md) for the architectural constraints that are easy to break
silently — weight compatibility, backend genericity, and the rules above.
