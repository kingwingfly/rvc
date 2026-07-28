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

## One binary per engine, plus one that has them all

| | what it is | install it if |
|---|---|---|
| **`rvc`** | voice conversion — `convert`, `serve`, `train`, `preprocess` | retimbring recordings is all you need |
| **`stt`** | speech recognition — PCM in, text out | transcription is all you need |
| **`tts`** | speech synthesis — text in, PCM out | you want a voice to read text |
| **`voice`** | the whole toolkit: `voice rvc …`, `voice stt`, `voice tts` | you want them together |

Each engine stands alone and pulls in only what it uses — installing `stt` costs
you no ONNX Runtime and none of the RVC stack. `voice` is an *integration*: it
depends on `rvc-cli`, `stt-cli` and `tts-cli` as libraries, so a flag cannot exist on one
spelling and not the other.

```sh
export ORT_DYLIB_PATH=/usr/lib/libonnxruntime.so   # onnxruntime is never bundled
export LIBTORCH=$PWD/libtorch                      # optional; must be 2.9.0
cargo build --release                              # all three binaries
cargo build --release -p rvc-cli                   # just `rvc`
cargo build --release -p stt-cli                   # just `stt`
cargo build --release -p tts-cli                   # just `tts`
```

A **debug build can run models at full speed** — the `dev` profile optimises
dependencies (`opt-level = 3`), where all the tensor math lives, while leaving
workspace crates cheap to recompile. Use `cargo build` for everything except
benchmarking; `--release` costs minutes of `lto` for a few percent.

Full setup, backend and device documentation lives in the `rvc` tool's README —
the requirements are the same for both binaries.

## Engines

| | status | docs |
|---|---|---|
| **`rvc`** — voice conversion (RVC v2) | **works**, inference + native training | [crates/rvc-cli/README.md](crates/rvc-cli/README.md) |
| **`stt`** — speech recognition (Whisper large-v3-turbo) | **works** — `voice stt` | below |
| **`tts`** — speech synthesis (GPT-SoVITS v2) | **works** — `voice tts` | below |
| **`translate`** | not started; pipe to any external tool meanwhile | — |

`rvc` runs on three interchangeable compute backends — ONNX Runtime, and native
Burn on either LibTorch or CubeCL/CUDA or WebGPU — chosen at run time with
`--backend`. New engines are Burn-only.

## `voice stt`

Raw f32le mono PCM at 16 kHz on stdin, text on stdout:

```sh
ffmpeg -i take.mp3 -f f32le -ar 16000 -ac 1 - | stt      # or: voice stt
```

`--format jsonl` adds per-segment timings and the detected language, which is
what a subtitle file — or a TTS training manifest — needs:

```json
{"start":2.300,"end":4.760,"language":"zh","text":"欺软怕硬的家伙算什么好汉"}
```

Two runtimes, one command. `--backend auto` reads the model directory: an
`optimum`-style ONNX export (`encoder_model.onnx` +
`decoder_model_merged.onnx`) runs on ONNX Runtime, `model.safetensors` runs on
the fastest available Burn backend. Name one explicitly with `--backend
onnx|tch|cuda|wgpu`. The two agree token for token on the same weights, which is
how the Burn port is checked against an independent implementation.

Weights come from `openai/whisper-large-v3-turbo`, downloaded on first use;
`--repo owner/name` picks a different one — `onnx-community/whisper-large-v3-turbo`
for the ONNX export — and every dimension is read from that repo's own
`config.json`, so a different size costs no code.

Input is **buffered, not streamed** — segmentation looks for silences across the
whole recording, and Whisper normalises each 30 s window against its own peak.
Segmentation matters more than its size suggests: Whisper is a language model
conditioned on audio, and handed a long quiet stretch it will invent fluent
sentences to fill it. `voice stt` reuses the sentence slicer from `rvc
preprocess`, which uses energy only to find silent gaps and never to gate
quiet-but-present sound, so soft and breathy speech survives the cut.
`--silence-db` and `--min-silence` tune where the cuts land.

## `voice tts`

Text on stdin, raw f32le mono PCM on stdout. GPT-SoVITS clones a voice from a
few seconds of reference audio:

```sh
echo "今天天气很好" \
  | tts --reference clip.wav --reference-text "不要再欺负他了" \
  | ffplay -f f32le -ar 32000 -ac 1 -
```

`--reference-text` is required and is not bookkeeping. `s1` generates by
*continuation*: it is primed with the reference's phonemes beside the
reference's semantic tokens and then asked to keep going with your text. Given
only the target text it sees phonemes and audio that disagree, finds nothing to
continue, and stops after a token or two.

`--sr` resamples, which is what makes the pipeline this toolkit exists for:

```sh
tts --reference clip.wav --reference-text "…" --sr 16000 < script.txt \
  | rvc serve -m voice.safetensors --model-sr 48000 > out.f32le
```

One line of stdin is one utterance. Weights (cnhubert, `s1`, `s2` and the ONNX
prosody encoder) download on first use.

**Chinese only for now.** `--language en|ja` errors rather than guessing, because
running Japanese through the Chinese front-end produces fluent-sounding wrong
audio. The accuracy ceiling on Chinese is the polyphone dictionary — see
`text-kit`.

### Fine-tuning a voice

Cloning from one reference clip gets the timbre. Fine-tuning adapts the
*delivery* — pacing, emphasis, where a speaker breathes — because those live in
the semantic token sequence `s1` predicts.

A corpus is audio beside transcripts, `<stem>.wav` next to `<stem>.txt`. `stt`
writes the transcripts:

```sh
for f in corpus/*.wav
  ffmpeg -v quiet -i $f -f f32le -ar 16000 -ac 1 - | stt > (string replace .wav .txt $f)
end

tts train corpus/ -o tuned/voice --epochs 10
tts --reference clip.wav --reference-text "…" --s1 tuned/voice.safetensors < script.txt
```

Training is plain next-token cross-entropy, so unlike an adversarial loop the
loss means something on its own: it should fall and keep falling. Two files are
written — `voice.safetensors` (the weight EMA, what you want) and
`voice.raw.safetensors` (the live weights). `s2` fine-tuning is not implemented
yet, so timbre still comes from the reference clip.

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
| `text-kit` | grapheme-to-phoneme for TTS: language splitting, Mandarin g2p, GPT-SoVITS's phoneme table. No model, no tensors |

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
| `stt-core` + `stt-cli` | speech recognition, Burn **or** ONNX Runtime → binary `stt` |
| `tts-core` + `tts-train` + `tts-cli` | speech synthesis → binary `tts` |
| `voice-cli` | the integration → binary `voice` |

Two rules keep it that way. **No engine depends on another engine** — anything
two of them need moves into the plumbing tier first. And **`voice-` is reserved
for the top**: a crate an engine depends on may not be named after the
integration, which is why the shared crates are `*-kit` rather than `voice-*`.

## Contributing

`cargo clippy --workspace` is kept clean and `cargo fmt` is enforced. See
[CLAUDE.md](CLAUDE.md) for the architectural constraints that are easy to break
silently — weight compatibility, backend genericity, and the rules above.
