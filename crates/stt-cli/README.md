# stt — speech recognition as a Unix filter

Turn a recording into text. **`stt` is Whisper**, running as pure-Rust inference
on either of two interchangeable runtimes — a native [Burn](https://burn.dev)
port on LibTorch, CubeCL/CUDA or WebGPU, or an `optimum`-style export on ONNX
Runtime — behind one command that reads raw PCM on stdin and writes transcripts
on stdout. No Python, at run time or at install time.

**New here?** [Requirements](#requirements) → [Install](#install) →
[Transcribe a file](#1-transcribe-a-file). Building on it?
[For developers](#for-developers).

This page documents the standalone `stt` binary. Every flag below is also
available from the `voice` binary as `voice stt …` — the same code, because
`voice` depends on this crate as a library rather than copying its arguments.
See the [repository README](../../README.md), and
[`rvc`'s README](../rvc-cli/README.md) if you want to pipe transcription into
voice conversion.

## Requirements

| | needed by | how it's found |
|---|---|---|
| **ffmpeg 8.1** | everyone — `audio-kit` is linked in for PCM I/O, and ffmpeg is what puts audio on stdin | system libraries |
| **ONNX Runtime** | only when you actually run `--backend onnx` | dlopened at run time from `ORT_DYLIB_PATH` |
| **LibTorch** | only `--backend tch` | linked at build time; found automatically at run time |

Unlike `rvc`, `stt` needs **neither** machine-learning runtime for its default
path: Whisper is a Burn port that loads Hugging Face weights directly, so a
`--backend cuda` or `--backend wgpu` build runs with nothing installed but
ffmpeg. The two runtimes below are choices, not layers — see
[Two runtimes, one command](#two-runtimes-one-command).

### ONNX Runtime

Needed to run an ONNX Whisper export. It is compiled in by default (the `onnx`
cargo feature) because `--backend auto` should be able to accept an export it is
handed, but the library itself is **dlopened on first use** — a run that never
touches ONNX Runtime never looks for it.

```sh
export ORT_DYLIB_PATH=/absolute/path/to/libonnxruntime.so   # if not installed system-wide
```

Because it is loaded dynamically this is a **run-time** variable: one binary
works against any 1.24-compatible build, CPU or CUDA. The CUDA execution
provider is tried first, then CPU.

### LibTorch (optional)

Only needed for `--backend tch`. Unlike ONNX Runtime, LibTorch is *linked*, so
it must be present when the binary is built. **Version must be 2.9.0** — that is
what the `tch 0.22` bindings are generated against, and a distro PyTorch is
usually too new.

```sh
# LibTorch 2.9.0 ships binary distributions with CUDA 12.6, 12.8 or 13.0 runtimes.
wget https://download.pytorch.org/libtorch/cu126/libtorch-shared-with-deps-2.9.0%2Bcu126.zip
unzip libtorch-shared-with-deps-2.9.0+cu126.zip     # -> ./libtorch
export LIBTORCH=$PWD/libtorch
```

At run time the binary finds LibTorch by itself — **`LD_LIBRARY_PATH` is never
needed**. `crates/stt-cli/build.rs` bakes a `RUNPATH` into the executable, which
searches in order:

1. the `LIBTORCH` it was built against,
2. `libtorch/` in the current working directory,
3. `libtorch/` next to the binary, then `../libtorch/` for `bin/` layouts.

That build script is a deliberate duplicate of `rvc-cli`'s, because an rpath is
a property of one linked executable. Without it a missing `libtorch.so` aborts
inside `ld.so` before `main` runs — on *every* invocation, including
`stt completions`, which never touches a GPU.

## Install

```sh
export LIBTORCH=$PWD/libtorch      # omit to build without the tch backend
cargo build --release -p stt-cli   # -> target/release/stt
```

The four backend features — `cuda`, `tch`, `wgpu`, `onnx` — are all on by
default, so one binary carries every runtime and `--backend` chooses at run
time. Drop the ones you do not want:

```sh
# Burn only: no ONNX Runtime anywhere in the build.
cargo build --release -p stt-cli --no-default-features --features cuda,tch,wgpu

# ONNX Runtime only: no LibTorch to install, no CUDA toolkit to have.
cargo build --release -p stt-cli --no-default-features --features onnx
```

Naming a backend that was not compiled in is an error that says so, and nothing
else changes.

**A debug build runs models at full speed.** The `dev` profile gives
dependencies `opt-level = 3`, which is where every tensor operation lives, while
leaving workspace crates cheap to recompile — transcribing 24 s of audio takes
19 s in debug against 18 s in release. Use `cargo build` for everything except
benchmarking.

## Two runtimes, one command

`stt-core` puts a single trait between the decode loop and whatever is running
the network: token ids in, `f32` logits out
([`crates/stt-core/src/engine.rs`](../stt-core/src/engine.rs)). Encoded audio
and key/value caches stay *inside* the engine, because they are backend-specific
tensors with no useful common type and hoisting them would mean a round trip to
host memory on every generated token.

Two consequences, and both are the point:

- `Transcriber` is **not generic over a compute backend**. Picking one is a
  constructor call, so nothing in this crate names a Burn type and the choice is
  purely run-time.
- **The two runtimes agree token for token on the same weights.** That is the
  best correctness check available for a port: ONNX Runtime is an entirely
  independent implementation of the same graph, so a matching transcript
  exercises the arithmetic, not just the module tree. Weight coverage says the
  names line up; this says the maths does. (This repo has already shipped a port
  that loaded at 100% and produced fluent garbage — see the causal-mask trap in
  [CLAUDE.md](../../CLAUDE.md).)

**How `auto` decides.** It reads the model directory, because the files settle
the question before preference can: an ONNX export cannot run on Burn and
safetensors cannot run on ONNX Runtime.

| what the directory holds | `--backend auto` picks |
|---|---|
| `encoder_model.onnx` + `decoder_model_merged.onnx` (or the same under `onnx/`) | `onnx` |
| `model.safetensors` | the fastest Burn backend available |

"Fastest available" is the shared cascade in `burn-kit`, identical to the one
`rvc` uses: LibTorch on a GPU, else CubeCL/CUDA, else WebGPU, else LibTorch on
CPU. It never picks a backend it can tell will not run, and where it cannot tell
it prefers WebGPU over CubeCL — WebGPU also runs on NVIDIA, while CubeCL fails
on anything else.

The ONNX side consumes the two-graph `optimum` export, where
`decoder_model_merged.onnx` folds the first pass and the cached passes into one
graph selected by a `use_cache_branch` flag. The cross-attention keys and values
are computed on that first pass and then reused verbatim for the whole segment,
which is why the first token of a segment is much slower than the rest.

## Compute backends & devices

| `--backend` | aliases | runtime | devices |
|---|---|---|---|
| `auto` *(default)* | | reads the model directory — see the table above | |
| `onnx` | | ONNX Runtime (`ort`) | CUDA EP, else CPU |
| `cuda` | `burn-cuda` | native Burn, CubeCL/CUDA kernels | NVIDIA only |
| `tch` | `libtorch`, `burn-tch` | native Burn, LibTorch | CUDA, MPS, Vulkan, CPU |
| `wgpu` | `webgpu`, `burn-wgpu` | native Burn, WebGPU | any Vulkan/Metal/DX12 GPU |

`--device` picks *which* device inside the chosen backend: `auto` (default),
`cpu`, `gpu`, `gpu:N`, `mps`, `vulkan`, with `cuda` and `cuda:N` accepted as
spellings of `gpu` and `gpu:N`. `cuda` is the only backend with no CPU device;
LibTorch and WebGPU both have one.

**`--backend onnx` ignores `--device`.** ONNX Runtime chooses its own execution
provider — CUDA if the shared library was built with it, else CPU — and there is
no honest way to map a `--device` onto that from here, so it is not pretended.

Naming a backend or device that is not available is an **error with a reason**,
never a silent fallback; only `auto` substitutes.

```sh
ffmpeg -v quiet -i take.mp3 -f f32le -ar 16000 -ac 1 - | stt --backend tch --device gpu:0
ffmpeg -v quiet -i take.mp3 -f f32le -ar 16000 -ac 1 - | stt --backend wgpu
```

## Usage

`stt` is one tool, so transcription is the **bare invocation** rather than a
subcommand; `completions` is the only subcommand beside it. Logs go to stderr,
so stdout carries nothing but transcripts.

### 1. Transcribe a file

Input is mono **f32le PCM at 16 kHz** — Whisper's analysis rate, and the same
wire format every other engine in the toolkit speaks. ffmpeg does the decoding
and resampling:

```sh
ffmpeg -v quiet -i take.mp3 -f f32le -ar 16000 -ac 1 - | stt
```

There is no `--input` flag on purpose. Anything that can produce PCM is a valid
source — a file, a microphone, another process:

```sh
# straight from a microphone
ffmpeg -v quiet -f alsa -i default -f f32le -ar 16000 -ac 1 - | stt

# every take in a directory, to a transcript beside each one (fish)
for f in corpus/*.wav
  ffmpeg -v quiet -i $f -f f32le -ar 16000 -ac 1 - | stt > (string replace .wav .txt $f)
end
```

Set the log level with `RUST_LOG` (the default is `info,ort=warn`, which mutes
ONNX Runtime's per-tensor allocation chatter). `RUST_LOG=warn` leaves only the
things worth acting on, such as the truncation warning below.

### 2. Output format — `--format text|jsonl`

`--format text` (the default) writes **one line per segment**, nothing else, so
it pipes straight into a translator, a `grep`, or `tts`:

```sh
ffmpeg -v quiet -i take.mp3 -f f32le -ar 16000 -ac 1 - | stt
```
```
今天天气很好
这是一段示例录音
```

`--format jsonl` writes one JSON object per line, adding the segment's position
in the original recording and the language the model used — which is what a
subtitle file, or a TTS training manifest, needs:

```sh
ffmpeg -v quiet -i take.mp3 -f f32le -ar 16000 -ac 1 - | stt --format jsonl
```
```json
{"start":2.300,"end":4.760,"language":"zh","text":"今天天气很好"}
{"start":5.120,"end":7.480,"language":"zh","text":"这是一段示例录音"}
```

`start` and `end` are seconds from the beginning of the input and are the
**slicer's** span, not the model's: Whisper is asked for plain text with
`<|notimestamps|>`, because the segment boundaries are already known from the
audio and a model's guess at them is strictly worse information. Segments whose
transcript comes back empty are dropped rather than emitted as blank lines.

JSON escaping is hand-rolled in `args.rs` rather than pulling `serde` in for one
field — the input is model-generated text, so quotes, backslashes and the
control characters below `0x20` are the whole surface.

### 3. Choosing a model — `--repo` and `--model`

The default is `openai/whisper-large-v3-turbo`, downloaded on first use into the
Hugging Face cache. Four files make a usable repo: `model.safetensors`,
`config.json`, `generation_config.json`, `tokenizer.json`.

```sh
# a different size, or a fine-tune
stt --repo openai/whisper-large-v3   < take.f32le

# the ONNX export of the same model — `--backend auto` will notice and use ORT
stt --repo onnx-community/whisper-large-v3-turbo   < take.f32le

# a local directory, skipping the downloader entirely
stt --model ~/models/whisper-large-v3-turbo   < take.f32le

# somewhere other than the default cache
stt --cache-dir /mnt/big/hf-cache   < take.f32le
```

**Every dimension is read from the repo's own `config.json`** — mel bands,
`d_model`, layer and head counts, vocabulary size — and every control-token id
from its `generation_config.json`. A different model size therefore costs no
code, and neither does the 80-vs-128 mel split that large-v3 introduced: the
log-mel front-end asks the loaded model how many bands it wants rather than
assuming. Hardcoding those ids would be worse than a hardcoded dimension,
because large-v3 added Cantonese and pushed all hundred language tokens up by
one — a stale table decodes fluent nonsense instead of failing.

Preferring the first-party `openai/…` repo to a community re-export is
deliberate for the Burn path: mirrors of converted weights move and disappear.
For the ONNX path a converted repo is exactly what you want, and
`onnx-community/whisper-large-v3-turbo` is the maintained one.

**If a download stalls, sidestep it.** Large Hugging Face fetches have been
observed to hang partway, and the in-process downloader does not finalise a
partial blob — so the next run starts over rather than resuming. Fetching the
repo by other means (`git clone`, `hf download`, a browser) and pointing
`--model` at the resulting directory is a perfectly valid workaround, and it is
also how you use weights that live on a machine with no network at all.

### 4. Segmentation — the part that matters most

**Whisper is a language model conditioned on audio, and that is also its failure
mode.** Handed a long quiet stretch it does not return nothing; it invents
fluent, confident sentences to fill it, and there is no signal in the output
that says so. Feeding it speech-shaped pieces is the standard defence, and it is
what `stt` does before it does anything else.

The slicer is the same one `rvc preprocess` uses
([`crates/audio-kit/src/slice.rs`](../audio-kit/src/slice.rs)), and its central
property is that **energy is used only to locate long silent gaps, never to gate
quiet-but-present sound**. A gap is a cut point only when it is both below the
energy floor *and* longer than `--min-silence`, so short internal pauses and
soft breathy tails stay inside their segment and a sentence is never split down
the middle. That was written for ASMR corpora, where the quietest material is
the material you most want to keep, and recognition inherits it unchanged.

| flag | default | what it does |
|---|---|---|
| `--silence-db` | `-40` | energy floor in dBFS. **Lower** it (e.g. `-50`) when soft speech is being cut into fragments; raise it when room noise is keeping everything glued together |
| `--min-silence` | `0.5` | how long a quiet gap must last to count as a boundary. Raise it if single sentences are being split |
| `--min-clip` | `0.2` | drop segments shorter than this, which are usually a cough or a door |
| `--max-clip` | `30.0` | hard cap on segment length. Must be in `(0, 30]` |

`--max-clip` cannot exceed 30 because **one segment has to fit one encoder
window**, and Whisper's is exactly 30 s wide. Passing more is rejected up front
rather than silently truncated. Runs longer than the cap are split at their
quietest interior frame until every piece fits, which is the least bad place to
cut when there is no real silence to use.

Note that `--min-silence` defaults to `0.5` here against `0.3` for
`rvc preprocess`: training wants many short clean clips, whereas recognition
wants whole sentences, since Whisper's context is what makes it accurate.

```sh
# a soft, close-mic recording: keep the quiet passages, and be slower to cut
ffmpeg -v quiet -i asmr.mp3 -f f32le -ar 16000 -ac 1 - \
  | stt --silence-db -50 --min-silence 0.8 --format jsonl
```

### 5. Language — `--language` and `--translate`

With no `--language`, the language is **detected**: one decoder step against the
encoded audio, reading only the language slots of the logits. That is the same
trick the reference implementation uses and it is far cheaper than decoding
twice, but on a short or breathy clip it is also the least reliable thing here.
Force it when you know the answer:

```sh
stt --language zh   < take.f32le      # ISO code: en, zh, ja, …
stt --translate     < take.f32le      # transcribe into English instead
```

`--translate` is Whisper's own task token, so it is X → English and nothing
else. Any other direction belongs to a translation model, which is why the
toolkit's pipeline diagram has a separate `translate` stage.

### 6. The token cap — `--max-tokens`

`--max-tokens` (default `224`) bounds how far the decoder may run on one
segment. Whisper's own default is 224, half of its 448-wide positional table,
and a cap is what keeps a hallucination loop from generating until the heat
death of the machine.

**Hitting the cap is reported, not passed off as the end of the utterance.** A
dense 30 s segment can legitimately need more tokens than that, so the run logs

```
WARN a segment hit the 224-token cap and was cut off — raise the token cap, or
     split it with a shorter maximum clip length
```

on stderr. Silent truncation is the failure this guards: without the warning the
output is a perfectly well-formed transcript that simply stops mid-sentence, and
nothing about it looks wrong. Raise `--max-tokens`, or lower `--max-clip` so the
segment arrives in two pieces.

### 7. Buffered, not streamed

`stt` reads **all** of stdin before it transcribes anything. Unlike `rvc serve`
this is not a streaming filter, and it cannot be, for two independent reasons:

- Segmentation looks for silences across the whole recording, so it needs the
  whole recording.
- Whisper's log-mel front-end floors each window 8 decades below **that
  window's own peak**. The normalisation is defined against the loudest bin in
  the window, so no frame in a window is final until the window is complete.

`--chunk` (default `16000` samples, one second) is only how many samples are
read from stdin per `read` call — a plumbing knob, not a latency knob. Feeding
`stt` from a live microphone works; you just get the transcript when the stream
closes.

### 8. Composing with the rest of the toolkit

Every stage is a filter over PCM or text, so the pipeline in the
[repository README](../../README.md) is a literal shell pipe:

```sh
# recognise, then re-synthesise in another voice
ffmpeg -v quiet -i take.mp3 -f f32le -ar 16000 -ac 1 - \
  | stt \
  | tts --reference clip.wav --reference-text "这是一段示例录音" --sr 16000 \
  | rvc serve -m models/voice.safetensors --model-sr 48000 \
  > out.f32le
```

Transcribing synthesised audio is also how `tts` is checked end to end: text in
and text out are compared by an independent model, which is about as close to
listening as an automated check gets.

`--format jsonl` is the input a TTS fine-tune wants. `tts train` reads a corpus
as `<stem>.wav` beside `<stem>.txt`, which one line of shell produces:

```sh
for f in corpus/*.wav
  ffmpeg -v quiet -i $f -f f32le -ar 16000 -ac 1 - | stt > (string replace .wav .txt $f)
end
```

### 9. Shell completions

`stt completions <shell>` prints a completion script to stdout — generated from
the actual flags, so it cannot drift from the binary. `bash`, `zsh`, `fish`,
`powershell` or `elvish`:

```sh
stt completions fish > ~/.config/fish/completions/stt.fish
stt completions zsh  > ~/.zfunc/_stt
stt completions bash > /etc/bash_completion.d/stt
```

## For developers

### Building from source

```sh
export ORT_DYLIB_PATH=/usr/lib/libonnxruntime.so
export LIBTORCH=$PWD/libtorch

cargo build -p stt-cli
cargo clippy --workspace --all-targets     # kept clean
cargo fmt

LD_LIBRARY_PATH=$PWD/libtorch/lib cargo test --workspace
```

`LD_LIBRARY_PATH` is a **test-only** requirement: cargo runs each test binary
with its own package directory as the working directory, so the relative search
paths baked into `stt` do not apply to them. The `stt` binary itself never needs
it.

### Checking the network port

`burn-whisper` has no unit tests for the network as a whole. Correctness starts
with loading real published weights and reporting coverage — for
`openai/whisper-large-v3-turbo` the answer must be **587 applied, 0 missing, 0
genuinely unused**, and it must be identical on every backend:

```sh
cargo run -p burn-whisper --example load -- ~/models/whisper-large-v3-turbo/model.safetensors
```

It takes `--backend ndarray|cuda|tch` (default `ndarray`, needs no GPU). A
backend that changes the counts is a bug.

Coverage checks the module *layout*, not the arithmetic. Three unit tests in
[`crates/burn-whisper/src/lib.rs`](../burn-whisper/src/lib.rs) cover what it
cannot, and each one exists because of a bug that got past a 100%-coverage load:

- the stride-2 convolution really halves the frame rate, so 30 s of mel becomes
  the 1500 positions the encoder's positional table is sized for;
- feeding a prompt all at once and feeding it a token at a time produce
  identical logits for the final position — which holds only if the causal mask
  *and* the positional offset are both right under a growing KV cache;
- the causal mask blocks the future and nothing else. Burn's `triu_mask` and
  `tril_mask` are named for the triangle they **keep**, so the obvious-looking
  one reverses time, and in a one-query/one-key decode step it masks the only
  position there is — a full row of `-inf` into softmax is `NaN`, not an error.
  Those tests assert finiteness explicitly, because `assert_approx_eq` compares
  `NaN` against `NaN` without complaint.

### The differential check

The strongest test this crate has costs one command:

```sh
ffmpeg -v quiet -i take.mp3 -f f32le -ar 16000 -ac 1 - > /tmp/take.f32le

stt --repo openai/whisper-large-v3-turbo        --backend tch  < /tmp/take.f32le > /tmp/burn.txt
stt --repo onnx-community/whisper-large-v3-turbo --backend onnx < /tmp/take.f32le > /tmp/ort.txt
diff /tmp/burn.txt /tmp/ort.txt      # must be empty
```

Decoding is greedy, so this is deterministic and the two must agree exactly. Run
it after touching anything in `burn-whisper`, `mel.rs` or `decode.rs`.

### Crate layout

| crate | role |
|-------|------|
| `stt-cli` | the `stt` binary (clap) — also a library, so `voice` hosts `voice stt` from this exact code |
| `stt-core` | the engine: log-mel front-end, BPE vocabulary and control tokens, greedy KV-cached decode, segmentation, and **both** runtimes behind one `Engine` trait |
| `burn-whisper` | the Whisper network alone, a standalone Burn port mirroring Hugging Face's `state_dict` layout; no app dependencies, no compute backend named |
| `burn-kit` | `--device` resolution, the `auto` backend cascade, and safetensors loading — shared with every other engine |
| `audio-kit` | ffmpeg decode/resample, raw-PCM I/O and the sentence slicer, all as `futures::Stream<f32>` |
| `hub-kit` | Hugging Face downloads |
| `cli-kit` | logging, `--device` parsing, shell completions |

**No engine depends on another engine.** `stt` shares no code with `rvc` or
`tts` beyond the `*-kit` tier — the log-mel front-end here and the one
`rvc-core` has for RMVPE share a shape and no numbers (different FFT size,
power rather than magnitude, `log10` rather than natural log, an extra rescale),
so the Slaney helpers are written twice on purpose.

## Status

- **Whisper large-v3-turbo runs end to end**, from PCM on stdin to text on
  stdout, with segmentation, language detection, forced language and
  X→English translation.
- **Two runtimes, one command**: the native Burn port on LibTorch, CubeCL/CUDA
  or WebGPU, and an `optimum` ONNX export on ONNX Runtime. `--backend auto`
  picks from the model directory. The two produce **byte-identical transcripts
  from the same weights**, which is how the port is validated against an
  independent implementation.
- **Model-agnostic within Whisper**: every dimension and every control-token id
  comes from the repo's own JSON, so a different size or a fine-tune is a
  `--repo` away and needs no code.
- **The whole network loads**: 587 / 0 / 0 against
  `openai/whisper-large-v3-turbo`, identically on every backend.
- **Not streaming.** Input is buffered by design; see
  [Buffered, not streamed](#7-buffered-not-streamed).

## Roadmap

- **Word-level timestamps.** The decoder currently runs with
  `<|notimestamps|>`; the timing in `--format jsonl` is the slicer's, which is
  segment-granular.
- **Beam search or temperature fallback.** Decoding is greedy today, which is
  what makes the Burn/ONNX differential check exact — any replacement has to
  keep a deterministic mode for that reason.
- **Batched segments.** Segments are independent, so a batch dimension across
  the encoder is free throughput that is simply not implemented yet.
- **Fine-tuning Whisper.** The Burn port is a complete model, so a training loop
  is a `train-kit` job rather than new network code — the same shape as
  `rvc-train` and `tts-train`.
- **A second model behind the same `Engine` trait**, for languages or latencies
  Whisper is wrong for. The trait is already the whole boundary.
