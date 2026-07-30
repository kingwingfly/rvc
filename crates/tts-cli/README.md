# tts — GPT-SoVITS speech synthesis

Text on stdin, raw f32le mono PCM on stdout. **`tts` clones a voice from a few
seconds of reference audio** and speaks your text in it — pure-Rust inference
(GPT-SoVITS v2 ported to Burn), a Unix filter by design, and a native Rust/Burn
fine-tuning loop when a few seconds of prompt is not enough. No Python anywhere
in the path.

**New here?** [Requirements](#requirements) → [Install](#install) →
[Speak a line](#speak-a-line). Adapting a particular voice?
[Fine-tune a voice](#4-fine-tune-a-voice-tts-train). Building on it?
[For developers](#for-developers).

This page documents the standalone `tts` binary. The same synthesis is available
from the `voice` binary as `voice tts` and `voice tts-train` — same code, same
flags — if you want recognition and voice conversion alongside it. See the
[repository README](../../README.md) for the toolkit as a whole, and
[`rvc-cli`'s README](../rvc-cli/README.md) for the conversion engine `tts` is
usually piped into.

## Requirements

| | needed by | how it's found |
|---|---|---|
| **ffmpeg 8.1** | everyone — the reference clip is decoded with it | system libraries |
| **ONNX Runtime** | the prosody encoder only; **its absence is a warning, not a failure** | dlopened at run time from `ORT_DYLIB_PATH` |
| **LibTorch** | optional — only `--backend tch` | linked at build time; found automatically at run time |
| a GPU | strongly recommended — `s1` runs one autoregressive step per 40 ms of speech | `--device`, default `auto` |

Neither runtime is bundled or downloaded. Point at your own:

```sh
export ORT_DYLIB_PATH=/absolute/path/to/libonnxruntime.so   # required if not installed system-wide
```

ONNX Runtime is loaded dynamically, so this is a **run-time** variable — the same
binary works against any 1.24-compatible build, CPU or CUDA. The CUDA execution
provider is tried first, then CPU.

**Today the prosody encoder is the only ONNX model in this path**, because it is
the only one that is frozen rather than fine-tuned. `s1` and `s2` are Burn ports
so that they can be trained here — an ONNX Runtime path for deploying them
frozen is on the [roadmap](#roadmap), and Burn stays available for fine-tuning
whatever inference backend you pick. If the encoder cannot be found or cannot be loaded, `tts` logs
a warning and synthesises with zero prosody features — **that costs
expressiveness, not intelligibility**. The speech is still correct; it is flatter
on Chinese, where tone contour is what the encoder mostly contributes. See
[`tts-core/src/prosody.rs`](../tts-core/src/prosody.rs) for why that model is
ONNX and for the `ProsodyEncoder` trait that leaves the slot open.

### LibTorch (optional)

Only needed for `--backend tch`, which is what `auto` prefers when a GPU is
present (it is ~9× faster than the CubeCL/CUDA backend on `rvc`'s generator; the
gap has not been measured on `s1`). Skip it and everything else still works.

Unlike ONNX Runtime, LibTorch is *linked*, so it must be present when the binary
is built. **Version must be 2.9.0** — that is what the `tch 0.22` bindings are
generated against, and a distro PyTorch is usually too new.

```sh
wget https://download.pytorch.org/libtorch/cu126/libtorch-shared-with-deps-2.9.0%2Bcu126.zip
unzip libtorch-shared-with-deps-2.9.0+cu126.zip     # -> ./libtorch
export LIBTORCH=$PWD/libtorch
```

At run time the binary finds LibTorch by itself — **`LD_LIBRARY_PATH` is never
needed**. `crates/tts-cli/build.rs` bakes `$LIBTORCH/lib` into the executable as
a `RUNPATH`, and the loader then searches, in order: the `LIBTORCH` it was built
against, `libtorch/` next to the binary, `libtorch/` in the current working
directory. Keeping a `./libtorch` in your project directory is enough even after
moving the binary.

## Install

```sh
export LIBTORCH=$PWD/libtorch      # omit to build without the tch backend
cargo build --release -p tts-cli   # -> target/release/tts
```

Without LibTorch:

```sh
cargo build --release -p tts-cli --no-default-features --features cuda,wgpu,onnx
```

`--backend tch` then reports that it wasn't compiled in, and everything else is
unchanged. Dropping `onnx` as well builds a binary with no ONNX Runtime
dependency at all — it synthesises with zero prosody features, which is the same
degradation described above, decided at build time instead of at run time.

`cargo build` (dev, no `--release`) is a reasonable default for *running* the
model too: the dev profile gives dependencies `opt-level = 3`, and that is where
every tensor op lives. Reach for `--release` to benchmark, not to test.

## Compute backends & devices

One `tts` binary carries every Burn compute backend it was built with;
`--backend` chooses at run time, and `tts train` takes the same flag with the
same meanings.

| `--backend` | aliases | runtime | devices |
|---|---|---|---|
| `cuda` | `burn-cuda` | native Burn, CubeCL/CUDA kernels | NVIDIA only |
| `tch` | `libtorch`, `burn-tch` | native Burn, LibTorch | CUDA, MPS, Vulkan, CPU |
| `wgpu` | `webgpu`, `burn-wgpu` | native Burn, WebGPU | any Vulkan/Metal/DX12 GPU |
| `auto` *(default)* | | the first of: LibTorch on a GPU, CubeCL/CUDA, WebGPU, LibTorch on CPU | |

`--device` picks *which* device inside the chosen backend: `auto` (default),
`cpu`, `gpu`, `gpu:N`, `mps`, `vulkan` (`cuda`/`cuda:N` are accepted spellings of
`gpu`). `cuda` is the only backend with no CPU device; LibTorch and WebGPU both
have one.

`auto` resolves through the same `burn_kit::auto_backend` every other subcommand
of every binary uses, so `tts`, `stt` and `rvc` cannot disagree about what `auto`
means. Naming a backend or device that isn't available is an **error with a
reason**, never a silent fallback; only `auto` substitutes.

`wgpu` needs no vendor toolkit and runs on AMD, Intel and Apple GPUs, so it is
the portable fallback where neither CUDA nor LibTorch is available.

```sh
tts --reference clip.wav --reference-text "这是一段示例录音" \
  --backend tch --device gpu:0 < script.txt > out.f32le
```

There is no `--backend onnx` for synthesis **today**: `s1` and `s2` are the two
networks this toolkit fine-tunes, so they load and run as Burn modules. An ONNX
Runtime path for frozen inference is being added — see
[Roadmap](#roadmap) — and Burn stays the fine-tuning backend either way.

**Where the time goes.** `s2` is one pass; `s1` is autoregressive at 25 tokens
per second of speech, so a ten-second line is 250 sequential decoder steps, each
one launch-bound rather than arithmetic-bound. That is why the GPU matters and
why `--max-tokens` is a useful bound rather than a formality.

## How it works: two stages and a 25 Hz boundary

GPT-SoVITS is two models with one interface between them, and knowing which is
which explains almost every flag on this page.

**`s1` (T2S) predicts *delivery*.** It is an autoregressive transformer that
reads phonemes and emits **semantic tokens**: pacing, emphasis, phrasing, where a
speaker breathes. It knows nothing about timbre.

**`s2` (SoVITS) renders *timbre*.** It is a VITS descendant that turns those
tokens back into a waveform, conditioned on a speaker vector taken from the
reference clip's spectrogram. It knows nothing about text beyond the phonemes it
is handed alongside the tokens.

**The boundary between them is a 25 Hz sequence of ids over a 1024-entry
codebook**, and that rate comes from exactly one stride: cnhubert produces
features at 50 Hz, and the quantiser's strided convolution halves it
([`burn-gptsovits/src/quantizer.rs`](../burn-gptsovits/src/quantizer.rs)).
Everything downstream follows from that one number — tokens per second, `s1`
sequence lengths, and what `--max-tokens 1500` means (a minute of speech).

```text
  reference audio ─┬─ cnhubert ─ quantiser ─── prompt semantic tokens ─┐
                   └─ spectrogram ─ ref_enc ── speaker vector ───────┐ │
  text ─ text-kit ─── phonemes ──────────────────────────────────┐   │ │
                 └─── BERT ─────── prosody ────────────────────┐ │   │ │
                                                               ▼ ▼   ▼ ▼
                                                     s1 ── semantic tokens
                                                                   │
                                                       s2 ─────────┴── audio
```

The reference clip therefore does **two** jobs, and they are easy to conflate:
its semantic tokens prime `s1`, and its spectrogram becomes the speaker vector
for `s2`. Both come from the same few seconds of audio.

## Usage

### Speak a line

Weights (cnhubert, `s1`, `s2` and the ONNX prosody encoder) download from Hugging
Face on first use and are cached; `--models` and `--prosody` point at a local
directory instead.

```sh
export ORT_DYLIB_PATH=/usr/lib/libonnxruntime.so

echo "今天天气很好" \
  | tts --reference clip.wav --reference-text "这是一段示例录音" \
  | ffplay -f f32le -ar 32000 -ac 1 -
```

**One line of stdin is one utterance**, so a script is a file of lines and the
output is the utterances concatenated. Logs go to **stderr**, which is what keeps
stdout a clean stream of samples — `tts … > out.f32le` is always just audio.

Blank lines are skipped. The model produces **32 kHz**; `--sr` resamples (see
[step 3](#3-sample-rate-and-feeding-rvc-serve)).

### 1. The reference clip

A few seconds of clean speech of the voice you want, in any format ffmpeg reads.
It is decoded to mono 16 kHz internally. Much more than a few seconds mostly
costs time — the speaker vector is an average — and under half a second is
rejected outright as too little to describe a voice.

**`--reference-text` is required, and it is not bookkeeping.** `s1` generates by
*continuation*: it is primed with the reference's phonemes beside the reference's
semantic tokens, and then asked to keep going with your text's phonemes. Give it
only the target text and it sees phonemes for one utterance sitting next to audio
of another, finds nothing to continue, and stops after a token or two. **That
failure looks exactly like a broken decoder and is not one** — it is a prompt
that does not cohere. So the transcript must be what is actually said in the
clip, not a description of it.

```sh
tts --reference clip.wav --reference-text "这是一段示例录音" < script.txt > out.f32le
```

Both flags are checked in `run` rather than by clap, so a missing one costs a
message rather than a model load. (`-t` is the short form of
`--reference-text`; `-r` of `--reference`.)

### 2. Steering the delivery

`s1` samples rather than taking the argmax, because it is choosing among a
thousand codebook entries and the greedy choice collapses into repeating a single
sound.

| flag | default | what it does |
|---|---|---|
| `--top-k` | `15` | sample from the `k` highest-scoring tokens. Lower is steadier, higher is more varied. |
| `--temperature` | `1.0` | below 1 sharpens the distribution, above 1 flattens it. |
| `--repetition-penalty` | `1.35` | pushes down tokens already generated. |
| `--seed` | `0` | a synthesis is reproducible from its seed. |
| `--max-tokens` | `1500` | cap on generated tokens per line. At 25 Hz, 1500 is a minute. |

**The repetition penalty is load-bearing, not a refinement.** Without it a run of
the same token becomes self-reinforcing and the utterance never ends — the model
simply keeps saying the same sound until `--max-tokens` cuts it off. It is
applied *before* top-k, so a penalised token can drop out of the candidate set
entirely, and it is asymmetric in the way upstream is: a positive score is
divided and a negative one multiplied, so both move *down*.

Hitting `--max-tokens` without an end-of-sequence token logs a warning and the
line may be cut off. If that happens repeatedly on short text, the prompt is the
suspect — re-read [step 1](#1-the-reference-clip).

```sh
# steadier, reproducible
echo "今天天气很好" | tts -r clip.wav -t "这是一段示例录音" --top-k 5 --seed 42
```

### 3. Sample rate, and feeding `rvc serve`

The synthesizer produces 32 kHz. `--sr` resamples the output, and that flag
exists for one pipeline in particular — synthesise, then retimbre:

```sh
tts --reference clip.wav --reference-text "这是一段示例录音" --sr 16000 < script.txt \
  | rvc serve -m models/voice.safetensors --model-sr 48000 \
  | ffplay -f f32le -ar 48000 -ac 1 -
```

16 kHz is what `rvc serve` reads (the RVC analysis rate), so the two filters
join with no intermediate file and no ffmpeg in between. The whole toolkit is
this shape: `voice -(stt)-> text -(tts)-> voice -(rvc)-> voice`, each stage a
plain Unix filter over raw f32le PCM.

Resampling is linear interpolation. The synthesizer's output is already
band-limited well below either rate's Nyquist, so that costs nothing audible and
saves a dependency on a filter design.

### 4. Fine-tune a voice (`tts train`)

Cloning from one reference clip already gets the timbre. Fine-tuning adapts the
**delivery** — pacing, emphasis, where a speaker breathes — because those live in
the semantic token sequence `s1` predicts, not in the waveform `s2` renders. It
is what you reach for when a particular voice is worth more than a few seconds of
prompt.

A corpus is audio beside transcripts: `<stem>.wav` next to `<stem>.txt`, in one
directory. `.mp3`, `.flac`, `.m4a`, `.ogg` and `.opus` are read too. Audio with
no transcript is skipped with a warning rather than failing the run.

**`stt` is how the transcripts get written**, which is the whole reason the two
engines sit in one toolkit:

```sh
# fish
for f in corpus/*.wav
  ffmpeg -v quiet -i $f -f f32le -ar 16000 -ac 1 - | stt > (string replace .wav .txt $f)
end

tts train corpus/ -o tuned/voice --epochs 10
tts --reference clip.wav --reference-text "这是一段示例录音" \
  --s1 tuned/voice.safetensors < script.txt > out.f32le
```

Read the generated transcripts before training. `stt` is good but not perfect,
and a wrong transcript teaches `s1` that a phoneme sequence maps to audio it does
not match — which is the same failure mode as a wrong `--reference-text`, spread
over the corpus.

**The loss means something on its own.** Training is plain next-token
cross-entropy over the semantic tokens: one model, one optimizer, one number.
Unlike `rvc train`'s adversarial loop — where `g` and `d` only have to stay
balanced — this loss should **fall and keep falling**. A loss that stops moving
is a run that has stopped learning.

**Two files are written**, the same `Checkpoint` family `rvc train` uses:

| file | what it is |
|---|---|
| `tuned/voice.safetensors` | the weight **EMA** — the one you deploy |
| `tuned/voice.raw.safetensors` | the live final-step weights |

The EMA is markedly steadier than the live weights on a corpus this small, which
is why it is the default output. `--ema-frac 0` turns it off, and then
`voice.safetensors` holds the live weights and no `.raw` twin is written at all.
There is no discriminator sidecar: `s1` has no adversary.

Deploy by pointing synthesis at the EMA file — it replaces the base `s1` and
nothing else, so cnhubert, `s2` and the prosody encoder still come from the
downloaded bundle:

```sh
tts --s1 tuned/voice.safetensors -r clip.wav -t "这是一段示例录音" < script.txt
```

You still pass a `--reference` clip after fine-tuning: `s1` continues from a
prompt whether or not it was tuned, and the timbre still comes from `s2`'s
speaker vector.

**Flags.**

| flag | default | what it does |
|---|---|---|
| `-o`, `--out` | `models/tts/voice` | output path family (`.safetensors`, `.raw.safetensors`) |
| `-m`, `--models` | auto-downloaded | directory holding the base models |
| `--prosody` | auto-downloaded | directory holding the ONNX prosody encoder |
| `--cache-dir` | the HF cache | where downloads land |
| `-l`, `--language` | `zh` | language of the transcripts |
| `-e`, `--epochs` | `10` | passes over the corpus |
| `-b`, `--batch-size` | `1` | clips per optimizer step |
| `--lr` | `1e-5` | starting learning rate |
| `--lr-final` | `0.1` | end-of-run LR as a fraction of `--lr` |
| `--ema-frac` | `0.1` | EMA window as a fraction of the run; `0` saves raw weights |
| `--max-tokens` | `1500` | skip clips longer than this many semantic tokens |
| `--backend` / `--device` | `auto` | as in [Compute backends & devices](#compute-backends--devices) |
| `--no-tui` | off | disable the dashboard and log to stderr |

Three of these repay a moment's thought:

- **`--batch-size` is accumulation, not padding.** Clips differ in length and
  this loop does not pad, so a batch is processed one clip at a time and the
  gradients accumulated. `1` is the honest default; raising it buys a steadier
  gradient at no extra VRAM, and costs proportionally more time per step.
- **`--max-tokens` skips rather than truncates.** A clip past the cap is left out
  of the run entirely, because a *cut-off* sequence would teach the model to stop
  early — precisely the failure that is hardest to notice and hardest to undo.
  Over-long clips are reported in a warning; if most of the corpus is skipped,
  slice it into sentences first.
- **The learning rate decays exponentially** from `--lr` to `--lr × --lr-final`
  across the whole run, the same shape `rvc train` uses. A small corpus overfits
  fast at a flat rate. `--lr-final 1.0` disables the decay.

**Preparation happens once, up front.** cnhubert, the quantiser and the prosody
BERT are all frozen, so the whole corpus is encoded to phonemes + prosody +
semantic tokens before the first optimizer step. Expect an idle-looking pause at
the start of a run proportional to the corpus, not to the epochs.

**Dashboard and early stop.** On a terminal (and without `--no-tui`) training
shows Burn's live TUI with the loss and learning rate plotted; **`q` stops early
and saves**. Two honest caveats: `tts train` logs to stderr regardless, so
redirect it (`2> train.log`) to keep the display clean, and **without the TUI
there is no early-stop key** — Ctrl-C ends the process and nothing is written.
Pick `--epochs` accordingly on a headless run.

`s2` fine-tuning — which would adapt the *timbre* rather than the delivery — is
**not implemented yet**; see [Roadmap](#roadmap). Until it lands, timbre comes
entirely from the reference clip, and that is usually enough.

### 5. Language support

`--language zh|en|ja`, default `zh`.

**Chinese works.** The front-end is `text-kit`: script-based language splitting,
jieba segmentation, pinyin, the opencpop phoneme mapping and tone sandhi, all
pure Rust with no model and no tensors. Its output addresses GPT-SoVITS's
732-symbol table by index, which is why that table is embedded verbatim rather
than rebuilt — off by one entry and the model produces confident nonsense rather
than an error.

**English is in progress.** Until it lands, `--language en` **errors rather than
guessing**, and that is deliberate: running text through the wrong front-end
produces fluent-sounding *wrong* audio, which is far harder to notice than a
refusal. `--language ja` errors for the same reason and has no work scheduled.

**The accuracy ceiling on Chinese is the polyphone dictionary.** Upstream reads
pronunciations with `pypinyin` and its ~130k-entry phrase dictionary; the
`pinyin` crate is per-character, so word-dependent polyphones (银行 as *hang*,
not *xing*) fall back to the commonest reading. `text-kit`'s `chinese.rs` patches
the frequent cases by hand. A real phrase dictionary, or g2pw, is the fix.

**Prosody features are Chinese-only** even upstream — the encoder is a Chinese
RoBERTa — so other languages are given zeros. That is a real input rather than an
error path, and it is what GPT-SoVITS itself does.

### 6. Shell completions

`tts completions <shell>` prints a completion script (generated from the actual
flags, so it never drifts) to stdout — `bash`, `zsh`, `fish`, `powershell`, or
`elvish`:

```sh
tts completions zsh  > ~/.zfunc/_tts
tts completions bash > /etc/bash_completion.d/tts
tts completions fish > ~/.config/fish/completions/tts.fish
```

## For developers

### Building from source

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
path baked into `tts` doesn't apply to them. The `tts` binary itself never needs
it.

### Checking the network port

There are no unit tests for the networks. Correctness is checked in three layers,
because **weight coverage alone is not enough** — this repo has already shipped a
port that loaded at 100% and produced garbage.

First, coverage: does the module tree match the checkpoint?

```sh
cargo run -p burn-gptsovits --example load -- hubert    <chinese-hubert-base/pytorch_model.bin>  # 210/0
cargo run -p burn-gptsovits --example load -- quantizer <s2G2333k.pth>   # 3/0
cargo run -p burn-gptsovits --example load -- sovits    <s2G2333k.pth>   # 773/0
cargo run -p burn-gptsovits --example load -- t2s       <s1*.ckpt>       # 295/0

# what names a new checkpoint expects, which is the first thing to run against one
cargo run -p burn-gptsovits --example keys -- --group <any checkpoint>
```

`sovits`'s three unused tensors are the codebook's EMA training statistics
(`embed_avg`, `cluster_size`, `inited`), which have no role in a lookup.

Second, arithmetic. `reconstruct` round-trips real audio through cnhubert, the
quantiser and the synthesizer, and writes the result back out as PCM:

```sh
cargo run -p burn-gptsovits --example reconstruct -- \
  <chinese-hubert-base/pytorch_model.bin> <s2G*.pth> <in.f32le@16k> <out.f32le@32k>
```

Comparing that output's energy envelope with the input's gives **r = 0.91**
against a chance baseline of 0.30, and spectral flatness 0.17 against 1.0 for
noise. Cheap, needs no reference implementation, and a mis-wired MRTE or a
mis-scaled style attention — both of which load at 100% coverage — fails it
loudly.

Third, end to end: synthesise, then transcribe the result with `stt` and compare
the text you put in with the text that comes out. An independent model listening
is as close to *listening* as an automated check gets.

### A known wart: `tts` leaks its models at exit

`args::run` and `backend::train_s1` both call `std::mem::forget` on the loaded
models rather than dropping them, and that is deliberate. **An ONNX Runtime
session on the CUDA execution provider must not be unwound in a process that also
drives a CUDA Burn backend**: doing so aborts with glibc's "corrupted
double-linked list" *after* every sample has been written — harmless to the
audio, fatal to the exit code, which for a filter in a pipeline is the part that
gets checked. It needs both runtimes to reproduce: omitting `--prosody` exits 0,
and `CUDA_VISIBLE_DEVICES=` exits 0. So a successful run exits **0** rather than
134, at the cost of handing memory back moments before the kernel reclaims it
anyway.

### Crate layout

| crate | role |
|-------|------|
| `text-kit` | grapheme-to-phoneme: script-based language splitting, Mandarin g2p (jieba + pinyin + opencpop + tone sandhi), and GPT-SoVITS's 732-symbol table. No model, no tensors, fully testable without weights |
| `burn-gptsovits` | the GPT-SoVITS network: cnhubert, the quantiser, `s2` (SoVITS) and `s1` (T2S) — no app deps, no compute backend named |
| `burn-vits` | the VITS blocks RVC and GPT-SoVITS share: attention, WaveNet, flow, posterior encoder, ResBlock1, the family's losses, the differentiable STFT |
| `tts-core` | the synthesis pipeline: reference analysis, `s1` sampling with a KV cache, `s2` decode, and the ONNX prosody encoder behind the `ProsodyEncoder` trait |
| `tts-train` | native Rust/Burn fine-tuning: corpus preparation with the frozen encoders, and the `s1` cross-entropy loop |
| `train-kit` | training scaffolding with no model knowledge: `Checkpoint`, `ema_update`, `accumulate`, the TUI `Dashboard` — shared with `rvc-train` |
| `burn-kit` | device resolution (`--device`, `auto_backend`) and checkpoint loading, shared by every network crate |
| `audio-kit` | ffmpeg decode/resample and raw-PCM I/O, as `futures::Stream` of mono `f32` |
| `hub-kit` | auto-download of cnhubert, `s1`, `s2` and the prosody encoder from Hugging Face |
| `cli-kit` | logging, shell completions, `--device` parsing — shared by all four binaries |
| `tts-cli` | the `tts` binary (clap) — also a library, so `voice` hosts the same subcommands from the same flag definitions |

## Status

- **End-to-end synthesis works**: every network of GPT-SoVITS v2 is ported to
  Burn (`crates/burn-gptsovits`), loads its published weights unchanged, and
  `tts` speaks Chinese from a reference clip on any of the three Burn backends.
- **`s1` fine-tuning works** (`tts train`): native Rust/Burn next-token
  cross-entropy with LR decay, weight EMA and the live dashboard. Adapts
  delivery; the loss falls and means something on its own.
- **`s2` is verified numerically, not just structurally**: `examples/reconstruct`
  correlates a round-tripped envelope at r=0.91 against a 0.30 chance baseline.
- **Chinese only.** `--language en` and `ja` error rather than guess. English is
  being written now; Japanese is not scheduled.
- **`s2` fine-tuning is not implemented.** Timbre comes from the reference clip.
- **No `--backend onnx` for synthesis yet.** `s1` and `s2` run as Burn modules;
  the only ONNX in the path today is the frozen prosody encoder.

## Roadmap

- **`s2` fine-tuning** — the VITS adversarial loop, so a fine-tune adapts timbre
  as well as delivery. Being written now.
- **An ONNX Runtime inference path for `s1` and `s2`**, so a frozen model can be
  deployed anywhere ONNX Runtime runs and any backend can be chosen for
  inference. Burn remains available for fine-tuning on every model, always.
  Being written now.
- **English front-end** in `text-kit`, then Japanese. Being written now.
- A real Mandarin phrase dictionary (or g2pw) to lift the polyphone ceiling.
- A Burn port of the prosody encoder, once reading
  `hfl/chinese-roberta-wwm-ext-large`'s own weights is worth more than trusting a
  third-party conversion of them. The `ProsodyEncoder` trait already leaves the
  slot open.
- Streaming synthesis: `s1` is autoregressive, so tokens could be handed to `s2`
  in chunks rather than a whole utterance at a time.
