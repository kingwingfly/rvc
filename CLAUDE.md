# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`voice` is a pure-Rust speech toolkit whose engines compose over Unix pipes:
`voice -(stt)-> text -(translate)-> text -(tts)-> voice -(rvc)-> voice`. Everything
but `translate` works today.

`rvc` is RVC v2 voice conversion: it retimbres a source voice into a trained
target voice while preserving content + F0 pitch (so breathy/expressive
vocalizations survive by construction). Inference runs on **three interchangeable
generator backends** — ONNX Runtime (`ort`), and native Burn on either LibTorch
(`burn-tch`, ~9x faster), CubeCL/CUDA or WebGPU — and training is native Rust/Burn
on any of the three.

### Three rules that are easy to break silently

**No Python.** Not for users, not for developers, not for setup. New models are
**ported to Burn** and load their original Hugging Face weights directly, training
loops included — never wrapped in a Python process, a `uv` project, or a
preprocessing script. Reading a cloned upstream repo as a porting reference is
fine; running or shipping it is not. This is not a style preference: driving the
RVC Python repo directly was tried and abandoned because it was unmaintainable
for the authors and unusable for anyone else. The single `.safetensors → ONNX`
exporter under `export/` is the deliberate exception, and it **gains**
responsibilities rather than losing them — Burn imports ONNX graphs and cannot
emit one, so that script is the only bridge from a model fine-tuned here to ONNX
Runtime, which this toolkit treats as a supported deployment target (see **Which
runtime a model gets, and why**). What keeps the rule intact is that it stays a
maintainer's build-time tool: no user, no test and no training run invokes it.

**One binary per engine, plus `voice`.** `rvc`, `stt` and `tts` each stand alone
and pull in only what they use — installing `stt` costs none of the RVC stack, and
its `onnx` feature is opt-out, so a Burn-only `stt` links no ORT at all.
`voice` is the *integration*: it depends on `rvc-cli` and
`stt-cli` **as libraries**, so an argument is defined exactly once and never
copied between binaries. Every `*-cli` crate is therefore a lib **and** a bin.
`rvc-cli::args::RvcCommand` is `#[command(flatten)]`ed by `rvc` and nested by
`voice`; `stt-cli::SttArgs` is `#[command(flatten)]`ed by `stt` and nested by
`voice`. User-facing strings in shared code must not name a binary ("train one
with the `train` subcommand", not "`rvc train`").

**No engine depends on another engine.** Voice conversion, recognition and
synthesis are siblings. Anything two of them need moves to a neutral crate first.

### Naming conventions
Three tiers, and the name says which tier a crate is in:

- **`*-kit`** — shared plumbing with no model and no engine knowledge, safe for
  anything to depend on: `burn-kit` (devices, checkpoints), `audio-kit` (ffmpeg
  I/O, the slicer), `hub-kit` (downloads), `cli-kit` (logging, completions).
- **`burn-*`** — one network each, named after the **model** (`burn-rvc` reads
  like `burn_dinov3`), holding no app dependencies and naming no compute backend.
- **`<engine>-core` / `<engine>-cli`** — one engine each, all the same shape:
  `rvc-core`+`rvc-train`+`rvc-cli`, `stt-core`+`stt-cli`, later `tts-core`+`tts-cli`.

**`voice-` is reserved for the top.** It marks the integration, so a crate that
an engine depends on must never be named `voice-*` — that is why the shared
crates are `*-kit`. `voice-cli` is the only `voice-*` crate.

### Where documentation goes
Four places, and putting a paragraph in the wrong one is exactly how `README.md`
once grew ninety lines of engine manual:

- **`README.md`** is an *index*: what `voice` is, the pipeline, which binary to
  install, one build block, links out. No engine documentation, ever — if a
  passage names a flag, it belongs in an engine README.
- **`crates/<engine>-cli/README.md`** is that engine's manual: every flag, which
  weights are fetched from where, which backends it accepts, its training loop.
  Setup shared by all four binaries (ORT, LibTorch, ffmpeg) is written once in
  `crates/rvc-cli/README.md` and linked, not copied.
- **`docs/*.typ`** are the long-form architecture papers, one per network —
  `rvc-architecture`, `gptsovits-architecture`, `whisper-architecture` — for
  *reviewing* a port rather than using it: what each block computes, what every
  loss term is for, why the training loop has the shape it does. Typst sources
  with the rendered PDF committed beside them, so reading needs no toolchain;
  rebuild with `typst compile docs/<name>.typ` and commit both.
- **this file** is the fourth: why a decision was made and which trap it avoids.
  A fact that would be equally true of any VITS repo belongs in a `.typ` paper;
  a fact that will bite whoever edits this code next belongs here.

## Build / run / verify

```sh
# Neither ONNX Runtime nor LibTorch is bundled or downloaded — point at your own:
export ORT_DYLIB_PATH=/usr/lib/libonnxruntime.so
export LIBTORCH=$PWD/libtorch   # must be exactly 2.9.0 (what tch 0.22 targets)

cargo build                     # all four binaries: `rvc`, `stt`, `tts`, `voice`
cargo build --release -p rvc-cli   # just `rvc`, optimised

# `cargo build` (dev) is the right default even for running models: the profile
# gives dependencies `opt-level = 3` — that is where every tensor op lives — and
# leaves workspace crates cheap to recompile. Transcribing 24 s of audio takes
# 19 s in debug against 18 s in release, for a fraction of the build. Reach for
# `--release` to benchmark, not to test.
cargo check -p rvc-core  --features cuda,tch
cargo check -p rvc-train --features cuda,tch    # where the trait bounds bite
cargo clippy --workspace        # workspace is kept clippy-clean
cargo fmt

# `--backend auto|onnx|cuda|tch` (aliases: burn/burn-cuda, libtorch/burn-tch);
# `--device auto|cpu|cuda|cuda:N|mps|vulkan`. Both default to auto.
cargo run -p rvc-cli   --      convert -m models/voice.safetensors --model-sr 48000 -o out/ in.mp3
cargo run -p voice-cli -- rvc  convert -m models/voice.safetensors --model-sr 48000 -o out/ in.mp3

# No LibTorch on the machine? Drop it (then `--backend tch` errors cleanly):
cargo build --release --no-default-features --features cuda

# Correctness of the Burn port is checked by loading REAL pretrained weights
# (there are no unit tests for the network) — reports applied/missing/unused:
cargo run -p burn-rvc --example load -- <path/to/f0G48k.pth>       # 560/0, 165/0
cargo run -p burn-whisper --example load -- <path/to/model.safetensors>  # 587/0
cargo run -p burn-gptsovits --example load -- hubert <chinese-hubert-base/pytorch_model.bin>  # 210/0
cargo run -p burn-gptsovits --example keys -- --group <any checkpoint>   # what names to mirror
```

Porting references are cloned under `/.reference` (gitignored) and **read, never
run** — `openai/whisper`, and RVC-Project tag `2.2.231006` for `burn-rvc`.

### Two runtimes for one model (`stt-core`)
`Engine` (`engine.rs`) is the whole boundary between the decode loop and a
runtime: token ids in, `f32` logits out. Encoded audio and KV caches stay inside
the engine, because they are backend-specific tensors with no useful common type.
That keeps `Transcriber` non-generic — the backend is a constructor call, not a
type parameter — and it buys the best correctness check available: **Burn and
ONNX Runtime produce byte-identical transcripts from the same weights**, which is
how the Burn port is validated against an independent implementation.

### Porting traps found the hard way
`Tensor::triu_mask`/`tril_mask` are named for the triangle they **keep**, not the
one they mask, so a causal mask is `tril_mask(shape, n_kv - n_q)`. Using
`triu_mask` reverses time, and when a decode step has one query and one key it
masks the only position there is — a full row of `-inf` into softmax is `NaN`,
not an error, and it propagates to every logit. Burn's `assert_approx_eq` also
compares `NaN` to `NaN` without complaint, so tests must assert finiteness
separately (`burn-whisper`'s do).

`burn_store::HalfPrecisionAdapter` reads as "widen fp16 on load" and is
**bidirectional** — it also *narrows* fp32 to fp16. `burn-kit`'s `Upcast` is the
one-way version and is what the safetensors loader uses. The bug is silent (the
model just loses mantissa) except on LibTorch, which rejects a conv whose bias
dtype stops matching its input. fp16 and fp32 checkpoints of the same model are
both common on the Hub, so test against both.

**A safetensors file is not one format but two**, and `burn-kit` has a loader for
each. `load_safetensors_into` applies `PyTorchToBurnAdapter`, which transposes
Linear weights from PyTorch's `[out, in]` to Burn's `[in, out]` — right for a
Hugging Face checkpoint, wrong for one `save_safetensors` wrote, which is already
in Burn's layout and gets transposed a second time. Reading our own checkpoint
back through the PyTorch path is what `load_burn_safetensors_into` exists to
prevent. Rectangular weights fail loudly on `ShapeMismatch`; a square one would
load "fine" and be silently scrambled. `burn-kit`'s round-trip test pins it.

**An ONNX Runtime session on the CUDA execution provider must never be dropped.**
Unwinding one aborts with glibc's "corrupted double-linked list", which turns an
otherwise successful run into exit 134 — and for a Unix filter the exit code is
the part that gets checked. `CUDA_VISIBLE_DEVICES=` exits 0, and `rvc convert`
drives ORT and Burn together without tripping it, so it is narrower than "ORT and
Burn conflict".

Two things this file previously claimed about it are **wrong**, and both were
found by testing rather than by reading: a CUDA *Burn* backend is **not** required
to reproduce it, and it is **not** confined to teardown after a successful run.
The first fix here leaked the models at the end of the happy path only, so every
early return — `--backend onnx` with no export, for one — still unwound a live
session and still aborted, printing the right error and then dying 134 instead of
1. The lesson is that the leak belongs on the **type that owns the session**
(`ManuallyDrop`), not at one call site: anything else silently obliges every
future caller, including every error path, to remember.

Requires **ffmpeg 8.1** dev libraries (and the `ffmpeg` binary for the realtime
`serve` example). ContentVec + RMVPE ONNX assets auto-download from Hugging Face
(`rvc models download` to prefetch). Only 48 kHz is supported today.

## Architecture

Data crosses every crate boundary as **mono `f32`**, and the whole conversion path
is `futures::Stream`-in → `futures::Stream`-out, which is why `rvc serve` is a plain
Unix filter (raw f32le PCM stdin→stdout) and batch `convert` is a thin wrapper.

| crate | role |
|-------|------|
| `burn-kit` | Burn plumbing with no model knowledge: `--device` resolution and checkpoint loading, shared by every network crate |
| `audio-kit` | ffmpeg decode/resample + WAV/raw-PCM I/O, all as `futures::Stream<f32>` |
| `rvc-core` | the voice-conversion pipeline: `FeatureExtractor` (ContentVec + RMVPE), coarse-pitch/upsample/pitch-shift DSP, streaming `Converter` (block/overlap with an **overlapping** crossfade — consecutive kept blocks share `xf_out` output samples so the blend adds, never deletes, audio), an optional post de-hiss stage (`denoise.rs`, `--denoise`), and **all three** generator backends (ort, Burn/LibTorch, Burn/CubeCL) behind one `Generator` trait |
| `burn-vits` | the VITS blocks RVC and GPT-SoVITS share (both descend from the same source, which is why their `state_dict` names line up): attention stack, `Wn`, flow, posterior encoder, `ResBlock1`, weight-norm convs, discriminators, the family's losses, the differentiable STFT |
| `burn-rvc` | what is RVC's alone: `SourceModule` (NSF), the 768-dim `TextEncoder`, `GeneratorNsf`, the synthesizer wiring; re-exports `burn-vits` so it still reads as one model |
| `burn-whisper` | the Whisper network (standalone Burn port); mirrors HF's `state_dict` layout so `openai/whisper-large-v3-turbo` loads unchanged |
| `burn-gptsovits` | the GPT-SoVITS network. `hubert` at 210/0, `quantizer` at 3/0, and `s2` complete at 773/0 (the 3 unused are the codebook's EMA training statistics). **`s2` is verified numerically, not just structurally**: `examples/reconstruct` round-trips real audio through cnhubert, the quantiser and the synthesizer, and the output tracks the source's energy envelope at r=0.91 against a chance baseline of 0.30. `t2s` (`s1`) is at 295/0. Every network of GPT-SoVITS is now ported; `tts-core`/`tts-cli` wire them into a working `tts`, and `tts-train` fine-tunes **both** stages — `s1` for delivery, `s2` for timbre. `SovitsPartial::forward_train` composes `enc_q` → `flow.forward` → random segment → `dec` and returns the five tensors the VITS losses need; the matching `s2D2333k.pth` discriminator loads at 111/0/0. `examples/keys` lists any checkpoint's tensors, which is the first thing to run against a new one |
| `rvc-train` | native Rust/Burn adversarial training loop (see `crates/rvc-train/ARCHITECTURE.md`) |
| `hub-kit` | auto-download every engine's assets from Hugging Face |
| `rvc-cli` | lib **and** the `rvc` binary (clap): `convert`, `serve`, `models`, `train`, `preprocess` |
| `stt-core` | speech recognition: Whisper log-mel front-end, BPE vocabulary, KV-cached greedy decode, segmentation via `audio-kit`'s slicer, and **two runtimes** (native Burn, ONNX Runtime) behind one `Engine` trait |
| `stt-cli` | lib **and** the `stt` binary |
| `tts-core` | speech synthesis: reference analysis, `s1` sampling with a KV cache, `s2` decode, and the ONNX prosody encoder behind a trait. **Two runtimes**, the same shape `stt-core` uses: `Engine` is the whole boundary, so `Synthesizer` is not generic and the backend is a constructor call rather than a type parameter. `--backend onnx` runs the whole stack — cnhubert, the quantiser, `ref_enc`, `s1` and `s2` — off four exported graphs, and `auto` picks it when the model directory holds an export |
| `tts-train` | fine-tuning GPT-SoVITS. `s1` is plain next-token cross-entropy over `T2s::forward_prompt_all` — one model, one optimizer, one loss, so unlike `rvc-train` the number means something on its own. `s2` is the other half: an adversarial VITS loop over `burn-vits`'s shared discriminators, inheriting `rvc-train`'s loss family (mel-L1 ×45, KL ×1, feature matching ×2, LSGAN) rather than inventing one, with GPT-SoVITS's five discriminator periods `[2,3,5,7,11]` against RVC's eight. Verified on 13 clips: mel falls 26.6 → 18.2 over two epochs on GPU and on CPU alike. `--stage s1|s2|both` prepares the corpus exactly once — preparation is the expensive half — and each stage writes its own checkpoint family. A corpus is `<stem>.wav` + `<stem>.txt` pairs, and `stt` is how the transcripts get written |
| `tts-cli` | lib **and** the `tts` binary |
| `cli-kit` | logging, shell completions and `--device` parsing, shared by all four binaries |
| `train-kit` | training scaffolding with no model knowledge: `Checkpoint`, `ema_update`, `accumulate`, `materialize`, `Dashboard`. Generic over the module trained, so a GAN and a cross-entropy loop share it |
| `text-kit` | grapheme-to-phoneme: script-based language splitting, Mandarin g2p (jieba + pinyin + opencpop + tone sandhi), and GPT-SoVITS's 732-symbol table. English g2p is an embedded CMUdict over upstream's deterministic cascade; Mandarin polyphones come from `pypinyin`'s own 47k phrase dictionary. Pure Rust, no ML, no backend — so it is fully testable without weights |
| `voice-cli` | the `voice` binary: `rvc-cli`, `stt-cli` and `tts-cli` nested as `voice rvc …`, `voice stt` and `voice tts` |

### Three runtimes, one path (the key abstraction)
Everything downstream of the generator is shared: the same `FeatureExtractor`, the
same DSP, and the same streaming `Converter` drive **either** backend through the
`Generator` trait (`crates/rvc-core/src/backend.rs`). `BurnGenerator<B>` is generic
over the Burn compute backend and stores its device; the concrete `cuda_generator`
/ `libtorch_generator` constructors return `impl Generator`, so `rvc-cli` never
names a Burn type and the choice is purely run-time. `auto` picks by the `-m`
extension (`.onnx` → ONNX Runtime), then LibTorch-on-CUDA → CubeCL/CUDA →
LibTorch-on-CPU. Naming a backend or device explicitly is an error if it is
unavailable, never a fallback. The CubeCL/CUDA generator is still slower than
realtime for `serve`; `--backend tch` is ~9x faster per file on an RTX 2060 and is
what `auto` picks for `.safetensors` weights.

The **alias sets are not yet uniform across engines** — `rvc` accepts a bare
`burn` for `cuda`, `stt` only `burn-cuda` — which is a genuine inconsistency, not
a documented distinction. Anything that unifies `--backend` parsing should move
it into `cli-kit` beside `--device`, which every binary already shares.

Device selection lives in `crates/burn-kit/src/device.rs` (`DeviceSpec`,
`cuda_device`, `libtorch_device`, `wgpu_device`, `guard_init`) — a crate that knows
about no model, so every engine and every subcommand resolves `auto` identically.
Two traps encoded there: `LibTorchDevice::default()` is **CPU** (unlike `CudaDevice::default()`), and
constructing `LibTorchDevice::Cuda` on a CPU-only LibTorch is a hard panic baked in
by `burn-tch`'s build script — so `libtorch_device` is the only place it is built.

### The `burn` / `cuda` / `tch` features
`rvc-core`'s Burn backend (`burn_backend.rs`, `pub BurnGenerator<B>`) is behind the
optional `burn` feature (off by default) so consumers that only need the shared
`FeatureExtractor` stay a lean `ort` crate. `burn` alone gives the *generic*
generator and no compute backend; `cuda` and `tch` each add one and can both be on
at once. `rvc-train` mirrors the same features. `rvc-cli` defaults to all three —
one binary, run-time choice. — and `crates/rvc-core/build.rs`
refuses a `tch` build with no `LIBTORCH`, because `burn-tch` hardcodes
`tch/download-libtorch` and cargo features are additive, so the silent fallback
would otherwise be a multi-GB download of a **CPU-only** LibTorch.
`crates/{rvc,stt,voice}-cli/build.rs` bake `$LIBTORCH/lib` into the binary as a
`RUNPATH`; without it a missing `libtorch.so` aborts in `ld.so` before `main`, on
every subcommand. They are deliberate duplicates — an rpath is per-executable.

### Which runtime a model gets, and why
The target is that **the user picks the backend — for inference and for
fine-tuning alike — and ONNX Runtime is a first-class deployment target for every
model, not legacy to be retired.** Two asymmetries decide how far each model gets
toward that:

- **ONNX Runtime cannot train.** So a model that is fine-tuned here *must* be a
  Burn port, whatever else it also runs on. That is not a preference for Burn; it
  is the only way a `train` subcommand can exist at all, and it is why Burn is
  always available for tuning even where ONNX is the faster inference path.
- **Burn imports ONNX and cannot emit it.** So a checkpoint trained here reaches
  ONNX Runtime only through `export/`. That is why the exporter gains scope
  instead of being deleted: `rvc`'s generator and all of GPT-SoVITS today.

Where a model is **frozen** and an export already exists, ONNX is often simply
the cheaper answer:

- `tts-core`'s prosody BERT is ONNX. It is frozen, an export exists, and one
  sentence through 24 layers is launch-overhead bound, so a port would not be
  meaningfully faster. The trait (`ProsodyEncoder`) leaves the slot open.
- `burn-gptsovits`'s cnhubert is a Burn port, done before that reasoning was
  settled. Keeping it costs nothing, it is verified at 210/0, and it is what lets
  `tts` run without ORT once `--prosody` is left off. It now *also* has an ONNX
  graph, which is the shape to aim for everywhere: both, chosen at run time.
- GPT-SoVITS `s1`/`s2` are Burn because they are fine-tuned, and they now run on
  ONNX Runtime as well — `alongside` Burn, never instead of it. `tts` is therefore
  the first engine to reach the target in full: every model it uses runs either
  way, and tuning stays on Burn.

**This reverses an earlier position on purpose, so do not "restore" it.** The old
plan was to port `rvc-core`'s ContentVec and RMVPE to Burn *in order to* drop
ONNX Runtime from the toolkit entirely and take `ORT_DYLIB_PATH` out of setup.
Backend choice turned out to be worth more than one environment variable: people
deploy where they deploy, and on plenty of targets ORT is the only runtime
available. Porting ContentVec and RMVPE is still worth doing — it would give
`rvc`'s feature extraction the same run-time choice its generator already has,
and demote ORT from a hard requirement to an option — but as an **addition**.
ContentVec is a HuBERT variant, so `burn-gptsovits`'s `hubert.rs` already covers
its architecture; doing it means lifting that module into a `burn-hubert` of its
own, since two engines would then share it.

### `s1` generates by continuation (`tts-core`)
The reference's **transcript** is part of the prompt, not metadata: `s1` is shown
the reference's phonemes beside the reference's semantic tokens and asked to
continue with the target's phonemes. Give it only the target text and it sees
text and audio that disagree, finds nothing to continue, and stops after a token
or two — a bug that looks like a broken decoder and is not. Hence
`--reference-text` being required.

**A `--reference-text` that is merely *wrong* fails differently, and worse.** An
absent one truncates loudly; a mismatched one degrades quietly: `s1` renders the
transcript it was given before reaching the target, so the output grows extra
leading speech and roughly doubles in length. Measured on the same clip, same
target (`今天天气很好`) and same seed: the clip's true transcript gives **1.32 s**
and the right words, a plausible-but-wrong transcript gives **2.36 s** with the
wrong words first. Nothing errors, and `stt` may still recover the target from the
tail, so a round-trip check can pass while the audio is wrong.

Worth knowing when writing test commands, because this is how it bites: pairing a
real recording with a *placeholder* transcript is the natural thing to do when
sanitising an example, and it manufactures exactly this artifact. Three separate
workers hit it that way and each diagnosed it as a decoder defect. **A duration
that does not match the text length is the tell.** Illustrative docs are safe —
`clip.wav` beside a generic transcript is self-consistent, since a reader supplies
both — but any command naming a real file must name what that file actually says.

### Verifying a port beyond weight coverage
Coverage says the module tree matches the checkpoint. It says nothing about
whether the forward pass computes the right thing, and this repo has already
shipped a port that loaded at 100% and produced garbage (Whisper's causal mask).
Each model therefore needs a second check that exercises arithmetic:

- `burn-whisper` — transcripts diffed against ONNX Runtime on the same weights,
  which came out byte-identical.
- `burn-gptsovits` `s2` — `examples/reconstruct` runs audio through the whole
  stage and correlates the output's energy envelope against the input's. r=0.91
  where shuffling gives 0.30, and spectral flatness 0.17 against 1.0 for noise.
  Cheap, needs no reference implementation, and a mis-wired MRTE or a
  mis-scaled attention fails it loudly.
- `tts` end to end — synthesise, then transcribe the result with `stt`. Text in
  and text out are compared by an independent model, which is as close to
  listening as an automated check gets.

### The semantic-token boundary (`burn-gptsovits::quantizer`)
25 Hz token ids over a 1024-entry codebook are what the two stages agree on: T2S
predicts them from text, SoVITS renders them to waveform, and building a training
set is running `Quantizer::encode` over the corpus. The rate comes from one
stride — cnhubert's 50 Hz halved — and everything downstream (tokens per second,
T2S sequence length) follows from it.

`s2G2333k.pth` confirms `text-kit`'s phoneme table independently:
`enc_p.text_embedding` is `[732, 192]`, and 732 is exactly the symbol count.

### The phoneme table is a compatibility contract (`text-kit`)
`text_kit::symbols::SYMBOLS` is GPT-SoVITS's v2 vocabulary verbatim, 732 entries
in order, because those indices address the T2S model's phoneme embedding. It is
embedded as data rather than rebuilt from upstream's construction (which sorts a
union of per-language sets and then appends two groups *unsorted*) — off by one
entry and the model produces confident nonsense rather than an error. Same reason
`opencpop-strict.txt` is `include_str!`d rather than read at run time.

The same table is what makes **English** cheap to add: GPT-SoVITS v2's symbol
list already carries the ARPAbet phones, so `--language en` needs no new indices
— only a g2p that emits them, which `english.rs` now does: an embedded CMUdict
(125,823 entries) behind upstream's deterministic cascade. It has no neural
out-of-vocabulary model, because Rust has no equivalent of `g2p_en`'s LSTM, so a
word that survives dictionary, possessive and compound handling is **spelled
letter by letter rather than guessed**. English also yields `word2ph: None`, so
the prosody encoder is fed zeros — intelligible, flatter than Chinese, and
upstream's own behaviour. Splitting is by
script, so a mixed sentence routes each run to its own front-end and the phoneme
streams concatenate into one sequence — the reason language selection is a
per-run property and not a global mode.

Mandarin is no longer per-character. The `pinyin` crate reads one character at a
time, so word-dependent polyphones (银行 as *hang*, not *xing*) used to fall back
to the commonest reading, patched by a twelve-entry table. `data/phrases.txt` now
embeds **`pypinyin`'s own 47,111-entry phrase dictionary** — the same table
upstream consults — so the coverage matches rather than approximating it, and
`POLYPHONES` is gone. Lookup is **longest match within each jieba token**, which
is deliberately better than upstream: upstream looks up the whole token only, so
it loses 银行 the moment jieba hands it 银行卡, because jieba's word list and
`pypinyin`'s table disagree about where words end.

The remaining ceiling is **syntactic**, not lexical: a reading that depends on
grammar rather than on the word is still wrong. 还 is its own jieba token in both
他还没来 (*hai*) and 把钱还他 (*huan*), so both come out *hai*. g2pw is the
endgame, and a phrase dictionary cannot reach it.

While porting that dictionary a live bug surfaced: the `pinyin` crate writes 绿 as
`lü4` while `opencpop-strict.txt` keys its finals `lv`/`nve`, so every word
containing 绿/女/略/律 missed the lookup and was emitted as `UNK` — a silently
dropped syllable, not an error. The fallback now rewrites ü to v.

### Lazy parameters (`train_kit::materialize`)
Burn allocates parameters lazily, and two things go wrong while a module is still
lazy: a clone taken beforehand gets **fresh `ParamId`s**, and parameters that
materialise *during* the differentiated pass yield no gradients at all. Either
way a data-parallel replica's gradients stop matching the master's and are
dropped silently — every extra device contributes nothing while the run looks
healthy. Warm-start and resume materialise on load; training from scratch does
not, so `materialize` is called before any replica is made.

### Moving a module between crates is free
Burn derives parameter paths from the field names of the struct that *contains* a
module, not from the crate it was declared in. That is what made `burn-vits`
extractable with the checkpoints untouched, and the check is exact: `burn-rvc`'s
`load` example still reports 560/0/0 and 165/0/0. Renaming a **field** does move
a path; renaming a *type* or moving a *file* does not.

### Weight-compatibility constraint (important when editing `burn-rvc`)
The Burn modules are kept **weight-compatible with RVC's PyTorch `state_dict`** so
they can warm-start from the public pretrained bases (`f0G48k.pth`/`f0D48k.pth`, HF
`lj1995/VoiceConversionWebUI`) — essential on a small (~1 h) corpus. The loader
(`crates/burn-rvc/src/store.rs`) remaps RVC's flat `attn_layers`/`norm_layers_*` lists
and the flow's even coupling indices onto the module tree and upcasts fp16→fp32.
Changing the module layout breaks warm-start and the ONNX exporter — keep names/
structure in step with the reference (RVC-Project tag `2.2.231006`,
`infer/lib/infer_pack/{models,attentions,modules}.py`).

### The Python boundary (`export/`)
The only Python: a standalone `uv` project that converts a Burn `.safetensors` to
ONNX. `rvc_infer.py` is a **clean-room** torch reimplementation mirroring the
`burn-rvc` layout (it does NOT depend on the RVC-Project repo). Native Burn inference
needs no export — this is only for ONNX Runtime / cross-framework deploy.

It is the **only** direction that needs Python, and only because Burn reads ONNX
without writing it. Extending it to a new model means adding another clean-room
mirror of that network's Burn layout beside `rvc_infer.py` — `gptsovits_infer.py`
is the second — never importing the upstream project, and never adding a step a
user has to run. `export_gptsovits.py` reads *either* weight layout: an original
`.pth`/`.ckpt` through the same key remaps the Rust loaders apply, or a Burn
`.safetensors` from `tts train` with its `Linear` weights transposed back. That
second path is the point of the whole exercise — it is how a voice fine-tuned
here reaches ONNX Runtime.

`s1` is emitted as **two** graphs, `s1_prompt` and `s1_step`, because a KV cache
cannot be a single static graph; the weights therefore appear twice on disk.
Sampling stays on the host, so a graph is a pure function of its inputs. The
numbers that make the port checkable are in `export/README.md`: both runtimes
sampled identical tokens from the same seed, agreeing to a **max absolute
difference of 7.9e-03** on RMS 5.2e-02 (correlation 0.99998), and `s1_prompt`
over a whole prompt agrees with `s1_prompt` + `s1_step` to 6.7e-06.

```sh
uv run --project export python export/export_onnx.py models/voice.safetensors models/voice.onnx
```

Exported graph contract (matches `rvc-core`):
`phone[1,T,768] f32, phone_lengths[1] i64, pitch[1,T] i64, pitchf[1,T] f32, ds[1] i64, rnd[1,192,T] f32 → audio[1,1,L] f32`.

## Training notes

Native Rust/Burn on a GPU; `trainer::run` is generic over `AutodiffBackend` and
`crates/rvc-train/src/lib.rs` picks the concrete one at run time from `--backend`.
All three train. One constraint shapes `burn-rvc`: burn 0.21's autodiff builds a
wrongly-shaped weight gradient for a *grouped, strided* `conv1d` whose padded length
isn't a multiple of the stride — CubeCL and WebGPU absorb it, LibTorch aborts.
`DiscriminatorS` is exactly that shape, so `DiscriminatorS::forward`
(`discriminator.rs`) reflect-pads its input to a length (`SCALE_ALIGN`) the whole
chain divides evenly. Don't remove it
without re-running `cargo run -p rvc-train --example convgrad --features tch,cuda,wgpu`. No `Learner` (the GAN loop doesn't fit it: `TrainStep::step`
takes `&self` and yields one `GradientsParams` for one optimizer, while a GAN needs
two models, two optimizers at different LRs, and D updated *between* the two
backward passes) — the
trainer drives Burn's `TuiMetricsRendererWrapper` directly (`crates/rvc-train/src/dashboard.rs`).
On a TTY a live dashboard shows `g`/`d`/`mel` losses; `q` stops early and saves. Off-TTY
(or `--no-tui`), it logs to stderr and Ctrl-C stops and saves. Losses match RVC exactly
(mel-L1 ×45, KL ×1, feature-matching ×2, LSGAN); the STFT front-end is n_fft=2048,
hop=480, 128 Slaney mels, center=False (`crates/rvc-train/src/spectral.rs`). Target GPU
is 6 GB (RTX 2060) → small batch. Warm-start from `--pretrained-g/-d` is strongly
recommended on a small corpus.

**Preprocess first (`rvc preprocess`).** `sample_batch` draws random 0.48 s
windows uniformly across each corpus file, so raw recordings full of
between-sentence dead-air collapse the generator to silence. `rvc preprocess
raw/*.mp3 -o clips/` then `rvc train clips/*.wav ...` slices the corpus into
clean per-sentence clips first — it removes between-sentence dead-air while
**preserving soft/breathy ASMR content** (energy is used only to find long
silent gaps, never to gate quiet-but-present sound). The shared slicer lives in
`crates/audio-kit/src/slice.rs` (`SliceOptions`, `slice`); the two tuning knobs
are `--silence-db` (energy floor; lower to keep the softest passages) and
`--min-silence` (how long a quiet gap must last to be a cut, so sentences are
never split). Training itself is unchanged — it just consumes the cleaned folder.
