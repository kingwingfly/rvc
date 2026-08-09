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

`seedvc` is the fourth engine and does the same job on opposite terms: **a
1–30 s reference clip is the whole speaker specification, so there is nothing to
train** (see **Seed-VC, and why the whole workspace is GPL-3.0**).

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

**One binary per engine, plus `voice`.** `rvc`, `stt`, `tts` and `seedvc` each
stand alone and pull in only what they use — installing `stt` costs none of the
RVC stack, and its `onnx` feature is opt-out, so a Burn-only `stt` links no ORT
at all. `voice` is the *integration*: it depends on `rvc-cli`, `stt-cli`,
`tts-cli` and `seedvc-cli`
**as libraries**, so an argument is defined exactly once and never
copied between binaries. Every `*-cli` crate is therefore a lib **and** a bin,
and each exports one clap type that its own `main` flattens and `voice` nests —
which is what makes `voice tts train` literally the same code path as
`tts train` rather than a second definition that has to be kept in step. User-facing
strings in shared code must not name a binary ("train one
with the `train` subcommand", not "`rvc train`").

**No engine depends on another engine.** Voice conversion, recognition and
synthesis are siblings. Anything two of them need moves to a neutral crate first.

### The shape every CLI has
**Running a binary with no subcommand is the stdin→stdout filter.** Subcommands
are for everything that is not streaming:
`rvc convert|train|preprocess|download|completions`,
`tts convert|train|preprocess|download|completions`,
`stt convert|download|completions`,
`seedvc convert|download|completions` — where `completions` is the binary's and
not the engine's, so it is the one that does **not** appear under `voice` (see
below). `stt` and `tts` were already this shape;
`rvc` reached it by promoting `rvc serve` to the bare invocation. **`convert` is
the batch counterpart of the bare invocation on all four** — same engine, files
instead of a pipe — which is why it is spelled identically everywhere rather than
`transcribe`, `synthesize` and `convert`. `seedvc` was born this shape, and its
missing subcommands say something rather than being unfinished: no `train`
because it is zero-shot, and no `preprocess` because a corpus is what a trainer
eats.

That is a deliberate promotion rather than a deletion. Streaming is the *primary*
mode of a Unix filter — it is the thing the whole `futures::Stream` pipeline
exists for — and hiding it behind a subcommand while `stt` and `tts` exposed it
directly meant the three engines could not be learned once. If a future engine
has a streaming mode, it goes on the bare invocation too.

**`stt` used to be the exception, and is not any more.** It read the whole of
stdin before transcribing anything, on the reasoning that segmentation needs to
see the recording — so a ten-minute file emitted nothing for ten minutes while a
short clip looked instant. What made the fix free is that `slice()`'s lookahead
is *bounded*: a voiced run's end can no longer move once
`max(min_silence, 2·pad)` of silence has followed it, so `audio_kit::Slicer`
finalises there and the ranges are the ones the whole recording would have
given. **A property test pins the streaming and batch slicers to identical cuts**
— that equivalence is the reason this is not a speed/accuracy trade, and it is
the thing to re-run before touching either. The single divergence is speech that
never pauses for longer than `--max-clip`: streaming cuts at the quietest frame
in the window it has, where batch balances the split across the whole run.

The corollary is that **`voice` nests, it never renames.** `voice tts train`, not
`voice tts-train`: `voice` hosts each engine's clap type unchanged, so a
subcommand added to `tts` appears under `voice tts` with no edit to `voice-cli`
at all. A hyphenated name is the tell that someone flattened a level by hand.

One exception to reading a hyphen that way, and it is the reverse case: clap
names a subcommand by kebab-casing its variant, so `Command::SeedVc` would
render as `voice seed-vc` while the binary is `seedvc`. The variant therefore
carries `#[command(name = "seedvc")]`. That is a rename **back** to the engine's
own name, which is the rule rather than an exception to it.

### `completions` belongs to the binary, not to the engine
It is the one subcommand in the list above that is **not** part of any engine's
clap type, and the reason is that a completion script describes *an executable*.
`<E>Command` therefore stops at the engine's own verbs, and each `main.rs`
flattens that enum into a private one that adds `Completions` beside it —
`#[command(flatten)]` on a subcommand variant, which is what makes this cost one
enum per binary rather than a second definition of anything.

**This is a correction, so do not undo it by moving the variant back.** While
`Completions` sat in the shared enum, every engine grew a nested copy under
`voice`, and `voice rvc completions bash` emitted a script beginning `_voice()`
— `run` was generic over the hosting binary precisely so that arm could build
`voice`'s tree, so the nested subcommand advertised `rvc`'s completions and
produced `voice`'s. Four spellings of one script, three of them lies. Now
`voice completions` is the only one, `rvc completions` still emits `_rvc()`, and
`run` needs no type parameter at all — the generic existed only for that arm.

The rule generalises: **anything true of the executable rather than of the engine
goes in `main.rs`.** A future `--version` banner or a self-update command is the
same shape.

**Every engine has a `download`, and it fetches exactly what a default bare
invocation would fetch on demand** — no more, so a synthesis-only user is never
charged for the `s2` discriminator, and no less, so the first real run needs no
network. It is one level deep, like every other subcommand: `rvc download` was
`rvc models download`, whose extra level bought nothing and could not be
mirrored on the other two without inventing a `models` noun for each. There is
deliberately no top-level `voice download`. The old `voice models` was one — it
announced "shared model assets" and fetched only voice conversion's two — and
the honest form of a cross-engine fetch is naming the engine whose gigabytes
are being spent: `voice stt download`.

### Where downloaded weights land
Split by **who reads the file**, because the two kinds have opposite lifetimes:

- **Inference assets** (ContentVec, RMVPE, Whisper, the prosody encoder,
  cnhubert, `s1*.ckpt`, `s2G*.pth`, Seed-VC's four networks) are shared across
  every run on the machine,
  so they go to a cache: `--cache-dir` → `{RVC,STT,TTS,SEEDVC}_CACHE_DIR` →
  `VOICE_CACHE_DIR` → `voice` under the XDG cache root, which is
  `$XDG_CACHE_HOME` when absolute and `~/.cache` otherwise. The resolved path is
  a **computed clap default**, so `-h` prints where this machine will actually
  put them rather than a placeholder. The last step is written as a *root* that
  `voice` is joined onto, not as two independent branches, because that is what
  makes `~/voice` unreachable — the toolkit's directory can only appear inside a
  cache, never beside the user's own folders — and a relative `XDG_CACHE_HOME`
  is ignored, since a CWD-relative cache is the exact failure this split exists
  to prevent.
- **Training warm-start bases** (`f0G48k.pth`, `f0D48k.pth`, `s2D*.pth`) go to
  `pretrained/` **inside that same cache**, stored flat under their upstream
  names rather than in the Hub's tree, so the directory can be read by eye and
  hand-populated by anyone who already has the weights. They are fetched only
  when a fine-tune asks for one, so a user who never trains never downloads
  them; `--no-pretrained` and `--resume` fetch nothing either.

**Nothing downloaded is ever written into an output directory.** An output
directory holds what a run *produced* — its checkpoint family, its `checkpoint/`
best family, its `.best.json`, and the dashboard's `train.log` — and nothing
else. The log used to go to the *current* directory on the reasoning that only
weights belong beside `-o`; it belongs with them, because it is the record of
the run that wrote them, and leaving it in whichever directory the user happened
to stand in meant two runs appended to one file. `-o` names a stem, so the
directory is its parent (`cli_kit::log_beside`) — that is one function rather
than two, so `rvc` and `tts` cannot drift.

**This is a correction of an earlier split, so do not restore it.** The bases
used to go to `pretrained/` beside the run's output, on the reasoning that they
"belong to one experiment". They do not: a base is the published upstream file,
byte for byte, identical for every voice ever trained on the machine — the same
kind of read-only shared input as ContentVec or Whisper. The old rule made
training *n* voices download the same 219 MB *n* times, and it made the split
turn on "who reads it" (inference vs training) when the property that actually
matters is **whether the file is reusable**. It is, so it is cached.

The reason there is a rule at all is that before it there was none, only a
per-engine convention, and the conventions had already diverged — the cache was
`$RVC_CACHE_DIR` / `~/.cache/rvc` no matter which binary asked. `--work-dir` was
a third notion of "where things go" on top of the cache and the output
directory; it is gone, and the working directory is the current directory, like
any other Unix tool.

Training also **refuses to overwrite an existing output `.safetensors` unless
`-y` is passed.** A voice is hours of GPU time and the corpus that produced it
may be gone; a re-run with the same `-o` is far more often a mistake than an
intent.

### Naming conventions
Three tiers, and the name says which tier a crate is in:

- **`*-kit`** — shared plumbing with no model and no engine knowledge, safe for
  anything to depend on: `burn-kit` (devices, checkpoints), `audio-kit` (ffmpeg
  I/O, the slicer), `hub-kit` (downloads and the cache), `cli-kit` (logging,
  completions, `--backend`/`--device`), `preprocess-kit` (the corpus slicer as a
  subcommand), `rpath-kit` (a build-dependency: where a binary looks for the
  libraries it links).
- **`burn-*`** — one network each, named after the **model** (`burn-rvc` reads
  like `burn_dinov3`), holding no app dependencies and naming no compute backend.
- **`<engine>-core` / `<engine>-cli`** — one engine each, all the same shape:
  `rvc-core`+`rvc-train`+`rvc-cli`, `stt-core`+`stt-cli`,
  `tts-core`+`tts-train`+`tts-cli`, `seedvc-core`+`seedvc-cli` (no `-train`,
  and its absence is the engine's defining property rather than a gap).

**`voice-` is reserved for the top.** It marks the integration, so a crate that
an engine depends on must never be named `voice-*` — that is why the shared
crates are `*-kit`. `voice-cli` is the only `voice-*` crate.

### Where documentation goes
Nine places, and putting a paragraph in the wrong one is exactly how `README.md`
once grew ninety lines of engine manual:

- **`README.md`** is an *index*: what `voice` is, the pipeline, which binary to
  install, one build block, links out. No engine documentation, ever — if a
  passage names a flag, it belongs in an engine README.
- **`docs/setup.md`** is everything that is true of all five binaries at once:
  ffmpeg, ORT and LibTorch, the build, the `--backend`/`--device` table, and
  where downloaded models land. It exists because that material was previously
  written once in `crates/rvc-cli/README.md` and linked from the other two — which
  reads as `stt` depending on `rvc`, obliges a reader who only wants `stt` to
  open the voice-conversion manual, and drifted anyway. **Anything an engine
  README would have to say identically belongs here instead.**
- **`crates/<engine>-cli/README.md`** is that engine's manual and only that:
  every flag, which weights it fetches from where, how to drive its training. It
  links to `docs/setup.md` and `docs/training.md` rather than repeating them.
- **`docs/training.md`** is to the trainers what `setup.md` is to the binaries:
  everything true of every training loop at once — the shared VITS objective,
  what is in `train-kit` and what is deliberately *not*, warm-start, the
  checkpoint family, devices and the multi-device plan. It replaced
  `crates/rvc-train/ARCHITECTURE.md`, which described one engine's loop from
  inside that engine's crate, and so had no place to put the two facts that
  matter most: that `rvc-train` and `tts-train`'s `s2` are the *same* loop over
  the same losses, and that `s1` is deliberately not. A per-crate document
  cannot say what two crates share. **A flag's default belongs in the engine
  README, not here** — this page explains the mechanism, the README drives it.
- **`docs/realtime.md`** is everything true of driving the engines *live*: the
  sample rate at each end of a pipe, ffmpeg capture and playback, turning a
  filter into a virtual microphone with `module-pipe-source`, and an honest
  per-engine latency budget. It exists because the `ffplay` one-liners had been
  copied into six files with three different output rates between them, and
  because "can `rvc` be a virtual mic" had a working answer — yes, with no code
  from us — that lived nowhere. Same division as `setup.md`: **a measurement
  belongs here, a flag's default belongs in the engine README.** Recipes it
  could not run are labelled untested *in the page*, not only in a PR body.
- **`docs/roadmap.md`** is what is *not* built and what each thing would cost.
  Before it, "not started" existed only as a cell in `README.md`'s engine table,
  which can record that `translate` is missing but not what blocks it. The
  division against **this** file is the tense: a decision already **taken** and
  the trap it avoids goes here, a decision still **open** goes there. Every
  entry names what actually blocks it, because the value is entirely in the
  constraint rather than in the wish.
- **`docs/*.typ`** are the long-form architecture papers, one per network —
  `rvc-architecture`, `gptsovits-architecture`, `whisper-architecture`,
  `seedvc-architecture` — for
  *reviewing* a port rather than using it: what each block computes, what every
  loss term is for, why the training loop has the shape it does. Typst sources
  with the rendered PDF committed beside them, so reading needs no toolchain;
  rebuild with `typst compile docs/<name>.typ` and commit both.
- **`export/README.md`** is the exporter's own manual, and it is a place rather
  than a footnote because it is the only documentation of the *other* side of a
  port: which graphs each model is split into and why, what to pass, and the
  cross-runtime agreement numbers that say the mirror is faithful. It is the
  only page describing a tool that **no user, no test and no training run ever
  invokes**, which is why its material cannot be folded into `docs/setup.md`.
- **this file** is the ninth: why a decision was made and which trap it avoids.
  A fact that would be equally true of any VITS repo belongs in a `.typ` paper;
  a fact that will bite whoever edits this code next belongs here.

That count has been wrong before — the closing bullet said "the fifth" while the
list held six — so **adding a page means editing the opening line and the
closing bullet in the same commit**, not only inserting a bullet.

## Build / run / verify

```sh
# Neither ONNX Runtime nor LibTorch is bundled or downloaded — point at your own:
export ORT_DYLIB_PATH=/usr/lib/libonnxruntime.so
export LIBTORCH=$PWD/libtorch   # must be exactly 2.9.0 (what tch 0.22 targets)

cargo build                     # all five binaries: `rvc`, `stt`, `tts`, `seedvc`, `voice`
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

# There is no CI, so those two plus the tests are the whole gate — nothing else
# will catch it. `--workspace` builds every member with default features, which
# includes `tch`, so LibTorch has to be loadable at run time as well as linked:
LD_LIBRARY_PATH=$PWD/libtorch/lib cargo test --workspace
cargo test -p text-kit                      # g2p, and the largest suite by far
cargo test -p seedvc-core streaming_matches_the_batch_path   # one test by name
# The one `#[ignore]`d test is `rvc-core`'s `model_probe`: it wants
# ORT_DYLIB_PATH and model/audio paths from the environment, so it runs only
# when asked for by name with `-- --ignored`.

# One `--backend auto|onnx|cuda|tch|wgpu` on every binary (aliases burn/burn-cuda,
# libtorch/burn-tch, webgpu/burn-wgpu); `--device auto|cpu|gpu|gpu:N|mps|vulkan`
# (cuda/cuda:N spell gpu). Both default to auto. See docs/setup.md.
cargo run -p rvc-cli   --      convert -m models/voice.safetensors --model-sr 48000 -o out/ in.mp3
cargo run -p voice-cli -- rvc  convert -m models/voice.safetensors --model-sr 48000 -o out/ in.mp3

# No LibTorch on the machine? Drop it (then `--backend tch` errors cleanly):
cargo build --release --no-default-features --features cuda

# Weight coverage is checked by loading REAL pretrained weights, which is what
# `cargo test` cannot do — none of them can be committed. The unit tests pin
# arithmetic against hand-written references; these pin the module tree against
# the checkpoint. Each reports applied/missing/unused, and every non-zero
# `unused` in this list has a recorded reason — read it, don't glance at it:
cargo run -p burn-rvc --example load -- <path/to/f0G48k.pth>       # 560/0, 165/0
cargo run -p burn-rvc --example load -- contentvec <path/to/hubert_base>  # 210/0
cargo run -p burn-rmvpe --example load -- <path/to/rmvpe.pt>       # 623/0/118
cargo run -p burn-mdx --example load -- <path/to/MDX23C-8KFFT-InstVoc_HQ.ckpt>  # 319/0/0
cargo run -p burn-whisper --example load -- <path/to/model.safetensors>  # 587/0
cargo run -p burn-gptsovits --example load -- hubert <chinese-hubert-base/pytorch_model.bin>  # 210/0
cargo run -p burn-seedvc --example load -- <DiT_seed_v2_...pruned.pth>   # module by module
cargo run -p seedvc-core --features tch --example coverage -- --dit ... --campplus ...
cargo run -p burn-gptsovits --example keys -- --group <any checkpoint>   # what names to mirror
cargo run -p burn-seedvc  --example keys -- <any checkpoint>            # likewise

# The second kind of check, the one coverage cannot make — see **Verifying a
# port beyond weight coverage**. These exercise arithmetic on real audio, so
# each takes weights and a clip; the exact usage and the numbers to expect are
# in the example's own module doc, which is where they stay current:
cargo run -p burn-gptsovits --example reconstruct  # `s2` round-trip, energy r=0.91
cargo run -p burn-seedvc --example speaker    # pairwise cosine: same speaker vs not
cargo run -p burn-seedvc --example vocode     # BigVGAN: does the waveform track the mel
cargo run -p burn-seedvc --example content    # whisper-small + the length regulator
cargo run -p burn-rvc    --example infer      # one generator forward pass
cargo run -p burn-mdx --example separate --features tch  # stems partition the mix, 41.5 dB
cargo run -p burn-mdx --example separate --features tch -- --mixture <song.wav> <ckpt>
# ...the second form is the *real*-recording mode: no stems exist, so it reports
# no SI-SDR against a source and every reading is a contrast. 5-6.5 dB of bed
# removal, and the reason that is not a hedge is in the example's module doc.
cargo run -p seedvc-core --features tch --example convert  # the engine, end to end
cargo run -p seedvc-core --features tch --example stream   # the same, through the filter
cargo run -p rvc-core --features tch --example f0_runtimes # Burn vs ORT F0, median 0.005–1.40 Hz
# `--features tch` there is not decoration: `f0_runtimes` carries
# `required-features`, so it fails to build rather than compiling to nothing
# when the backend it exists to compare against is absent.
cargo run -p text-kit --example phonemize -- ja "私は東京にいます" <naist-jdic-dir>
# ...the `ja` form specifically: the dictionary is a run-time asset, so it is
# the one text-kit path `cargo test` cannot reach. `zh` and `en` are covered.
```

Porting references are cloned under `/.reference` (gitignored) and **read, never
run** — `openai/whisper`, and RVC-Project for `burn-rvc`, which is pinned to
`2.3.260718` and audited against `2.2.231006` (see **What the 2.3 audit found**).

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

**PyTorch's weight norm has two spellings on disk, and which one a checkpoint
uses says nothing about the model.** `torch.nn.utils.weight_norm` writes
`weight_g`/`weight_v`; `torch.nn.utils.parametrizations.weight_norm`, which
supersedes it, writes `parametrizations.weight.original0`/`original1` for the
same two tensors in the same order. Two HuBERT checkpoints of the *same*
architecture differ by exactly this — `chinese-hubert-base` is the old spelling,
RVC's ContentVec the new one — so `burn-hubert` accepts both. Getting it wrong is
silent: the two parameters simply go missing and the positional convolution keeps
its initialised values, which is a mis-scaled position embedding rather than an
error. `missing` in an `ApplyResult` is the only thing that shows it, which is
why a coverage report is *read* rather than glanced at (ContentVec first loaded
at 208/2). Accepting both spellings left `chinese-hubert-base` untouched, checked
rather than assumed: `burn-gptsovits`'s `examples/load -- hubert` still reports
210/0 afterwards.

**A layer upstream builds conditionally is an `Option`, always.** RMVPE's
`ConvBlockRes` creates its 1×1 residual shortcut only when the block changes
channel count, and probes it with `hasattr`, so 45 of the 56 blocks in `rmvpe.pt`
have no `shortcut.*` key at all. Modelling it as `Option<Conv2d<B>>` is what
makes that a load with **nothing** missing, rather than 90 absent parameters a
reader has to talk themselves out of. `burn-seedvc`'s CAMPPlus has the same shape
for the same reason, so this is the pattern and not a one-off.

**`torch.nn.GRU`'s storage is not Burn's, and the difference is silent.** PyTorch
concatenates the three gates into one `weight_ih_l0` of `[3 * hidden, input]`,
while Burn's `nn::Gru` holds three separate `GateController`s — and a
`KeyRemapper` renames keys, it cannot cut one tensor into three. So
`burn_rmvpe::gru::BiGru` carries PyTorch's own field names (`weight_ih_l0`,
`weight_hh_l0`, `bias_ih_l0`, `bias_hh_l0` and their `_reverse` twins) and its own
forward pass. Two details of that pass have to be right and neither is visible to
a coverage count: the gate order is **`r, z, n`**, and `bias_ih` and `bias_hh`
stay **separate**, because PyTorch computes
`n = tanh(W_in x + b_in + r * (W_hn h + b_hn))` — `b_hn` is applied *inside* the
reset gating, so fusing the biases moves it outside and changes the answer
wherever `r != 1`, which is everywhere. Either mistake loads at 100%, produces
finite output, and predicts the wrong pitch confidently. A scalar reference
written straight from PyTorch's documented equations pins both, and a second test
pins that the reverse direction actually travels backwards — running it forwards
and storing it at the same index passes every shape and finiteness check there is.

**tokio's `BufWriter` bypasses its own buffer for any single write at or above
capacity (8 KiB)**, so a filter that writes large chunks and forgets to flush
*looks* like it streams: at realistic sample rates most of each chunk goes
straight out, and only the sub-8-KiB tail is stranded. That is why `tts`'s
missing per-utterance flush went unnoticed, and it is why **a latency test has to
use chunks smaller than the buffer to see the defect at all**. Measured on a
three-line script: at `--sr 16000` the before/after difference sat inside
run-to-run variance, while at `--sr 1600`, where a whole utterance fits the
buffer, it was unambiguous — two lines released in one lump before, one release
per utterance after, with byte totals identical either way.

**Two ways of driving a filter live look identical to a hung model, and neither
is one** (both measured, in `docs/realtime.md`). Writing into a pipe source with
**nothing capturing**: the FIFO holds about 64 KiB — a third of a second at
48 kHz mono `f32` — the sound server only drains it while some application is
recording, so the engine blocks until a consumer opens the device. A writer
feeding 10 s of audio was still blocked 60 s later. With **something capturing
but the engine behind**: there is no backpressure to apply, so the device
underruns and substitutes **silence**, and the delay therefore never grows — an
output that is quietly part silence rather than a glitch you can hear. `rvc`'s
model load is the most visible case, 18 s of `underrun 0 < 8192` before the first
sample. Into a *pipe* the same shortfall shows up as the opposite symptom, an
unbounded delay, because the reader does apply backpressure.

Requires **ffmpeg 8.1** dev libraries (and the `ffmpeg` binary for the realtime
filter examples, which is what captures and plays PCM at either end of the pipe).
A system package needs no configuration; `FFMPEG_DIR` names your own build at
**build time**, exactly as `LIBTORCH` does and with the same run-time search
afterwards, and
`crates/audio-kit/build.rs` refuses a build that has an unpacked `./ffmpeg` at
the project root without naming it — `ffmpeg-sys-next` would not look there, and
its pkg-config failure never mentions the directory sitting in front of you.
ContentVec + RMVPE auto-download from Hugging Face in whichever of the two
weight formats the chosen backend reads (the `download` subcommand prefetches
them, and takes `--backend` so it knows which). Only 48 kHz is supported today.

### Exit 134 when an ORT session drops (RTX 2060, accepted)
Dropping an ONNX Runtime session on the CUDA execution provider aborts with
glibc's "corrupted double-linked list" on **this maintainer's RTX 2060**. It was
worked around by leaking every session (`ManuallyDrop`, `std::mem::forget`).
**That workaround has been removed deliberately and must not be reinstated** —
it obliged every present and future call site, including every error path, to
remember a leak that buys nothing on any other machine.

Measured on the affected machine, `dev` with the workaround against the same
tree without it:

| | workaround in place | removed |
|---|---|---|
| `--backend onnx`, export present | 0, 0, 0 | **134 on 5 of 7 runs** |
| `--backend onnx`, no export (early return) | 1, 1 | 134, 134 |
| `--backend tch` + ONNX prosody encoder | 0, 0 | 134, 134, 0 |

Four things make it easy to misdiagnose, which is the reason for the numbers:

- **It is intermittent, at roughly 70–80%.** A single trial can exit 0 and look
  fixed; the first run after removal did exactly that. Never conclude anything
  here from one invocation.
- **It is not the prosody encoder.** With prosody disabled, so `OnnxEngine`'s
  four sessions are the only ORT sessions in the process, 3 of 3 aborted.
- **It is not confined to ONNX as the generator.** `--backend tch` with the ORT
  prosody encoder aborts too, so it tracks the session, not the backend choice.
- **The audio is unaffected** — byte-identical 1.56 s output that `stt`
  transcribes correctly. Only the exit code differs.

The user accepts exit 134 on this hardware. Treat it as a **known local defect
with a recorded measurement**, not as a rule about how ORT sessions must be
handled anywhere else; the old framing as a general rule is what made the leak
spread. If it shows up elsewhere, the thing to record is which driver.

## Architecture

Data crosses every crate boundary as **mono `f32`**, and the whole conversion path
is `futures::Stream`-in → `futures::Stream`-out, which is why a bare `rvc` is a plain
Unix filter (raw f32le PCM stdin→stdout) and batch `convert` is a thin wrapper.

| crate | role |
|-------|------|
| `burn-kit` | Burn plumbing with no model knowledge: `--device` resolution and checkpoint loading, shared by every network crate |
| `audio-kit` | ffmpeg decode/resample + WAV/raw-PCM I/O, all as `futures::Stream<f32>` |
| `rvc-core` | the voice-conversion pipeline: `FeatureExtractor` (ContentVec + RMVPE), coarse-pitch/upsample/pitch-shift DSP, streaming `Converter` (block/overlap with an **overlapping** crossfade — consecutive kept blocks share `xf_out` output samples so the blend adds, never deletes, audio), an optional post de-hiss stage (`denoise.rs`, `--denoise`), and **all three** generator backends (ort, Burn/LibTorch, Burn/CubeCL) behind one `Generator` trait. The two *feature* models have the same run-time choice behind `ContentEncoder`/`PitchEstimator` (`analysis.rs`, implemented in `encoder.rs`/`f0.rs` for ORT and `burn_features.rs` for Burn), picked independently of each other and of the generator |
| `burn-vits` | the VITS blocks RVC and GPT-SoVITS share (both descend from the same source, which is why their `state_dict` names line up): attention stack, `Wn`, flow, posterior encoder, `ResBlock1`, weight-norm convs, discriminators, the family's losses, the differentiable STFT |
| `burn-rvc` | what is RVC's alone: `SourceModule` (NSF), the 768-dim `TextEncoder`, `GeneratorNsf`, the synthesizer wiring; re-exports `burn-vits` so it still reads as one model. Also `ContentVec` — RVC's *readout* of `burn-hubert` and nothing more, since the network is shared. **RVC v2 takes the final (12th) encoder layer directly** where v1 took layer 9 through `final_proj`, so that head sits in the checkpoint wired to nothing, and `hubert_base/config.json` is `HubertConfig::chinese_base()` field for field (pinned as a constant rather than parsed, so a disagreeing checkpoint fails as a shape mismatch). `examples/load -- contentvec <hubert_base>` reports 210/0 |
| `burn-hubert` | the HuBERT SSL encoder, its own crate because **two engines read it**: GPT-SoVITS calls it cnhubert, and RVC's ContentVec is the same architecture with other weights. `hidden_states` returns every layer rather than only the last, which is what makes a variant that reads a different layer a choice of index instead of a second port. Lifted out of `burn-gptsovits` with no field renamed, and that extraction is the cleanest proof on record of **Moving a module between crates is free** — the load example still reports 210/0 afterwards. It carries no `cuda`/`tch` features, because those exist to give a crate's *examples* a backend and this network's coverage harness stays `burn-gptsovits`'s |
| `burn-rmvpe` | the RMVPE pitch network, upstream's `E2E(4, 1, (2, 2))`: a five-level U-net, a `Conv2d(16 → 3, 3×3)` head, one bidirectional GRU (384 → 256 each way) and `Linear(512, 360)`. `[batch, 128, T]` log-mel in, `[batch, T, 360]` cents salience out — the mel front end (`rvc-core`'s `mel.rs`) and the salience→Hz decode (`dsp::rmvpe_decode`) stay in `rvc-core` so both runtimes share them, rather than giving the two backends a chance to disagree about something neither computes. `rmvpe.pt` loads at **623/0/118**, the unused being one `num_batches_tracked` per `BatchNorm`. Aligning the frame count to a multiple of 32 is `forward`'s job, not the caller's |
| `burn-mdx` | MDX23C (TFC-TDF-UNet v3), the source-separation network UVR ships: a complex STFT front end, five TFC-TDF U-net levels over a subband-folded spectrum, and one waveform per stem. What lets a corpus recorded over music be cleaned before anything else touches it. `MDX23C-8KFFT-InstVoc_HQ.ckpt` loads at **319/0/0** — no unused at all, because the norms are `InstanceNorm2d` and so carry no running statistics and no `num_batches_tracked`. The **older MDX-Net v2 models are ONNX-only and deliberately not ported**; see the crate docs for why a Burn port of them cannot be verified |
| `burn-whisper` | the Whisper network (standalone Burn port); mirrors HF's `state_dict` layout so `openai/whisper-large-v3-turbo` loads unchanged |
| `burn-gptsovits` | the GPT-SoVITS network. `hubert` at 210/0, `quantizer` at 3/0, and `s2` complete at 773/0 (the 3 unused are the codebook's EMA training statistics). **`s2` is verified numerically, not just structurally**: `examples/reconstruct` round-trips real audio through cnhubert, the quantiser and the synthesizer, and the output tracks the source's energy envelope at r=0.91 against a chance baseline of 0.30. `t2s` (`s1`) is at 295/0. Every network of GPT-SoVITS is now ported; `tts-core`/`tts-cli` wire them into a working `tts`, and `tts-train` fine-tunes **both** stages — `s1` for delivery, `s2` for timbre. `SovitsPartial::forward_train` composes `enc_q` → `flow.forward` → random segment → `dec` and returns the five tensors the VITS losses need; the matching `s2D2333k.pth` discriminator loads at 111/0/0. `examples/keys` lists any checkpoint's tensors, which is the first thing to run against a new one |
| `rvc-train` | native Rust/Burn adversarial training loop (see `docs/training.md`) |
| `hub-kit` | auto-download every engine's assets from Hugging Face |
| `rvc-cli` | lib **and** the `rvc` binary (clap): the bare invocation streams, plus `convert`, `train`, `preprocess`, `download`, `completions` |
| `stt-core` | speech recognition: Whisper log-mel front-end, BPE vocabulary, KV-cached greedy decode, segmentation via `audio-kit`'s slicer, and **two runtimes** (native Burn, ONNX Runtime) behind one `Engine` trait |
| `stt-cli` | lib **and** the `stt` binary |
| `tts-core` | speech synthesis: reference analysis, `s1` sampling with a KV cache, `s2` decode, and the ONNX prosody encoder behind a trait. **Two runtimes**, the same shape `stt-core` uses: `Engine` is the whole boundary, so `Synthesizer` is not generic and the backend is a constructor call rather than a type parameter. `--backend onnx` runs the whole stack — cnhubert, the quantiser, `ref_enc`, `s1` and `s2` — off four exported graphs, and `auto` picks it when the model directory holds an export |
| `tts-train` | fine-tuning GPT-SoVITS. `s1` is plain next-token cross-entropy over `T2s::forward_prompt_all` — one model, one optimizer, one loss, so unlike `rvc-train` the number means something on its own. `s2` is the other half: an adversarial VITS loop over `burn-vits`'s shared discriminators, inheriting `rvc-train`'s loss family (mel-L1 ×45, KL ×1, feature matching ×2, LSGAN) rather than inventing one, with GPT-SoVITS's five discriminator periods `[2,3,5,7,11]` against RVC's eight. Verified on 13 clips: mel falls 26.6 → 18.2 over two epochs on GPU and on CPU alike. `--stage s1|s2|both` prepares the corpus exactly once — preparation is the expensive half — and each stage writes its own checkpoint family. A corpus is `<stem>.wav` + `<stem>.txt` pairs, and `stt` is how the transcripts get written |
| `tts-cli` | lib **and** the `tts` binary |
| `burn-seedvc` | the Seed-VC network: the diffusion transformer and its flow-matching sampler, the length regulator, CAMPPlus, BigVGAN. **The crate the workspace's GPL-3.0 comes from** — see its own section below |
| `seedvc-core` | zero-shot voice conversion: `reference` (one clip → timbre vector + mel prefix + length-regulated content), `convert` (the chunk arithmetic and the equal-power crossfade), the streaming `Converter`, and the Burn backends behind one `Model` trait — the shape `stt-core` and `tts-core` use, so the backend is a constructor call rather than a type parameter. **No `-train` sibling**, and the `onnx` feature adds an `OnnxModel` reading the six graphs `export/export_seedvc.py` writes, behind `--backend onnx --onnx <dir>` |
| `seedvc-cli` | lib **and** the `seedvc` binary: the bare invocation streams, plus `convert`, `download`, `completions` |
| `cli-kit` | logging, shell completions, and the shared `--backend`/`--device`/`--cache-dir` flags — one enum and one alias set for all five binaries, so the spellings cannot drift apart again |
| `preprocess-kit` | the `preprocess` subcommand `rvc` and `tts` both expose: decode, slice on silence, write `<stem>_<NNN>.wav`. One definition, so the flags and the slicing cannot differ between the two engines |
| `train-kit` | training scaffolding with no model knowledge: `Checkpoint`, `ema_update`, `accumulate`, `materialize`, `Dashboard`. Generic over the module trained, so a GAN and a cross-entropy loop share it |
| `rpath-kit` | a **build-dependency**, not a runtime one: where each binary's `build.rs` gets the loader search order for the two linked libraries, ffmpeg and LibTorch |
| `text-kit` | grapheme-to-phoneme: script-based language splitting, Mandarin g2p (jieba + pinyin + opencpop + tone sandhi), and GPT-SoVITS's 732-symbol table. English g2p is an embedded CMUdict over upstream's deterministic cascade; Mandarin polyphones come from `pypinyin`'s own 47k phrase dictionary; Japanese is `jpreprocess` (a pure-Rust OpenJTalk rewrite) with the dictionary supplied by the caller. Pure Rust, no ML, no backend — so it is fully testable without weights |
| `voice-cli` | the `voice` binary: `rvc-cli`, `stt-cli`, `tts-cli` and `seedvc-cli` nested as `voice rvc …`, `voice stt`, `voice tts` and `voice seedvc` |

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
realtime for the streaming filter; `--backend tch` is ~9x faster per file on an
RTX 2060 and is what `auto` picks for `.safetensors` weights.

`--backend` is parsed **once**, by `cli_kit::Backend`, with one alias set for
every binary (`burn`/`burn-cuda` → `cuda`, `libtorch`/`burn-tch` → `tch`,
`webgpu`/`burn-wgpu` → `wgpu`). It used to be four separate enums, and they had
drifted: `rvc` accepted a bare `burn` where `stt` insisted on `burn-cuda`, which
is the kind of difference nobody decides on and everybody has to learn. Adding
an engine means reusing that enum, never declaring another one beside it.

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
Every `*-cli` crate's `build.rs` bakes the linked libraries' search paths into
its binary as a `RUNPATH`; without it a missing `libtorch.so` — or
`libavcodec.so` — aborts in `ld.so` before `main`, on every subcommand,
including ones that touch neither.

**The call is per-executable; the search order is not.** An rpath is a property
of one linked binary, so each `build.rs` still has to emit its own, but the four
of them were byte-identical copies of the same 40 lines, and ffmpeg would have
made that eighty. The order now lives once in `rpath-kit`, a **build-dependency**
with no runtime code: for each of `libtorch` and `ffmpeg`, `<name>/lib` relative
to the **working directory**, then `$ORIGIN/<name>/lib` and
`$ORIGIN/../<name>/lib` relative to the **binary**, then whatever `ld.so.cache`
knows. Dropping a self-contained tree beside a binary therefore works with no
environment at all, and a distribution's own package keeps working untouched.
`ffmpeg-sys-next` and `torch-sys` emit only a *link* search path, which the
loader never reads — that is why this is not free.

**Every entry is relative, and that is the rule to keep.** `LIBTORCH` and
`FFMPEG_DIR` say where to *link* against and nothing more; the build machine's
absolute paths used to be baked in front of the relative ones, which meant `ldd`
on a user's machine reported the maintainer's directory layout, and a build-time
environment decided a run-time answer. It also cannot be what a user wants:
these libraries are linked, so `ld.so` resolves them before `main` and no
variable of ours could be read in time. **The run-time variable is
`LD_LIBRARY_PATH`**, which glibc consults *before* `DT_RUNPATH` — so it
overrides all of this already and is the documented escape hatch. The cost of
dropping the absolute entry is that running a binary from a directory holding
neither `./libtorch` nor `./ffmpeg` needs it; `cargo test` already did.

The LibTorch entries are gated on the `tch` feature; the ffmpeg ones never are,
because every engine decodes audio. **A binary that happens not to decode pays
nothing for them** — it references no ffmpeg symbol, so it has no `NEEDED` entry
to resolve and the search path is simply never consulted. That is what lets the rule stay "always emit them" rather than
tracking which engine currently decodes: `stt` used to read PCM and never open a
container, and gaining `convert` made it link ffmpeg with no build change at all.

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
  ONNX Runtime as well — `alongside` Burn, never instead of it. Seed-VC now
  belongs in the same sentence, the third engine to reach the target in full:
  every model either of them uses runs either way, and tuning stays on Burn
  wherever a fine-tune exists.
- Seed-VC has an ONNX path now, and it was the exporter's work rather than a
  change of heart. **`export/export_seedvc.py` writes the six graphs — `content`,
  `style`, `mel`, `regulator`, `dit`, `bigvgan` — and `seedvc-core`'s `OnnxModel`
  runs them behind `--backend onnx --onnx <dir>`.** A Seed-VC checkpoint is four
  files rather than one, so `auto` cannot settle the runtime from a `-m` the way
  `rvc` does from a `.onnx`: it still resolves by hardware unless `--onnx` names
  a bundle, the one artefact on disk that could decide it. `--backend onnx` with
  no `--onnx` is refused with a reason naming the missing directory — not "not
  supported", which would send a user hunting for a feature flag that cannot
  exist. **The mirror of that refusal matters as much and was the one that used
  to be silent**: `--onnx` beside `--backend cuda|tch|wgpu` asks for two runtimes
  at once, and the loader took the graphs regardless, discarding a backend the
  user had named out loud. Both refusals live in `seedvc_core::backend::resolve`
  so the fetch never happens, and `--onnx` alone still selects ONNX Runtime.

**This reverses an earlier position on purpose, so do not "restore" it.** The old
plan was to port `rvc-core`'s ContentVec and RMVPE to Burn *in order to* drop
ONNX Runtime from the toolkit entirely and take `ORT_DYLIB_PATH` out of setup.
Backend choice turned out to be worth more than one environment variable: people
deploy where they deploy, and on plenty of targets ORT is the only runtime
available. So the port happened as an **addition**, and ORT stays a supported
target for both models rather than being demoted.

**That port has landed, and the prediction it replaces was right about the
shape.** ContentVec is a HuBERT variant, so `burn-gptsovits`'s `hubert.rs` did
cover its architecture, and it was lifted into `burn-hubert` first because two
engines now share it — the "anything two of them need moves to a neutral crate
first" rule applied *before* the second reader arrived rather than after. RMVPE
needed a network of its own (`burn-rmvpe`). `rvc` therefore joins `tts` in
reaching the target: every model it runs runs either way, chosen at run time.

Four things about that choice are worth having written down:

- **The two feature models choose separately from the generator and from each
  other.** `--content-vec-backend` and `--rmvpe-backend` override `--backend`
  per model and default to it, because a conversion is three models and only one
  of them was ever a choice. They default to the **already-resolved** generator
  backend, never the raw `--backend`: resolving `auto` twice would let a `.onnx`
  generator settle on ORT while its feature models independently re-derived
  `auto` from the hardware. `--rmvpe-backend auto` therefore means *inherit*,
  not *re-derive* — `Backend::Auto` is not a runtime and would reach
  `weight_format` matching no arm.
- **A backend decides which *file* is downloaded, and that mapping lives in
  `rvc-cli` (`args::weight_format`) because it is the only crate entitled to
  know both halves.** `cli-kit` owns `Backend` and states that it knows nothing
  about weight formats; `hub-kit` owns `WeightFormat` and must not depend on
  `cli-kit`. RMVPE's two formats are a one-word filename change inside the same
  first-party `lj1995/VoiceConversionWebUI` repo (`rmvpe.onnx` / `rmvpe.pt`), so
  the torch arm was nearly free — but ContentVec's torch form is a **directory**
  (`hubert_base/` = weights, config, preprocessor) where the ONNX one is a single
  community-mirrored file, so `fetch_contentvec` follows `fetch_whisper`'s
  multi-file shape. Only the engine in the middle knows that asymmetry.
  `rvc download` gained `--backend` for the same reason: with no `-m` to inspect
  it could only guess, it used to guess ONNX, and prefetching the wrong format
  leaves the first real run downloading anyway.
- **An `.onnx` generator is the one combination that cannot be mixed, and
  `build_converter` refuses it rather than composing something.** `RvcModel` is
  *one fused pipeline*, built from a single `RvcConfig`/`ModelPaths` naming all
  three graphs at once, so an ONNX generator with a LibTorch RMVPE cannot be
  described to it — and the generic path cannot host it either, because
  `Generator::convert_segment` takes raw 16 kHz audio and the only ONNX
  implementor of that trait *is* the fused `RvcModel`. Every constructor that
  accepts a prebuilt `FeatureExtractor` is a Burn one, so the mix has no third
  place to go. It is therefore an `ensure!` naming the offending flag, **before
  any download**.

  **This is a correction, so do not restore the old shape.** The early return
  used to test that all three models agreed on ONNX and let a disagreement fall
  through to the generic path — where the match arm is
  `Backend::Onnx => unreachable!()`, so `-m voice.onnx --rmvpe-backend tch`
  panicked two downloads and two model loads after the flag that caused it was
  parsed. Widening the condition back only moves that panic; deleting the check
  instead forces both feature models onto ORT and silently discards whatever the
  two override flags said. A `.safetensors` generator still mixes freely, which
  is what the flags are for.
- **`rvc train`'s feature backends default to `onnx`, not to `--backend`, and the
  reason is not that ORT cannot train.** Feature extraction is not training — it
  runs once over the corpus before the loop starts. It is that `rvc-train` builds
  its own `FeatureExtractor` from two paths and that constructor is ONNX-only, so
  inheriting `--backend tch` would fetch `rmvpe.pt`, hand it to an ORT session
  builder and break the recommended training command. A Burn backend named there
  is an **error with a reason**, never a quiet downgrade — somebody who asked for
  it must not come away believing the corpus was analysed on Burn. When the
  trainer takes a prebuilt extractor, the exception goes away.

**The numbers are the point, because coverage would not have caught a wrong gate
order.** ContentVec on Burn against ContentVec on ORT: per-frame cosine
**1.000000** as both mean and minimum, with identical frame counts — where
comparing frame *i* against frame *i + T/2* gives 0.004, so that is agreement and
not a degenerate metric. (`examples/load -- contentvec` is 210/0/57, the unused
being 54 LayerNorms counted under Burn's `gamma`/`beta`, v1's `final_proj`, and
SpecAugment's mask token.) RMVPE, network against network on one shared mel:
**0.70 Hz** mean absolute F0 difference over 406 jointly voiced frames at
correlation **0.9932–0.9996**.

**Driven through `FeatureExtractor`, read the median and not the mean** — this is
the one number here that a favourable sample can flatter, and it did.
`f0_runtimes` over eight clips of this repository's own breathy close-mic
corpus: median absolute difference **0.005–1.40 Hz**, sub-hertz on seven of
eight, while the *mean* over the same frames spans **0.49–24.4 Hz** and
correlation drops to 0.31. Both describe the same contours, because the
disagreement is a handful of frames rather than a drift — one clip has a median
of 0.0047 Hz across 502 voiced frames and a mean of 4.53, which is one frame in
five hundred. `max |Δ|` prints both readings and their ratio to identify them,
and on half the clips it is **an octave** (0.42–0.45, or 2.0): where the salience
map has two comparable peaks an octave apart, whichever is fractionally higher
wins, so a perturbation far too small to move a confident frame flips an
ambiguous one outright. An earlier "0.11–1.11 Hz, 0–6 voicing disagreements" here
was measured on clean clips; this toolkit exists for the material that is not.

**A swapped GRU gate or a fused bias is not content-dependent**, which is what
makes the median the check: it would not spare the confident frames. The
threshold is not a second effect either — both runtimes use the same
`F0_THRESHOLD` of 0.03, so the 7–87 voicing disagreements are the same
perturbation at that boundary.

**That residual is a known difference and not a defect to reconcile.** The U-net
wants a multiple of 32 frames; `Rmvpe::forward` pads with **zeros**, which is
upstream's `F.pad(mode="constant")` and therefore what the published weights were
actually run with, while `rvc-core`'s `f0.rs` reflect-pads by hand for the ONNX
path. Both trim afterwards, so the two can only differ over the last ≤ 31 frames
— except that the GRU's reverse half carries a little of it back across the whole
clip, which is why the disagreement is not confined to the tail.
`crates/rvc-core/examples/f0_runtimes` is the check.

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

### `seedvc-core` depends on `cli-kit`, and that is allowed
It is the only `*-core` that does, which reads as a layering slip and is not one.
`seedvc_core::load` erases the Burn backend behind `Box<dyn Model>`, so it has to
name a backend, and the rule above is that there is **one** `--backend` enum for
the whole workspace — declaring a second one in `seedvc-core` to avoid the
dependency is the thing explicitly forbidden. `cli-kit` is a `*-kit` crate,
listed as safe for anything to depend on, so the dependency is the sanctioned
half of that trade.

The other three engines differ only because their cores have no loader at all:
`stt-cli` and `tts-cli` build the `Engine` themselves. Moving `backend.rs` up to
`seedvc-cli` to match would strand `seedvc-core`'s three examples, which are the
crate's only harness against **real weights** — its unit tests run on synthetic
tensors and a stub `Model`, because none of the four checkpoints can be
committed — and which take `--backend` themselves.

### `seedvc convert` drives the batch path, not the streaming one
Both exist and both are live, which is deliberate. `crate::convert` knows a
file's length before the first chunk, so it balances the last chunk against the
one before it (`MIN_CHUNK_FRAMES`); `Converter` cannot, having already emitted
the audio it would need to hand back. So the file command uses the file path and
the filter uses the stream, and `streaming_matches_the_batch_path` pins them to
each other.

**That equivalence test is the reason neither may be deleted as a duplicate.**
It is what caught `flush` rounding a frame count where the batch path floors —
the frame count sets the noise buffer's width, so one frame re-indexes every
value and the sampler integrates a different field entirely.

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
- `burn-mdx` — `examples/separate` mixes a known voice with a known
  instrumental bed and reads three things, of which **the first is the one that
  proves the port**: the two stems sum back to the mixture at **41.5 dB**
  SI-SDR, and a solo source splits in *opposite directions* depending on which
  one went in (4.7 dB and 7.3 dB rejection). Neither survives a transposed
  U-net, a batch norm where an instance norm belongs or a scrambled stem axis,
  because nothing downstream re-imposes them. That synthetic mixture says
  nothing about **quality**, because it is out of distribution on both sides —
  every mixture-level SI-SDR in it sits within a decibel of doing nothing.

  **The quality question has since been measured on a real mixture, and the two
  answers must not be merged.** The user supplied a 19.8-minute stream — a
  streamer talking over somebody else's music — and `--mixture <file>` is the
  reference-free mode that reads it: no source exists, so no SI-SDR against one
  is reported, and every number is a contrast the mixture is measured under the
  same way. **It separates, modestly.** Across four excerpts the vocals stem's
  loud/quiet contrast comes out *above* the mixture's (16.8–25.7 against
  12.9–22.5 dB) while the instrumental stem's comes out *below* it (7.7–16.7),
  which is one stem following the intermittent speech and the other the
  continuous bed; the music removed from the vocals stem is **5–6.5 dB** where
  the bed is continuous, against the 15–20 dB this model reaches on a song. The
  partition holds at 34–37 dB across a whole file. Overlap-add seams are part of
  the drop from 41.5 and **not all of it** — the mono fold below has the same
  file and the same seams and reads 37.5 — so that reading is content-sensitive
  and a change in it is not on its own a regression.

  Three things about that measurement not to re-derive:

  - **Read the gaps at 250 ms, not at one second.** A between-sentence gap is a
    few hundred milliseconds, so at a one-second window no "quiet" frame is
    speech-free and the verdict *inverts*: the same stems on the same 60 s gave
    2.1 dB of removal and a vocals contrast below the mixture's, which reads as
    a model that separated nothing, where 250 ms gives 6.5 dB and a contrast
    above it.
  - **The input is near-mono, and it is not the explanation.** The side channel
    sits 14–19 dB under the mid, so the stereo cue a stereo-native separator
    wants is mostly absent — but folding to true mono and re-running costs only
    1.4 dB (6.5 → 5.1). What is left is the material: a *speaking* voice is not
    a sung one. Which also means a mono corpus loses almost nothing.
  - **The practical gain is bigger than the decibels.** Transcribing the same
    60 s with `stt convert -l zh`, the mixture yields **5** segments — one of
    them 18.8 s of merged speech — and the vocals stem **14**, one per
    utterance, with the same words. `audio_kit`'s slicer cuts on silence and a
    continuous bed leaves none, so a corpus recorded behind music cannot be
    sliced into sentences at all until the bed comes off. One of the 14 is a
    clear Whisper hallucination in a newly-emptied gap and one is a 0.37 s
    fragment, so a consumer wants a duration floor. **Pin `--language`**: left to
    detect, the two files disagree and the comparison stops meaning anything.

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

The same table is what makes **English and Japanese** cheap to add: GPT-SoVITS
v2's symbol list already carries the ARPAbet phones *and* all 38 Japanese ones
(interleaved alphabetically, because upstream builds the table with
`sorted(set(...))`), so neither needs new indices. English `--language en`
needs no new indices
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

### Japanese, and the one asset `text-kit` does not embed
`--language ja` runs on [`jpreprocess`](https://crates.io/crates/jpreprocess)
(BSD-3), a **pure-Rust OpenJTalk rewrite** — so it costs no C and no Python, and
its `extract_fullcontext` emits the same HTS full-context labels that
`pyopenjtalk.make_label(run_frontend(…))` does. `japanese.rs` therefore ports
upstream's `pyopenjtalk_g2p_prosody` (espnet's, via `text/japanese.py`) rather
than inventing anything: `p3` per label, unvoiced vowels lowercased, `sil` →
`^`/`$` and `pau` → `_`, then `#`, `[` and `]` from `a1`,`a2`,`a3`,`f1` and the
**next** label's `a2`.

**`_numeric_feature_by_regex` returns `-50` when a field does not match, and that
sentinel is load-bearing** — the `a1 == 0` and `a2 != f1` comparisons only behave
at a sentence boundary because a missing field is a large negative rather than
`0` or an `Option`. Substituting either changes which frames get an accent mark.

**`#` is deliberately not in the 732-symbol table**, so it folds to `UNK` via
`symbols::id()`. That is not a gap being papered over: it is bit-for-bit what
upstream feeds the checkpoint, since `cleaner.py` maps any out-of-table phone to
`UNK` too. `^`/`$` never reach the table at all — upstream strips them with a
`[1:-1]`. Japanese yields `word2ph: None` like English, and upstream marks tones
and `word2ph` as unfinished on its own side.

**The dictionary is the exception to `text-kit`'s "no weights" rule, and it is
kept outside the crate to preserve it.** NAIST-JDic is 28.7 MB, so it is a
run-time asset fetched into the shared cache on the first Japanese run —
`JapaneseDict::open` takes the path, and `phonemize`/`phonemize_mixed` take an
`Option<&JapaneseDict>`. The label parsing and the prosody rules stay testable
against hand-written label strings with no dictionary at all, which is most of
what could go wrong. **Do not enable `jpreprocess`'s `naist-jdic` feature to
avoid the plumbing**: that feature's `build.rs` downloads during `cargo build`,
which is exactly the shape `crates/rvc-core/build.rs` exists to prevent for
LibTorch. Only `tokenizer` is on.

`examples/phonemize` exists because the dictionary path is the one thing
`cargo test` cannot cover.

**Bare Han follows the caller's language, and getting this wrong is silent.**
`segment.rs` decides Japanese by the presence of kana, which is right for mixed
text — but 東京, 日本語 and 人々 are valid in *either* language, and choosing
Chinese regardless meant `--language ja` phonemized them as Mandarin, with a
plausible phoneme sequence and no error anywhere. `default` now breaks that tie;
Chinese is still the answer for every caller that did not ask for Japanese. In
the same vein U+3005 (々) is **Han, not neutral** — upstream's
`_japanese_characters` includes it, and as a neutral it split 人々 in two.

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
(`Synthesizer::load_pytorch`, `crates/burn-rvc/src/synthesizer.rs`, on top of
`burn-kit`'s generic `load_pytorch_into`) remaps RVC's flat
`attn_layers`/`norm_layers_*` lists and the flow's even coupling indices onto the
module tree and upcasts fp16→fp32. Changing the module layout breaks warm-start
and the ONNX exporter — keep names/structure in step with the reference
(RVC-Project tag `2.3.260718`, `infer/module/{models,attentions,modules}.py`,
which 2.2 spelled `infer/lib/infer_pack/`).

### What the 2.3 audit found
**There is no RVC v3.** Upstream's latest is `2.3.260718` (21 July 2026), whose
notes say "Base model unchanged"; the v3 promise has sat unshipped since
`2.1.230814`. So the question is never "port v3", it is "did 2.3's *fixes* reach
the maths we implement". The whole of `models.py`, `attentions.py`, `modules.py`,
`commons.py`, `transforms.py`, `configs/v2/48k.json`, `rmvpe.py`, `losses.py`,
`mel_processing.py` and the inference pipeline were diffed against 2.2. **The
answer for inference is no, and the port is unchanged** — this section exists so
nobody has to re-derive that.

What 2.3 actually did to the network is **nothing**: the file moved from
`infer/lib/infer_pack/` to `infer/module/`, `TextEncoder256`/`TextEncoder768`
collapsed into one `TextEncoder(in_channels, …)`, `SynthesizerTrnMs768NSFsid`
became a subclass of the 256 variant, and every TorchScript annotation and
`__prepare_scriptable__` hook was deleted. **None of that moves a `state_dict`
key**, which is why our 560/0/0 and 165/0/0 still hold. `SineGen` and
`SourceModuleHnNSF` differ only by black reformatting, the noise scale is still
`0.66666`, `MultiPeriodDiscriminatorV2`'s periods are still `[2,3,5,7,11,17,23,37]`,
and `losses.py` is byte-identical. `infer` gained `skip_head`/`return_length`/
`return_length2` and the two generators gained `n_res`, but all default to `None`
and the batch path through them is what 2.2 computed.

Three changes *are* real, and each was deliberately not adopted:

- **F0 is now interpolated across unvoiced frames** — `uv = f0 == 0; f0[uv] =
  np.interp(…)` in both `infer/vc/pipeline.py::get_f0` and
  `train/dataset/extract_f0.py`, applied before the key shift so it moves both
  `pitchf` and the coarse pitch. Consequence: `SineGen._f02uv` never sees a zero,
  so the unvoiced branch that swaps harmonic excitation for noise-only **stops
  firing at all** — and that branch is precisely what renders breath and whisper.
  Adopting it would need a retrain, would invalidate every voice users have
  already trained, and works against the soft/breathy content this toolkit exists
  to preserve. `dsp::f0_to_coarse` keeps mapping 0 to bin 1 and `shift_pitch`
  keeps leaving zeros at zero, which is both internally consistent with our
  trainer and correct for the 2.2-era bases we warm-start from.
- **The mel front-end's floor dropped**: `clip_val` 1e-5 → 2e-6 (≈14 dB more
  range under the old floor) and the linear-spectrogram epsilon 1e-6 → 2e-7.
  Ours is already *finer* than either release on the linear side (1e-9) and still
  at 1e-5 on the mel. This is a training-objective knob — it touches no weight and
  no inference path — but `burn_vits::spectral` is shared with `tts-train`, where
  GPT-SoVITS upstream still uses 1e-5, so lowering it silently would move two
  engines' loss scales and invalidate every recorded number. **Worth trying for
  soft, close-mic material, as a measured change with both engines re-baselined,
  not as a fix.**
- **The long-file split search was fixed**: 2.2 accumulated *signed* samples
  (`audio_sum += audio_pad[i : i - window]`) and only then took `np.abs`, so a
  loud symmetric waveform sums to ≈0 and could be chosen as the quietest cut
  point; 2.3 sums `np.abs` first. A genuine bug, but it is upstream's
  chunk-the-long-file heuristic, which we do not have — `rvc-core`'s `Converter`
  splits on fixed blocks with an overlapping crossfade, and our corpus slicer is
  `slicer2.py`'s RMS algorithm, which 2.3 did not touch.

Also of note without being ours to copy: 2.3's real-time `infer` no longer runs
the flow on a bare truncated tail but gives it 24 frames of left context
(`flow_head = max(head - 24, 0)`) and drops them afterwards. That is upstream's
answer to the seam artefact our crossfade answers, so it is the place to look
first if block joins ever become audible. Everything else in the release —
UVR5→PyMSS separation, FCPE as a pitch option, CUDA graphs, single-GPU without
DDP, the WebUI rewrite — is packaging and never reaches a tensor we own.

### Seed-VC, and why the whole workspace is GPL-3.0 (`burn-seedvc`)
The fourth engine — `burn-seedvc` ported, `seedvc-core` and `seedvc-cli` wiring
it into a binary that `voice` nests. Seed-VC converts a voice **without training
on it** — a
1–30 s reference clip is the entire speaker specification, where `rvc` and
`tts` each want a fine-tune. That is why it sits beside them rather than
replacing them, and it is the answer to "is there something more advanced than
RVC v2": yes, but it is a different bargain, not a newer RVC.

**Upstream is GPL-3.0, and a port written from reading it is a derivative work**,
so the licence carries to everything that links `burn-seedvc`. The workspace was
relicensed from `MIT OR Apache-2.0` to `GPL-3.0-only` for exactly this reason —
one line, since every crate inherits `license.workspace = true`. `-only` rather
than `-or-later` because upstream ships the bare GPL-3.0 text with no "or any
later version" statement, and `-only` is the reading that is valid either way.
Releases made before that stay MIT for whoever holds a copy; the change applies
forward. **Anything that must stay permissive cannot depend on this crate.**

The target is the v1 `seed-uvit-whisper-small-wavenet` preset, chosen for one
reason that outweighs the rest: **its content encoder is `openai/whisper-small`,
which `burn-whisper` already loads.** The v2 CFM+AR pair needs an
ASTRAL-Quantization tokeniser ported first.

`DiT_seed_v2_uvit_whisper_small_wavenet_bigvgan_pruned.pth` is 302 tensors /
110M parameters, and `examples/load` prints where they live so a prefix nobody
claims is visibly a piece nobody ported. **Two of the six networks are in
nobody's checkpoint but their own** — the content encoder is whisper-small and
the vocoder is `nvidia/bigvgan_v2_22khz_80band_256x` — which is worth knowing
before hunting for their tensors in the wrong file.

Two traps found while porting, both of the same kind — *the checkpoint contains
more than the model runs*:

- **`net.style_encoder.*` (18 tensors) is dead weight.** Upstream's
  `build_model` assembles only `cfm` and `length_regulator`, so `load_checkpoint`
  reads straight past it; inference instead builds a **separate** CAMPPlus from
  `campplus_cn_common.bin`, and *that* is what conditions the transformer. The
  18 tensors are a fossil of the training-time model. They are ported anyway,
  because a subtree nobody claims is indistinguishable from one somebody forgot
  — but **wiring inference to them would feed the transformer a timbre vector
  Seed-VC was never conditioned on.** The real timbre encoder is `campplus.rs`,
  loading `campplus_cn_common.bin` from HF **`funasr/campplus`** (Apache-2.0,
  named verbatim in three of upstream's entry points) at 815/0/0.
- **`net.vq.*` is the same story**, and the length regulator's 2048-entry
  codebook is allocated and never indexed, because this preset sets
  `is_discrete: false`. Port faithfully, document what is live.

Also: `sampling_ratios: [1,1,1,1]` **is not a ratio.** Upstream reads only its
length — one conv stage per entry — so it means "four stages", not "rate
unchanged". The length regulator is in fact the only thing in the model that
changes the frame rate, 50 Hz from Whisper to ≈86.13 Hz for the mel.

**Three front ends, three rates, and none of them interchangeable.** Whisper's
content encoder eats **16 kHz** and its log-mel is *not* `burn_vits::Spectral` —
it centres its STFT, takes power rather than magnitude, and ends in `log10` with
a peak-relative floor. CAMPPlus eats a **Kaldi filterbank** at 16 kHz,
mean-normalised over time, and takes `[batch, frames, bins]` — the **opposite**
order to `StyleEncoder::forward`. Everything from the diffusion transformer
onward is **22.05 kHz**, where `Spectral` *is* exact, for BigVGAN as well as for
Seed-VC. Every one of those pairings has matching frame counts and 80 bands, so
substituting one for another runs happily and computes something else.

**That mean subtraction is upstream's *call site*, not its encoder**, and it is
the reason `fbank`'s `forward` does it rather than `campplus.rs`. A port written
by reading the model is faithful and still wrong: CAMPPlus's own module never
normalises, so the step lives in whatever code assembles its input, and
inference passes an already-normalised filterbank in. The failure is silent —
the embedding starts keying on the recording's channel rather than on the
speaker, which looks like a mediocre conversion and not like a bug. It was
dropped twice in one week while this engine was being built, which is why it now
sits inside the front end that cannot be used without it.

**The reference and the source share one window, and nothing warns.**
`max_context_window` is `22050 / 256 * 30` = **2580** mel frames — integer
division first, so it is not 2584 — and the reference's mel prefix comes out of
the same 2580 the source chunk does. A 25 s reference (the cap
`seedvc_core::reference::REFERENCE_SECONDS` applies) is 2153 of them and leaves
under 5 s per chunk; 5 s of reference leaves nearly 25. So **a longer reference
buys a saturating timbre vector and costs source per chunk.**
`seedvc_core::convert` states that arithmetic in its error rather than reporting
whatever symptom it produces, because trimming the clip is the only thing the
user can do about it, and the subtraction underflows rather than failing if
nobody checks it.

**The engine's `download` fetches from four repos, not one.** `tts` has a single
snapshot directory; here the checkpoint (`Plachta/Seed-VC`) holds only the
transformer and the length regulator, and the timbre encoder
(`funasr/campplus`), the vocoder (`nvidia/bigvgan_v2_22khz_80band_256x`, weights
*and* its `config.json`, from which the band count and upsampling rates are
read) and the content encoder (`openai/whisper-small`) are three other projects'
releases used unmodified. All four are inference assets, so all four go to the
shared cache and there is no `pretrained/` counterpart — a warm-start base is a
training input, and this engine has no training.

**`examples/load` now collides six ways.** `burn-rvc`, `burn-whisper`,
`burn-gptsovits`, `burn-seedvc`, `burn-rmvpe` and `burn-mdx` each have one, so cargo's
"output filename collision" warning names six crates rather than two. It is the
hazard the next section describes, not noise.

### The shared target directory is unsafe for concurrent worktrees
`target/debug/examples/<name>` is **not** hashed per worktree, so two checkouts
building an example of the same name overwrite each other's binary and
`cargo run --example` silently executes whichever landed last. Two workers hit
this independently while porting Seed-VC: one got a complete, plausible, *wrong*
coverage report out of it, and another had `cargo test` run a sibling's test
binary — a fifteen-test set including tests its own tree did not contain. The
same fingerprinting confusion also produces compile errors that contradict the
file on disk (a `pub use` reported missing when it is plainly there), where the
fix is `touch`, not debugging.

This is why cargo's "output filename collision" warning on the `load` and `keys`
examples is **not** the harmless noise it looks like. Working in parallel
checkouts means an isolated `CARGO_TARGET_DIR`, or copying the built binary out
and running the copy — which is where every coverage number in `burn-seedvc`'s
module docs comes from.

### The Python boundary (`export/`)
The only Python: a standalone `uv` project that converts a Burn `.safetensors` to
ONNX. `rvc_infer.py` is a **clean-room** torch reimplementation mirroring the
`burn-rvc` layout (it does NOT depend on the RVC-Project repo). Native Burn inference
needs no export — this is only for ONNX Runtime / cross-framework deploy.

It is the **only** direction that needs Python, and only because Burn reads ONNX
without writing it. Extending it to a new model means adding another clean-room
mirror of that network's Burn layout beside `rvc_infer.py` — `gptsovits_infer.py`
is the second, and `seedvc_infer/` is the third, a *directory* rather than a file
because it mirrors the 4800 Burn lines of `burn-seedvc` as six modules — never
importing the upstream project, and never adding a step a user has to run. The
Seed-VC mirror is the one this batch added, split across five parallel worktrees
on its way in: the five mirror PRs landed first, then the driver that assembles
them. `export_gptsovits.py` accepts *either* weight layout: an original
`.pth`/`.ckpt` through the same key remaps the Rust loaders apply, or a Burn
`.safetensors` from `tts train`. That second path is the point of the whole
exercise — it is how a voice fine-tuned here would reach ONNX Runtime.

**`reference.onnx` is built from the `s2` checkpoint too, and forgetting that is
the one real trap here.** It carries the quantiser and `ref_enc` — the prompt
tokens and the speaker vector — which live in the same file as the decoder. So
`--only s2 --s2 tuned.safetensors` writes a bundle whose decoder is the fine-tune
and whose front end is whatever was in the output directory. It loads, runs, and
sounds wrong, and nothing downstream can detect it — and `--only reference --s2
tuned.safetensors` is the same bundle mirrored, a tuned front end feeding a base
decoder. `export_gptsovits.py` now refuses `--s2` with any `--only` that names
one of the two and not the other.

**This is what an earlier entry here reported as a bug in the Burn-`.safetensors`
branch. That report was wrong and is retracted.** The branch is faithful; the
measurement that condemned it compared a tuned `s2.onnx` against a *base*
`reference.onnx`, so the two runtimes were decoding different prompts. Measured
again with the whole bundle exported from the same weights, at a fixed seed with
the caller-drawn noise held identical:

| weights exported from | ONNX vs Burn |
|---|---|
| `s2G2333k.pth` | RMS-diff/RMS **0.005**, log-spectrogram corr 0.99998 |
| a fine-tuned `.safetensors` | RMS-diff/RMS **0.00002**, corr **1.000000**, 0.00 dB |

Three checks that were run and should not be repeated from scratch: the Burn
`.safetensors` applies to `SovitsPartial` at **773/0/0**; every one of the 539
parameters `build_state_dict` produces from it is within 7% of the base model's,
as two epochs of fine-tuning should be; and the tensors it emits appear verbatim
in the exported graph. **The lesson is about the metric, not the weights** —
sample-wise RMS on a vocoder is phase-sensitive, and two runs of the *same* model
can differ hugely by it. Compare log-spectra or energy envelopes when asking
whether two graphs are the same model.

`s1` is emitted as **two** graphs, `s1_prompt` and `s1_step`, because a KV cache
cannot be a single static graph; the weights therefore appear twice on disk.
Sampling stays on the host, so a graph is a pure function of its inputs. The
numbers that make the port checkable are in `export/README.md`: both runtimes
sampled identical tokens from the same seed, agreeing to a **max absolute
difference of 7.9e-03** on RMS 5.2e-02 (correlation 0.99998), and `s1_prompt`
over a whole prompt agrees with `s1_prompt` + `s1_step` to 6.7e-06.

```sh
uv run --project export python export/export_rvc.py models/voice.safetensors models/voice.onnx
```

Exported graph contract (matches `rvc-core`):
`phone[1,T,768] f32, phone_lengths[1] i64, pitch[1,T] i64, pitchf[1,T] f32, ds[1] i64, rnd[1,192,T] f32 → audio[1,1,L] f32`.

## Training notes

How the loops actually work — the shared objective, `train-kit`, warm-start, the
checkpoint family, the multi-device plan — is `docs/training.md`. What follows is
only what will bite whoever edits them.

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
hop=480, 128 Slaney mels, center=False (`crates/burn-vits/src/spectral.rs`, shared
with `tts-train` — which is why its 1e-5 mel floor is a two-engine decision, see
**What the 2.3 audit found**). Target GPU
is 6 GB (RTX 2060) → small batch. Warm-start from `--pretrained-g/-d` is strongly
recommended on a small corpus.

**Preprocess first (`rvc preprocess`).** `sample_batch` draws random 0.48 s
windows uniformly across each corpus file, so raw recordings full of
between-sentence dead-air collapse the generator to silence. `rvc preprocess
raw/*.mp3 -o clips/` then `rvc train clips/*.wav ...` slices the corpus into
clean per-sentence clips first — it removes between-sentence dead-air while
**preserving soft, breathy, close-mic content** (energy is used only to find long
silent gaps, never to gate quiet-but-present sound). The shared slicer lives in
`crates/audio-kit/src/slice.rs` (`SliceOptions`, `slice`); the two tuning knobs
are `--silence-db` (energy floor; lower to keep the softest passages) and
`--min-silence` (how long a quiet gap must last to be a cut, so sentences are
never split). Training itself is unchanged — it just consumes the cleaned folder.
