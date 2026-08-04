# seedvc — zero-shot voice conversion

Convert one voice into another with **nothing to train**. `seedvc` is pure-Rust
[Seed-VC](https://github.com/Plachtaa/seed-vc): a frozen Whisper encoder carries
*what was said*, a diffusion transformer predicts a mel in the target's timbre,
and BigVGAN vocodes it. The target is specified by a **1–30 s recording**, handed
to `-r` on the command line, so any voice you can record is a voice you can
convert into a few seconds later.

**That is the whole difference from [`rvc`](../rvc-cli/README.md) and
[`tts`](../tts-cli/README.md), and it is a different bargain rather than a newer
one.** They spend hours of GPU time learning one voice and then render it very
well; this spends none and renders any voice adequately. There is no `train`
subcommand here and there will not be one.

Runtimes, drivers and ffmpeg: [`docs/setup.md`](../../docs/setup.md). What the
six networks compute:
[`docs/seedvc-architecture.pdf`](../../docs/seedvc-architecture.pdf). Build it
with `cargo build --release -p seedvc-cli`. Every command below is also
`voice seedvc …` — same code, same flags.

## Commands

| | |
|---|---|
| `seedvc -r <clip>` | **the bare invocation is the filter** — f32le mono PCM @16 kHz on stdin, converted PCM @22.05 kHz on stdout |
| `seedvc convert <files…> -r <clip>` | convert whole files, one `<stem>.wav` per input |
| `seedvc download` | prefetch the four networks a conversion fetches on its first run |
| `seedvc completions <shell>` | completion script for bash, zsh, fish, powershell or elvish |

Logs go to stderr, so stdout is only ever samples.

**`-r/--reference` is required by both conversion paths, but clap does not
enforce it** — the shared options are flattened beside `download` and
`completions`, which want no reference at all, so marking it required would
demand one of them too. A missing `-r` is therefore an error at the start of the
run rather than a usage message, and it says what a reference is for.

```sh
ffmpeg -v quiet -i take.mp3 -f f32le -ar 16000 -ac 1 - \
  | seedvc -r target-voice.wav \
  | ffplay -f f32le -ar 22050 -ac 1 -
```

**The output is 22.05 kHz**, not `rvc`'s 48 kHz — it is BigVGAN's rate, and
anything downstream in the pipe has to be told so.

## The reference clip

Anything ffmpeg opens, at any rate and channel count. It is decoded **twice**, at
16 kHz for the content encoder and the timbre encoder and at 22.05 kHz for the
mel, because it conditions the transformer three separate ways:

- a **timbre vector**, the clip through a Kaldi filterbank and CAMPPlus;
- a **mel prefix** the output is generated *after*, in context — the transformer
  is asked to continue the reference rather than to imitate it, and the prefix is
  sliced back off afterwards;
- its own **length-regulated content**, prepended to the source's, so what it is
  asked to continue is a prompt whose audio and words agree.

**Only the first of those saturates.** The timbre vector is a pooled average, so
past a few seconds more reference buys very little.

### Longer is not better, and this is the one number to know

The reference's mel prefix and the source share **one context window**. The
transformer sees 2580 mel frames at once — 30 s at 22.05 kHz over a hop of 256,
integer division first — and the source's share is what is left after the
reference:

```text
frames per chunk = 2580 − reference frames
```

At ≈86.13 frames a second that means a 5 s reference leaves nearly 25 s of source
per chunk, and a 25 s reference leaves under 5 s. Neither is wrong and nothing
warns: **a long reference silently buys shorter chunks.** A few seconds of clean,
representative speech is the useful setting.

Anything past **25 s is read and discarded** — upstream's own cap, applied at
both decode rates so the two halves of the reference cannot describe different
durations. That cap is also what keeps the window from ever being filled by the
reference alone: 25 s is 2153 of the 2580 frames, leaving 427. Should a reference
ever get past it, the refusal states this arithmetic rather than a symptom,
because trimming the clip is the only thing you could do about it.

The **source** has no length limit at all: chunking is the engine's job, and
consecutive chunks are joined with an equal-power crossfade. Each is an
independent generation, so they agree on timbre and content and on nothing about
phase — which is why the fades are cos²/sin², summing to exactly 1, rather than
linear.

## Flags

`seedvc --help` and `seedvc convert --help` list every flag with its default.
Three are worth understanding before you turn them.

**`--steps` (default 30) buys smoothness linearly and costs time linearly.** The
sampler is explicit Euler, which is first-order accurate: doubling the steps
roughly halves the error, and no more. Do not read the step count as if it bought
a Runge–Kutta rate. 4–10 steps is the fast end, 25–50 the polished one.

**`--guidance` (default 0.7) pushes each step *past* the conditioned prediction,
away from the unconditioned one.** Raise it to follow the reference harder, at
the cost of artefacts. Zero is not "no conditioning" — it is exactly the
conditioned velocity with no extrapolation, and because the unconditional pass is
then skipped entirely it **halves the work per step**. `--guidance 0 --steps 60`
and `--guidance 0.7 --steps 30` cost about the same.

`--length-adjust` scales the output's duration against the source's — above 1 is
slower, below is faster, and pitch is untouched, because the content is resampled
onto the new frame count rather than the waveform being stretched. Compressing
hard eventually puts more source behind each chunk than the content encoder's own
30 s window holds, which is refused in terms of the flag that caused it.

`--seed` fixes the sampler's noise, so a conversion repeats exactly. Every chunk
is an independent generation from fresh noise, which is why two runs of the same
file at different seeds differ audibly even though neither is wrong.

`--chunk` is the filter's stdin read size in samples and nothing more; it does
not change what the model is shown.

## Backends

One binary carries every backend it was built with, and `--backend` picks at run
time.

| `--backend` | aliases | runtime | devices |
|---|---|---|---|
| `auto` *(default)* | | the fastest Burn backend available — LibTorch on a GPU, CubeCL/CUDA, WebGPU, LibTorch on CPU | |
| `cuda` | `burn`, `burn-cuda` | native Burn, CubeCL/CUDA | NVIDIA only |
| `tch` | `libtorch`, `burn-tch` | native Burn, LibTorch | CUDA, MPS, Vulkan, CPU |
| `wgpu` | `webgpu`, `burn-wgpu` | native Burn, WebGPU | any Vulkan/Metal/DX12 GPU |

**There is no `onnx` here, and it is not a missing feature flag.** Nothing
exports Seed-VC — [`export/`](../../export/README.md) mirrors RVC and GPT-SoVITS
only, and Burn reads ONNX graphs without being able to write one — so
`--backend onnx` fails immediately with that reason rather than falling back. No
rebuild changes it.

Unlike the other engines `auto` here resolves by hardware alone: there is no
artefact on disk that could decide it. Naming a backend or device that is not
available is an **error with a reason**, never a silent fallback.

`--device auto|cpu|gpu|gpu:N|mps|vulkan` picks *which* device inside the chosen
backend (`cuda`/`cuda:N` are accepted spellings of `gpu`). The spellings are
identical on all five binaries — see
[`docs/setup.md`](../../docs/setup.md#backends-and-devices).

**This is not realtime today.** Measured on an RTX 2060 with
`--backend tch --device gpu` at default `--steps 30 --guidance 0.7`, the batch
path converted **7.79 s of audio in 10.9 s**. The cost is 30 transformer
evaluations per chunk plus, at any positive guidance, a second evaluation for the
unconditional branch; `--guidance 0` and a lower `--steps` are the two knobs that
move it.

## Recipes

**Convert a pile of files.** One `<stem>.wav` per input in `-o`; the model and
the reference are loaded and analysed once for the whole batch, which is the
reason this is a subcommand rather than a shell loop over the filter.

```sh
seedvc convert take1.mp3 take2.mp3 -o out/ -r target-voice.wav
```

**Trade quality for speed.** Halve the evaluations per step, then halve the steps
again if that is still too slow:

```sh
seedvc convert take.mp3 -o out/ -r target-voice.wav --guidance 0 --steps 10
```

**Repeat a conversion exactly** — the same seed and the same reference give the
same audio, which is how two backends are held against each other:

```sh
seedvc convert take.mp3 -o out/tch  -r clip.wav --seed 42 --backend tch
seedvc convert take.mp3 -o out/wgpu -r clip.wav --seed 42 --backend wgpu
```

**Compose with the rest of the toolkit.** Every stage is a filter, so the
pipeline is a literal shell pipe. `tts` speaks at 32 kHz and `seedvc` reads
16 kHz, so `--sr` does the resampling in-process rather than through a file:

```sh
tts -r clip.wav -t "<what clip.wav actually says>" --sr 16000 < script.txt \
  | seedvc -r target-voice.wav \
  | ffplay -f f32le -ar 22050 -ac 1 -
```

## Where files land

Four networks from **four separate Hugging Face repos**, because three of them
are other projects' releases used unmodified:

| | repo | what for |
|---|---|---|
| `DiT_seed_v2_uvit_whisper_small_wavenet_bigvgan_pruned.pth` | `Plachta/Seed-VC` | the transformer **and** the length regulator |
| `campplus_cn_common.bin` | `funasr/campplus` | the timbre encoder |
| `bigvgan_generator.pt` + its `config.json` | `nvidia/bigvgan_v2_22khz_80band_256x` | the vocoder — the band count and upsampling rates are read from that JSON, not assumed |
| the repo directory | `openai/whisper-small` | the content encoder; only its encoder half is read |

All four are **inference assets**, so they go to the shared cache —
`--cache-dir`, else `$SEEDVC_CACHE_DIR`, else `$VOICE_CACHE_DIR`, else
`~/.cache/voice`, resolved as
[`docs/setup.md`](../../docs/setup.md#where-models-are-stored) describes and
printed by `-h`. There is no `pretrained/` counterpart here, because a warm-start
base is a training input and this engine has no training.

`seedvc download` fetches all four up front, which is how a machine that will be
offline later gets set up. There is nothing optional to leave out: a conversion
opens every one of them. `--checkpoint`, `--campplus`, `--bigvgan` and
`--content` name weights you already hold instead — the last of those takes the
Whisper **directory**, since its `config.json` and the vocoder's share a name and
a flat layout would hand each loader the other's.

Converted audio goes where you point stdout, or under `convert` into `-o`, and
nothing else is written.

## Limits

**Zero-shot is the ceiling as well as the point.** No amount of reference audio
makes this a fine-tune; if a single voice matters enough to spend GPU hours on,
[`rvc`](../rvc-cli/README.md) or [`tts`](../tts-cli/README.md) is the engine, and
their loops are [`docs/training.md`](../../docs/training.md).

The port targets the v1 `seed-uvit-whisper-small-wavenet` preset only. The v2
CFM+AR pair needs an ASTRAL-Quantization tokeniser that is not written.

Two subtrees of the checkpoint are **fossils of the training-time model that the
released inference path never builds** — `net.style_encoder.*` and `net.vq.*`.
They are ported so that a prefix nobody claims is visibly nobody's rather than
possibly forgotten, but the timbre vector comes from CAMPPlus and the length
regulator's codebook is never indexed. Wiring inference to either would condition
the transformer on something Seed-VC was never trained against.

Chunk boundaries carry less content context than upstream's: each chunk's audio
is encoded with 0.19 s of left context where upstream gives it 5 s. **That is a
named guess, not a measured defect** — it is inaudible on the clips this was
developed against, and those are single-chunk, which is what a source under 25 s
with a short reference always is. If a seam ever does become audible, this is the
first thing to suspect.

The networks have no unit tests. Coverage is checked by loading the real released
weights (`cargo run -p burn-seedvc --example load`), and the solver, the chunk
arithmetic and the crossfade are tested directly.

## See also

- [repository README](../../README.md) — what `voice` is and how the engines compose
- [`docs/setup.md`](../../docs/setup.md) — ffmpeg, LibTorch, backends and devices
- [`docs/seedvc-architecture.pdf`](../../docs/seedvc-architecture.pdf) — what each of the six networks computes
- [`rvc`](../rvc-cli/README.md), [`stt`](../stt-cli/README.md) and [`tts`](../tts-cli/README.md) — the other three engines
- [`CLAUDE.md`](../../CLAUDE.md) — why the code is shaped the way it is
