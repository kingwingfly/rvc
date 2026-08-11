# Training

Everything that is true of **every** training loop in this toolkit. Per-flag
detail belongs to the engine manuals
([`rvc`](../crates/rvc-cli/README.md#train-a-voice),
[`tts`](../crates/tts-cli/README.md#fine-tune-a-voice)) and per-network detail to the
architecture papers ([`rvc-architecture.typ`](rvc-architecture.typ),
[`gptsovits-architecture.typ`](gptsovits-architecture.typ)); this page is the
part that would otherwise be written twice and drift.

Training is **native Rust on Burn**. There is no Python in the loop — not a
preprocessing script, not a `uv` project, not a shelled-out trainer. Models are
ported to Burn and load their original Hugging Face weights directly, which is
what makes fine-tuning possible at all: ONNX Runtime has no training path, so a
model this repo tunes *must* be a Burn port whatever else it also runs on.

## The three loops

| loop | what it adapts | objective | unit of work |
|---|---|---|---|
| `rvc-train` | a whole voice — timbre and pitch handling | VITS GAN | a fixed 48-frame (0.48 s) window, decoder on a 36-frame segment (`--window-frames`, `--segment-frames`) |
| `tts-train` `s2` | timbre, for GPT-SoVITS | the same VITS GAN | one whole utterance, decoder on a 32-frame (0.64 s) segment |
| `tts-train` `s1` | delivery — pacing, emphasis, where a speaker breathes | next-token cross-entropy | one whole token sequence |

Two of the three are the *same* adversarial loop, because RVC and GPT-SoVITS's
`s2` are the same architecture from the same source — which is why `burn-vits`
exists, and why `s2` fine-tuning **inherited** RVC's loss family instead of
inventing one. Two implementations of an objective this fiddly would drift.

`s1` is the odd one and deliberately so: one model, one optimizer, one loss. Its
number means something on its own, which no GAN loss does.

### The unit of work is where they genuinely differ

`rvc-train` samples fixed 48-frame windows at random across the corpus, which it
can only do because RVC's conditioning is frame-aligned content features with no
sequence structure — any 0.48 s of a clip is a valid training example. `enc_q`
and the flow see the window; the decoder renders a 36-frame slice of it.

Both counts are `--window-frames` and `--segment-frames`, and they were constants
here until the two loops were brought to parity. `--segment-frames` is `tts
train`'s knob of the same name; **`--window-frames` has no counterpart there at
all**, because `s2` never draws a fixed window. That is also why it carries a
second job here: since the window is a *fixed* size drawn out of a clip, a clip
shorter than one window is not a valid example at all, so the same number is the
corpus's minimum clip length and clips below it are dropped before the loop
starts. On this repository's own 92-clip corpus, 48 → 400 leaves 32. `s2` has no
such floor, because its unit of work is whatever the utterance happens to be.

`s2` cannot. Its conditioning is *text*, attended to through the MRTE
cross-attention, so a window of frames no longer matches the phonemes it was
transcribed from. Each micro-batch is therefore one whole utterance: `enc_p`,
`enc_q` and the flow see all of it and only the decoder runs on a random
segment, exactly as upstream's `rand_slice_segments` does. The consequence is
that **memory grows with the longest clip rather than with the batch**, which is
why there is a frame cap at all to keep a 6 GB card alive, and why a batch here
is processed one clip at a time and accumulated rather than padded.

That cap is `--max-frames`, and it is *separate* from `--max-tokens` for a
reason worth stating once: `--max-tokens` bounds `s1`'s sequence, in semantic
tokens at 25 Hz, while the cap that governs `s2`'s peak memory is in latent
frames at 50 Hz. It defaults to twice `--max-tokens` because a token is two
frames, which is the right default and was the wrong *only* option — while it
was derived, raising `--max-tokens` so `s1` could see longer lines doubled `s2`'s
memory as an unrelated side effect.

`s1` is the same story without the segment: sequences differ in length, the loop
does not pad, so a batch is an accumulation count.

## The shared objective

Four losses in fixed proportions, identical in `rvc-train` and in `s2` because
VITS is their common ancestor:

```
mel-L1 × 45      KL × 1      feature matching × 2      LSGAN adversarial
```

Each step runs **D before G**: the discriminator is stepped on a detached fake
first, so the generator's adversarial term then faces the discriminator that has
just learned from it. The multi-period discriminator is shared
(`burn_vits::MultiPeriodDiscriminator`); only the periods differ, RVC's
`[2,3,5,7,11,17,23,37]` against GPT-SoVITS's `[2,3,5,7,11]`. Both also carry the
single scale discriminator, which is always entry 0 of an upstream checkpoint's
flat `discriminators.N` list.

The differentiable STFT/mel front-end is `burn_vits::Spectral`: `n_fft=2048`,
`hop=480`, 128 Slaney mels, `center=False` for RVC at 48 kHz, and the
GPT-SoVITS 32 kHz configuration for `s2`.

## Why there is no Burn `Learner`

`TrainStep::step` takes `&self` and returns one `GradientsParams` for one
`Optimizer`. A GAN needs two models, two optimizers at different learning rates,
and D updated *between* the two backward passes. `MultiGradientsParams` is not a
way around it — it accumulates one parameter across *devices*, not across
models.

Opting out costs `Learner`'s multi-device strategies, which is why the
data-parallel plan below is reimplemented by hand, and it costs the metrics
plumbing, which is why `train_kit::Dashboard` drives Burn's
`TuiMetricsRendererWrapper` directly. `s1` could have used `Learner` — it is one
model and one loss — but sharing the dashboard, the checkpoint family and the
schedule with the other two was worth more than the skeleton.

## `train-kit`

Scaffolding that knows about no model, generic over the module being trained, so
a GAN over a vocoder and a cross-entropy loop over a transformer share it.

**`Checkpoint`** — a checkpoint is one *family* of files sharing a stem, and
every save writes the whole family:

```
<out>.safetensors        the deployable model — the EMA when it is on
<out>.raw.safetensors    its raw (non-EMA) live twin        [only with EMA on]
<out>.disc.safetensors   the discriminator that co-evolved with that twin
```

`Checkpoint::new` accepts any spelling of a family — with or without the suffix,
generator, raw twin or discriminator sidecar — and `resume_from` picks the twin
when it exists, so `--resume` continues from the `raw-G ↔ live-D` pairing that
actually played the adversarial game rather than from the EMA, which never
itself faced D. Members are built from the *stem* rather than
`Path::with_extension`: that only sees the last dot, and on
`voice.best.raw.safetensors` it would strip `.best` along with `.raw` and point
resume at a different run's discriminator.

**`Schedule`** — the LR decay and the EMA window, both expressed as fractions of
a *run* rather than as per-epoch quantities. `cur_lr = lr · lr_final^(step/total)`
and `ema_decay = 1 − 1/(ema_frac · total_steps)` (clamped ≤ 0.9999). A per-epoch
gamma would make `--epochs` quietly mean something different in every loop, and
a constant LR bounces around the minimum instead of settling — the decay is what
lets late-training oscillation quiet down.

**`ema_update` / `materialize`** — the EMA is kept device-side by a
`ModuleVisitor` that collects live parameters as rank-erased `TensorPrimitive`s
and a `ModuleMapper` that blends `keep·ema + (1−keep)·live`, order-paired by
traversal, so there is no CPU round-trip and no `ParamId` dependency. **EMA is a
saved snapshot only** — it never feeds back into the optimizer. `--ema-frac 0`
turns it off and saves the raw weights as `<out>`.

**`accumulate`** — sums gradients over micro-batches and devices as
`incoming · (1/N)` into a running `GradientsParams`, matched by `ParamId`, on the
inner (non-autodiff) backend. It is what makes an effective batch of `batch × N`
fit at single-micro-batch VRAM, at ~N× compute.

**`Best`** — the best-so-far snapshot, `<out-dir>/checkpoint/` with a `.best`
stem, written whenever the mel loss hits a new minimum. A run that drifts late
still leaves its best model behind, and that model deploys and resumes exactly
like the final one. Three rules make the comparison mean anything:

- **Window.** The per-step mel is noisy enough that its single-step minimum is
  luck, so what is compared is the mean over `(total_steps/20).clamp(1, 50)`
  steps. The cap is load-bearing: `total_steps` is the *scheduled* count, and
  runs are normally ended by hand long before it, so an uncapped window would
  first close only in runs nobody completes.

  **The clamp's other end is the one worth overriding**, and `Best::new` takes
  the window as an `Option` for it: a short schedule divides down to zero and
  the floor of `1` puts back a single-step mean, which is precisely the noise
  the window exists to average out. So the derived value is right for a run
  scheduled the length it will actually take, and wrong for a long schedule
  somebody intends to interrupt — a distinction only the caller can make, which
  is why it is a parameter rather than a wider clamp. `rvc train --best-window`
  exposes it; `s2` keeps the derived one, having no equivalent habit.
- **Persistence.** The score lives in a `<out>.best.json` sidecar and is read
  back at startup. Without it every process starts from an empty best and its
  first window overwrites the previous run's model however much worse it is —
  which under the recommended stop-early/`--resume` workflow reduces `.best` to
  "the last run's exit snapshot". A sidecar whose weights are missing is
  ignored; a corrupt one is a warning and no more.
- **Tail.** The partial window an early stop leaves is judged only when it is at
  least half a full window, because a one-step mean has many times the variance
  of a full one. Below that it still counts when nothing is on disk, so a
  stopped-early run leaves *something*.

A family counts as the new best only once all its files land — sidecar last — so
a failed write can neither pair a generator with a stale discriminator nor
advance a score a later run inherits. Every run reports on exit what it kept, or
warns that it wrote no best at all; the failure mode is invisible otherwise.

**`Dashboard`** — on a TTY, Burn's `TuiMetricsRendererWrapper` plotting the
loop's numeric metrics and the learning rate, with CPU and GPU usage as text
lines (Burn's own `CpuUse`/`CudaMetric` probes behind the `metrics` feature —
NVML for the GPU, so used/total, utilisation and power). Pressing **`q`** flips
the shared `Interrupter` and the loop stops and saves. Off a TTY, or with
`--no-tui`, it logs to stderr and **Ctrl-C** does the same through a SIGINT flag
threaded into the loop. The same call emits the throttled `step/total … eta`
line either way: with a TUI up the CLI has routed `tracing` to `train.log` in the
output directory, so the log records the run's curves instead of scribbling over
the display.

**`distinct`**, **`Rng`**, **`human`**, **`scalar`** — the small shared pieces:
rejecting a `--device` list that names one device twice (`auto,gpu:0` are two
spellings of device 0, and on WebGPU `auto`, `vulkan` and `mps` all resolve to
the default adapter — a repeat silently doubles that device's share of the work
and its memory, which looks like training working), a seeded sampler so a run
repeats, a
duration formatter, and the one call that syncs a loss tensor back to `f32`.

### What is deliberately *not* shared

**The shape of a step.** A GAN needs two models, two optimizers and a
discriminator update wedged between two backward passes; a cross-entropy loop
needs one of each. Forcing one skeleton over both would cost more than the
duplication it removes, so each trainer writes its own loop out of these parts.

## Warm-start

On a small corpus — an hour of audio, often much less — warm-starting is not
optional. Every loop here starts from published weights:

- `rvc-train` from RVC's `f0G48k.pth` + `f0D48k.pth`
  (HF `lj1995/VoiceConversionWebUI`),
- `s2` from an upstream `s2G*.pth` (or an earlier run's `.safetensors`) plus the
  matching `s2D2333k.pth` discriminator,
- `s1` from the upstream `s1*.ckpt`.

A missing *discriminator* is survivable but not advised: a fresh adversary
spends its early steps learning what speech is rather than what this voice is.
A missing or partial *generator* is refused outright — `missing` and `errors`
are both checked, because the applier drops a path that failed to apply from
both lists, so a shape mismatch would otherwise read as full coverage while the
parameter silently kept its initialised value.

Warm-start is also the reason the Burn modules stay weight-compatible with the
original `state_dict` layouts. Changing a module's *field* names moves parameter
paths and breaks it; renaming a type or moving a file does not.

**Where the bases land:** `pretrained/` inside the **asset cache**, alongside
every other download. A base is the published upstream file and is the same for
every voice trained on the machine, so it is fetched once and reused; an output
directory holds only what a run produced. `--no-pretrained` fetches nothing at
all, and neither does RVC's
`--resume`: it continues from weights that already exist, so a base would be
loaded and immediately overwritten. See
[where models are stored](setup.md#where-models-are-stored).

## Corpus preparation

**Both trainers eat a corpus that `preprocess` produced, and it is its own
binary.** `rvc preprocess` and `tts preprocess` existed and are **gone** —
removed rather than deprecated, so an old command line fails to parse rather
than doing something else. What replaced them is not a rename: corpus
preparation grew stages that run models (source separation, speaker
diarisation), which is more than a subcommand on two engines that neither of
them is about. Every flag and default is
[`crates/preprocess-cli/README.md`](../crates/preprocess-cli/README.md); what
follows is only why a trainer cares.

**RVC**: `preprocess clip raw/*.mp3 -o clips/` first, always. `sample_batch`
draws random 0.48 s windows uniformly across each file, so raw recordings full
of between-sentence dead air collapse the generator to silence. The slicer
(`audio_kit::slice`) cuts on long silent gaps only — energy is used to *find*
gaps, never to gate quiet-but-present sound, so breathy and whispered passages
survive. `--silence-db` lowers the floor, `--min-silence` sets how long a gap
must last to be a cut, and neither ever splits a sentence.

**GPT-SoVITS**: a corpus is `<stem>.wav` + `<stem>.txt` pairs. `preprocess clip`
— the same slicer, run the same way — cuts raw recordings into per-sentence
clips, and `stt` writes the transcript beside each. Preparation runs cnhubert and the quantiser over
each clip to produce the 25 Hz semantic tokens both stages agree on, and it is
the expensive half of a fine-tune — which is why `--stage s1|s2|both` prepares
exactly once and each stage then writes its own checkpoint family.

**When the source is a stream rather than a studio take**, slicing is not the
first step and running it first tells you nothing about why. `audio_kit::slice`
cuts on silence, and a continuous music bed means the recording has none — the
whole file comes back as a handful of minutes-long clips, from which
`sample_batch` then draws windows of somebody else's music. `preprocess analyze`
is what reports that condition rather than leaving it to be inferred: it runs
the slicer at the floor you asked for *and* at one measured from the recording,
and says outright when no floor can work. The fix is `preprocess separate` to
take the bed off, then `preprocess diarize` if a second voice is in it, and
`clip` last. The stages compose because each one reads audio files and writes
audio files.

## Stability knobs

The loss *magnitudes* match upstream exactly; these shape the *dynamics* on a
small corpus, where a constant-LR GAN tends to converge the timbre but leave the
mel loss plateauing or oscillating in the back half — audible as muffled
(under-resolved highs) or faintly staticky output. Two are on by default.

- **Generator weight EMA** (`--ema-frac`, default `0.1`, on). Averaged over the
  adversarial oscillation, so what deploys is cleaner than any single step.
- **LR decay** (`--lr-final`, default `0.1×`, on). `1.0` disables it.
- **Gradient accumulation** (`--grad-accum`, RVC). The effective-batch lever
  where the batch is a fixed window; the GPT-SoVITS loops get the same effect
  from `--batch-size`, since they accumulate by construction.
- **Discriminator balancing** (`--d-lr-ratio`, `--d-interval`, both no-ops by
  default, both GANs). Buzzy high-frequency static is D overpowering G: weaken
  it with `--d-lr-ratio 0.5` or update it every other step. Skipped D steps
  carry the previous `d` loss forward on the dashboard.
- **SNR clip weighting** (`--snr-weight`, RVC, default uniform). Biases which
  clip a window is drawn from by `snr^alpha`, where `snr` is the clip's
  *noise-floor* SNR — p75 frame-RMS over p10 frame-RMS, a ratio and never
  absolute loudness, so a soft-but-clean clip still scores high and the
  breathy passages the slicer works to preserve are not penalised.

## Never clobbering a trained voice

A voice is hours of GPU time and the corpus that produced it may be gone, so a
run **refuses to start** when its output `.safetensors` already exists, naming
the file and saying to pass `-y`. The check happens before the corpus is
prepared or a model is loaded — the point is not to discover it after an hour.
`--resume` is itself the statement that overwriting is intended.

There is no `--work-dir`: the working directory is the current directory, as it
is for any other Unix tool. The one file a run writes that is not weights is its
log, and that goes with the weights — `-o models/voice` puts the dashboard's
`tracing` output in `models/train.log`, next to the checkpoint family it
describes.

## Backends and devices

Every loop is generic over `AutodiffBackend` and the concrete backend is chosen
at run time from `--backend`; the cargo features (`tch`, `cuda`, `wgpu`) decide
what is linked, and the CLIs enable all of them. `onnx` is rejected with a
reason rather than being a separate type — ONNX Runtime cannot train.

`AB::InnerBackend` is where gradients, the EMA and every saved weight live.
`AutodiffBackend` guarantees `InnerBackend: Backend<Device = Self::Device>`, so
one device value serves the whole loop.

**All three backends train**, but one burn 0.21 quirk dictates a detail of the
port: autodiff builds a weight gradient one or more kernel-taps too long for a
*grouped, strided* `conv1d` whose padded input length is not a multiple of the
stride. CubeCL and WebGPU absorb it; LibTorch's strict `copy_` aborts.
`DiscriminatorS` is precisely that shape (`k=41, s=4, groups=4..256`), so
`DiscriminatorS::forward` reflect-pads its input to a length the whole four-layer
chain divides evenly (`SCALE_ALIGN`). Do not remove that padding without
re-running the probe:

```sh
cargo run -p rvc-train --example convgrad --features tch,cuda,wgpu
```

Its minimal failing case is `conv1d(16 → 64, k=4, s=2, p=1, groups=4)` on length
101, and it also shows the padded chain passing.

### Lazy parameters

Burn allocates parameters lazily, and two things go wrong while a module is
still lazy: a clone taken beforehand gets **fresh `ParamId`s**, and parameters
that materialise *during* the differentiated pass yield no gradients at all.
Either way a replica's gradients stop matching the master's and are dropped
silently — every extra device contributes nothing while the run looks healthy.
Warm-start and resume materialise on load; training from scratch does not, so
`train_kit::materialize` is called before any replica is made.

### Multi-device

`run` takes a device *slice*. One entry behaves exactly as single-device — no
clone, no transfer. More is data-parallel on the master-device plan, which is
what `Learner`'s `MultiDeviceOptim::OptimMainDevice` does and what has to be
reimplemented here because the GAN loop cannot use `Learner`: per-step replicas
via `Module::to_device`, one micro-step per device, then
`GradientsParams::to_device` back to the master before `accumulate` sums them.
Only the master's weights, optimizers and EMA advance.

Two limits worth knowing. Devices are dispatched sequentially, so the win
depends entirely on backends queueing work asynchronously — which is why a
micro-step returns loss *tensors* and the loop reads them only after every
device and both optimizers have been dispatched. **One `into_data()` in the
middle of the fan-out would sync each device before the next was queued and make
N devices strictly serial.** `MultiDevicesTrainStep`'s thread-per-device is the
next step for real scaling. Replicas are also copied every step; avoiding that
needs persistent replicas plus an all-reduce (`burn-collective`), out of scope.

**N>1 is verified only as GPU+CPU on LibTorch** — the development machine has
one GPU, so two-GPU sharding has never run. The two things that would fail
silently have unit tests: that a replica's gradients reach the master under the
same `ParamId`s (they did not, until lazy parameter init was forced before the
first clone), and that the gradient scale averages over devices as well as
accumulation steps.

## Deploying what was trained

Burn imports ONNX graphs and cannot emit one, so a checkpoint trained here
reaches ONNX Runtime through `export/` — the toolkit's one deliberate Python
exception, a maintainer's build-time tool that no user, test or training run
invokes. Native Burn inference needs no export: pass the `.safetensors` wherever
the base weights went. See [`export/README.md`](../export/README.md).
