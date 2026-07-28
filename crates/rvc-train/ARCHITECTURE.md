# rvc-train — native RVC training

Goal: replace the Python RVC training pipeline with a pure-Rust one. Python is
used **only** for the final weight → ONNX conversion (RVC's own exporter), which
produces the `phone/phone_lengths/pitch/pitchf/ds/rnd → audio` graph that
`rvc-core` already runs.

## Pipeline

1. **Dataset prep** — *implemented* (`lib.rs`).
   Decode each corpus clip with `audio-kit` to 16 kHz (for analysis) and to the
   model rate (the generator target), then extract per-frame **ContentVec**
   features and an **RMVPE F0** contour with `rvc_core::FeatureExtractor` — the
   same ONNX models the inference path uses, so features are identical at train
   and inference time.

2. **Generator training** — *done*. The network is a standalone crate,
   [`burn-rvc`](../burn-rvc), a faithful `burn` port of RVC v2's
   `SynthesizerTrnMs768NSFsid` (`enc_p`, `enc_q`, `flow`, `dec`/`GeneratorNSF`
   with the NSF `SineGen` source, `emb_g`) plus the `MultiPeriodDiscriminator`.
   Reference was RVC-Project tag `2.2.231006`
   (`infer/lib/infer_pack/{models,attentions,modules}.py`).

   Correctness is checked by **loading the real pretrained weights** (no unit
   tests): `cargo run -p burn-rvc --example load -- <f0G48k.pth>` reports
   applied/missing/unused. **The whole generator loads: 560/560 params, 0
   missing, 0 unused, 0 errors** — including weight-norm convs. The loader
   stamps names from the pickle keys, remaps RVC's flat `attn_layers/
   norm_layers_*` lists and the flow's even coupling indices onto our tree, and
   upcasts the fp16 checkpoint to fp32.

   The loop (`trainer.rs`) warm-starts G + D, then per step does a discriminator
   step (fake detached) and a generator step, with two `AdamW` optimizers
   (lr 1e-4, β 0.8/0.99). Losses match RVC exactly: mel-L1 ×45, KL ×1 (VITS's
   sampled `logs_p - logs_q - 0.5 + 0.5·(z_p-m_p)²·exp(-2·logs_p)`), feature
   matching ×2, and LSGAN adversarial. The differentiable STFT/mel front-end
   (`spectral.rs`) is `n_fft=2048`, `hop=480`, 128 Slaney mels, center=False.

   **Dashboard & early stop.** On a TTY, `dashboard.rs` drives Burn's
   `TuiMetricsRendererWrapper` directly, because our GAN loop cannot use Burn's
   `Learner`: `TrainStep::step` takes `&self` and returns one `GradientsParams`
   for one `Optimizer`, whereas a GAN needs two models, two optimizers at
   different LRs, and D updated *between* the two backward passes so G's
   adversarial loss sees the updated D. (`MultiGradientsParams` is not a way
   around this — it accumulates one parameter across *devices*, not across
   models.) The cost of opting out is that `Learner`'s multi-device strategies
   are unavailable, which is why training is single-device. The dashboard
   registers the metrics once and pushes values + progress each
   step. Plotted (numeric) metrics are the `g`/`d`/`mel` losses **and the
   learning rate**; CPU and GPU usage are shown as text lines (reusing Burn's own
   `CpuUse`/`CudaMetric` system probes, behind the `metrics` feature — NVML for
   the GPU, so `<used>/<total> Gb`, utilisation, and power). The renderer shares a
   Burn `Interrupter` — pressing `q`
   flips it and the loop stops and saves. Off-TTY (or `--no-tui`), it logs to
   stderr and **Ctrl-C** (a SIGINT flag threaded through `TrainRequest.stop`)
   stops and saves. When the TUI is on, the CLI routes `tracing` logs to
   `{work-dir}/train.log` so they don't corrupt the display.

   Notes:
   - Warm-start from RVC's `assets/pretrained_v2/f0G48k.pth` / `f0D48k.pth` is
     essential for a small (~1 h) single-speaker corpus. This requires our burn
     module layout to map onto RVC's `state_dict` keys.
   - Target GPU is 6 GB (RTX 2060) → small batch (≈4), fp16 where stable.
   - Backend: `burn` with the cuda backend at train time; keep the
     model generic over `burn::tensor::backend::Backend`.

### Training-stability controls

The loss *magnitudes* match RVC exactly; these knobs shape the training
*dynamics* on a small corpus, where a constant-LR GAN tends to converge the
timbre but leave `mel_loss` plateauing/oscillating in the back half — audible as
muffled (under-resolved highs) or faintly staticky output. All are `TrainSettings`
fields with matching `rvc train` flags; two are on by default.

- **Generator weight EMA** (`--ema-frac`, default `0.1`, **on**). The trainer
  keeps an exponential moving average of the generator weights and **saves the EMA
  as the model** — averaged over the adversarial oscillation, so cleaner and less
  prone to shipping a noisy final step. The window is **derived from the run
  length**: `ema_decay = 1 − 1/(ema_frac · total_steps)` (clamped ≤ 0.9999), so a
  fraction `ema_frac` of the run is averaged regardless of epoch count. The raw
  live weights are written alongside as `<out>.raw.safetensors` (for resume/debug);
  `--ema-frac 0` disables EMA and saves the raw weights as `<out>`. Implemented
  device-side via a `ModuleVisitor` that collects the live params as rank-erased
  `TensorPrimitive`s and a `ModuleMapper` that blends `keep·ema + (1-keep)·live`,
  order-paired by traversal (no CPU round-trip, no `ParamId` dependency). EMA is a
  *saved snapshot only* — it does not feed back into the optimizer.

- **LR schedule** (`--lr` base, `--lr-final` end fraction, default `1e-4` → `0.1×`,
  **on**). The LR decays **exponentially over the whole run**,
  `cur_lr = lr · lr_final^(step/total_steps)`, applied to *both* optimizers — so it
  is independent of the epoch count (unlike a per-epoch gamma, whose effect
  silently depends on `-e`). A constant LR bounces around the minimum instead of
  settling; the decay is what lets the late-training oscillation quiet down. Set
  `--lr-final 1.0` to disable.

- **Gradient accumulation** (`--grad-accum`, default `1`). Sums gradients over N
  micro-batches before one optimizer step, giving an *effective* batch of
  `batch × N` at single-micro-batch VRAM (each micro-batch's graph is freed after
  its backward). Costs ~N× compute per epoch. Accumulation is a `ModuleVisitor`
  that adds `incoming · (1/N)` into a running `GradientsParams`, matched by
  `ParamId`; gradients live on the inner (non-autodiff) backend.

- **Discriminator balancing** (`--d-lr-ratio`, `--d-interval`, both no-op by
  default). If the output has buzzy/high-frequency static — the discriminator
  overpowering the generator — weaken D with `--d-lr-ratio 0.5` (scales D's LR)
  or `--d-interval 2` (updates D every other step). Skipped D steps carry the
  previous `d` loss forward on the dashboard.

- **SNR clip weighting** (`--snr-weight`, default `0.0` = uniform). Biases which
  clip each training window is drawn from by `snr^alpha`, where `snr` is each
  clip's **noise-floor SNR** — p75 frame-RMS over p10 frame-RMS (a *ratio*, never
  absolute loudness), so a soft-but-clean ASMR clip still scores high and the
  breathy passages the slicer works to preserve are not penalised. `frame_snr`
  and the cumulative-weight sampling live in `dataset.rs`.

### Checkpoints

`checkpoint.rs` owns everything about saved weights. A checkpoint is one family
of files sharing a stem, and *every* save writes the same family:

```
<out>.safetensors        the deployable generator — the EMA when it's on
<out>.raw.safetensors    its raw (non-EMA) live twin        [only with EMA on]
<out>.disc.safetensors   the discriminator that co-evolved with that twin
```

`Checkpoint::new` accepts any spelling of a family (with or without the suffix,
generator, raw twin or discriminator sidecar) and `resume_from` picks the twin
when it exists, so
`--resume` continues from the `raw-G ↔ live-D` pairing that actually played the
adversarial game rather than from the EMA, which never itself faced D. Members
are built from the *stem* rather than `Path::with_extension` — that only sees the
last dot, and on `voice.best.raw.safetensors` it would strip `.best` along with
`.raw` and point resume at a different run's discriminator.

There is no *periodic*-checkpoint machinery; there is one extra snapshot:

- **Best checkpoint** (**on**; `--no-save-best` disables). `Checkpoint::best()` is
  the same family under `<out-dir>/checkpoint/` with a `.best` stem, written
  whenever the `mel` loss hits a new minimum — so a run that drifts late still
  leaves its best model behind, and that model deploys and resumes exactly like
  the final one. What's saved is what a deploy uses (the EMA), even though the mel
  scored is the live generator's — which is exactly why the raw twin matters: it's
  the G that faced the D beside it. Three rules make the comparison mean something:

  - **Window.** The per-step mel is noisy enough that its single-step minimum is
    luck, so what's compared is the *mean* over `(total_steps/20).clamp(1, 50)`
    steps. The cap is load-bearing: `total_steps` is the *scheduled* count, and
    since runs are normally ended by hand (`q`/Ctrl-C) long before that, an
    uncapped window would first close only in runs nobody actually completes.
  - **Persistence.** The score lives in a `<out>.best.json` sidecar
    (`{"mel", "step"}`, hand-parsed — two fields don't earn a serde dependency)
    and is read back at startup. Without it every process starts from an empty
    best and its first window overwrites the previous run's model however much
    worse it is, which under the recommended stop-early/`--resume` workflow
    reduces `.best` to "the last run's exit snapshot". A sidecar whose weights are
    missing is ignored; a corrupt one is a `warn!` and no more.
  - **Tail.** The partial window an early stop leaves behind is judged only when
    it's at least half a full window — a one-step mean has many times the variance
    of a full one and would unseat a genuinely better best. Below that it still
    counts when nothing is on disk, so a stopped-early run leaves *something*.

  A family counts as the new best only once all its files land — sidecar last —
  so a failed write can neither leave a generator paired with a stale
  discriminator nor advance the score a later run inherits. Every run reports on
  exit what it kept, or warns that it wrote no best at all — the failure mode here
  is invisible otherwise.

3. **ONNX export** — the one allowed Python step, kept **minimal and standalone**.
   A small self-contained `uv` project (~one torch file) defines the inference
   graph (`SynthesizerTrnMsNSFsidM`-equivalent), loads the trained weights, and
   runs `torch.onnx.export`. It does **not** depend on the RVC-Project repo.
   Flow: burn training writes weights as safetensors → the exporter loads them
   into the torch inference model → `phone/phone_lengths/pitch/pitchf/ds/rnd ->
   audio` ONNX → drops straight into `rvc convert -m …`.

## Why match RVC's layout

Keeping the burn modules weight-compatible with RVC's `state_dict` lets us
warm-start from the public pretrained bases (essential on ~1 h of audio) and
keeps the standalone torch exporter a faithful, trivial mirror. The RVC-Project
repo (tag `2.2.231006`, `infer/lib/infer_pack/{models,attentions,modules}.py`)
was the read-only reference for transcribing the architecture; the port is
complete and the toolkit no longer needs it. The pretrained warm-start bases
`f0G48k.pth`/`f0D48k.pth` (HF `lj1995/VoiceConversionWebUI`) are what
`rvc train --pretrained-g/-d` consumes.

## Backend

`trainer::run<AB: AutodiffBackend>(req, clips, &[AB::Device])` is generic over the
compute backend; `lib.rs::dispatch` instantiates the concrete one from
`TrainRequest::backend` at run time. Two cargo features (`tch`, `cuda`) decide what
is linked; the CLI enables both.

All three backends train, but one burn 0.21 quirk dictates a detail of the port:
the autodiff builds a weight gradient one or more kernel-taps too long for a
*grouped, strided* `conv1d` whose padded input length is not a multiple of the
stride. CubeCL and WebGPU absorb it; LibTorch's strict `copy_` aborts.
`DiscriminatorS` is precisely that shape (`k=41, s=4, groups=4..256` over a
17280-sample segment), so `DiscriminatorS::forward` reflect-pads its input up to
`len % SCALE_ALIGN == 1`, which the whole four-layer chain then divides evenly. `examples/convgrad.rs` is the probe: its minimal failing case is
`conv1d(16 -> 64, k=4, s=2, p=1, groups=4)` on length 101, and it also shows the
padded chain passing.

`AB::InnerBackend` is where gradients, the EMA and every saved weight live.
`AutodiffBackend` guarantees `InnerBackend: Backend<Device = Self::Device>`, so a
single `device` value serves the whole loop — there is no second device to thread
through. The bound really is just `AB: AutodiffBackend`: `GradientsParams::{remove,
register}` are bound on plain `Backend`, `AdamW` implements `SimpleOptimizer<B>` for
any `B: Backend`, and the `Module` derive emits a concrete
`type InnerModule = Synthesizer<B::InnerBackend>` that rustc normalises without help.

Device selection is `rvc_core::DeviceSpec` (`--device auto|cpu|gpu|gpu:N|mps|vulkan`),
shared with the inference path so `auto` means the same thing in both.

`run` takes a device *slice*. With one entry it behaves exactly as before — no
clone, no transfer. With more it is data-parallel on the master-device plan, which
is what `Learner`'s `MultiDeviceOptim::OptimMainDevice` does and what we have to
reimplement because the GAN loop can't use `Learner`: per-step replicas via
`Module::to_device`, one `micro_step` per device, then
`GradientsParams::to_device` back to the master before the existing `accumulate`
sums them. Only the master's weights, optimizers and EMA advance.

Two limits worth knowing. Devices are dispatched sequentially, so the win depends
entirely on backends queueing work asynchronously — which is why `micro_step`
returns loss *tensors* and `run` reads them only after every device and both
optimizers have been dispatched. One `into_data()` in the middle of the fan-out
would sync each device before the next was queued and make N devices strictly
serial. `MultiDevicesTrainStep`'s thread-per-device is the next step for real
scaling. And replicas are copied every step; avoiding that needs persistent
replicas plus an all-reduce (`burn-collective`), which is out of scope.

**N>1 is verified only as GPU+CPU on LibTorch** — the development machine has one
GPU, so two-GPU sharding has never run. The two things that would fail silently
have unit tests: that a replica's gradients reach the master under the same
`ParamId`s (they did not, until burn's lazy parameter init was forced before the
first clone), and that the gradient scale averages over devices as well as
accumulation steps.
