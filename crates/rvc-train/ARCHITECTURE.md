# rvc-train — native RVC training

Goal: replace the Python RVC training pipeline with a pure-Rust one. Python is
used **only** for the final weight → ONNX conversion (RVC's own exporter), which
produces the `phone/phone_lengths/pitch/pitchf/ds/rnd → audio` graph that
`rvc-core` already runs.

## Pipeline

1. **Dataset prep** — *implemented* (`lib.rs`).
   Decode each corpus clip with `rvc-audio` to 16 kHz (for analysis) and to the
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
   `TuiMetricsRendererWrapper` directly (our GAN loop doesn't fit Burn's
   `Learner`): it registers the metrics once and pushes values + progress each
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

The final model is `<out>.safetensors` (the EMA when enabled) plus
`<out>.raw.safetensors` and the `<out>.disc.safetensors` sidecar. There is no
*periodic*-checkpoint machinery; there is one extra snapshot:

- **Best checkpoint** (**on**; `--no-save-best` disables). Writes the generator and
  its discriminator to `<out-dir>/checkpoint/<name>.best[.disc].safetensors`
  whenever the `mel` loss hits a new minimum, so a run that drifts late still
  leaves its best model behind. The comparison is on the *mean* mel over a window
  of `total_steps/20` steps — the per-step loss is noisy enough that its single-step
  minimum is luck — which also bounds the run to ~20 full G+D writes. The G/D pair
  is saved together and only counts as the new best once both land, so `--resume`
  on it always finds the *matching* discriminator (`best_path`/`save_checkpoint`
  in `trainer.rs`; the `.best` stem feeds the existing `disc_sidecar_path`).
  What's saved is what a deploy uses (the EMA when enabled), even though the mel
  scored is the live generator's.

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

`burn` with the `ndarray` backend by default (CPU, always compiles); a `cuda`
feature selects the GPU backend for the RTX 2060 at train time. The model is
generic over `burn::tensor::backend::Backend` / `AutodiffBackend`.
