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
   `Learner`): it registers the `g`/`d`/`mel` losses once and pushes a value +
   progress each step. The renderer shares a Burn `Interrupter` — pressing `q`
   flips it and the loop stops and saves. Off-TTY (or `--no-tui`), it logs to
   stderr and **Ctrl-C** (a SIGINT flag threaded through `TrainRequest.stop`)
   stops and saves. When the TUI is on, the CLI routes `tracing` logs to
   `{work-dir}/train.log` so they don't corrupt the display.

   Notes:
   - Warm-start from RVC's `assets/pretrained_v2/f0G48k.pth` / `f0D48k.pth` is
     essential for a small (~1 h) single-speaker corpus. This requires our burn
     module layout to map onto RVC's `state_dict` keys.
   - Target GPU is 6 GB (RTX 2060) → small batch (≈4), fp16 where stable.
   - Backend: `burn` with the wgpu (or cuda) backend at train time; keep the
     model generic over `burn::tensor::backend::Backend`.

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

`burn` with the `ndarray` backend by default (CPU, always compiles); a `wgpu`
feature selects the GPU backend for the RTX 2060 at train time. The model is
generic over `burn::tensor::backend::Backend` / `AutodiffBackend`.
