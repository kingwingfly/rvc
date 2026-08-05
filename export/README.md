# onnx-export

Convert this toolkit's Burn models to ONNX for deployment with ONNX Runtime.

This is the **only** Python in the toolkit — a small, standalone `python` project managed by `uv`.
It depends on none of the upstream projects — RVC-Project, GPT-SoVITS or
Plachta/Seed-VC: `rvc_infer.py`, `gptsovits_infer.py` and the `seedvc_infer/`
package are clean-room torch reimplementations of the three inference paths, each
mirroring its Burn module layout (`burn-rvc`, `burn-gptsovits` + `burn-vits`,
`burn-seedvc`) so the trained weights load with a direct key mapping — `Linear`
weights transposed, weight-norm folded.

It exists because Burn can *import* ONNX but cannot *emit* it. A model
fine-tuned here (`rvc train`, `tts train`) reaches ONNX Runtime only through a
torch reimplementation of the same layout, so this is the one path by which a
fine-tuned voice is ever deployable to a runtime other than Burn.

## RVC

```sh
uv run --project export python export/export_rvc.py \
    models/voice.safetensors  models/voice.onnx

rvc convert --backend onnx -m models/voice.onnx --model-sr 48000 -o out/  in.mp3
```

The exported graph matches the `rvc-core` contract:

```
inputs : phone [1,T,768] f32, phone_lengths [1] i64, pitch [1,T] i64,
         pitchf [1,T] f32, ds [1] i64, rnd [1,192,T] f32
output : audio [1,1,L] f32
```

## GPT-SoVITS

```sh
uv run --project export python export/export_gptsovits.py \
    models/gptsovits  models/gptsovits/onnx

tts --models models/gptsovits --backend onnx -r clip.wav -t "…" < script.txt
```

The first argument is the directory `tts --models` points at
(`chinese-hubert-base/`, an `s1*.ckpt` and an `s2G*.pth`); the second is where
the graphs are written, and `tts` looks for them in `<models>/onnx` or
`<models>`. `--s1`/`--s2` replace either stage with a Burn `.safetensors` — what
`tts train` writes — which is the case this exporter is for. `--only
reference|s1|s2` re-exports one stage, which is what you want after a fine-tune
touched one of them.

### What the export was checked against

Synthesising one sentence twice — `tts --backend tch` and `tts --backend onnx`,
same seed, same reference, the base `s1v2.ckpt` + `s2G2333k.pth` — produced
**93 440 samples both times**, so `s1` sampled the identical token sequence on
both runtimes, and the two waveforms differ by max 7.9e-03 on a signal of RMS
5.2e-02 (RMS difference 3.2e-04, correlation 0.99998). That is f32 accumulation
noise through five HiFiGAN upsample stages, not a difference in what the graphs
compute, and it exercises the whole chain: cnhubert, the quantiser, `ref_enc`,
`s1`'s prompt pass, 73 cached decode steps, and `s2`. Both transcribe back
through `stt` to the same text.

`s1_prompt` over a whole prompt also agrees with `s1_prompt` + `s1_step` to
6.7e-06 — the incremental-versus-one-shot check `burn-gptsovits`'s own tests
make, and the one that catches a wrong mask or a wrong position offset.

### Not covered

- **Training.** These are inference graphs. The discriminators, `enc_q` and the
  losses have no ONNX form and are not going to get one — fine-tuning is Burn,
  which is the point of the split.
- **Anything but batch 1.** The graphs trace one utterance at a time, which is
  what both CLIs do.
- **`s1` sampling.** Token choice, the repetition penalty and the stop rule stay
  on the host in `tts-core`, identically for both runtimes. The two runtimes
  therefore do not produce identical waveforms even at one seed: `s1` samples,
  and a logit that differs in the last bit can change a token.

## Seed-VC

```sh
uv run --project export python export/export_seedvc.py \
    models/seedvc  models/seedvc/onnx

seedvc convert --backend onnx --onnx models/seedvc/onnx -r clip.wav -o out/ in.wav
```

The first argument is a directory holding the four released checkpoints — the
Seed-VC checkpoint (`DiT_seed_v2_uvit_whisper_small_wavenet_bigvgan_pruned.pth`),
`campplus_cn_common.bin` from `funasr/campplus`, `bigvgan_generator.pt` from
`nvidia/bigvgan_v2_22khz_80band_256x`, and whisper-small's `model.safetensors` —
each matched by what identifies it rather than by its full upstream name, so a
cache, a clone or a hand-made copy all work. The second is where the six graphs
are written. `--dit`/`--campplus`/`--bigvgan`/`--content` name any of the four
checkpoints directly, and `--only content|style|mel|regulator|dit|bigvgan`
re-exports one of the graphs, which is what you want after re-downloading a
single network.

The six graph contracts:

```
content.onnx    : audio [1,480000] f32 → content [1,1500,768] f32       (static)
style.onnx      : audio [1,S] f32      → style [1,192] f32
mel.onnx        : audio [1,S] f32      → mel [1,80,S/256] f32
regulator.onnx  : content [1,C,768] f32, picks [T] i64 → cond [1,T,512] f32
dit.onnx        : x [B,80,T] f32, prompt_x [B,80,T] f32, t [B] f32,
                  style [B,192] f32, cond [B,T,512] f32 → v [B,80,T] f32
bigvgan.onnx    : mel [1,80,T] f32     → audio [1,1,256T] f32
```

Every dynamic axis is declared `Dim.DYNAMIC`; `content.onnx` is the one graph
with none, because whisper-small's encoder is trained on padded 30 s windows and
always returns all 1500 frames — the host pads on the way in and slices on the
way out, exactly as `content.rs` does. `dit.onnx`'s batch axis is dynamic on
purpose: classifier-free guidance stacks the conditioned and unconditional
inputs as batch 2 in a single call, which is how `flow.rs` pays for one forward
pass instead of two. Everything else traces one clip.

`dit.onnx` and `regulator.onnx` are built from the same checkpoint — the
transformer and the length regulator are the two things `Plachta/Seed-VC` holds.
The exporter reads the 440 MB file once and applies each graph's remap set.
Weight-norm is not folded: the transformer and the WaveNet keep their
`weight_g`/`weight_v` parameters and ONNX Runtime constant-folds the
reconstruction at session init, which is simpler than folding in Python and
what the two modules were written for. Only BigVGAN's plain convolutions, whose
checkpoint stores the pair, are folded by the exporter.

### What the export was checked against

`seedvc convert` on the same source, reference, seed, steps and guidance against
`--backend tch` and `--backend onnx --onnx <bundle>` produced two waveforms whose
**log-energy envelopes correlate at r=1.000000**, mean |dB difference| < 0.001
dB at 64-, 256- and 1024-sample block sizes, and **`stt` transcribes both to
identical text**. That exercises all six graphs end to end — content, style, mel,
regulator, the diffusion transformer at two batch sizes (the CFG pair), Euler
integration on the host, and BigVGAN — on the real checkpoints, at the real
preset.

### Not covered

- **Training.** Seed-VC is zero-shot — a reference clip is the whole speaker
  specification — so there is nothing to fine-tune and no training graph to
  export. The flow-matching Euler loop stays on the host in `seedvc-core`,
  identically for any runtime.
- **Anything but batch 1**, except `dit`'s classifier-free-guidance pair, which
  is exactly batch 2 and is what the graph's dynamic batch axis exists for.
- **`net.style_encoder.*` and `net.vq.*`**, the two fossil subtrees of the
  Seed-VC checkpoint that upstream's `build_model` never assembles — inference
  builds a *separate* CAM++ from `campplus_cn_common.bin` for the timbre vector,
  and this preset is not discrete, so the length regulator's codebook is never
  indexed. The exporter reads straight past both.

## Notes

All three exporters use PyTorch's modern **dynamo** ONNX exporter
(`torch.onnx.export(..., dynamo=True)` with `dynamic_shapes`), so they are
warning-free and require `torch>=2.7`. Dynamic axes are declared with
`Dim.DYNAMIC` rather than `Dim.AUTO`: DYNAMIC *asserts* the axis stays dynamic
and raises if the trace specialises it to the dummy length, whereas AUTO
silently bakes in a fixed one — and a fixed sequence length would make every one
of these graphs useless. Each saved model is validated with
`onnx.checker.check_model(..., full_check=True)`, which also runs shape
inference and so catches an internally inconsistent graph that the default
structural check would pass.

Native Burn inference needs no export — this is only for ONNX Runtime /
cross-framework deploy.
