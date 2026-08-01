# onnx-export

Convert this toolkit's Burn models to ONNX for deployment with ONNX Runtime.

This is the **only** Python in the toolkit — a small, standalone `python` project managed by `uv`. 
It depends on neither the RVC-Project nor the GPT-SoVITS repository: `rvc_infer.py`
and `gptsovits_infer.py` are clean-room torch reimplementations of the two
inference paths, each mirroring its Burn module layout (`burn-rvc`,
`burn-gptsovits` + `burn-vits`) so the trained weights load with a direct key
mapping — `Linear` weights transposed, weight-norm folded.

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

## Notes

Both exporters use PyTorch's modern **dynamo** ONNX exporter
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
