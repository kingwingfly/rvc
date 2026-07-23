# rvc-onnx-export

Convert a Burn-trained RVC generator (`.safetensors`, from `rvc train`) to an
ONNX model for deployment with ONNX Runtime.

This is the **only** Python in the toolkit — a small, standalone `uv` project. It
does **not** depend on the RVC-Project repository: `rvc_infer.py` is a clean-room
torch reimplementation of the inference generator that mirrors the `burn-rvc`
module layout, so the trained weights load with a direct mapping (Linear weights
transposed, weight-norm folded).

## Use

```sh
uv run --project export python export/export_onnx.py \
    models/voice.safetensors  models/voice.onnx
```

Then deploy through ONNX Runtime:

```sh
rvc convert --backend onnx -m models/voice.onnx --model-sr 48000 -o out/  in.mp3
```

The exported graph matches the `rvc-core` contract:

```
inputs : phone [1,T,768] f32, phone_lengths [1] i64, pitch [1,T] i64,
         pitchf [1,T] f32, ds [1] i64, rnd [1,192,T] f32
output : audio [1,1,L] f32
```

Native Burn inference (`rvc convert --backend burn -m voice.safetensors`) needs
no export — this is only for ONNX Runtime / cross-framework deploy.
