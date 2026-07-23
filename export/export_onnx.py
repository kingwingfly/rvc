"""Convert a Burn-trained RVC generator (safetensors) to ONNX.

Usage:
    uv run --project export python export_onnx.py voice.safetensors voice.onnx

The exported graph matches the `asmr-vc` inference contract:
    inputs : phone [1,T,768] f32, phone_lengths [1] i64, pitch [1,T] i64,
             pitchf [1,T] f32, ds [1] i64, rnd [1,192,T] f32
    output : audio [1,1,L] f32
so the result drops straight into `asmr convert -m voice.onnx`.
"""

from __future__ import annotations

import sys

import torch
from safetensors.torch import load_file

from rvc_infer import Config, Synthesizer

# Burn stores Linear weight as [in, out]; torch wants [out, in].
LINEAR_WEIGHTS = {"enc_p.emb_phone.weight", "dec.m_source.l_linear.weight"}


def fold_weight_norm(g: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Reconstruct `weight = g * v / ‖v‖` (norm over all dims except output)."""
    dims = list(range(1, v.dim()))
    norm = v.pow(2).sum(dim=dims, keepdim=True).sqrt()
    return g * v / norm


def build_state_dict(model: Synthesizer, st: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    missing: list[str] = []
    for name, param in model.named_parameters():
        if name in LINEAR_WEIGHTS:
            out[name] = st[name].t().contiguous()
        elif name in st:
            out[name] = st[name]
        elif name.endswith(".weight") and (name[:-7] + ".weight_g") in st:
            out[name] = fold_weight_norm(st[name[:-7] + ".weight_g"], st[name[:-7] + ".weight_v"])
        else:
            missing.append(name)
        if name in out and out[name].shape != param.shape:
            raise SystemExit(f"shape mismatch for {name}: got {tuple(out[name].shape)}, want {tuple(param.shape)}")
    if missing:
        raise SystemExit(f"no weights for {len(missing)} params, e.g. {missing[:5]}")
    return out


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: export_onnx.py <in.safetensors> <out.onnx>")
    src, dst = sys.argv[1], sys.argv[2]

    st = load_file(src)
    model = Synthesizer(Config())
    model.load_state_dict(build_state_dict(model, st), strict=True)
    model.eval()

    t = 100
    dummy = (
        torch.rand(1, t, 768),
        torch.tensor([t], dtype=torch.int64),
        torch.randint(1, 255, (1, t), dtype=torch.int64),
        torch.rand(1, t) * 200.0,
        torch.zeros(1, dtype=torch.int64),
        torch.randn(1, 192, t),
    )
    torch.onnx.export(
        model,
        dummy,
        dst,
        input_names=["phone", "phone_lengths", "pitch", "pitchf", "ds", "rnd"],
        output_names=["audio"],
        dynamic_axes={
            "phone": {1: "t"},
            "pitch": {1: "t"},
            "pitchf": {1: "t"},
            "rnd": {2: "t"},
            "audio": {2: "l"},
        },
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
