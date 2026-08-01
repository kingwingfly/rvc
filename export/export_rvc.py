"""Convert a Burn-trained RVC generator (safetensors) to ONNX.

Usage:
    uv run --project export python export_rvc.py voice.safetensors voice.onnx

The exported graph matches the `rvc-core` inference contract:
    inputs : phone [1,T,768] f32, phone_lengths [1] i64, pitch [1,T] i64,
             pitchf [1,T] f32, ds [1] i64, rnd [1,192,T] f32
    output : audio [1,1,L] f32
so the result drops straight into `rvc convert -m voice.onnx`.
"""

from __future__ import annotations

import sys
import warnings

import onnx
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


def dummy_inputs(t: int = 100) -> tuple[torch.Tensor, ...]:
    """The six graph inputs in positional order, matching the rvc-core contract."""
    return (
        torch.rand(1, t, 768),                              # phone        [1, T, 768] f32
        torch.tensor([t], dtype=torch.int64),               # phone_lengths[1]        i64
        torch.randint(1, 255, (1, t), dtype=torch.int64),   # pitch        [1, T]      i64
        torch.rand(1, t) * 200.0,                           # pitchf       [1, T]      f32
        torch.zeros(1, dtype=torch.int64),                  # ds           [1]         i64
        torch.randn(1, 192, t),                             # rnd          [1, 192, T] f32
    )


def export_model(model: Synthesizer, dst: str) -> None:
    """Export `model` to ONNX via the dynamo exporter, preserving the exact
    rvc-core graph contract, then validate the saved file.

    The sequence length is dynamic on phone dim1, pitch dim1, pitchf dim1 and
    rnd dim2; the trace ties them to one symbol, and the output audio length
    `L` follows from it. phone_lengths and ds stay static [1]. We use
    `Dim.DYNAMIC` (not `Dim.AUTO`): DYNAMIC *asserts* the axes stay dynamic and
    raises if the trace specializes one to the dummy length, whereas AUTO would
    silently bake in a fixed length — the whole point of this export is a
    variable sequence length. The exporter still infers the shared symbol from
    the trace without emitting the "axis name shares constraints" warning a
    single named `Dim` would. `dynamic_shapes` is a tuple aligned to the
    positional args of forward().
    """
    dyn = torch.export.Dim.DYNAMIC
    dynamic_shapes = (
        {1: dyn},    # phone
        {},          # phone_lengths (static [1])
        {1: dyn},    # pitch
        {1: dyn},    # pitchf
        {},          # ds (static [1])
        {2: dyn},    # rnd
    )
    with warnings.catch_warnings():
        # The dynamo export path (observed on the resolved torch 2.13)
        # deepcopies pytree TreeSpecs during run_decompositions, tripping
        # torch's own deprecated LeafSpec shim. That FutureWarning is internal
        # to torch and unrelated to how we call the exporter, so silence just
        # that one message to keep export clean; drop it once torch stops
        # emitting it.
        warnings.filterwarnings("ignore", message=r".*LeafSpec.*", category=FutureWarning)
        torch.onnx.export(
            model,
            dummy_inputs(),
            dst,
            input_names=["phone", "phone_lengths", "pitch", "pitchf", "ds", "rnd"],
            output_names=["audio"],
            dynamic_shapes=dynamic_shapes,
            opset_version=18,
            dynamo=True,
        )
    # full_check=True also runs shape inference, catching an internally
    # inconsistent graph (e.g. a mis-propagated dynamic dim) that the default
    # structural check would pass.
    onnx.checker.check_model(dst, full_check=True)
    print(f"wrote {dst} (onnx.checker: ok)")


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: export_rvc.py <in.safetensors> <out.onnx>")
    src, dst = sys.argv[1], sys.argv[2]

    st = load_file(src)
    model = Synthesizer(Config())
    model.load_state_dict(build_state_dict(model, st), strict=True)
    model.eval()

    export_model(model, dst)


if __name__ == "__main__":
    main()
