"""Convert GPT-SoVITS weights to the four ONNX graphs `tts --backend onnx` runs.

Usage:
    uv run --project export python export/export_gptsovits.py <models-dir> <out-dir>
        [--s1 fine-tuned.safetensors] [--s2 fine-tuned.safetensors]

`<models-dir>` is what `tts --models` points at: `chinese-hubert-base/`, an
`s1*.ckpt` and an `s2G*.pth`. `--s1`/`--s2` override either stage with a Burn
`.safetensors` — what `tts train` writes — which is the whole reason this
exporter exists: Burn can import ONNX but cannot emit it, so a fine-tuned voice
reaches ONNX Runtime only through a torch reimplementation of the same layout.

Both weight layouts load. An original `.pth`/`.ckpt` needs the same key remaps
the Rust loaders apply (they are repeated below so the two can be diffed); a
Burn `.safetensors` needs its `Linear` weights transposed and its norm
parameters renamed from `gamma`/`beta`. Weight-norm is folded either way.

The graph contracts are in `export/README.md`.
"""

from __future__ import annotations

import argparse
import re
import sys
import warnings
from pathlib import Path

import onnx
import torch
from safetensors.torch import load_file

from gptsovits_infer import (
    Hubert,
    HubertConfig,
    ReferenceGraph,
    S1Prompt,
    S1Step,
    S2,
    Sovits,
    SovitsConfig,
    T2s,
    T2sConfig,
)

# The remaps `SovitsPartial::load_pytorch`, `T2s::load_pytorch` and
# `Hubert::load_pytorch` apply, in the same order. Kept verbatim so a change on
# either side is a visible diff against the other.
S2_REMAPS = [
    (r"^enc_p\.(encoder_ssl|encoder_text|encoder2)\.attn_layers\.(\d+)\.", r"enc_p.\1.layers.\2.attn."),
    (r"^enc_p\.(encoder_ssl|encoder_text|encoder2)\.norm_layers_1\.(\d+)\.", r"enc_p.\1.layers.\2.norm_1."),
    (r"^enc_p\.(encoder_ssl|encoder_text|encoder2)\.ffn_layers\.(\d+)\.", r"enc_p.\1.layers.\2.ffn."),
    (r"^enc_p\.(encoder_ssl|encoder_text|encoder2)\.norm_layers_2\.(\d+)\.", r"enc_p.\1.layers.\2.norm_2."),
    (r"^ssl_proj\.", "quantizer.ssl_proj."),
    (r"^quantizer\.vq\.layers\.(\d+)\._codebook\.", r"quantizer.vq.layers.\1."),
    (r"^ref_enc\.spectral\.3\.", "ref_enc.spectral.1."),
    (r"^flow\.flows\.2\.", "flow.flows.1."),
    (r"^flow\.flows\.4\.", "flow.flows.2."),
    (r"^flow\.flows\.6\.", "flow.flows.3."),
]
S1_REMAPS = [(r"^model\.", "")]
HUBERT_REMAPS = [(r"^encoder\.pos_conv_embed\.conv\.", "encoder.pos_conv_embed.")]


def read_checkpoint(path: Path, top_level_key: str | None, remaps) -> tuple[dict, bool]:
    """A checkpoint's tensors under this project's module names, and whether they
    came out of Burn.

    A `.safetensors` here is always one this toolkit saved (`tts train`), so it
    is already in the module tree's own naming and needs no remapping — only the
    layout fixes `build_state_dict` applies.
    """
    if path.suffix == ".safetensors":
        return load_file(str(path)), True

    raw = torch.load(str(path), map_location="cpu", weights_only=False)
    if top_level_key is not None:
        raw = raw[top_level_key]
    out = {}
    for name, tensor in raw.items():
        for pattern, replacement in remaps:
            name = re.sub(pattern, replacement, name)
        out[name] = tensor
    return out, False


def fold_weight_norm(g: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Reconstruct `weight = g * v / ‖v‖`.

    Which axes the norm runs over is read from `g`'s shape rather than assumed:
    the VITS convolutions normalise over output channels (`g` is `[out,1,1]`)
    and HuBERT's positional convolution over the kernel (`g` is `[1,1,k]`), and
    using one rule for both loads fine while scaling half the model wrongly.
    """
    dims = [d for d in range(v.dim()) if g.shape[d] == 1]
    return g * v / v.pow(2).sum(dim=dims, keepdim=True).sqrt()


def build_state_dict(model: torch.nn.Module, src: dict, burn: bool, what: str) -> dict:
    """Map `src` onto `model`'s parameters, folding weight-norm as it goes."""
    # Which parameters need a layout fix is decided by the module that holds
    # them, not by their name: `norm1.weight` is a LayerNorm's and `linear1.weight`
    # is a Linear's, and only one of each pair moves.
    transpose, renamed = set(), {}
    for name, module in model.named_modules():
        prefix = f"{name}." if name else ""
        if isinstance(module, torch.nn.Linear):
            transpose.add(f"{prefix}weight")
        elif isinstance(module, (torch.nn.LayerNorm, torch.nn.GroupNorm)):
            renamed[f"{prefix}weight"] = f"{prefix}gamma"
            renamed[f"{prefix}bias"] = f"{prefix}beta"

    # Buffers (the sinusoidal table, the DFT kernels) are computed, not trained;
    # they are carried through so `load_state_dict` can stay strict, which is
    # what catches a parameter this mapping forgot.
    out = {name: buffer for name, buffer in model.named_buffers()}
    missing: list[str] = []
    for name, param in model.named_parameters():
        key = renamed.get(name, name) if burn else name
        if key in src:
            tensor = src[key]
            if burn and name in transpose:
                tensor = tensor.t()
        elif name.endswith(".weight") and f"{name[:-7]}.weight_g" in src:
            base = name[:-7]
            tensor = fold_weight_norm(src[f"{base}.weight_g"], src[f"{base}.weight_v"])
        else:
            missing.append(name)
            continue
        tensor = tensor.to(torch.float32).contiguous()
        if tensor.shape != param.shape:
            raise SystemExit(
                f"{what}: shape mismatch for {name}: got {tuple(tensor.shape)}, "
                f"want {tuple(param.shape)}"
            )
        out[name] = tensor

    if missing:
        raise SystemExit(f"{what}: no weights for {len(missing)} params, e.g. {missing[:5]}")
    return out


def load_models(models: Path, s1_override: Path | None, s2_override: Path | None):
    """The three networks, loaded and in eval mode."""
    hubert_path = next(
        (p for p in [models / "chinese-hubert-base/pytorch_model.bin", models / "pytorch_model.bin"] if p.exists()),
        None,
    )
    if hubert_path is None:
        raise SystemExit(f"chinese-hubert-base/pytorch_model.bin not found under {models}")

    def find(prefix: str, ext: str) -> Path:
        for directory in [models / "gsv-v2final-pretrained", models]:
            if not directory.is_dir():
                continue
            for entry in sorted(directory.iterdir()):
                if entry.name.startswith(prefix) and entry.name.endswith(ext):
                    return entry
        raise SystemExit(f"no {prefix}*{ext} under {models}")

    hubert = Hubert(HubertConfig())
    state, burn = read_checkpoint(hubert_path, None, HUBERT_REMAPS)
    hubert.load_state_dict(build_state_dict(hubert, state, burn, "cnhubert"), strict=True)

    sovits = Sovits(SovitsConfig())
    s2_path = s2_override or find("s2G", ".pth")
    state, burn = read_checkpoint(s2_path, "weight", S2_REMAPS)
    sovits.load_state_dict(build_state_dict(sovits, state, burn, "s2"), strict=True)

    t2s = T2s(T2sConfig())
    s1_path = s1_override or find("s1", ".ckpt")
    state, burn = read_checkpoint(s1_path, "weight", S1_REMAPS)
    t2s.load_state_dict(build_state_dict(t2s, state, burn, "s1"), strict=True)

    for model in (hubert, sovits, t2s):
        model.eval()
    print(f"loaded cnhubert {hubert_path.name}, s2 {s2_path.name}, s1 {s1_path.name}")
    return hubert, sovits, t2s


def export(model: torch.nn.Module, args, dst: Path, inputs, outputs, dynamic) -> None:
    """Export one graph and validate the file that comes out.

    `Dim.DYNAMIC` rather than `Dim.AUTO`: DYNAMIC *asserts* the axis stays
    dynamic and raises if the trace specialises it to the dummy length, whereas
    AUTO silently bakes in a fixed one — and a fixed sequence length would make
    every one of these graphs useless.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    with warnings.catch_warnings():
        # Internal to torch's dynamo path (deepcopying pytree TreeSpecs during
        # run_decompositions trips its own deprecated LeafSpec shim), unrelated
        # to how the exporter is called here. Drop once torch stops emitting it.
        warnings.filterwarnings("ignore", message=r".*LeafSpec.*", category=FutureWarning)
        torch.onnx.export(
            model,
            args,
            str(dst),
            input_names=inputs,
            output_names=outputs,
            dynamic_shapes=dynamic,
            opset_version=18,
            dynamo=True,
        )
    onnx.checker.check_model(str(dst), full_check=True)
    print(f"wrote {dst} (onnx.checker: ok)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", type=Path, help="directory tts --models points at")
    parser.add_argument("out", type=Path, help="where to write the .onnx graphs")
    parser.add_argument("--s1", type=Path, help="fine-tuned s1 (.safetensors from `tts train`)")
    parser.add_argument("--s2", type=Path, help="fine-tuned s2 (.safetensors)")
    parser.add_argument(
        "--only",
        choices=["reference", "s1", "s2"],
        action="append",
        help="export only these graphs (repeatable); default is all four",
    )
    opts = parser.parse_args()
    wanted = set(opts.only or ["reference", "s1", "s2"])

    # `reference.onnx` is built from `sovits` too — it carries the quantiser and
    # `ref_enc`, which live in the same checkpoint as the decoder. Leaving either
    # half at whatever was in the output directory produces a bundle that loads,
    # runs and sounds wrong: the prompt tokens and the speaker vector come from
    # one model and the decoder from another. Nothing downstream can detect it,
    # so refuse unless *both* are re-exported — `--only reference` alone is the
    # same mismatch mirrored, with a tuned front end feeding a base decoder.
    if opts.s2 and not {"reference", "s2"} <= wanted:
        stale = sorted({"reference", "s2"} - wanted)
        raise SystemExit(
            "--s2 changes reference.onnx and s2.onnx together: the quantiser and "
            "ref_enc come from the same checkpoint as the decoder, so exporting "
            f"one without the other leaves {', '.join(f'{s}.onnx' for s in stale)} "
            "from a different model. Pass `--only reference --only s2`, or drop "
            "--only and export the whole bundle."
        )

    hubert, sovits, t2s = load_models(opts.models, opts.s1, opts.s2)
    dyn = torch.export.Dim.DYNAMIC
    cfg = T2sConfig()
    n_layer, d_model = cfg.n_layer, cfg.model_dim

    with torch.no_grad():
        if "reference" in wanted:
            export(
                ReferenceGraph(hubert, sovits),
                (torch.randn(1, 48_000) * 0.1,),
                opts.out / "reference.onnx",
                ["audio"],
                ["codes", "speaker"],
                ({1: dyn},),
            )

        if "s2" in wanted:
            tokens, phones = 40, 12
            export(
                S2(sovits),
                (
                    torch.randint(0, 1024, (1, tokens)),
                    torch.randint(0, 732, (1, phones)),
                    torch.randn(1, 512, 1),
                    torch.randn(1, 192, tokens * 2),
                ),
                opts.out / "s2.onnx",
                ["codes", "text", "speaker", "noise"],
                ["audio"],
                ({1: dyn}, {1: dyn}, {}, {2: dyn}),
            )

        if "s1" in wanted:
            phones, prompt = 20, 30
            export(
                S1Prompt(t2s),
                (
                    torch.randint(0, 732, (1, phones)),
                    torch.randn(1, phones, cfg.bert_dim),
                    torch.randint(0, 1024, (1, prompt)),
                ),
                opts.out / "s1_prompt.onnx",
                ["phones", "bert", "prompt"],
                ["logits"] + cache_names("present", n_layer),
                ({1: dyn}, {1: dyn}, {1: dyn}),
            )

            past = [torch.randn(1, phones + prompt, d_model) for _ in range(2 * n_layer)]
            export(
                S1Step(t2s),
                (torch.randint(0, 1024, (1, 1)), torch.tensor([prompt]), *past),
                opts.out / "s1_step.onnx",
                ["token", "position"] + cache_names("past", n_layer),
                ["logits"] + cache_names("present", n_layer),
                # The cache arrives as `*past`, so `torch.export` sees three
                # arguments, not fifty: its shapes go in one nested list.
                ({}, {}, tuple({1: dyn} for _ in past)),
            )


def cache_names(prefix: str, n_layer: int) -> list[str]:
    return [f"{prefix}.{i}.{slot}" for i in range(n_layer) for slot in ("key", "value")]


if __name__ == "__main__":
    sys.exit(main())
