"""Convert Seed-VC's four checkpoints to the six ONNX graphs `seedvc --onnx <dir>` runs.

Usage:
    uv run --project export python export/export_seedvc.py <models-dir> <out-dir>
        [--dit DiT*.pth] [--campplus campplus_cn_common.bin]
        [--bigvgan bigvgan_generator.pt] [--content whisper-small/model.safetensors]

`<models-dir>` is a directory holding the four released checkpoints — the
Seed-VC checkpoint (`DiT_seed_v2_uvit_whisper_small_wavenet_bigvgan_pruned.pth`,
which holds the transformer *and* the length regulator), `campplus_cn_common.bin`
from `funasr/campplus`, `bigvgan_generator.pt` from
`nvidia/bigvgan_v2_22khz_80band_256x`, and `openai/whisper-small`'s
`model.safetensors` — each matched by what identifies it, exactly as the Rust
`seedvc_paths` does. The four flags override any of them, for a hand-assembled
set or a cache the names do not line up with.

Six graphs come out, one per `Graph` class under `seedvc_infer/`:
`content.onnx` (whisper-small's encoder), `style.onnx` (CAM++ + the Kaldi
filterbank), `mel.onnx` (the 22.05 kHz log-mel, weightless), `regulator.onnx`,
`dit.onnx` (the transformer and its WaveNet tail) and `bigvgan.onnx`. Each
module is a clean-room torch mirror of its Burn counterpart, so the released
weights load through the same key remaps the Rust loaders apply — repeated here
so a change on either side is a visible diff against the other.

**Weight-norm is not folded.** The transformer and WaveNet keep their
`weight_g`/`weight_v` parameters, and ONNX Runtime constant-folds the
`‖v‖` reconstruction at session init; only BigVGAN's plain convolutions, whose
checkpoint also stores the pair, are folded here. See `build_state_dict`.

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

from seedvc_infer import (
    bigvgan,
    campplus,
    content,
    dit,
    length_regulator,
    spectral,
)

# The remaps `Dit::load_pytorch`, `InterpolateRegulator::load_pytorch`,
# `CamPPlus::load_pytorch` and `BigVgan::load_pytorch` apply, in the same order.
# Kept verbatim so a change on either side is a visible diff against the other.
DIT_REMAPS = [
    (r"^net\.cfm\.module\.estimator\.", ""),
    (r"\.conv\.conv\.", "."),
    (r"\.mlp\.2\.", ".mlp.1."),
    (r"^final_layer\.adaLN_modulation\.1\.", "final_layer.ada_ln_modulation."),
]
# Upstream's length regulator is one `nn.Sequential`, so its keys are flat
# indices over a repeating conv/norm/Mish triple whose activation carries no
# tensors: `model.0`, `model.1`, then `model.3`, `model.4`, and so on, with the
# final width-1 conv at `model.12`. Generated from the block count, as the Rust
# side generates them, so a deeper preset stays in step with `Self::new`.
def regulator_remaps(n_blocks: int) -> list[tuple[str, str]]:
    return [
        (r"^net\.length_regulator\.module\.", ""),
        *((rf"^model\.{3 * i}\.", f"blocks.{i}.conv.") for i in range(n_blocks)),
        *((rf"^model\.{3 * i + 1}\.", f"blocks.{i}.norm.") for i in range(n_blocks)),
        (rf"^model\.{3 * n_blocks}\.", "out_conv."),
    ]


# CAM++'s three static remaps, all undoing an `nn.Sequential`'s positional or
# hard-coded child names. The `tdnnd{i+1}` → `{i}` pair is generated separately,
# because a `ModuleList` names its children one-based while this mirror numbers
# from zero — see `campplus_remaps`.
CAMPPLUS_REMAPS = [
    (r"\.batchnorm\.", "."),
    (r"\.shortcut\.0\.", ".shortcut.conv."),
    (r"\.shortcut\.1\.", ".shortcut.norm."),
]


def campplus_remaps(model: campplus.CamPPlus) -> list[tuple[str, str]]:
    """CAM++'s full remap list, `tdnnd` pairs generated from the block depths actually built."""
    longest = max(len(model.xvector.block1), len(model.xvector.block2), len(model.xvector.block3))
    return [
        *CAMPPLUS_REMAPS,
        *((rf"\.tdnnd{i + 1}\.", f".{i}.") for i in range(longest)),
    ]


BIGVGAN_REMAPS = [(r"^ups\.(\d+)\.0\.", r"ups.\1.")]
# The whole decoder lands in `unused`, and that is the expected result rather
# than a coverage gap: upstream deletes it, because Seed-VC wants acoustic
# features and never generates text.
CONTENT_REMAPS = [(r"^model\.encoder\.", "")]


def read_checkpoint(path: Path, top_level_key: str | None, remaps) -> dict:
    """A checkpoint's tensors under this project's module names.

    `top_level_key` selects the state dict inside the file (`"generator"` for
    BigVGAN's); `None` for the Seed-VC checkpoint, whose tensors sit under a
    nested `net` map and are flattened here rather than read flat. A
    `.safetensors` is whisper-small's, in Hugging Face's own PyTorch layout —
    Seed-VC has no training, so there is no Burn-produced file to transpose.
    """
    raw = (
        load_file(str(path))
        if path.suffix == ".safetensors"
        else torch.load(str(path), map_location="cpu", weights_only=True)
    )
    if top_level_key is not None:
        raw = raw[top_level_key]
    out = {}
    for name, tensor in _flatten(raw).items():
        for pattern, replacement in remaps:
            name = re.sub(pattern, replacement, name)
        out[name] = tensor
    return out


def _flatten(state: dict) -> dict:
    """PyTorch's nested state dicts under the dotted names `pytorch_keys` reads."""
    out = {}
    for key, value in state.items():
        if isinstance(value, dict):
            for name, tensor in _flatten(value).items():
                out[f"{key}.{name}"] = tensor
        else:
            out[key] = value
    return out


def fold_weight_norm(g: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Reconstruct `weight = g * v / ‖v‖`.

    Which axes the norm runs over is read from `g`'s shape rather than assumed:
    the BigVGAN convolutions weight-norm with `dim=0`, so `g` is `[out,1,1]`
    and the norm runs over the input and kernel axes.
    """
    dims = [d for d in range(v.dim()) if g.shape[d] == 1]
    return g * v / v.pow(2).sum(dim=dims, keepdim=True).sqrt()


def build_state_dict(model: torch.nn.Module, src: dict, what: str) -> dict:
    """Map `src` onto `model`'s parameters, folding weight-norm as it goes.

    A parameter found verbatim loads directly — that is the transformer's and
    the WaveNet's whole story, whose `weight_g`/`weight_v` stay as three
    initializers per layer and are reconstructed by ONNX Runtime's constant
    folder at session init. A parameter that exists only as the folded `.weight`
    (BigVGAN's plain convolutions) is rebuilt from its pair. Buffers come from
    `src` when the checkpoint carries them (CAM++'s running statistics are
    trained numbers) and are derived otherwise (the DFT kernels every front end
    builds), which is what lets `load_state_dict` stay strict — that is what
    catches a parameter this mapping forgot.
    """
    out = {}
    for name, buffer in model.named_buffers():
        if name in src:
            out[name] = src[name].to(torch.float32).contiguous()
        else:
            out[name] = buffer

    missing: list[str] = []
    for name, param in model.named_parameters():
        if name in src:
            tensor = src[name]
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


def resolve(flag: Path | None, models: Path, what: str, find) -> Path:
    """A flag's checkpoint, or the path under `models` that identifies it.

    Each checkpoint is matched by what identifies it rather than by its full
    upstream name, because a cache, a clone and a hand-made copy name these
    files differently — the same tolerance `seedvc_paths` has.

    `find` is handed the entry **and its lowercased name**, and the matchers
    below compare against the second. That is not tidiness: `seedvc_paths`
    lowercases every filename before matching, and the released checkpoint is
    `DiT_seed_v2_uvit_whisper_small_wavenet_bigvgan_pruned.pth`, so a
    case-sensitive `startswith("dit")` finds nothing and the two sides disagree
    about a directory both claim to read the same way.
    """
    if flag is not None:
        return flag
    if not models.is_dir():
        raise SystemExit(f"{models} is not a directory")
    for entry in sorted(models.iterdir()):
        if found := find(entry, entry.name.lower()):
            return found
    raise SystemExit(f"no {what} under {models}")


def export(graph: torch.nn.Module, dst: Path) -> None:
    """Export one graph and validate the file that comes out.

    `Dim.DYNAMIC` rather than `Dim.AUTO`: DYNAMIC *asserts* the axis stays
    dynamic and raises if the trace specialises it to the dummy length, whereas
    AUTO silently bakes in a fixed one — and a fixed sequence length would make
    every one of these graphs useless.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    graph.eval()
    with warnings.catch_warnings():
        # Internal to torch's dynamo path (deepcopying pytree TreeSpecs during
        # run_decompositions trips its own deprecated LeafSpec shim), unrelated
        # to how the exporter is called here. Drop once torch stops emitting it.
        warnings.filterwarnings("ignore", message=r".*LeafSpec.*", category=FutureWarning)
        torch.onnx.export(
            graph,
            graph.dummy(),
            str(dst),
            input_names=graph.INPUTS,
            output_names=graph.OUTPUTS,
            dynamic_shapes=graph.dynamic_shapes(),
            opset_version=18,
            dynamo=True,
        )
    onnx.checker.check_model(str(dst), full_check=True)
    print(f"wrote {dst} (onnx.checker: ok)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", type=Path, help="directory holding the four checkpoints")
    parser.add_argument("out", type=Path, help="where to write the .onnx graphs")
    parser.add_argument("--dit", type=Path, help="the DiT checkpoint (default: the DiT*.pth under <models>)")
    parser.add_argument("--campplus", type=Path, help="campplus_cn_common.bin (default: <models>/campplus*.bin)")
    parser.add_argument("--bigvgan", type=Path, help="bigvgan_generator.pt (default: <models>/bigvgan*.pt)")
    parser.add_argument("--content", type=Path, help="whisper-small's model.safetensors (default: <models>/<whisper dir>/model.safetensors)")
    parser.add_argument(
        "--only",
        choices=["content", "style", "mel", "regulator", "dit", "bigvgan"],
        action="append",
        help="export only these graphs (repeatable); default is all six",
    )
    opts = parser.parse_args()
    wanted = set(opts.only or ["content", "style", "mel", "regulator", "dit", "bigvgan"])

    with torch.no_grad():
        if "content" in wanted:
            encoder = content.ContentEncoder()
            path = resolve(
                opts.content,
                opts.models,
                "a whisper-small directory (holding model.safetensors)",
                lambda e, _n: e / "model.safetensors" if e.is_dir() and (e / "model.safetensors").exists() else None,
            )
            state = read_checkpoint(path, None, CONTENT_REMAPS)
            encoder.encoder.load_state_dict(build_state_dict(encoder.encoder, state, "content"), strict=True)
            print(f"loaded content encoder from {path.parent.name} (decoder unused)")
            export(content.Graph(encoder), opts.out / "content.onnx")

        if "style" in wanted:
            model = campplus.CamPPlus()
            path = resolve(
                opts.campplus,
                opts.models,
                "campplus_cn_common.bin",
                lambda e, n: e if n.startswith("campplus") and n.endswith(".bin") else None,
            )
            state = read_checkpoint(path, None, campplus_remaps(model))
            model.load_state_dict(build_state_dict(model, state, "campplus"), strict=True)
            print(f"loaded campplus from {path.name}")
            export(campplus.Graph(model), opts.out / "style.onnx")

        if "mel" in wanted:
            export(spectral.Graph(spectral.Spectral(spectral.SpectralConfig())), opts.out / "mel.onnx")

        # The transformer and the length regulator share one checkpoint, so the
        # 440 MB file is read once here and each graph takes its own remap set.
        if "dit" in wanted or "regulator" in wanted:
            path = resolve(
                opts.dit,
                opts.models,
                "a DiT*.pth checkpoint",
                lambda e, n: e if n.startswith("dit") and n.endswith(".pth") else None,
            )
            if "dit" in wanted:
                model = dit.Dit()
                state = read_checkpoint(path, None, DIT_REMAPS)
                model.load_state_dict(build_state_dict(model, state, "dit"), strict=True)
                print(f"loaded dit from {path.name}")
                export(dit.Graph(model), opts.out / "dit.onnx")
            if "regulator" in wanted:
                model = length_regulator.InterpolateRegulator(length_regulator.RegulatorConfig())
                state = read_checkpoint(path, None, regulator_remaps(model.cfg.n_blocks))
                model.load_state_dict(build_state_dict(model, state, "regulator"), strict=True)
                print(f"loaded regulator from {path.name} (embedding, mask_token unused)")
                export(length_regulator.Graph(model), opts.out / "regulator.onnx")

        if "bigvgan" in wanted:
            model = bigvgan.BigVgan(bigvgan.BigVganConfig())
            path = resolve(
                opts.bigvgan,
                opts.models,
                "bigvgan_generator.pt",
                lambda e, n: e if "bigvgan" in n and n.endswith(".pt") and "discriminator" not in n else None,
            )
            state = read_checkpoint(path, "generator", BIGVGAN_REMAPS)
            model.load_state_dict(build_state_dict(model, state, "bigvgan"), strict=True)
            print(f"loaded bigvgan from {path.name}")
            export(bigvgan.Graph(model), opts.out / "bigvgan.onnx")


if __name__ == "__main__":
    sys.exit(main())
