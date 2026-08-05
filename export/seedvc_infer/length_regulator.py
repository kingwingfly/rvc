"""Seed-VC's length regulator, mirroring `burn-seedvc`'s `InterpolateRegulator`.

Clean-room: written from `crates/burn-seedvc/src/length_regulator.rs`, not from
the Seed-VC repository, which is read as a porting reference and neither run nor
vendored.

One small convolutional stack sits between the frozen content encoder and the
diffusion transformer and does two things: it changes the width from Whisper's
768 to the transformer's 512, and it changes the **rate**, from Whisper's 50 Hz
to the mel's 22050/256 ≈ 86.13 Hz. It is the only thing in the model that
changes the frame rate, which is worth knowing before reading the next
paragraph.

**`sampling_ratios: [1, 1, 1, 1]` is not a ratio.** Upstream reads only its
*length* — one `conv → GroupNorm → Mish` stage per entry — and never looks at
the values, so it means "four stages", not "rate unchanged". A reader who takes
the name at face value concludes this module preserves the frame rate when it is
precisely what does not. The mirror therefore stores the stage *count* and says
where it came from, rather than carrying a list of ones nothing indexes.

**The nearest-neighbour resample is a graph input, not graph logic.** Upstream's
`F.interpolate(..., mode='nearest')` picks source frame
`min(floor(i · source / frames), source - 1)`, where `frames` is the mel length
the caller wants; expressed inside the graph that is data-dependent control flow
over a length ONNX would have to be told anyway. So `picks` comes in as a
`[T] i64` tensor the host computes — [`pick_indices`] below is the arithmetic,
and the Rust runtime is its real implementation — and the graph is a gather
followed by the conv stack. The same split is why the flow-matching Euler loop
stays on the host.

`embedding` (2048 × 512) and `mask_token` are the **discrete** path, which this
preset does not take: `config.yml` for `seed-uvit-whisper-small-wavenet` sets
`is_discrete: false`, so content stays continuous and the codebook is never
indexed. They are not built here, because a graph is not a checkpoint — nothing
would reference them and ONNX would drop them anyway. That is the opposite call
from `burn-seedvc`, which *does* hold them, and deliberately so: there a missing
parameter shows up as a false `unused` in a coverage report, whereas here the
exporter loads a named subset and an absent key is simply not asked for.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


# ---- the preset -------------------------------------------------------------


@dataclass
class RegulatorConfig:
    """The `seed-uvit-whisper-small-wavenet` numbers, from `SeedVcConfig`."""

    content_dim: int = 768
    hidden_dim: int = 512
    #: `len(sampling_ratios)` — see the module docstring for why it is a count.
    n_blocks: int = 4


def pick_indices(source: int, frames: int) -> torch.Tensor:
    """The nearest-neighbour source index per output frame — the `picks` input.

    Named apart from the tensor so it cannot be confused with the `picks`
    parameter the two `forward`s take. Kept beside the module it feeds so the
    two cannot drift, but this is **host** arithmetic and `seedvc-core` owns the
    copy that runs in anger. The
    float multiply-then-truncate is PyTorch's own `F.interpolate` rule and is
    reproduced rather than simplified: `floor(i * source / frames)` in integers
    disagrees with it wherever the division is inexact, which is every frame of
    a 50 Hz → 86.13 Hz resample.
    """
    scale = source / frames
    return torch.tensor(
        [min(int(i * scale), source - 1) for i in range(frames)], dtype=torch.int64
    )


# ---- the stack --------------------------------------------------------------


class RegulatorBlock(nn.Module):
    """One `Conv1d(k=3, pad=1) → GroupNorm → Mish` stage, length-preserving.

    The GroupNorm has a single group — upstream's `groups` defaults to 1 and
    nothing overrides it — so it normalises each frame across all 512 channels
    at once, which is a LayerNorm in everything but the parameter names.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, 3, padding=1)
        self.norm = nn.GroupNorm(1, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.mish(self.norm(self.conv(x)))


class InterpolateRegulator(nn.Module):
    """Content features in, transformer conditioning out.

    Field names are `burn-seedvc`'s, not upstream's flat `model.{0,1,3,4,...}`
    `nn.Sequential` indices: the exporter applies the same remap the Rust loader
    generates from the block count, so the two trees stay one tree.
    """

    def __init__(self, cfg: RegulatorConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.content_in_proj = nn.Linear(cfg.content_dim, cfg.hidden_dim)
        self.blocks = nn.ModuleList(RegulatorBlock(cfg.hidden_dim) for _ in range(cfg.n_blocks))
        self.out_conv = nn.Conv1d(cfg.hidden_dim, cfg.hidden_dim, 1)

    def forward(self, content: torch.Tensor, picks: torch.Tensor) -> torch.Tensor:
        """`[batch, source, content_dim]` at 50 Hz → `[batch, len(picks), hidden_dim]`.

        Upstream then multiplies by a sequence mask built from the per-sample
        `ylens`, which is all ones unless a batch mixes lengths — it never does
        at inference, where the batch is one clip — so the mask is not modelled.
        """
        x = self.content_in_proj(content).transpose(1, 2)
        x = x.index_select(2, picks)
        for block in self.blocks:
            x = block(x)
        return self.out_conv(x).transpose(1, 2)


# ---- the exported graph -----------------------------------------------------


class Graph(nn.Module):
    """`content [1, C, 768]`, `picks [T] i64` → `cond [1, T, 512]`.

    `C` and `T` are independent dynamic axes, which is the whole point of taking
    `picks` as an input: the graph never has to relate the two, and a caller
    that wants upstream's `length_adjust` stretch just hands over different
    indices.
    """

    INPUTS = ["content", "picks"]
    OUTPUTS = ["cond"]

    def __init__(self, regulator: InterpolateRegulator) -> None:
        super().__init__()
        self.regulator = regulator

    def forward(self, content: torch.Tensor, picks: torch.Tensor) -> torch.Tensor:
        return self.regulator(content, picks)

    def dummy(self) -> tuple[torch.Tensor, ...]:
        # A second of audio: 50 Whisper frames resampled onto 86 mel frames, so
        # the trace sees the two lengths actually differing.
        source, frames = 50, 86
        return (
            torch.randn(1, source, self.regulator.cfg.content_dim),
            pick_indices(source, frames),
        )

    def dynamic_shapes(self) -> tuple:
        dyn = torch.export.Dim.DYNAMIC
        return ({1: dyn}, {0: dyn})
