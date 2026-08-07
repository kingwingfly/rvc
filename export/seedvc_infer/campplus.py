"""CAM++ — the speaker embedding Seed-VC actually conditions on.

Mirrors `crates/burn-seedvc/src/campplus.rs` field for field, so
`campplus_cn_common.bin` loads by a direct key mapping — `export_seedvc.py`'s
`CAMPPLUS_REMAPS` and `campplus_remaps` are the same remaps
`CamPPlus::load_pytorch` applies, and they live there rather than here because
every one of the six mirrors is loaded by that one driver. This is a clean-room
reimplementation: it imports neither Seed-VC nor 3D-Speaker.

A reference clip in, one 192-dim timbre vector out. This is the network
upstream builds as `CAMPPlus(feat_dim=80, embedding_size=192)`, and its output
is the `style2` the diffusion transformer is conditioned on — so without it
there is no zero-shot conversion at all. It is **not** `net.style_encoder.*`,
the 18-tensor subtree of the Seed-VC checkpoint that upstream's `build_model`
never assembles; the weights here live in somebody else's release, Hugging Face
`funasr/campplus` (Apache-2.0), which upstream names verbatim.

`Norm` always normalises by the stored running statistics, where PyTorch's
`BatchNorm` switches on a `training` flag. The speaker encoder is frozen in
every Seed-VC path, so the batch-statistics branch would be dead code whose only
effect could be to make an embedding depend on what else was in the batch.

[`Graph`] at the bottom is `style.onnx`.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from seedvc_infer.fbank import Fbank, FbankConfig

# PyTorch's `BatchNorm` default, which upstream never overrides. No tensor in
# the checkpoint pins it — this follows from reading the constructor.
EPS = 1e-5

# Frames per segment in the context-aware mask's segment pooling. Upstream's
# `seg_len`, and at the 50 Hz this runs at it is a two-second window — long
# enough to average over a phrase rather than a phoneme.
SEG_LEN = 100


@dataclass(frozen=True)
class CamPPlusConfig:
    """The shape of the network. The defaults are `campplus_cn_common.bin`."""

    feat_dim: int = 80
    embedding_size: int = 192
    m_channels: int = 32
    init_channels: int = 128
    growth_rate: int = 32
    bn_size: int = 4
    # `(layers, kernel, dilation)` per dense block.
    blocks: tuple[tuple[int, int, int], ...] = ((12, 3, 1), (24, 3, 2), (16, 3, 2))

    @property
    def fcm_channels(self) -> int:
        """Channels the `Fcm` front end emits — three stride-2 stages take the
        bins down by 8, and what is left is flattened onto the channel axis.

        320 on the released shape, which is the first TDNN layer's input width.
        """
        return self.m_channels * (self.feat_dim // 8)


# ---- batch norm, in evaluation mode -----------------------------------------


class Norm(nn.Module):
    """`BatchNorm` frozen at its running statistics, with the checkpoint's own
    parameter names.

    Written out rather than reached for from `torch.nn` for two reasons. One
    module serves both ranks — `nn.BatchNorm1d` and `nn.BatchNorm2d` would be
    two, splitting a set of weights the checkpoint keeps under one shape. And
    normalising by the running statistics is unconditional here, where a
    `BatchNorm` decides it from a `training` flag: the speaker encoder is frozen
    in every Seed-VC path, so that branch could only ever make an embedding
    depend on what else was in the batch, and forgetting an `.eval()` would be
    the way it happened.

    `num_batches_tracked` has no inference role and is deliberately not a
    buffer; it is the one key per norm the exporter leaves unclaimed, 122 of
    them on the released file.
    """

    def __init__(self, channels: int, affine: bool = True) -> None:
        super().__init__()
        self.register_parameter("weight", nn.Parameter(torch.ones(channels)) if affine else None)
        self.register_parameter("bias", nn.Parameter(torch.zeros(channels)) if affine else None)
        self.register_buffer("running_mean", torch.zeros(channels))
        self.register_buffer("running_var", torch.ones(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normalise over the channel axis, whatever the rank — the same weights
        # serve `BatchNorm1d` on `[batch, channels, time]` and `BatchNorm2d` on
        # `[batch, channels, bins, time]`, because in both the channel is axis 1.
        shape = [1] * x.dim()
        shape[1] = -1
        y = (x - self.running_mean.view(shape)) / (self.running_var + EPS).sqrt().view(shape)
        if self.weight is not None:
            y = y * self.weight.view(shape) + self.bias.view(shape)
        return y


# ---- FCM: the 2-D front end -------------------------------------------------


class Shortcut(nn.Module):
    """The projection on a strided `ResBlock`'s skip path.

    Its own module because upstream builds it as an `nn.Sequential`, so the
    checkpoint spells it `shortcut.0` and `shortcut.1`.
    """

    def __init__(self, in_channels: int, channels: int, stride: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, channels, 1, stride=(stride, 1), bias=False)
        self.norm = Norm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.conv(x))


class ResBlock(nn.Module):
    """A 3x3 residual block over `[batch, channels, bins, time]`.

    The stride is `(2, 1)`: frequency is halved, time is untouched. That
    asymmetry is the whole point of the front end — it compresses *what the
    spectrum looks like* while leaving the frame rate for the TDNN stack.
    """

    def __init__(self, in_channels: int, channels: int, stride: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, channels, 3, stride=(stride, 1), padding=1, bias=False)
        self.bn1 = Norm(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = Norm(channels)
        # Present only where the stride makes the input and output shapes
        # differ, which is 2 of the released model's 4 residual blocks — the
        # first of each stage. Modelling it as optional is what makes this a
        # load with *nothing* missing, rather than ten absent parameters a
        # reader has to talk themselves out of.
        self.shortcut = (
            Shortcut(in_channels, channels, stride)
            if stride != 1 or in_channels != channels
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        skip = x if self.shortcut is None else self.shortcut(x)
        return F.relu(out + skip)


class Fcm(nn.Module):
    """An image of the filterbank in, a 320-channel sequence out."""

    def __init__(self, cfg: CamPPlusConfig) -> None:
        super().__init__()
        m = cfg.m_channels
        self.conv1 = nn.Conv2d(1, m, 3, padding=1, bias=False)
        self.bn1 = Norm(m)
        self.layer1 = nn.ModuleList([ResBlock(m, m, 2), ResBlock(m, m, 1)])
        self.layer2 = nn.ModuleList([ResBlock(m, m, 2), ResBlock(m, m, 1)])
        self.conv2 = nn.Conv2d(m, m, 3, stride=(2, 1), padding=1, bias=False)
        self.bn2 = Norm(m)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """`[batch, bins, frames]` → `[batch, channels * bins / 8, frames]`."""
        out = F.relu(self.bn1(self.conv1(x.unsqueeze(1))))
        for block in [*self.layer1, *self.layer2]:
            out = block(out)
        out = F.relu(self.bn2(self.conv2(out)))

        # Every dimension taken from the output rather than the input: the frame
        # count is meant to survive untouched, and reading it off the input
        # would turn a stride that quietly changed it into a reshape that
        # scrambles the tensor instead of failing.
        batch, channels, bins, frames = out.shape
        return out.reshape(batch, channels * bins, frames)


# ---- the TDNN stack ---------------------------------------------------------


class TdnnLayer(nn.Module):
    """A convolution followed by batch norm and a ReLU."""

    def __init__(self, in_channels: int, channels: int, kernel: int, stride: int) -> None:
        super().__init__()
        self.linear = nn.Conv1d(
            in_channels, channels, kernel, stride=stride, padding=(kernel - 1) // 2, bias=False
        )
        self.nonlinear = Norm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.nonlinear(self.linear(x)))


class TransitLayer(nn.Module):
    """Batch norm and a ReLU followed by a 1x1 convolution that halves the width.

    **The mirror image of `TdnnLayer`, and the order is what matters**: a dense
    block grows by concatenation, so something has to bring the width back down
    before the next block, and it does it *after* activating rather than before.
    Swapping the two runs, loads at 100% and changes every value downstream.
    """

    def __init__(self, in_channels: int, channels: int) -> None:
        super().__init__()
        self.nonlinear = Norm(in_channels)
        self.linear = nn.Conv1d(in_channels, channels, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(F.relu(self.nonlinear(x)))


class CamLayer(nn.Module):
    """Context-aware masking: a local convolution gated by pooled context.

    This is the "CAM" and the one piece worth reading twice. Every frame is
    scaled by what its neighbourhood and the utterance as a whole look like,
    which is how a speaker-level network suppresses content-level detail.
    """

    def __init__(self, bn_channels: int, channels: int, kernel: int, dilation: int) -> None:
        super().__init__()
        self.linear_local = nn.Conv1d(
            bn_channels,
            channels,
            kernel,
            dilation=dilation,
            padding=(kernel - 1) // 2 * dilation,
            bias=False,
        )
        # `reduction=2`, and upstream leaves these two with their default bias.
        self.linear1 = nn.Conv1d(bn_channels, bn_channels // 2, 1)
        self.linear2 = nn.Conv1d(bn_channels // 2, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        local = self.linear_local(x)
        # Two scales of context: the whole utterance, and a 2 s neighbourhood.
        context = x.mean(dim=2, keepdim=True) + segment_pool(x)
        gate = torch.sigmoid(self.linear2(F.relu(self.linear1(context))))
        return local * gate


def segment_pool(x: torch.Tensor) -> torch.Tensor:
    """Average each `SEG_LEN`-frame segment and broadcast it back over the
    segment's own frames.

    `ceil_mode` is what gives the final short window an average of its own
    rather than dropping it — but **what that window is divided by is not
    agreed between torch and ONNX**, and dividing the pooled ones by themselves
    is what makes this immune to the disagreement. `torch.avg_pool1d` divides a
    ceil-mode window by the frames actually in it; `torch.onnx.export` emits
    `AveragePool` with `count_include_pad=1`, so ONNX Runtime divides the same
    window by the full kernel of 100. Whatever the divisor, it is the same one
    for `x` and for the ones, so the ratio is the true mean either way.

    Measured before the fix, torch against ONNX Runtime on the same weights: a
    249-frame clip — one partial window and nothing else — embedded to cosine
    **0.940**, where a 3612-frame clip, whose tail is 6 frames in 1806, reached
    0.99995. A wrong divisor on the tail is therefore worst on exactly the
    1–30 s references this engine exists to take.
    """
    pooled = F.avg_pool1d(x, SEG_LEN, stride=SEG_LEN, ceil_mode=True)
    frames = F.avg_pool1d(torch.ones_like(x[:, :1]), SEG_LEN, stride=SEG_LEN, ceil_mode=True)
    return (pooled / frames).repeat_interleave(SEG_LEN, dim=2)[..., : x.shape[2]]


class CamDenseTdnnLayer(nn.Module):
    """One layer of a dense block: bottleneck, then a context-aware convolution."""

    def __init__(
        self, in_channels: int, channels: int, bn_channels: int, kernel: int, dilation: int
    ) -> None:
        super().__init__()
        self.nonlinear1 = Norm(in_channels)
        self.linear1 = nn.Conv1d(in_channels, bn_channels, 1, bias=False)
        self.nonlinear2 = Norm(bn_channels)
        self.cam_layer = CamLayer(bn_channels, channels, kernel, dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(F.relu(self.nonlinear1(x)))
        return self.cam_layer(F.relu(self.nonlinear2(x)))


class DenseLayer(nn.Module):
    """The final `1024 → 192` projection, with a non-affine batch norm after it."""

    def __init__(self, in_channels: int, channels: int) -> None:
        super().__init__()
        self.linear = nn.Conv1d(in_channels, channels, 1, bias=False)
        self.nonlinear = Norm(channels, affine=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.nonlinear(self.linear(x))


class XVector(nn.Module):
    """Everything after the front end, up to and including the embedding.

    Named for the `nn.Sequential` upstream calls `xvector`, because that is the
    prefix the checkpoint uses — including `xvector.dense`, which upstream has
    since moved out of the sequential and carries a compatibility remap for. The
    released file predates the move, and mirroring the file is what keeps the
    loader free of a remap that only exists to undo somebody else's refactor.
    """

    def __init__(self, cfg: CamPPlusConfig) -> None:
        super().__init__()
        # The only thing that halves the frame rate, 100 Hz to 50 Hz.
        self.tdnn = TdnnLayer(cfg.fcm_channels, cfg.init_channels, 5, 2)

        channels = cfg.init_channels
        blocks, transits = [], []
        for layers, kernel, dilation in cfg.blocks:
            blocks.append(
                nn.ModuleList(
                    CamDenseTdnnLayer(
                        channels + i * cfg.growth_rate,
                        cfg.growth_rate,
                        cfg.bn_size * cfg.growth_rate,
                        kernel,
                        dilation,
                    )
                    for i in range(layers)
                )
            )
            channels += layers * cfg.growth_rate
            transits.append(TransitLayer(channels, channels // 2))
            channels //= 2

        # Flat fields rather than a list, because the checkpoint names them
        # `block1`…`block3` and `transit1`…`transit3` rather than indexing them.
        self.block1, self.block2, self.block3 = blocks
        self.transit1, self.transit2, self.transit3 = transits
        self.out_nonlinear = Norm(channels)
        # Statistics pooling concatenates a mean and a deviation, so the
        # projection sees twice the channels the stack ends on.
        self.dense = DenseLayer(channels * 2, cfg.embedding_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """`[batch, channels, frames]` → `[batch, embedding, 1]`, the trailing
        axis being the one-frame sequence the pointwise projection works on."""
        h = self.tdnn(x)
        for block, transit in [
            (self.block1, self.transit1),
            (self.block2, self.transit2),
            (self.block3, self.transit3),
        ]:
            for layer in block:
                # Dense connectivity: every layer's output is kept *beside* its
                # input rather than replacing it, which is where the width goes.
                h = torch.cat([h, layer(h)], dim=1)
            h = transit(h)
        h = F.relu(self.out_nonlinear(h))

        # Statistics pooling — mean and standard deviation over time — is what
        # makes the reference length irrelevant. The deviation is
        # **Bessel-corrected**, matching `torch.std(unbiased=True)`, which is
        # what the Burn port's `var` is and what upstream computes.
        stats = torch.cat(
            [h.mean(dim=2, keepdim=True), h.var(dim=2, keepdim=True).sqrt()], dim=1
        )
        return self.dense(stats)


class CamPPlus(nn.Module):
    """The speaker encoder: a Kaldi filterbank in, a 192-dim timbre vector out."""

    def __init__(self, cfg: CamPPlusConfig = CamPPlusConfig()) -> None:
        super().__init__()
        self.head = Fcm(cfg)
        self.xvector = XVector(cfg)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """`features`: a Kaldi filterbank, `[batch, frames, bins]` → `[batch, embedding]`.

        **Frames before bins**, which is upstream's order and the opposite of
        every other tensor in this model — and both axes are 80 wide, so the two
        transpositions of the same clip differ only in which axis is which and
        neither will fail loudly. Keeping the transpose here, where upstream has
        it on the first line of `CAMPPlus.forward`, is what makes the two
        readable side by side.
        """
        return self.xvector(self.head(features.transpose(1, 2))).squeeze(2)


# ---- the exported graph -----------------------------------------------------


class Graph(nn.Module):
    """`audio [1, S]` mono f32 at 16 kHz → `style [1, 192]`.

    **The filterbank is inside the graph, mean subtraction included.** Leaving
    it to the host would put the one step this network cannot run without on the
    far side of a language boundary, where its absence is invisible: the
    embedding stays finite and repeatable and quietly stops separating speakers.

    `S` is dynamic because a reference is 1–30 s and the statistics pool is what
    makes the length irrelevant. There is no batch axis to make dynamic — a
    reference is one clip, and statistics pooled over another clip's padding
    would be wrong rather than merely different.
    """

    INPUTS = ["audio"]
    OUTPUTS = ["style"]

    def __init__(self, campplus: CamPPlus, cfg: FbankConfig = FbankConfig()) -> None:
        super().__init__()
        self.fbank = Fbank(cfg)
        self.campplus = campplus

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        return self.campplus(self.fbank(audio))

    def dummy(self) -> tuple[torch.Tensor, ...]:
        # Three seconds: past `SEG_LEN` frames, so the trace sees a partial
        # final segment rather than only a partial one.
        return (torch.randn(1, 3 * self.fbank.cfg.sample_rate) * 0.1,)

    def dynamic_shapes(self) -> tuple:
        return ({1: torch.export.Dim.DYNAMIC},)
