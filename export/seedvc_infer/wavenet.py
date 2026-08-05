"""Standalone torch implementation of Seed-VC's WaveNet tail.

Mirrors `crates/burn-seedvc/src/wavenet.rs` field for field, so the released
checkpoint loads onto it by direct key mapping. This is a clean-room
reimplementation written from the Burn port: it does not import, vendor or run
the Seed-VC repository.

The block has no graph of its own — it is the second half of `dit.onnx` — so
there is no `Graph` class here. What it does own is the one piece of arithmetic
a reader cannot get from the call site: **the convolutions reflect-pad**, which
is upstream's `SConv1d` ignoring the `padding` argument it is handed. See
[`reflect_pad`].

Weight-norm is **not** folded, unlike `gptsovits_infer.py`. The exporter loads
the checkpoint's `weight_g`/`weight_v` pairs straight onto these parameters, so
folding here would mean the mirror's parameter names stopped matching the file
it is fed.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


# ---- weight norm ------------------------------------------------------------


class WeightNormConv1d(nn.Module):
    """`torch.nn.utils.weight_norm(nn.Conv1d)`, mirroring `burn_vits::WeightNormConv1d`.

    `weight_g` is `[out, 1, 1]` and `weight_v` is `[out, in, kernel]` — weight
    norm's default `dim=0`, one magnitude per output channel. The norm therefore
    runs over the input and kernel axes and nothing else.
    """

    def __init__(self, in_ch: int, out_ch: int, kernel: int) -> None:
        super().__init__()
        self.weight_g = nn.Parameter(torch.ones(out_ch, 1, 1))
        self.weight_v = nn.Parameter(torch.randn(out_ch, in_ch, kernel) * 0.02)
        self.bias = nn.Parameter(torch.zeros(out_ch))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        v = self.weight_v
        weight = v * (self.weight_g / v.pow(2).sum(dim=(1, 2), keepdim=True).sqrt())
        return F.conv1d(x, weight, self.bias)


# ---- the block --------------------------------------------------------------


def reflect_pad(x: torch.Tensor, pad: int) -> torch.Tensor:
    """Pad `x` by reflection, `pad` frames at each end.

    `WN` builds each layer as `SConv1d(…, padding=(k·d − d)/2, …)`, but
    `SConv1d.__init__` never forwards `padding` to the `nn.Conv1d` it wraps: the
    argument disappears into `**kwargs` and the convolution is built unpadded.
    `SConv1d.forward` then reflect-pads the *input* instead. **Reading the call
    site rather than the wrapper is how this mirror would have acquired zero
    padding**, which is not an error anywhere — it simply pulls the first and
    last `pad` frames of every layer towards silence, eight times over.
    """
    return x if pad == 0 else F.pad(x, (pad, pad), mode="reflect")


class WaveNet(nn.Module):
    """Seed-VC's `WN`: gated convolutions with **per-layer** conditioning.

    Not VITS's `WN`, which adds one broadcast `g` to every layer. This one
    projects `g` once into `2 · hidden · layers` channels and reads a different
    window of it per layer — which is what the checkpoint's `cond_layer` weight
    of `[8192, 512, 1]` records, `2 × 512 × 8`.

    Upstream's `x_mask` and its `p_dropout: 0.2` are both absent: dropout is a
    training-time term, and the mask is all ones for the single clip inference
    ever passes (the classifier-free-guidance pair being that clip twice).
    """

    def __init__(self, hidden: int, layers: int, kernel: int, dilation: int) -> None:
        super().__init__()
        if dilation != 1:
            raise ValueError("only dilation_rate 1 is mirrored; a dilating preset needs per-layer padding")
        self.hidden = hidden
        self.pad = (kernel - 1) // 2
        self.in_layers = nn.ModuleList(
            WeightNormConv1d(hidden, 2 * hidden, kernel) for _ in range(layers)
        )
        # The last layer feeds only the skip sum, so it needs no residual half —
        # which is why `res_skip_layers.7` is `[512, …]` where the other seven
        # are `[1024, …]`.
        self.res_skip_layers = nn.ModuleList(
            WeightNormConv1d(hidden, 2 * hidden if i + 1 < layers else hidden, 1)
            for i in range(layers)
        )
        self.cond_layer = WeightNormConv1d(hidden, 2 * hidden * layers, 1)

    def forward(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """`x`: `[batch, hidden, frames]`, `g`: `[batch, hidden, 1]` → `[batch, hidden, frames]`."""
        g = self.cond_layer(g)
        n, h = len(self.in_layers), self.hidden

        output = torch.zeros_like(x)
        for i in range(n):
            acts = self.in_layers[i](reflect_pad(x, self.pad)) + g[:, i * 2 * h : (i + 1) * 2 * h]
            # `fused_add_tanh_sigmoid_multiply`: first half the value, second the gate.
            acts = torch.tanh(acts[:, :h]) * torch.sigmoid(acts[:, h:])

            res_skip = self.res_skip_layers[i](acts)
            if i + 1 < n:
                x = x + res_skip[:, :h]
                output = output + res_skip[:, h:]
            else:
                output = output + res_skip
        return output
