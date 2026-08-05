"""Standalone torch implementation of Seed-VC's U-ViT diffusion transformer.

Mirrors `crates/burn-seedvc/src/dit.rs` field for field, so the released
checkpoint — 255 of its 302 tensors are this network and its WaveNet tail —
loads by direct key mapping. This is a clean-room reimplementation written from
the Burn port: it does not import, vendor or run the Seed-VC repository.

What the network computes is a **velocity field**: given a partly-noised mel and
a flow time `t`, it predicts the direction that mel should move in. Integrating
that is [`crates/burn-seedvc/src/flow.rs`]'s job and stays on the host, so
[`Graph`] is one velocity evaluation and nothing more — the same division the
`s1` graphs make, and what keeps the exported file a pure function of its
inputs.

Five things here are silent if mirrored wrongly, all of them invisible to a
weight-coverage count:

- **RoPE rotates adjacent pairs**, `(x[0], x[1])`, `(x[2], x[3])`, … — the
  `gpt-fast` convention, not Llama's half-and-half split.
- [`AdaLayerNorm`] splits its projection **scale then shift**; [`FinalLayer`]
  splits **shift then scale**. Same-looking arithmetic, opposite order, because
  upstream writes the two chunks in opposite orders in the two places.
- The U-ViT skips are a **LIFO stack**: layers below the middle push, layers
  above it pop, and the middle layer does neither. Depth 13 is odd by design —
  that is what makes the two lists the same length.
- `t_embedder` and `t_embedder2` are **two independent embedders**, one for the
  transformer and the output head, one for the WaveNet. Wiring one to both loads
  fine and halves the conditioning capacity.
- The final norm is affine-free with **eps 1e-6**, where every RMSNorm in the
  stack uses 1e-5.

Four tensors upstream allocates and never reaches — `x_embedder`,
`cond_embedder`, `content_mask_embedder`, `f0_embedder` — plus the `input_pos`
buffer are **deliberately absent**. `dit.rs` holds them so its coverage report
stays honest; a mirror that built them would only oblige the exporter's loader
to feed parameters no graph reads.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from seedvc_infer.wavenet import WaveNet

# `ModelArgs.norm_eps`, shared by every RMSNorm in the stack.
NORM_EPS = 1e-5
# The output head's affine-free LayerNorm. Deliberately not `NORM_EPS`.
FINAL_NORM_EPS = 1e-6
# Base of the rotary ladder and of the timestep code — upstream reuses 10000 for both.
FREQ_BASE = 10_000.0
# The period both sinusoidal ladders are reduced by — see `rope_tables`.
TWO_PI = 2.0 * math.pi
# Width of the sinusoidal timestep code before the MLP sees it, pinned
# independently by `mlp.0.weight` being `[512, 256]`.
TIME_FREQ_DIM = 256


@dataclass
class DitConfig:
    """The `seed-uvit-whisper-small-wavenet` preset, from
    `SeedVcConfig::uvit_whisper_small_wavenet` — only the fields this network reads.

    `sampling_ratios`, the codebook sizes and the audio rates belong to the
    modules that use them and are not repeated here.
    """

    n_mels: int = 80
    hidden_dim: int = 512
    style_dim: int = 192
    depth: int = 13
    heads: int = 8
    block_size: int = 8192
    wavenet_layers: int = 8
    wavenet_kernel: int = 5
    wavenet_dilation: int = 1

    def ffn_hidden(self) -> int:
        """`ModelArgs.intermediate_size` when unset: `find_multiple(2/3 · 4 · dim, 256)`.

        1536 at dim 512 — exactly what `w1 [1536, 512]` records.
        """
        return -(-8 * self.hidden_dim // 3 // 256) * 256

    def merged_dim(self) -> int:
        """The one projection the whole of the conditioning enters through.

        80 (noisy mel) + 80 (prompt mel) + 512 (content) + 192 (timbre) = 864,
        which is what `cond_x_merge_linear [512, 864]` records. Guidance can be
        done by zeroing inputs precisely because there is no other inlet.
        """
        return 2 * self.n_mels + self.hidden_dim + self.style_dim


# ---- norms and the gated feed-forward ---------------------------------------


class WeightNormLinear(nn.Module):
    """`torch.nn.utils.weight_norm(nn.Linear)`, mirroring `dit.rs`'s own holder.

    **`weight_v` keeps PyTorch's `[out, in]`.** Burn's transposing adapter fires
    on its `Linear` and nothing else, so the Burn side hand-rolls this to be
    handed the checkpoint's orientation unchanged — and a torch mirror is in
    that orientation to begin with.
    """

    def __init__(self, d_in: int, d_out: int) -> None:
        super().__init__()
        self.weight_g = nn.Parameter(torch.ones(d_out, 1))
        self.weight_v = nn.Parameter(torch.randn(d_out, d_in) * 0.02)
        self.bias = nn.Parameter(torch.zeros(d_out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        v = self.weight_v
        weight = v * (self.weight_g / v.pow(2).sum(dim=1, keepdim=True).sqrt())
        return F.linear(x, weight, self.bias)


class RmsNorm(nn.Module):
    """One learned scale, no bias, no mean subtraction."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + NORM_EPS) * self.weight


class AdaLayerNorm(nn.Module):
    """`scale ⊙ rms(x) + shift`, both read off the conditioning vector.

    The conditioning is the flow-time embedding broadcast over every frame, so
    **this is how the transformer knows where along the trajectory it is** —
    nothing else inside a block depends on `t`.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.project_layer = nn.Linear(dim, 2 * dim)
        self.norm = RmsNorm(dim)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """`x`: `[batch, frames, dim]`, `c`: `[batch, 1, dim]` — broadcast over frames."""
        # Scale first, shift second. `FinalLayer` splits its own modulation the
        # other way round.
        scale, shift = self.project_layer(c).chunk(2, dim=2)
        return self.norm(x) * scale + shift


class FeedForward(nn.Module):
    """SwiGLU, `w2(silu(w1 x) ⊙ w3 x)`.

    A two-matrix MLP would consume the same tensors in the same order and simply
    compute something else, which is why `w1` and `w3` both being `[1536, 512]`
    is the shape to read this off.
    """

    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(hidden, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


# ---- rotary positions and attention -----------------------------------------


def rope_tables(frames: torch.Tensor | int, head_dim: int, device, dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """`(cos, sin)` of shape `[1, frames, 1, head_dim / 2]`.

    Recomputed per forward rather than cached, as on the Burn side: the tables
    are a function of the sequence length, which is a dynamic axis of the graph.

    **The angle is accumulated in `float64` and reduced modulo 2π before it is
    narrowed**, because computing `pos · inv` in `float32` loses precision that
    grows with position — 1.9e-06 at 64 frames but 3.0e-04 at the trained
    `block_size` of 8192, an order of magnitude above the 1e-5 this repo
    compares runtimes at. `dit.rs` accumulates in `f64` and pushes `as f32`, so
    a `float32` mirror would have surfaced as a plausible port bug the first
    time a long clip was measured ONNX against Burn.

    **The reduction is what keeps the trigonometry in `float32`, and it is not
    optional.** ONNX Runtime has no `float64` kernel for `Cos`/`Sin`, so taking
    the Burn side literally — cosine in `f64`, then narrow — exports and passes
    `onnx.checker` but fails at session creation with `NOT_IMPLEMENTED`. Since
    both are 2π-periodic, reducing first is exact in real arithmetic and leaves
    an argument in `[0, 2π)`, where `float32` spacing is 2.4e-07 — three orders
    below the threshold that mattered. So this is the accurate answer *and* the
    portable one, rather than a trade between them.
    """
    half = head_dim // 2
    pos = torch.arange(frames, device=device, dtype=torch.float64)
    inv = FREQ_BASE ** (-2.0 * torch.arange(half, device=device, dtype=torch.float64) / head_dim)
    theta = (pos[:, None] * inv[None, :]).remainder(TWO_PI).to(dtype)
    return theta.cos()[None, :, None, :], theta.sin()[None, :, None, :]


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate `x` (`[batch, frames, heads, head_dim]`) along its feature pairs.

    **The pairs are adjacent, not half-and-half.** `gpt-fast` reshapes the last
    dimension to `(head_dim/2, 2)`, so it rotates `(x[0], x[1])`, `(x[2], x[3])`
    and so on, where the split-halves convention of Llama-style code rotates
    `(x[i], x[i + head_dim/2])`. Both are called "RoPE", they are not
    interchangeable, and choosing wrongly costs nothing at load time and
    everything at inference.
    """
    b, t, h, d = x.shape
    pairs = x.reshape(b, t, h, d // 2, 2)
    x0, x1 = pairs[..., 0], pairs[..., 1]
    return torch.stack([x0 * cos - x1 * sin, x1 * cos + x0 * sin], dim=-1).reshape(b, t, h, d)


class Attention(nn.Module):
    """Unmasked self-attention, one fused `wqkv` and rotary positions.

    **There is no causal mask and adding one is the classic mistake here.** This
    is a denoiser, not a language model: it sees the whole utterance at once.
    Upstream does build a mask, but it is a *padding* mask over keys and is all
    ones for the single clip inference passes.
    """

    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        # `total_head_dim = (n_head + 2 · n_local_heads) · head_dim`, and this
        # preset has no grouped-query attention, so it comes to a plain 3 × dim.
        self.wqkv = nn.Linear(dim, 3 * dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)
        self.heads = heads
        self.head_dim = dim // heads

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        b, t, dim = x.shape
        q, k, v = (
            part.reshape(b, t, self.heads, self.head_dim)
            for part in self.wqkv(x).split(dim, dim=2)
        )
        q = apply_rope(q, cos, sin).transpose(1, 2)
        k = apply_rope(k, cos, sin).transpose(1, 2)
        v = v.transpose(1, 2)

        scores = q @ k.transpose(2, 3) / math.sqrt(self.head_dim)
        y = (torch.softmax(scores, dim=3) @ v).transpose(1, 2).reshape(b, t, dim)
        return self.wo(y)


# ---- the U-ViT stack --------------------------------------------------------


class TransformerBlock(nn.Module):
    """One pre-norm block, plus the U-ViT skip inlet."""

    def __init__(self, dim: int, heads: int, ffn_hidden: int) -> None:
        super().__init__()
        self.attention = Attention(dim, heads)
        self.feed_forward = FeedForward(dim, ffn_hidden)
        self.ffn_norm = AdaLayerNorm(dim)
        self.attention_norm = AdaLayerNorm(dim)
        # Present on **every** block, including the seven that never receive a
        # skip: upstream builds it from a config flag rather than from the
        # block's index, so the dead ones are in the checkpoint all the same.
        self.skip_in_linear = nn.Linear(2 * dim, dim)

    def forward(self, x, c, skip_in, cos, sin) -> torch.Tensor:
        if skip_in is not None:
            x = self.skip_in_linear(torch.cat([x, skip_in], dim=2))
        h = x + self.attention(self.attention_norm(x, c), cos, sin)
        return h + self.feed_forward(self.ffn_norm(h, c))


class Transformer(nn.Module):
    """The stack, with the U-ViT long skips wired across it.

    `i < depth/2` emits and `i > depth/2` receives, popping from the end — so the
    last emitter feeds the first receiver and the pairing is symmetric about the
    middle layer, which does neither. Upstream's depth of 13 is odd, which is
    what makes the two lists the same length; an even depth would leave one
    emitted tensor unclaimed, and upstream would not error either.
    """

    def __init__(self, cfg: DitConfig) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            TransformerBlock(cfg.hidden_dim, cfg.heads, cfg.ffn_hidden()) for _ in range(cfg.depth)
        )
        self.norm = AdaLayerNorm(cfg.hidden_dim)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        head_dim = self.layers[0].attention.head_dim
        cos, sin = rope_tables(x.shape[1], head_dim, x.device, x.dtype)

        half = len(self.layers) // 2
        skips: list[torch.Tensor] = []
        for i, layer in enumerate(self.layers):
            x = layer(x, c, skips.pop() if i > half else None, cos, sin)
            if i < half:
                skips.append(x)
        return self.norm(x, c)


# ---- flow time and the output head ------------------------------------------


class TimestepEmbedder(nn.Module):
    """Sinusoidal flow-time code through a two-layer MLP.

    Upstream is `nn.Sequential(Linear, SiLU, Linear)`, so its checkpoint keys are
    `mlp.0` and `mlp.2` — the gap is the activation, which carries no tensors.
    The exporter's remap closes it, matching `Dit::load_pytorch`.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.mlp = nn.ModuleList([nn.Linear(TIME_FREQ_DIM, dim), nn.Linear(dim, dim)])

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """`t`: `[batch]` in `[0, 1]` → `[batch, dim]`."""
        half = TIME_FREQ_DIM // 2
        # `scale · exp(-ln(10000) · i / half)` with `scale = 1000`: the flow time
        # is in [0, 1] where a diffusion step index would be in [0, 1000), so
        # upstream rescales the argument rather than re-deriving the ladder.
        # Accumulated in `float64` and reduced modulo 2π before narrowing, for
        # the reasons `rope_tables` gives: the argument reaches 1000 at `t = 1`,
        # where `float32` spacing is already 6e-05, and ONNX Runtime has no
        # `float64` `Cos`/`Sin`.
        i = torch.arange(half, device=t.device, dtype=torch.float64)
        args = t[:, None].double() * (1000.0 * FREQ_BASE ** (-i / half))[None, :]
        args = args.remainder(TWO_PI).to(t.dtype)
        # `cat([cos, sin])`, in that order — reversing it is silent.
        code = torch.cat([args.cos(), args.sin()], dim=1)
        return self.mlp[1](F.silu(self.mlp[0](code)))


class FinalLayer(nn.Module):
    """Modulate by the flow time, then project.

    Upstream is `nn.Sequential(SiLU, Linear)`, hence the checkpoint's
    `adaLN_modulation.1`; the exporter's remap drops the index and snake-cases
    the name, matching `Dit::load_pytorch`.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.linear = WeightNormLinear(dim, dim)
        self.ada_ln_modulation = nn.Linear(dim, 2 * dim)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """`x`: `[batch, frames, dim]`, `c`: `[batch, dim]`."""
        # **Shift is the first half here, scale the second** — the reverse of
        # `AdaLayerNorm`. Swapping them is a silent quality loss, not a crash.
        shift, scale = self.ada_ln_modulation(F.silu(c)).chunk(2, dim=1)
        normed = F.layer_norm(x, x.shape[-1:], eps=FINAL_NORM_EPS)
        return self.linear(normed * (1.0 + scale[:, None]) + shift[:, None])


# ---- the transformer --------------------------------------------------------


class Dit(nn.Module):
    """Seed-VC's diffusion transformer, the checkpoint's `net.cfm.module.estimator.*`."""

    def __init__(self, cfg: DitConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg = cfg or DitConfig()
        dim = cfg.hidden_dim

        self.cond_projection = nn.Linear(dim, dim)
        self.cond_x_merge_linear = nn.Linear(cfg.merged_dim(), dim)
        self.t_embedder = TimestepEmbedder(dim)
        # A **second** timestep embedder, feeding the WaveNet tail while
        # `t_embedder` feeds the transformer and the output head. Same
        # architecture, separate weights.
        self.t_embedder2 = TimestepEmbedder(dim)
        self.transformer = Transformer(cfg)
        # The **long** skip, concatenating the input mel onto the transformer's
        # output — hence 592 = 512 + 80, where the per-layer `skip_in_linear` is
        # the U-ViT one.
        self.skip_linear = nn.Linear(dim + cfg.n_mels, dim)
        self.conv1 = nn.Linear(dim, dim)
        self.wavenet = WaveNet(dim, cfg.wavenet_layers, cfg.wavenet_kernel, cfg.wavenet_dilation)
        self.res_projection = nn.Linear(dim, dim)
        self.final_layer = FinalLayer(dim)
        self.conv2 = nn.Conv1d(dim, cfg.n_mels, 1)

    def forward(
        self,
        x: torch.Tensor,
        prompt_x: torch.Tensor,
        t: torch.Tensor,
        style: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        """Predict the flow velocity at time `t`.

        - `x`: `[batch, n_mels, frames]` — the current, partly-noised mel.
        - `prompt_x`: `[batch, n_mels, frames]` — the reference mel in the
          leading frames and zero after. Shaped like `x`, **not** like the
          reference: the sampler writes the reference into a zero tensor of `x`'s
          length.
        - `t`: `[batch]` — flow time in `[0, 1]`, one per batch element.
        - `style`: `[batch, style_dim]` — the timbre vector, broadcast over frames.
        - `cond`: `[batch, frames, hidden_dim]` — the length regulator's output.

        Returns `[batch, n_mels, frames]`.

        Classifier-free guidance is the caller's business and is done by
        **input** rather than by a flag: stack the batch twice and zero
        `prompt_x`, `style` and `cond` in the second half. Nothing here reduces
        over dimension 0, so the two rows cannot blend.
        """
        t1 = self.t_embedder(t)
        x = x.transpose(1, 2)
        x_in = self.cond_x_merge_linear(
            torch.cat(
                [
                    x,
                    prompt_x.transpose(1, 2),
                    self.cond_projection(cond),
                    style[:, None, :].expand(-1, x.shape[1], -1),
                ],
                dim=2,
            )
        )

        x_res = self.transformer(x_in, t1[:, None, :])
        x_res = self.skip_linear(torch.cat([x_res, x], dim=2))

        h = self.wavenet(self.conv1(x_res).transpose(1, 2), self.t_embedder2(t)[:, :, None])
        # The long residual: the WaveNet refines the transformer's output rather
        # than replacing it.
        h = h.transpose(1, 2) + self.res_projection(x_res)

        return self.conv2(self.final_layer(h, t1).transpose(1, 2))


# ---- the exported graph -----------------------------------------------------


class Graph(nn.Module):
    """`x [B,80,T]`, `prompt_x [B,80,T]`, `t [B]`, `style [B,192]`, `cond [B,T,512]`
    → `v [B,80,T]`.

    One velocity evaluation. **The Euler loop stays on the host**, exactly as
    `s1`'s sampling does for `tts`: a graph that is a pure function of its inputs
    can be compared run to run and runtime to runtime, and the loop carries no
    weights to export.

    **The batch axis is dynamic and must stay so** — classifier-free guidance
    runs the conditioned and unconditional inputs as batch 2 in a single call,
    which is how `flow.rs` stacks them and what pays for one forward pass instead
    of two.
    """

    INPUTS = ["x", "prompt_x", "t", "style", "cond"]
    OUTPUTS = ["v"]

    def __init__(self, dit: Dit) -> None:
        super().__init__()
        self.dit = dit

    def forward(self, x, prompt_x, t, style, cond) -> torch.Tensor:
        return self.dit(x, prompt_x, t, style, cond)

    def dummy(self) -> tuple[torch.Tensor, ...]:
        """Trace inputs at the guidance pair's own batch of 2.

        Tracing at batch 1 would leave a specialised-batch graph one step away
        from a `Dim.DYNAMIC` assertion failure rather than an obvious one, and
        the frame count has to clear the WaveNet's reflect padding either way.
        """
        cfg = self.dit.cfg
        batch, frames = 2, 64
        return (
            torch.randn(batch, cfg.n_mels, frames),
            torch.randn(batch, cfg.n_mels, frames),
            torch.rand(batch),
            torch.randn(batch, cfg.style_dim),
            torch.randn(batch, frames, cfg.hidden_dim),
        )

    def dynamic_shapes(self) -> tuple:
        dyn = torch.export.Dim.DYNAMIC
        return (
            {0: dyn, 2: dyn},
            {0: dyn, 2: dyn},
            {0: dyn},
            {0: dyn},
            {0: dyn, 1: dyn},
        )
