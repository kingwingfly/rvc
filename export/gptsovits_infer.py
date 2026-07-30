"""Standalone torch implementation of the GPT-SoVITS inference path.

Mirrors the Burn modules (`burn-gptsovits`, `burn-vits`) field for field, so a
Burn `.safetensors` — what `tts train` writes — loads with a direct key mapping,
exactly as `rvc_infer.py` mirrors `burn-rvc`. This is a clean-room
reimplementation: it does not import the GPT-SoVITS repository.

Only what inference needs is here. `enc_q` (the posterior encoder) and the
discriminators are training-only and omitted. Weight-norm is folded at load
time, so the convolutions here are plain.

The four graphs the exporter emits are the classes at the bottom:
[`ReferenceGraph`], [`S1Prompt`], [`S1Step`] and [`S2`].
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import nn

LRELU_SLOPE = 0.1


def lrelu(x: torch.Tensor) -> torch.Tensor:
    return F.leaky_relu(x, LRELU_SLOPE)


# ---- cnhubert ---------------------------------------------------------------


@dataclass
class HubertConfig:
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    conv_dim: tuple[int, ...] = (512,) * 7
    conv_kernel: tuple[int, ...] = (10, 3, 3, 3, 3, 2, 2)
    conv_stride: tuple[int, ...] = (5, 2, 2, 2, 2, 2, 2)
    num_conv_pos_embeddings: int = 128
    num_conv_pos_embedding_groups: int = 16


class PosConv(nn.Module):
    """The encoder's convolutional position embedding.

    Weight-normalised over the **last** axis rather than over output channels,
    which is why it cannot share the folding rule the VITS decoders use: its
    `weight_g` is `[1, 1, kernel]`, not `[out, 1, 1]`.
    """

    def __init__(self, cfg: HubertConfig) -> None:
        super().__init__()
        c, k = cfg.hidden_size, cfg.num_conv_pos_embeddings
        self.groups = cfg.num_conv_pos_embedding_groups
        self.padding = k // 2
        self.weight = nn.Parameter(torch.zeros(c, c // self.groups, k))
        self.bias = nn.Parameter(torch.zeros(c))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.conv1d(x, self.weight, self.bias, padding=self.padding, groups=self.groups)
        # An even kernel with `kernel // 2` padding leaves one frame too many on
        # the right; the reference drops it rather than padding asymmetrically.
        return F.gelu(y[:, :, :-1])


class ConvLayer(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int, stride: int, norm: bool) -> None:
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel, stride, bias=False)
        # `feat_extract_norm: "group"` means layer 0 only, one group per channel.
        self.layer_norm = nn.GroupNorm(out_ch, out_ch) if norm else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.layer_norm is not None:
            x = self.layer_norm(x)
        return F.gelu(x)


class HubertAttention(nn.Module):
    def __init__(self, cfg: HubertConfig) -> None:
        super().__init__()
        d = cfg.hidden_size
        self.q_proj = nn.Linear(d, d)
        self.k_proj = nn.Linear(d, d)
        self.v_proj = nn.Linear(d, d)
        self.out_proj = nn.Linear(d, d)
        self.n_head = cfg.num_attention_heads

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        dh = d // self.n_head
        heads = lambda y: y.view(b, t, self.n_head, dh).transpose(1, 2)  # noqa: E731
        q = heads(self.q_proj(x)) * dh**-0.5
        k = heads(self.k_proj(x))
        v = heads(self.v_proj(x))
        out = torch.softmax(q @ k.transpose(2, 3), dim=3) @ v
        return self.out_proj(out.transpose(1, 2).reshape(b, t, d))


class HubertFeedForward(nn.Module):
    def __init__(self, cfg: HubertConfig) -> None:
        super().__init__()
        self.intermediate_dense = nn.Linear(cfg.hidden_size, cfg.intermediate_size)
        self.output_dense = nn.Linear(cfg.intermediate_size, cfg.hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output_dense(F.gelu(self.intermediate_dense(x)))


class HubertLayer(nn.Module):
    """One **post**-norm block (`do_stable_layer_norm: false`)."""

    def __init__(self, cfg: HubertConfig) -> None:
        super().__init__()
        self.attention = HubertAttention(cfg)
        self.layer_norm = nn.LayerNorm(cfg.hidden_size)
        self.feed_forward = HubertFeedForward(cfg)
        self.final_layer_norm = nn.LayerNorm(cfg.hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer_norm(x + self.attention(x))
        return self.final_layer_norm(x + self.feed_forward(x))


class HubertEncoder(nn.Module):
    def __init__(self, cfg: HubertConfig) -> None:
        super().__init__()
        self.pos_conv_embed = PosConv(cfg)
        self.layer_norm = nn.LayerNorm(cfg.hidden_size)
        self.layers = nn.ModuleList(HubertLayer(cfg) for _ in range(cfg.num_hidden_layers))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pos = self.pos_conv_embed(x.transpose(1, 2)).transpose(1, 2)
        x = self.layer_norm(x + pos)
        for layer in self.layers:
            x = layer(x)
        return x


class FeatureProjection(nn.Module):
    def __init__(self, cfg: HubertConfig) -> None:
        super().__init__()
        self.layer_norm = nn.LayerNorm(cfg.conv_dim[-1])
        self.projection = nn.Linear(cfg.conv_dim[-1], cfg.hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(self.layer_norm(x))


class FeatureExtractor(nn.Module):
    def __init__(self, cfg: HubertConfig) -> None:
        super().__init__()
        self.conv_layers = nn.ModuleList(
            ConvLayer(
                1 if i == 0 else cfg.conv_dim[i - 1],
                cfg.conv_dim[i],
                cfg.conv_kernel[i],
                cfg.conv_stride[i],
                norm=(i == 0),
            )
            for i in range(len(cfg.conv_dim))
        )

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        x = wav.unsqueeze(1)
        for layer in self.conv_layers:
            x = layer(x)
        return x


class Hubert(nn.Module):
    """`[batch, samples]` at 16 kHz → `[batch, frames, hidden]` at 50 Hz."""

    def __init__(self, cfg: HubertConfig) -> None:
        super().__init__()
        self.feature_extractor = FeatureExtractor(cfg)
        self.feature_projection = FeatureProjection(cfg)
        self.encoder = HubertEncoder(cfg)

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        frames = self.feature_extractor(wav)
        return self.encoder(self.feature_projection(frames.transpose(1, 2)))


# ---- the semantic-token boundary --------------------------------------------


@dataclass
class QuantizerConfig:
    dim: int = 768
    codebook_size: int = 1024
    stride: int = 2


class Codebook(nn.Module):
    def __init__(self, cfg: QuantizerConfig) -> None:
        super().__init__()
        self.embed = nn.Parameter(torch.zeros(cfg.codebook_size, cfg.dim))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Nearest entry to each frame of `[batch, frames, dim]`.

        `|x - e|² = |x|² - 2·x·eᵀ + |e|²`, with `|x|²` dropped: it is the same
        for every candidate of a given frame, so it cannot change the winner.
        """
        sq = self.embed.pow(2).sum(dim=1)
        return (x @ self.embed.t() * 2.0 - sq).argmax(dim=2)

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        return F.embedding(codes, self.embed)


class Quantizer(nn.Module):
    """`extract_latent`: a stride-2 convolution to 25 Hz, then a lookup."""

    def __init__(self, cfg: QuantizerConfig) -> None:
        super().__init__()
        self.ssl_proj = nn.Conv1d(cfg.dim, cfg.dim, cfg.stride, cfg.stride)
        self.vq = nn.Module()
        self.vq.layers = nn.ModuleList([Codebook(cfg)])

    def encode(self, ssl: torch.Tensor) -> torch.Tensor:
        """`[batch, dim, frames]` at 50 Hz → `[batch, frames/2]` token ids."""
        return self.vq.layers[0].encode(self.ssl_proj(ssl).transpose(1, 2))

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        return self.vq.layers[0].decode(codes).transpose(1, 2)


# ---- linear spectrogram -----------------------------------------------------


class Spectral(nn.Module):
    """Framing, windowing and the DFT fused into two convolutions.

    Not a `torch.stft` call: this mirrors `burn-vits`' `Spectral`, which is built
    that way because Burn has no STFT, and a DFT written as a convolution is also
    the form that exports cleanly to ONNX.
    """

    def __init__(self, n_fft: int = 2048, hop: int = 640) -> None:
        super().__init__()
        window = torch.zeros(n_fft)
        window[:n_fft] = 0.5 - 0.5 * torch.cos(2 * math.pi * torch.arange(n_fft) / n_fft)
        n_bins = n_fft // 2 + 1
        b = torch.arange(n_bins).unsqueeze(1)
        k = torch.arange(n_fft).unsqueeze(0)
        angle = 2 * math.pi * b * k / n_fft
        self.register_buffer("cos_kernel", (window * torch.cos(angle)).unsqueeze(1))
        self.register_buffer("sin_kernel", (-window * torch.sin(angle)).unsqueeze(1))
        self.hop = hop
        # center=False with (n_fft-hop)/2 reflect padding, so the frame count is
        # exactly L/hop and one frame is one decoder output hop.
        self.pad = (n_fft - hop) // 2

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """`[batch, samples]` → `[batch, n_bins, frames]` magnitude."""
        x = F.pad(wav.unsqueeze(1), (self.pad, self.pad), mode="reflect")
        real = F.conv1d(x, self.cos_kernel, stride=self.hop)
        imag = F.conv1d(x, self.sin_kernel, stride=self.hop)
        return (real.pow(2) + imag.pow(2) + 1e-9).sqrt()


# ---- ref_enc: the speaker vector --------------------------------------------


@dataclass
class ReferenceConfig:
    in_dim: int = 704
    hidden: int = 128
    out_dim: int = 512
    kernel_size: int = 5
    n_head: int = 2


def mish(x: torch.Tensor) -> torch.Tensor:
    return x * torch.tanh(F.softplus(x))


class LinearNorm(nn.Module):
    """A plain `Linear` under the wrapper whose only lasting effect is the extra
    `fc` level in the parameter names."""

    def __init__(self, inp: int, out: int) -> None:
        super().__init__()
        self.fc = nn.Linear(inp, out)


class ConvNorm(nn.Module):
    def __init__(self, inp: int, out: int, kernel: int) -> None:
        super().__init__()
        self.conv = nn.Conv1d(inp, out, kernel, padding=kernel // 2)


class Conv1dGlu(nn.Module):
    def __init__(self, channels: int, kernel: int) -> None:
        super().__init__()
        self.conv1 = ConvNorm(channels, channels * 2, kernel)
        self.channels = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv1.conv(x)
        signal, gate = y[:, : self.channels], y[:, self.channels :]
        return x + signal * torch.sigmoid(gate)


class StyleAttention(nn.Module):
    """Self-attention over the reference frames.

    Scaled by **√d_model, not √d_head** — upstream passes
    `temperature=d_model ** 0.5`. With two heads over 128 channels that is √128
    where the conventional choice is √64, and the conventional one loads just as
    well.
    """

    def __init__(self, d_model: int, n_head: int) -> None:
        super().__init__()
        self.w_qs = nn.Linear(d_model, d_model)
        self.w_ks = nn.Linear(d_model, d_model)
        self.w_vs = nn.Linear(d_model, d_model)
        self.fc = nn.Linear(d_model, d_model)
        self.n_head = n_head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        dh = d // self.n_head
        heads = lambda y: y.view(b, t, self.n_head, dh).transpose(1, 2)  # noqa: E731
        q, k, v = heads(self.w_qs(x)), heads(self.w_ks(x)), heads(self.w_vs(x))
        scores = (q @ k.transpose(2, 3)) / math.sqrt(d)
        out = (torch.softmax(scores, dim=3) @ v).transpose(1, 2).reshape(b, t, d)
        return self.fc(out) + x


class ReferenceEncoder(nn.Module):
    def __init__(self, cfg: ReferenceConfig) -> None:
        super().__init__()
        self.spectral = nn.ModuleList(
            [LinearNorm(cfg.in_dim, cfg.hidden), LinearNorm(cfg.hidden, cfg.hidden)]
        )
        self.temporal = nn.ModuleList(
            [Conv1dGlu(cfg.hidden, cfg.kernel_size), Conv1dGlu(cfg.hidden, cfg.kernel_size)]
        )
        self.slf_attn = StyleAttention(cfg.hidden, cfg.n_head)
        self.fc = LinearNorm(cfg.hidden, cfg.out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """`[batch, in_dim, frames]` → `[batch, out_dim, 1]`."""
        h = x.transpose(1, 2)
        for layer in self.spectral:
            h = mish(layer.fc(h))
        h = h.transpose(1, 2)
        for layer in self.temporal:
            h = layer(h)
        h = self.slf_attn(h.transpose(1, 2))
        h = self.fc.fc(h)
        # The speaker is a property of the whole reference, not of any frame.
        return h.mean(dim=1, keepdim=True).transpose(1, 2)


# ---- the VITS attention stack (shared by all three of enc_p's encoders) ------


class VitsLayerNorm(nn.Module):
    """Channel-wise LayerNorm over `[batch, channels, time]`."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        centered = x - mean
        var = centered.pow(2).mean(dim=1, keepdim=True)
        normed = centered / (var + 1e-5).sqrt()
        return normed * self.gamma.view(1, -1, 1) + self.beta.view(1, -1, 1)


class RelativeAttention(nn.Module):
    """Multi-head self-attention with relative position embeddings."""

    def __init__(self, channels: int, n_heads: int, window_size: int) -> None:
        super().__init__()
        self.conv_q = nn.Conv1d(channels, channels, 1)
        self.conv_k = nn.Conv1d(channels, channels, 1)
        self.conv_v = nn.Conv1d(channels, channels, 1)
        self.conv_o = nn.Conv1d(channels, channels, 1)
        self.k_channels = channels // n_heads
        self.emb_rel_k = nn.Parameter(torch.zeros(1, window_size * 2 + 1, self.k_channels))
        self.emb_rel_v = nn.Parameter(torch.zeros(1, window_size * 2 + 1, self.k_channels))
        self.n_heads = n_heads
        self.window_size = window_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, d, t = x.shape
        h, dk = self.n_heads, self.k_channels
        shape = lambda y: y.view(b, h, dk, t).transpose(2, 3)  # noqa: E731
        q = shape(self.conv_q(x)) / math.sqrt(dk)
        k = shape(self.conv_k(x))
        v = shape(self.conv_v(x))

        scores = q @ k.transpose(2, 3)
        key_rel = self._relative_embeddings(self.emb_rel_k, t)
        scores = scores + self._rel_to_abs(q @ key_rel.transpose(1, 2).unsqueeze(0))

        p_attn = torch.softmax(scores, dim=3)
        out = p_attn @ v
        value_rel = self._relative_embeddings(self.emb_rel_v, t)
        out = out + self._abs_to_rel(p_attn) @ value_rel.unsqueeze(0)
        return self.conv_o(out.transpose(2, 3).reshape(b, d, t))

    def _relative_embeddings(self, emb: torch.Tensor, length: int) -> torch.Tensor:
        w = self.window_size
        pad = max(length - (w + 1), 0)
        start = max((w + 1) - length, 0)
        if pad > 0:
            emb = F.pad(emb, [0, 0, pad, pad, 0, 0])
        return emb[:, start : start + 2 * length - 1]

    @staticmethod
    def _rel_to_abs(x: torch.Tensor) -> torch.Tensor:
        b, h, l, _ = x.shape
        x = F.pad(x, [0, 1])
        x = x.view(b, h, l * 2 * l)
        x = F.pad(x, [0, l - 1])
        return x.view(b, h, l + 1, 2 * l - 1)[:, :, :l, l - 1 :]

    @staticmethod
    def _abs_to_rel(x: torch.Tensor) -> torch.Tensor:
        b, h, l, _ = x.shape
        x = F.pad(x, [0, l - 1])
        x = x.view(b, h, l * l + l * (l - 1))
        x = F.pad(x, [l, 0])
        return x.view(b, h, l, 2 * l)[:, :, :, 1:]


class Ffn(nn.Module):
    def __init__(self, channels: int, filter_channels: int, kernel: int) -> None:
        super().__init__()
        pad = ((kernel - 1) // 2, kernel // 2)
        self.conv_1 = nn.Conv1d(channels, filter_channels, kernel)
        self.conv_2 = nn.Conv1d(filter_channels, channels, kernel)
        self.pad = pad

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.conv_1(F.pad(x, self.pad)))
        return self.conv_2(F.pad(x, self.pad))


class EncoderLayer(nn.Module):
    def __init__(self, channels: int, filter_channels: int, n_heads: int, kernel: int) -> None:
        super().__init__()
        self.attn = RelativeAttention(channels, n_heads, window_size=4)
        self.norm_1 = VitsLayerNorm(channels)
        self.ffn = Ffn(channels, filter_channels, kernel)
        self.norm_2 = VitsLayerNorm(channels)


class Encoder(nn.Module):
    def __init__(
        self, channels: int, filter_channels: int, n_heads: int, n_layers: int, kernel: int
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            EncoderLayer(channels, filter_channels, n_heads, kernel) for _ in range(n_layers)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer.norm_1(x + layer.attn(x))
            x = layer.norm_2(x + layer.ffn(x))
        return x


# ---- enc_p ------------------------------------------------------------------


@dataclass
class TextEncoderConfig:
    n_symbols: int = 732
    ssl_dim: int = 768
    hidden_channels: int = 192
    filter_channels: int = 768
    n_heads: int = 2
    n_layers: int = 6
    kernel_size: int = 3
    out_channels: int = 192
    mrte_hidden: int = 512
    mrte_heads: int = 4


class CrossAttention(nn.Module):
    """The MRTE's cross-attention.

    No relative-position embeddings, and not for want of them in the checkpoint:
    query and key live in different sequences, so the distance between two of
    their indices means nothing.
    """

    def __init__(self, channels: int, n_heads: int) -> None:
        super().__init__()
        self.conv_q = nn.Conv1d(channels, channels, 1)
        self.conv_k = nn.Conv1d(channels, channels, 1)
        self.conv_v = nn.Conv1d(channels, channels, 1)
        self.conv_o = nn.Conv1d(channels, channels, 1)
        self.n_heads = n_heads

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        b, c, tq = x.shape
        dh = c // self.n_heads
        heads = lambda y: y.view(b, self.n_heads, dh, y.shape[2]).transpose(2, 3)  # noqa: E731
        q = heads(self.conv_q(x)) / math.sqrt(dh)
        k = heads(self.conv_k(context))
        v = heads(self.conv_v(context))
        out = torch.softmax(q @ k.transpose(2, 3), dim=3) @ v
        return self.conv_o(out.transpose(2, 3).reshape(b, c, tq))


class Mrte(nn.Module):
    def __init__(self, cfg: TextEncoderConfig) -> None:
        super().__init__()
        h = cfg.mrte_hidden
        self.cross_attention = CrossAttention(h, cfg.mrte_heads)
        self.c_pre = nn.Conv1d(cfg.hidden_channels, h, 1)
        self.text_pre = nn.Conv1d(cfg.hidden_channels, h, 1)
        self.c_post = nn.Conv1d(h, cfg.hidden_channels, 1)

    def forward(self, ssl: torch.Tensor, text: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        ssl = self.c_pre(ssl)
        text = self.text_pre(text)
        return self.c_post(self.cross_attention(ssl, text) + ssl + g)


class TextEncoder(nn.Module):
    def __init__(self, cfg: TextEncoderConfig) -> None:
        super().__init__()
        stack = lambda n: Encoder(  # noqa: E731
            cfg.hidden_channels, cfg.filter_channels, cfg.n_heads, n, cfg.kernel_size
        )
        self.ssl_proj = nn.Conv1d(cfg.ssl_dim, cfg.hidden_channels, 1)
        self.encoder_ssl = stack(cfg.n_layers // 2)
        self.text_embedding = nn.Embedding(cfg.n_symbols, cfg.hidden_channels)
        self.encoder_text = stack(cfg.n_layers)
        self.mrte = Mrte(cfg)
        self.encoder2 = stack(cfg.n_layers // 2)
        self.proj = nn.Conv1d(cfg.hidden_channels, cfg.out_channels * 2, 1)
        self.out_channels = cfg.out_channels

    def forward(
        self, ssl: torch.Tensor, text: torch.Tensor, g: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        y = self.encoder_ssl(self.ssl_proj(ssl))
        t = self.encoder_text(self.text_embedding(text).transpose(1, 2))
        y = self.encoder2(self.mrte(y, t, g))
        stats = self.proj(y)
        return torch.split(stats, self.out_channels, dim=1)


# ---- flow -------------------------------------------------------------------


class Wn(nn.Module):
    def __init__(self, hidden: int, kernel: int, dilation_rate: int, n_layers: int, gin: int) -> None:
        super().__init__()
        self.hidden = hidden
        self.n_layers = n_layers
        self.cond_layer = nn.Conv1d(gin, 2 * hidden * n_layers, 1)
        self.in_layers = nn.ModuleList()
        self.res_skip_layers = nn.ModuleList()
        for i in range(n_layers):
            d = dilation_rate**i
            pad = (kernel * d - d) // 2
            self.in_layers.append(nn.Conv1d(hidden, 2 * hidden, kernel, dilation=d, padding=pad))
            out = 2 * hidden if i < n_layers - 1 else hidden
            self.res_skip_layers.append(nn.Conv1d(hidden, out, 1))

    def forward(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        out = torch.zeros_like(x)
        g = self.cond_layer(g)
        h = self.hidden
        for i in range(self.n_layers):
            acts = self.in_layers[i](x) + g[:, i * 2 * h : (i + 1) * 2 * h]
            acts = torch.tanh(acts[:, :h]) * torch.sigmoid(acts[:, h:])
            rs = self.res_skip_layers[i](acts)
            if i < self.n_layers - 1:
                x = x + rs[:, :h]
                out = out + rs[:, h:]
            else:
                out = out + rs
        return out


class ResidualCouplingLayer(nn.Module):
    def __init__(self, channels: int, hidden: int, gin: int, n_layers: int) -> None:
        super().__init__()
        self.half = channels // 2
        self.pre = nn.Conv1d(self.half, hidden, 1)
        self.enc = Wn(hidden, 5, 1, n_layers, gin)
        self.post = nn.Conv1d(hidden, self.half, 1)

    def reverse(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        x0, x1 = x[:, : self.half], x[:, self.half :]
        return torch.cat([x0, x1 - self.post(self.enc(self.pre(x0), g))], 1)


class ResidualCouplingBlock(nn.Module):
    def __init__(self, channels: int, hidden: int, gin: int, n_flows: int, n_layers: int) -> None:
        super().__init__()
        self.flows = nn.ModuleList(
            ResidualCouplingLayer(channels, hidden, gin, n_layers) for _ in range(n_flows)
        )

    def reverse(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """Latent ← prior. Flips carry no parameters and sit between couplings."""
        for flow in reversed(self.flows):
            x = torch.flip(x, [1])
            x = flow.reverse(x, g)
        return x


# ---- the HiFiGAN decoder ----------------------------------------------------


@dataclass
class DecoderConfig:
    in_channels: int = 192
    upsample_initial_channel: int = 512
    upsample_rates: tuple[int, ...] = (10, 8, 2, 2, 2)
    upsample_kernel_sizes: tuple[int, ...] = (16, 16, 8, 2, 2)
    resblock_kernel_sizes: tuple[int, ...] = (3, 7, 11)
    resblock_dilation_sizes: tuple[tuple[int, ...], ...] = ((1, 3, 5), (1, 3, 5), (1, 3, 5))
    gin_channels: int = 512

    def samples_per_frame(self) -> int:
        n = 1
        for r in self.upsample_rates:
            n *= r
        return n


class ResBlock1(nn.Module):
    def __init__(self, channels: int, kernel: int, dilations: tuple[int, ...]) -> None:
        super().__init__()
        pad = lambda d: d * (kernel - 1) // 2  # noqa: E731
        self.convs1 = nn.ModuleList(
            nn.Conv1d(channels, channels, kernel, dilation=d, padding=pad(d)) for d in dilations
        )
        self.convs2 = nn.ModuleList(
            nn.Conv1d(channels, channels, kernel, dilation=1, padding=pad(1)) for _ in dilations
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for c1, c2 in zip(self.convs1, self.convs2):
            x = x + c2(lrelu(c1(lrelu(x))))
        return x


class Decoder(nn.Module):
    def __init__(self, cfg: DecoderConfig) -> None:
        super().__init__()
        uic = cfg.upsample_initial_channel
        self.num_kernels = len(cfg.resblock_kernel_sizes)
        self.conv_pre = nn.Conv1d(cfg.in_channels, uic, 7, padding=3)
        self.ups = nn.ModuleList()
        self.resblocks = nn.ModuleList()
        for i, (rate, kernel) in enumerate(zip(cfg.upsample_rates, cfg.upsample_kernel_sizes)):
            self.ups.append(
                nn.ConvTranspose1d(uic >> i, uic >> (i + 1), kernel, rate, (kernel - rate) // 2)
            )
            for k, d in zip(cfg.resblock_kernel_sizes, cfg.resblock_dilation_sizes):
                self.resblocks.append(ResBlock1(uic >> (i + 1), k, d))
        last = uic >> len(cfg.upsample_rates)
        self.conv_post = nn.Conv1d(last, 1, 7, padding=3, bias=False)
        self.cond = nn.Conv1d(cfg.gin_channels, uic, 1)

    def forward(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        # The speaker vector is added once, before any upsampling: it colours the
        # whole utterance rather than varying along it.
        x = self.conv_pre(x) + self.cond(g)
        for i, up in enumerate(self.ups):
            x = up(lrelu(x))
            blocks = self.resblocks[i * self.num_kernels : (i + 1) * self.num_kernels]
            x = sum(block(x) for block in blocks) / self.num_kernels
        return torch.tanh(self.conv_post(lrelu(x)))


# ---- s2 ---------------------------------------------------------------------


@dataclass
class SovitsConfig:
    inter_channels: int = 192
    hidden_channels: int = 192
    gin_channels: int = 512
    n_flows: int = 4
    flow_layers: int = 4
    decoder: DecoderConfig = field(default_factory=DecoderConfig)
    text_encoder: TextEncoderConfig = field(default_factory=TextEncoderConfig)
    reference: ReferenceConfig = field(default_factory=ReferenceConfig)
    quantizer: QuantizerConfig = field(default_factory=QuantizerConfig)


class Sovits(nn.Module):
    """The inference half of `s2`. `enc_q` is training-only and absent."""

    def __init__(self, cfg: SovitsConfig) -> None:
        super().__init__()
        self.flow = ResidualCouplingBlock(
            cfg.inter_channels, cfg.hidden_channels, cfg.gin_channels, cfg.n_flows, cfg.flow_layers
        )
        self.dec = Decoder(cfg.decoder)
        self.enc_p = TextEncoder(cfg.text_encoder)
        self.ref_enc = ReferenceEncoder(cfg.reference)
        self.quantizer = Quantizer(cfg.quantizer)
        self.reference_width = cfg.reference.in_dim

    def speaker(self, spec: torch.Tensor) -> torch.Tensor:
        # Upstream slices `refer[:, :704]`; the width is a constant, not a mel
        # count, which is why it is not derived from the spectrogram.
        return self.ref_enc(spec[:, : self.reference_width])

    def decode(
        self, codes: torch.Tensor, text: torch.Tensor, g: torch.Tensor, noise: torch.Tensor
    ) -> torch.Tensor:
        # The quantiser works at 25 Hz and `enc_p` at 50, so every token is
        # repeated once. Interpolating *between* codes would be wrong: they name
        # codebook entries, and a point between two of them is not a third entry.
        quantized = self.quantizer.decode(codes).repeat_interleave(2, dim=2)
        m, logs = self.enc_p(quantized, text, g)
        z_p = m + torch.exp(logs) * noise
        return self.dec(self.flow.reverse(z_p, g), g)


# ---- s1 ---------------------------------------------------------------------


@dataclass
class T2sConfig:
    model_dim: int = 512
    n_head: int = 16
    n_layer: int = 24
    ffn_dim: int = 2048
    phoneme_vocab_size: int = 732
    vocab_size: int = 1025
    bert_dim: int = 1024


class SinePosition(nn.Module):
    """Sinusoidal positions with one learnable scalar.

    The table is fixed; `alpha` decides how loudly it speaks. `sin` and `cos`
    are interleaved per pair — not the two halves concatenated, which is the
    other common convention and would load identically.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1))
        k = torch.arange(dim // 2, dtype=torch.float32)
        self.register_buffer("inv_freq", torch.exp(-math.log(10000.0) * 2 * k / dim))
        self.dim = dim

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """`x`: `[batch, seq, dim]`; `positions`: `[seq]`, the index of each."""
        angle = positions.to(self.inv_freq.dtype).unsqueeze(1) * self.inv_freq.unsqueeze(0)
        pe = torch.stack([torch.sin(angle), torch.cos(angle)], dim=-1).reshape(1, -1, self.dim)
        return x + pe * self.alpha


class FusedAttention(nn.Module):
    """Self-attention with the three projections fused, as
    `nn.MultiheadAttention` stores them: one `[3 * d_model, d_model]` weight
    holding query, key and value in that order."""

    def __init__(self, cfg: T2sConfig) -> None:
        super().__init__()
        d = cfg.model_dim
        self.in_proj_weight = nn.Parameter(torch.zeros(3 * d, d))
        self.in_proj_bias = nn.Parameter(torch.zeros(3 * d))
        self.out_proj = nn.Linear(d, d)
        self.n_head = cfg.n_head

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None,
        past: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, seq, d = x.shape
        dh = d // self.n_head
        q, k, v = F.linear(x, self.in_proj_weight, self.in_proj_bias).split(d, dim=2)
        if past is not None:
            k = torch.cat([past[0], k], dim=1)
            v = torch.cat([past[1], v], dim=1)

        heads = lambda y: y.view(b, y.shape[1], self.n_head, dh).transpose(1, 2)  # noqa: E731
        scores = (heads(q) / math.sqrt(dh)) @ heads(k).transpose(2, 3)
        if mask is not None:
            scores = scores.masked_fill(mask, float("-inf"))
        out = (torch.softmax(scores, dim=3) @ heads(v)).transpose(1, 2).reshape(b, seq, d)
        return self.out_proj(out), k, v


class T2sLayer(nn.Module):
    """One **post**-norm block: normalise *after* the residual add. The pre-norm
    arrangement is one line away and loads identically."""

    def __init__(self, cfg: T2sConfig) -> None:
        super().__init__()
        self.self_attn = FusedAttention(cfg)
        self.linear1 = nn.Linear(cfg.model_dim, cfg.ffn_dim)
        self.linear2 = nn.Linear(cfg.ffn_dim, cfg.model_dim)
        self.norm1 = nn.LayerNorm(cfg.model_dim)
        self.norm2 = nn.LayerNorm(cfg.model_dim)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None,
        past: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h, k, v = self.self_attn(x, mask, past)
        x = self.norm1(x + h)
        return self.norm2(x + self.linear2(torch.relu(self.linear1(x)))), k, v


class TokenEmbedding(nn.Module):
    """An embedding under the extra `word_embeddings` level the checkpoint has."""

    def __init__(self, vocab: int, dim: int) -> None:
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab, dim)


class T2sStack(nn.Module):
    def __init__(self, cfg: T2sConfig) -> None:
        super().__init__()
        self.layers = nn.ModuleList(T2sLayer(cfg) for _ in range(cfg.n_layer))


class T2s(nn.Module):
    def __init__(self, cfg: T2sConfig) -> None:
        super().__init__()
        self.bert_proj = nn.Linear(cfg.bert_dim, cfg.model_dim)
        self.ar_text_embedding = TokenEmbedding(cfg.phoneme_vocab_size, cfg.model_dim)
        self.ar_text_position = SinePosition(cfg.model_dim)
        self.ar_audio_embedding = TokenEmbedding(cfg.vocab_size, cfg.model_dim)
        self.ar_audio_position = SinePosition(cfg.model_dim)
        self.h = T2sStack(cfg)
        self.ar_predict_layer = nn.Linear(cfg.model_dim, cfg.vocab_size, bias=False)

    def embed_text(self, phones: torch.Tensor, bert: torch.Tensor) -> torch.Tensor:
        # Added, not concatenated: prosody colours each phoneme rather than
        # extending the sequence.
        x = self.ar_text_embedding.word_embeddings(phones) + self.bert_proj(bert)
        return self.ar_text_position(x, torch.arange(phones.shape[1], device=phones.device))

    def embed_audio(self, tokens: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        return self.ar_audio_position(self.ar_audio_embedding.word_embeddings(tokens), positions)


def prompt_mask(n_text: torch.Tensor, n_audio: torch.Tensor, device) -> torch.Tensor:
    """The prompt's block mask; `True` blocks.

    Not one causal triangle across the join. Text attends within itself in both
    directions — it is given, not predicted — while audio attends over all the
    text and causally over itself. A plain causal mask would stop each phoneme
    seeing the ones after it, which nothing would report.

    Written as boolean algebra rather than the `torch.where` it reads as, and
    not for elegance: ONNX Runtime's CUDA provider has no `Where` for boolean
    outputs, and refuses to *load* a graph containing one ("Provider type for
    Where node … is not set") rather than falling back for that node.
    """
    n = n_text + n_audio
    idx = torch.arange(n, device=device)
    row, col = idx.unsqueeze(1), idx.unsqueeze(0)
    # A column in the audio half is blocked from every text row, and from an
    # audio row that has not reached it yet. Nothing in the text half is blocked.
    return (col >= n_text) & ((row < n_text) | (col > row))


# ---- the four exported graphs -----------------------------------------------


class ReferenceGraph(nn.Module):
    """`audio [1, S]` at 16 kHz → `codes [1, T]`, `speaker [1, 512, 1]`.

    The reference clip does two jobs and they are easy to conflate: its
    *semantic tokens* prime `s1`, and its *spectrogram* becomes the speaker
    vector. Both come from the same few seconds of audio, so both come out of
    one graph.
    """

    def __init__(self, hubert: Hubert, sovits: Sovits) -> None:
        super().__init__()
        self.hubert = hubert
        self.sovits = sovits
        self.spectral = Spectral()

    def forward(self, audio: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        ssl = self.hubert(audio).transpose(1, 2)
        codes = self.sovits.quantizer.encode(ssl)
        # The speaker vector wants the synthesizer's 32 kHz. Duplicating samples
        # is a crude 2x resample and adequate: `ref_enc` averages over time and
        # the imaging artefacts land above the bins it reads.
        spec = self.spectral(audio.repeat_interleave(2, dim=1))
        return codes, self.sovits.speaker(spec)


class S1Prompt(nn.Module):
    """`phones [1, P]`, `bert [1, P, 1024]`, `prompt [1, K]` → `logits [1, 1025]`
    and the key/value cache for every layer."""

    def __init__(self, t2s: T2s) -> None:
        super().__init__()
        self.t2s = t2s

    def forward(
        self, phones: torch.Tensor, bert: torch.Tensor, prompt: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        text = self.t2s.embed_text(phones, bert)
        # Audio positions restart at zero and are independent of the text's:
        # upstream applies the two positional encodings separately and only then
        # concatenates.
        audio = self.t2s.embed_audio(
            prompt, torch.arange(prompt.shape[1], device=prompt.device)
        )
        x = torch.cat([text, audio], dim=1)
        mask = prompt_mask(text.shape[1], audio.shape[1], x.device)

        cache: list[torch.Tensor] = []
        for layer in self.t2s.h.layers:
            x, k, v = layer(x, mask, None)
            cache += [k, v]
        logits = self.t2s.ar_predict_layer(x[:, -1])
        return (logits, *cache)


class S1Step(nn.Module):
    """One decode step: `token [1, 1]`, `position [1]` and the cache in, the same
    cache one position longer out.

    No mask: a single query against the whole cache can see all of it, and every
    position in the cache is in its past by construction.
    """

    def __init__(self, t2s: T2s) -> None:
        super().__init__()
        self.t2s = t2s

    def forward(self, token: torch.Tensor, position: torch.Tensor, *past: torch.Tensor):
        x = self.t2s.embed_audio(token, position)
        cache: list[torch.Tensor] = []
        for i, layer in enumerate(self.t2s.h.layers):
            x, k, v = layer(x, None, (past[2 * i], past[2 * i + 1]))
            cache += [k, v]
        logits = self.t2s.ar_predict_layer(x[:, -1])
        return (logits, *cache)


class S2(nn.Module):
    """`codes [1, T]`, `text [1, P]`, `speaker [1, 512, 1]`, `noise [1, 192, 2T]`
    → `audio [1, 1, 1280 T]`.

    `noise` is a graph input rather than a `randn` inside it, matching the `rnd`
    input of the RVC export: it makes the graph a pure function, so a Burn run
    and an ONNX run of the same tokens can be compared directly.
    """

    def __init__(self, sovits: Sovits) -> None:
        super().__init__()
        self.sovits = sovits

    def forward(
        self,
        codes: torch.Tensor,
        text: torch.Tensor,
        speaker: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        return self.sovits.decode(codes, text, speaker, noise)
