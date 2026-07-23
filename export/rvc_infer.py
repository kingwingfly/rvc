"""Standalone torch implementation of the RVC v2 inference generator.

Mirrors the Burn `burn-rvc` model (same module names/structure) so a
Burn-trained safetensors loads with a direct key mapping. Only the inference
path is implemented (`enc_p`, `flow`, `dec`, `emb_g`); the posterior encoder is
training-only and omitted. Weight-norm is folded at load time, so convolutions
here are plain. This is a clean-room reimplementation — it does not import the
RVC-Project repository.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class Config:
    spec_channels: int = 1025
    inter_channels: int = 192
    hidden_channels: int = 192
    filter_channels: int = 768
    n_heads: int = 2
    n_layers: int = 6
    kernel_size: int = 3
    window_size: int = 10
    resblock_kernel_sizes: tuple[int, ...] = (3, 7, 11)
    resblock_dilation_sizes: tuple[tuple[int, ...], ...] = ((1, 3, 5), (1, 3, 5), (1, 3, 5))
    upsample_rates: tuple[int, ...] = (12, 10, 2, 2)
    upsample_initial_channel: int = 512
    upsample_kernel_sizes: tuple[int, ...] = (24, 20, 4, 4)
    spk_embed_dim: int = 109
    gin_channels: int = 256
    sample_rate: int = 48000

    def hop(self) -> int:
        h = 1
        for r in self.upsample_rates:
            h *= r
        return h


# ---- prior / text encoder ---------------------------------------------------


class RvcLayerNorm(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))
        self.channels = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, -1)
        x = F.layer_norm(x, (self.channels,), self.gamma, self.beta, 1e-5)
        return x.transpose(1, -1)


class MultiHeadAttention(nn.Module):
    def __init__(self, channels: int, n_heads: int, window_size: int) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.window_size = window_size
        self.k_channels = channels // n_heads
        self.conv_q = nn.Conv1d(channels, channels, 1)
        self.conv_k = nn.Conv1d(channels, channels, 1)
        self.conv_v = nn.Conv1d(channels, channels, 1)
        self.conv_o = nn.Conv1d(channels, channels, 1)
        self.emb_rel_k = nn.Parameter(torch.zeros(1, window_size * 2 + 1, self.k_channels))
        self.emb_rel_v = nn.Parameter(torch.zeros(1, window_size * 2 + 1, self.k_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q, k, v = self.conv_q(x), self.conv_k(x), self.conv_v(x)
        b, d, t = q.size()
        q = q.view(b, self.n_heads, self.k_channels, t).transpose(2, 3)
        k = k.view(b, self.n_heads, self.k_channels, t).transpose(2, 3)
        v = v.view(b, self.n_heads, self.k_channels, t).transpose(2, 3)

        scores = torch.matmul(q / math.sqrt(self.k_channels), k.transpose(-2, -1))
        key_rel = self._get_relative_embeddings(self.emb_rel_k, t)
        rel_logits = torch.matmul(q / math.sqrt(self.k_channels), key_rel.unsqueeze(0).transpose(-2, -1))
        scores = scores + self._rel_to_abs(rel_logits)

        p_attn = F.softmax(scores, dim=-1)
        out = torch.matmul(p_attn, v)
        rel_weights = self._abs_to_rel(p_attn)
        value_rel = self._get_relative_embeddings(self.emb_rel_v, t)
        out = out + torch.matmul(rel_weights, value_rel.unsqueeze(0))
        out = out.transpose(2, 3).contiguous().view(b, d, t)
        return self.conv_o(out)

    def _get_relative_embeddings(self, emb: torch.Tensor, length: int) -> torch.Tensor:
        w = self.window_size
        pad = max(length - (w + 1), 0)
        start = max((w + 1) - length, 0)
        end = start + 2 * length - 1
        if pad > 0:
            emb = F.pad(emb, [0, 0, pad, pad, 0, 0])
        return emb[:, start:end]

    @staticmethod
    def _rel_to_abs(x: torch.Tensor) -> torch.Tensor:
        b, h, l, _ = x.size()
        x = F.pad(x, [0, 1, 0, 0, 0, 0, 0, 0])
        x = x.view(b, h, l * 2 * l)
        x = F.pad(x, [0, l - 1, 0, 0, 0, 0])
        return x.view(b, h, l + 1, 2 * l - 1)[:, :, :l, l - 1 :]

    @staticmethod
    def _abs_to_rel(x: torch.Tensor) -> torch.Tensor:
        b, h, l, _ = x.size()
        x = F.pad(x, [0, l - 1, 0, 0, 0, 0, 0, 0])
        x = x.view(b, h, l * l + l * (l - 1))
        x = F.pad(x, [l, 0, 0, 0, 0, 0])
        return x.view(b, h, l, 2 * l)[:, :, :, 1:]


class Ffn(nn.Module):
    def __init__(self, channels: int, filter_channels: int, kernel: int) -> None:
        super().__init__()
        self.conv_1 = nn.Conv1d(channels, filter_channels, kernel)
        self.conv_2 = nn.Conv1d(filter_channels, channels, kernel)
        self.pad_l = (kernel - 1) // 2
        self.pad_r = kernel // 2

    def _pad(self, x: torch.Tensor) -> torch.Tensor:
        return F.pad(x, [self.pad_l, self.pad_r])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.conv_1(self._pad(x)))
        return self.conv_2(self._pad(x))


class EncoderLayer(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.attn = MultiHeadAttention(cfg.hidden_channels, cfg.n_heads, cfg.window_size)
        self.norm_1 = RvcLayerNorm(cfg.hidden_channels)
        self.ffn = Ffn(cfg.hidden_channels, cfg.filter_channels, cfg.kernel_size)
        self.norm_2 = RvcLayerNorm(cfg.hidden_channels)


class Encoder(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.layers = nn.ModuleList(EncoderLayer(cfg) for _ in range(cfg.n_layers))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer.norm_1(x + layer.attn(x))
            x = layer.norm_2(x + layer.ffn(x))
        return x


class TextEncoder(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.emb_phone = nn.Linear(768, cfg.hidden_channels)
        self.emb_pitch = nn.Embedding(256, cfg.hidden_channels)
        self.encoder = Encoder(cfg)
        self.proj = nn.Conv1d(cfg.hidden_channels, cfg.inter_channels * 2, 1)
        self.hidden_channels = cfg.hidden_channels
        self.out_channels = cfg.inter_channels

    def forward(self, phone: torch.Tensor, pitch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.emb_phone(phone) + self.emb_pitch(pitch)
        x = x * math.sqrt(self.hidden_channels)
        x = F.leaky_relu(x, 0.1)
        x = x.transpose(1, 2)
        x = self.encoder(x)
        stats = self.proj(x)
        return torch.split(stats, self.out_channels, dim=1)


# ---- flow -------------------------------------------------------------------


class WN(nn.Module):
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
            res_skip = 2 * hidden if i < n_layers - 1 else hidden
            self.res_skip_layers.append(nn.Conv1d(hidden, res_skip, 1))

    def forward(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        out = torch.zeros_like(x)
        g = self.cond_layer(g)
        h = self.hidden
        for i in range(self.n_layers):
            x_in = self.in_layers[i](x)
            g_l = g[:, i * 2 * h : (i + 1) * 2 * h, :]
            acts = torch.tanh(x_in[:, :h] + g_l[:, :h]) * torch.sigmoid(x_in[:, h:] + g_l[:, h:])
            rs = self.res_skip_layers[i](acts)
            if i < self.n_layers - 1:
                x = x + rs[:, :h]
                out = out + rs[:, h:]
            else:
                out = out + rs
        return out


class ResidualCouplingLayer(nn.Module):
    def __init__(self, channels: int, hidden: int, gin: int) -> None:
        super().__init__()
        self.half = channels // 2
        self.pre = nn.Conv1d(self.half, hidden, 1)
        self.enc = WN(hidden, 5, 1, 3, gin)
        self.post = nn.Conv1d(hidden, self.half, 1)

    def _mean(self, x0: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        return self.post(self.enc(self.pre(x0), g))

    def forward(self, x: torch.Tensor, g: torch.Tensor, reverse: bool) -> torch.Tensor:
        x0, x1 = x[:, : self.half], x[:, self.half :]
        m = self._mean(x0, g)
        x1 = x1 - m if reverse else x1 + m
        return torch.cat([x0, x1], 1)


class ResidualCouplingBlock(nn.Module):
    def __init__(self, channels: int, hidden: int, gin: int, n_flows: int) -> None:
        super().__init__()
        self.flows = nn.ModuleList(ResidualCouplingLayer(channels, hidden, gin) for _ in range(n_flows))

    def forward(self, x: torch.Tensor, g: torch.Tensor, reverse: bool) -> torch.Tensor:
        if reverse:
            for flow in reversed(self.flows):
                x = torch.flip(x, [1])
                x = flow(x, g, reverse=True)
        else:
            for flow in self.flows:
                x = flow(x, g, reverse=False)
                x = torch.flip(x, [1])
        return x


# ---- NSF-HiFiGAN decoder ----------------------------------------------------


class SineGen(nn.Module):
    def __init__(self, sampling_rate: int) -> None:
        super().__init__()
        self.sr = sampling_rate
        self.amp = 0.1

    def forward(self, f0: torch.Tensor, upp: int) -> torch.Tensor:
        f0 = f0[:, None].transpose(1, 2)  # [b, t, 1]
        rad = (f0 / self.sr) % 1
        tmp = torch.cumsum(rad, 1) * upp
        tmp = F.interpolate(tmp.transpose(2, 1), scale_factor=float(upp), mode="linear", align_corners=True).transpose(2, 1)
        rad_up = F.interpolate(rad.transpose(2, 1), scale_factor=float(upp), mode="nearest").transpose(2, 1)
        tmp = tmp % 1
        wrap = (tmp[:, 1:, :] - tmp[:, :-1, :]) < 0
        shift = torch.zeros_like(rad_up)
        shift[:, 1:, :] = wrap * -1.0
        sine = torch.sin(torch.cumsum(rad_up + shift, dim=1) * 2 * math.pi) * self.amp
        uv = F.interpolate((f0 > 0).float().transpose(2, 1), scale_factor=float(upp), mode="nearest").transpose(2, 1)
        noise = uv * 0.003 + (1 - uv) * self.amp / 3
        sine = sine * uv + noise * torch.randn_like(sine)
        return sine


class SourceModule(nn.Module):
    def __init__(self, sampling_rate: int) -> None:
        super().__init__()
        self.l_sin_gen = SineGen(sampling_rate)
        self.l_linear = nn.Linear(1, 1)

    def forward(self, f0: torch.Tensor, upp: int) -> torch.Tensor:
        sine = self.l_sin_gen(f0, upp)
        return torch.tanh(self.l_linear(sine)).transpose(1, 2)  # [b, 1, t*upp]


class ResBlock1(nn.Module):
    def __init__(self, channels: int, kernel: int, dilations: tuple[int, ...]) -> None:
        super().__init__()

        def pad(d: int) -> int:
            return d * (kernel - 1) // 2

        self.convs1 = nn.ModuleList(
            nn.Conv1d(channels, channels, kernel, 1, dilation=d, padding=pad(d)) for d in dilations
        )
        self.convs2 = nn.ModuleList(
            nn.Conv1d(channels, channels, kernel, 1, dilation=1, padding=pad(1)) for _ in dilations
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = c2(F.leaky_relu(c1(F.leaky_relu(x, 0.1)), 0.1))
            x = x + xt
        return x


class GeneratorNsf(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        uic = cfg.upsample_initial_channel
        self.num_kernels = len(cfg.resblock_kernel_sizes)
        self.upp = cfg.hop()
        self.conv_pre = nn.Conv1d(cfg.inter_channels, uic, 7, 1, padding=3)
        self.m_source = SourceModule(cfg.sample_rate)
        self.ups = nn.ModuleList()
        self.noise_convs = nn.ModuleList()
        n_up = len(cfg.upsample_rates)
        for i in range(n_up):
            u, k = cfg.upsample_rates[i], cfg.upsample_kernel_sizes[i]
            self.ups.append(nn.ConvTranspose1d(uic >> i, uic >> (i + 1), k, u, padding=(k - u) // 2))
            if i + 1 < n_up:
                sf = 1
                for r in cfg.upsample_rates[i + 1 :]:
                    sf *= r
                self.noise_convs.append(nn.Conv1d(1, uic >> (i + 1), sf * 2, sf, padding=sf // 2))
            else:
                self.noise_convs.append(nn.Conv1d(1, uic >> (i + 1), 1))
        self.resblocks = nn.ModuleList()
        ch = uic
        for i in range(n_up):
            ch = uic >> (i + 1)
            for k, d in zip(cfg.resblock_kernel_sizes, cfg.resblock_dilation_sizes):
                self.resblocks.append(ResBlock1(ch, k, d))
        self.conv_post = nn.Conv1d(ch, 1, 7, 1, padding=3, bias=False)
        self.cond = nn.Conv1d(cfg.gin_channels, uic, 1)

    def forward(self, x: torch.Tensor, f0: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        har = self.m_source(f0, self.upp)
        x = self.conv_pre(x) + self.cond(g)
        for i, ups in enumerate(self.ups):
            x = F.leaky_relu(x, 0.1)
            x = ups(x) + self.noise_convs[i](har)
            xs = None
            for j in range(self.num_kernels):
                out = self.resblocks[i * self.num_kernels + j](x)
                xs = out if xs is None else xs + out
            x = xs / self.num_kernels
        x = self.conv_post(F.leaky_relu(x))
        return torch.tanh(x)


# ---- full generator (ONNX export contract) ----------------------------------


class Synthesizer(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.enc_p = TextEncoder(cfg)
        self.dec = GeneratorNsf(cfg)
        self.flow = ResidualCouplingBlock(cfg.inter_channels, cfg.hidden_channels, cfg.gin_channels, 4)
        self.emb_g = nn.Embedding(cfg.spk_embed_dim, cfg.gin_channels)

    def forward(
        self,
        phone: torch.Tensor,        # [1, T, 768]
        phone_lengths: torch.Tensor,  # [1] (unused, full length)
        pitch: torch.Tensor,        # [1, T]
        pitchf: torch.Tensor,       # [1, T]
        ds: torch.Tensor,           # [1]
        rnd: torch.Tensor,          # [1, 192, T]
    ) -> torch.Tensor:
        g = self.emb_g(ds).unsqueeze(-1)
        m_p, logs_p = self.enc_p(phone, pitch)
        # Sequence mask from phone_lengths (all ones at full length); keeps
        # phone_lengths as a graph input for the rvc-core inference contract.
        idx = torch.arange(phone.shape[1], device=phone.device).unsqueeze(0)
        mask = (idx < phone_lengths.unsqueeze(1)).to(m_p.dtype).unsqueeze(1)
        m_p = m_p * mask
        logs_p = logs_p * mask
        z_p = m_p + torch.exp(logs_p) * rnd
        z = self.flow(z_p, g, reverse=True)
        return self.dec(z, pitchf, g)
