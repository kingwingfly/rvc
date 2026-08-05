"""Standalone torch implementation of BigVGAN, the last stage of Seed-VC.

Mirrors `crates/burn-seedvc/src/bigvgan.rs` field for field, so
`bigvgan_generator.pt` loads onto it by direct key mapping — the same
arrangement `rvc_infer.py` and `gptsovits_infer.py` have against their crates.
This is a clean-room reimplementation: it does not import the Seed-VC repository
or NVIDIA's.

Weight-norm is folded at load time by the exporter, so the convolutions here are
plain `nn.Conv1d`/`nn.ConvTranspose1d` whose `.weight` the driver reconstructs
from the checkpoint's `weight_g`/`weight_v` pair.

The mel this eats is the **natural-log** one BigVGAN was trained against —
magnitude spectrogram through a Slaney filterbank, clamped at 1e-5 and logged,
at n_fft 1024 / hop 256 on 22.05 kHz audio. A dB-scaled or power mel runs
happily and vocodes something else.

[`Graph`] at the bottom is `bigvgan.onnx`.

Provenance: `modules/bigvgan/` of Seed-VC (https://github.com/Plachtaa/seed-vc,
GPL-3.0), which vendors NVIDIA's BigVGAN (MIT) and julius' alias-free resampling
(Apache-2.0), read as a reference and never run or vendored.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

# Ratio the anti-aliasing wrapper resamples by, up and down, and the taps in its
# low-pass — upstream's `Activation1d` defaults, which nothing in BigVGAN
# overrides.
AA_RATIO = 2
AA_TAPS = 12


@dataclass
class BigVganConfig:
    """`nvidia/bigvgan_v2_22khz_80band_256x`, verbatim from its `config.json`.

    The upsample rates multiply to 256 — the `256x` of the name, and the mel hop
    the rest of Seed-VC agrees on, so one mel frame is exactly 256 samples.
    """

    num_mels: int = 80
    upsample_initial_channel: int = 1536
    upsample_rates: tuple[int, ...] = (4, 4, 2, 2, 2, 2)
    upsample_kernel_sizes: tuple[int, ...] = (8, 8, 4, 4, 4, 4)
    resblock_kernel_sizes: tuple[int, ...] = (3, 7, 11)
    resblock_dilation_sizes: tuple[tuple[int, ...], ...] = ((1, 3, 5),) * 3
    # `tanh` at the output, or a hard clamp to ±1. **False on v2** — these
    # weights were trained against the clamp, and `tanh` would compress every
    # peak instead of passing it. `use_bias_at_final` is false for the same
    # release, so `conv_post` has no bias tensor at all rather than a zero one.
    use_tanh_at_final: bool = False
    use_bias_at_final: bool = False
    snake_logscale: bool = True

    @property
    def hop(self) -> int:
        return math.prod(self.upsample_rates)


def get_padding(kernel: int, dilation: int) -> int:
    return (kernel * dilation - dilation) // 2


# ---- the snake activation ---------------------------------------------------


class SnakeBeta(nn.Module):
    """`x + sin²(exp(α)·x) / (exp(β) + 1e-9)`, α and β learned per channel.

    `"activation": "snakebeta"` with `"snake_logscale": true` in the weight
    repo's `config.json`, so this is **not** the plain `Snake` whose single α
    appears in both places. Two ways to get it wrong load at 100% and produce a
    plausible-looking waveform: using α where β belongs, which is right only
    where the two coincide and after training they do not, and dropping the
    `exp`, which on a log-scale checkpoint whose values sit near zero multiplies
    the periodic term by roughly 1e9.

    The 1e-9 is upstream's and is load-bearing rather than cosmetic: β is an
    exponential of a learned value and nothing stops it underflowing to zero.
    """

    def __init__(self, channels: int, logscale: bool) -> None:
        super().__init__()
        # Upstream's own initialisation: zeros in log space, ones in linear —
        # both meaning "gain 1", so an unloaded module is `x + sin²(x)`.
        init = torch.zeros(channels) if logscale else torch.ones(channels)
        self.alpha = nn.Parameter(init.clone())
        self.beta = nn.Parameter(init.clone())
        self.logscale = logscale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        alpha = self.alpha.view(1, -1, 1)
        beta = self.beta.view(1, -1, 1)
        if self.logscale:
            alpha, beta = alpha.exp(), beta.exp()
        return x + torch.sin(x * alpha).pow(2) / (beta + 1e-9)


# ---- alias-free resampling --------------------------------------------------


def kaiser_sinc_filter1d(cutoff: float, half_width: float, taps: int) -> torch.Tensor:
    """A Kaiser-windowed sinc low-pass, unit sum, `[1, 1, taps]`.

    Julius' `LowPassFilters` design as upstream vendors it: β comes from the
    transition width by Kaiser's own attenuation estimate. The sum
    normalisation is not cosmetic — without it the filter passes a constant at a
    gain slightly off 1, and that error compounds over the 218 resampling sites
    in one forward pass.

    Computed in float64 and narrowed once, so the result is the same twelve
    numbers `burn_seedvc::bigvgan::kaiser_sinc_filter1d` derives.
    """
    half = taps // 2
    atten = 2.285 * (half - 1) * math.pi * (4 * half_width) + 7.95
    if atten > 50.0:
        beta = 0.1102 * (atten - 8.7)
    elif atten >= 21.0:
        beta = 0.5842 * (atten - 21.0) ** 0.4 + 0.07886 * (atten - 21.0)
    else:
        beta = 0.0

    window = torch.kaiser_window(taps, periodic=False, beta=beta, dtype=torch.float64)
    # An even tap count has no centre sample, so its grid sits half a step off;
    # an odd one is centred on zero.
    offset = 0.5 if taps % 2 == 0 else 0.0
    time = torch.arange(taps, dtype=torch.float64) - half + offset
    taps_ = 2 * cutoff * window * torch.sinc(2 * cutoff * time)
    return (taps_ / taps_.sum()).to(torch.float32).view(1, 1, taps)


def replicate_pad(x: torch.Tensor, left: int, right: int) -> torch.Tensor:
    """`F.pad(..., mode="replicate")`, which is what the resamplers use rather
    than zeros: a zero-padded signal has a step at each end, and a 12-tap sinc
    turns that step into ringing inside the output."""
    return F.pad(x, (left, right), mode="replicate")


class UpSample1d(nn.Module):
    """2× interpolation through the low-pass, as one transposed convolution.

    The trims either side are the fiddliest arithmetic in the file, and getting
    them wrong slides every activation against the signal it is applied to
    rather than erroring.
    """

    def __init__(self, channels: int, taps: int) -> None:
        super().__init__()
        self.register_buffer("filter", kaiser_sinc_filter1d(0.5 / AA_RATIO, 0.6 / AA_RATIO, taps))
        self.channels = channels
        self.taps = taps
        self.pad = taps // AA_RATIO - 1
        self.trim_left = self.pad * AA_RATIO + (taps - AA_RATIO) // 2
        self.trim_right = self.pad * AA_RATIO + math.ceil((taps - AA_RATIO) / 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = replicate_pad(x, self.pad, self.pad)
        # One kernel shared by every channel, so the convolution is grouped and
        # the filter is broadcast rather than stored per channel.
        weight = self.filter.expand(self.channels, 1, self.taps)
        y = F.conv_transpose1d(x, weight, stride=AA_RATIO, groups=self.channels)
        # Interpolation spreads each input sample over `AA_RATIO` outputs, so
        # the filter's unit sum has to be undone to keep the amplitude.
        y = y * AA_RATIO
        return y[:, :, self.trim_left : y.shape[-1] - self.trim_right]


class LowPassFilter1d(nn.Module):
    """2× decimation through the same low-pass, as one strided convolution.

    The one place this mirror is *shallower* than nothing and deeper than
    `bigvgan.rs`, deliberately: Burn's `DownSample1d` holds the filter directly
    and remaps `.downsample.lowpass.filter` away on load, whereas keeping
    upstream's wrapper here means the checkpoint's own key names apply with no
    remap. Nothing is riding on it either way — the filter is a *buffer*, so the
    exporter derives it rather than reading it from the file — but the derived
    kernel can then be held against the stored copy under the key the file uses.
    """

    def __init__(self, channels: int, taps: int) -> None:
        super().__init__()
        self.register_buffer("filter", kaiser_sinc_filter1d(0.5 / AA_RATIO, 0.6 / AA_RATIO, taps))
        self.channels = channels
        self.taps = taps
        # Asymmetric for an even tap count, because the filter's centre falls
        # between two samples.
        self.pad_left = taps // 2 - (1 if taps % 2 == 0 else 0)
        self.pad_right = taps // 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = replicate_pad(x, self.pad_left, self.pad_right)
        weight = self.filter.expand(self.channels, 1, self.taps)
        return F.conv1d(x, weight, stride=AA_RATIO, groups=self.channels)


class DownSample1d(nn.Module):
    def __init__(self, channels: int, taps: int) -> None:
        super().__init__()
        self.lowpass = LowPassFilter1d(channels, taps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lowpass(x)


class AliasFreeActivation(nn.Module):
    """upsample → activate → downsample: the periodic activation applied where
    it cannot alias.

    sin² doubles the bandwidth of whatever it is fed, so applying it at the
    signal rate folds everything above Nyquist back down as audible aliasing.
    """

    def __init__(self, channels: int, logscale: bool) -> None:
        super().__init__()
        self.act = SnakeBeta(channels, logscale)
        self.upsample = UpSample1d(channels, AA_TAPS)
        self.downsample = DownSample1d(channels, AA_TAPS)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.downsample(self.act(self.upsample(x)))


# ---- the residual block and the generator -----------------------------------


class AmpBlock1(nn.Module):
    """BigVGAN's `AMPBlock1`: HiFi-GAN's ResBlock1 conv stacks with an
    alias-free snake wherever the leaky-ReLU was.

    The activations are one flat list of `2 × dilations` because the checkpoint
    indexes them that way. The forward pass takes the even entries before the
    dilated convolutions and the odd ones before the width-1 convolutions —
    upstream's `activations[::2]` and `activations[1::2]`.
    """

    def __init__(self, channels: int, kernel: int, dilations: tuple[int, ...], logscale: bool) -> None:
        super().__init__()
        self.convs1 = nn.ModuleList(
            nn.Conv1d(channels, channels, kernel, 1, get_padding(kernel, d), dilation=d)
            for d in dilations
        )
        self.convs2 = nn.ModuleList(
            nn.Conv1d(channels, channels, kernel, 1, get_padding(kernel, 1))
            for _ in dilations
        )
        self.activations = nn.ModuleList(
            AliasFreeActivation(channels, logscale) for _ in range(2 * len(dilations))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, (c1, c2) in enumerate(zip(self.convs1, self.convs2)):
            xt = self.activations[2 * i](x)
            xt = c1(xt)
            xt = self.activations[2 * i + 1](xt)
            x = x + c2(xt)
        return x


class BigVgan(nn.Module):
    """The vocoder: `[batch, num_mels, frames]` → `[batch, 1, frames · 256]`."""

    def __init__(self, cfg: BigVganConfig) -> None:
        super().__init__()
        if cfg.use_bias_at_final:
            raise ValueError("only the bias-free output convolution is modelled")

        initial = cfg.upsample_initial_channel
        self.conv_pre = nn.Conv1d(cfg.num_mels, initial, 7, 1, 3)
        self.ups = nn.ModuleList(
            nn.ConvTranspose1d(initial >> i, initial >> (i + 1), kernel, rate, (kernel - rate) // 2)
            for i, (rate, kernel) in enumerate(zip(cfg.upsample_rates, cfg.upsample_kernel_sizes))
        )
        # Three blocks per stage, stored flat: the checkpoint's `resblocks.7` is
        # stage 2's middle block, not stage 7's.
        self.resblocks = nn.ModuleList(
            AmpBlock1(initial >> (i + 1), kernel, dilations, cfg.snake_logscale)
            for i in range(len(cfg.upsample_rates))
            for kernel, dilations in zip(cfg.resblock_kernel_sizes, cfg.resblock_dilation_sizes)
        )

        final_channels = initial >> len(cfg.upsample_rates)
        self.activation_post = AliasFreeActivation(final_channels, cfg.snake_logscale)
        self.conv_post = nn.Conv1d(final_channels, 1, 7, 1, 3, bias=False)
        self.num_kernels = len(cfg.resblock_kernel_sizes)
        self.use_tanh_at_final = cfg.use_tanh_at_final

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        x = self.conv_pre(mel)
        for i in range(len(self.ups)):
            x = self.ups[i](x)
            # The three blocks are averaged, not summed: they are alternative
            # receptive fields over the same signal, and summing would triple
            # the level going into the next stage.
            blocks = self.resblocks[i * self.num_kernels : (i + 1) * self.num_kernels]
            acc = blocks[0](x)
            for block in blocks[1:]:
                acc = acc + block(x)
            x = acc / self.num_kernels

        x = self.conv_post(self.activation_post(x))
        return torch.tanh(x) if self.use_tanh_at_final else torch.clamp(x, -1.0, 1.0)


# ---- the exported graph -----------------------------------------------------


class Graph(nn.Module):
    """`mel [1, 80, T]` → `audio [1, 1, 256 T]` — `bigvgan.onnx`.

    Nothing here is data-dependent: the whole model is convolutions over a
    single dynamic time axis, so `T` is the only symbol in the graph and the
    output length follows from it.
    """

    INPUTS = ["mel"]
    OUTPUTS = ["audio"]

    def __init__(self, bigvgan: BigVgan) -> None:
        super().__init__()
        self.bigvgan = bigvgan

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        return self.bigvgan(mel)

    def dummy(self) -> tuple[torch.Tensor, ...]:
        # Plausible log-mel rather than white noise: the floor `Spectral` clamps
        # at is log(1e-5) ≈ -11.5, and speech sits a few units above it. Noise
        # centred on zero would trace the same graph but exercise the snake
        # activation in a range the weights never see.
        mel = torch.randn(1, 80, 64) * 1.5 - 5.0
        return (mel.clamp_min(math.log(1e-5)),)

    def dynamic_shapes(self) -> tuple:
        return ({2: torch.export.Dim.DYNAMIC},)
