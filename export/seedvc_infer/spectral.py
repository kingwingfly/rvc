"""Seed-VC's 22.05 kHz log-mel front end, mirroring `burn-vits`' `Spectral`.

Clean-room: written from `crates/burn-vits/src/spectral.rs` and the preset that
`crates/burn-seedvc/src/content.rs::mel_config` hands it, not from the Seed-VC
repository, which is read as a porting reference and neither run nor vendored.

**`gptsovits_infer.py` already carries a mirror of the same Burn module, and
this is deliberately a second one rather than a shared import.** That one stops
at the *linear* magnitude spectrogram for GPT-SoVITS' posterior encoder at
32 kHz / n_fft 2048 / hop 640; Seed-VC's diffusion transformer and BigVGAN eat
an *80-band log mel* at 22 050 / 1024 / 256. The two differ in both halves that
matter — the preset and where the transform stops — so the shared version would
be a config object threaded through two exporters to save nine lines.

**This is one of three 80-band front ends in the toolkit, and none of them is
interchangeable with the others.** Whisper's log-mel (16 kHz, centred STFT,
power rather than magnitude, `log10` under a peak-relative floor) and CAMPPlus'
Kaldi filterbank (16 kHz, mean-normalised over time, `[batch, frames, bins]`)
also emit 80 bands, at frame counts that line up. Substituting one for another
therefore runs happily and computes something else. This is the 22.05 kHz one,
and it is the only one BigVGAN was trained against.

`Spectral` holds **no weights at all** — every kernel is derived from the config
— so `mel.onnx` is a pure transform graph and the exporter has nothing to load
into it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


# ---- the preset -------------------------------------------------------------


@dataclass
class SpectralConfig:
    """Mirrors `burn_vits::SpectralConfig`; the defaults are Seed-VC's preset.

    Which is to say `mel_config(SeedVcConfig::uvit_whisper_small_wavenet())`:
    the model's own 22 050 Hz, `fmax` at Nyquist, and a hop that makes one mel
    frame one vocoder hop.
    """

    sample_rate: int = 22_050
    n_fft: int = 1024
    hop: int = 256
    win_length: int = 1024
    n_mels: int = 80
    fmin: float = 0.0
    fmax: float = 11_025.0

    @property
    def n_bins(self) -> int:
        return self.n_fft // 2 + 1


# ---- the Slaney mel scale ---------------------------------------------------
#
# librosa's, which is what `burn-vits` implements: linear at 200/3 Hz per mel
# below 1 kHz and logarithmic above it, with the two halves meeting at the same
# value so the curve is continuous.


def hz_to_mel(f: float) -> float:
    f_sp, min_log_hz = 200.0 / 3.0, 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = math.log(6.4) / 27.0
    if f >= min_log_hz:
        return min_log_mel + math.log(f / min_log_hz) / logstep
    return f / f_sp


def mel_to_hz(m: torch.Tensor) -> torch.Tensor:
    f_sp, min_log_hz = 200.0 / 3.0, 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = math.log(6.4) / 27.0
    return torch.where(
        m >= min_log_mel,
        min_log_hz * torch.exp(logstep * (m - min_log_mel)),
        f_sp * m,
    )


# ---- log-mel spectrogram ----------------------------------------------------


class Spectral(nn.Module):
    """Framing, windowing and the DFT fused into two convolutions, then mel.

    Not a `torch.stft` call. `burn-vits` is built this way because Burn has no
    STFT, and mirroring *that* rather than upstream's `torch.stft` is what makes
    the graph exportable at all: a fused `[n_bins, 1, n_fft]` kernel traces as
    plain `F.conv1d`, where `torch.stft` does not trace cleanly.
    """

    def __init__(self, cfg: SpectralConfig) -> None:
        super().__init__()
        n_fft, n_bins = cfg.n_fft, cfg.n_bins

        # A **periodic** Hann — `i / win`, not `i / (win - 1)` — zero-padded and
        # centred inside the frame. Seed-VC's preset has `win_length == n_fft`,
        # so the padding is empty here and the window fills the frame; the
        # centring is kept because that is what `burn-vits`' `hann` does and a
        # preset that shortens the window would otherwise silently misalign.
        window = torch.zeros(n_fft)
        pad = (n_fft - cfg.win_length) // 2
        i = torch.arange(cfg.win_length)
        window[pad : pad + cfg.win_length] = 0.5 - 0.5 * torch.cos(2 * math.pi * i / cfg.win_length)

        b = torch.arange(n_bins).unsqueeze(1)
        k = torch.arange(n_fft).unsqueeze(0)
        angle = 2 * math.pi * b * k / n_fft
        self.register_buffer("cos_kernel", (window * torch.cos(angle)).unsqueeze(1))
        self.register_buffer("sin_kernel", (-window * torch.sin(angle)).unsqueeze(1))

        fft_freqs = torch.arange(n_bins, dtype=torch.float32) * cfg.sample_rate / n_fft
        mel_min, mel_max = hz_to_mel(cfg.fmin), hz_to_mel(cfg.fmax)
        steps = torch.arange(cfg.n_mels + 2, dtype=torch.float32)
        # Multiply before dividing, which is the association `spectral.rs` uses;
        # in float32 the other grouping differs by an ulp per band.
        hz = mel_to_hz(mel_min + (mel_max - mel_min) * steps / (cfg.n_mels + 1))
        lower, centre, upper = hz[:-2, None], hz[1:-1, None], hz[2:, None]
        triangle = torch.minimum(
            (fft_freqs - lower) / (centre - lower),
            (upper - fft_freqs) / (upper - centre),
        )
        # Slaney area normalisation: each filter integrates to a constant rather
        # than peaking at one, so wide high-frequency bands do not dominate.
        self.register_buffer("mel_fb", triangle.clamp_min(0.0) * (2.0 / (upper - lower)))

        self.hop = cfg.hop
        # center=False with (n_fft - hop)/2 reflect padding, so the frame count
        # is exactly L // hop — the grid everything downstream of the length
        # regulator lives on, and the reason a mel frame is one vocoder hop.
        self.pad = (n_fft - cfg.hop) // 2

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """`[batch, samples]` → `[batch, n_mels, samples // hop]` log mel."""
        x = F.pad(wav.unsqueeze(1), (self.pad, self.pad), mode="reflect")
        real = F.conv1d(x, self.cos_kernel, stride=self.hop)
        imag = F.conv1d(x, self.sin_kernel, stride=self.hop)
        # The epsilon goes **inside** the square root, which is where
        # `burn-vits` puts it. Outside it would be a constant added to every
        # magnitude rather than a floor under the small ones, and it would leave
        # the derivative of `sqrt` unbounded at a silent bin — which is what the
        # epsilon is there for, since `burn-vits` backprops the mel-L1 loss
        # through this transform.
        mag = (real.pow(2) + imag.pow(2) + 1e-9).sqrt()
        # A **natural** log under a 1e-5 floor, not the dB or `log10` the other
        # two front ends take. This is the scale BigVGAN was trained against.
        return torch.matmul(self.mel_fb, mag).clamp_min(1e-5).log()


# ---- the exported graph -----------------------------------------------------


class Graph(nn.Module):
    """`audio [1, S]` at 22 050 Hz → `mel [1, 80, S // 256]`.

    Weightless, so the exporter constructs it and writes it out with nothing to
    load. `S` is dynamic; the frame count follows from it by the convolution's
    own arithmetic rather than by anything the host has to compute.
    """

    INPUTS = ["audio"]
    OUTPUTS = ["mel"]

    def __init__(self, spectral: Spectral) -> None:
        super().__init__()
        self.spectral = spectral

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        return self.spectral(audio)

    def dummy(self) -> tuple[torch.Tensor, ...]:
        # One second, which is comfortably longer than the 384-sample reflect
        # pad — a clip shorter than that cannot be padded at all.
        return (torch.randn(1, 22_050) * 0.1,)

    def dynamic_shapes(self) -> tuple:
        return ({1: torch.export.Dim.DYNAMIC},)
