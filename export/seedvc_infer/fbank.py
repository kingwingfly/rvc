"""Kaldi's filterbank — the only input CAM++ has ever been shown.

Mirrors `crates/burn-seedvc/src/fbank.rs` field for field, as `rvc_infer.py`
mirrors `burn-rvc`. This is a clean-room reimplementation: it imports neither
Seed-VC nor 3D-Speaker, and it deliberately does **not** call
`torchaudio.compliance.kaldi`.

That last point is a tractability argument as much as a licence one. Upstream's
`kaldi.fbank` is a loop over framing, per-frame DC removal, preemphasis, a
window and an FFT, and it does not trace cleanly into a graph with a dynamic
sample count. The Burn port already folded that whole chain into two precomputed
`[bins, 1, 400]` conv1d kernels — an exact rewrite, because everything Kaldi
does between a frame and its spectrum is linear in that frame — so mirroring
*Burn* leaves a front end that is two `F.conv1d` calls and a matmul, and exports
without argument.

There is no `Graph` class here. This front end is not a graph of its own: it
lives **inside** `campplus.Graph`, which is what `style.onnx` is, and pulling it
out would put the mean subtraction below one forgotten line away — see
[`Fbank.forward`].
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

# The exponent that turns a symmetric Hann window into Kaldi's Povey window.
POVEY_EXPONENT = 0.85


# ---- configuration ----------------------------------------------------------


@dataclass(frozen=True)
class FbankConfig:
    """Everything `kaldi.fbank` is called with, resolved.

    The defaults are Seed-VC's own call — `num_mel_bins=80, dither=0,
    sample_frequency=16000` — plus torchaudio's defaults for everything it
    leaves out, so the default is the only configuration Seed-VC ever uses.
    """

    sample_rate: int = 16_000
    num_mel_bins: int = 80
    frame_length_ms: float = 25.0
    frame_shift_ms: float = 10.0
    low_freq: float = 20.0
    # Kaldi spells this `0.0` and means "the Nyquist rate"; resolved here so
    # nothing downstream has to know that convention.
    high_freq: float = 8_000.0
    preemphasis: float = 0.97

    @property
    def window(self) -> int:
        """Samples in one analysis window, 400."""
        return int(self.sample_rate * self.frame_length_ms * 1e-3)

    @property
    def shift(self) -> int:
        """Samples between consecutive windows, 160."""
        return int(self.sample_rate * self.frame_shift_ms * 1e-3)

    @property
    def padded_window(self) -> int:
        """The FFT length, 512: `window` rounded up to a power of two."""
        return 1 << (self.window - 1).bit_length()


def mel(hz: float) -> float:
    """Kaldi's mel scale, `1127·ln(1 + f/700)`.

    Not the Slaney scale the 22.05 kHz mel and Whisper both use: that one is
    linear below 1 kHz and logarithmic above, where this is one expression over
    the whole range. They disagree by tens of hertz in the middle of the band.
    """
    return 1127.0 * math.log(1.0 + hz / 700.0)


# ---- the transform ----------------------------------------------------------


class Fbank(nn.Module):
    """`audio [batch, samples]` at 16 kHz → `[batch, frames, bins]`.

    **Frames before bins**, which is what `campplus.CamPPlus.forward` takes, and
    **the per-clip mean is already subtracted**. Upstream writes the speaker
    encoder's input in two lines and both of them are this method:

        feat = kaldi.fbank(wave_16k, num_mel_bins=80, dither=0, sample_frequency=16000)
        feat = feat - feat.mean(dim=0, keepdim=True)

    **The subtraction is not a nicety.** CAM++ has no input normalisation of its
    own, and leaving it out is invisible — the embedding stays finite, stays
    repeatable, and quietly starts keying on the recording's channel rather than
    on the speaker. It was dropped twice in one week while this engine was being
    built, which is why the raw filterbank is not offered separately.

    Every difference from the 22.05 kHz 80-band mel the transformer predicts is
    a place where substituting one for the other would run and compute something
    else: per-frame DC removal, preemphasis 0.97 against a replicate tap, the
    Povey window, a **right** zero-pad to 512, `snip_edges` framing so
    `frames = 1 + (samples − 400) / 160`, a **power** spectrum, Kaldi's own mel
    scale from 20 Hz, and a Nyquist bin that carries no weight at all.

    Waveform scale does not matter: a gain of `a` adds `2·ln a` to every bin of
    every frame and the mean subtraction removes exactly that. The one place it
    is visible is the epsilon floor, so this follows upstream's `[-1, 1]` float
    convention rather than Kaldi's int16 one — the floor has to sit where the
    released model saw it.
    """

    def __init__(self, cfg: FbankConfig = FbankConfig()) -> None:
        super().__init__()
        window, padded = cfg.window, cfg.padded_window
        bins = padded // 2 + 1
        self.cfg = cfg
        self.shift = cfg.shift
        # `torch.finfo(torch.float).eps`, the value `_get_epsilon` hands
        # `torch.max`, and what the released model's log was floored at.
        self.epsilon = torch.finfo(torch.float32).eps

        # Built in float64 and narrowed once at the end. In float32 the taps
        # accumulate visible error at the top of the band, and the frame grid
        # itself is worse: `16000 * 10.0 * 1e-3` comes out 159.99999 and
        # truncates to 159, a grid that drifts a sample per window and fails no
        # assertion anywhere.
        n = torch.arange(window, dtype=torch.float64)
        hann = 0.5 - 0.5 * torch.cos(2.0 * math.pi * n / (window - 1))
        povey = hann.pow(POVEY_EXPONENT)

        # The windowed DFT basis, `[bins, window]`. The frame is zero-padded on
        # the right to `padded`, so only its first `window` taps can meet a
        # sample and the padding never has to be materialised.
        angle = 2.0 * math.pi * torch.arange(bins, dtype=torch.float64).unsqueeze(1) * n / padded

        # Preemphasis, folded in from the right: sample `j` reaches outputs `j`
        # and `j + 1`, so a tap on input `j` picks up `basis[j]` less
        # `0.97 · basis[j + 1]`. The replicate pad at the left edge is why tap 0
        # keeps only `1 - 0.97` of itself. DC removal folds in the same way:
        # subtracting a frame's own mean from each of its samples subtracts the
        # row's mean from each of its taps.
        own = torch.ones(window, dtype=torch.float64)
        own[0] = 1.0 - cfg.preemphasis

        def kernel(basis: torch.Tensor) -> torch.Tensor:
            taps = own * basis - cfg.preemphasis * F.pad(basis[:, 1:], (0, 1))
            return (taps - taps.mean(dim=1, keepdim=True)).unsqueeze(1).float()

        self.register_buffer("cos_kernel", kernel(povey * torch.cos(angle)))
        self.register_buffer("sin_kernel", kernel(povey * -torch.sin(angle)))

        # Kaldi lays `num_mel_bins` triangles over `[low_freq, high_freq]` at
        # even spacing *in the mel domain*, each spanning two spacings and
        # peaking in the middle — hence the `+ 1` in the denominator, which is
        # the end effect of the first and last triangles hanging off the edges.
        num_fft_bins = padded // 2
        fft_bin_width = cfg.sample_rate / padded
        delta = (mel(cfg.high_freq) - mel(cfg.low_freq)) / (cfg.num_mel_bins + 1)
        centres = torch.tensor(
            [mel(fft_bin_width * k) for k in range(num_fft_bins)], dtype=torch.float64
        )
        left = mel(cfg.low_freq) + torch.arange(cfg.num_mel_bins, dtype=torch.float64).unsqueeze(1) * delta
        up = (centres - left) / delta
        down = (left + 2.0 * delta - centres) / delta

        # Left `bins` wide with the Nyquist column zero, which is exactly the
        # zero column torchaudio pads on: Kaldi's bank stops one bin short of
        # what an `rfft` returns.
        mel_fb = torch.zeros(cfg.num_mel_bins, bins, dtype=torch.float64)
        mel_fb[:, :num_fft_bins] = torch.minimum(up, down).clamp_min(0.0)
        self.register_buffer("mel_fb", mel_fb.float())

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        x = wav.unsqueeze(1)
        real = F.conv1d(x, self.cos_kernel, stride=self.shift)
        imaginary = F.conv1d(x, self.sin_kernel, stride=self.shift)
        power = real.square() + imaginary.square()
        feats = (self.mel_fb @ power).clamp_min(self.epsilon).log().transpose(1, 2)
        return feats - feats.mean(dim=1, keepdim=True)
