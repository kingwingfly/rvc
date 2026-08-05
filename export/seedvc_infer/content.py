"""Standalone torch implementation of Seed-VC's content encoder.

Mirrors `burn-seedvc`'s `content.rs` and the encoder half of `burn-whisper`
field for field, so `openai/whisper-small`'s `model.safetensors` loads with a
direct key mapping — strip `model.encoder.` and every name lines up, exactly as
`burn-seedvc` does it. This is a clean-room reimplementation: it imports neither
the Seed-VC repository nor `openai/whisper`.

Nothing here is Seed-VC's own network. Seed-VC's content encoder *is*
whisper-small's encoder, frozen; what belongs to Seed-VC is the decision to read
acoustic features off a speech-recognition encoder and throw the decoder away,
which upstream spells `del whisper_model.decoder` and which this file spells by
never building one.

**Whisper's log-mel is not the mel the rest of the model runs on**, and the two
are close enough to substitute in silence: this one centres its STFT (reflect
padding of `n_fft / 2`), takes power rather than magnitude, and finishes in
`log10` under a floor eight decades below the window's own peak, at 16 kHz. The
transform every module downstream of the length regulator speaks is the
22 050 Hz HiFi-GAN one. Their frame counts both look plausible, so crossing them
shifts every feature by half a hop and rescales it, with nothing to see.

The exported graph is [`Graph`], and it is static in every axis — its docstring
says why the 30 s window is sidestepped rather than fought.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

# 16 kHz is not a choice: it is Whisper's analysis rate, baked into the weights.
CONTENT_SR = 16000
# Samples of 16 kHz audio behind one content frame — two 160-sample mel hops,
# because the encoder's second convolution has stride 2. So content arrives at
# 50 Hz, and every content-side length in the model derives from this number
# rather than from the mel's hop.
CONTENT_STRIDE = 320
WINDOW_SAMPLES = 30 * CONTENT_SR
MEL_N_FFT = 400
MEL_HOP = 160
# Mel frames in one window, before the stride-2 convolution halves them to the
# 1500 the positional table is sized for.
WINDOW_FRAMES = WINDOW_SAMPLES // MEL_HOP


@dataclass
class WhisperConfig:
    """`openai/whisper-small`'s encoder, transcribed from its `config.json`.

    Only the encoder's fields are here, because only the encoder is built. The
    decoder's are not omitted for brevity — a field nobody reads is a field that
    can drift from the checkpoint without anything noticing.
    """

    num_mel_bins: int = 80
    max_source_positions: int = 1500
    d_model: int = 768
    encoder_attention_heads: int = 12
    encoder_layers: int = 12
    encoder_ffn_dim: int = 3072


# ---- the log-mel front end --------------------------------------------------


def slaney_filterbank(n_mels: int) -> torch.Tensor:
    """`librosa.filters.mel(sr=16000, n_fft=400, n_mels=n_mels)`, `[n_mels, n_bins]`.

    Slaney scale with Slaney area normalisation, which is how Whisper's shipped
    `mel_filters.npz` was generated. Computed rather than loaded so the port has
    no data file to keep in step with.
    """
    f_sp = 200.0 / 3.0
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = math.log(6.4) / 27.0

    def hz_to_mel(f: float) -> float:
        if f < min_log_hz:
            return f / f_sp
        return min_log_mel + math.log(f / min_log_hz) / logstep

    def mel_to_hz(m: float) -> float:
        if m < min_log_mel:
            return m * f_sp
        return min_log_hz * math.exp(logstep * (m - min_log_mel))

    n_bins = MEL_N_FFT // 2 + 1
    freqs = torch.arange(n_bins, dtype=torch.float32) * CONTENT_SR / MEL_N_FFT
    mel_min, mel_max = hz_to_mel(0.0), hz_to_mel(CONTENT_SR / 2)
    edges = [
        mel_to_hz(mel_min + (mel_max - mel_min) * i / (n_mels + 1)) for i in range(n_mels + 2)
    ]

    fb = torch.zeros(n_mels, n_bins)
    for m in range(n_mels):
        lo, ctr, hi = edges[m], edges[m + 1], edges[m + 2]
        lower = (freqs - lo) / (ctr - lo)
        upper = (hi - freqs) / (hi - ctr)
        fb[m] = torch.clamp(torch.minimum(lower, upper), min=0.0) * (2.0 / (hi - lo))
    return fb


class LogMel(nn.Module):
    """`[batch, 480000]` at 16 kHz → `[batch, 80, 3000]`.

    Framing, windowing and the DFT fused into two convolutions rather than a
    `torch.stft` call. That is not a stylistic preference — it mirrors
    `content.rs`, which is written that way because Burn has no STFT, and it is
    also the form that traces to a graph of plain `Conv` nodes instead of an
    STFT op ONNX Runtime would have to be persuaded to support.

    The window is a **periodic** Hann (`torch.hann_window`'s default), which is
    what Whisper was trained on; the symmetric one is off by a sample and
    detunes every bin slightly.
    """

    def __init__(self, n_mels: int) -> None:
        super().__init__()
        n_bins = MEL_N_FFT // 2 + 1
        k = torch.arange(MEL_N_FFT, dtype=torch.float32)
        window = 0.5 - 0.5 * torch.cos(2 * math.pi * k / MEL_N_FFT)
        angle = 2 * math.pi * torch.arange(n_bins, dtype=torch.float32).unsqueeze(1) * k / MEL_N_FFT
        self.register_buffer("cos_kernel", (window * torch.cos(angle)).unsqueeze(1))
        self.register_buffer("sin_kernel", (-window * torch.sin(angle)).unsqueeze(1))
        self.register_buffer("filters", slaney_filterbank(n_mels))

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        # `torch.stft(center=True)`: reflect padding of half the FFT size, so
        # frame `t` is centred on sample `t * hop` rather than half a hop later.
        x = F.pad(audio.unsqueeze(1), (MEL_N_FFT // 2, MEL_N_FFT // 2), mode="reflect")
        real = F.conv1d(x, self.cos_kernel, stride=MEL_HOP)
        imag = F.conv1d(x, self.sin_kernel, stride=MEL_HOP)
        # `center=True` yields `1 + L/hop` frames and Whisper drops the last —
        # the one whose window reaches past the end of the audio.
        power = (real.pow(2) + imag.pow(2))[:, :, :WINDOW_FRAMES]

        log = torch.log10(torch.clamp(self.filters @ power, min=1e-10))
        # Floor eight decades below *this window's* peak, then rescale. Both are
        # taken over the whole window, which is why this is not a streaming
        # transform: a window must be complete before any of its frames are
        # final. The span is therefore always exactly 2, but where it sits
        # follows the loudest bin, so the output is not bounded to [-1, 1].
        floor = log.flatten(1).max(dim=1).values.view(-1, 1, 1) - 8.0
        return (torch.maximum(log, floor) + 4.0) / 4.0


# ---- whisper-small's encoder ------------------------------------------------


class Attention(nn.Module):
    """Multi-head self-attention, unmasked.

    Field names are Hugging Face's because those are the names in the
    checkpoint. One shape to remember: **`k_proj` has no bias** — Whisper omits
    it, since a constant added to every key shifts a row of logits equally and
    softmax cancels it.

    The decoder's causal mask is the trap this repo has already been bitten by,
    and it is absent here for the plain reason that an encoder has no future to
    hide: every frame of a 30 s window attends to every other.
    """

    def __init__(self, cfg: WhisperConfig) -> None:
        super().__init__()
        d = cfg.d_model
        self.q_proj = nn.Linear(d, d)
        self.k_proj = nn.Linear(d, d, bias=False)
        self.v_proj = nn.Linear(d, d)
        self.out_proj = nn.Linear(d, d)
        self.n_head = cfg.encoder_attention_heads

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        dh = d // self.n_head
        heads = lambda y: y.view(b, t, self.n_head, dh).transpose(1, 2)  # noqa: E731
        # The reference scales q and k by `d_head ** -0.25` each, an fp16
        # overflow guard; the product is the same and this mirror is fp32.
        q = heads(self.q_proj(x)) * dh**-0.5
        k = heads(self.k_proj(x))
        v = heads(self.v_proj(x))
        out = torch.softmax(q @ k.transpose(2, 3), dim=3) @ v
        return self.out_proj(out.transpose(1, 2).reshape(b, t, d))


class EncoderLayer(nn.Module):
    """One **pre**-norm residual block: normalise, transform, add."""

    def __init__(self, cfg: WhisperConfig) -> None:
        super().__init__()
        self.self_attn = Attention(cfg)
        self.self_attn_layer_norm = nn.LayerNorm(cfg.d_model)
        self.fc1 = nn.Linear(cfg.d_model, cfg.encoder_ffn_dim)
        self.fc2 = nn.Linear(cfg.encoder_ffn_dim, cfg.d_model)
        self.final_layer_norm = nn.LayerNorm(cfg.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(self.self_attn_layer_norm(x))
        return x + self.fc2(F.gelu(self.fc1(self.final_layer_norm(x))))


class AudioEncoder(nn.Module):
    """`mel [batch, 80, 3000]` → features `[batch, 1500, 768]`."""

    def __init__(self, cfg: WhisperConfig) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(cfg.num_mel_bins, cfg.d_model, 3, padding=1)
        # Stride 2: 3000 mel frames (30 s at 100 fps) become 1500.
        self.conv2 = nn.Conv1d(cfg.d_model, cfg.d_model, 3, stride=2, padding=1)
        # Sinusoidal, but shipped in the checkpoint as a plain table rather than
        # recomputed — so porting it exactly means *loading* it, and rebuilding
        # it from the formula would be a second implementation to keep in step.
        self.embed_positions = nn.Embedding(cfg.max_source_positions, cfg.d_model)
        self.layers = nn.ModuleList(EncoderLayer(cfg) for _ in range(cfg.encoder_layers))
        self.layer_norm = nn.LayerNorm(cfg.d_model)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.conv1(mel))
        x = F.gelu(self.conv2(x))
        x = x.transpose(1, 2)
        x = x + self.embed_positions.weight[: x.shape[1]].unsqueeze(0)
        for layer in self.layers:
            x = layer(x)
        return self.layer_norm(x)


class ContentEncoder(nn.Module):
    """Whisper's encoder as Seed-VC uses it: a frozen feature extractor.

    Holds the log-mel front end beside the network because the two are one
    interface — the encoder is only correct on mel computed exactly this way,
    and separating them is how a caller ends up feeding it the other mel.
    """

    def __init__(self, cfg: WhisperConfig | None = None) -> None:
        super().__init__()
        cfg = cfg or WhisperConfig()
        self.log_mel = LogMel(cfg.num_mel_bins)
        self.encoder = AudioEncoder(cfg)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        return self.encoder(self.log_mel(audio))


def load_encoder(model: ContentEncoder, state: dict[str, torch.Tensor]) -> dict[str, list[str]]:
    """Apply `openai/whisper-small`'s `model.safetensors` to `model`.

    The remap strips `model.encoder.`, so **the whole decoder lands in `unused`,
    and that is the expected result rather than a coverage gap**: upstream does
    `del whisper_model.decoder` for the same reason, because Seed-VC wants
    acoustic features and never generates text.

    Returns the three lists a Burn `ApplyResult` carries, so the numbers can be
    read against `burn-seedvc`'s. That crate measures **187 applied, 0 missing,
    0 errors, 342 unused**; the count here should be 187 / 0 / 0 / **292**,
    because its 342 include the 50 `weight`/`bias` entries of the encoder's own
    25 LayerNorms, which its adapter consumes under Burn's `gamma`/`beta` names
    and its store then records as unconsumed. `nn.LayerNorm` needs no such
    rename, so this side has nothing to overcount.

    Nothing is transposed on the way in. HF stores `Linear` as `[out, in]` and
    so does torch, which is the whole reason a mirror written against the *Burn*
    layout still loads a *PyTorch* checkpoint directly here — the transposition
    Burn needs happens inside Burn's adapter, not on disk.
    """
    prefix = "model.encoder."
    encoder = {k[len(prefix) :]: v for k, v in state.items() if k.startswith(prefix)}

    applied, missing, errors = [], [], []
    for name, param in model.encoder.named_parameters():
        found = encoder.get(name)
        if found is None:
            missing.append(name)
        elif tuple(found.shape) != tuple(param.shape):
            errors.append(f"{name}: checkpoint {tuple(found.shape)} vs module {tuple(param.shape)}")
        else:
            with torch.no_grad():
                param.copy_(found.to(torch.float32))
            applied.append(name)

    consumed = set(applied)
    unused = [k for k in state if not (k.startswith(prefix) and k[len(prefix) :] in consumed)]
    return {"applied": applied, "missing": missing, "errors": errors, "unused": unused}


# ---- the exported graph -----------------------------------------------------


class Graph(nn.Module):
    """`audio [1, 480000]` at 16 kHz → `content [1, 1500, 768]`.

    **Static in every axis, and that is what sidesteps Whisper's fixed window
    rather than fighting it.** Whisper's positional table is 1500 frames wide
    and the weights were trained on padded 30 s windows, so the graph always
    computes the whole window and always returns all 1500 frames. `content.rs`
    additionally slices to `min(samples / 320 + 1, 1500)` — the frames that
    describe real audio rather than the padding — but that bound is a function
    of the *sample count*, which a static graph does not have. So the host pads
    the clip up to 480 000 samples on the way in and takes the leading
    `samples / 320 + 1` frames on the way out, the same host-side arithmetic
    that carries the length regulator's gather indices and the Euler loop.

    A clip longer than 30 s is the host's problem too: upstream chunks it with a
    5 s overlap and stitches the features, and `content.rs` truncates instead.
    Neither belongs in a graph that cannot see how long the audio was.
    """

    INPUTS = ["audio"]
    OUTPUTS = ["content"]

    def __init__(self, encoder: ContentEncoder) -> None:
        super().__init__()
        self.encoder = encoder

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        return self.encoder(audio)

    def dummy(self) -> tuple[torch.Tensor, ...]:
        return (torch.randn(1, WINDOW_SAMPLES) * 0.1,)

    def dynamic_shapes(self) -> tuple:
        # One entry per input, and it is empty: no axis of this graph moves.
        return ({},)
