// RVC v2 — Architecture, GAN training, and the science of neural voice conversion.
// Compile:  typst compile docs/rvc-architecture.typ
// Pure native Typst — no external packages required.

#set document(title: "Retrieval-based Voice Conversion — Architecture & Training", author: "rvc")
#set page(
  paper: "a4",
  margin: (x: 2.4cm, top: 2.6cm, bottom: 2.4cm),
  numbering: "1",
  number-align: center,
)
#set text(font: ("New Computer Modern", "Latin Modern Roman", "DejaVu Serif"), size: 10.5pt, lang: "en")
#set par(justify: true, leading: 0.68em)
#set heading(numbering: "1.1")
#show heading: set block(above: 1.3em, below: 0.7em)

// ---- palette -------------------------------------------------------------
#let ink      = rgb("#1c2530")
#let primary  = rgb("#2d4a7c")
#let accent   = rgb("#b1541f")
#let c-feat   = rgb("#e8f0fb")
#let c-prior  = rgb("#e9f6ec")
#let c-lat    = rgb("#fff4dc")
#let c-dec    = rgb("#fdeede")
#let c-disc   = rgb("#f4e8f3")
#let c-loss   = rgb("#eeeeee")
#let stroke-c = rgb("#9fb0c8")

#show heading.where(level: 1): set text(fill: primary)
#show heading.where(level: 2): set text(fill: primary.darken(10%))
#show heading.where(level: 3): set text(fill: ink)
#set raw(theme: none)
#show raw.where(block: false): it => box(fill: rgb("#f2f4f8"), inset: (x: 3pt, y: 0pt), outset: (y: 3pt), radius: 2pt, text(size: 9pt, it))

// ---- diagram helpers -----------------------------------------------------
#let node(body, fill: c-prior, w: auto) = box(
  fill: fill, inset: (x: 9pt, y: 7pt), radius: 5pt,
  stroke: 0.7pt + stroke-c, width: w,
)[#set text(size: 9pt); #set par(justify: false, leading: 0.5em); #align(center)[#body]]

#let ar = text(fill: primary, weight: "bold", size: 12pt)[#h(5pt)→#h(5pt)]
#let dn = align(center, text(fill: primary, weight: "bold", size: 12pt)[↓])

#let panel(body, caption: none) = figure(
  box(fill: rgb("#fbfcfe"), stroke: 0.6pt + rgb("#dbe3ee"), inset: 13pt, radius: 7pt, width: 100%, body),
  caption: caption,
  kind: image,
  supplement: [Figure],
)

// =========================================================================
#align(center)[
  #v(1.2cm)
  #text(size: 21pt, weight: "bold", fill: primary)[Retrieval-based Voice Conversion]
  #v(2pt)
  #text(size: 14pt, fill: accent)[The architecture and training of a VITS-derived, GAN-trained retimbre model]
  #v(6pt)
  #text(size: 10pt, fill: ink.lighten(20%))[A conceptual walkthrough of RVC v2 — grounded in the `rvc` pure-Rust reimplementation]
  #v(0.5cm)
  #line(length: 40%, stroke: 0.8pt + stroke-c)
]
#v(0.4cm)

#block(inset: (x: 6pt))[
  #set text(size: 9.8pt)
  *Abstract.* — RVC (Retrieval-based Voice Conversion) takes a *source* recording and
  re-renders it in a *target* voice, keeping *what was said and how it was
  pitched* while replacing *who is saying it*. This document explains the ideas
  that make that possible: how the model separates linguistic content and pitch
  from speaker timbre, how it is structured as a conditional variational
  autoencoder wrapped around a neural vocoder, and how a *generative
  adversarial network* is used to teach that vocoder to synthesise waveforms
  indistinguishable from real audio. Throughout, the concepts are anchored to a
  concrete open implementation so the abstract picture stays checkable against
  running code.
]

#outline(depth: 2, indent: auto)
#v(0.3cm)

// =========================================================================
= Provenance: which RVC this is

Everything below describes *RVC v2*, mirrored from the reference implementation
`RVC-Project/Retrieval-based-Voice-Conversion-WebUI` at tag `2.2.231006` — and
within it `infer/lib/infer_pack/{models,attentions,modules}.py`, where
`SynthesizerTrnMs768NSFsid` and `MultiPeriodDiscriminatorV2` are defined. The
Rust modules keep that reference's *field names* and nesting, so a published
PyTorch `state_dict` maps onto the module tree parameter for parameter. That is a
standing constraint on the port rather than a convenience: without it there is no
warm-start (§7), and on a one-hour corpus there is no useful training without
warm-start.

== "v2" is not a version to be behind on

The name invites the assumption that a v3 exists and this port has not caught up.
*It does not.* A v3 has been announced in upstream's release notes since
`2.1.230814` (August 2023) and has never shipped. Upstream's most recent release,
`2.3.260718` of 21 July 2026, says so in as many words — *base model unchanged* —
and what it does change sits outside the network entirely:

#block(inset: (x: 10pt))[
  #set text(size: 9.8pt)
  - *packaging and WebUI ergonomics* — installation, defaults, the browser front
    end. None of it reaches the model.
  - *vocal separation swapped from UVR5 to PyMSS* — a corpus-preparation tool that
    runs before training ever starts. This toolkit does not use either; its
    corpus preparation is the sentence-safe slicer of §10.
  - *FCPE added as a pitch extractor* — an alternative to RMVPE (§4) in the
    analysis front end, which is *frozen and speaker-independent* and therefore
    swappable without touching a single trained weight.
  - *CUDA-graph speed-ups on the real-time path* — an inference-scheduling change.
]

So the tag pinned above fixes the *source text* being mirrored, not the generation
of the model: v2 is upstream's current architecture, and a v2 port is level with
it. If v3 ever lands it will be a new network to port, not a migration.

== The warm-start bases

A "base" is a generator/discriminator pair trained on many speakers, which
fine-tuning then specialises to one voice (§7). Several exist for *this same
architecture*, so they differ only in the weights, never in the shapes of §6:

#align(center, block(width: 96%)[
  #set text(size: 9pt)
  #set par(justify: false)
  #table(
    columns: (auto, auto, 1fr),
    align: (left, center, left),
    stroke: 0.4pt + stroke-c,
    inset: 6pt,
    fill: (_, row) => if row == 0 { c-loss } else { white },
    [*base*], [*rates*], [*notes*],
    [`f0G48k.pth` / `f0D48k.pth`], [32/40/48 kHz],
      [the stock v2 base, from HF `lj1995/VoiceConversionWebUI`; the 48 kHz pair
       is what this toolkit fetches (76 MB + 143 MB)],
    [TITAN-Medium], [32/40/48 kHz],
      [community, Apache-2.0, ships G *and* D at every rate; trained on 11.15 h of
       the Expresso corpus],
    [Ov2Super], [32/40 kHz], [community; no 48 kHz weights published],
    [RIN_E3], [32/40 kHz], [community; no 48 kHz weights published],
  )
])

Because the architecture is identical, swapping a base is a matter of pointing the
trainer's pretrained generator and discriminator flags at different files — the
loader remaps and applies them exactly as it does the stock pair.

*TITAN-Medium is the only alternative reachable at 48 kHz, and it is deliberately
not wired up.* Nothing in the toolkit fetches or selects it, and the reason is
that it has not been compared against the stock base on this project's own
corpora; making it a default would ship an untested claim about output quality on
soft, breathy material, which is the material that matters here. Passing its G/D
pair by hand is expected to work and *has not been tried* — an honest untested
path, not a supported one. Ov2Super and RIN_E3 are out of reach for a blunter
reason: they publish nothing at 48 kHz, and 48 kHz is the only rate this trainer
accepts.

== Why 48 kHz is the only rate

The trainer rejects any other `--model-sr` outright
(`crates/rvc-train/src/lib.rs:140-143`). That limit is *not* architectural:
`SynthesizerConfig::v2_40k()` already exists (`crates/burn-rvc/src/config.rs:67-76`)
and departs from the 48 kHz configuration in two lines — upsample rates
$10 · 10 · 2 · 2$ instead of $12 · 10 · 2 · 2$, giving a hop of 400 samples rather
than 480, with transposed-conv kernels to match. Everything *around* the network
is what assumes 480: the analysis frame grid, the training STFT front end
($n_"fft" = 2048$, hop 480), the dataset windowing, and the base weights that
would have to be fetched at the new rate. Reaching 40 kHz is therefore a plumbing
job rather than a modelling one — and until someone does it, the two 32/40 kHz
community bases cannot be used at all.

// =========================================================================
= What voice conversion actually is

A speech signal braids together several independent things at once: the
*linguistic content* (the sequence of phonemes — the words), the *prosody*
(the pitch contour, or fundamental frequency $F_0$, and rhythm), and the
*timbre* (the resonances of a particular vocal tract — the thing that makes a
voice recognisable). Text-to-speech generates all three from scratch. Voice
conversion is a narrower, in some ways harder, problem: keep content and
prosody *exactly*, and swap only the timbre.

The design principle that follows is *disentanglement*. If we can extract a
representation of a frame of audio that captures *what is being said* but is
*blind to who is saying it*, then we can hand that content-only representation,
together with the desired pitch, to a decoder that has learned one specific
target voice — and the decoder has no choice but to render the same content in
that voice. RVC leans on this hard, and it is why breathy, non-lexical
vocalizations (sighs, gasps) survive conversion by construction: they are part
of the content and pitch streams, not something the model has to "recognise."

#panel(caption: [The end-to-end conversion path. Everything above the decoder is
speaker-independent analysis; the decoder is the only speaker-specific part.])[
  #set align(center)
  #node([source waveform\ #text(size: 7.5pt, fill: ink.lighten(30%))[any voice, mono f32]], fill: c-feat)
  #dn
  #node([resample → 16 kHz], fill: c-feat)
  #dn
  #grid(columns: (auto, 20pt, auto), align: horizon, gutter: 0pt,
    node([*ContentVec* (SSL)\ → content `[T,768]`], fill: c-feat),
    [],
    node([*RMVPE*\ → $F_0$ contour `[T]`], fill: c-feat),
  )
  #dn
  #grid(columns: (auto, 20pt, auto), align: horizon, gutter: 0pt,
    node([content vectors], fill: c-feat),
    [],
    node([coarse-pitch ids + fine $F_0$ (Hz)], fill: c-feat),
  )
  #dn
  #node([*Prior encoder* `enc_p` → $(mu_p, log sigma_p)$ ; sample the prior $z_p$], fill: c-prior, w: 78%)
  #dn
  #node([*Normalizing flow* (reverse) : $z_p → z$], fill: c-prior, w: 78%)
  #dn
  #node([*NSF-HiFiGAN decoder* `dec` + speaker embedding $g$ #sym.arrow.r waveform], fill: c-dec, w: 78%)
  #dn
  #node([target waveform → resample to output rate], fill: c-dec)
]

The rest of this document unfolds that diagram top to bottom, then explains how
the speaker-specific decoder is *trained* — which is where the GAN enters.

// =========================================================================
= Background: a few building blocks

Readers new to audio machine learning may want these six ideas first; every later
section leans on them. Experienced readers can skip ahead.

#block(inset: (x: 10pt))[
  #set text(size: 9.8pt)
  / Digital audio — samples & sample rate: a sound wave is stored as a long list
    of numbers, the *samples*, measured many thousands of times per second. The
    *sample rate* is how many per second. Analysis here runs at 16 kHz (16 000
    samples/s — enough to read content and pitch), while the target voice is
    synthesised at 48 kHz for full fidelity.
  / Spectrogram (STFT): the raw sample list is hard to reason about directly.
    Sliding a short window along the signal and taking a *Fourier transform* of
    each window gives the *short-time Fourier transform* — a 2-D picture of *how
    much energy sits at each frequency, moment by moment*. Columns are time
    frames; rows are frequency bins.
  / The mel scale & mel spectrogram: human hearing resolves low frequencies far
    more finely than high ones. The *mel scale* warps frequency to match,
    $"mel"(f) = 1127 · ln(1 + f/700)$. A *mel spectrogram* is an STFT whose
    frequency rows have been regrouped into a smaller set of perceptually spaced
    *mel bands* (128 of them here). Because closeness on a mel spectrogram tracks
    "sounds alike," the main training loss is simply an $L_1$ distance between the
    mel spectrograms of the real and generated audio (§7).
  / Embeddings: a network cannot consume a raw category ("speaker #3", "pitch bin
    57") directly. An *embedding* is a learned lookup table turning each discrete
    id into a short vector the network can use; because it is learned, similar ids
    drift to similar vectors. RVC embeds both the *speaker* ($g$) and the *coarse
    pitch*.
  / Convolutions: a *convolution* slides a small learnable filter along the signal
    and fires wherever its local pattern appears. Stacking them detects
    increasingly complex structure (edges → harmonics → textures) while reusing
    the same weights everywhere — efficient and natural for audio, where the same
    waveform shapes recur throughout. Both the vocoder and the discriminators are
    convolutional.
  / Training — loss & gradient descent: a *loss* is one number measuring how wrong
    the output is. *Gradient descent* nudges every weight a tiny step in the
    direction that most reduces it; repeated over many examples, the network
    learns. The step size is the *learning rate*, and how it is scheduled turns
    out to matter for stability (§9).
]

// =========================================================================
= Analysis: turning audio into content and pitch

Two pretrained, frozen models do the analysis. Neither is trained by RVC; they
are universal front-ends, run identically at training and inference time (in
this implementation they are shared ONNX sessions behind one
`FeatureExtractor`).

== ContentVec — a speaker-blind content representation

ContentVec is a *self-supervised* speech encoder (a HuBERT-family transformer)
that maps 16 kHz audio to a sequence of 768-dimensional vectors, roughly one
per 20 ms frame. Its crucial property for our purposes is that it was trained
with a speaker-disentanglement objective: two people saying the same words
produce nearly the same ContentVec sequence. That vector is therefore a good
proxy for "linguistic content, timbre removed" — exactly the input a voice
converter wants.

In code the content stream is extracted at ~50 Hz and *upsampled ×2* to the
100 Hz frame grid so it aligns one-to-one with the pitch stream:

```
let up = upsample_rows(&feats.content, 2);   // 50 Hz -> 100 Hz
let n  = up.len().min(feats.f0.len());        // align to F0 length
```

== RMVPE — a robust pitch tracker

RMVPE estimates the fundamental frequency $F_0$ — the pitch — for each frame,
and marks frames as *voiced* or *unvoiced* (silence / breath has no pitch). It
is prized for staying accurate on noisy, expressive, or quiet material, which
matters enormously for ASMR-style corpora full of soft phonation.

Pitch feeds the model in *two* forms, and understanding why is worth a moment:

#block(inset: (x: 10pt))[
  #set text(size: 10pt)
  / Coarse pitch (integer ids `[1,255]`): the continuous $F_0$ in Hz is warped
    onto a *mel* scale and quantized into 255 bins. This becomes an *embedding
    lookup* inside the prior encoder — it is treated like a discrete symbol,
    letting the network learn a smooth notion of "pitch class."
  / Fine $F_0$ (Hz, continuous): the raw value drives the decoder's harmonic
    oscillator directly, so the synthesised waveform sits at *precisely* the
    right pitch, not the nearest bin.
]

The mel-warping quantization (from `dsp::f0_to_coarse`) is:
$ "mel"(f) = 1127 · ln(1 + f/700), wide
  "id" = "round"("clamp"(1..255)) . $
Unvoiced frames ($f = 0$) map to id $1$; the map is monotonic, so higher pitch
always yields a higher (or equal) bin.

// =========================================================================
= The generative core: a conditional VAE around a vocoder

RVC's network is a descendant of *VITS*, adapted for conversion and named
`SynthesizerTrnMs768NSFsid`. It is best understood as a *conditional
variational autoencoder* (CVAE) whose decoder happens to be a full neural
vocoder. Five submodules cooperate:

#panel(caption: [The five modules of the synthesizer and the two data paths.
The posterior encoder (dashed) exists *only during training* — it is the "cheat
sheet" that teaches the prior what a good latent looks like.])[
  #set align(center)
  #set text(size: 9pt)
  #grid(
    columns: (1fr, 1fr),
    column-gutter: 22pt, row-gutter: 8pt, align: center + horizon,
    // left column: training-only posterior branch
    box(stroke: (paint: accent, dash: "dashed", thickness: 0.8pt), radius: 6pt, inset: 9pt)[
      #text(fill: accent, weight: "bold", size: 8pt)[TRAINING ONLY]
      #v(4pt)
      #node([linear spectrogram\ of *real* target audio], fill: c-feat)
      #dn
      #node([*Posterior encoder* `enc_q`\ (WaveNet) → $(mu_q, log sigma_q)$], fill: c-prior)
      #dn
      #node([sample $z = mu_q + epsilon e^(log sigma_q)$], fill: c-lat)
    ],
    // right column: prior branch (always)
    box(stroke: 0.7pt + stroke-c, radius: 6pt, inset: 9pt)[
      #text(fill: primary, weight: "bold", size: 8pt)[ALWAYS (train + infer)]
      #v(4pt)
      #node([content `[T,768]` + coarse pitch], fill: c-feat)
      #dn
      #node([*Prior encoder* `enc_p`\ (transformer) → $(mu_p, log sigma_p)$], fill: c-prior)
      #dn
      #node([prior sample $z_p$], fill: c-lat)
    ],
  )
  #v(6pt)
  #grid(columns: (auto, auto, auto), align: horizon, gutter: 6pt,
    node([$z$ (train) / $z_p$ (infer)], fill: c-lat),
    text(fill: primary, weight: "bold")[→ *Flow* → ],
    node([*NSF-HiFiGAN* `dec` + speaker $g$ → waveform], fill: c-dec),
  )
]

== The two encoders and the latent space

Both encoders emit a *distribution*, not a point: a per-frame mean $mu$ and
log-standard-deviation $log sigma$ over a 192-dimensional latent. This is the
VAE idea — the latent $z$ is stochastic, sampled with the reparameterization
trick $z = mu + epsilon dot.op e^(log sigma)$, $epsilon tilde cal(N)(0,1)$.

- The *posterior encoder* `enc_q` sees the *real answer*: the linear
  spectrogram of the ground-truth target waveform. It is a stack of 16 dilated
  WaveNet layers and produces the posterior $(mu_q, log sigma_q)$. Because it
  gets the actual audio, its latent $z$ is easy for the decoder to turn back
  into that audio. But at inference we do *not* have the target audio — that is
  the whole point — so `enc_q` cannot be used then.
- The *prior encoder* `enc_p` sees only *content + pitch* — the information we
  *will* have at inference. It is a 6-layer relative-position transformer
  producing the prior $(mu_p, log sigma_p)$.

The trick that ties them together is a *normalizing flow*.

== The normalizing flow — bridging prior and posterior

There is a gap: the posterior latent $z$ (rich, from real audio) and the prior
$z_p$ (from content only) live in different-looking distributions. The flow is
an *invertible* neural function $f_g$ (conditioned on the speaker $g$) that
warps between them, and invertibility is what lets us run it *both directions*:

#panel(caption: [The flow is one function used two ways. Training pushes the
posterior latent toward the prior; inference pulls a prior sample into the
decoder's latent space.])[
  #set align(center)
  #grid(columns: (auto, 60pt, auto), align: horizon, column-gutter: 6pt,
    node([$z$ #text(size:7.5pt)[(from real audio)]], fill: c-lat),
    [#text(fill: primary, weight: "bold", size: 9pt)[`flow.forward`] \ #text(fill:primary)[$==>$]],
    node([$z_p$ #text(size:7.5pt)[(prior space)]], fill: c-lat),
  )
  #v(4pt)
  #text(size: 8pt, fill: ink.lighten(20%))[TRAIN: compare against prior → KL loss]
  #v(9pt)
  #line(length: 60%, stroke: (paint: stroke-c, dash: "dotted"))
  #v(9pt)
  #grid(columns: (auto, 60pt, auto), align: horizon, column-gutter: 6pt,
    node([$z_p$ #text(size:7.5pt)[(sampled from prior)]], fill: c-lat),
    [#text(fill: accent, weight: "bold", size: 9pt)[`flow.reverse`] \ #text(fill:accent)[$<==$]],
    node([$z$ #text(size:7.5pt)[(→ decoder)]], fill: c-lat),
  )
  #v(4pt)
  #text(size: 8pt, fill: ink.lighten(20%))[INFER: turn content-derived prior into a decodable latent]
]

RVC's flow is a `ResidualCouplingBlock` of four *additive coupling layers*.
Each layer splits the channels in half, leaves the first half untouched, and
adds to the second half a WaveNet-predicted shift computed *from the first
half*:
$ x_1 arrow.l x_1 + m(x_0, g) wide ("forward") , wide
  x_1 arrow.l x_1 - m(x_0, g) wide ("reverse") . $
Because the shift depends only on the *untouched* half, the layer is trivially
invertible — reverse just subtracts the very same predicted mean. A channel
*flip* between layers ensures every dimension eventually gets transformed. The
layers are `mean_only`, so the log-determinant is zero and there is nothing
extra to track:

```
pub fn forward(&self, x, g)  { cat(x0, x1 + mean(x0, g)) }   // add
pub fn reverse(&self, x, g)  { cat(x0, x1 - mean(x0, g)) }   // subtract
```

At inference the sampling temperature is deliberately cooled —
$z_p = mu_p + sigma_p dot.op epsilon dot.op 0.666$ — trading a little
diversity for cleaner, more stable output.

// =========================================================================
= The vocoder: NSF-HiFiGAN

The decoder `dec` turns the 192-channel latent (at 100 Hz) into a waveform at
48 kHz. That is a 480× increase in sample rate — the *hop length*, the product
of the upsample factors $12 · 10 · 2 · 2 = 480$. It is a *HiFi-GAN* generator
with a *Neural Source-Filter* (NSF) twist.

== Why a source-filter model

Classic HiFi-GAN upsamples the latent through transposed convolutions and lets
the network invent all the fine structure. For *singing and expressive speech*
that struggles: the network has to hallucinate an exact pitch. NSF fixes this
by *handing the network the pitch as an explicit excitation signal*. The
`SourceModule` builds a sine wave whose instantaneous frequency tracks the fine
$F_0$ contour — a clean harmonic *source* — and injects it into every upsampling
stage. The convolutional stack then acts as a learned *filter* shaping that
source into a natural voice. Pitch accuracy comes for free because it is built
into the source.

#panel(caption: [Inside the NSF-HiFiGAN decoder. The sine source (bottom) is
generated deterministically from $F_0$ and mixed in at each upsampling stage;
the speaker embedding $g$ conditions the head.])[
  #set align(center)
  #set text(size: 8.7pt)
  #grid(columns: (auto, 16pt, auto), align: horizon, gutter: 4pt,
    node([latent $z$ `[192,T]`], fill: c-lat), ar, node([`conv_pre` + speaker cond `+ cond(g)`], fill: c-dec),
  )
  #dn
  #node([× 4 upsample stages : `LeakyReLU → ConvTranspose1d (×12,×10,×2,×2)` \ then `+ noise_conv(source)` , then average of 3 `ResBlock1` (kernels 3/7/11, dilations 1·3·5)], fill: c-dec, w: 92%)
  #dn
  #grid(columns: (auto, 16pt, auto), align: horizon, gutter: 4pt,
    node([`LeakyReLU → conv_post → tanh`], fill: c-dec), ar, node([waveform `[1, T·480]`], fill: c-dec),
  )
  #v(9pt)
  #line(length: 70%, stroke: (paint: accent, dash: "dashed"))
  #v(7pt)
  #grid(columns: (auto, 16pt, auto, 16pt, auto), align: horizon, gutter: 4pt,
    node([$F_0$ (Hz)], fill: c-feat),
    ar,
    node([`SineGen`: phase-accumulate sine at $F_0$ + voiced/unvoiced mask + noise], fill: c-feat),
    ar,
    node([harmonic source `[1, T·480]` → strided into each stage], fill: c-feat),
  )
]

== The deterministic sine source

The sine generator is *not learned* (RVC runs it under `no_grad`; here it is
plain arithmetic). For each frame it computes a phase increment
$f_0 slash "sr"$, accumulates phase across the whole clip, upsamples the
cumulative phase (linear, `align_corners`) and takes its sine — carefully
detecting phase wraps so the oscillator stays continuous:

```
phase += rad_up[j] + shift;                 // shift = -1 on a detected wrap
sine[j] = (phase * 2.0 * PI).sin() * SINE_AMP;
```

Voiced frames get the sine; unvoiced frames get low-level Gaussian noise
(breath). The only learnable part of the source is a single `l_linear`
projection feeding a `tanh`. This is precisely why *breathy, unvoiced ASMR
content survives*: unvoiced frames are handled explicitly by the noise branch,
not discarded.

== A note on dimensions (v2, 48 kHz)

#align(center, block(width: 92%)[
  #set text(size: 9pt)
  #table(
    columns: (auto, auto, 1fr),
    align: (left, center, left),
    stroke: 0.4pt + stroke-c,
    inset: 6pt,
    fill: (_, row) => if row == 0 { c-loss } else { white },
    [*quantity*], [*value*], [*meaning*],
    [content dim], [768], [ContentVec output width, per frame],
    [latent `inter_channels`], [192], [shared width of prior / posterior / flow],
    [`hidden_channels`], [192], [transformer & WaveNet width],
    [`spec_channels`], [1025], [linear-spectrogram bins ($n_"fft"/2+1$) into `enc_q`],
    [`gin_channels`], [256], [speaker-embedding width $g$],
    [transformer], [6 layers, 2 heads], [prior encoder `enc_p`],
    [flow], [4 coupling layers], [additive, `mean_only`, with flips],
    [upsample rates], [12·10·2·2], [product = hop = 480 samples/frame],
    [frame rate], [100 Hz], [both content and pitch land on this grid],
  )
])

// =========================================================================
= Training I: the reconstruction objectives

Training is *fine-tuning*: warm-start every module from the public pretrained
base (`f0G48k.pth` / `f0D48k.pth`), then adapt to one target voice on a small
corpus. The module layout is deliberately kept weight-compatible with the
reference PyTorch `state_dict` so this warm-start is possible — essential when
the corpus is only ~1 hour long. Other bases exist for the same architecture; §1
lists them and says why none of them is wired up.

Each step draws a batch of short random windows from the corpus, runs the
*training forward* (posterior → flow → decode a random 0.36 s segment), and
computes a compound loss. Three of its terms are *reconstruction* terms; the
other two (next section) are *adversarial*.

/ Mel-spectrogram L1 ($times 45$): the perceptual backbone. Convert both the
  real segment and the generated one to a 128-band mel spectrogram and take
  the mean absolute difference. This is what actually makes the output *sound*
  like the target; its large weight (45) reflects that.
  $ L_"mel" = || "mel"(y) - "mel"(hat(y)) ||_1 . $

/ KL divergence ($times 1$): the CVAE glue. It pushes the flow-transformed
  posterior $z_p$ to look like a sample from the prior $(mu_p, log sigma_p)$,
  so that at inference — when only the prior is available — the decoder still
  receives a latent it knows how to render. RVC uses VITS's Monte-Carlo form:
  $ L_"kl" = log sigma_p - log sigma_q - 1/2
    + 1/2 (z_p - mu_p)^2 e^(-2 log sigma_p) . $

/ Feature matching ($times 2$): defined via the discriminator, covered below.

A subtle but important detail: the posterior's input spectrogram is
`.detach()`-ed, and in feature matching the *real* features are detached — they
are fixed targets, and gradients must not flow into them.

// =========================================================================
= Training II: the generative adversarial network

Mel-L1 alone produces *oversmoothed, buzzy* audio: averaging many plausible
waveforms that share a mel spectrogram gives a muddy one. The fix is a *GAN*.
We introduce a second network — the *discriminator* — whose only job is to tell
*real* target audio from the *generated* audio. The generator (our whole
synthesizer) is then additionally trained to *fool* it. In the limit, "audio
the discriminator can't distinguish from real" is exactly "audio that has the
crisp fine structure of real speech."

== A tale of two networks

#panel(caption: [The adversarial game. $G$ wants $D$ to score its output as
real; $D$ wants to score real as 1 and fake as 0. They are trained in
alternation, each with its own AdamW optimiser.])[
  #set align(center)
  #grid(columns: (auto, 46pt, auto), align: horizon, column-gutter: 8pt, row-gutter: 6pt,
    node([*Generator $G$*\ (synthesizer)\ produces $hat(y)$], fill: c-dec, w: 120pt),
    align(center)[
      #text(fill: primary, weight: "bold")[$hat(y)$ →] \
      #v(10pt)
      #text(fill: accent, weight: "bold")[← gradient\ "make it real"]
    ],
    node([*Discriminator $D$*\ real $y$ → 1 ?\ fake $hat(y)$ → 0 ?], fill: c-disc, w: 120pt),
  )
]

== Why *many* discriminators — multi-period + multi-scale

A single discriminator looking at the raw 1-D waveform tends to miss *periodic*
artefacts — the buzz and roughness that live in the pitch harmonics. RVC (from
HiFi-GAN) therefore uses a *bank* of discriminators, a
`MultiPeriodDiscriminatorV2`:

- One *scale* discriminator (`DiscriminatorS`) reads the waveform as a plain
  1-D signal — good at broad envelope and noise texture.
- Eight *period* discriminators (`DiscriminatorP`), one for each prime period
  in ${2,3,5,7,11,17,23,37}$. Each *folds* the 1-D waveform into a 2-D grid
  with that period as the row stride, then runs 2-D convolutions. Folding on a
  period exposes structure that repeats at that spacing — precisely where
  pitch-related artefacts hide. Using *coprime* periods means the discriminators
  cover a wide range of periodicities without redundantly overlapping.

#panel(caption: [How a period discriminator "sees" the signal: a period-3 fold
turns a 1-D strip into a 2-D image whose columns line up periodic samples, then
convolves it in 2-D. Eight coprime periods cover many spacings.])[
  #set align(center)
  #set text(size: 8.5pt)
  1-D waveform (reshaped by period $p=3$):
  #v(5pt)
  #table(columns: 12, inset: 4pt, align: center, stroke: 0.4pt + stroke-c,
    fill: c-disc.lighten(30%),
    [$x_0$],[$x_1$],[$x_2$],[$x_3$],[$x_4$],[$x_5$],[$x_6$],[$x_7$],[$x_8$],[$x_9$],[$x_10$],[$x_11$],
  )
  #v(6pt)
  #text(fill: primary, weight: "bold", size: 11pt)[↓ fold every 3 samples into rows]
  #v(6pt)
  #grid(columns: 3, column-gutter: 0pt,
    ..([$x_0$],[$x_1$],[$x_2$],[$x_3$],[$x_4$],[$x_5$],[$x_6$],[$x_7$],[$x_8$],[$x_9$],[$x_10$],[$x_11$]).map(c =>
      box(fill: c-disc, stroke: 0.4pt + stroke-c, inset: 5pt, width: 34pt)[#c])
  )
  #v(4pt)
  #text(size: 8pt, fill: ink.lighten(20%))[columns now align samples 3 apart → 2-D convs detect period-3 structure]
]

Both discriminator families also expose their *intermediate feature maps*,
which powers the feature-matching loss.

== The four adversarial/perceptual terms, precisely

RVC uses the *least-squares* GAN (LSGAN) formulation — it is more stable than
the log-loss original. Writing $D_k$ for the $k$-th discriminator's score:

#block(inset: (x: 8pt))[
  #set text(size: 10pt)
  / Discriminator loss (push real→1, fake→0):
    $ L(D) = sum_k EE[(D_k (y) - 1)^2] + EE[D_k (hat(y))^2] . $
  / Generator adversarial loss (push fake→1):
    $ L_"adv"(G) = sum_k EE[(D_k (hat(y)) - 1)^2] . $
  / Feature matching ($times 2$): match $G$'s audio to the target *inside* the
    discriminator, layer by layer — a perceptual $L_1$ that stabilises training:
    $ L_"fm" = sum_k sum_l norm(D_k^((l)) (y) - D_k^((l)) (hat(y)))_1 . $
]

The complete generator objective is the weighted sum
$ L_G = L_"adv"(G) + 2 L_"fm" + 45 L_"mel" + L_"kl" . $

== The alternating step

One training step touches both networks. For the *discriminator* term the
generated audio is *detached*, so $D$'s gradient never leaks into $G$. Both the
$D$ and the $G$ gradients are computed from the *same start-of-step weights*, and
only *then* are the two optimisers stepped — so within a step $G$ is pushed
against the discriminator as it stood at the start, not a half-updated one. (With
gradient accumulation, §9, these gradients are summed over several micro-batches
before the step.)

#panel(caption: [One training iteration. Both losses are measured against the
same start-of-step weights; the two optimisers then step together.])[
  #set align(center)
  #set text(size: 8.8pt)
  #node([sample batch of windows → forward: posterior, flow, decode a random 0.36 s segment $hat(y)$ ; take matching real segment $y$], fill: c-prior, w: 92%)
  #dn
  #node([*① Discriminator loss* — score $y$ and `detach`$(hat(y))$ across scale + 8 periods; \ $L(D) = sum_k$ LSGAN$(D_k)$ → accumulate $nabla D$], fill: c-disc, w: 92%)
  #dn
  #node([*② Generator loss* (against the start-of-step $D$) — $L_G = L_"adv" + 2 L_"fm" + 45 L_"mel" + L_"kl"$ → accumulate $nabla G$], fill: c-dec, w: 92%)
  #dn
  #node([*③ Step* — `opt_d.step` then `opt_g.step`; update the generator's EMA snapshot], fill: c-lat, w: 92%)
  #dn
  #node([log `g`, `d`, `mel`, `lr` to the live dashboard; repeat], fill: c-loss, w: 92%)
]

Both optimisers are AdamW ($beta_1{=}0.8$, $beta_2{=}0.99$), matching the
reference recipe, with a base learning rate of $10^(-4)$ that is *decayed over the
run* (§9). What ships is not the raw final weights but a smoothed *exponential
moving average* of them — also §9.

== Reading the loss curves

Because $G$ and $D$ are *adversaries*, their losses do *not* both fall to zero —
they seek an equilibrium. Practical intuition:

#block(inset: (x: 10pt))[
  #set text(size: 9.8pt)
  - *`mel_loss` is the one to watch.* It is the honest reconstruction signal and
    *should trend down*. Some *shaking around a plateau* in the back half is
    normal — it is the adversarial equilibrium, and the stabilisation techniques
    of §9 (LR decay, weight EMA) exist to tame exactly that. But if it *never*
    descends and the output is silent, suspect a *data* problem (§10), not a
    hyperparameter one.
  - *`d_loss` near a small positive constant* (not collapsing to 0) means the
    discriminator is appropriately challenged. A `d_loss` crashing to 0 means
    $D$ has won and $G$ gets no useful gradient.
  - *`g_loss` is dominated by the $45 times$ mel term*, so it largely tracks
    `mel_loss` plus adversarial jitter.
]

// =========================================================================
= Training III: making training stable and reliable

The recipe so far — LSGAN, feature matching, mel-L1, KL — is faithful to the
reference. But an adversarial game trained on a *small* corpus and a *small* GPU
is a jittery thing: with a tiny batch each step's gradient is noisy, and once $G$
and $D$ settle into their tug-of-war the mel loss stops descending and instead
*oscillates* around a plateau. Left alone, whatever noisy point the final step
lands on is what gets saved. Five techniques — all optional `rvc train` flags, two
on by default — make the outcome steadier and the saved model better. None of them
touches the *objective*; they shape the *dynamics*.

== A decaying learning-rate schedule

A constant learning rate keeps taking full-size steps forever, so near the optimum
the weights *bounce around* it instead of settling in. The remedy is to shrink the
step as training proceeds. Here the rate decays *exponentially over the whole run*,
$ "lr"(t) = "lr"_0 · rho^(t slash T) , $
from $"lr"_0$ (the `--lr` base) at the first step down to $"lr"_0 · rho$ at the
last, where $rho =$ `--lr-final` and $T$ is the total number of steps. Tying the
schedule to the *fraction of the run completed* rather than to an epoch count
makes it behave identically whether you train for two epochs or forty.

== Averaging the weights: the EMA snapshot

Even with a decaying rate the weights still wobble step to step. A cheap, standard
trick from GAN vocoders is to keep a second, *shadow* copy of the generator that
trails the live weights — an *exponential moving average*:
$ theta_"ema" arrow.l d · theta_"ema" + (1 - d) · theta_"live" quad "(every step)." $
Averaging over the recent past cancels the oscillation, so the EMA weights are
*cleaner and less buzzy* than any single step. Two points worth stressing: the EMA
is a *read-only snapshot* — it never feeds back into the optimiser, so it does
*not* slow learning — and it is what gets *saved as the model* (the raw live
weights are written beside it, so nothing is lost, and on a very short run that
never plateaus you can prefer them). The averaging window is set as a fraction of
the run (`--ema-frac`), so, like the LR schedule, it needs no re-tuning when the
run length changes.

== Bigger effective batches for free: gradient accumulation

A 6 GB GPU forces a tiny batch, and tiny batches make noisy gradients. Gradient
*accumulation* recovers a larger *effective* batch without more memory: run
`--grad-accum N` micro-batches, *sum* their gradients, and step once. Peak memory
stays that of a single micro-batch — each one's computation graph is freed after
its backward pass — while the summed gradient is as steady as a batch $N$ times
larger, at $N times$ the compute.

== Keeping the discriminator in check

If $D$ learns much faster than $G$ it starts winning outright: its gradients stop
being informative and instead inject *buzzy, high-frequency artefacts* into the
generator. Two knobs rebalance the game — `--d-lr-ratio` scales $D$'s learning
rate below $G$'s, and `--d-interval` updates $D$ only every few steps — both
handing $G$ a little room to catch up.

== Weighting the data by cleanliness

Finally, corpus clips are not equally clean. `--snr-weight` biases sampling toward
the clips with a lower noise floor, weighting each by $"SNR"^alpha$. Crucially the
score is a *signal-to-noise ratio* — a clip's loud-percentile level over its
noise-floor level — and *not loudness*, so a soft, breathy ASMR take still scores
high and is never penalised for being quiet. This steers away from hiss while
keeping the very content the corpus exists to capture. (Contrast §10, which removes
between-sentence *silence*; this weights whole clips by *noise*.)

// =========================================================================
= Where training goes wrong: data, not just model

The most common failure on real-world corpora is a generator that *collapses to
silence* — `mel_loss` stuck and shaking, output pure silence, even though the
pretrained base works. The architecture is fine; the *training distribution* is
poisoned.

The trainer draws *random short windows uniformly across each file*. If the raw
recordings are full of between-sentence dead air, then most randomly sampled
0.36 s segments *are silence*, and the fastest way to minimise mel-L1 on a
silence-heavy batch is to *emit silence*. The generator obliges, globally.

The remedy is a *sentence-safe slicer* run as a preprocessing pass
(`rvc preprocess`): it uses short-time energy with *hysteresis* to cut only
*genuine, sustained* dead air between sentences, while:

#block(inset: (x: 10pt))[
  #set text(size: 9.8pt)
  - *never splitting a complete sentence* — a gap must be both below the energy
    floor *and* longer than a minimum duration before it counts as a cut; and
  - *preserving soft, breathy content* — energy is used only to *locate* long
    silent gaps, never to gate quiet-but-present sound, and cut boundaries are
    edge-padded so onsets and breathy tails are kept.
]

The result is a corpus of clean per-sentence clips, so uniform window sampling
now lands on real phonation. Training itself is unchanged — it simply consumes
a healthier distribution.

// =========================================================================
= Inference recap: the reverse path

At inference the posterior encoder and the whole discriminator bank are gone.
The path collapses to: analyse source → prior encoder → *sample* $z_p$ →
*reverse* the flow → decode with the target speaker embedding. Content and
pitch came from the *source*; timbre came from the *trained decoder + speaker
embedding*. That asymmetry — universal analysis in, speaker-specific synthesis
out — is the whole idea of voice conversion, and every architectural choice
above exists to keep those two halves cleanly separated.

#v(0.6cm)
#line(length: 100%, stroke: 0.5pt + stroke-c)
#v(4pt)
#align(center, text(size: 8.5pt, fill: ink.lighten(30%))[
  Concepts anchored to the `rvc` implementation: `burn-rvc` (network),
  `rvc-core` (feature extraction + DSP + backends), `rvc-train` (adversarial
  training loop). RVC v2, 48 kHz, weight-compatible with the reference
  `SynthesizerTrnMs768NSFsid` / `MultiPeriodDiscriminatorV2`
  (RVC-Project, tag `2.2.231006`).
])
