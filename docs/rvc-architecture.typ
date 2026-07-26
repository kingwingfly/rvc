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
the corpus is only ~1 hour long.

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

One training step updates the two networks in turn. The generated audio is
*detached* for the discriminator step, so $D$'s gradient never leaks into $G$;
then $G$ is updated through the (now fixed-weights) discriminators.

#panel(caption: [One training iteration. The order matters: $D$ first on
detached audio, then $G$ against the updated $D$.])[
  #set align(center)
  #set text(size: 8.8pt)
  #node([sample batch of windows → forward: posterior, flow, decode a random 0.36 s segment $hat(y)$ ; take matching real segment $y$], fill: c-prior, w: 92%)
  #dn
  #node([*① Discriminator step* — score $y$ and `detach`$(hat(y))$ across scale + 8 periods; \ $L(D) = sum_k$ LSGAN$(D_k)$; `opt_d.step`], fill: c-disc, w: 92%)
  #dn
  #node([*② Generator step* — $L_G = L_"adv" + 2 L_"fm" + 45 L_"mel" + L_"kl"$; `opt_g.step`], fill: c-dec, w: 92%)
  #dn
  #node([log `g`, `d`, `mel` to the live dashboard; repeat], fill: c-loss, w: 92%)
]

Both optimisers are AdamW ($beta_1{=}0.8$, $beta_2{=}0.99$, lr $10^(-4)$),
matching the reference recipe.

== Reading the loss curves

Because $G$ and $D$ are *adversaries*, their losses do *not* both fall to zero —
they seek an equilibrium. Practical intuition:

#block(inset: (x: 10pt))[
  #set text(size: 9.8pt)
  - *`mel_loss` is the one to watch.* It is the honest reconstruction signal and
    *should trend down*. If it shakes around a plateau and never descends, the
    generator is not learning the voice — most often a *data* problem (see
    below), not a hyperparameter one.
  - *`d_loss` near a small positive constant* (not collapsing to 0) means the
    discriminator is appropriately challenged. A `d_loss` crashing to 0 means
    $D$ has won and $G$ gets no useful gradient.
  - *`g_loss` is dominated by the $45 times$ mel term*, so it largely tracks
    `mel_loss` plus adversarial jitter.
]

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
  `SynthesizerTrnMs768NSFsid` / `MultiPeriodDiscriminatorV2`.
])
