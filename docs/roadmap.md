# Roadmap

What is not built yet, why it is wanted, and what it would actually cost. This
is the repo's first roadmap — before it, "not started" lived only as a cell in
[`README.md`](../README.md)'s engine table, which could say that `translate` was
missing but not what building it would involve.

**This is not a schedule and not a promise.** Nothing here is in progress unless
it says so. Each entry is written so that whoever picks it up starts from the
constraints rather than rediscovering them, which means every one names the thing
that actually blocks it — usually not the part that looks hard.

The counterpart page is [`CLAUDE.md`](../CLAUDE.md), which records decisions
already taken and the traps behind them. A choice that has been *made* belongs
there; a choice still open belongs here.

- [`translate` — the missing pipeline stage](#translate--the-missing-pipeline-stage)
- [Seed-VC to ONNX](#seed-vc-to-onnx)
- [`seedvc train`](#seedvc-train)
- [A native audio-device backend](#a-native-audio-device-backend)
- [g2pw — Mandarin's syntactic polyphones](#g2pw--mandarins-syntactic-polyphones)
- [ContentVec and RMVPE, both ways](#contentvec-and-rmvpe-both-ways)
- [The RVC 2.3 mel floor](#the-rvc-23-mel-floor)

## `translate` — the missing pipeline stage

The pipeline this toolkit is named for is

```
voice -(stt)-> text -(translate)-> text -(tts)-> voice -(rvc)-> voice
```

and **`translate` is the one stage that has never been built.** Everything
around it works. A user who wants it today pipes to any external tool, which is
what `README.md` says, and the shape of the toolkit makes that genuinely fine —
a text-to-text filter is the easiest thing in the world to substitute.

*What it needs.* A `translate-core` + `translate-cli` pair in the shape every
other engine has: the bare invocation is the stdin→stdout filter, one line in and
one line out, plus `convert`, `download` and `completions`. The engine work is
choosing a model that can be ported to Burn and loads its Hugging Face weights
directly — the no-Python rule applies here exactly as it does everywhere else, so
wrapping an existing translation library that shells out is not an option.

*What it costs.* The plumbing is nearly free: `cli-kit` already owns
`--backend`/`--device`/`--cache-dir`, `hub-kit` owns the download and the cache,
and `voice-cli` nests a new engine by hosting its clap type. The model is the
whole job, and the size of it depends entirely on which one — a small
sequence-to-sequence translator is a much smaller port than anything else in this
repo, and an LLM-class one is not a port at all but a different kind of project.

*The one design question worth settling first* is whether `translate` streams per
line, which is the only shape consistent with the rest of the toolkit, or wants
document context to translate well. If it wants context, it is the first engine
whose bare invocation cannot be a pure line filter, and that deserves a decision
rather than a default.

## Seed-VC to ONNX

`seedvc` is the only engine with **no** ONNX path, and
[`CLAUDE.md`](../CLAUDE.md) is explicit that this is an asymmetry rather than an
omission: Burn imports ONNX graphs and cannot emit one, so a model reaches ONNX
Runtime only through [`export/`](../export/README.md). `seedvc-core`'s loader
therefore refuses `--backend onnx` before it reads a file, with a message saying
*why* — a user told "not compiled in" goes looking for a feature flag that cannot
exist.

*Why it cannot be borrowed.* There is no public ONNX Seed-VC to point at. The
Hugging Face model index carries 27 Seed-VC repositories and every one of them is
PyTorch; `onnx-community` has never touched the model. So the export has to be
written here or not exist.

*What it needs.* Another clean-room torch mirror beside
[`rvc_infer.py`](../export/rvc_infer.py) and
[`gptsovits_infer.py`](../export/gptsovits_infer.py) — mirroring the Burn layout,
never importing upstream — plus the exporter that drives it. That is roughly
1300 lines against seven Burn modules:

| module | lines |
|---|---|
| `dit.rs` | 841 |
| `bigvgan.rs` | 735 |
| `campplus.rs` | 724 |
| `fbank.rs` | 423 |
| `content.rs` | 379 |
| `length_regulator.rs` | 225 |
| `wavenet.rs` | 187 |

and six graphs: content, style, mel, regulator, dit-velocity and BigVGAN.

*The trap to know before starting.* **`Spectral` and `Fbank` run on Burn inside
`BurnModel` today**, so they are part of the model rather than host-side
preprocessing, and they must be *traced into the graphs* rather than assumed to
be somebody else's problem. Getting that wrong produces an export that runs and
is fed the wrong front end — and
[`CLAUDE.md`](../CLAUDE.md) records that the three front ends here have matching
frame counts and band counts, so substituting one for another computes something
else without failing. CAMPPlus's mean subtraction is upstream's *call site* and
not its encoder, which is exactly the kind of step a naive trace drops.

*What it buys.* Deployment where ORT is the only runtime available, which is the
reason `export/` gains scope rather than losing it. It buys no training path —
ONNX Runtime cannot train — so this is orthogonal to the entry below.

## `seedvc train`

**Seed-VC is fine-tunable.** The engine README's "there is nothing to train" is
about *zero-shot inference* and stays true — a reference clip really is the whole
speaker specification at inference time — but upstream ships a `train.py`, and a
fine-tune on a target domain is a real quality lever. `seedvc-core` having no
`-train` sibling is currently presented as the engine's defining property; if
this is built, that framing has to be revised rather than quietly contradicted.

*What upstream does.* `build_model` (`commons.py`) constructs exactly two things,
`cfm` and `length_regulator`, and those are the only two optimised. It
warm-starts from the released checkpoint, eats a **bare recursive directory of
1–30 s audio with no transcripts and no speaker labels** — which is a far lighter
corpus requirement than either `rvc train` or `tts train` — and minimises a
single L1 on the flow-matching velocity with the prompt region masked out. AdamW
at 1e-5, `ExponentialLR(0.999996)` stepped per batch, gradient clip 10.

This also **confirms** `CLAUDE.md`'s reading of `net.style_encoder.*` as a
training-time fossil: even upstream's *training* entry point never instantiates
it.

*What it needs, in order.*

1. **A loss.** `flow.rs` has `Sampler` — upstream's `CFM` reduced to the Euler
   solver and its guidance scale, `steps: 30` and `guidance: 0.7` — and
   `sample()` is inference only. There is no loss method at all; one has to be
   written against upstream's `flow_matching.py`.
2. **Masking back in `Dit`.** `Dit::forward(x, prompt_x, t, style, cond)` dropped
   upstream's `x_lens` and `prompt_lens` on the grounds that batch-1 inference
   makes the padding mask trivial, and the code says so. Training at batch > 1
   makes it non-trivial again, so that has to come back before anything else is
   worth measuring — a wrong mask trains happily on padding.
3. **The `S_alt` substitute**, which is the actual blocker.

*The blocker.* Upstream builds its timbre-shifted content `S_alt` — the
perturbed input that stops the model from simply copying the source — with
OpenVoice's `ToneColorConverter` plus a 22 MB `se_db.pt`. **Neither exists in
this workspace, and pulling them in means another model, another licence and
another download for a step that is only ever used during training.** The
decision taken is to substitute a NANSY-style DSP perturbation instead — formant
shift, random resampling and a parametric EQ applied before the content encoder —
and to **document that as a deliberate divergence rather than as a port**. It is
not upstream's recipe and results should not be compared to upstream's numbers as
though it were.

*What it costs.* Real work, but the scaffolding is there: `train-kit` already
owns checkpoints, EMA, gradient accumulation and the dashboard, and is generic
over the module trained. Note that this is a **single L1 loss on one model with
one optimizer**, so unlike `rvc-train` the loss number means something on its own
— the same property that makes `tts-train`'s `s1` easy to reason about.

## A native audio-device backend

[`realtime.md`](realtime.md) documents the thing this would replace, and the
honest summary is that **it already works without us**. On Linux,
`pactl load-module module-pipe-source` turns a named pipe into a real capture
device, so `rvc > /tmp/voice.pipe` *is* a virtual microphone that OBS, Discord
and a browser all see — with zero code in this repo. macOS and Windows reach the
same place through BlackHole and VB-Cable.

*What a native backend would buy.* One fewer process at each end of the pipe, and
the latency that process costs — the ffmpeg hop is a buffer, though a small one
next to the 0.5 s block `rvc` buffers anyway. It would also let an engine name a device
directly instead of the user assembling one, and it could avoid the FIFO stall
`realtime.md` describes, where nothing drains the pipe until an application starts
capturing.

*What it costs, and why that is the whole argument.* A platform-specific audio
dependency — `cpal`, or PipeWire and PulseAudio directly — inside engines that
are currently pure Unix filters with no I/O beyond stdin and stdout. That sits
badly against the design: the `futures::Stream`-in → `futures::Stream`-out shape
is what makes every engine composable, and audio APIs are the least portable
dependency available. It would also be the first thing in the toolkit that cannot
be built and tested identically on every platform.

*So the shape it should take, if it is taken at all*, is an **opt-in feature on
the CLI crates rather than a change to any `-core`**, leaving the filter path
exactly as it is and adding a device path beside it. An engine that grows a
microphone must not thereby lose its pipe.

## g2pw — Mandarin's syntactic polyphones

`text-kit`'s Mandarin front end is at the ceiling a lexical approach can reach.
Embedding `pypinyin`'s own 47,111-entry phrase dictionary and doing longest-match
lookup within each jieba token already beats upstream — upstream looks up the
whole token only, so it loses 银行 the moment jieba hands it 银行卡.

**The remaining errors are syntactic, not lexical, and no phrase table can reach
them.** `chinese.rs` states the case: 还 stands alone as a word in both 他还没来
(*hai*) and 把钱还他 (*huan*), so there is no phrase to look up and both come out
*hai*. Only syntax separates them.

*What it needs.* g2pw, which is a BERT polyphone disambiguator — so a Burn port
of a BERT encoder plus its polyphone classification head and the character-to-
candidate table it selects over. `tts-core` already carries a BERT for prosody,
which is a useful precedent for the shape but not the same model.

*What it costs.* A model download and a forward pass on a path that is currently
pure Rust with no tensors at all. That is the real price: `text-kit` is today
"no model, no backend — so it is fully testable without weights", and g2pw ends
that. Keeping the dictionary path as the fallback when the model is absent is
what would preserve the property, and it should be designed in from the start
rather than retrofitted.

*How wrong it is today* is worth measuring before doing any of this. The failure
is confined to characters whose reading is grammatically determined, which is a
short list; a corpus count would say whether this is a rare blemish or a
persistent one, and that number should decide the priority.

## ContentVec and RMVPE, both ways

`rvc-core`'s feature extraction — ContentVec and RMVPE — runs on ONNX Runtime
only, which is why `rvc` cannot be built without ORT the way a Burn-only `stt`
can. Burn ports of both are being added, giving feature extraction the same
run-time backend choice the generator already has and demoting ORT from a hard
requirement to an option.

**The mirror-image gap is what belongs on this list.** Once those models can be
*fine-tuned* here, a fine-tuned one cannot reach ONNX Runtime, because Burn
imports ONNX without emitting it. `export/` is where that changes, and the work
is the same shape as every other entry there: a clean-room torch mirror of the
Burn layout, plus the exporter.

*Why this is not urgent.* Neither model is fine-tuned today — both are frozen
inference assets — so the gap is theoretical until something trains them. It is
listed because the asymmetry is easy to forget, and because the moment a `train`
subcommand touches either one, deployment under ORT silently stops being possible
for the result.

*Worth doing alongside.* ContentVec is a HuBERT variant, so `burn-gptsovits`'s
`hubert.rs` already covers its architecture. Sharing it means lifting that module
into a `burn-hubert` of its own — two engines would then depend on it, and **no
engine may depend on another engine**, so the shared code has to move to a
neutral crate first. That is a naming and layering job, not a modelling one.

## The RVC 2.3 mel floor

Upstream RVC 2.3 dropped its mel front-end's `clip_val` from 1e-5 to 2e-6 —
about 14 dB more range under the old floor — and the linear-spectrogram epsilon
from 1e-6 to 2e-7. `CLAUDE.md`'s audit of that release deliberately did not adopt
it, and flags it as **worth trying for soft, breathy material as a measured
change with both engines re-baselined, not as a fix.**

*Why it is not a one-line change.* `burn_vits::spectral` is **shared with
`tts-train`**, where GPT-SoVITS upstream still uses 1e-5. Lowering it silently
would move two engines' loss scales at once and invalidate every recorded number
in this repo — every mel figure in `docs/training.md`, every baseline a future
change would be compared against. It touches no weight and no inference path;
it is purely a training-objective knob, which is what makes it cheap to try and
expensive to get wrong.

*What the experiment looks like.* Train the same corpus twice at each floor, on
both engines, and compare on the material this is for — soft and breathy content,
where the extra range under the floor is the whole point. Re-baseline the
recorded numbers in `docs/training.md` if it is adopted. **Report both engines
even if only one improves**, because the shared front end means one engine's gain
is the other engine's regression risk.

*Note the linear side is already finer than either release.* This repo uses 1e-9
where 2.2 used 1e-6 and 2.3 uses 2e-7, so only the mel floor is actually in
question.
