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
- [`seedvc train`](#seedvc-train)
- [A native audio-device backend](#a-native-audio-device-backend)
- [g2pw — Mandarin's syntactic polyphones](#g2pw--mandarins-syntactic-polyphones)
- [Exporting a fine-tuned ContentVec or RMVPE](#exporting-a-fine-tuned-contentvec-or-rmvpe)
- [The duplicated `session()` builder](#the-duplicated-session-builder)
- [The RVC 2.3 mel floor](#the-rvc-23-mel-floor)
- [Separation attenuates a bed, it does not remove it](#separation-attenuates-a-bed-it-does-not-remove-it)
- [`preprocess diarize` without a reference](#preprocess-diarize-without-a-reference)

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

## `seedvc train`

**Seed-VC is fine-tunable.** The engine README's "there is nothing to train" is
about *zero-shot inference* and stays true — a reference clip really is the whole
speaker specification at inference time — but upstream ships a `train.py`, and a
fine-tune on a target domain is a real quality lever. `seedvc-core` having no
`-train` sibling is currently presented as the engine's defining property — the
`onnx` arm the exporter batch added does not touch that framing, since ONNX
Runtime cannot train either; if this is built, that framing has to be revised
rather than quietly contradicted.

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

## Exporting a fine-tuned ContentVec or RMVPE

**A fine-tuned ContentVec or RMVPE could not reach ONNX Runtime**, because Burn
imports ONNX without emitting it. `export/` is where that changes, and the work
is the same shape as every other entry there: a clean-room torch mirror of the
Burn layout, plus the exporter.

Only the *export* direction is open. Running these two on either runtime is
done — `burn-rmvpe` and `burn-rvc`'s `ContentVec` (over the extracted
`burn-hubert`) give feature extraction the same run-time backend choice the
generator has, selected by `--content-vec-backend` and `--rmvpe-backend`. What
that decision cost and which traps it avoided is [`CLAUDE.md`](../CLAUDE.md)'s
business, not this page's.

*Why this is not urgent.* Neither model is fine-tuned today — both are frozen
inference assets — so the gap is theoretical until something trains them. It is
listed because the asymmetry is easy to forget, and because the moment a `train`
subcommand touches either one, deployment under ORT silently stops being possible
for the result.

## The duplicated `session()` builder

**The six-line CUDA-then-CPU session builder now exists in four engines.**
`session()` — `Session::builder()` with CUDA first and CPU as the fallback — is
the shape every ONNX path here converges on, and `seedvc-core`'s copy is the
fourth: `stt-core`, `rvc-core` and `tts-core` each carry their own, and they are
not shared because no engine may depend on another engine.

*What it costs.* Four definitions of one ordering decision. The execution-provider
list already had to be right in three places; it now has to be right in four, and
a change — a new provider, a preference order that stops being correct — is a
four-file diff that any one of the four can silently miss.

*What the fix looks like.* Lifting it into `cli-kit`, which already owns
`Backend` and is the crate the docs list as safe for anything to depend on,
turning four copies into one call. Four is the count at which the extraction pays
for itself: at three, the shared home is an indirection for its own sake; at
four, a provider change is already a four-file diff.

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
## Separation attenuates a bed, it does not remove it

`preprocess separate` runs MDX23C, which takes **10–12 dB** of a continuous
music bed out of the vocals stem on the material this toolkit is actually for —
a person talking over somebody else's music — and about **4 dB** where the bed
is intermittent, since there is less of it in the speech gaps to take out. The
stems buy far more than those numbers suggest: a mixture with a continuous bed
has no silence to cut on, so the difference between "five clips holding a
minute" and "fourteen holding sentences" is the difference between a corpus and
a file. But the bed is attenuated rather than gone, and any consumer should
expect that.

*This entry used to read "5–6.5 dB", and used to argue that the shortfall was
the training distribution — MDX23C learned* sung *vocals and a speaking voice is
not one. That is retracted.* Every figure behind it was measured on LibTorch
while `burn_mdx::TfcTdfNet::forward` was letting the encoder overwrite the
network's head in place; the model separates speech from music about as well as
it separates singing. See CLAUDE.md's **`swap_dims` on LibTorch returns a view
burn-tch forgets the provenance of**.

*The stereo cue is real but small.* The model eats a stereo complex STFT and
finds a centre-panned vocal partly by where it sits in the field, so the obvious
worry is that a near-mono stream starves it. Measured: the side channel sits
16–19 dB under the mid on every excerpt, and folding to true mono and re-running
costs **2.9 dB** (10.4 → 7.5). Real, worth knowing before recording a corpus in
one channel, and not the difference between working and not.

*What would move it further* is a separator trained on speech-over-music rather
than on productions — a different checkpoint, and possibly a different
architecture, not a change to `burn-mdx`. Nothing here blocks that:
`burn_mdx::MdxConfig` already describes the family by shape, and `examples/keys`
reads a new checkpoint's architecture off its tensors. What blocks it is that no
such published checkpoint has been identified — and with 10–12 dB in hand the
case for looking is weaker than it was.

*And the measurement is content-sensitive*, which is the thing not to
re-derive: read at a 250 ms gap window the removal is 10.4 dB, and at a
one-second window the verdict can **invert**, because a between-sentence gap is
a few hundred milliseconds and at one second no "quiet" frame is speech-free.
Any future comparison has to pin the window or it is not comparing anything.

## `preprocess diarize` without a reference

`diarize` is target-speaker extraction: `--reference` is required, and there is
no mode that discovers how many speakers a recording holds. That is the useful
half first — somebody pulling their own speech out of their own stream *has* a
clean sample of themselves, and matching one known voice is far more robust than
clustering — but it is a genuine gap rather than a statement, unlike `seedvc`'s
missing `train`.

*What blocks it* is not the embedding, which already exists in `burn-campplus`
and already separates speakers well enough to threshold. It is that clustering
needs a number of speakers or a stopping rule, and every cheap answer
(agglomerative with a distance cutoff, say) reintroduces exactly the calibration
problem `--threshold` solves by being handed a reference. Worse, the
calibration would be *per recording* rather than per model: what CAM++'s known
failure mode keys on is the channel and the room, which is why two clips of one
speaker from one session score 0.87–0.90 while the same speaker across sessions
scores 0.36–0.78 — a spread wider than the gap between speakers.

*So the honest shape*, if it is built, is clustering that reports its own
confidence and refuses rather than guessing when the modes overlap — the same
thing `preprocess-core`'s `examples/timbre` already does when it prints a gap
that does not exist.
