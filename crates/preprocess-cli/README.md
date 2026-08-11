# preprocess — turn recordings into a corpus

`rvc train` and `tts train` both draw short random windows out of whatever files
they are given. Hand them raw recordings and most of those windows are
between-sentence dead air, a music bed, or somebody else talking — and the model
learns that instead. `preprocess` is the phase in front of them: eight stages, of
which the seven that write anything read audio files and write audio files — so
they compose in whatever order your material needs.

Runtimes, drivers and ffmpeg: [`docs/setup.md`](../../docs/setup.md). Build it
with `cargo build --release -p preprocess-cli`. Every command below is also
`voice preprocess …` — same code, same flags. The exception is `completions`,
which describes a binary rather than a phase: use `voice completions` there.

**There is no bare invocation, and it is the only engine-shaped binary here
without one.** `rvc`, `stt`, `tts` and `seedvc` each name *one transformation*,
so "the engine on a pipe" is a complete description of what a bare invocation
does.
`preprocess` names *a phase*, and which stage of it to run is exactly what the
subcommand chooses — a bare `preprocess` would have to pick one silently, and
whichever it picked would be wrong for everybody who wanted a different one.

**`rvc preprocess` and `tts preprocess` are gone**, removed rather than
deprecated, so an old command line fails to parse instead of quietly doing
something else. `preprocess clip` is the slicer they were.

## Commands

| | |
|---|---|
| `preprocess analyze <files…>` | report what a corpus *is*, and what to set the rest of these to. Runs no model, writes nothing |
| `preprocess separate <files…>` | split a recording into a voice stem and a music stem (MDX23C) |
| `preprocess diarize <files…> -r <clip>` | keep only what one voice is saying (CAM++) |
| `preprocess denoise <files…>` | strip a steady background hiss |
| `preprocess normalize <files…>` | match every recording to one level, by peak or by loudness |
| `preprocess trim <files…>` | strip the silence at each end, without splitting anything |
| `preprocess resample <files…>` | one sample rate and one channel count, stated rather than implied |
| `preprocess clip <files…>` | slice into clean per-utterance clips — the last stage, and the one a trainer eats |
| `preprocess completions <shell>` | completion script for bash, zsh, fish, powershell or elvish |

Inputs are files **or directories** — a directory is expanded recursively to the
audio in it (`mp3`, `wav`, `flac`, `m4a`, `ogg`, `opus`, `aac`, `wma`). One
undecodable file is skipped with a line on stderr rather than aborting the batch.
Every stage writes into a directory that is **not** its input by default, so a
badly chosen knob is always recoverable by re-running with a different one.

## The multi-speaker recipe: `separate` then `diarize`

This is the case most people arrive with, and it needs both stages because
neither can do the other's job. A stream with somebody else's music under it
holds two problems: the bed, and the singer in it.

```sh
preprocess separate raw/     -o vocals/ --stem vocals
preprocess diarize  vocals/  -o mine/   --reference me.wav
preprocess clip     mine/    -o dataset/
```

**`separate` asks whether something is a voice, never whose it is.** Every
speaker in the mixture lands in the vocals stem — the person talking *and* the
singer on the backing track. That is not a shortcoming of this checkpoint; it is
what source separation is. So a recording with one speaker over instrumental
music is finished after `separate`, and a recording with a *sung* voice under it
needs `diarize` after it.

**`diarize` is target-speaker extraction, not clustering.** `--reference` is
required and there is no mode without it: each window of the input is embedded
by CAM++ and scored by cosine against the reference, and runs of windows that
clear `--threshold` are written out.

**Take the reference from the recording you are filtering.** This is the single
likeliest reason a run keeps nothing, and it is a property of the embedding
rather than of this stage: CAM++ keys partly on the microphone and the room, so
two clips of one speaker *from one session* score 0.87–0.90 while the same
speaker across sessions scores 0.36–0.78 — a spread wider than the gap between
different speakers that `--threshold` has to sit in (measured by
`preprocess-core`'s `examples/timbre`; see [`diarize`](#diarize) for what was and
was not re-run here). Ten seconds cut out of the file you are about to filter
beats a clean studio sample of the same person.

Run `analyze` first if you are not sure the bed is even the problem — it says so
in one decode and costs nothing.

**What the three stages actually did**, run here on 60 s of a real stream with a
song under it, on `--backend tch`:

```
preprocess separate at_180s.wav        --stem both    →  60.0s in 21 passes
                                                          vocals rms 0.0323, instrumental rms 0.0258
preprocess diarize  at_180s.vocals.wav -r ref.wav     →  4 segments, 55.0 of 60.0s kept
                                                          47 of 58 windows, mean cosine 0.684
preprocess clip     at_180s.vocals.wav                →  13 clips, 41.4 of 60.0s kept
```

The reference was 14 s of the same speaker from a different part of the same
recording, passed through `separate` the same way — which is the rule above,
followed. 47 of 58 windows on an excerpt that is mostly this one person is the
retention the stage is tuned for; on material where somebody else is singing
under him it is the *rejection* that matters, and that number is in
[`diarize`](#diarize) below.

## `analyze`

The only stage that loads no model, takes no `--backend`, downloads nothing and
writes nothing. It decodes and does arithmetic, so it runs over a whole corpus
at decode speed. The report goes to **stdout**; skips and advisories go to
stderr, so `--format json` is always exactly one parseable document.

| flag | default | |
|---|---|---|
| `--sr` | `48000` | rate to decode at. Nothing is written, so this only sets the grid the measurement is taken on |
| `--silence-db` | `-40` | the `clip` settings to report *against* — the clip counts below are what `clip` would produce from these very flags |
| `--min-silence` | `0.3` | |
| `--min-clip` | `1.0` | |
| `--max-clip` | `0` | |
| `--pad` | `0.15` | |
| `--format` | `text` | `text` to read, `json` to drive something with |
| `--per-file` | off | a line per file as well as the corpus summary |

**The suggested `--silence-db` is checked rather than asserted**, which is the
reason this stage is worth a decode: the slicer is run **twice**, once at the
floor you passed and once at one derived from the recording's own quiet frames,
and both verdicts are printed. That is also the only way to see the case that
matters most — **a recording with no dead air at either floor**. Speech over a
continuous music bed is that shape: the floor sits under the music, the slicer
finds nothing quiet, and no `--silence-db` can rescue it because the quiet is not
there to find. `analyze` says so instead of suggesting a threshold that could not
work, and what to do about it is `separate`.

Measured with this binary on this repository's own material, at the default
`--silence-db -40` and then at the floor read off each recording:

| | length | floor | SNR | at −40 dB | at the measured floor |
|---|---|---|---|---|---|
| `dataset/`, 92 clips a previous `clip` run wrote | 1207 s | −47.2 dBFS | 13.0 dB | 72 clips / 4% dead air | 285 / 15% |
| the source stream, whole | 1191 s | −40.3 | 14.2 | 64 / 4% | 196 / 15% |
| a 60 s excerpt of it with the music bed running | 60 s | −38.7 | 11.4 | **3 / 0%** | 12 / 14% |
| the same 60 s after `separate --stem vocals` | 60 s | −51.0 | **22.3** | 13 / 31% | 13 / 18% |

**Read the third row first, because it is what a slicer failing looks like.**
Three clips out of sixty seconds of speech, holding 59.9 of the 60 — `clip` did
not cut anything, it renamed the file three times. A trainer then draws 0.48 s
windows out of that, and most of them are somebody else's music. The fourth row
is the same audio after `separate`: the floor falls 12 dB, the SNR nearly
doubles, and the same `clip` invocation returns **13** clips holding 41.4 s,
which are sentences. Confirmed by running it —

```
preprocess clip at_180s.wav        →  3 clips (60.0s in -> 59.9s kept)
preprocess clip at_180s.vocals.wav → 13 clips (60.0s in -> 41.4s kept)
```

The first two rows are the same point at corpus scale: the default floor is
wrong for this material by a factor of three to four in clip count, and it is
wrong in the direction that merges sentences into paragraphs rather than the one
that fragments them. The suggested floor is what closes that, and it is worth
running before `clip` on anything not recorded in a quiet room.

**The suggestion can land on either side of the default, and that is not it
being arbitrary.** Per file it is the measured floor plus a margin — 8 dB, or
40% of the way up to the speech level, whichever is nearer the floor — so the
stream, whose floor is −40.3 with speech at −26.1, suggests **−34.7**. Over a
corpus it is the **lowest** of the per-file suggestions rather than their
middle, which is why 92 clips whose floors spread from −77.6 to −33.1 suggest
**−69.6**: a floor that is too low leaves some dead air, which `--min-clip` and
`--pad` absorb, while one that is too high eats the soft tails this toolkit
exists to preserve. Both directions raise the clip count here because −40 was
not where either recording's dead air actually sat. The spread is printed beside
the floor so a corpus that disagrees with itself is visible rather than averaged
away.

There is no loudness (LUFS) column. ffmpeg reports that through its log rather
than through the samples, so reading it would mean installing a global log
callback and parsing stderr for one number that peak, floor and SNR already
answer between them.

## `separate`

MDX23C (TFC-TDF-UNet v3), the separator UVR ships. One recording in,
`<stem>.vocals.wav` and/or `<stem>.instrumental.wav` out.

| flag | default | |
|---|---|---|
| `-o`, `--output-dir` | `separated` | |
| `--stem` | `vocals` | `vocals`, `instrumental` or `both` — both come out of the same pass, so `both` costs only the second file |
| `-m`, `--model` | auto-downloaded | an MDX23C checkpoint you already have |
| `--cache-dir` | the shared cache | where the download lands; `-h` prints the resolved path for this machine |
| `--backend` | `auto` | `cuda`, `tch`, `wgpu` and their aliases. **No `onnx`** — this model is published as a PyTorch checkpoint only, and `--backend onnx` is refused with that reason rather than pointing at a feature flag that cannot exist |
| `--device` | `auto` | `cpu`, `gpu`, `gpu:N`, `mps`, `vulkan` |
| `--download-timeout`, `--retries` | | how patient the 448 MB fetch is |

**There is no `--sr` here, unlike every other stage.** The stems are written at
the model's own **44.1 kHz** and in **stereo**: the network is stereo-native and
folding the field away would throw out one of the two cues it separates on.
Writing them at anything else would be a resample stacked on a separation, and
the next stage's decode performs one anyway.

**Run it before `clip`, and expect the music attenuated rather than gone.** The
network reaches 15–20 dB of bed removal on a song and **5–6.5 dB** on a person
talking over one, because it was trained on *sung* vocals and a speaking voice is
not one. Those two figures are `burn-mdx`'s own harness (`cargo run -p burn-mdx
--example separate --features tch -- --mixture <file>`) and were **not re-run for
this page**; the example's module doc is where they stay current.

What was re-run here is the consequence, which is out of proportion to the
decibels: slicing cuts on silence, and a continuous bed means the recording has
none, so a corpus recorded behind music cannot be cut into sentences *at all*
until the bed comes off. On 60 s of a real stream, `clip` gave **3** clips
holding 59.9 of the 60 seconds before this stage and **13** holding 41.4 after —
and the noise floor `analyze` reads fell from −38.7 to −51.0 dBFS with the SNR
going 11.4 → 22.3 dB.

Two consequences of emptying those gaps, both worth expecting downstream: a
recogniser has more room to invent in a newly-silent gap, and very short
fragments appear — so whatever consumes these stems wants a duration floor
(`clip --min-clip`, which defaults to 1 s, is one).

## `diarize`

CAM++ speaker embedding. Windows of the input scored against a reference clip;
retained runs written as `<stem>_<NNN>.wav`.

| flag | default | |
|---|---|---|
| `-r`, `--reference` | **required** | a few clean seconds of the voice to keep. Only the first 30 s are read — the embedding saturates |
| `-o`, `--output-dir` | `diarized` | |
| `--threshold` | `0.55` | cosine similarity a window must reach. Raise to drop anything doubtful, lower to keep more of the target at the cost of admitting other speakers |
| `--window` | `3.0` | seconds per comparison |
| `--hop` | `1.0` | seconds between windows — the resolution of the output. Halving it doubles the work |
| `--min-segment` | `1.0` | drop a retained stretch shorter than this |
| `--sr` | `48000` | rate the retained segments are written at |
| `-m`, `--model` | auto-downloaded | |
| `--cache-dir`, `--backend`, `--device` | | as `separate`, including the absence of `onnx` and for the same reason |

**The output is window-quantised, not sample-accurate.** Edges land on `--hop`
boundaries, a window spanning a speaker change belongs to whoever dominates it,
and a kept run is widened to cover its last window in full. So a segment can
carry a fraction of a second of the other voice at each end, the seconds written
are always more than the seconds matched, and a one-word interjection shorter
than a window will not be removed on its own. The window counts in the per-file
line are the honest measure of how much of a recording matched.

**`--window` below about 2 s stops working**, and that is the speaker model's own
context pooling showing up as a number rather than a tuning preference: at 1.5 s
the target's lower quartile sits *below* the rejected material's upper one, so no
threshold separates them. 5 s separates marginally better than 3 s and quantises
every boundary five seconds wide, which is why 3 s is the default. At 3 s,
`--threshold 0.55` kept **80%** of the target's windows and rejected **86%** of
music-and-singer ones.

Those figures and the 0.87–0.90 / 0.36–0.78 spread above are
`preprocess-core`'s own harness — `cargo run -p preprocess-core --features tch
--example timbre` — measured on a separated 60 s of stream against a 14 s
reference from the same recording. **They were not re-run for this page**; the
harness is where they stay current, and it is what to re-run if the model ever
changes. Two limits on them, both bounding from above: reference and material
share a session, which is the case CAM++ flatters; and the rejected material is a
music bed with a sung voice in it rather than a second person talking, because no
clean recording of a second speaker was available.

There is **no RMS gate** on a window, and that is a measurement rather than an
oversight: near-silent windows already score far below any threshold that keeps
the target, so a floor would be a second knob agreeing with the first.

One run that *was* made here, for what a healthy invocation looks like: the
separated 60 s excerpt above, against a 14 s reference from elsewhere in the same
recording, kept **47 of 58 windows** at a mean cosine of **0.684** and wrote 4
segments holding 55.0 of the 60 seconds. The per-file line reports all of those,
and the window count — not the seconds — is the number to read, because a kept
run is widened to its last window's end.

## `denoise`

The same `anlmdn` de-hisser `rvc --denoise` runs on its output, pointed at a
corpus instead — which is the better place for it, because hiss that survives
into the training clips is hiss the model learns to reproduce. One file in, one
`<stem>.wav` out.

| flag | default | |
|---|---|---|
| `-o`, `--output-dir` | `denoised` | not the input directory, so a bad `--denoise-strength` is recoverable |
| `--sr` | `48000` | decode and write rate. Hiss is broadband, so a stage that resampled on the way through would change what it is removing |
| `--denoise-strength` | `0.008` | raise to remove more hiss; lower if soft breathy texture starts to smear |
| `--denoise-patch` | `0.002` | seconds — the unit compared for self-similarity; smaller keeps finer detail |
| `--denoise-research` | `0.006` | seconds — how far in time it looks for similar patches. Must exceed the patch |

There is deliberately no switch to turn it off, unlike `rvc --denoise`: here the
stage *is* the subcommand, so a flag disabling it would leave a command that
copies files.

`anlmdn` averages each short patch with self-similar patches found nearby in
time, so steady hiss averages away while ever-changing breath texture — which is
spectrally indistinguishable from hiss, and is exactly what this toolkit exists
to preserve — has few close matches and survives. A spectral-subtraction
de-noiser keyed on level and frequency does not have that property, which is why
this one is not that.

## `normalize`

One constant gain per file, and nothing else. `--peak` and `--lufs` answer
different questions.

| flag | default | |
|---|---|---|
| `-o`, `--output-dir` | `normalized` | |
| `--sr` | `48000` | rate the output is written at |
| `--peak` | `0.95` | the loudest sample lands here, as a fraction of full scale. Exact, instant, and blind to everything but that one sample — so a stray thump sets the gain for a whole recording |
| `--lufs` | — | EBU R128 integrated loudness, e.g. `-23`. K-weighted and gated, so the silence between sentences does not drag the reading down. Unbothered by the thump |

Ask for one or the other, never both; clap refuses the pair by name before a file
is opened.

**Nothing here compresses, limits or rides the level**, and that is the reason
ffmpeg's own `loudnorm` is not what runs: in single pass that filter is a
*dynamic* normalizer, which is exactly the processing breathy close-mic material
must not get. The consequence is that a `--lufs` run can push the peak past full
scale on a recording with a wide range. The per-file line reports the peak it
reached, and **it is reported rather than limited** — a note on stderr counts how
many files it happened to.

Loudness is measured with `ebur128` and measured **again after the gain**, so the
second number in the report is a reading rather than arithmetic. A file with no
gated reading at all — under 0.4 s, or entirely silent — is skipped with that
reason, never quietly normalized by peak instead.

Note both flags take their value with `=` or a space (`--lufs=-23` or
`--lufs -23`); every negative-valued flag in this toolkit accepts both.

## `trim`

The silence at each end, and nothing in the middle. One file in, one file out.

| flag | default | |
|---|---|---|
| `-o`, `--output-dir` | `trimmed` | |
| `--sr` | `48000` | rate the output is written at |
| `--silence-db` | `-40` | energy floor in dBFS |
| `--pad` | `0.15` | seconds of bordering quiet kept at each edge |
| `--measure-floor` | off | read the floor off the recording's own quiet frames instead of using `--silence-db` |

`clip` finds these boundaries already and then throws the whole-file case away,
because its job is to cut a recording into sentences. **A take that is already
*one* utterance wants the same measurement and one file back**, which is the
entire difference between the two stages. A file with nothing above the floor is
written through whole rather than emptied, and counted on stderr — that usually
means the floor is wrong for that take.

## `resample`

The conversion every other stage does on the way past, stated rather than
implied.

| flag | default | |
|---|---|---|
| `-o`, `--output-dir` | `resampled` | |
| `--sr` | `48000` | the rate everything is written at |
| `--channels` | `mono` | `mono` folds a stereo input to `(L + R) / 2`; `stereo` writes a mono input to both |

A corpus assembled from several sources arrives at a trainer as several rates,
because each stage writes what it produced. This is the stage that says which
one. The channel count is reported per file because it is the half a user is
least likely to have thought about: **`separate` writes stereo on purpose**, and
folding that to mono is a decision rather than a detail.

## `clip`

The slicer, and the stage a trainer's corpus comes out of. Each input becomes
`<stem>_<NNN>.wav`.

| flag | default | |
|---|---|---|
| `-o`, `--output-dir` | `dataset` | |
| `--sr` | `48000` | rate the clips are written at. A trainer re-decodes at whatever rate it needs, so this only decides what is on disk; matching the rate you will train at avoids a resample |
| `--silence-db` | `-40` | energy floor in dBFS. Lower it (e.g. `-50`) to keep the very softest passages |
| `--min-silence` | `0.3` | seconds a quiet gap must last to count as a sentence boundary, so a shorter internal pause never splits a sentence |
| `--min-clip` | `1.0` | drop anything shorter |
| `--max-clip` | `0` | hard cap in seconds; `0` means never split a long sentence |
| `--pad` | `0.15` | seconds of bordering quiet kept at each edge, so onsets and soft breathy tails are not clipped |
| `--normalize` | off | peak-normalize each clip to ~0.95 full scale |

**Energy is used only to *find* long silent gaps, never to gate quiet-but-present
sound.** That is the whole point for soft, breathy, close-mic material: the
softest passages are content, not noise floor. The slicer itself is
`audio_kit::slice`, the same one `stt` segments on, so the two cannot disagree
about where a sentence ends.

`--silence-db` and `--min-silence` are the two knobs worth touching, and
`analyze` is how to pick them rather than guessing. **Listen to a few output
clips before committing a training run to them.**

## Where the weights come from

| stage | model | repository |
|---|---|---|
| `separate` | `MDX23C-8KFFT-InstVoc_HQ.ckpt`, 448 MB | `Politrees/UVR_resources`, pinned to a revision — the MDX family shares filename prefixes across incompatible architectures, so the revision is load-bearing rather than decoration |
| `diarize` | `campplus_cn_common.bin`, 28 MB | `funasr/campplus` |

`clip`, `denoise` and `analyze` fetch nothing at all.

Both are **inference assets**, so both go to the shared cache — `--cache-dir`,
then `$VOICE_CACHE_DIR`, then `~/.cache/voice`, resolved as
[`docs/setup.md`](../../docs/setup.md#where-models-are-stored) describes and
printed by `-h`. Neither has a `pretrained/` counterpart: those are warm-start
bases, and nothing here trains.

**There is no `preprocess download`, and that is the rule rather than a gap.** A
`download` subcommand fetches what a default *bare invocation* would fetch on
demand, and this binary has no bare invocation to have a default — three of its
eight stages fetch nothing whatever. `separate` and `diarize` fetch on their first
run like everything else, and `--model` on either points at a copy you already
have.

## Limits

`diarize` has no mode that runs without `--reference`: discovering how many
speakers a recording holds is blind clustering, which is a different job and is
not built — see [`docs/roadmap.md`](../../docs/roadmap.md) for what actually
blocks it. `separate` is a music/voice split and always will be; it cannot be
made to tell two speakers apart. Neither model has an ONNX path, because neither
is published as a graph and nothing here exports them.

## See also

- [repository README](../../README.md) — what `voice` is and how the engines compose
- [`docs/setup.md`](../../docs/setup.md) — ffmpeg, LibTorch, backends and devices
- [`docs/training.md`](../../docs/training.md) — why a trainer needs any of this
- [`rvc`](../rvc-cli/README.md), [`stt`](../stt-cli/README.md), [`tts`](../tts-cli/README.md) and [`seedvc`](../seedvc-cli/README.md) — the engines this feeds
- [`CLAUDE.md`](../../CLAUDE.md) — why the code is shaped the way it is
