# tts — GPT-SoVITS speech synthesis

Speak text in a voice cloned from a few seconds of reference audio. `tts` is
pure-Rust GPT-SoVITS v2: a transformer predicts *delivery* as semantic tokens and
a VITS decoder renders them in the reference's *timbre*. Inference runs on ONNX
Runtime or native Burn; fine-tuning runs on any Burn backend.

Runtimes, drivers and ffmpeg: [`docs/setup.md`](../../docs/setup.md). What the
networks compute:
[`docs/gptsovits-architecture.pdf`](../../docs/gptsovits-architecture.pdf).
Build it with `cargo build --release -p tts-cli`. Every command below is also
`voice tts …` — same code, same flags. The exception is `completions`, which
describes a binary rather than an engine: use `voice completions` there.

## Commands

| | |
|---|---|
| `tts -r <clip> -t <transcript>` | **the bare invocation is the filter** — one line of text per utterance on stdin, f32le mono PCM on stdout |
| `tts convert <text files…>` | speak whole files, one `<stem>.wav` per input |
| `tts train <corpus>` | fine-tune `s1`, `s2` or both on a voice |
| `tts preprocess <files…>` | slice recordings into clean per-sentence clips, ready for `stt` |
| `tts download` | prefetch what a synthesis fetches on its first run |
| `tts completions <shell>` | completion script for bash, zsh, fish, powershell or elvish |

Logs go to stderr, so stdout is only ever samples. Blank input lines are skipped.

`tts download` fetches what a default run fetches on demand — the GPT-SoVITS
bundle and the prosody encoder — and prints where each landed, which is what
`--models` and `--prosody` take. It is the way to set up a machine that will be
offline later, or to keep a batch job from spending its first minutes on the
network. `--no-prosody` leaves out the ~1.3 GB encoder; the `s2` discriminator
is never fetched here at all, because only fine-tuning opens one and `train`
gets it when asked.

```sh
echo "今天天气很好" \
  | tts -r clip.wav -t "<what clip.wav actually says>" \
  | ffplay -f f32le -ar 32000 -ac 1 -
```

`tts convert` is the same synthesis with files at both ends —
`{output_dir}/{stem}.wav`, one per input. **A file's non-blank lines are still
one utterance each; they are concatenated into that file's single WAV**, so the
line breaks are how you tell the model where an utterance ends, not how many
files come out. The reference is decoded and analysed once for the whole batch,
so a directory of scripts costs one startup rather than one per file, and the
first failure stops the run rather than being skipped past. Here the two
reference flags are plain required arguments — nothing else hangs off this
subcommand, so leaving one out is a usage error rather than a diagnosis after a
gigabyte has loaded.

```sh
tts convert script.txt chapter2.txt -o out/ \
  -r clip.wav -t "<what clip.wav actually says>"
```

The model produces 32 kHz; `--sr` resamples, which is how synthesis feeds
conversion with no intermediate file:

```sh
tts -r clip.wav -t "<transcript>" --sr 16000 < script.txt \
  | rvc -m models/voice.safetensors --model-sr 48000 > out.f32le
```

## The reference clip

A few seconds of clean speech in any format ffmpeg reads, decoded to mono 16 kHz.
It does two jobs — its semantic tokens prime `s1`, its spectrogram becomes the
speaker vector `s2` renders with. Under half a second is rejected; much more than
a few seconds mostly costs time.

**`--reference-text` is required, and it must be what the clip actually says.**
`s1` generates by *continuation*: primed with the reference's phonemes beside the
reference's semantic tokens, then asked to keep going with your text's phonemes.
Given only the target text it sees phonemes for one utterance next to audio of
another, finds nothing to continue, and stops after a token or two — which looks
exactly like a broken decoder and is not one.

**A transcript that is merely *wrong* fails worse, because it fails quietly**:
`s1` renders what it was handed before reaching your text, so the output grows
extra leading speech and roughly doubles in length, with nothing to error about.
A duration that does not match the text length is the tell.

## Flags

`tts --help` and `tts train --help` list every flag with its default. Two are
worth knowing before you hit them:

**The repetition penalty is load-bearing, not a refinement.** Without it a run of
the same token becomes self-reinforcing and the utterance never ends, until
`--max-tokens` cuts it off with a warning. If that happens on short text, suspect
the prompt rather than the sampler.

The prosody encoder is Chinese-only (a Chinese RoBERTa) and its absence is a
**warning, not a failure**: synthesis continues with zero prosody features,
costing expressiveness rather than intelligibility.

## Backends

One binary carries every backend it was built with, and `tts train` takes the
same flag with the same meanings.

| `--backend` | aliases | runtime |
|---|---|---|
| `auto` *(default)* | | an ONNX export in `--models` if there is one, else the fastest Burn backend |
| `onnx` | | ONNX Runtime, over the graphs `export/export_gptsovits.py` writes |
| `cuda` | `burn`, `burn-cuda` | native Burn, CubeCL/CUDA |
| `tch` | `libtorch`, `burn-tch` | native Burn, LibTorch |
| `wgpu` | `webgpu`, `burn-wgpu` | native Burn, WebGPU — no vendor toolkit, runs on AMD/Intel/Apple |

`--device auto|cpu|gpu|gpu:N|mps|vulkan` picks *which* device inside the chosen
backend (`cuda`/`cuda:N` are accepted spellings of `gpu`; `--backend onnx`
ignores it). Naming a backend or device that is unavailable is an **error with a
reason**, never a silent fallback — only `auto` substitutes.

**Where the time goes.** `s2` is one pass; `s1` is autoregressive at 25 tokens
per second of speech, so a ten-second line is 250 sequential, launch-bound
decoder steps. That is why a GPU matters here.

## Fine-tune a voice

Two stages adapting different things, which explains most of the flags. **`s1`
carries delivery** — pacing, emphasis, where a speaker breathes. **`s2` carries
timbre** — what makes a clone sound like the speaker rather than like whichever
seconds of reference audio it was prompted with. They write independent files, so
either deploys without the other.

A corpus is `<stem>.wav` beside `<stem>.txt` in one directory (`.mp3`, `.flac`,
`.m4a`, `.ogg` and `.opus` are read too; audio with no transcript is skipped with
a warning). Build one from raw recordings in two steps — `preprocess` cuts them
into per-sentence clips, `stt` writes a transcript beside each. **Read the
transcripts before training**: a wrong one is the corpus-wide version of a wrong
reference text.

```sh
tts preprocess raw/*.mp3 -o corpus/ --sr 32000

# `stt` reads f32le mono 16 kHz on stdin, so ffmpeg decodes into it.
fd -e wav . corpus -j 1 -x sh -c \
  'ffmpeg -v quiet -i "$1" -f f32le -ar 16000 -ac 1 - | stt > "$2"' _ {} {.}.txt

tts train corpus/ -o models/mine --stage both --epochs 10
tts -r clip.wav -t "<transcript>" \
  --s1 models/mine.s1.safetensors --s2 models/mine.s2.safetensors < script.txt
```

`--stage` prepares the corpus **exactly once** — encoding it through the frozen
cnhubert, quantiser and prosody BERT is the expensive half — then trains what was
asked for, so expect an idle-looking pause at the start proportional to the
corpus rather than to the epochs.

`--max-tokens` **skips** a clip longer than it rather than truncating one: a
cut-off clip teaches the model to stop early. `-b` accumulates rather than pads,
so raising it costs time rather than VRAM, and `--backend onnx` is rejected here
because ONNX Runtime cannot train.

Each stage writes `<stem>.<s1|s2>.safetensors` — the weight **EMA**, markedly
steadier on a small corpus and the one to deploy — beside a `.raw.safetensors`
twin holding the live final-step weights. **`s1`'s loss means something on its
own**: plain next-token cross-entropy, one model, one optimizer, one number, and
it should fall and keep falling. `s2`'s is adversarial, where `g` and `d` only
have to stay balanced and `mel` is the number to watch.

On a terminal a live dashboard plots them and **`q` stops early and saves**. Off
a TTY (or with `--no-tui`) logging is on stderr, and Ctrl-C stops and saves.

How the two loops work — `s2` is the same VITS GAN `rvc` trains with, `s1` is
not — is [`docs/training.md`](../../docs/training.md).

## Limits

**Chinese is the language that works fully.** `--language en` is intelligible but
flatter: its front-end has no per-character phoneme count to give the prosody
encoder, which is upstream's own behaviour. `--language ja` errors rather than
guessing, because the wrong front-end produces fluent-sounding *wrong* audio,
which is far harder to notice than a refusal. Mandarin readings that depend on
grammar rather than on the word are still wrong; g2pw is the fix and is not
written.

Synthesis is per utterance, not streaming — `s1` finishes a line before `s2`
renders it. The networks have no unit tests, and **weight coverage alone is not
enough**: `burn-gptsovits`'s `load`, `keys` and `reconstruct` examples check
coverage and arithmetic, and synthesising then transcribing with `stt` is the
end-to-end check.
