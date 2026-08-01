# tts — GPT-SoVITS speech synthesis

Speak text in a voice cloned from a few seconds of reference audio. `tts` is
pure-Rust GPT-SoVITS v2: a transformer predicts *delivery* as semantic tokens and
a VITS decoder renders them in the reference's *timbre*. Inference runs on ONNX
Runtime or native Burn; fine-tuning runs on any Burn backend.

Runtimes, drivers and ffmpeg: [`docs/setup.md`](../../docs/setup.md). What the
networks compute:
[`docs/gptsovits-architecture.pdf`](../../docs/gptsovits-architecture.pdf).
Build it with `cargo build --release -p tts-cli`. Every command below is also
`voice tts …` — same code, same flags.

## Commands

| | |
|---|---|
| `tts -r <clip> -t <transcript>` | **the bare invocation is the filter** — one line of text per utterance on stdin, f32le mono PCM on stdout |
| `tts train <corpus>` | fine-tune `s1`, `s2` or both on a voice |
| `tts completions <shell>` | completion script for bash, zsh, fish, powershell or elvish |

Logs go to stderr, so stdout is only ever samples. Blank input lines are skipped.

```sh
echo "今天天气很好" \
  | tts -r clip.wav -t "<what clip.wav actually says>" \
  | ffplay -f f32le -ar 32000 -ac 1 -
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

| flag | default | |
|---|---|---|
| `-r`, `--reference` / `-t`, `--reference-text` | *required* | the clip and what it says, as above |
| `-m`, `--models` | auto-downloaded | directory holding cnhubert, `s1*.ckpt` and `s2G*.pth` |
| `--s1`, `--s2` | base weights | fine-tuned weights, each overridable on its own |
| `--prosody` | auto-downloaded | directory holding the ONNX prosody encoder |
| `--cache-dir` | see [Where files land](#where-files-land) | where downloads land |
| `-l`, `--language` | `zh` | `zh`, `en` or `ja` |
| `--sr` | `32000` | output sample rate |
| `--top-k` | `15` | sample from the `k` highest-scoring tokens; lower is steadier |
| `--temperature` | `1.0` | below 1 sharpens the distribution, above 1 flattens it |
| `--repetition-penalty` | `1.35` | pushes down tokens already generated |
| `--seed` | `0` | a synthesis is reproducible from its seed |
| `--max-tokens` | `1500` | cap per line; at 25 tokens per second that is a minute |

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
a warning). **`stt` is how the transcripts get written** — read them before
training: a wrong transcript is the corpus-wide version of a wrong reference one.

```sh
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

| flag | default | |
|---|---|---|
| `-o`, `--out` | `models/voice` | stem; each stage appends `.s1` / `.s2` |
| `-y` | off | overwrite an existing output instead of refusing |
| `--stage` | `both` | `s1`, `s2` or `both` |
| `-l`, `--language` | `zh` | language of the transcripts |
| `-e`, `--epochs` | `10` | passes over the corpus |
| `-b`, `--batch-size` | `1` | clips per step — accumulated, not padded, so raising it costs time rather than VRAM |
| `--lr` / `--s2-lr` | `1e-5` / `1e-4` | per-stage learning rates: cross-entropy on a large transformer wants a smaller one than a warm-started GAN |
| `--lr-final` | `0.1` | end-of-run LR as a fraction of the start; decays exponentially |
| `--ema-frac` | `0.1` | EMA window as a fraction of the run; `0` saves the raw weights |
| `--max-tokens` | `1500` | **skip** clips longer than this, never truncate — a cut-off clip teaches an early stop |
| `--segment-frames` | `32` | latent frames `s2` renders per step; trades VRAM against little else |
| `--d-lr-ratio`, `--d-interval` | `1.0`, `1` | hold off an `s2` discriminator that is winning |
| `--no-save-best` | off | stop keeping a best-so-far `s2` checkpoint |
| `--backend`, `--device` | `auto` | as above, except that `onnx` is rejected — ONNX Runtime cannot train. `--device` takes a comma-separated list for data-parallel training, the first being the master |
| `--no-tui` | off | disable the dashboard and log to stderr |

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

## Where files land

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
