# tts — GPT-SoVITS speech synthesis

Text on stdin, raw f32le mono PCM on stdout, logs on stderr. **`tts` clones a
voice from a few seconds of reference audio** and speaks your text in it —
GPT-SoVITS v2 ported to Burn, with a native Rust fine-tuning loop when a few
seconds of prompt is not enough. No Python anywhere in the path.

Setup (ffmpeg, ONNX Runtime, LibTorch) is shared by every binary and lives in
[`docs/setup.md`](../../docs/setup.md). `cargo build --release -p tts-cli` gives
`target/release/tts`; the `voice` binary hosts the same code as `voice tts` and
`voice tts train`. See the [repository README](../../README.md) for the toolkit
and [`rvc`](../rvc-cli/README.md) for the conversion engine `tts` usually feeds.

## Quick start

```sh
echo "今天天气很好" \
  | tts --reference clip.wav --reference-text "<what clip.wav actually says>" \
  | ffplay -f f32le -ar 32000 -ac 1 -
```

**One line of stdin is one utterance**; blank lines are skipped and the output is
the utterances concatenated. The model produces 32 kHz — `--sr` resamples, which
is how synthesis feeds conversion with no intermediate file:

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

## Synthesis flags

| flag | default | what it does |
|---|---|---|
| `-r`, `--reference` / `-t`, `--reference-text` | *required* | the clip and what it says, as above |
| `-m`, `--models` | auto-downloaded | directory holding cnhubert, `s1*.ckpt`, `s2G*.pth` |
| `--s1`, `--s2` | base weights | fine-tuned weights, independently overridable |
| `--prosody` | auto-downloaded | directory holding the ONNX prosody encoder |
| `--cache-dir` | see [Where files live](#where-files-live) | where downloads land |
| `-l`, `--language` | `zh` | `zh`, `en` or `ja` |
| `--sr` | `32000` | output sample rate |
| `--top-k` | `15` | sample from the `k` highest-scoring tokens; lower is steadier |
| `--temperature` | `1.0` | below 1 sharpens the distribution, above 1 flattens it |
| `--repetition-penalty` | `1.35` | pushes down tokens already generated |
| `--seed` | `0` | a synthesis is reproducible from its seed |
| `--max-tokens` | `1500` | cap per line; at 25 Hz that is a minute of speech |
| `--backend`, `--device` | `auto` | see below |

**The repetition penalty is load-bearing, not a refinement.** Without it a run of
the same token becomes self-reinforcing and the utterance never ends, until
`--max-tokens` cuts it off with a warning. If that happens on short text, suspect
the prompt rather than the sampler.

The prosody encoder is Chinese-only (a Chinese RoBERTa) and its absence is a
**warning, not a failure**: synthesis continues with zero prosody features,
costing expressiveness rather than intelligibility. `--language en` is flatter
for the same reason — its front-end has no per-character phoneme count to give.
`--language ja` errors rather than guessing, because the wrong front-end produces
fluent-sounding *wrong* audio, which is far harder to notice than a refusal.

## Backends and devices

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

## Fine-tuning (`tts train`)

Two stages adapting different things, which explains most of the flags. **`s1`
carries delivery** — pacing, emphasis, where a speaker breathes. **`s2` carries
timbre** — what makes a clone sound like the speaker rather than like whichever
seconds of reference audio it was prompted with. They write independent files, so
either deploys without the other.

A corpus is `<stem>.wav` beside `<stem>.txt` in one directory (`.mp3`, `.flac`,
`.m4a`, `.ogg`, `.opus` read too; audio with no transcript is skipped with a
warning). **`stt` is how the transcripts get written** — read them before
training: a wrong transcript is the corpus-wide version of a wrong reference one.

```sh
# fish. `stt` reads f32le mono 16 kHz on stdin, so ffmpeg decodes into it.
for f in corpus/*.wav
  ffmpeg -v quiet -i $f -f f32le -ar 16000 -ac 1 - | stt > (path change-extension txt $f)
end

tts train corpus/ -o models/mine --stage both --epochs 10
tts -r clip.wav -t "<transcript>" \
  --s1 models/mine.s1.safetensors --s2 models/mine.s2.safetensors < script.txt
```

`--stage` prepares the corpus **exactly once** — encoding it through the frozen
cnhubert, quantiser and prosody BERT is the expensive half — then trains what was
asked for, so expect an idle-looking pause at the start proportional to the
corpus rather than to the epochs.

| flag | default | what it does |
|---|---|---|
| `-o`, `--out` | `models/voice` | stem; each stage appends `.s1` / `.s2` |
| `-y` | off | overwrite an existing output instead of refusing |
| `--stage` | `both` | `s1`, `s2` or `both` |
| `-e`, `--epochs` | `10` | passes over the corpus |
| `-b`, `--batch-size` | `1` | clips per step — accumulated, not padded, so raising it costs time rather than VRAM |
| `--lr` / `--s2-lr` | `1e-5` / `1e-4` | per-stage learning rates: cross-entropy on a large transformer wants a smaller one than a warm-started GAN |
| `--lr-final` | `0.1` | end-of-run LR as a fraction of the start; decays exponentially |
| `--ema-frac` | `0.1` | EMA window as a fraction of the run; `0` saves raw weights |
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
it should fall and keep falling. `s2`'s is adversarial like `rvc train`'s, where
`g` and `d` only have to stay balanced and `mel` is the number to watch.

On a terminal a live dashboard plots them and **`q` stops early and saves**; the
run's log goes to `train.log` beside the weights so it cannot scribble over the
display. Off a TTY (or with `--no-tui`) logging is on stderr, and Ctrl-C stops
and saves.

## Where files live

Inference assets — cnhubert, `s1*.ckpt`, `s2G*.pth`, the prosody BERT — download
once to a cache resolved in this order: `--cache-dir`, `$TTS_CACHE_DIR`,
`$VOICE_CACHE_DIR`, `$XDG_CACHE_HOME/voice`, `~/.cache/voice`; `tts --help`
prints the path it resolved. `--models` and `--prosody` name a local directory
instead and download nothing. The `s2` discriminator base (`s2D*.pth`) is read
only by training, so it lands beside the run in `<-o's directory>/pretrained/`
rather than in the shared cache. **Nothing is ever written to a pipeline's output
directory.**

`tts completions bash|zsh|fish|powershell|elvish` prints a completion script
generated from the actual flags.

## Reviewing the port

There are no unit tests for the networks, and **weight coverage alone is not
enough** — this repo has shipped a port that loaded at 100% and produced garbage.
`burn-gptsovits`'s examples check coverage (`load`, `keys`) and arithmetic
(`reconstruct`, which round-trips real audio and correlates the energy envelope
at r = 0.91 against a 0.30 chance baseline); end to end is synthesise, then
transcribe with `stt` and compare.

```sh
cargo run -p burn-gptsovits --example load -- sovits <s2G2333k.pth>    # 773/0
cargo run -p burn-gptsovits --example keys -- --group <any checkpoint> # what a new one expects
cargo run -p burn-gptsovits --example reconstruct -- \
  <chinese-hubert-base/pytorch_model.bin> <s2G*.pth> <in.f32le@16k> <out.f32le@32k>
```

[`docs/gptsovits-architecture.pdf`](../../docs/gptsovits-architecture.pdf) is the
long-form paper on what each block computes.
