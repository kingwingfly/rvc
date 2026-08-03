# stt — speech recognition as a Unix filter

Turn a recording into text. **`stt` is Whisper**, running as pure-Rust inference
on either of two interchangeable runtimes — a native [Burn](https://burn.dev)
port on LibTorch, CubeCL/CUDA or WebGPU, or an `optimum`-style export on ONNX
Runtime — behind one command that reads raw PCM on stdin and writes transcripts
on stdout. No Python, at run time or at install time.

Every flag below is also `voice stt …`, from this crate as a library rather than
a copy of its arguments.

## Install

Toolchain setup — ffmpeg, and ONNX Runtime or LibTorch if you want those
backends — is written once in [`docs/setup.md`](../../docs/setup.md).

```sh
cargo build --release -p stt-cli    # -> target/release/stt
```

The four backend features — `cuda`, `tch`, `wgpu`, `onnx` — are all on by
default, so one binary carries every runtime and `--backend` chooses at run time.
Drop what you do not want: `--no-default-features --features cuda,tch,wgpu`
builds with no ONNX Runtime anywhere, `--features onnx` alone needs neither
LibTorch nor a CUDA toolkit. Naming a backend that was not compiled in is an
error that says so.

**A debug build runs models at full speed**, because the `dev` profile gives
dependencies `opt-level = 3` and that is where every tensor operation lives —
24 s of audio takes 19 s in debug against 18 s in release. Reach for `--release`
to benchmark, not to test.

## Shape

Recognition is the **bare invocation**, the same as `rvc` and `tts`: stdin to
stdout, logs on stderr, no `--input` flag. Two subcommands sit beside it —
`stt download`, which fetches the weights the first run would otherwise pull
from Hugging Face, and `stt completions <shell>`.

```sh
ffmpeg -v quiet -i take.mp3 -f f32le -ar 16000 -ac 1 - | stt
```

Input is mono **f32le PCM at 16 kHz** — Whisper's analysis rate, and the wire
format every engine here speaks. **Each line is written and flushed as its
segment closes**, so a four-minute recording starts producing text seconds in
rather than at the end, and a live microphone is transcribed as you speak.

Nothing is traded away for that. A segment is handed to the decoder only once no
later sample could move its boundaries — `--min-silence` of continued quiet, or
twice the edge padding, whichever is longer — so the transcript is the one the
whole recording would have given; a property test in
[`audio-kit`](../audio-kit/src/slice.rs) pins the two paths to identical cuts.
The wait for a line is therefore that segment's own length, plus `--min-silence`
to prove it ended, plus its decode. The **one** case that differs is speech that
never pauses for longer than `--max-clip`: it is cut at the quietest frame inside
that window instead of being balanced across the whole run.

## Flags

`stt --help` lists every flag with its default. What is worth knowing beyond
that is **segmentation**, because Whisper handed a long quiet stretch invents
fluent sentences to fill it and nothing in the output says so. The slicer
([`audio-kit`](../audio-kit/src/slice.rs), shared with `rvc preprocess`) cuts
only where audio is both below `--silence-db` *and* quiet for longer than
`--min-silence`, so soft breathy tails stay inside their segment. Lower the floor
(`--silence-db=-50`) when soft speech is being cut into fragments; raise
`--min-silence` when single sentences are being split.

`--max-clip` **must be in `(0, 30]`** — one segment has to fit one 30 s encoder
window — and is rejected up front rather than silently truncated. Hitting
`--max-tokens` logs a warning for the same reason: a cut-off transcript looks
perfectly well-formed and simply stops mid-sentence.

Every model dimension and every control-token id is read from the repo's own
JSON, so another size or a fine-tune costs no code. `RUST_LOG` sets the log
level; the default is `info,ort=warn`.

## Backends

| `--backend` | aliases | runtime | devices |
|---|---|---|---|
| `auto` *(default)* | | reads the model directory: an ONNX export → `onnx`, `model.safetensors` → the fastest Burn backend available | |
| `onnx` | | ONNX Runtime (`ort`) | CUDA EP, else CPU — **`--device` is ignored**, ORT picks its own provider |
| `cuda` | `burn`, `burn-cuda` | native Burn, CubeCL kernels | NVIDIA only |
| `tch` | `libtorch`, `burn-tch` | native Burn, LibTorch | CUDA, MPS, Vulkan, CPU |
| `wgpu` | `webgpu`, `burn-wgpu` | native Burn, WebGPU | any Vulkan/Metal/DX12 GPU |

The alias set is identical across `rvc`, `stt`, `tts` and `voice`. Naming a
backend or device that is not available is an **error with a reason**, never a
silent fallback — only `auto` substitutes.

An ONNX repo (`onnx-community/whisper-large-v3-turbo`) is the two-graph
`optimum` export; a Burn repo is the first-party `openai/…` one, preferred there
because mirrors of converted weights move and disappear.

## Recipes

**Transcribe a directory, one transcript beside each take.** This is also how a
`tts train` corpus gets its `<stem>.txt` files.

```sh
fd -e wav . corpus -j 1 -x sh -c \
  'ffmpeg -v quiet -i "$1" -f f32le -ar 16000 -ac 1 - | stt > "$2"' _ {} {.}.txt
```

**`-j 1` is load-bearing**: without it a second `stt` loads a second copy of the
model onto the same GPU.

**Subtitles and manifests — `--format jsonl`.** Adds the segment's position in
the recording and the language the model used. `start` and `end` are the
*slicer's* span, in seconds: decoding runs with `<|notimestamps|>`, since the
boundaries are already known from the audio.

```sh
ffmpeg -v quiet -i take.mp3 -f f32le -ar 16000 -ac 1 - | stt --format jsonl
```
```json
{"start":2.300,"end":4.760,"language":"zh","text":"今天天气很好"}
{"start":5.120,"end":7.480,"language":"zh","text":"这是一段示例录音"}
```

**A soft, close-mic recording.** Keep the quiet passages and be slower to cut:

```sh
ffmpeg -v quiet -i asmr.mp3 -f f32le -ar 16000 -ac 1 - \
  | stt --silence-db=-50 --min-silence 0.8 --language zh --format jsonl
```

**Recognise, then re-synthesise in another voice.** Every stage is a filter, so
the toolkit's pipeline is a literal shell pipe:

```sh
ffmpeg -v quiet -i take.mp3 -f f32le -ar 16000 -ac 1 - \
  | stt \
  | tts --reference clip.wav --reference-text "the transcript of clip.wav" --sr 16000 \
  | rvc -m models/voice.safetensors --model-sr 48000 \
  > out.f32le
```

**The differential check**, which is the strongest test this crate has. Decoding
is greedy, so the two runtimes are deterministic and must agree exactly; run it
after touching `burn-whisper`, `mel.rs` or `decode.rs`.

```sh
stt --repo openai/whisper-large-v3-turbo         --backend tch  < take.f32le > burn.txt
stt --repo onnx-community/whisper-large-v3-turbo --backend onnx < take.f32le > ort.txt
diff burn.txt ort.txt      # must be empty
```

**If a download stalls**, fetch the repo by other means (`git clone`,
`hf download`, a browser) and point `--model` at the directory. Large Hugging
Face fetches have been observed to hang partway and the in-process downloader
does not resume a partial blob, so the next run would otherwise start over.

## Where files land

Whisper's weights go to the shared cache — `--cache-dir` or `$STT_CACHE_DIR`,
resolved as [`docs/setup.md`](../../docs/setup.md#where-models-are-stored)
describes and printed by `-h`. Transcripts go where you point stdout, and
nothing else is written.

`stt download` fetches them without transcribing anything, which is how a
machine that will be offline later — or a batch job that must not spend its
first minutes downloading — gets set up. It prints the directory it filled, and
that is exactly what `--model` takes:

```sh
stt --model "$(stt download | sed -n 's/^whisper: //p')" < take.f32le
```

## Status and limits

Whisper large-v3-turbo runs end to end and loads at **587 / 0 / 0**, identically
on every backend, and the two runtimes produce **byte-identical transcripts from
the same weights** — which is how the Burn port is validated against an
independent implementation rather than against itself.

Not yet here: word-level timestamps, so the timings in `--format jsonl` are
segment-granular; beam search or temperature fallback, since greedy decoding is
what makes the differential check exact; batched segments; and fine-tuning — the
Burn port is a complete model, so a training loop would be a `train-kit` job
rather than new network code.

## See also

- [repository README](../../README.md) — what `voice` is and how the engines compose
- [`docs/setup.md`](../../docs/setup.md) — ffmpeg, ONNX Runtime, LibTorch
- [`rvc`](../rvc-cli/README.md) and [`tts`](../tts-cli/README.md) — the other two engines
- [`CLAUDE.md`](../../CLAUDE.md) — why the code is shaped the way it is
