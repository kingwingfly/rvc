# Realtime — playing, capturing, and virtual microphones

Everything true of driving the engines **live** rather than over files: which
sample rate each end of a pipe is carrying, how to reach a microphone and a
speaker, how to make `rvc` or `seedvc` appear to OBS and Discord as a capture
device, and how much delay to expect from each engine.

Per-flag detail belongs to the engine manuals ([`rvc`](../crates/rvc-cli/README.md),
[`stt`](../crates/stt-cli/README.md), [`tts`](../crates/tts-cli/README.md),
[`seedvc`](../crates/seedvc-cli/README.md)); toolchain and driver setup is
[`setup.md`](setup.md). This page is the part that would otherwise be written
into five READMEs and drift.

- [There is no audio-device code, on purpose](#there-is-no-audio-device-code-on-purpose)
- [The rates](#the-rates)
- [Capture and playback with ffmpeg](#capture-and-playback-with-ffmpeg)
- [A virtual microphone](#a-virtual-microphone)
- [OBS](#obs)
- [The latency budget](#the-latency-budget)

## There is no audio-device code, on purpose

No crate in this workspace links `cpal`, `pipewire`, `libpulse`, ALSA, JACK or
any other audio API. Every engine reads raw mono `f32le` PCM on stdin and writes
it on stdout, and **capture and playback are delegated entirely to `ffmpeg` and
`ffplay`**.

That is the design rather than a gap. The whole conversion path is
`futures::Stream`-in → `futures::Stream`-out, so a bare invocation is a plain
Unix filter and gains a microphone the same way `grep` gains one — from whatever
is on the left of the pipe. It costs one extra process and buys three things: the
engines stay portable with no per-platform audio backend to maintain, ffmpeg
already speaks every capture API on every platform, and anything that can write
PCM to a pipe is a valid source. A native backend is
[on the roadmap](roadmap.md#a-native-audio-device-backend) as an *option*, not as
a replacement.

The consequence to keep in mind is that **nothing in the toolkit knows what
sample rate the process on the other side of the pipe expects.** Raw PCM carries
no header, so a mismatch is not an error — it is playback at the wrong speed and
pitch. That is what the next section is for.

## The rates

**Every engine reads 16 kHz.** They differ on the way out, and the differences
are real: 48 kHz out of `rvc` fed to a player told 22.05 kHz plays at less than
half speed and nothing warns.

| | stdin | stdout | fixed by |
|---|---|---|---|
| **`rvc`** | mono f32le **16 kHz** — the RVC analysis rate | mono f32le at `--model-sr`, **48000** | the trained model; only 48 kHz is supported today |
| **`stt`** | mono f32le **16 kHz** — Whisper's analysis rate | text, one line or JSONL per segment | — |
| **`tts`** | text, one line per utterance | mono f32le **32000** | GPT-SoVITS's `s2`; `--sr` resamples in-process |
| **`seedvc`** | mono f32le **16 kHz** — the content encoder's rate | mono f32le **22050** | BigVGAN's rate, and not `rvc`'s |

Logs go to stderr on all four, so stdout only ever carries the payload.

`--sr` on `tts` is what lets synthesis feed conversion with no intermediate file,
because the resample happens in-process rather than through a third ffmpeg:

```sh
tts -r clip.wav -t "<what clip.wav actually says>" --sr 16000 < script.txt \
  | rvc -m models/voice.safetensors --model-sr 48000 \
  | ffplay -f f32le -ar 48000 -ac 1 -
```

The four-stage pipeline, which is the toolkit's shape in one command — note each
stage naming the rate the stage before it produces:

```sh
ffmpeg -v quiet -i take.mp3 -f f32le -ar 16000 -ac 1 - \
  | stt \
  | tts --reference clip.wav --reference-text "<what clip.wav actually says>" --sr 16000 \
  | rvc -m models/voice.safetensors --model-sr 48000 \
  > out.f32le
```

## Capture and playback with ffmpeg

Playback is always the same shape — `ffplay` told the format, the rate and the
channel count, reading `-`:

```sh
... | ffplay -f f32le -ar 48000 -ac 1 -     # rvc
... | ffplay -f f32le -ar 32000 -ac 1 -     # tts
... | ffplay -f f32le -ar 22050 -ac 1 -     # seedvc
```

Capture is the mirror image: an input device instead of `-i in.mp3`, with the
output side pinned to 16 kHz mono because that is what every engine reads.

| platform | live input |
|---|---|
| Linux, PipeWire or PulseAudio | `ffmpeg -f pulse -i default -f f32le -ar 16000 -ac 1 -` |
| Linux, ALSA directly | `ffmpeg -f alsa -i default -f f32le -ar 16000 -ac 1 -` |
| macOS | `ffmpeg -f avfoundation -i :0 -f f32le -ar 16000 -ac 1 -` |
| Windows | `ffmpeg -f dshow -i audio="<device name>" -f f32le -ar 16000 -ac 1 -` |

`ffmpeg -sources pulse`, `ffmpeg -f avfoundation -list_devices true -i ""` and
`ffmpeg -list_devices true -f dshow -i dummy` print what the names are on the
machine in front of you. Only the Linux rows were run here; the macOS and Windows
device syntax is ffmpeg's documented form and is **untested by this project**.

So a live microphone through voice conversion into the speakers is:

```sh
ffmpeg -v quiet -f pulse -i default -f f32le -ar 16000 -ac 1 - \
  | rvc -m models/voice.safetensors --model-sr 48000 --backend tch \
  | ffplay -v quiet -f f32le -ar 48000 -ac 1 -nodisp -
```

Read [the latency budget](#the-latency-budget) before expecting that to be
usable for conversation.

## A virtual microphone

**This already works and needs no code from us.** The operating system's own
sound server can present a named pipe as a capture device, so redirecting an
engine's stdout into that pipe *is* a virtual microphone — one that OBS, Discord,
a browser and every other application see beside the real ones.

### Linux — PipeWire or PulseAudio

`pactl` drives both: PipeWire ships a PulseAudio-compatible server, so the same
command works either way. The `Server Name` line of `pactl info` says which one
is answering.

```sh
# Create the device. The module makes the FIFO itself; the id it prints is the
# handle for removing it again.
pactl load-module module-pipe-source \
    source_name=voice file=/tmp/voice.pipe \
    format=float32le rate=48000 channels=1

pactl list sources short          # `voice` is now in the list
```

```sh
# Feed it. This is the ordinary filter pipeline with a redirect instead of ffplay.
ffmpeg -v quiet -f pulse -i default -f f32le -ar 16000 -ac 1 - \
  | rvc -m models/voice.safetensors --model-sr 48000 --backend tch \
  > /tmp/voice.pipe
```

```sh
pactl unload-module <id>          # tear it down
```

**`rate=` and `channels=` must match the engine's output exactly**, because the
module resamples nothing — it reinterprets whatever bytes arrive as the format it
was given. That is `rate=48000` for `rvc` and `rate=22050` for `seedvc`, from
[the table above](#the-rates), and `channels=1` for both. Getting it wrong
produces a device that works and sounds wrong.

**The engine stalls while nothing is capturing from the device.** A FIFO holds
about 64 KiB, which at 48 kHz mono `f32` is a third of a second of audio; once it
is full the write blocks, and the sound server only drains it while some
application is actually recording from the source. So the pipeline appears to
hang until OBS — or `parecord`, or anything else — opens the device, and then
runs normally. Start the consumer first, or expect the first third of a second to
sit there. This is FIFO semantics rather than a defect in anything, but it looks
exactly like a hung model, which is why it is written down.

**Once a client *is* capturing, a slow engine underruns instead of blocking**,
and the device fills the gap with silence rather than stretching time. That is
the opposite failure to the stall above and the far more likely one in practice,
because it is what a generator that cannot keep up looks like from OBS's side:
not an error, not a dropout you can hear as a glitch, just an output that is
quietly part silence. The most visible case is startup — `rvc` spends its first
several seconds loading the model and opening its ORT sessions before it emits a
single sample, and every one of those seconds is silence on the device. **Start
the pipeline, wait for its "stdin … -> stdout …" line on stderr, and only then
go live.**

**What was tested, and how.** All of the above was run end to end on Arch Linux
against PipeWire 1.6.8, whose `pactl info` answers
`Server Name: PulseAudio (on PipeWire 1.6.8)`. The module loaded, it created the
FIFO itself, `pactl list sources short` reported
`voice … float32le 1ch 48000Hz`, and audio written into the FIFO came back out of
a recording taken *from the device* — mono `f32le` at 48 kHz at a plausible level
rather than silence. Unloading the module removed the source from the list again.

That was then repeated with **the real engine rather than a test tone**: a corpus
clip decoded to 16 kHz, piped through `rvc -m … --model-sr 48000 --backend tch`
on `libtorch<cuda>`, redirected into the FIFO, and captured from the `voice`
device. The pipeline exited 0 and the converted audio arrived at the device. It
also demonstrated the underrun above unmistakably: the engine logged 18 s of
model loading before its first sample, and the server logged a continuous
`underrun 0 < 8192` for exactly that period.

Two more things that are not obvious from the commands. **A session manager
(`wireplumber` or equivalent) must be running** or the source is never drained at
all — on any desktop it already is, but it is the reason this fails on a bare
`pipewire` with no session. And the stall above was reproduced deliberately: with
nothing capturing, a writer feeding 10 s of audio was still blocked 60 s later.

### macOS — BlackHole

**Untested by this project**; this is the documented shape of the tool rather
than a measurement. [BlackHole](https://github.com/ExistentialAudio/BlackHole) is
a virtual audio driver that installs as both an output and an input device, so
the route is to play into it rather than to write a pipe:

```sh
ffmpeg -f avfoundation -i :0 -f f32le -ar 16000 -ac 1 - \
  | rvc -m models/voice.safetensors --model-sr 48000 \
  | ffmpeg -f f32le -ar 48000 -ac 1 -i - -f audiotoolbox -audio_device_index <n> -
```

`ffmpeg -f audiotoolbox -list_devices true -i ""` gives the index of the
BlackHole output. Anything then recording from the BlackHole *input* hears the
converted voice. There is no macOS equivalent of `module-pipe-source`, which is
why this route goes through a device rather than a FIFO.

### Windows — VB-Cable

**Untested by this project.**
[VB-Cable](https://vb-audio.com/Cable/) installs a paired playback/recording
device with the same shape as BlackHole: play into `CABLE Input`, and
applications record from `CABLE Output`.

```sh
ffmpeg -f dshow -i audio="<microphone>" -f f32le -ar 16000 -ac 1 - ^
  | rvc -m models/voice.safetensors --model-sr 48000 ^
  | ffplay -f f32le -ar 48000 -ac 1 -nodisp -
```

`ffplay` has no device-selection flag, so routing its output to `CABLE Input`
means making that the default playback device, or using a player that can be
pointed at one. The `^` line continuations are `cmd.exe`; PowerShell uses a
backtick and any POSIX shell uses `\`.

## OBS

Once a virtual capture device exists, OBS needs nothing special: **Sources → +
→ Audio Input Capture**, and pick `voice` (or `BlackHole`, or `CABLE Output`).
The engine is upstream of everything OBS knows about, so filters, the mixer and
recording all behave as they would with a real microphone.

Two things worth knowing before wiring a stream around it:

- **Add the engine's delay to OBS's own.** OBS does not know the audio it is
  given is late, so lip-sync against a camera needs the video delayed to match —
  a *Render Delay* filter on the video source, set from
  [the budget below](#the-latency-budget).
- **A dead pipeline is silence, not an error.** If the engine exits, the device
  stays in OBS's list and produces nothing. Check the pipeline's stderr, which is
  where every engine's logs go and which a redirect into a FIFO does not touch.

There is no OBS *plugin* here and none is planned; a capture device is the whole
integration, and it is one an OBS upgrade cannot break.

## The latency budget

**Every engine buffers a block, not a sample.** Each stage is a window rather
than a filter, so nothing can come out until a whole block has gone in, and that
block length is the floor under the delay no matter how fast the hardware is.
Model time sits on top of it.

| | buffering floor | on top of that |
|---|---|---|
| **`rvc`** | **0.5 s** — `StreamParams::realtime`'s block, with 0.25 s of look-back re-analysed per block and a 0.05 s crossfade | one block of model time; `--backend tch` is ~9x faster per file than CubeCL/CUDA on an RTX 2060, and the CubeCL generator is still slower than realtime for the streaming filter |
| **`stt`** | one **voiced segment** — the slicer finalises a run once `max(min_silence, 2·pad)` of silence has followed it | one decode of that segment |
| **`tts`** | one **utterance** — a line of stdin, flushed as soon as it is synthesised | `s1` sampling plus `s2` decode for that line |
| **`seedvc`** | **2.2 s** — `StreamParams::realtime` is 172 frames, so 2.0 s of new audio per chunk plus the crossfade | one chunk of model time, and **the model is slower than realtime on the hardware this was developed against** |

Read the last row as the engine's own README states it: **`seedvc` is not
realtime today.** Measured on an RTX 2060 with `--backend tch --device gpu` at
the default `--steps 30 --guidance 0.7`, the batch path converted 7.79 s of audio
in 10.9 s. What the `realtime` preset buys there is that the audio which does
come out comes out in 2 s steps rather than after the whole recording — a live
pipe still falls behind, without bound, for as long as it runs. `--guidance 0`
halves the transformer evaluations per step and `--steps 10` cuts them again;
those two knobs are what move it.

`seedvc`'s `--chunk` does **not** move the figure — it is only how much stdin is
read at a time. The window arithmetic that does set it is `seedvc_core::stream`'s.

For `rvc`, 0.5 s plus model time is the honest number to design around, and it is
low enough for streaming-with-a-delay while being too high for conversation.
Whether it is usable live comes down to whether the generator keeps up on the
machine, and **falling behind looks like two different faults depending on what
is downstream.** Into a pipe — `ffplay`, a file, another engine — the reader
applies backpressure and the *delay* grows without bound, because nothing here
drops audio to catch up. Into a
[virtual capture device](#a-virtual-microphone) there is no backpressure to
apply, so the device underruns and substitutes *silence* instead, and the delay
never grows because the audio simply is not there. Watching the first minute is
the test either way — a pipeline that is going to fall behind starts doing so
immediately.

`tts` releasing per utterance is a property to rely on rather than a coincidence:
a script of lines produces speech line by line rather than at the end.
[`CLAUDE.md`](../CLAUDE.md) records why that was once broken and invisible —
tokio's `BufWriter` bypasses its own buffer for any write at or above its 8 KiB
capacity, so a missing flush stranded only the sub-8 KiB tail and looked like
streaming at realistic sample rates.
