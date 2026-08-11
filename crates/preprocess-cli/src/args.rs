//! Command-line argument definitions (clap derive).
//!
//! The shape differs from every engine's in exactly one way, and the reason is
//! in [`crate`]'s own docs: there is **no bare invocation**, because
//! `preprocess` names a stage rather than an engine and which stage is what the
//! subcommand picks. Everything else is the house shape — one clap type for the
//! whole tree, `completions` added by whichever `main` is hosting it.

use std::path::PathBuf;

use anyhow::Result;
use clap::{Args, Subcommand, ValueEnum};

// `Backend` arrived with `separate`, the first stage that runs a model. It is
// `cli_kit`'s, like every other binary's — there is one such enum for the whole
// workspace and a second must not appear beside it.
pub use cli_kit::{Backend, CompletionsArgs};

/// The whole of corpus preparation, defined once and worn two ways: the
/// `preprocess` binary flattens it at its top level, `voice` nests it under a
/// `preprocess` subcommand.
///
/// `completions` is deliberately not in here — it describes the *binary* being
/// completed, not the stages, so each `main` adds it beside this. See
/// [`PreprocessCommand`].
#[derive(Debug, Args)]
pub struct PreprocessCli {
    #[command(subcommand)]
    pub command: PreprocessCommand,
}

/// The stages, and nothing about the executable that hosts them.
///
/// **Not an `Option`, unlike every engine's**: there is nothing a bare
/// `preprocess` could do. See [`crate`] for why that is not the streaming rule
/// being broken.
///
/// A new stage is one module in `preprocess-core`, one module under
/// `commands/`, and one variant here. That is the whole extension point, and it
/// is the reason this crate exists rather than a second subcommand on two
/// engines that neither of them is about.
#[derive(Debug, Subcommand)]
pub enum PreprocessCommand {
    /// Slice recordings into clean per-utterance clips (dead-air removed).
    Clip(ClipArgs),
    /// Remove steady background hiss from recordings.
    Denoise(DenoiseArgs),
    /// Split recordings into a voice stem and a music stem (removes a backing
    /// track).
    // The long help sits on the *variant*, not on `SeparateArgs`: clap builds a
    // subcommand's `about`/`long_about` from the enum, and a `long_about` on the
    // args struct is silently ignored — the command then answers `--help` with
    // the one-line summary and everything that makes the stage honest goes
    // unread.
    #[command(long_about = SEPARATE_LONG_ABOUT)]
    Separate(SeparateArgs),
    /// Keep only the parts spoken by one voice, given a reference clip.
    // Same placement and the same reason as `separate`'s above.
    #[command(long_about = DIARIZE_LONG_ABOUT)]
    Diarize(DiarizeArgs),
    /// Report what a corpus is — noise floor, SNR, peaks, silence — and what
    /// to set the other stages to.
    // Same placement and the same reason as the two above.
    #[command(long_about = ANALYZE_LONG_ABOUT)]
    Analyze(AnalyzeArgs),
}

/// What every stage takes: files to read and a directory to write into.
///
/// Flattened rather than repeated, so the input handling cannot differ between
/// two stages that are meant to compose — the output of one is a legitimate
/// input to the next, and that only holds while they agree on what an input is.
///
/// `-o` is deliberately *not* here: every stage has one, but each names its own
/// default after what it produces, and a shared default would have to be one of
/// them or a directory that describes nothing.
#[derive(Debug, Args, Clone)]
pub struct IoArgs {
    /// Input audio files and/or directories (directories are expanded
    /// recursively to their audio files: mp3, wav, flac, m4a, ogg, opus, aac,
    /// wma).
    #[arg(required = true)]
    pub input: Vec<PathBuf>,
    /// Sample rate of the written audio. A later stage re-decodes at whatever
    /// rate it needs, so this only decides what is on disk; matching the rate
    /// you will train at avoids a resample.
    #[arg(long, alias = "model-sr", default_value_t = 48000)]
    pub sr: u32,
}

impl IoArgs {
    /// Reject a rate nothing could be written at.
    pub fn verify(&self) -> Result<()> {
        anyhow::ensure!(self.sr > 0, "--sr must be positive");
        Ok(())
    }
}

/// Where a slicer cuts.
///
/// Flattened by `clip`, which does the cutting, and by `analyze`, which reports
/// what that cutting would produce. One definition for the same reason `IoArgs`
/// is one: the numbers `analyze` prints have to come from the very knobs `clip`
/// will act on, and two copies of five defaults would drift the first time one
/// of them moved.
#[derive(Debug, Args, Clone)]
pub struct SliceArgs {
    /// Energy floor in dBFS: audio quieter than this counts as between-sentence
    /// dead-air. Lower it (e.g. -50) to keep the very softest passages —
    /// energy is used only to find silent gaps, never to gate quiet content.
    #[arg(long, default_value_t = -40.0)]
    pub silence_db: f32,
    /// Minimum silent-gap length (seconds) that counts as a sentence boundary.
    /// Shorter pauses stay inside the clip, so complete sentences are never
    /// split.
    #[arg(long, default_value_t = 0.3)]
    pub min_silence: f32,
    /// Drop any clip shorter than this (seconds).
    #[arg(long, default_value_t = 1.0)]
    pub min_clip: f32,
    /// Hard cap on clip length (seconds); 0 means never split a long sentence.
    #[arg(long, default_value_t = 0.0)]
    pub max_clip: f32,
    /// Edge-pad each clip by up to this many seconds of bordering quiet so
    /// onsets and soft breathy tails are not clipped.
    #[arg(long, default_value_t = 0.15)]
    pub pad: f32,
}

impl SliceArgs {
    /// Reject a slicer configuration that would silently produce no clips.
    pub fn verify(&self) -> Result<()> {
        anyhow::ensure!(
            self.silence_db <= 0.0,
            "--silence-db is dBFS, so it must be at most 0 (full scale); {} would \
             treat every sample as silence",
            self.silence_db
        );
        anyhow::ensure!(
            self.min_silence > 0.0,
            "--min-silence must be positive: a zero-length gap is every sample boundary"
        );
        anyhow::ensure!(self.min_clip >= 0.0, "--min-clip must not be negative");
        anyhow::ensure!(self.pad >= 0.0, "--pad must not be negative");
        // `0` is the documented "never split" sentinel, so it is the one value
        // allowed below `--min-clip`.
        anyhow::ensure!(
            self.max_clip == 0.0 || self.max_clip >= self.min_clip,
            "--max-clip ({}) is below --min-clip ({}), so every clip would be cut \
             to a length that is then discarded (use 0 to never split)",
            self.max_clip,
            self.min_clip
        );
        Ok(())
    }

    /// Where these flags say to cut.
    pub fn options(&self) -> audio_kit::SliceOptions {
        audio_kit::SliceOptions {
            silence_db: self.silence_db,
            min_silence: self.min_silence,
            min_clip: self.min_clip,
            max_clip: self.max_clip,
            pad: self.pad,
        }
    }
}

/// Slice a corpus into clean per-utterance clips.
#[derive(Debug, Args)]
pub struct ClipArgs {
    #[command(flatten)]
    pub io: IoArgs,
    /// Directory to write the sliced `<stem>_<NNN>.wav` clips into.
    #[arg(short = 'o', long, default_value = "dataset")]
    pub output_dir: PathBuf,
    #[command(flatten)]
    pub slice: SliceArgs,
    /// Peak-normalize each written clip to ~0.95 full-scale.
    #[arg(long)]
    pub normalize: bool,
}

impl ClipArgs {
    /// Reject a slicer configuration that would silently produce no clips.
    pub fn verify(&self) -> Result<()> {
        self.io.verify()?;
        self.slice.verify()
    }

    /// The stage settings these flags describe.
    pub fn options(&self) -> preprocess_core::clip::ClipOptions {
        preprocess_core::clip::ClipOptions {
            sr: self.io.sr,
            slice: self.slice.options(),
            normalize: self.normalize,
        }
    }
}

/// De-hiss a corpus.
///
/// There is no `--denoise` switch to go with the tuning flags, unlike voice
/// conversion's: here the stage *is* the subcommand, so a flag turning it off
/// would leave a command that copies files.
#[derive(Debug, Args)]
pub struct DenoiseArgs {
    #[command(flatten)]
    pub io: IoArgs,
    /// Directory to write the de-hissed `<stem>.wav` files into. Not the input
    /// directory by default, because a stage that overwrote its own input would
    /// make a bad `--denoise-strength` unrecoverable.
    #[arg(short = 'o', long, default_value = "denoised")]
    pub output_dir: PathBuf,
    #[command(flatten)]
    pub tuning: cli_kit::DenoiseOpts,
}

impl DenoiseArgs {
    /// Reject a de-hiss configuration `anlmdn` would refuse or misbehave on.
    pub fn verify(&self) -> Result<()> {
        self.io.verify()?;
        self.tuning.verify()
    }

    /// The stage settings these flags describe.
    pub fn options(&self) -> preprocess_core::denoise::DenoiseOptions {
        preprocess_core::denoise::DenoiseOptions {
            sr: self.io.sr,
            params: self.tuning.params(),
        }
    }
}

/// Which half of a recording is written out.
///
/// A clap mirror of [`preprocess_core::separate::Stems`], for the reason
/// [`ClipArgs::options`] maps its flags one at a time: the stage's types stay
/// free of `clap`, and the flag's spelling stays here where the rest of the
/// command line is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum Stem {
    /// The voice. What a corpus wants.
    Vocals,
    /// The music that was removed.
    Instrumental,
    /// Both files. Free but for the disk: one pass produces the pair.
    Both,
}

/// What `separate --help` says beyond its one-line summary.
///
/// Every number in it is measured rather than claimed, and the order is the
/// order a user needs them in: what to do with the stage first, what it is
/// worth second, and what it does *not* do last — because the last is the one
/// somebody will otherwise assume.
const SEPARATE_LONG_ABOUT: &str = "\
Split recordings into a voice stem and a music stem — for a corpus recorded \
over a backing track, which nothing downstream can handle until the music is \
off it.

Run this BEFORE `clip`. Slicing cuts on silence, and a continuous music bed \
means the recording has none: on a 60 s excerpt of a stream, slicing the \
mixture gave 5 clips holding 58 of its 60 seconds, and slicing the vocals stem \
gave 12 holding 39 — sentences rather than minutes. Recognition tells the same \
story, 5 segments against 14 with the same words. That is the gain, and it is \
larger than the decibels suggest.

The separation itself is modest and measured: 5-6.5 dB of a continuous bed \
comes out of the vocals stem across the excerpts the model was measured on, and \
4.1 dB through this stage — read on the mono downmix, which is what everything \
downstream decodes — against the 15-20 dB this model reaches on a song, because \
it was trained on sung vocals and a speaking voice is not one. Expect the music \
attenuated, not gone.

It separates voice from music, NOT one speaker from another. Every voice lands \
in the vocals stem, including a singer on the backing track — so somebody \
talking over another person's singing still gets both. Keeping one speaker is a \
different model and a stage of its own — run `diarize` on these stems \
afterwards; this one cannot do it and does not try.

Emptying the gaps also gives a recogniser room to invent in them, and can leave \
very short fragments, so whatever consumes these stems wants a duration floor.

The stems are written at the model's own 44.1 kHz — hence no --sr here — and \
in stereo, because the model is stereo-native and folding it away would throw \
out one of the two cues it separates on.";

/// Split recordings into a voice stem and a music stem.
///
/// Deliberately without `--sr`, which every other stage has: the separation
/// model's rate is not a choice, so the stems are written at its own 44.1 kHz.
/// Writing them at anything else would be a resample on top of a separation,
/// and the next stage's decode performs one anyway.
#[derive(Debug, Args)]
pub struct SeparateArgs {
    /// Input audio files and/or directories (directories are expanded
    /// recursively to their audio files: mp3, wav, flac, m4a, ogg, opus, aac,
    /// wma).
    #[arg(required = true)]
    pub input: Vec<PathBuf>,
    /// Directory to write the `<stem>.vocals.wav` / `<stem>.instrumental.wav`
    /// stems into.
    #[arg(short = 'o', long, default_value = "separated")]
    pub output_dir: PathBuf,
    /// Which stem to write. Both come out of the same pass, so `both` costs
    /// only the second file.
    #[arg(long, value_enum, default_value_t = Stem::Vocals)]
    pub stem: Stem,
    /// Separation checkpoint (MDX23C) [default: auto-downloaded from Hugging
    /// Face].
    #[arg(short, long)]
    pub model: Option<PathBuf>,
    /// Directory the downloaded model is cached in. Shared by every engine
    /// unless `$VOICE_CACHE_DIR` says otherwise.
    #[arg(long, default_value_os_t = hub_kit::default_cache_dir())]
    pub cache_dir: PathBuf,
    /// Inference backend: `cuda` (aliases `burn`, `burn-cuda`), `tch`
    /// (`libtorch`, `burn-tch`) or `wgpu` (`webgpu`, `burn-wgpu`); `auto` picks
    /// the fastest compiled in. There is no `onnx`: this model is published as
    /// a PyTorch checkpoint only.
    #[arg(long, value_enum, default_value_t = Backend::Auto)]
    pub backend: Backend,
    /// Compute device: `auto` (fastest visible), `cpu`, `gpu`, `gpu:N`, `mps` or
    /// `vulkan` (`cuda`/`cuda:N` also accepted). The `cuda` backend has GPUs only.
    #[arg(long, default_value = "auto", value_name = "DEVICE", value_parser = cli_kit::parse_device)]
    pub device: burn_kit::DeviceSpec,
    #[command(flatten)]
    pub download: cli_kit::DownloadOpts,
}

impl SeparateArgs {
    /// Reject what cannot run, **before** the 448 MB the model weighs.
    ///
    /// Both checks are here for that reason and not for tidiness: a backend
    /// this build cannot run and a download policy that abandons every transfer
    /// are equally cheap to notice now and equally expensive to notice after
    /// the fetch.
    pub fn verify(&self) -> Result<()> {
        self.download.verify()?;
        preprocess_core::backend::resolve(self.backend)?;
        Ok(())
    }

    /// The stage settings these flags describe.
    pub fn options(&self) -> preprocess_core::separate::SeparateOptions {
        preprocess_core::separate::SeparateOptions {
            stems: match self.stem {
                Stem::Vocals => preprocess_core::separate::Stems::Vocals,
                Stem::Instrumental => preprocess_core::separate::Stems::Instrumental,
                Stem::Both => preprocess_core::separate::Stems::Both,
            },
        }
    }
}

/// What `diarize --help` says beyond its one-line summary.
///
/// Same placement and the same reason as [`SEPARATE_LONG_ABOUT`], and the same
/// rule about its contents: every number in it is measured, and the order is
/// what a user needs first — where the stage goes in the pipeline, then what
/// the output actually looks like, then the thing that most often makes it
/// perform badly.
const DIARIZE_LONG_ABOUT: &str = "\
Keep only the parts of a recording spoken by one voice.

Each window of the input is compared to a reference clip by speaker \
similarity, and runs of windows that clear --threshold are written out as \
<stem>_<NNN>.wav. Source separation cannot do this on its own: a vocals stem \
holds every voice in the mixture, the person talking and a singer on the \
backing track alike, because separation asks whether something is a voice and \
not whose it is.

So for a stream recorded over somebody else's music, the order is the recipe:

    preprocess separate raw/    -o vocals/ --stem vocals
    preprocess diarize  vocals/ -o mine/   --reference me.wav
    preprocess clip     mine/   -o dataset/

The output is window-quantised, not sample-accurate. Edges land on --hop \
boundaries, a window spanning a speaker change belongs to whoever dominates \
it, and a kept run is widened to cover its last window in full — so a segment \
can carry a fraction of a second of the other voice at each end, and the \
seconds written are always more than the seconds matched. The window counts in \
the per-file line are the honest measure of how much of a recording matched.

TAKE THE REFERENCE FROM THE RECORDING YOU ARE FILTERING. Measured on this \
repository's own corpus, two clips of one speaker from one session score \
0.87-0.90; the same speaker across sessions scores 0.36-0.78. That spread is \
wider than the gap between speakers this threshold works in, because the \
embedding keys partly on the microphone and the room — so a reference from \
somewhere else is the likeliest single reason a run keeps nothing.

At the default 3 s window, a threshold of 0.55 kept 80% of the target's \
windows and rejected 86% of the music-and-singer ones on a separated 60 s of \
stream. At 1.5 s the two stop separating at all, which is the speaker model's \
own 2 s context window showing up as a number.

There is no mode that runs without --reference: this is target-speaker \
extraction, and discovering how many speakers a recording holds is a different \
job that is not built.";

/// Keep one speaker and drop the rest.
///
/// `-r/--reference` is **required**, and that is the honest shape rather than a
/// placeholder: blind clustering — the mode that would need no reference — is
/// not built, and an `Option` that errored on `None` would advertise it.
#[derive(Debug, Args)]
pub struct DiarizeArgs {
    #[command(flatten)]
    pub io: IoArgs,
    /// Directory to write the retained `<stem>_<NNN>.wav` segments into. Not
    /// the input directory by default, for `denoise`'s reason: a stage that
    /// overwrote its own input would make a badly chosen `--threshold`
    /// unrecoverable.
    #[arg(short = 'o', long, default_value = "diarized")]
    pub output_dir: PathBuf,
    /// A recording of the voice to keep — a few clean seconds of that person
    /// alone, from the same recording you are filtering. Only the first 30 s
    /// are read, since the embedding saturates.
    #[arg(short = 'r', long)]
    pub reference: PathBuf,
    /// Cosine similarity to the reference a window must reach to be kept.
    /// Raise it to drop anything doubtful, lower it to keep more of the target
    /// voice at the cost of admitting other speakers.
    #[arg(long, default_value_t = 0.55)]
    pub threshold: f32,
    /// Analysis window in seconds. Shorter follows a speaker change more
    /// closely and gives a noisier embedding; the speaker model pools its
    /// context over 2 s, and below that the distributions stop separating.
    #[arg(long, default_value_t = 3.0)]
    pub window: f32,
    /// Step between windows in seconds. Segment edges land on this grid, so it
    /// is the resolution of the output; halving it doubles the work.
    #[arg(long, default_value_t = 1.0)]
    pub hop: f32,
    /// Drop any retained stretch shorter than this (seconds).
    #[arg(long, default_value_t = 1.0)]
    pub min_segment: f32,
    /// CAM++ speaker-embedding weights [default: auto-downloaded from Hugging
    /// Face].
    #[arg(short, long)]
    pub model: Option<PathBuf>,
    /// Directory the downloaded model is cached in. Shared by every engine
    /// unless `$VOICE_CACHE_DIR` says otherwise.
    #[arg(long, default_value_os_t = hub_kit::default_cache_dir())]
    pub cache_dir: PathBuf,
    /// Inference backend: `cuda` (aliases `burn`, `burn-cuda`), `tch`
    /// (`libtorch`, `burn-tch`) or `wgpu` (`webgpu`, `burn-wgpu`); `auto` picks
    /// the fastest compiled in. There is no `onnx`: this model is published as
    /// a PyTorch checkpoint only.
    #[arg(long, value_enum, default_value_t = Backend::Auto)]
    pub backend: Backend,
    /// Compute device: `auto` (fastest visible), `cpu`, `gpu`, `gpu:N`, `mps` or
    /// `vulkan` (`cuda`/`cuda:N` also accepted). The `cuda` backend has GPUs only.
    #[arg(long, default_value = "auto", value_name = "DEVICE", value_parser = cli_kit::parse_device)]
    pub device: burn_kit::DeviceSpec,
    #[command(flatten)]
    pub download: cli_kit::DownloadOpts,
}

impl DiarizeArgs {
    /// Reject what cannot run, **before** the weights are fetched.
    ///
    /// The backend and download checks are here for `separate`'s reason, and
    /// the window arithmetic joins them because it is equally cheap to notice
    /// now: a `--hop` larger than `--window` skips audio nothing ever scores,
    /// and it does so silently.
    pub fn verify(&self) -> Result<()> {
        self.io.verify()?;
        self.download.verify()?;
        preprocess_core::embed::resolve(self.backend)?;
        anyhow::ensure!(
            self.window > 0.0,
            "--window must be positive: it is the length of audio each speaker \
             comparison is made over"
        );
        anyhow::ensure!(
            self.hop > 0.0 && self.hop <= self.window,
            "--hop ({}) must be positive and no larger than --window ({}), or \
             the step would skip audio nothing ever scores",
            self.hop,
            self.window
        );
        // Cosine similarity is bounded, so a value outside is a unit mistake —
        // a percentage, most likely — and would keep everything or nothing
        // rather than failing.
        anyhow::ensure!(
            (-1.0..=1.0).contains(&self.threshold),
            "--threshold is a cosine similarity and must be between -1 and 1; \
             {} would {} every window",
            self.threshold,
            if self.threshold > 1.0 {
                "reject"
            } else {
                "keep"
            }
        );
        anyhow::ensure!(
            self.min_segment >= 0.0,
            "--min-segment must not be negative"
        );
        Ok(())
    }

    /// The stage settings these flags describe.
    pub fn options(&self) -> preprocess_core::diarize::DiarizeOptions {
        preprocess_core::diarize::DiarizeOptions {
            sr: self.io.sr,
            window: self.window,
            hop: self.hop,
            threshold: self.threshold,
            min_segment: self.min_segment,
        }
    }
}

/// What `analyze --help` says beyond its one-line summary.
///
/// Same placement and the same reason as [`SEPARATE_LONG_ABOUT`]. What it has
/// to say that the others do not is the *negative* space — no model, no
/// backend, no download — because that is what makes it free to run on a whole
/// corpus before deciding anything, and nothing else in this binary is.
const ANALYZE_LONG_ABOUT: &str = "\
Measure a corpus and report what it is: noise floor, speech level, SNR, peak, \
how much of it is dead air, and what `clip` would cut it into.

This stage is MODEL-FREE. No --backend, no --device, no weights, no downloads — \
it decodes the audio and does arithmetic, so it runs over a whole corpus at \
decode speed and can be run before anything is decided.

It writes nothing. The report goes to stdout: --format text to read, --format \
json to drive something with. Per-file lines are opt-in (--per-file); the \
corpus summary is the answer to 'is this corpus usable', and comes last.

The suggested --silence-db is derived from the recordings' own quiet frames, \
and it is CHECKED rather than asserted: the slicer is run twice, once at the \
floor you passed and once at the measured one, and both verdicts are reported. \
That is what makes the last line worth acting on.

If a recording holds no dead air at either floor, no --silence-db can help — \
the quiet is not there to find. Speech over a continuous music bed is that \
shape, and `separate` is what removes the bed; a lower floor cannot. This stage \
says so rather than suggesting a threshold that could not work.

There is no loudness (LUFS) column: ffmpeg reports that through its log rather \
than through the audio, and peak, floor and SNR answer the same question here.";

/// How the report is printed.
///
/// Spelled `--format`, and its text arm spelled `text`, to match the one other
/// command in this workspace that has a choice of output shape (`stt`). `json`
/// rather than `jsonl` because this stage's answer is one document — a corpus
/// summary with its files inside it — and not a stream of records.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, ValueEnum)]
pub enum Format {
    /// Aligned columns for a terminal.
    #[default]
    Text,
    /// One JSON document: `{summary, files}`.
    Json,
}

/// Report what a corpus is.
///
/// `input` and `--sr` are declared here rather than flattened from [`IoArgs`],
/// for `separate`'s reason mirrored: that struct's `--sr` documents "the sample
/// rate of the written audio", and this stage writes none. What the rate
/// decides here is the grid the measurement is taken on.
#[derive(Debug, Args)]
pub struct AnalyzeArgs {
    /// Input audio files and/or directories (directories are expanded
    /// recursively to their audio files: mp3, wav, flac, m4a, ogg, opus, aac,
    /// wma).
    #[arg(required = true)]
    pub input: Vec<PathBuf>,
    /// Sample rate to decode at for the measurement. Nothing is written, so
    /// this only sets the grid; matching the rate you will train at is what
    /// makes the report describe the audio a trainer will see.
    #[arg(long, alias = "model-sr", default_value_t = 48000)]
    pub sr: u32,
    /// The slicer settings to report against: the clip counts, the length
    /// histogram and the silence ratio are what `clip` would produce from these
    /// very flags.
    #[command(flatten)]
    pub slice: SliceArgs,
    /// Output format.
    #[arg(long, value_enum, default_value_t = Format::Text)]
    pub format: Format,
    /// Print a line per file as well as the corpus summary. The summary is the
    /// answer for a corpus of hundreds; this is for the handful you then go
    /// looking at.
    #[arg(long)]
    pub per_file: bool,
}

impl AnalyzeArgs {
    /// Reject a configuration whose report would describe nothing.
    pub fn verify(&self) -> Result<()> {
        anyhow::ensure!(self.sr > 0, "--sr must be positive");
        // The same checks `clip` makes, because these are the same flags and
        // the report would otherwise print what an invalid `clip` invocation
        // "would produce".
        self.slice.verify()
    }

    /// The stage settings these flags describe.
    pub fn options(&self) -> preprocess_core::analyze::AnalyzeOptions {
        preprocess_core::analyze::AnalyzeOptions {
            sr: self.sr,
            slice: self.slice.options(),
        }
    }
}
