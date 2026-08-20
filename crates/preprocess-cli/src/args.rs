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
    /// Match recordings to one level, by peak or by EBU R128 loudness.
    #[command(long_about = NORMALIZE_LONG_ABOUT)]
    Normalize(NormalizeArgs),
    /// Strip leading and trailing silence, without splitting the recording.
    #[command(long_about = TRIM_LONG_ABOUT)]
    Trim(TrimArgs),
    /// Convert recordings to WAV at one sample rate.
    #[command(long_about = RESAMPLE_LONG_ABOUT)]
    Resample(ResampleArgs),
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
    // `allow_negative_numbers` because this flag's value is *always* negative
    // and clap would otherwise read `-50` as a cluster of short flags. See
    // `cli_kit`'s module docs.
    #[arg(long, allow_negative_numbers = true, default_value_t = -40.0)]
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
mixture gave 5 clips holding 58.2 of its 60 seconds, and slicing the vocals \
stem gave 14 holding 35.9 — sentences rather than minutes. Recognition tells \
the same story, 5 segments against 17. That is the gain, and it is larger than \
the decibels suggest.

The separation itself is measured rather than promised: 10-12 dB of a \
continuous bed comes out of the vocals stem, and 4 dB where the bed is \
intermittent because there is less of it in the gaps to take out — read on the \
mono downmix, which is what everything downstream decodes. Expect the music \
strongly attenuated, not gone.

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
    // `allow_negative_numbers` because a cosine similarity starts at -1 and
    // `verify` accepts the whole of that range, so `--threshold -0.2` is a
    // legal invocation — and without this clap reads `-0.2` as a cluster of
    // short flags and answers `unexpected argument '-0'`, which reads as a
    // shell-quoting problem and is not one. A negative threshold is what
    // somebody reaches for when a reference keeps nothing and they want the
    // whole distribution written out before re-picking one. See `cli_kit`'s
    // module docs.
    #[arg(long, allow_negative_numbers = true, default_value_t = 0.55)]
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
    /// and it does so silently. `--reference` joins them for the same reason
    /// and is checked first, being the cheapest of the lot.
    pub fn verify(&self) -> Result<()> {
        self.io.verify()?;
        // The one check here that touches the filesystem, and it earns that:
        // the reference is the entire specification of what to keep, and a
        // mistyped path was being reported *after* the speaker model had been
        // fetched and loaded — a quarter of a gigabyte and a model load spent
        // on a run that could not have started. `separate`'s ordering rule
        // says a refusal that costs nothing belongs before the download, and a
        // path that is not there is exactly that refusal.
        //
        // Existence only, deliberately: whether the file holds decodable audio
        // is a question for the decoder, whose error names the format problem
        // far better than a guess from here could.
        anyhow::ensure!(
            self.reference.is_file(),
            "--reference {} is not a readable file; it is the recording of the \
             voice to keep, so there is nothing to compare against without it",
            self.reference.display()
        );
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

/// What `normalize --help` says beyond its one-line summary.
///
/// Same placement and the same reason as [`SEPARATE_LONG_ABOUT`]: the choice
/// between the two targets is the whole of the stage, and a one-line summary
/// cannot say which one answers which question.
const NORMALIZE_LONG_ABOUT: &str = "\
Match recordings to one level, by applying a single constant gain to each.

--peak and --lufs are two ways of saying what level, and they answer different \
questions:

    --peak 0.95   the loudest sample lands at 95% of full scale. Exact and
                  instant, and blind to everything but that one sample, so a
                  stray thump sets the gain for a whole recording.

    --lufs -23    EBU R128 integrated loudness, which is what 'as loud as each
                  other' means to a listener: K-weighted, and gated so the
                  silence between sentences does not drag the reading down.
                  Unbothered by the thump.

Ask for one or the other, never both. Use peak when you want a known headroom, \
loudness when you want two takes to sit at the same level.

The gain is CONSTANT over each file. Nothing here compresses, limits or rides \
the level, which is why ffmpeg's own loudnorm filter is not what runs: in \
single pass that filter is a dynamic normalizer, and this toolkit exists for \
material whose dynamics are the content. The consequence is that a --lufs run \
can push the peak past full scale on a recording with a wide range. The \
per-file line reports the peak it reached, and it is reported rather than \
limited.

Loudness is measured with ffmpeg's ebur128 scanner, and measured AGAIN after \
the gain, so the number in the report is a reading rather than arithmetic. A \
file with no gated reading at all -- under 0.4 s, or entirely silent -- is \
skipped with that reason, never quietly normalized by peak instead.

No model, no weights, no download: this runs at disk speed over a whole corpus.";

/// What `trim --help` says beyond its one-line summary.
///
/// Placed on the variant for [`SEPARATE_LONG_ABOUT`]'s reason. Its job is to
/// draw the line against `clip`, which is the stage a reader will otherwise
/// assume this one duplicates.
const TRIM_LONG_ABOUT: &str = "\
Strip the silence off the head and tail of a recording. One file in, one file \
out, and never a split: everything between the first voiced sample and the \
last is kept exactly as it was, pauses included.

That is the difference from `clip`, which cuts the same recording into one file \
per sentence. The silence detection is shared, so --silence-db and --pad mean \
here exactly what they mean there; only the treatment of the gaps in the middle \
differs.

--measure-floor is for a recording whose room tone sits ABOVE the fixed floor, \
where -40 dBFS finds no silence to strip at all. It reads the floor off the \
recording's own quiet frames instead. Off by default, because a moved default \
would silently re-cut every corpus prepared so far.

A recording with nothing above the floor is written through UNCHANGED rather \
than emptied. The likeliest cause is a floor set above the whole recording, and \
a stage that answered that by deleting the audio would be indistinguishable \
from one that lost it.

No model, no weights, no download: this runs at disk speed over a whole corpus.";

/// What `resample --help` says beyond its one-line summary.
///
/// Placed on the variant for [`SEPARATE_LONG_ABOUT`]'s reason. The channel
/// paragraph is the one that has to be here: `separate` writes stereo on
/// purpose, and a mono default is a choice a user must be able to see.
const RESAMPLE_LONG_ABOUT: &str = "\
Convert recordings to WAV at one sample rate — mp3, m4a, flac, ogg, opus or \
wav in, <stem>.wav out. Every other stage resamples on the way through because \
every one of them decodes; this is that step with nothing else attached, so a \
corpus can be given one obvious normalising pass.
CHANNELS. Mono by default, because mono is what every engine in this toolkit \
decodes to and a training corpus has no use for a second channel. That matters \
in one place: `separate` writes its stems at 44.1 kHz STEREO on purpose — the \
separation model is stereo-native and folding its output to mono throws away \
one of the two cues it separates on — so piping `separate` into a default \
`resample` discards that second channel. Pass --channels stereo to keep it. \
Nothing downstream in this workspace reads it, so the fold is a loss only if \
something outside does.
--sr is the point of the stage, so it has no clever default: 48000 is what \
voice conversion trains at, 44100 is what `separate` emits, 22050 is Seed-VC's \
vocoder and 16000 is what recognition eats.
No model, no weights, no download: this runs at disk speed over a whole corpus.";

/// Match recordings to one level.
///
/// The two targets are `Option`s rather than one defaulted flag and one
/// override, because "which target" is the question and a default on both would
/// make asking for neither mean something different from asking for peak.
#[derive(Debug, Args)]
pub struct NormalizeArgs {
    #[command(flatten)]
    pub io: IoArgs,
    /// Directory to write the level-matched `<stem>.wav` files into. Not the
    /// input directory by default, for `denoise`'s reason: a stage that
    /// overwrote its own input would make a badly chosen target unrecoverable.
    #[arg(short = 'o', long, default_value = "normalized")]
    pub output_dir: PathBuf,
    /// Peak target as a fraction of full scale: the loudest sample lands here.
    /// This is the default target [default: 0.95].
    // `conflicts_with` rather than a check in `verify`: clap refuses the pair by
    // name, before an argument is read or a file is opened, and its message
    // names both flags — which is the whole requirement.
    #[arg(long, conflicts_with = "lufs")]
    pub peak: Option<f32>,
    /// EBU R128 integrated loudness target in LUFS (negative), e.g. -23 for the
    /// broadcast reference or -16 for a louder corpus.
    // See `cli_kit`'s module docs. Full scale is 0 LUFS, so every usable value here
    // is negative and this flag was unreachable in the form its own help gives.
    #[arg(long, allow_negative_numbers = true)]
    pub lufs: Option<f32>,
}

impl NormalizeArgs {
    /// Reject a target that is not a level.
    pub fn verify(&self) -> Result<()> {
        self.io.verify()?;
        if let Some(peak) = self.peak {
            // Above 1.0 is not headroom, it is a request to clip; at or below 0
            // there is no gain that gets there.
            anyhow::ensure!(
                peak > 0.0 && peak <= 1.0,
                "--peak is a fraction of full scale and must be in (0, 1]; {peak} would {}",
                if peak > 1.0 {
                    "clip every peak it lifted"
                } else {
                    "have no level to normalize to"
                }
            );
        }
        if let Some(lufs) = self.lufs {
            // Full scale is 0 LUFS, so a positive target is a sign mistake and
            // would ask for a gain nothing can hold.
            anyhow::ensure!(
                lufs < 0.0,
                "--lufs is a loudness in LUFS, which is negative below full scale; \
                 {lufs} is at or above it (did you mean {})",
                -lufs.abs()
            );
        }
        Ok(())
    }

    /// The stage settings these flags describe.
    pub fn options(&self) -> preprocess_core::normalize::NormalizeOptions {
        preprocess_core::normalize::NormalizeOptions {
            sr: self.io.sr,
            target: match (self.peak, self.lufs) {
                // Both is refused by clap before this is reached.
                (_, Some(lufs)) => preprocess_core::normalize::Target::Lufs(lufs),
                (Some(peak), None) => preprocess_core::normalize::Target::Peak(peak),
                (None, None) => preprocess_core::normalize::Target::Peak(
                    preprocess_core::normalize::DEFAULT_PEAK,
                ),
            },
        }
    }
}

/// Strip leading and trailing silence.
///
/// The two slicer knobs it carries are spelled exactly as `clip`'s, because
/// they are the same knobs on the same detector. The ones it does not carry are
/// the ones that only mean something when a recording is being cut into pieces:
/// `--min-silence`, `--min-clip` and `--max-clip` all describe interior gaps,
/// and this stage has no interior.
#[derive(Debug, Args)]
pub struct TrimArgs {
    #[command(flatten)]
    pub io: IoArgs,
    /// Directory to write the trimmed `<stem>.wav` files into. Not the input
    /// directory by default, for `denoise`'s reason: a stage that overwrote its
    /// own input would make a badly chosen `--silence-db` unrecoverable.
    #[arg(short = 'o', long, default_value = "trimmed")]
    pub output_dir: PathBuf,
    /// Energy floor in dBFS: audio quieter than this counts as silence at the
    /// edges. Lower it (e.g. -50) to keep the very softest onsets and tails.
    // See `cli_kit`'s module docs.
    #[arg(long, allow_negative_numbers = true, default_value_t = -40.0)]
    pub silence_db: f32,
    /// Keep up to this many seconds of the bordering quiet at each edge, so
    /// onsets and soft breathy tails are not clipped.
    #[arg(long, default_value_t = 0.15)]
    pub pad: f32,
    /// Read the floor off the recording's own quiet frames instead of using
    /// `--silence-db`. For a recording whose room tone is above the fixed
    /// floor, where a fixed pass finds no silence to strip at all.
    #[arg(long)]
    pub measure_floor: bool,
}

impl TrimArgs {
    /// Reject a floor nothing could be quieter than.
    pub fn verify(&self) -> Result<()> {
        self.io.verify()?;
        anyhow::ensure!(
            self.silence_db <= 0.0,
            "--silence-db is dBFS, so it must be at most 0 (full scale); {} would \
             treat every sample as silence",
            self.silence_db
        );
        anyhow::ensure!(self.pad >= 0.0, "--pad must not be negative");
        Ok(())
    }

    /// The stage settings these flags describe.
    pub fn options(&self) -> preprocess_core::trim::TrimOptions {
        preprocess_core::trim::TrimOptions {
            sr: self.io.sr,
            silence_db: self.silence_db,
            pad: self.pad,
            measure_floor: self.measure_floor,
        }
    }
}

/// How many channels [`ResampleArgs`] writes.
///
/// A clap mirror of [`preprocess_core::resample::Channels`], for [`Stem`]'s
/// reason: the stage's types stay free of `clap`, and the flag's spelling stays
/// here with the rest of the command line.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum ChannelCount {
    /// One channel; a stereo input is folded to `(L + R) / 2`.
    Mono,
    /// Two channels; a mono input is written to both.
    Stereo,
}

/// Convert recordings to WAV at one sample rate.
#[derive(Debug, Args)]
pub struct ResampleArgs {
    #[command(flatten)]
    pub io: IoArgs,
    /// Directory to write the converted `<stem>.wav` files into.
    #[arg(short = 'o', long, default_value = "resampled")]
    pub output_dir: PathBuf,
    /// Channels to write. `mono` is what every engine here decodes to;
    /// `stereo` is what a `separate` stem needs to survive this stage intact.
    #[arg(long, value_enum, default_value_t = ChannelCount::Mono)]
    pub channels: ChannelCount,
}

impl ResampleArgs {
    /// Reject a rate nothing could be written at.
    pub fn verify(&self) -> Result<()> {
        self.io.verify()
    }

    /// The stage settings these flags describe.
    pub fn options(&self) -> preprocess_core::resample::ResampleOptions {
        preprocess_core::resample::ResampleOptions {
            sr: self.io.sr,
            channels: match self.channels {
                ChannelCount::Mono => preprocess_core::resample::Channels::Mono,
                ChannelCount::Stereo => preprocess_core::resample::Channels::Stereo,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    /// The `preprocess` binary's tree, minus `completions` — which belongs to
    /// the binary rather than to the stages, so it is not part of what is
    /// being tested here.
    #[derive(Debug, Parser)]
    #[command(name = "preprocess")]
    struct Bin {
        #[command(subcommand)]
        command: PreprocessCommand,
    }

    /// `voice`'s tree, so a flag is exercised in the *nested* position too.
    ///
    /// This is the whole reason `allow_negative_numbers` goes on the argument
    /// and not on the command: a `*-cli` crate exports an `Args` type that
    /// somebody else's `Command` hosts, and a `Command`-level annotation is
    /// left behind when that happens.
    #[derive(Debug, Parser)]
    #[command(name = "voice")]
    struct Nested {
        #[command(subcommand)]
        command: NestedCommand,
    }

    #[derive(Debug, Subcommand)]
    enum NestedCommand {
        Preprocess(PreprocessCli),
    }

    fn parse(argv: &[&str]) -> PreprocessCommand {
        Bin::parse_from(argv).command
    }

    // Every test below passes the value SPLIT from its flag (`--flag -1`
    // rather than `--flag=-1`). That is the only form that reproduces the
    // defect: clap reads a leading `-` as the start of a short-flag cluster
    // unless the argument opted out, and the `=` form is never ambiguous. A
    // test written the natural way — the way anybody writes one for a flag
    // they have just added — passes against a broken flag.

    #[test]
    fn a_negative_diarize_threshold_parses_when_split_from_its_flag() {
        let PreprocessCommand::Diarize(a) = parse(&[
            "preprocess",
            "diarize",
            "--threshold",
            "-0.2",
            "--reference",
            "me.wav",
            "in.wav",
        ]) else {
            panic!("expected diarize");
        };
        assert_eq!(a.threshold, -0.2);
    }

    #[test]
    fn the_whole_documented_threshold_range_is_reachable_and_accepted() {
        // `verify` accepts -1..=1, so the bottom of that range has to survive
        // the parser as well: a validator guarding values the parser refuses
        // is a promise the command line cannot keep.
        let PreprocessCommand::Diarize(a) = parse(&[
            "preprocess",
            "diarize",
            "--threshold",
            "-1",
            "--reference",
            "me.wav",
            "in.wav",
        ]) else {
            panic!("expected diarize");
        };
        assert_eq!(a.threshold, -1.0);
    }

    #[test]
    fn a_negative_threshold_parses_nested_under_voice_too() {
        let NestedCommand::Preprocess(cli) = Nested::parse_from([
            "voice",
            "preprocess",
            "diarize",
            "--threshold",
            "-0.2",
            "--reference",
            "me.wav",
            "in.wav",
        ])
        .command;
        let PreprocessCommand::Diarize(a) = cli.command else {
            panic!("expected diarize");
        };
        assert_eq!(a.threshold, -0.2);
    }

    #[test]
    fn the_silence_floor_clips_help_recommends_parses_when_split_from_its_flag() {
        // `--silence-db`'s own help says "e.g. -50", and this is the form that
        // advice is written in.
        let PreprocessCommand::Clip(a) =
            parse(&["preprocess", "clip", "--silence-db", "-50", "in.wav"])
        else {
            panic!("expected clip");
        };
        assert_eq!(a.slice.silence_db, -50.0);
    }

    #[test]
    fn the_silence_floor_trims_help_recommends_parses_when_split_from_its_flag() {
        let PreprocessCommand::Trim(a) =
            parse(&["preprocess", "trim", "--silence-db", "-50", "in.wav"])
        else {
            panic!("expected trim");
        };
        assert_eq!(a.silence_db, -50.0);
    }

    #[test]
    fn the_slice_floor_analyze_reports_against_parses_when_split_from_its_flag() {
        // `analyze` flattens the very same `SliceArgs`, so the annotation has
        // to reach it through the flatten as well.
        let PreprocessCommand::Analyze(a) =
            parse(&["preprocess", "analyze", "--silence-db", "-50", "in.wav"])
        else {
            panic!("expected analyze");
        };
        assert_eq!(a.slice.silence_db, -50.0);
    }

    #[test]
    fn the_broadcast_loudness_target_parses_when_split_from_its_flag() {
        // Every usable LUFS value is negative — full scale is 0 — so this flag
        // is unreachable in its documented form without the annotation.
        let PreprocessCommand::Normalize(a) =
            parse(&["preprocess", "normalize", "--lufs", "-23", "in.wav"])
        else {
            panic!("expected normalize");
        };
        assert_eq!(a.lufs, Some(-23.0));
    }

    // The tests below are the other half of the same worry. Above: a flag that
    // cannot be typed. Here: a flag that is typed, accepted, and then dropped
    // on the way into the stage — which `-h` cannot show and a stage cannot
    // report, because it never learns the value existed. Each one sets every
    // knob to something that is NOT its default, so a conversion that hard-codes
    // a field or forgets one fails rather than coincidentally agreeing.

    #[test]
    fn every_slicer_knob_reaches_the_slicer() {
        let PreprocessCommand::Clip(a) = parse(&[
            "preprocess",
            "clip",
            "--silence-db",
            "-50",
            "--min-silence",
            "0.7",
            "--min-clip",
            "2.5",
            "--max-clip",
            "9.0",
            "--pad",
            "0.4",
            "--sr",
            "44100",
            "--normalize",
            "in.wav",
        ]) else {
            panic!("expected clip");
        };
        let opts = a.options();
        assert_eq!(opts.sr, 44100);
        assert!(opts.normalize);
        assert_eq!(opts.slice.silence_db, -50.0);
        assert_eq!(opts.slice.min_silence, 0.7);
        assert_eq!(opts.slice.min_clip, 2.5);
        assert_eq!(opts.slice.max_clip, 9.0);
        assert_eq!(opts.slice.pad, 0.4);
    }

    #[test]
    fn the_slicer_defaults_are_the_ones_the_shared_slicer_would_have_picked() {
        // `SliceArgs` restates `audio_kit::SliceOptions::default()` rather than
        // deferring to it, so the two can drift in silence — and `-h` would go
        // on printing whichever one clap holds. This is what catches that.
        let PreprocessCommand::Clip(a) = parse(&["preprocess", "clip", "in.wav"]) else {
            panic!("expected clip");
        };
        // Compared field by field because `SliceOptions` is `audio-kit`'s and
        // carries no `PartialEq`; naming each field is also what makes a field
        // added there and forgotten here show up as a compile error rather
        // than as a silently unchecked knob.
        let (got, want) = (a.options().slice, audio_kit::SliceOptions::default());
        assert_eq!(got.silence_db, want.silence_db);
        assert_eq!(got.min_silence, want.min_silence);
        assert_eq!(got.min_clip, want.min_clip);
        assert_eq!(got.max_clip, want.max_clip);
        assert_eq!(got.pad, want.pad);
    }

    #[test]
    fn analyze_reports_against_the_very_settings_clip_would_cut_with() {
        // The two stages flatten one `SliceArgs` for this reason: the numbers
        // `analyze` prints are only worth acting on while they come from the
        // knobs `clip` will act on.
        let PreprocessCommand::Analyze(an) = parse(&[
            "preprocess",
            "analyze",
            "--silence-db",
            "-50",
            "--min-silence",
            "0.7",
            "in.wav",
        ]) else {
            panic!("expected analyze");
        };
        let PreprocessCommand::Clip(cl) = parse(&[
            "preprocess",
            "clip",
            "--silence-db",
            "-50",
            "--min-silence",
            "0.7",
            "in.wav",
        ]) else {
            panic!("expected clip");
        };
        let (got, want) = (an.options().slice, cl.options().slice);
        assert_eq!(got.silence_db, want.silence_db);
        assert_eq!(got.min_silence, want.min_silence);
        assert_eq!(got.min_clip, want.min_clip);
        assert_eq!(got.max_clip, want.max_clip);
        assert_eq!(got.pad, want.pad);
    }

    #[test]
    fn a_missing_reference_is_refused_before_anything_is_fetched() {
        // `--reference` is the whole specification of what to keep, so a
        // mistyped path cannot produce a run — and it used to be reported only
        // after the speaker model had been fetched and loaded. The check is
        // first in `verify` because it is the cheapest one there, and `verify`
        // is the first thing `commands::diarize::run` does.
        let PreprocessCommand::Diarize(a) = parse(&[
            "preprocess",
            "diarize",
            "--reference",
            "no/such/reference.wav",
            "in.wav",
        ]) else {
            panic!("expected diarize");
        };
        let err = a.verify().expect_err("a missing reference must be refused");
        let msg = format!("{err:#}");
        assert!(
            msg.contains("--reference"),
            "the message must name the offending flag, got: {msg}"
        );
        assert!(
            msg.contains("no/such/reference.wav"),
            "and the path it could not find, got: {msg}"
        );
    }

    #[test]
    fn every_diarize_knob_reaches_the_stage() {
        let PreprocessCommand::Diarize(a) = parse(&[
            "preprocess",
            "diarize",
            "--reference",
            "me.wav",
            "--threshold",
            "-0.2",
            "--window",
            "4.5",
            "--hop",
            "0.25",
            "--min-segment",
            "2.0",
            "--sr",
            "16000",
            "in.wav",
        ]) else {
            panic!("expected diarize");
        };
        let opts = a.options();
        assert_eq!(opts.sr, 16000);
        assert_eq!(opts.threshold, -0.2);
        assert_eq!(opts.window, 4.5);
        assert_eq!(opts.hop, 0.25);
        assert_eq!(opts.min_segment, 2.0);
    }

    #[test]
    fn every_trim_knob_reaches_the_stage() {
        let PreprocessCommand::Trim(a) = parse(&[
            "preprocess",
            "trim",
            "--silence-db",
            "-55",
            "--pad",
            "0.4",
            "--measure-floor",
            "--sr",
            "22050",
            "in.wav",
        ]) else {
            panic!("expected trim");
        };
        let opts = a.options();
        assert_eq!(opts.sr, 22050);
        assert_eq!(opts.silence_db, -55.0);
        assert_eq!(opts.pad, 0.4);
        assert!(opts.measure_floor);
    }

    #[test]
    fn every_denoise_knob_reaches_the_filter() {
        let PreprocessCommand::Denoise(a) = parse(&[
            "preprocess",
            "denoise",
            "--denoise-strength",
            "0.02",
            "--denoise-patch",
            "0.003",
            "--denoise-research",
            "0.01",
            "--sr",
            "44100",
            "in.wav",
        ]) else {
            panic!("expected denoise");
        };
        let opts = a.options();
        assert_eq!(opts.sr, 44100);
        assert_eq!(opts.params.strength, 0.02);
        assert_eq!(opts.params.patch_secs, 0.003);
        assert_eq!(opts.params.research_secs, 0.01);
    }

    #[test]
    fn resample_carries_both_the_rate_and_the_channel_count() {
        let PreprocessCommand::Resample(a) = parse(&[
            "preprocess",
            "resample",
            "--sr",
            "16000",
            "--channels",
            "stereo",
            "in.wav",
        ]) else {
            panic!("expected resample");
        };
        let opts = a.options();
        assert_eq!(opts.sr, 16000);
        assert_eq!(opts.channels, preprocess_core::resample::Channels::Stereo);
    }

    #[test]
    fn resample_writes_mono_unless_asked_otherwise() {
        // The default that costs a `separate` stem its second channel, so it
        // is the one worth stating in a test as well as in the long help.
        let PreprocessCommand::Resample(a) = parse(&["preprocess", "resample", "in.wav"]) else {
            panic!("expected resample");
        };
        assert_eq!(
            a.options().channels,
            preprocess_core::resample::Channels::Mono
        );
    }

    #[test]
    fn each_stem_choice_reaches_the_separator() {
        for (flag, want) in [
            ("vocals", preprocess_core::separate::Stems::Vocals),
            (
                "instrumental",
                preprocess_core::separate::Stems::Instrumental,
            ),
            ("both", preprocess_core::separate::Stems::Both),
        ] {
            let PreprocessCommand::Separate(a) =
                parse(&["preprocess", "separate", "--stem", flag, "in.wav"])
            else {
                panic!("expected separate");
            };
            assert_eq!(a.options().stems, want);
        }
        let PreprocessCommand::Separate(a) = parse(&["preprocess", "separate", "in.wav"]) else {
            panic!("expected separate");
        };
        assert_eq!(
            a.options().stems,
            preprocess_core::separate::Stems::Vocals,
            "the default stem is the one a corpus wants"
        );
    }

    #[test]
    fn normalizes_default_target_is_the_peak_its_help_prints() {
        // The one default in this file that clap does not own: `--peak` is an
        // `Option`, so its `[default: 0.95]` is written by hand into the help
        // text and the value actually used lives in `preprocess-core`. Nothing
        // but this test connects the two, and a drift between them would print
        // one number and apply another.
        let PreprocessCommand::Normalize(a) = parse(&["preprocess", "normalize", "in.wav"]) else {
            panic!("expected normalize");
        };
        assert_eq!(
            a.options().target,
            preprocess_core::normalize::Target::Peak(preprocess_core::normalize::DEFAULT_PEAK)
        );
        assert_eq!(preprocess_core::normalize::DEFAULT_PEAK, 0.95);
    }

    #[test]
    fn each_normalize_target_reaches_the_stage_and_the_pair_is_refused() {
        let PreprocessCommand::Normalize(a) =
            parse(&["preprocess", "normalize", "--peak", "0.7", "in.wav"])
        else {
            panic!("expected normalize");
        };
        assert_eq!(
            a.options().target,
            preprocess_core::normalize::Target::Peak(0.7)
        );

        let PreprocessCommand::Normalize(a) =
            parse(&["preprocess", "normalize", "--lufs", "-16", "in.wav"])
        else {
            panic!("expected normalize");
        };
        assert_eq!(
            a.options().target,
            preprocess_core::normalize::Target::Lufs(-16.0)
        );

        // Asking for both is refused by clap, by name, before a file is opened
        // — which is why `options()` may treat the pair as unreachable.
        assert!(
            Bin::try_parse_from([
                "preprocess",
                "normalize",
                "--peak",
                "0.7",
                "--lufs",
                "-16",
                "in.wav"
            ])
            .is_err()
        );
    }

    #[test]
    fn a_flag_whose_values_are_all_positive_still_refuses_a_negative_one() {
        // The annotation widens what a value may look like, so it is scoped to
        // the arguments that need it rather than applied to the command. These
        // are the flags that must keep rejecting a negative, and the check is
        // that they fail at the *parser* rather than reaching `verify`.
        for (stage, flag) in [
            ("clip", "--sr"),
            ("clip", "--pad"),
            ("clip", "--min-silence"),
            ("diarize", "--window"),
            ("normalize", "--peak"),
        ] {
            let mut argv = vec!["preprocess", stage, flag, "-1"];
            if stage == "diarize" {
                argv.extend(["--reference", "me.wav"]);
            }
            argv.push("in.wav");
            assert!(
                Bin::try_parse_from(&argv).is_err(),
                "{stage} {flag} accepted a negative value"
            );
        }
    }
}
