//! Command-line argument definitions (clap derive).
//!
//! The shape differs from every engine's in exactly one way, and the reason is
//! in [`crate`]'s own docs: there is **no bare invocation**, because
//! `preprocess` names a stage rather than an engine and which stage is what the
//! subcommand picks. Everything else is the house shape — one clap type for the
//! whole tree, `completions` added by whichever `main` is hosting it.

use std::path::PathBuf;

use anyhow::Result;
use clap::{Args, Subcommand};

// No `Backend` re-export yet: no stage here runs a model, and a `--backend`
// that decides nothing is worse than none. The first stage that needs one takes
// `cli_kit::Backend`, like every other binary — there is one such enum for the
// whole workspace and a second must not appear beside it.
pub use cli_kit::CompletionsArgs;

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

/// Slice a corpus into clean per-utterance clips.
#[derive(Debug, Args)]
pub struct ClipArgs {
    #[command(flatten)]
    pub io: IoArgs,
    /// Directory to write the sliced `<stem>_<NNN>.wav` clips into.
    #[arg(short = 'o', long, default_value = "dataset")]
    pub output_dir: PathBuf,
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
    /// Peak-normalize each written clip to ~0.95 full-scale.
    #[arg(long)]
    pub normalize: bool,
}

impl ClipArgs {
    /// Reject a slicer configuration that would silently produce no clips.
    pub fn verify(&self) -> Result<()> {
        self.io.verify()?;
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

    /// The stage settings these flags describe.
    pub fn options(&self) -> preprocess_core::clip::ClipOptions {
        preprocess_core::clip::ClipOptions {
            sr: self.io.sr,
            slice: audio_kit::SliceOptions {
                silence_db: self.silence_db,
                min_silence: self.min_silence,
                min_clip: self.min_clip,
                max_clip: self.max_clip,
                pad: self.pad,
            },
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
