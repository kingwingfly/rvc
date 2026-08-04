//! Command-line argument definitions (clap derive).
//!
//! The shape is fixed by the house rule every engine follows: the bare
//! invocation is the stdin→stdout filter, and everything that is not streaming
//! is a subcommand beside it.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use clap::{Args, Subcommand};

pub use cli_kit::{Backend, CompletionsArgs};

/// The whole of the `seedvc` **engine**, defined once and worn two ways: the
/// `seedvc` binary flattens it at its top level, `voice` nests it under a
/// `seedvc` subcommand.
///
/// `completions` is deliberately not in here — it describes the *binary* being
/// completed, not the engine, so each `main` adds it beside this. See
/// [`SeedVcCommand`].
#[derive(Debug, Args)]
pub struct SeedVcCli {
    #[command(flatten)]
    pub filter: FilterArgs,
    #[command(subcommand)]
    pub command: Option<SeedVcCommand>,
}

/// What this engine can do, and nothing about the executable that hosts it.
///
/// **`completions` is not a member, on purpose.** A completion script describes
/// one binary, so a nested `voice seedvc completions` could only ever emit
/// `voice`'s — which is exactly what it used to do. Each `main.rs` flattens this
/// enum into its own and adds `Completions` beside it, so the standalone binary
/// keeps the subcommand and `voice` grows only one, at its top level.
#[derive(Debug, Subcommand)]
pub enum SeedVcCommand {
    /// Convert audio files into the reference's voice (WAV output).
    // Boxed because it carries every model path and every sampler knob on top of
    // its own two, and clippy is right that `completions` shouldn't pay for that.
    Convert(Box<ConvertArgs>),
    /// Prefetch the weights a conversion needs, so the first run is offline.
    Download(DownloadArgs),
}

/// Where the four networks come from, and which voice to convert into.
///
/// The reference lives here rather than beside the input files because both the
/// filter and `convert` need it, and neither can be given it by clap: these
/// options are flattened beside the subcommands for the bare invocation, so
/// marking it `required` would demand one of `download` and `completions` too.
#[derive(Debug, Args, Clone)]
pub struct ModelOpts {
    /// A 1–30 s recording of the target voice. This is the whole speaker
    /// specification — there is nothing to train.
    #[arg(short, long)]
    pub reference: Option<PathBuf>,
    /// Seed-VC checkpoint: the transformer and the length regulator
    /// [default: auto-downloaded from Hugging Face].
    #[arg(long)]
    pub checkpoint: Option<PathBuf>,
    /// CAMPPlus timbre encoder [default: auto-downloaded from Hugging Face].
    #[arg(long)]
    pub campplus: Option<PathBuf>,
    /// BigVGAN vocoder weights [default: auto-downloaded from Hugging Face].
    #[arg(long)]
    pub bigvgan: Option<PathBuf>,
    /// Whisper content encoder directory [default: auto-downloaded from
    /// Hugging Face].
    #[arg(long)]
    pub content: Option<PathBuf>,
    /// Directory the downloaded models are cached in. Shared by every engine
    /// unless `$SEEDVC_CACHE_DIR` (or `$VOICE_CACHE_DIR`) says otherwise.
    #[arg(long, default_value_os_t = hub_kit::cache_dir_for("SEEDVC_CACHE_DIR"))]
    pub cache_dir: PathBuf,
}

impl ModelOpts {
    /// The reference recording every conversion path needs.
    ///
    /// Optional to clap for the reason the struct's own docs give; "required" is
    /// decided here, and callers check it before anything is fetched or loaded.
    pub fn reference(&self) -> Result<&Path> {
        self.reference.as_deref().context(
            "-r/--reference is required: a 1-30 s recording of the voice to \
             convert into (there is no model to train — the clip is the whole \
             speaker specification)",
        )
    }
}

/// The sampler's knobs, shared by the filter and `convert`.
#[derive(Debug, Args, Clone, Copy)]
pub struct SamplerOpts {
    /// Euler steps the flow-matching sampler takes from noise to mel. Cost and
    /// smoothness are both linear in it.
    #[arg(long, default_value_t = 30)]
    pub steps: usize,
    /// Classifier-free guidance scale: how far each step is pushed away from the
    /// unconditioned prediction. Raise to follow the reference harder, at the
    /// cost of artefacts; zero or below skips the unconditional pass entirely
    /// and halves the work per step.
    #[arg(long, default_value_t = 0.7)]
    pub guidance: f64,
    /// Scales the output's duration against the source's. Above 1 is slower,
    /// below is faster; pitch is unchanged.
    #[arg(long, default_value_t = 1.0)]
    pub length_adjust: f64,
    /// Seed for the sampler's noise, so a conversion can be repeated exactly.
    #[arg(long, default_value_t = 0)]
    pub seed: u64,
}

impl SamplerOpts {
    /// Reject values clap's types accept but the sampler cannot use.
    pub fn verify(&self) -> Result<()> {
        anyhow::ensure!(
            self.steps > 0,
            "--steps must be at least 1: the sampler integrates the flow over \
             that many Euler steps, and zero of them leaves pure noise"
        );
        anyhow::ensure!(
            self.length_adjust > 0.0,
            "--length-adjust must be positive (1.0 keeps the source's duration)"
        );
        Ok(())
    }

    /// The engine's own conversion settings.
    pub fn options(&self) -> seedvc_core::ConvertOptions {
        seedvc_core::ConvertOptions {
            sampler: seedvc_core::Sampler {
                steps: self.steps,
                guidance: self.guidance,
            },
            length_adjust: self.length_adjust,
            seed: self.seed,
        }
    }
}

/// The bare invocation: raw f32le mono PCM in at 16 kHz, raw f32le mono PCM out
/// at the vocoder's 22.05 kHz.
#[derive(Debug, Args)]
pub struct FilterArgs {
    #[command(flatten)]
    pub models: ModelOpts,
    #[command(flatten)]
    pub sampler: SamplerOpts,
    /// Samples per input read chunk from stdin (16 kHz mono f32le).
    #[arg(long, default_value_t = 1600)]
    pub chunk: usize,
    /// Inference backend: `cuda` (aliases `burn`, `burn-cuda`), `tch`
    /// (`libtorch`, `burn-tch`) or `wgpu` (`webgpu`, `burn-wgpu`); `auto` picks
    /// the fastest compiled in. Nothing exports Seed-VC to ONNX, so `onnx` is an
    /// error rather than a fallback.
    #[arg(long, value_enum, default_value_t = Backend::Auto)]
    pub backend: Backend,
    /// Compute device: `auto` (fastest visible), `cpu`, `gpu`, `gpu:N`, `mps` or
    /// `vulkan` (`cuda`/`cuda:N` also accepted). The `cuda` backend has GPUs only.
    #[arg(long, default_value = "auto", value_name = "DEVICE", value_parser = cli_kit::parse_device)]
    pub device: burn_kit::DeviceSpec,
}

impl FilterArgs {
    /// Reject values clap's types accept but the pipeline cannot use.
    pub fn verify(&self) -> Result<()> {
        // A zero-sample read would spin on stdin forever without ever handing
        // the converter a block to work on.
        anyhow::ensure!(self.chunk > 0, "--chunk must be at least 1 sample");
        self.sampler.verify()
    }
}

/// Batch conversion: files in, one WAV each out.
#[derive(Debug, Args)]
pub struct ConvertArgs {
    #[command(flatten)]
    pub models: ModelOpts,
    #[command(flatten)]
    pub sampler: SamplerOpts,
    /// One or more input audio files (mp3, wav, ...).
    #[arg(required = true)]
    pub input: Vec<PathBuf>,
    /// Directory to write converted `<stem>.wav` files into.
    #[arg(short = 'o', long, default_value = ".")]
    pub output_dir: PathBuf,
    /// Inference backend: `cuda` (aliases `burn`, `burn-cuda`), `tch`
    /// (`libtorch`, `burn-tch`) or `wgpu` (`webgpu`, `burn-wgpu`); `auto` picks
    /// the fastest compiled in. Nothing exports Seed-VC to ONNX, so `onnx` is an
    /// error rather than a fallback.
    #[arg(long, value_enum, default_value_t = Backend::Auto)]
    pub backend: Backend,
    /// Compute device: `auto` (fastest visible), `cpu`, `gpu`, `gpu:N`, `mps` or
    /// `vulkan` (`cuda`/`cuda:N` also accepted). The `cuda` backend has GPUs only.
    #[arg(long, default_value = "auto", value_name = "DEVICE", value_parser = cli_kit::parse_device)]
    pub device: burn_kit::DeviceSpec,
}

impl ConvertArgs {
    /// Reject values clap's types accept but the pipeline cannot use.
    pub fn verify(&self) -> Result<()> {
        self.sampler.verify()
    }
}

/// What a default run would fetch on demand, fetched up front instead.
///
/// All four networks, because a conversion opens all four. There is nothing
/// optional to leave out as `tts` has, and no warm-start base as `rvc` has —
/// Seed-VC is zero-shot, so nothing it downloads is ever a training input.
#[derive(Debug, Args)]
pub struct DownloadArgs {
    /// Directory the downloaded models are cached in. Shared by every engine
    /// unless `$SEEDVC_CACHE_DIR` (or `$VOICE_CACHE_DIR`) says otherwise.
    #[arg(long, default_value_os_t = hub_kit::cache_dir_for("SEEDVC_CACHE_DIR"))]
    pub cache_dir: PathBuf,
}
