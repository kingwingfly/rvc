//! Command-line argument definitions (clap derive).

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use clap::{Args, Subcommand};
pub use cli_kit::{Backend, CompletionsArgs};
pub use preprocess_kit::PreprocessArgs;

/// The whole of the `rvc` command tree, defined once and worn two ways: the
/// `rvc` binary flattens it at its top level, `voice` nests it under an `rvc`
/// subcommand. Every engine here has the same shape — the bare invocation is
/// the stdin→stdout filter, and everything else is a subcommand.
#[derive(Debug, Args)]
pub struct RvcCli {
    #[command(flatten)]
    pub filter: FilterArgs,
    #[command(subcommand)]
    pub command: Option<RvcCommand>,
}

/// What this engine can do, and nothing about the executable that hosts it.
///
/// **`completions` is not a member, on purpose.** A completion script describes
/// one binary, so a nested `voice rvc completions` could only ever emit
/// `voice`'s — which is exactly what it used to do. Each `main.rs` flattens this
/// enum into its own and adds `Completions` beside it, so the standalone binary
/// keeps the subcommand and `voice` grows only one, at its top level.
#[derive(Debug, Subcommand)]
pub enum RvcCommand {
    /// Batch-convert audio files into the target timbre (WAV output).
    Convert(ConvertArgs),
    /// Train an RVC generator on a corpus (native Rust / burn).
    // Boxed because `TrainArgs` alone is ~320 bytes against 72 for the next
    // largest variant, and clippy is right that every parse shouldn't pay it.
    Train(Box<TrainArgs>),
    /// Slice a corpus into clean per-sentence training clips (dead-air removed).
    Preprocess(PreprocessArgs),
    /// Prefetch the weights a conversion needs, so the first run is offline.
    Download(DownloadArgs),
}

/// Shared options for locating the three ONNX models.
#[derive(Debug, Args, Clone)]
pub struct ModelOpts {
    /// Trained RVC generator weights (`.safetensors`, or `.onnx` for an export).
    #[arg(short, long)]
    pub model: Option<PathBuf>,
    /// Generator output sample rate (40000 or 48000).
    #[arg(long, default_value_t = 48000)]
    pub model_sr: u32,
    /// ContentVec encoder ONNX [default: auto-downloaded from Hugging Face].
    #[arg(long)]
    pub content: Option<PathBuf>,
    /// RMVPE F0 ONNX [default: auto-downloaded from Hugging Face].
    #[arg(long)]
    pub rmvpe: Option<PathBuf>,
    /// Directory the downloaded models are cached in. Shared by every engine
    /// unless `$RVC_CACHE_DIR` (or `$VOICE_CACHE_DIR`) says otherwise.
    #[arg(long, default_value_os_t = hub_kit::cache_dir_for("RVC_CACHE_DIR"))]
    pub cache_dir: PathBuf,
    /// Speaker id fed to the generator (single-speaker models use 0).
    #[arg(long, default_value_t = 0)]
    pub speaker_id: i64,
}

impl ModelOpts {
    /// The generator weights every conversion path needs.
    ///
    /// Optional to clap only because these options are also flattened beside
    /// the subcommands, for the bare filter invocation — requiring `-m` there
    /// would require it of `train` and `preprocess` too. So "required" is
    /// decided here, and callers check it before anything is fetched or loaded.
    pub fn model(&self) -> Result<&Path> {
        self.model.as_deref().context(
            "-m/--model is required: the trained generator weights \
             (train one with the `train` subcommand)",
        )
    }

    /// Reject values clap's types accept but the pipeline cannot use.
    pub fn verify(&self) -> Result<()> {
        anyhow::ensure!(
            matches!(self.model_sr, 40_000 | 48_000),
            "--model-sr must be 40000 or 48000, not {}: it is a property of the \
             trained generator, not a resampling request",
            self.model_sr
        );
        anyhow::ensure!(
            self.speaker_id >= 0,
            "--speaker-id must not be negative (single-speaker models use 0)"
        );
        Ok(())
    }
}

#[derive(Debug, Args)]
pub struct ConvertArgs {
    #[command(flatten)]
    pub models: ModelOpts,
    /// One or more input audio files (mp3, wav, ...).
    #[arg(required = true)]
    pub input: Vec<PathBuf>,
    /// Directory to write converted `<stem>_<model>.wav` files into.
    #[arg(short = 'o', long, default_value = ".")]
    pub output_dir: PathBuf,
    /// Pitch shift in semitones.
    #[arg(short = 't', long, default_value_t = 0)]
    pub transpose: i32,
    /// Inference backend: `auto`, `onnx`, `cuda` (aliases `burn`, `burn-cuda`),
    /// `tch` (`libtorch`, `burn-tch`) or `wgpu` (`webgpu`, `burn-wgpu`).
    #[arg(long, value_enum, default_value_t = Backend::Auto)]
    pub backend: Backend,
    /// Compute device: `auto` (fastest visible), `cpu`, `gpu`, `gpu:N`, `mps` or
    /// `vulkan` (`cuda`/`cuda:N` also accepted). The `cuda` backend has GPUs only.
    #[arg(long, default_value = "auto", value_name = "DEVICE", value_parser = cli_kit::parse_device)]
    pub device: burn_kit::DeviceSpec,
    #[command(flatten)]
    pub denoise: DenoiseOpts,
}

impl ConvertArgs {
    /// Reject values clap's types accept but the pipeline cannot use.
    pub fn verify(&self) -> Result<()> {
        self.models.verify()?;
        self.denoise.verify()
    }
}

/// The bare invocation: raw f32le mono PCM in, raw f32le mono PCM out.
#[derive(Debug, Args)]
pub struct FilterArgs {
    #[command(flatten)]
    pub models: ModelOpts,
    /// Pitch shift in semitones.
    #[arg(short = 't', long, default_value_t = 0)]
    pub transpose: i32,
    /// Samples per input read chunk from stdin (16 kHz mono f32le).
    #[arg(long, default_value_t = 1600)]
    pub chunk: usize,
    /// Inference backend: `auto`, `onnx`, `cuda` (aliases `burn`, `burn-cuda`),
    /// `tch` (`libtorch`, `burn-tch`) or `wgpu` (`webgpu`, `burn-wgpu`). Prefer
    /// `tch` or `onnx` here — `cuda` does not keep up with realtime.
    #[arg(long, value_enum, default_value_t = Backend::Auto)]
    pub backend: Backend,
    /// Compute device: `auto` (fastest visible), `cpu`, `gpu`, `gpu:N`, `mps` or
    /// `vulkan` (`cuda`/`cuda:N` also accepted). The `cuda` backend has GPUs only.
    #[arg(long, default_value = "auto", value_name = "DEVICE", value_parser = cli_kit::parse_device)]
    pub device: burn_kit::DeviceSpec,
    #[command(flatten)]
    pub denoise: DenoiseOpts,
}

impl FilterArgs {
    /// Reject values clap's types accept but the pipeline cannot use.
    pub fn verify(&self) -> Result<()> {
        self.models.verify()?;
        // A zero-sample read would spin on stdin forever without ever handing
        // the converter a block to work on.
        anyhow::ensure!(self.chunk > 0, "--chunk must be at least 1 sample");
        self.denoise.verify()
    }
}

/// De-hiss options shared by the streaming filter and `convert`. Off unless
/// `--denoise` is given; the tuning flags only take effect when it is. Defaults
/// mirror
/// [`rvc_core::DenoiseParams::default`]; `--denoise-strength` is the main knob.
#[derive(Debug, Args, Clone)]
pub struct DenoiseOpts {
    /// Remove steady background hiss from the output (ffmpeg `anlmdn`
    /// non-local-means de-noise, run in-process, tuned to preserve the soft
    /// broadband texture of quiet, breathy content).
    #[arg(long)]
    pub denoise: bool,
    /// De-hiss strength: raise to remove more hiss, lower if soft/breathy
    /// texture starts to smear. Only used with `--denoise`.
    #[arg(long = "denoise-strength", default_value_t = 0.008)]
    pub denoise_strength: f32,
    /// `anlmdn` patch duration (seconds): the unit compared for self-similarity;
    /// smaller keeps finer detail. Only used with `--denoise`.
    #[arg(long = "denoise-patch", default_value_t = 0.002)]
    pub denoise_patch: f32,
    /// `anlmdn` research window (seconds): how far in time it looks for similar
    /// patches. Must exceed the patch, and sets the de-hiss latency. Only used
    /// with `--denoise`.
    #[arg(long = "denoise-research", default_value_t = 0.006)]
    pub denoise_research: f32,
}

impl DenoiseOpts {
    /// Reject a de-hiss configuration `anlmdn` would refuse or misbehave on.
    ///
    /// Only when `--denoise` is on: the tuning flags have defaults, so checking
    /// them unconditionally would reject a run that never denoises anything.
    pub fn verify(&self) -> Result<()> {
        if !self.denoise {
            return Ok(());
        }
        anyhow::ensure!(
            self.denoise_strength > 0.0,
            "--denoise-strength must be positive (drop --denoise to disable the stage)"
        );
        anyhow::ensure!(self.denoise_patch > 0.0, "--denoise-patch must be positive");
        // Not cosmetic: `anlmdn` searches a window for patches to compare, so a
        // window no larger than the patch leaves it nothing to average over.
        anyhow::ensure!(
            self.denoise_research > self.denoise_patch,
            "--denoise-research ({}) must exceed --denoise-patch ({}): the research \
             window is where similar patches are looked for",
            self.denoise_research,
            self.denoise_patch
        );
        Ok(())
    }

    /// The [`rvc_core::DenoiseParams`] these flags describe, or `None` when
    /// `--denoise` was not passed (stage disabled).
    pub fn params(&self) -> Option<rvc_core::DenoiseParams> {
        self.denoise.then_some(rvc_core::DenoiseParams {
            strength: self.denoise_strength,
            patch_secs: self.denoise_patch,
            research_secs: self.denoise_research,
        })
    }
}

#[derive(Debug, Args)]
pub struct DownloadArgs {
    /// Directory the downloaded models are cached in. Shared by every engine
    /// unless `$RVC_CACHE_DIR` (or `$VOICE_CACHE_DIR`) says otherwise.
    #[arg(long, default_value_os_t = hub_kit::cache_dir_for("RVC_CACHE_DIR"))]
    pub cache_dir: PathBuf,
    /// Override the ContentVec repo as `owner/name:file`
    /// [default: the toolkit's ContentVec ONNX on Hugging Face].
    #[arg(long)]
    pub content: Option<String>,
    /// Override the RMVPE repo as `owner/name:file`
    /// [default: the toolkit's RMVPE ONNX on Hugging Face].
    #[arg(long)]
    pub rmvpe: Option<String>,
}

#[derive(Debug, Args)]
pub struct TrainArgs {
    /// Corpus: one or more audio files of the target voice.
    #[arg(required = true)]
    pub data: Vec<PathBuf>,
    /// Output path for the trained generator (writes `<path>.safetensors`).
    #[arg(short = 'o', long, default_value = "models/voice")]
    pub out: PathBuf,
    /// Generator output sample rate (48000).
    #[arg(long, default_value_t = 48000)]
    pub model_sr: u32,
    /// Number of training epochs; stop early with `q` or Ctrl-C (saves).
    #[arg(short = 'e', long, default_value_t = 5)]
    pub epochs: u32,
    /// Mini-batch size (lower it if a 6 GB GPU runs out of memory).
    #[arg(short = 'b', long, default_value_t = 4)]
    pub batch_size: usize,
    /// Speaker id embedded in the generator (single-speaker: 0).
    #[arg(long, default_value_t = 0)]
    pub speaker_id: i64,
    /// Base AdamW learning rate.
    #[arg(long, default_value_t = 1e-4)]
    pub lr: f64,
    /// End-of-run LR as a fraction of `--lr`, decayed over the whole run
    /// (`1.0` = no decay).
    #[arg(long, default_value_t = 0.1)]
    pub lr_final: f64,
    /// EMA smoothing window as a fraction of the run; the saved model is the EMA
    /// (`0` = save the raw weights).
    #[arg(long, default_value_t = 0.1)]
    pub ema_frac: f64,
    /// Micro-batches per optimizer step (effective batch = batch×N). `1` = off.
    #[arg(long, default_value_t = 1)]
    pub grad_accum: usize,
    /// Discriminator LR multiplier; `<1.0` tames buzzy static.
    #[arg(long, default_value_t = 1.0)]
    pub d_lr_ratio: f64,
    /// Update the discriminator every N steps.
    #[arg(long, default_value_t = 1)]
    pub d_interval: usize,
    /// Bias clip sampling toward cleaner clips by `snr^alpha` (noise-floor SNR,
    /// not loudness). `0` = uniform.
    #[arg(long, default_value_t = 0.0)]
    pub snr_weight: f32,
    /// Don't keep the best-so-far (lowest mel) weights in
    /// `<out-dir>/checkpoint/<name>.best[.disc].safetensors`; save only the
    /// final ones.
    #[arg(long)]
    pub no_save_best: bool,
    /// Compute backend: `auto`, `cuda` (alias `burn-cuda`), `tch` (`libtorch`,
    /// `burn-tch`) or `wgpu` (`webgpu`, `burn-wgpu`). All three train, and the
    /// saved weights are the same whichever you pick.
    #[arg(long, value_enum, default_value_t = Backend::Auto)]
    pub backend: Backend,
    /// Compute device(s): `auto`, `cpu`, `gpu`, `gpu:N`, `mps`, `vulkan`
    /// (`cuda`/`cuda:N` are accepted spellings of `gpu`). Comma-separate for
    /// data-parallel training across devices — the first is the master.
    #[arg(
        long,
        alias = "devices",
        default_value = "auto",
        value_name = "DEVICE",
        value_delimiter = ',',
        value_parser = cli_kit::parse_device
    )]
    pub device: Vec<burn_kit::DeviceSpec>,
    /// Overwrite weights already at `-o` instead of refusing to start.
    /// `--resume` implies it: continuing a run means writing over its files.
    #[arg(short = 'y', long)]
    pub yes: bool,
    /// Resume from a generator `.safetensors` (+ its `.disc.safetensors`
    /// sidecar). Pass either the EMA output or its `.raw.safetensors` twin —
    /// the raw (non-EMA) live weights are preferred automatically when present,
    /// as they pair faithfully with the saved discriminator. Conflicts with
    /// `--pretrained-g`.
    #[arg(
        long,
        alias = "continue",
        value_name = "SAFETENSORS",
        conflicts_with = "pretrained_g"
    )]
    pub resume: Option<PathBuf>,
    /// Pretrained generator base to warm-start from [default:
    /// `<cache-dir>/pretrained/f0G48k.pth`, downloaded from HF
    /// `lj1995/VoiceConversionWebUI` on first use and reused by every run].
    #[arg(long, conflicts_with = "no_pretrained")]
    pub pretrained_g: Option<PathBuf>,
    /// Pretrained discriminator base to warm-start from [default:
    /// `<cache-dir>/pretrained/f0D48k.pth`, downloaded on first use].
    #[arg(long, conflicts_with = "no_pretrained")]
    pub pretrained_d: Option<PathBuf>,
    /// Train from scratch: no warm-start, and no download of the bases. Poor on
    /// a small corpus, which is why they are fetched by default. `--resume`
    /// fetches nothing either: it continues from weights that already exist.
    #[arg(long)]
    pub no_pretrained: bool,
    /// ContentVec encoder ONNX [default: auto-downloaded].
    #[arg(long)]
    pub content: Option<PathBuf>,
    /// RMVPE F0 ONNX [default: auto-downloaded].
    #[arg(long)]
    pub rmvpe: Option<PathBuf>,
    /// Directory the downloaded models are cached in. Shared by every engine
    /// unless `$RVC_CACHE_DIR` (or `$VOICE_CACHE_DIR`) says otherwise.
    #[arg(long, default_value_os_t = hub_kit::cache_dir_for("RVC_CACHE_DIR"))]
    pub cache_dir: PathBuf,
    /// Disable the TUI dashboard and log to stderr (auto-off when not a TTY).
    #[arg(long)]
    pub no_tui: bool,
}

impl TrainArgs {
    /// Reject a training configuration that cannot converge, or cannot start.
    ///
    /// Checked before the corpus is decoded and before a base is downloaded: a
    /// typo in a learning rate should cost a message, not an hour of GPU and a
    /// model full of `NaN`.
    pub fn verify(&self) -> Result<()> {
        anyhow::ensure!(
            matches!(self.model_sr, 40_000 | 48_000),
            "--model-sr must be 40000 or 48000, not {}",
            self.model_sr
        );
        anyhow::ensure!(self.epochs > 0, "--epochs must be at least 1");
        anyhow::ensure!(self.batch_size > 0, "--batch-size must be at least 1");
        anyhow::ensure!(
            self.grad_accum > 0,
            "--grad-accum must be at least 1 (1 = off)"
        );
        anyhow::ensure!(self.d_interval > 0, "--d-interval must be at least 1");
        anyhow::ensure!(self.speaker_id >= 0, "--speaker-id must not be negative");
        anyhow::ensure!(
            self.lr > 0.0 && self.lr.is_finite(),
            "--lr must be a positive, finite number"
        );
        anyhow::ensure!(
            self.d_lr_ratio > 0.0 && self.d_lr_ratio.is_finite(),
            "--d-lr-ratio must be a positive, finite number (1.0 = same LR as the generator)"
        );
        // `1.0` is "no decay" and is the upper bound; at or below zero the LR
        // would reach zero or go negative partway through the run.
        anyhow::ensure!(
            self.lr_final > 0.0 && self.lr_final <= 1.0,
            "--lr-final is a fraction of --lr and must be in (0, 1], not {}: at or \
             below 0 the run would end with no learning rate at all, and above 1 it \
             would end faster than it started",
            self.lr_final
        );
        // `0` disables the EMA and saves the raw weights; `1` would mean a window
        // as long as the run, which never updates.
        anyhow::ensure!(
            (0.0..1.0).contains(&self.ema_frac),
            "--ema-frac is a fraction of the run and must be in [0, 1); 0 saves the \
             raw weights"
        );
        anyhow::ensure!(
            self.snr_weight >= 0.0 && self.snr_weight.is_finite(),
            "--snr-weight must be zero (uniform) or a positive, finite exponent"
        );
        anyhow::ensure!(!self.device.is_empty(), "--device names no device");
        Ok(())
    }
}
