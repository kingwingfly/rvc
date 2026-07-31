//! Command-line argument definitions (clap derive).

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use clap::{Args, Subcommand};
pub use cli_kit::{Backend, CompletionsArgs};

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
    /// Download/prefetch shared ONNX assets from Hugging Face.
    Models(ModelsArgs),
    /// Print a shell completion script (bash, zsh, fish, powershell, elvish).
    Completions(CompletionsArgs),
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

/// De-hiss options shared by the streaming filter and `convert`. Off unless
/// `--denoise` is given; the tuning flags only take effect when it is. Defaults
/// mirror
/// [`rvc_core::DenoiseParams::default`]; `--denoise-strength` is the main knob.
#[derive(Debug, Args, Clone)]
pub struct DenoiseOpts {
    /// Remove steady background hiss from the output (ffmpeg `anlmdn`
    /// non-local-means de-noise, run in-process, tuned to preserve the soft
    /// broadband texture of ASMR/breathy content).
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
pub struct ModelsArgs {
    #[command(subcommand)]
    pub command: ModelsCommand,
}

#[derive(Debug, Subcommand)]
pub enum ModelsCommand {
    /// Download the shared ContentVec + RMVPE ONNX assets.
    Download(ModelsDownloadArgs),
}

#[derive(Debug, Args)]
pub struct ModelsDownloadArgs {
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
pub struct PreprocessArgs {
    /// Input audio files and/or directories (directories are expanded to their
    /// audio files: mp3, wav, flac, m4a, ogg, opus, aac, wma).
    #[arg(required = true)]
    pub input: Vec<PathBuf>,
    /// Directory to write the sliced `<stem>_<NNN>.wav` clips into.
    #[arg(short = 'o', long, default_value = "dataset")]
    pub output_dir: PathBuf,
    /// Sample rate of the written clips (match your training `--model-sr`).
    #[arg(long, default_value_t = 48000)]
    pub model_sr: u32,
    /// Energy floor in dBFS: audio quieter than this counts as between-sentence
    /// dead-air. ASMR users can lower it (e.g. -50) to keep the very softest
    /// passages — energy is used only to find silent gaps, never to gate quiet
    /// content.
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
    /// Directory for the training log and saved weights.
    #[arg(long, default_value = "models/train")]
    pub work_dir: PathBuf,
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
    /// `<-o's directory>/pretrained/f0G48k.pth`, downloaded from HF
    /// `lj1995/VoiceConversionWebUI` on first use].
    #[arg(long, conflicts_with = "no_pretrained")]
    pub pretrained_g: Option<PathBuf>,
    /// Pretrained discriminator base to warm-start from [default:
    /// `<-o's directory>/pretrained/f0D48k.pth`, downloaded on first use].
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
