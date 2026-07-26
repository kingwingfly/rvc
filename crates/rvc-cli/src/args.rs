//! Command-line argument definitions (clap derive).

use std::path::PathBuf;

use clap::{Args, Parser, Subcommand, ValueEnum};

/// Which generator backend runs inference.
#[derive(Debug, Clone, Copy, Default, ValueEnum)]
pub enum InferBackend {
    /// Pick by weights extension: `.onnx` → onnx-runtime, else Burn.
    #[default]
    Auto,
    /// Native Burn generator (`.pth`/`.safetensors`).
    Burn,
    /// ONNX Runtime generator (`.onnx`).
    Onnx,
}

/// rvc — RVC voice conversion toolkit (ONNX Runtime + native Burn).
#[derive(Debug, Parser)]
#[command(name = "rvc", version, about)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Debug, Subcommand)]
pub enum Command {
    /// Batch-convert audio files into the target timbre (WAV output).
    Convert(ConvertArgs),
    /// Realtime Unix filter: raw f32le mono PCM on stdin -> stdout.
    Serve(ServeArgs),
    /// Download/prefetch shared ONNX assets from Hugging Face.
    Models(ModelsArgs),
    /// Train an RVC generator on a corpus (native Rust / burn).
    Train(TrainArgs),
    /// Slice a corpus into clean per-sentence training clips (dead-air removed).
    Preprocess(PreprocessArgs),
}

/// Shared options for locating the three ONNX models.
#[derive(Debug, Args, Clone)]
pub struct ModelOpts {
    /// Trained RVC generator ONNX (`voice.onnx`).
    #[arg(short, long)]
    pub model: PathBuf,
    /// Generator output sample rate (40000 or 48000).
    #[arg(long, default_value_t = 48000)]
    pub model_sr: u32,
    /// ContentVec encoder ONNX [default: auto-downloaded from Hugging Face].
    #[arg(long)]
    pub content: Option<PathBuf>,
    /// RMVPE F0 ONNX [default: auto-downloaded from Hugging Face].
    #[arg(long)]
    pub rmvpe: Option<PathBuf>,
    /// Cache directory for downloaded assets
    /// [default: the Hugging Face cache, e.g. ~/.cache/huggingface/hub].
    #[arg(long)]
    pub cache_dir: Option<PathBuf>,
    /// Speaker id fed to the generator (single-speaker models use 0).
    #[arg(long, default_value_t = 0)]
    pub speaker_id: i64,
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
    /// Inference backend (default: auto by `-m` extension).
    #[arg(long, value_enum, default_value_t = InferBackend::Auto)]
    pub backend: InferBackend,
    #[command(flatten)]
    pub denoise: DenoiseOpts,
}

#[derive(Debug, Args)]
pub struct ServeArgs {
    #[command(flatten)]
    pub models: ModelOpts,
    /// Pitch shift in semitones.
    #[arg(short = 't', long, default_value_t = 0)]
    pub transpose: i32,
    /// Samples per input read chunk from stdin (16 kHz mono f32le).
    #[arg(long, default_value_t = 1600)]
    pub chunk: usize,
    /// Inference backend (default: auto by `-m` extension). Note: the Burn
    /// (GPU/cuda) generator is currently slower than realtime for `serve`.
    #[arg(long, value_enum, default_value_t = InferBackend::Auto)]
    pub backend: InferBackend,
    #[command(flatten)]
    pub denoise: DenoiseOpts,
}

/// De-hiss options shared by `convert` and `serve`. Off unless `--denoise` is
/// given; the tuning flags only take effect when it is. Defaults mirror
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
    /// Cache directory for downloaded assets
    /// [default: the Hugging Face cache, e.g. ~/.cache/huggingface/hub].
    #[arg(long)]
    pub cache_dir: Option<PathBuf>,
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
    /// Output path for the trained generator; a `.safetensors` file is written
    /// at this path (deploy directly with `rvc convert`, or export to ONNX).
    #[arg(short = 'o', long, default_value = "models/voice")]
    pub out: PathBuf,
    /// Generator output sample rate (40000 or 48000).
    #[arg(long, default_value_t = 48000)]
    pub model_sr: u32,
    /// Number of training epochs. Fine-tuning a warm-started base on a small
    /// (~30 min) corpus converges in a few dozen; stop early any time with
    /// Ctrl-C (the model is saved) or `q` in the dashboard.
    #[arg(short = 'e', long, default_value_t = 20)]
    pub epochs: u32,
    /// Mini-batch size (keep small for a 6 GB GPU; 2 is safe on an RTX 2060).
    #[arg(short = 'b', long, default_value_t = 2)]
    pub batch_size: usize,
    /// Speaker id embedded in the generator (single-speaker corpora use 0).
    #[arg(long, default_value_t = 0)]
    pub speaker_id: i64,
    /// Directory for the training log, checkpoints, and the saved weights.
    #[arg(long, default_value = "models/train")]
    pub work_dir: PathBuf,
    /// Continue a previous run: resume from a generator `.safetensors` written
    /// by an earlier `rvc train` (e.g. after Ctrl-C) instead of a pretrained
    /// base. If a matching `<stem>.disc.safetensors` sits next to it (written
    /// automatically), the discriminator resumes too; otherwise it falls back to
    /// `--pretrained-d`. Conflicts with `--pretrained-g`.
    #[arg(
        long,
        alias = "continue",
        value_name = "SAFETENSORS",
        conflicts_with = "pretrained_g"
    )]
    pub resume: Option<PathBuf>,
    /// Pretrained generator base (`f0G48k.pth`) to warm-start from (strongly
    /// recommended on a small corpus). Download the `f0G48k.pth`/`f0D48k.pth`
    /// bases from Hugging Face `lj1995/VoiceConversionWebUI`
    /// (`assets/pretrained_v2/`) and pass their paths, e.g.
    /// `models/pretrained/f0G48k.pth`.
    #[arg(long)]
    pub pretrained_g: Option<PathBuf>,
    /// Pretrained discriminator base (`f0D48k.pth`) to warm-start from — see
    /// `--pretrained-g` for where to get it.
    #[arg(long)]
    pub pretrained_d: Option<PathBuf>,
    /// ContentVec encoder ONNX [default: auto-downloaded from Hugging Face].
    #[arg(long)]
    pub content: Option<PathBuf>,
    /// RMVPE F0 ONNX [default: auto-downloaded from Hugging Face].
    #[arg(long)]
    pub rmvpe: Option<PathBuf>,
    /// Cache directory for downloaded feature-extractor assets
    /// [default: the Hugging Face cache, e.g. ~/.cache/huggingface/hub].
    #[arg(long)]
    pub cache_dir: Option<PathBuf>,
    /// Disable the interactive training dashboard (TUI) and log to stderr
    /// instead. The TUI is auto-disabled when stderr is not a terminal.
    #[arg(long)]
    pub no_tui: bool,
}
