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

/// ASMR voice toolkit: RVC voice conversion via ONNX Runtime.
#[derive(Debug, Parser)]
#[command(name = "asmr", version, about)]
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
    /// Override the ContentVec encoder ONNX (default: auto-download).
    #[arg(long)]
    pub content: Option<PathBuf>,
    /// Override the RMVPE F0 ONNX (default: auto-download).
    #[arg(long)]
    pub rmvpe: Option<PathBuf>,
    /// Cache directory for downloaded assets.
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
    /// Directory to write converted `<stem>.wav` files into.
    #[arg(short = 'o', long, default_value = ".")]
    pub output_dir: PathBuf,
    /// Pitch shift in semitones.
    #[arg(short = 't', long, default_value_t = 0)]
    pub transpose: i32,
    /// Inference backend (default: auto by `-m` extension).
    #[arg(long, value_enum, default_value_t = InferBackend::Auto)]
    pub backend: InferBackend,
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
    /// Cache directory for downloaded assets.
    #[arg(long)]
    pub cache_dir: Option<PathBuf>,
    /// Override ContentVec repo as `owner/name:file`.
    #[arg(long)]
    pub content: Option<String>,
    /// Override RMVPE repo as `owner/name:file`.
    #[arg(long)]
    pub rmvpe: Option<String>,
}

#[derive(Debug, Args)]
pub struct TrainArgs {
    /// Corpus: one or more audio files of the target voice.
    #[arg(required = true)]
    pub data: Vec<PathBuf>,
    /// Output path for the trained generator ONNX.
    #[arg(short = 'o', long, default_value = "models/voice.onnx")]
    pub out: PathBuf,
    /// Generator output sample rate (40000 or 48000).
    #[arg(long, default_value_t = 48000)]
    pub model_sr: u32,
    /// Number of training epochs.
    #[arg(short = 'e', long, default_value_t = 200)]
    pub epochs: u32,
    /// Mini-batch size (keep small for a 6 GB GPU).
    #[arg(short = 'b', long, default_value_t = 4)]
    pub batch_size: usize,
    /// Speaker id embedded in the generator (single-speaker corpora use 0).
    #[arg(long, default_value_t = 0)]
    pub speaker_id: i64,
    /// Directory for checkpoints and the intermediate weight file.
    #[arg(long, default_value = "models/train")]
    pub work_dir: PathBuf,
    /// Pretrained generator (`f0G48k.pth`) to warm-start from (recommended).
    #[arg(long)]
    pub pretrained_g: Option<PathBuf>,
    /// Pretrained discriminator (`f0D48k.pth`) to warm-start from (recommended).
    #[arg(long)]
    pub pretrained_d: Option<PathBuf>,
    /// Override the ContentVec encoder ONNX (default: auto-download).
    #[arg(long)]
    pub content: Option<PathBuf>,
    /// Override the RMVPE F0 ONNX (default: auto-download).
    #[arg(long)]
    pub rmvpe: Option<PathBuf>,
    /// Cache directory for downloaded feature-extractor assets.
    #[arg(long)]
    pub cache_dir: Option<PathBuf>,
}
