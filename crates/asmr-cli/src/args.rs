//! Command-line argument definitions (clap derive).

use std::path::PathBuf;

use clap::{Args, Parser, Subcommand};

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
    /// Train an RVC model on a corpus via the Python/uv pipeline.
    Train(TrainArgs),
    /// Text-to-speech in the target voice (GPT-SoVITS via Python/uv sidecar).
    Tts(TtsArgs),
}

/// Shared options for locating the three ONNX models.
#[derive(Debug, Args, Clone)]
pub struct ModelOpts {
    /// Trained RVC generator ONNX (`voice.onnx`).
    #[arg(short = 'm', long)]
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
pub struct TtsArgs {
    #[command(subcommand)]
    pub command: TtsCommand,
}

#[derive(Debug, Subcommand)]
pub enum TtsCommand {
    /// Synthesize speech from text using a fine-tuned voice.
    Speak(TtsSpeakArgs),
    /// Few-shot fine-tune GPT-SoVITS on a target-voice corpus.
    Finetune(TtsFinetuneArgs),
}

/// Shared: path to the training project (uv) and passthrough args.
#[derive(Debug, Args, Clone)]
pub struct SidecarOpts {
    /// Path to the training/sidecar project (contains pyproject.toml for uv).
    #[arg(long, default_value = "training")]
    pub project: PathBuf,
    /// Extra arguments passed through to the sidecar entrypoint (after `--`).
    #[arg(last = true)]
    pub extra: Vec<String>,
}

#[derive(Debug, Args)]
pub struct TtsSpeakArgs {
    /// Text to synthesize.
    #[arg(short = 'x', long)]
    pub text: String,
    /// Reference audio clip that defines the voice/prosody.
    #[arg(short = 'r', long = "ref")]
    pub ref_audio: PathBuf,
    /// Output WAV path.
    #[arg(short = 'o', long, default_value = "tts.wav")]
    pub out: PathBuf,
    /// Fine-tuned model directory/checkpoint (default: sidecar's configured one).
    #[arg(short = 'm', long)]
    pub model: Option<PathBuf>,
    /// Language hint (e.g. `zh`, `en`, `auto`).
    #[arg(short = 'l', long, default_value = "auto")]
    pub lang: String,
    #[command(flatten)]
    pub sidecar: SidecarOpts,
}

#[derive(Debug, Args)]
pub struct TtsFinetuneArgs {
    /// Corpus: one or more target-voice audio files.
    #[arg(required = true)]
    pub data: Vec<PathBuf>,
    /// Output directory for the fine-tuned model.
    #[arg(short = 'o', long, default_value = "models/tts")]
    pub out: PathBuf,
    #[command(flatten)]
    pub sidecar: SidecarOpts,
}

#[derive(Debug, Args)]
pub struct TrainArgs {
    /// Corpus: one or more audio files of the target voice.
    #[arg(required = true)]
    pub data: Vec<PathBuf>,
    /// Output path for the trained generator ONNX.
    #[arg(short = 'o', long, default_value = "models/voice.onnx")]
    pub out: PathBuf,
    /// Path to the training project (contains pyproject.toml for uv).
    #[arg(long, default_value = "training")]
    pub project: PathBuf,
    /// Extra arguments passed through to the training entrypoint.
    #[arg(last = true)]
    pub extra: Vec<String>,
}
