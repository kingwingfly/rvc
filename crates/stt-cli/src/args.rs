//! `stt` — speech recognition as a Unix filter.
//!
//! Raw f32le mono PCM at 16 kHz on stdin, text on stdout, logs on stderr:
//!
//! ```sh
//! ffmpeg -i take.mp3 -f f32le -ar 16000 -ac 1 - | stt
//! ```
//!
//! `--format text` (the default) writes one line per segment so it pipes
//! straight into a translator or `voice tts`. `--format jsonl` adds timings and
//! the detected language, which is what a subtitle file or a TTS training
//! manifest needs.
//!
//! Unlike the voice-conversion filter this is **not** streaming: the whole
//! input is read before anything is transcribed, because segmentation looks for
//! silences across the recording and Whisper's own mel normalisation is per
//! 30 s window.

use anyhow::{Context, Result};
use clap::{Args, Subcommand, ValueEnum};
use cli_kit::CompletionsArgs;
use futures::StreamExt;
use std::path::PathBuf;
use stt_core::{DecodeOptions, TranscribeOptions};
use tokio::io::{AsyncWriteExt, BufWriter};

use crate::backend::load_transcriber;
pub use cli_kit::Backend;

/// The whole of the `stt` command tree, defined once and worn two ways: the
/// `stt` binary flattens it at its top level, `voice` nests it under an `stt`
/// subcommand. Every engine here has the same shape — the bare invocation is
/// the stdin→stdout filter, and everything else is a subcommand.
#[derive(Debug, Args)]
pub struct SttCli {
    #[command(flatten)]
    pub transcribe: SttArgs,
    #[command(subcommand)]
    pub command: Option<SttCommand>,
}

#[derive(Debug, Subcommand)]
pub enum SttCommand {
    /// Transcribe audio files to `<stem>.txt` (or `.jsonl`) in a directory.
    Convert(crate::convert::ConvertArgs),
    /// Prefetch the weights recognition needs, so the first run is offline.
    Download(DownloadArgs),
    /// Print a shell completion script (bash, zsh, fish, powershell, elvish).
    Completions(CompletionsArgs),
}

/// What a default `stt` run would fetch on demand, fetched up front instead.
///
/// Nothing else: the repo is the only thing recognition downloads, and there is
/// no training-only weight here to leave out.
#[derive(Debug, Args)]
pub struct DownloadArgs {
    /// Override the model repo as `owner/name`
    /// [default: openai/whisper-large-v3-turbo].
    #[arg(long, value_name = "OWNER/NAME")]
    pub repo: Option<String>,
    /// Directory the downloaded models are cached in. Shared by every engine
    /// unless `$STT_CACHE_DIR` (or `$VOICE_CACHE_DIR`) says otherwise.
    #[arg(long, default_value_os_t = hub_kit::cache_dir_for("STT_CACHE_DIR"))]
    pub cache_dir: PathBuf,
}

pub async fn download(args: DownloadArgs) -> Result<()> {
    let assets = hub_kit::fetch_whisper(args.repo.as_deref(), &args.cache_dir)
        .await
        .context("failed to fetch the Whisper model")?;
    // Where it landed is the point of the command, so it goes to stdout — it is
    // also what `--model` takes, which is how an offline machine is set up.
    println!("whisper: {}", assets.dir.display());
    Ok(())
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, ValueEnum)]
pub enum Format {
    /// One line of text per segment — pipes into anything.
    #[default]
    Text,
    /// One JSON object per line: start, end, text, language.
    Jsonl,
}

#[derive(Debug, Args)]
pub struct SttArgs {
    /// Directory holding a Hugging Face Whisper repo
    /// [default: auto-downloaded from Hugging Face].
    #[arg(short, long)]
    pub model: Option<PathBuf>,
    /// Override the model repo as `owner/name`
    /// [default: openai/whisper-large-v3-turbo].
    #[arg(long, value_name = "OWNER/NAME")]
    pub repo: Option<String>,
    /// Directory the downloaded models are cached in. Shared by every engine
    /// unless `$STT_CACHE_DIR` (or `$VOICE_CACHE_DIR`) says otherwise.
    #[arg(long, default_value_os_t = hub_kit::cache_dir_for("STT_CACHE_DIR"))]
    pub cache_dir: PathBuf,
    /// Output format.
    #[arg(long, value_enum, default_value_t = Format::Text)]
    pub format: Format,
    /// Force a language by ISO code (`en`, `zh`, `ja`) instead of detecting it.
    /// Worth setting on short or breathy clips, where detection is least sure.
    #[arg(short, long)]
    pub language: Option<String>,
    /// Translate to English rather than transcribing verbatim.
    #[arg(long)]
    pub translate: bool,
    /// Recognition backend; `auto` takes an ONNX export from `--model` if there
    /// is one, else the fastest Burn backend.
    #[arg(long, value_enum, default_value_t = Backend::Auto)]
    pub backend: Backend,
    /// Compute device: `auto`, `cpu`, `gpu`, `gpu:N`, `mps` or `vulkan`.
    #[arg(long, default_value = "auto", value_name = "DEVICE", value_parser = cli_kit::parse_device)]
    pub device: burn_kit::DeviceSpec,
    /// Samples per input read chunk. Read from stdin, so the `convert`
    /// subcommand — which decodes files instead — ignores it.
    #[arg(long, default_value_t = 16000)]
    pub chunk: usize,
    /// Energy floor in dBFS: quieter than this counts as a gap between
    /// utterances. Lower it to keep very soft passages in one segment.
    #[arg(long, default_value_t = -40.0)]
    pub silence_db: f32,
    /// Minimum silent-gap length (seconds) that counts as a segment boundary.
    #[arg(long, default_value_t = 0.5)]
    pub min_silence: f32,
    /// Drop any segment shorter than this (seconds).
    #[arg(long, default_value_t = 0.2)]
    pub min_clip: f32,
    /// Hard cap on segment length (seconds). Cannot exceed 30 — one segment must
    /// fit one encoder window.
    #[arg(long, default_value_t = 30.0)]
    pub max_clip: f32,
    /// Cap on tokens generated per segment. Raise it if dense speech is being
    /// cut off (a warning says so); lower it to bound a hallucination loop.
    #[arg(long, default_value_t = 224)]
    pub max_tokens: usize,
}

impl SttArgs {
    /// Reject values clap's types accept but the decoder cannot use.
    pub fn verify(&self) -> Result<()> {
        // 30 s is not a tuning choice: it is the width of Whisper's encoder
        // window, and a longer segment simply would not fit one.
        anyhow::ensure!(
            self.max_clip > 0.0 && self.max_clip <= 30.0,
            "--max-clip must be in (0, 30]: one segment has to fit Whisper's 30 s encoder window"
        );
        anyhow::ensure!(self.chunk > 0, "--chunk must be at least 1 sample");
        anyhow::ensure!(self.max_tokens > 0, "--max-tokens must be at least 1");
        anyhow::ensure!(self.min_clip >= 0.0, "--min-clip must not be negative");
        anyhow::ensure!(
            self.min_silence > 0.0,
            "--min-silence must be positive: a zero-length gap is every sample boundary"
        );
        anyhow::ensure!(
            self.silence_db <= 0.0,
            "--silence-db is dBFS, so it must be at most 0 (full scale); {} would \
             treat every sample as silence",
            self.silence_db
        );
        anyhow::ensure!(
            self.min_clip <= self.max_clip,
            "--min-clip ({}) exceeds --max-clip ({}), so every segment would be cut \
             to a length that is then discarded",
            self.min_clip,
            self.max_clip
        );
        Ok(())
    }

    /// The Whisper directory this run reads, fetched on demand when `--model`
    /// names none.
    pub async fn model_dir(&self) -> Result<PathBuf> {
        match &self.model {
            Some(dir) => Ok(dir.clone()),
            None => {
                tracing::info!("resolving Whisper weights from Hugging Face...");
                Ok(
                    hub_kit::fetch_whisper(self.repo.as_deref(), &self.cache_dir)
                        .await
                        .context("failed to fetch the Whisper model")?
                        .dir,
                )
            }
        }
    }

    /// Where to cut and what to ask the model for, as `stt-core` wants it.
    pub fn options(&self) -> TranscribeOptions {
        TranscribeOptions {
            slice: audio_kit::SliceOptions {
                silence_db: self.silence_db,
                min_silence: self.min_silence,
                min_clip: self.min_clip,
                max_clip: self.max_clip,
                ..Default::default()
            },
            decode: DecodeOptions {
                language: self.language.clone(),
                translate: self.translate,
                max_tokens: self.max_tokens,
            },
        }
    }
}

/// One segment as a line of output, in the requested format.
///
/// Shared with `convert`, so a file on disk and the same audio down the pipe
/// cannot come out differently formatted.
pub(crate) fn segment_line(format: Format, s: &stt_core::Segment) -> String {
    match format {
        Format::Text => format!("{}\n", s.text),
        Format::Jsonl => format!(
            "{{\"start\":{:.3},\"end\":{:.3},\"language\":\"{}\",\"text\":{}}}\n",
            s.start,
            s.end,
            s.language,
            json_string(&s.text)
        ),
    }
}

pub async fn transcribe(args: SttArgs) -> Result<()> {
    args.verify()?;

    let dir = args.model_dir().await?;

    let mut stt =
        tokio::task::block_in_place(|| load_transcriber(&dir, args.backend, args.device))?;

    // Buffered, not streamed: see the module docs.
    let mut input = Box::pin(audio_kit::read_f32le(tokio::io::stdin(), args.chunk));
    let mut audio: Vec<f32> = Vec::new();
    while let Some(chunk) = input.next().await {
        audio.extend_from_slice(&chunk.context("reading stdin")?);
    }
    tracing::info!(
        "transcribing {:.1} s of audio",
        audio.len() as f32 / stt_core::SAMPLE_RATE as f32
    );

    let opts = args.options();

    let segments = tokio::task::block_in_place(|| stt.transcribe(&audio, &opts))
        .context("transcription failed")?;

    let mut out = BufWriter::new(tokio::io::stdout());
    for s in &segments {
        out.write_all(segment_line(args.format, s).as_bytes())
            .await
            .context("writing stdout")?;
    }
    out.flush().await.context("final flush")?;
    tracing::info!("{} segments", segments.len());
    Ok(())
}

/// Escape a transcript as a JSON string.
///
/// Hand-rolled rather than pulling `serde` into the CLI for one field: the input
/// is model-generated text, so the escapes that matter are quotes, backslashes
/// and the control characters below 0x20.
fn json_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}
