//! `stt convert` — batch file transcription, one transcript per input.
//!
//! The filter reads PCM off stdin, which makes a whole directory a shell loop
//! that reloads Whisper for every take. This is the same recognition with the
//! model loaded once and ffmpeg doing the decoding, so any container works.

use anyhow::{Context, Result};
use audio_kit::DecodeOptions;
use clap::Args;
use futures::StreamExt;
use std::path::{Path, PathBuf};

use crate::args::{Format, SttArgs, segment_line};
use crate::backend::load_transcriber;

#[derive(Debug, Args)]
pub struct ConvertArgs {
    #[command(flatten)]
    pub stt: SttArgs,
    /// One or more input audio files (mp3, wav, ...).
    #[arg(required = true)]
    pub input: Vec<PathBuf>,
    /// Directory to write `<stem>.txt` — or `<stem>.jsonl` — files into.
    #[arg(short = 'o', long, default_value = ".")]
    pub output_dir: PathBuf,
}

pub async fn run(args: ConvertArgs) -> Result<()> {
    args.stt.verify()?;
    let dir = args.stt.model_dir().await?;

    // One loaded model (GPU/ORT init is expensive), reused across files.
    let mut stt =
        tokio::task::block_in_place(|| load_transcriber(&dir, args.stt.backend, args.stt.device))?;
    let opts = args.stt.options();

    tokio::fs::create_dir_all(&args.output_dir)
        .await
        .with_context(|| format!("creating output dir {}", args.output_dir.display()))?;

    // The extension has to match what is inside, so it follows `--format`.
    let ext = match args.stt.format {
        Format::Text => "txt",
        Format::Jsonl => "jsonl",
    };

    for input in &args.input {
        let audio = decode(input).await?;
        tracing::info!(
            "transcribing {} ({:.1} s)",
            input.display(),
            audio.len() as f32 / stt_core::SAMPLE_RATE as f32
        );
        let segments = tokio::task::block_in_place(|| stt.transcribe(&audio, &opts))
            .with_context(|| format!("transcribing {}", input.display()))?;

        let stem = input.file_stem().and_then(|s| s.to_str()).unwrap_or("out");
        let out_path = args.output_dir.join(format!("{stem}.{ext}"));
        let text: String = segments
            .iter()
            .map(|s| segment_line(args.stt.format, s))
            .collect();
        tokio::fs::write(&out_path, text)
            .await
            .with_context(|| format!("writing {}", out_path.display()))?;
        tracing::info!("wrote {} ({} segments)", out_path.display(), segments.len());
    }
    Ok(())
}

/// Decode a file to mono f32 at Whisper's analysis rate.
async fn decode(input: &Path) -> Result<Vec<f32>> {
    let stream = audio_kit::decode_path(input, DecodeOptions::new(stt_core::SAMPLE_RATE));
    let mut stream = std::pin::pin!(stream);
    let mut audio = Vec::new();
    while let Some(chunk) = stream.next().await {
        audio.extend_from_slice(&chunk.with_context(|| format!("decoding {}", input.display()))?);
    }
    Ok(audio)
}
