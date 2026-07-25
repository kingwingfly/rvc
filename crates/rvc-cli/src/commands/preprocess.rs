//! `rvc preprocess` — slice a corpus into clean per-sentence training clips.
//!
//! Removes *between-sentence dead-air only* (never quiet-but-present ASMR
//! content) so downstream `rvc train` draws its random windows from voiced
//! sentences instead of dead air. Directories in the input are expanded to
//! their audio files; each file is sliced and its segments written as
//! `<stem>_<NNN>.wav` at `--model-sr`.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use futures::{StreamExt, stream};
use rvc_audio::{DecodeOptions, SliceOptions, decode_paths, slice, write_wav_file};

use crate::args::PreprocessArgs;

/// Audio extensions we expand directories into (case-insensitive).
const AUDIO_EXTS: &[&str] = &["mp3", "wav", "flac", "m4a", "ogg", "opus", "aac", "wma"];

pub async fn run(args: PreprocessArgs) -> Result<()> {
    let opts = SliceOptions {
        silence_db: args.silence_db,
        min_silence: args.min_silence,
        min_clip: args.min_clip,
        max_clip: args.max_clip,
        pad: args.pad,
    };

    let files = expand_inputs(&args.input)?;
    if files.is_empty() {
        anyhow::bail!("no audio files found in the given input paths");
    }

    tokio::fs::create_dir_all(&args.output_dir)
        .await
        .with_context(|| format!("creating output dir {}", args.output_dir.display()))?;

    let sr = args.model_sr;
    let mut total_clips = 0usize;
    let mut total_in_secs = 0.0f64;
    let mut total_kept_secs = 0.0f64;

    for file in &files {
        let samples = decode_mono(file, sr).await?;
        let in_secs = samples.len() as f64 / sr as f64;
        total_in_secs += in_secs;

        let segments = slice(&samples, sr, &opts);
        let stem = file.file_stem().and_then(|s| s.to_str()).unwrap_or("clip");

        let mut kept_secs = 0.0f64;
        for (i, (start, end)) in segments.iter().enumerate() {
            let mut clip = samples[*start..*end].to_vec();
            if args.normalize {
                peak_normalize(&mut clip, 0.95);
            }
            kept_secs += clip.len() as f64 / sr as f64;
            let out_path = args.output_dir.join(format!("{stem}_{i:03}.wav"));
            let out_stream = stream::iter([Ok::<_, rvc_audio::AudioError>(clip)]);
            write_wav_file(&out_path, sr, out_stream)
                .await
                .with_context(|| format!("writing {}", out_path.display()))?;
        }

        total_clips += segments.len();
        total_kept_secs += kept_secs;
        println!(
            "{}: {} clips  ({:.1}s in -> {:.1}s kept)",
            file.display(),
            segments.len(),
            in_secs,
            kept_secs,
        );
    }

    println!(
        "total: {} clips from {} files  ({:.1}s in -> {:.1}s kept)  -> {}",
        total_clips,
        files.len(),
        total_in_secs,
        total_kept_secs,
        args.output_dir.display(),
    );
    Ok(())
}

/// Expand input paths: files are kept as-is; directories yield their audio
/// files (by extension, case-insensitive). Result is sorted and de-duplicated.
fn expand_inputs(inputs: &[PathBuf]) -> Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    for path in inputs {
        if path.is_dir() {
            let entries = std::fs::read_dir(path)
                .with_context(|| format!("reading directory {}", path.display()))?;
            for entry in entries {
                let entry = entry?;
                let p = entry.path();
                if p.is_file() && has_audio_ext(&p) {
                    files.push(p);
                }
            }
        } else {
            files.push(path.clone());
        }
    }
    files.sort();
    files.dedup();
    Ok(files)
}

/// Whether `path`'s extension is a recognised audio format (case-insensitive).
fn has_audio_ext(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .map(|e| {
            let e = e.to_ascii_lowercase();
            AUDIO_EXTS.contains(&e.as_str())
        })
        .unwrap_or(false)
}

/// Decode a file to mono `f32` at `sr`, draining the whole stream into a `Vec`.
async fn decode_mono(input: &Path, sr: u32) -> Result<Vec<f32>> {
    let decode = decode_paths(vec![input.to_path_buf()], DecodeOptions::new(sr));
    let mut decode = std::pin::pin!(decode);
    let mut out = Vec::new();
    while let Some(chunk) = decode.next().await {
        out.extend(chunk.with_context(|| format!("decoding {}", input.display()))?);
    }
    Ok(out)
}

/// Peak-normalize `samples` in place to `target` full-scale. No-op if silent.
fn peak_normalize(samples: &mut [f32], target: f32) {
    let peak = samples.iter().fold(0.0f32, |m, x| m.max(x.abs()));
    if peak > 1e-9 {
        let gain = target / peak;
        for s in samples.iter_mut() {
            *s *= gain;
        }
    }
}
