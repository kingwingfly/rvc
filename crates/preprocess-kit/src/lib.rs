//! The `preprocess` subcommand: slice a corpus into clean per-sentence clips.
//!
//! Removes *between-sentence dead-air only*, never quiet-but-present content, so
//! a training run draws its windows from voiced sentences instead of from
//! silence. Directories in the input are expanded to their audio files; each
//! file is sliced and its segments written as `<stem>_<NNN>.wav`.
//!
//! Shared rather than owned by an engine: slicing a corpus is decode, find the
//! gaps, write WAVs, and knows nothing about what will be trained on the result.
//! `rvc` needs it so random windows do not land in dead air; `tts` needs it so a
//! clip is one utterance with one transcript, which is what makes
//! `preprocess` → `stt` → `train` a corpus.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use audio_kit::{DecodeOptions, SliceOptions, decode_paths, slice, write_wav_file};
use clap::Args;
use futures::{StreamExt, stream};

#[derive(Debug, Args)]
pub struct PreprocessArgs {
    /// Input audio files and/or directories (directories are expanded to their
    /// audio files: mp3, wav, flac, m4a, ogg, opus, aac, wma).
    #[arg(required = true)]
    pub input: Vec<PathBuf>,
    /// Directory to write the sliced `<stem>_<NNN>.wav` clips into.
    #[arg(short = 'o', long, default_value = "dataset")]
    pub output_dir: PathBuf,
    /// Sample rate of the written clips. Training re-decodes them at whatever
    /// rate it needs, so this only decides what is on disk; matching the rate
    /// you will train at avoids a resample.
    #[arg(long, alias = "model-sr", default_value_t = 48000)]
    pub sr: u32,
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

impl PreprocessArgs {
    /// Reject a slicer configuration that would silently produce no clips.
    pub fn verify(&self) -> Result<()> {
        anyhow::ensure!(self.sr > 0, "--sr must be positive");
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
}

/// Audio extensions we expand directories into (case-insensitive).
const AUDIO_EXTS: &[&str] = &["mp3", "wav", "flac", "m4a", "ogg", "opus", "aac", "wma"];

pub async fn run(args: PreprocessArgs) -> Result<()> {
    args.verify()?;
    let opts = SliceOptions {
        silence_db: args.silence_db,
        min_silence: args.min_silence,
        min_clip: args.min_clip,
        max_clip: args.max_clip,
        pad: args.pad,
    };

    tokio::fs::create_dir_all(&args.output_dir)
        .await
        .with_context(|| format!("creating output dir {}", args.output_dir.display()))?;

    let files = expand_inputs(&args.input, &args.output_dir)?;
    if files.is_empty() {
        anyhow::bail!("no audio files found in the given input paths");
    }

    let sr = args.sr;
    let mut total_clips = 0usize;
    let mut total_in_secs = 0.0f64;
    let mut total_kept_secs = 0.0f64;
    let mut failed = 0usize;
    // Files sharing a stem (e.g. `a/song.mp3` and `b/song.mp3`) would otherwise
    // overwrite each other's clips; disambiguate the output base per stem.
    let mut used_stems: HashMap<String, usize> = HashMap::new();

    for file in &files {
        // One undecodable file must not abort the whole batch.
        let samples = match decode_mono(file, sr).await {
            Ok(s) => s,
            Err(e) => {
                eprintln!("skipping {}: {e:#}", file.display());
                failed += 1;
                continue;
            }
        };
        let in_secs = samples.len() as f64 / sr as f64;
        total_in_secs += in_secs;

        let segments = slice(&samples, sr, &opts);

        let stem = file
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("clip")
            .to_string();
        let seen = used_stems.entry(stem.clone()).or_insert(0);
        let base = if *seen == 0 {
            stem.clone()
        } else {
            format!("{stem}__{seen}")
        };
        *seen += 1;

        let mut kept_secs = 0.0f64;
        for (i, (start, end)) in segments.iter().enumerate() {
            let mut clip = samples[*start..*end].to_vec();
            if args.normalize {
                peak_normalize(&mut clip, 0.95);
            }
            kept_secs += clip.len() as f64 / sr as f64;
            let out_path = args.output_dir.join(format!("{base}_{i:03}.wav"));
            let out_stream = stream::iter([Ok::<_, audio_kit::AudioError>(clip)]);
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

    let skipped = if failed > 0 {
        format!(" ({failed} skipped)")
    } else {
        String::new()
    };
    println!(
        "total: {} clips from {} files{}  ({:.1}s in -> {:.1}s kept)  -> {}",
        total_clips,
        files.len() - failed,
        skipped,
        total_in_secs,
        total_kept_secs,
        args.output_dir.display(),
    );
    Ok(())
}

/// Expand input paths: files are kept as-is; directories are walked
/// **recursively** for audio files (by extension, case-insensitive). The output
/// dir is skipped so a re-run never re-ingests its own clips. Result is sorted
/// and de-duplicated.
fn expand_inputs(inputs: &[PathBuf], output_dir: &Path) -> Result<Vec<PathBuf>> {
    let skip = std::fs::canonicalize(output_dir).ok();
    let mut files = Vec::new();
    for path in inputs {
        if path.is_dir() {
            collect_dir(path, skip.as_deref(), &mut files)?;
        } else {
            files.push(path.clone());
        }
    }
    files.sort();
    files.dedup();
    Ok(files)
}

/// Recursively collect audio files under `dir`, skipping the output dir.
fn collect_dir(dir: &Path, skip: Option<&Path>, out: &mut Vec<PathBuf>) -> Result<()> {
    if let (Some(skip), Ok(here)) = (skip, std::fs::canonicalize(dir)) {
        if here == skip {
            return Ok(());
        }
    }
    let entries =
        std::fs::read_dir(dir).with_context(|| format!("reading directory {}", dir.display()))?;
    for entry in entries {
        let p = entry?.path();
        if p.is_dir() {
            collect_dir(&p, skip, out)?;
        } else if p.is_file() && has_audio_ext(&p) {
            out.push(p);
        }
    }
    Ok(())
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
