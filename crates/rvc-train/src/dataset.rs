//! Corpus → training clips. Each clip holds the analysis features and the
//! ground-truth 48 kHz waveform, all aligned on the 100 Hz frame grid.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow};
use audio_kit::{DecodeOptions, decode_path};
use futures::StreamExt;
use rvc_core::{DEFAULT_CHUNK, FeatureExtractor, f0_to_coarse};
use train_kit::Rng;

/// Samples per latent frame (48 kHz, hop 480).
pub const HOP: usize = 480;
/// ContentVec feature width.
pub const CONTENT_DIM: usize = 768;

/// One preprocessed clip, aligned to `frames` on the 100 Hz grid.
pub struct Clip {
    /// Flattened content features, `frames * 768`.
    pub content: Vec<f32>,
    /// Coarse pitch ids, `frames`.
    pub coarse: Vec<i64>,
    /// F0 in Hz, `frames`.
    pub nsff0: Vec<f32>,
    /// Ground-truth waveform at the model rate, `frames * HOP`.
    pub gt: Vec<f32>,
    /// Number of aligned frames.
    pub frames: usize,
    /// Noise-floor SNR (linear): the clip's loud-percentile frame RMS over its
    /// quiet-percentile (noise-floor) frame RMS. A *ratio*, so a soft-but-clean
    /// take still scores high — unlike a loudness measure, which would
    /// penalise exactly the breathy passages we want to keep. Used only for
    /// optional SNR-weighted sampling ([`clip_weights`]).
    pub snr: f32,
}

/// Noise-floor SNR of a model-rate waveform: loud-percentile frame RMS divided
/// by quiet-percentile frame RMS. Frames are `HOP` samples (the training grid).
fn frame_snr(gt: &[f32]) -> f32 {
    let mut rms: Vec<f32> = gt
        .chunks_exact(HOP)
        .map(|f| (f.iter().map(|s| s * s).sum::<f32>() / HOP as f32).sqrt())
        .collect();
    if rms.len() < 8 {
        return 1.0;
    }
    rms.sort_by(|a, b| a.total_cmp(b));
    let pct = |p: f32| rms[((rms.len() - 1) as f32 * p) as usize];
    let floor = pct(0.10).max(1e-6); // noise floor (quiet frames)
    let signal = pct(0.75); // representative "loud" level
    (signal / floor).max(1.0)
}

/// Per-clip cumulative sampling weights for `snr^alpha`, or `None` for uniform
/// sampling (`alpha <= 0`). Returned as a cumulative distribution over clips so
/// [`sample_batch`] can pick with one binary search.
pub fn clip_weights(clips: &[Clip], alpha: f32) -> Option<Vec<f32>> {
    if alpha <= 0.0 {
        return None;
    }
    let mut cum = 0.0f32;
    let cdf: Vec<f32> = clips
        .iter()
        .map(|c| {
            cum += c.snr.powf(alpha);
            cum
        })
        .collect();
    (cum > 0.0).then_some(cdf)
}

/// Decode + extract features for every corpus file.
pub async fn prepare_clips(
    data: &[PathBuf],
    content_onnx: &Path,
    rmvpe_onnx: &Path,
    model_sr: u32,
    min_frames: usize,
) -> Result<Vec<Clip>> {
    let mut fx = FeatureExtractor::load(content_onnx, rmvpe_onnx)
        .map_err(|e| anyhow!("loading ContentVec/RMVPE ONNX: {e}"))?;

    let mut clips = Vec::new();
    for path in data {
        let wav16k = decode_mono(path, FeatureExtractor::ANALYSIS_SR).await?;
        anyhow::ensure!(!wav16k.is_empty(), "{} decoded to nothing", path.display());
        let gt = decode_mono(path, model_sr).await?;

        // Chunked extraction bounds ONNX memory on long clips; content is
        // already upsampled to the 100 Hz F0 grid and aligned to f0.
        let (content, f0) = fx
            .extract_aligned(&wav16k, DEFAULT_CHUNK)
            .map_err(|e| anyhow!("features for {}: {e}", path.display()))?;

        let frames = content.len().min(gt.len() / HOP);
        if frames < min_frames {
            tracing::warn!(
                "skipping {} ({} frames < {})",
                path.display(),
                frames,
                min_frames
            );
            continue;
        }

        let coarse = f0_to_coarse(&f0[..frames]);
        let nsff0 = f0[..frames].to_vec();
        let mut cflat = Vec::with_capacity(frames * CONTENT_DIM);
        for row in &content[..frames] {
            cflat.extend_from_slice(row);
        }
        let gt = gt[..frames * HOP].to_vec();

        let snr = frame_snr(&gt);
        tracing::debug!("  {}: {} frames (snr {:.1})", path.display(), frames, snr);
        clips.push(Clip {
            content: cflat,
            coarse,
            nsff0,
            gt,
            frames,
            snr,
        });
    }

    anyhow::ensure!(
        !clips.is_empty(),
        "no usable training clips (need >= {min_frames} frames each)"
    );

    let frames: usize = clips.iter().map(|c| c.frames).sum();
    let mut snrs: Vec<f32> = clips.iter().map(|c| c.snr).collect();
    snrs.sort_by(f32::total_cmp);
    tracing::info!(
        "corpus: {} clips, {:.1} min audio, snr {:.0}/{:.0}/{:.0} dB (min/median/max){}",
        clips.len(),
        (frames * HOP) as f64 / 16_000.0 / 60.0,
        snrs[0],
        snrs[snrs.len() / 2],
        snrs[snrs.len() - 1],
        match data.len() - clips.len() {
            0 => String::new(),
            n => format!("; {n} skipped as too short"),
        }
    );
    Ok(clips)
}

/// A batch of fixed-length windows, as flat CPU buffers ready for tensors.
pub struct Batch {
    /// `batch * window * 768`.
    pub phone: Vec<f32>,
    /// `batch * window`.
    pub coarse: Vec<i64>,
    /// `batch * window`.
    pub nsff0: Vec<f32>,
    /// `batch * window * HOP`.
    pub gt: Vec<f32>,
}

/// Sample `batch` random `window`-frame windows from the clips.
///
/// `cdf`, when `Some`, is a cumulative clip-weight distribution (see
/// [`clip_weights`]) used to bias which clip each window is drawn from; `None`
/// samples clips uniformly. Windows within a clip are always uniform.
pub fn sample_batch(
    clips: &[Clip],
    batch: usize,
    window: usize,
    rng: &mut Rng,
    cdf: Option<&[f32]>,
) -> Batch {
    let mut phone = Vec::with_capacity(batch * window * CONTENT_DIM);
    let mut coarse = Vec::with_capacity(batch * window);
    let mut nsff0 = Vec::with_capacity(batch * window);
    let mut gt = Vec::with_capacity(batch * window * HOP);

    for _ in 0..batch {
        let idx = match cdf {
            Some(c) => weighted(rng, c),
            None => rng.below(clips.len()),
        };
        let clip = &clips[idx];
        let start = rng.below(clip.frames - window + 1);
        phone.extend_from_slice(&clip.content[start * CONTENT_DIM..(start + window) * CONTENT_DIM]);
        coarse.extend_from_slice(&clip.coarse[start..start + window]);
        nsff0.extend_from_slice(&clip.nsff0[start..start + window]);
        gt.extend_from_slice(&clip.gt[start * HOP..(start + window) * HOP]);
    }
    Batch {
        phone,
        coarse,
        nsff0,
        gt,
    }
}

/// Decode one file to a single mono `f32` buffer at `sample_rate`.
async fn decode_mono(path: &Path, sample_rate: u32) -> Result<Vec<f32>> {
    let mut stream = std::pin::pin!(decode_path(path, DecodeOptions::new(sample_rate)));
    let mut out = Vec::new();
    while let Some(chunk) = stream.next().await {
        out.extend(chunk.with_context(|| format!("decoding {}", path.display()))?);
    }
    Ok(out)
}

/// Sample an index from a cumulative weight distribution (`cdf` ascending, last
/// element the total). Draws a point in `[0, total)` and returns the first bucket
/// whose cumulative weight exceeds it.
///
/// Here rather than on [`Rng`] because SNR-biased clip sampling is this
/// trainer's alone — `s2` draws its utterances uniformly.
fn weighted(rng: &mut Rng, cdf: &[f32]) -> usize {
    let total = *cdf.last().expect("non-empty cdf") as f64;
    let point = (rng.unit() * total) as f32;
    cdf.partition_point(|&c| c <= point).min(cdf.len() - 1)
}
