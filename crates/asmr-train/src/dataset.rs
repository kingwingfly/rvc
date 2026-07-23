//! Corpus → training clips. Each clip holds the analysis features and the
//! ground-truth 48 kHz waveform, all aligned on the 100 Hz frame grid.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow};
use asmr_audio::{DecodeOptions, decode_path};
use asmr_vc::{DEFAULT_CHUNK, FeatureExtractor, f0_to_coarse};
use futures::StreamExt;

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

        tracing::info!("  {}: {} frames", path.display(), frames);
        clips.push(Clip {
            content: cflat,
            coarse,
            nsff0,
            gt,
            frames,
        });
    }

    anyhow::ensure!(
        !clips.is_empty(),
        "no usable training clips (need >= {min_frames} frames each)"
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
pub fn sample_batch(clips: &[Clip], batch: usize, window: usize, rng: &mut Rng) -> Batch {
    let mut phone = Vec::with_capacity(batch * window * CONTENT_DIM);
    let mut coarse = Vec::with_capacity(batch * window);
    let mut nsff0 = Vec::with_capacity(batch * window);
    let mut gt = Vec::with_capacity(batch * window * HOP);

    for _ in 0..batch {
        let clip = &clips[rng.below(clips.len())];
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

/// Tiny xorshift RNG (no external `rand` dep).
pub struct Rng(u64);

impl Rng {
    /// Seed the RNG.
    pub fn new(seed: u64) -> Self {
        Self(seed | 1)
    }

    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    /// Uniform integer in `0..n`.
    pub fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}
