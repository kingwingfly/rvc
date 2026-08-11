//! Decode anything to WAV at one rate.
//!
//! Every other stage resamples on the way through, because every one of them
//! decodes; this is the same operation with nothing else attached, and it
//! exists so a corpus can be given one normalising step. mp3, m4a, flac and
//! opus in, `<base>.wav` out, at whatever `--sr` says.
//!
//! # Channels
//!
//! Mono by default, because mono `f32` is what crosses every crate boundary in
//! this toolkit and a training corpus has no use for a second channel.
//!
//! The exception is worth stating rather than discovering: [`crate::separate`]
//! writes **44.1 kHz stereo** on purpose — MDX23C is stereo-native and folding
//! its stems to mono throws away one of the two cues it separates on — so
//! `separate` piped into a mono `resample` silently discards that. It is only a
//! loss if something downstream still wanted the pair; nothing in this
//! workspace does, since every engine decodes to mono anyway. [`Channels`]
//! makes it a choice either way, and `--help` says which one is being made.

use std::path::Path;

use anyhow::{Context, Result};
use audio_kit::{DecodeOptions, decode_path_stereo, write_wav_stereo_file};
use futures::{StreamExt, stream};

use crate::InputFile;

/// How many channels to write.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Channels {
    /// One channel: a stereo input is folded to `(L + R) / 2`.
    Mono,
    /// Two channels. A mono input is written to both, so the output is uniform
    /// whatever went in.
    Stereo,
}

/// What to convert, and to what.
#[derive(Debug, Clone, Copy)]
pub struct ResampleOptions {
    /// Sample rate to write at.
    pub sr: u32,
    /// Channel count to write.
    pub channels: Channels,
}

/// What one file produced.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ResampleReport {
    /// Duration written.
    pub out_secs: f64,
    /// Channels written — the report of a conversion says what it converted to.
    pub channels: Channels,
}

/// Convert one file into the output directory.
pub async fn file(
    input: &InputFile,
    opts: &ResampleOptions,
    output_dir: &Path,
) -> Result<ResampleReport> {
    let out_path = output_dir.join(format!("{}.wav", input.base));
    let out_secs = match opts.channels {
        Channels::Mono => {
            let samples = crate::decode_mono(&input.path, opts.sr).await?;
            let out_secs = samples.len() as f64 / opts.sr as f64;
            let out_stream = stream::iter([Ok::<_, audio_kit::AudioError>(samples)]);
            audio_kit::write_wav_file(&out_path, opts.sr, out_stream)
                .await
                .with_context(|| format!("writing {}", out_path.display()))?;
            out_secs
        }
        Channels::Stereo => {
            // Drained rather than piped straight through so the duration in the
            // report is a count of what was written, and so a decode error
            // arrives before the file is created.
            let decode = decode_path_stereo(&input.path, DecodeOptions::new(opts.sr));
            let mut decode = std::pin::pin!(decode);
            let mut chunks = Vec::new();
            let mut frames = 0usize;
            while let Some(chunk) = decode.next().await {
                let chunk = chunk.with_context(|| format!("decoding {}", input.path.display()))?;
                frames += chunk.frames();
                chunks.push(Ok::<_, audio_kit::AudioError>(chunk));
            }
            write_wav_stereo_file(&out_path, opts.sr, stream::iter(chunks))
                .await
                .with_context(|| format!("writing {}", out_path.display()))?;
            frames as f64 / opts.sr as f64
        }
    };

    Ok(ResampleReport {
        out_secs,
        channels: opts.channels,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const SR: u32 = 44_100;

    fn scratch(name: &str) -> std::path::PathBuf {
        let dir =
            std::env::temp_dir().join(format!("preprocess-core-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("create scratch dir");
        dir
    }

    /// A tone at `hz`, `secs` long, at this test's own rate.
    fn tone(secs: f32, hz: f64) -> Vec<f32> {
        let n = (secs * SR as f32) as usize;
        (0..n)
            .map(|i| (0.5 * (std::f64::consts::TAU * hz * i as f64 / SR as f64).sin()) as f32)
            .collect()
    }

    /// Two channels carrying *different* tones, so a fold and a swap are both
    /// visible: written at 44.1 kHz stereo, which is what `separate` produces
    /// and therefore the input this stage's channel question is about.
    async fn write_stereo(path: &Path, left: Vec<f32>, right: Vec<f32>) {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).expect("create parent");
        }
        let s = stream::iter([Ok::<_, audio_kit::AudioError>(audio_kit::StereoSamples {
            left,
            right,
        })]);
        write_wav_stereo_file(path, SR, s)
            .await
            .expect("write stereo");
    }

    async fn read_stereo(path: &Path, sr: u32) -> audio_kit::StereoSamples {
        let decode = decode_path_stereo(path, DecodeOptions::new(sr));
        let mut decode = std::pin::pin!(decode);
        let mut out = audio_kit::StereoSamples::default();
        while let Some(chunk) = decode.next().await {
            let chunk = chunk.expect("decode");
            out.left.extend(chunk.left);
            out.right.extend(chunk.right);
        }
        out
    }

    /// The rate changes, the audio does not: a 44.1 kHz input written at
    /// 16 kHz keeps its duration, which is the only thing a resample must not
    /// get wrong. (A frame count that tracked the *input* rate is exactly how a
    /// conversion ends up playing at the wrong speed.)
    #[tokio::test]
    async fn the_rate_changes_and_the_duration_does_not() {
        let dir = scratch("resample-rate");
        write_stereo(&dir.join("take.wav"), tone(2.0, 220.0), tone(2.0, 220.0)).await;

        let out = dir.join("out");
        std::fs::create_dir_all(&out).expect("create out");
        let files = crate::plan(std::slice::from_ref(&dir), &out).expect("plan");
        let opts = ResampleOptions {
            sr: 16_000,
            channels: Channels::Mono,
        };
        let report = file(&files[0], &opts, &out).await.expect("resample");
        assert!((report.out_secs - 2.0).abs() < 0.05, "{report:?}");

        let written = crate::decode_mono(&out.join("take.wav"), 16_000)
            .await
            .expect("decode");
        assert!(
            (written.len() as f64 / 16_000.0 - 2.0).abs() < 0.05,
            "{} samples",
            written.len()
        );
    }

    /// The channel question, pinned in both directions on the one input that
    /// raises it: a `separate` stem, whose two channels differ.
    #[tokio::test]
    async fn stereo_is_kept_when_asked_for_and_folded_when_not() {
        let dir = scratch("resample-channels");
        write_stereo(&dir.join("take.wav"), tone(1.0, 220.0), tone(1.0, 880.0)).await;

        let out = dir.join("out");
        std::fs::create_dir_all(&out).expect("create out");
        let files = crate::plan(std::slice::from_ref(&dir), &out).expect("plan");

        let opts = ResampleOptions {
            sr: SR,
            channels: Channels::Stereo,
        };
        let report = file(&files[0], &opts, &out).await.expect("resample");
        assert_eq!(report.channels, Channels::Stereo);
        let kept = read_stereo(&out.join("take.wav"), SR).await;
        let apart: f32 = kept
            .left
            .iter()
            .zip(&kept.right)
            .map(|(l, r)| (l - r).abs())
            .sum::<f32>()
            / kept.frames() as f32;
        assert!(apart > 0.1, "the two channels were merged: {apart}");

        let opts = ResampleOptions {
            sr: SR,
            channels: Channels::Mono,
        };
        file(&files[0], &opts, &out).await.expect("resample");
        let folded = read_stereo(&out.join("take.wav"), SR).await;
        // A mono WAV decodes to two identical channels, which is the observable
        // form of "one of the two cues is gone".
        let apart: f32 = folded
            .left
            .iter()
            .zip(&folded.right)
            .map(|(l, r)| (l - r).abs())
            .sum::<f32>()
            / folded.frames() as f32;
        assert!(
            apart < 1e-6,
            "a mono write kept two distinct channels: {apart}"
        );
    }
}
