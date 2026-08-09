//! Slice a recording into clean per-utterance clips.
//!
//! Removes *between-sentence dead-air only*, never quiet-but-present content,
//! so a training run draws its windows from voiced sentences instead of from
//! silence. Each input file becomes `<base>_<NNN>.wav` in the output directory.
//!
//! Energy is used only to find long silent gaps — it never gates
//! quiet-but-present sound, which is the whole point for soft, breathy,
//! close-mic material: the softest passages are content, not noise floor. The
//! slicing itself is [`audio_kit::slice`], shared with the streaming segmenter
//! that speech recognition cuts on, so the two cannot disagree about where a
//! sentence ends.

use std::path::Path;

use anyhow::{Context, Result};
use audio_kit::{SliceOptions, write_wav_file};
use futures::stream;

use crate::InputFile;

/// What to cut, and what to write.
#[derive(Debug, Clone)]
pub struct ClipOptions {
    /// Sample rate of the written clips. Training re-decodes them at whatever
    /// rate it needs, so this only decides what is on disk.
    pub sr: u32,
    /// Where to cut.
    pub slice: SliceOptions,
    /// Peak-normalize each written clip to ~0.95 full-scale.
    pub normalize: bool,
}

/// What one file produced.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ClipReport {
    /// Clips written.
    pub clips: usize,
    /// Duration decoded from the input.
    pub in_secs: f64,
    /// Duration written out across all clips — the rest was dead air.
    pub kept_secs: f64,
}

/// Slice one file into the output directory.
///
/// The caller creates `output_dir` (see [`crate::plan`], which needs it to
/// exist) and decides what to do with a file that will not decode — one bad
/// recording in a corpus of hundreds must not abort the batch.
pub async fn file(input: &InputFile, opts: &ClipOptions, output_dir: &Path) -> Result<ClipReport> {
    let samples = crate::decode_mono(&input.path, opts.sr).await?;
    let in_secs = samples.len() as f64 / opts.sr as f64;
    let segments = audio_kit::slice(&samples, opts.sr, &opts.slice);

    let mut kept_secs = 0.0f64;
    for (i, (start, end)) in segments.iter().enumerate() {
        let mut clip = samples[*start..*end].to_vec();
        if opts.normalize {
            peak_normalize(&mut clip, 0.95);
        }
        kept_secs += clip.len() as f64 / opts.sr as f64;
        let out_path = output_dir.join(format!("{}_{i:03}.wav", input.base));
        let out_stream = stream::iter([Ok::<_, audio_kit::AudioError>(clip)]);
        write_wav_file(&out_path, opts.sr, out_stream)
            .await
            .with_context(|| format!("writing {}", out_path.display()))?;
    }

    Ok(ClipReport {
        clips: segments.len(),
        in_secs,
        kept_secs,
    })
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

#[cfg(test)]
mod tests {
    use super::*;

    const SR: u32 = 48_000;

    #[test]
    fn normalizing_lifts_the_peak_to_the_target() {
        let mut s = vec![0.5, -0.25, 0.0, 0.5];
        peak_normalize(&mut s, 0.95);
        assert!((s[0] - 0.95).abs() < 1e-6, "{s:?}");
        assert!((s[1] + 0.475).abs() < 1e-6, "{s:?}");
        // Gain, not a rewrite: the shape is untouched.
        assert_eq!(s[2], 0.0);
    }

    #[test]
    fn a_silent_clip_is_left_alone() {
        // Without the guard this divides by ~0 and writes infinities into a
        // WAV — silence in, silence out is the only sane answer.
        let mut s = vec![0.0, 0.0, 0.0];
        peak_normalize(&mut s, 0.95);
        assert_eq!(s, vec![0.0, 0.0, 0.0]);

        let mut tiny = vec![1e-12, -1e-12];
        peak_normalize(&mut tiny, 0.95);
        assert_eq!(tiny, vec![1e-12, -1e-12]);
    }

    /// A 220 Hz sine at half scale — RMS about -9 dBFS, well above the -40 dB
    /// floor. A real tone rather than the ±0.5 alternation `audio_kit`'s own
    /// slicer tests use, because this signal makes a round trip through a WAV
    /// file and a sample at Nyquist is the one thing that would not survive it.
    fn voiced(secs: f32) -> Vec<f32> {
        let n = (secs * SR as f32) as usize;
        (0..n)
            .map(|i| (0.5 * (std::f64::consts::TAU * 220.0 * i as f64 / SR as f64).sin()) as f32)
            .collect()
    }

    fn silence(secs: f32) -> Vec<f32> {
        vec![0.0; (secs * SR as f32) as usize]
    }

    fn scratch(name: &str) -> std::path::PathBuf {
        let dir =
            std::env::temp_dir().join(format!("preprocess-core-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("create scratch dir");
        dir
    }

    async fn write_tone(path: &Path, samples: Vec<f32>) {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).expect("create parent");
        }
        let s = stream::iter([Ok::<_, audio_kit::AudioError>(samples)]);
        write_wav_file(path, SR, s).await.expect("write tone");
    }

    /// The one check that exercises the whole stage against ffmpeg: two files
    /// with the same stem in different directories must not overwrite each
    /// other's clips, and the cuts must land on the silence rather than
    /// anywhere in the tone.
    #[tokio::test]
    async fn two_files_sharing_a_stem_both_survive_the_batch() {
        let dir = scratch("clip-batch");
        let mut sig = voiced(2.0);
        sig.extend(silence(1.0));
        sig.extend(voiced(2.0));
        write_tone(&dir.join("a/take.wav"), sig.clone()).await;
        write_tone(&dir.join("b/take.wav"), sig).await;

        let out = dir.join("out");
        std::fs::create_dir_all(&out).expect("create out");
        let files = crate::plan(&[dir.clone()], &out).expect("plan");
        assert_eq!(files.len(), 2);

        let opts = ClipOptions {
            sr: SR,
            slice: SliceOptions {
                silence_db: -40.0,
                min_silence: 0.3,
                min_clip: 1.0,
                max_clip: 0.0,
                pad: 0.15,
            },
            normalize: false,
        };
        for f in &files {
            let report = file(f, &opts, &out).await.expect("clip");
            assert_eq!(report.clips, 2, "one clip per voiced run");
            assert!(report.in_secs > 4.9 && report.in_secs < 5.1, "{report:?}");
            // The 1 s gap goes, minus the padding kept at each edge of it.
            assert!(report.kept_secs < report.in_secs, "{report:?}");
            assert!(report.kept_secs > 4.0, "{report:?}");
        }

        let mut written: Vec<_> = std::fs::read_dir(&out)
            .expect("read out")
            .map(|e| e.expect("entry").file_name().to_string_lossy().into_owned())
            .collect();
        written.sort();
        assert_eq!(
            written,
            vec![
                "take__1_000.wav",
                "take__1_001.wav",
                "take_000.wav",
                "take_001.wav",
            ]
        );
    }
}
