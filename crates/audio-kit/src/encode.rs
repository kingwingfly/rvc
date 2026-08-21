//! Minimal WAV encoding for batch `convert` output.
//!
//! We write 32-bit IEEE-float WAV (format tag 3) so converted audio is
//! preserved losslessly for A/B listening. mp3 output can be layered on later
//! via ffmpeg; the realtime path never touches this module (it emits raw PCM).
//!
//! Mono is what every engine writes. [`write_wav_stereo`] is the counterpart of
//! [`decode_path_stereo`](crate::decode::decode_path_stereo), so a two-channel
//! signal that a separator produced can reach a file without being flattened
//! first; the two share one header writer so the RIFF layout cannot drift.

use std::path::Path;

use futures::{Stream, StreamExt};
use tokio::fs::File;
use tokio::io::{AsyncWrite, AsyncWriteExt, BufWriter};

use crate::error::{AudioError, Result};
use crate::{Samples, StereoSamples};

/// Collect a stream of mono `f32` chunks and write them to `path` as a
/// 32-bit float mono WAV at `sample_rate`.
pub async fn write_wav_file<S>(path: impl AsRef<Path>, sample_rate: u32, stream: S) -> Result<()>
where
    S: Stream<Item = Result<Samples>> + Unpin,
{
    let file = File::create(path).await?;
    write_wav(BufWriter::new(file), sample_rate, stream).await
}

/// Write a stream of mono `f32` chunks to `writer` as a 32-bit float mono WAV.
///
/// The whole signal is buffered first so the RIFF/`data` sizes can be filled
/// in; the voice clips handled here are short enough that this is not a concern.
pub async fn write_wav<W, S>(writer: W, sample_rate: u32, mut stream: S) -> Result<()>
where
    W: AsyncWrite + Unpin,
    S: Stream<Item = Result<Samples>> + Unpin,
{
    let mut samples: Vec<f32> = Vec::new();
    while let Some(item) = stream.next().await {
        samples.extend_from_slice(&item?);
    }
    write_wav_pcm(writer, sample_rate, 1, &samples).await
}

/// Collect a stream of [`StereoSamples`] chunks and write them to `path` as a
/// 32-bit float **stereo** WAV at `sample_rate`.
pub async fn write_wav_stereo_file<S>(
    path: impl AsRef<Path>,
    sample_rate: u32,
    stream: S,
) -> Result<()>
where
    S: Stream<Item = Result<StereoSamples>> + Unpin,
{
    let file = File::create(path).await?;
    write_wav_stereo(BufWriter::new(file), sample_rate, stream).await
}

/// Write a stream of [`StereoSamples`] chunks to `writer` as a 32-bit float
/// stereo WAV.
///
/// The channels are interleaved here and nowhere earlier: a WAV's `data` chunk
/// is `[L, R, L, R, …]` by definition, so this is the one place the layout is
/// not a choice. Everything upstream keeps the planar pair, which is what an
/// STFT wants — see [`StereoSamples`].
///
/// Fails with [`AudioError::ChannelLengthMismatch`] if a chunk's two channels
/// disagree in length.
pub async fn write_wav_stereo<W, S>(writer: W, sample_rate: u32, mut stream: S) -> Result<()>
where
    W: AsyncWrite + Unpin,
    S: Stream<Item = Result<StereoSamples>> + Unpin,
{
    let mut interleaved: Vec<f32> = Vec::new();
    while let Some(item) = stream.next().await {
        let chunk = item?;
        // This is the one place `StereoSamples`'s equal-length invariant is
        // *relied on*, and the type's own guard is a `debug_assert!` inside
        // `frames()` that a release build drops and this function never calls.
        // So it is checked rather than trusted: interleaving a mismatched pair
        // has no harmless reading — truncating drops audio from the tail of the
        // longer channel, and padding shifts every frame after the shortfall
        // against the other channel. Refusing says which chunk was wrong.
        if chunk.left.len() != chunk.right.len() {
            return Err(AudioError::ChannelLengthMismatch {
                left: chunk.left.len(),
                right: chunk.right.len(),
            });
        }
        for (l, r) in chunk.left.iter().zip(&chunk.right) {
            interleaved.push(*l);
            interleaved.push(*r);
        }
    }
    write_wav_pcm(writer, sample_rate, 2, &interleaved).await
}

/// Write a 32-bit IEEE-float WAV header for `channels` channels followed by
/// `interleaved`, which must already be in the file's own frame order.
///
/// One writer for both channel counts, so the RIFF header cannot say one thing
/// in the mono path and another in the stereo one.
async fn write_wav_pcm<W>(
    mut writer: W,
    sample_rate: u32,
    channels: u16,
    interleaved: &[f32],
) -> Result<()>
where
    W: AsyncWrite + Unpin,
{
    let bits_per_sample: u16 = 32;
    let byte_rate = sample_rate * u32::from(channels) * u32::from(bits_per_sample / 8);
    let block_align = channels * (bits_per_sample / 8);
    let data_bytes = (interleaved.len() * 4) as u32;
    let riff_size = 36 + data_bytes;

    let mut header = Vec::with_capacity(44);
    header.extend_from_slice(b"RIFF");
    header.extend_from_slice(&riff_size.to_le_bytes());
    header.extend_from_slice(b"WAVE");
    header.extend_from_slice(b"fmt ");
    header.extend_from_slice(&16u32.to_le_bytes()); // fmt chunk size
    header.extend_from_slice(&3u16.to_le_bytes()); // format = IEEE float
    header.extend_from_slice(&channels.to_le_bytes());
    header.extend_from_slice(&sample_rate.to_le_bytes());
    header.extend_from_slice(&byte_rate.to_le_bytes());
    header.extend_from_slice(&block_align.to_le_bytes());
    header.extend_from_slice(&bits_per_sample.to_le_bytes());
    header.extend_from_slice(b"data");
    header.extend_from_slice(&data_bytes.to_le_bytes());
    writer.write_all(&header).await?;

    let mut body = Vec::with_capacity(interleaved.len() * 4);
    for s in interleaved {
        body.extend_from_slice(&s.to_le_bytes());
    }
    writer.write_all(&body).await?;
    writer.flush().await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn u16_at(bytes: &[u8], off: usize) -> u16 {
        u16::from_le_bytes([bytes[off], bytes[off + 1]])
    }

    fn u32_at(bytes: &[u8], off: usize) -> u32 {
        u32::from_le_bytes([bytes[off], bytes[off + 1], bytes[off + 2], bytes[off + 3]])
    }

    fn f32_at(bytes: &[u8], off: usize) -> f32 {
        f32::from_le_bytes([bytes[off], bytes[off + 1], bytes[off + 2], bytes[off + 3]])
    }

    /// The bytes are read back against the RIFF spec's own offsets rather than
    /// through [`crate::decode`], for the reason `decode`'s `float_wav` states
    /// from the other side: a writer and a reader that swap left for right in
    /// the same direction round-trip perfectly and are both wrong. Reading the
    /// header where the spec says the header is settles it independently.
    #[tokio::test]
    async fn stereo_wav_declares_two_channels_and_interleaves_left_first() {
        const SR: u32 = 16_000;
        // Two chunks, so the joint between them is exercised too.
        let chunks = vec![
            Ok(StereoSamples {
                left: vec![0.25, 0.5],
                right: vec![-0.25, -0.5],
            }),
            Ok(StereoSamples {
                left: vec![0.75],
                right: vec![-0.75],
            }),
        ];
        let mut out: Vec<u8> = Vec::new();
        write_wav_stereo(&mut out, SR, futures::stream::iter(chunks))
            .await
            .expect("write stereo wav");

        assert_eq!(&out[0..4], b"RIFF");
        assert_eq!(&out[8..12], b"WAVE");
        assert_eq!(u16_at(&out, 20), 3, "format tag must stay IEEE float");
        assert_eq!(u16_at(&out, 22), 2, "channel count");
        assert_eq!(u32_at(&out, 24), SR);
        assert_eq!(u32_at(&out, 28), SR * 2 * 4, "byte rate");
        assert_eq!(u16_at(&out, 32), 8, "block align: two 4-byte samples");
        assert_eq!(u16_at(&out, 34), 32, "bits per sample");
        assert_eq!(&out[36..40], b"data");
        assert_eq!(u32_at(&out, 40), 3 * 2 * 4, "three frames, two channels");
        assert_eq!(out.len(), 44 + 3 * 2 * 4);

        // Left leads each frame, and the chunk boundary does not reorder it.
        let frames: Vec<(f32, f32)> = (0..3)
            .map(|i| (f32_at(&out, 44 + i * 8), f32_at(&out, 48 + i * 8)))
            .collect();
        assert_eq!(frames, vec![(0.25, -0.25), (0.5, -0.5), (0.75, -0.75)]);
    }

    /// `StereoSamples` has public fields, so nothing stops a caller building a
    /// mismatched pair. The writer refuses it: there is no harmless way to
    /// interleave one, and a truncated or shifted channel is inaudible as a
    /// defect and fatal to a separation.
    #[tokio::test]
    async fn mismatched_channels_are_refused_not_truncated() {
        let chunk = StereoSamples {
            left: vec![0.1, 0.2, 0.3],
            right: vec![-0.1, -0.2],
        };
        let mut out: Vec<u8> = Vec::new();
        let err = write_wav_stereo(&mut out, 16_000, futures::stream::iter([Ok(chunk)]))
            .await
            .expect_err("mismatched channels must not be written");
        assert!(
            matches!(err, AudioError::ChannelLengthMismatch { left: 3, right: 2 }),
            "{err}"
        );
    }

    /// A directory of this test's own, named after the process so parallel
    /// worktrees cannot collide — the same hazard the shared cargo target
    /// directory has.
    fn scratch(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("audio-kit-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("create scratch dir");
        dir
    }

    /// The header tests above read the bytes where the spec says they are, and
    /// that is deliberately *independent* of [`crate::decode`]. This is the
    /// other half: writer and reader pinned **against each other** on the one
    /// path where they meet, which is what a stage like `preprocess normalize`
    /// actually does — write a file, and be read back by the next stage.
    ///
    /// Sample-**exact** rather than approximate, and the exactness is the
    /// point: at a decode rate equal to the file's own there is no resampler
    /// in the way, so anything but equality is a lost frame, a dropped tail or
    /// a format conversion nobody asked for. It is checked rather than assumed
    /// because "decoding at the same rate is a no-op" is a claim about
    /// swresample, not about this crate.
    #[tokio::test]
    async fn a_mono_wav_decodes_back_to_the_samples_it_was_written_from() {
        const SR: u32 = 48_000;
        let dir = scratch("mono-round-trip");
        let path = dir.join("round-trip.wav");

        // Values a float WAV stores exactly, plus two that only survive if
        // nothing quantises to 16-bit on the way through.
        let written: Vec<f32> = (0..SR as usize)
            .map(|i| {
                0.37 * (std::f32::consts::TAU * 440.0 * i as f32 / SR as f32).sin()
                    + 1e-6 * (i % 7) as f32
            })
            .collect();
        // Three chunks, so the joints are inside the file rather than at its ends.
        let chunks: Vec<Result<Samples>> = written
            .chunks(SR as usize / 3 + 1)
            .map(|c| Ok(c.to_vec()))
            .collect();
        write_wav_file(&path, SR, futures::stream::iter(chunks))
            .await
            .expect("write mono wav");

        let read = decode_all(&path, SR).await;
        assert_eq!(read.len(), written.len(), "sample count changed");
        assert_eq!(read, written, "samples changed");
    }

    /// The stereo counterpart, which also pins the one thing the mono path
    /// cannot: that the interleave written here and the de-interleave
    /// [`crate::decode::decode_path_stereo`] performs are inverses. Two
    /// functions that swap left for right in the same direction round-trip
    /// perfectly against *each other* — so this test is only worth anything
    /// beside `stereo_wav_declares_two_channels_and_interleaves_left_first`,
    /// which settles which channel is which against the spec.
    #[tokio::test]
    async fn a_stereo_wav_decodes_back_to_the_channels_it_was_written_from() {
        const SR: u32 = 48_000;
        let dir = scratch("stereo-round-trip");
        let path = dir.join("round-trip.wav");

        let frames = SR as usize / 2;
        // Deliberately dissimilar channels: two identical ones would survive a
        // swap, which is exactly the mistake worth catching.
        let left: Vec<f32> = (0..frames)
            .map(|i| 0.6 * (std::f32::consts::TAU * 220.0 * i as f32 / SR as f32).sin())
            .collect();
        let right: Vec<f32> = (0..frames)
            .map(|i| -0.3 * (std::f32::consts::TAU * 997.0 * i as f32 / SR as f32).sin())
            .collect();
        let chunks: Vec<Result<StereoSamples>> = left
            .chunks(frames / 3 + 1)
            .zip(right.chunks(frames / 3 + 1))
            .map(|(l, r)| {
                Ok(StereoSamples {
                    left: l.to_vec(),
                    right: r.to_vec(),
                })
            })
            .collect();
        write_wav_stereo_file(&path, SR, futures::stream::iter(chunks))
            .await
            .expect("write stereo wav");

        let mut read = StereoSamples::default();
        let stream = crate::decode::decode_path_stereo(&path, crate::DecodeOptions::new(SR));
        let mut stream = std::pin::pin!(stream);
        while let Some(chunk) = stream.next().await {
            let chunk = chunk.expect("decode stereo");
            read.left.extend_from_slice(&chunk.left);
            read.right.extend_from_slice(&chunk.right);
        }
        assert_eq!(read.left.len(), left.len(), "left sample count changed");
        assert_eq!(read.left, left, "left channel changed");
        assert_eq!(read.right, right, "right channel changed");
    }

    /// Decode `path` to mono `f32` at `sr`, draining the whole stream.
    async fn decode_all(path: &std::path::Path, sr: u32) -> Samples {
        let stream = crate::decode::decode_path(path, crate::DecodeOptions::new(sr));
        let mut stream = std::pin::pin!(stream);
        let mut out = Vec::new();
        while let Some(chunk) = stream.next().await {
            out.extend(chunk.expect("decode"));
        }
        out
    }

    /// The mono writer is unchanged by sharing a header writer with the stereo
    /// one — the risk in factoring it out is that `channels` silently becomes 2
    /// for everybody.
    #[tokio::test]
    async fn mono_wav_still_declares_one_channel() {
        const SR: u32 = 48_000;
        let mut out: Vec<u8> = Vec::new();
        write_wav(&mut out, SR, futures::stream::iter([Ok(vec![0.1, 0.2])]))
            .await
            .expect("write mono wav");

        assert_eq!(u16_at(&out, 22), 1, "channel count");
        assert_eq!(u32_at(&out, 28), SR * 4, "byte rate");
        assert_eq!(u16_at(&out, 32), 4, "block align");
        assert_eq!(u32_at(&out, 40), 2 * 4, "two samples");
        assert_eq!(out.len(), 44 + 2 * 4);
    }
}
