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

use crate::error::Result;
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
pub async fn write_wav_stereo<W, S>(writer: W, sample_rate: u32, mut stream: S) -> Result<()>
where
    W: AsyncWrite + Unpin,
    S: Stream<Item = Result<StereoSamples>> + Unpin,
{
    let mut interleaved: Vec<f32> = Vec::new();
    while let Some(item) = stream.next().await {
        let chunk = item?;
        // `zip` stops at the shorter channel. `StereoSamples` promises the two
        // are equal, and a caller who breaks that promise loses the odd sample
        // rather than shifting every frame after it by one — which is what
        // writing the longer channel against silence would do.
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
