//! Minimal WAV encoding for batch `convert` output.
//!
//! We write 32-bit IEEE-float mono WAV (format tag 3) so converted audio is
//! preserved losslessly for A/B listening. mp3 output can be layered on later
//! via ffmpeg; the realtime path never touches this module (it emits raw PCM).

use std::path::Path;

use futures::{Stream, StreamExt};
use tokio::fs::File;
use tokio::io::{AsyncWrite, AsyncWriteExt, BufWriter};

use crate::error::Result;
use crate::Samples;

/// Collect a stream of mono `f32` chunks and write them to `path` as a
/// 32-bit float mono WAV at `sample_rate`.
pub async fn write_wav_file<S>(
    path: impl AsRef<Path>,
    sample_rate: u32,
    stream: S,
) -> Result<()>
where
    S: Stream<Item = Result<Samples>> + Unpin,
{
    let file = File::create(path).await?;
    write_wav(BufWriter::new(file), sample_rate, stream).await
}

/// Write a stream of mono `f32` chunks to `writer` as a 32-bit float mono WAV.
///
/// The whole signal is buffered first so the RIFF/`data` sizes can be filled
/// in; ASMR clips are short enough that this is not a concern.
pub async fn write_wav<W, S>(mut writer: W, sample_rate: u32, mut stream: S) -> Result<()>
where
    W: AsyncWrite + Unpin,
    S: Stream<Item = Result<Samples>> + Unpin,
{
    let mut samples: Vec<f32> = Vec::new();
    while let Some(item) = stream.next().await {
        samples.extend_from_slice(&item?);
    }

    let bits_per_sample: u16 = 32;
    let channels: u16 = 1;
    let byte_rate = sample_rate * u32::from(channels) * u32::from(bits_per_sample / 8);
    let block_align = channels * (bits_per_sample / 8);
    let data_bytes = (samples.len() * 4) as u32;
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

    let mut body = Vec::with_capacity(samples.len() * 4);
    for s in &samples {
        body.extend_from_slice(&s.to_le_bytes());
    }
    writer.write_all(&body).await?;
    writer.flush().await?;
    Ok(())
}
