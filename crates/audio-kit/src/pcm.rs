//! Raw interleaved `f32` little-endian PCM helpers for the Unix-filter path.
//!
//! The voice-conversion filter reads raw PCM from stdin and writes raw PCM to
//! stdout so it
//! composes with ffmpeg pipes. These helpers turn any [`AsyncRead`] into a
//! stream of mono `f32` chunks and drain a stream back into any [`AsyncWrite`].

use futures::{Stream, StreamExt};
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};

use crate::Samples;
use crate::error::Result;

/// Read raw `f32le` mono PCM from `reader`, yielding chunks of at most
/// `chunk_samples` samples. A trailing partial chunk (if the stream ends
/// mid-sample-group) is emitted with whatever whole samples were read.
pub fn read_f32le<R>(mut reader: R, chunk_samples: usize) -> impl Stream<Item = Result<Samples>>
where
    R: AsyncRead + Unpin + Send + 'static,
{
    let chunk_samples = chunk_samples.max(1);
    async_stream::stream! {
        // Byte buffer sized to the requested chunk; refilled each iteration.
        let mut buf = vec![0u8; chunk_samples * 4];
        // Carry for a partial (<4 byte) sample straddling reads.
        let mut carry: Vec<u8> = Vec::with_capacity(4);
        loop {
            match reader.read(&mut buf).await {
                Ok(0) => break,
                Ok(n) => {
                    carry.extend_from_slice(&buf[..n]);
                    let whole = carry.len() / 4;
                    if whole > 0 {
                        let mut out = Vec::with_capacity(whole);
                        for i in 0..whole {
                            let b = [carry[i * 4], carry[i * 4 + 1], carry[i * 4 + 2], carry[i * 4 + 3]];
                            out.push(f32::from_le_bytes(b));
                        }
                        carry.drain(..whole * 4);
                        yield Ok(out);
                    }
                }
                Err(e) => {
                    yield Err(e.into());
                    break;
                }
            }
        }
    }
}

/// Drain a stream of mono `f32` chunks into `writer` as raw `f32le` PCM.
///
/// Errors carried inside the stream are propagated; the writer is flushed on
/// completion.
pub async fn write_f32le<W, S>(mut writer: W, mut stream: S) -> Result<()>
where
    W: AsyncWrite + Unpin,
    S: Stream<Item = Result<Samples>> + Unpin,
{
    let mut bytes: Vec<u8> = Vec::new();
    while let Some(item) = stream.next().await {
        let chunk = item?;
        bytes.clear();
        bytes.reserve(chunk.len() * 4);
        for s in chunk {
            bytes.extend_from_slice(&s.to_le_bytes());
        }
        writer.write_all(&bytes).await?;
    }
    writer.flush().await?;
    Ok(())
}
