//! Raw interleaved `f32` little-endian PCM helpers for the Unix-filter path.
//!
//! An engine that filters raw PCM reads it from stdin and writes it to stdout so
//! it composes with ffmpeg pipes. These helpers turn any [`AsyncRead`] into a
//! stream of mono `f32` chunks and drain a stream — or a single chunk — back into
//! any [`AsyncWrite`], and [`resample_linear`] adapts the rate when the two ends
//! of such a pipe disagree.

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
    while let Some(item) = stream.next().await {
        write_f32le_chunk(&mut writer, &item?).await?;
    }
    writer.flush().await?;
    Ok(())
}

/// Write one chunk of mono `f32` samples to `writer` as raw `f32le` PCM.
///
/// The writer is **not** flushed, so a realtime caller can flush per chunk for
/// latency and a batch one can leave it to its `BufWriter`.
pub async fn write_f32le_chunk<W>(mut writer: W, samples: &[f32]) -> Result<()>
where
    W: AsyncWrite + Unpin,
{
    let mut bytes = Vec::with_capacity(samples.len() * 4);
    for s in samples {
        bytes.extend_from_slice(&s.to_le_bytes());
    }
    writer.write_all(&bytes).await?;
    Ok(())
}

/// Resample mono PCM from `from_sr` to `to_sr` by linear interpolation.
///
/// Adequate wherever the signal is already band-limited well below either
/// rate's Nyquist — a synthesizer's output is, so this costs nothing audible
/// and saves a dependency on a filter design. It is *not* a general-purpose
/// resampler: feeding it full-band audio being downsampled will alias, and
/// [`decode`](crate::decode) (which resamples through ffmpeg) is the right tool
/// for that.
///
/// Empty input yields empty output. A zero rate is meaningless — the ratio
/// would be non-finite and the output length arbitrary — so it yields empty
/// output too rather than panicking.
pub fn resample_linear(pcm: &[f32], from_sr: u32, to_sr: u32) -> Samples {
    if from_sr == to_sr {
        return pcm.to_vec();
    }
    if pcm.is_empty() || from_sr == 0 || to_sr == 0 {
        return Vec::new();
    }
    let ratio = from_sr as f64 / to_sr as f64;
    let n = (pcm.len() as f64 / ratio) as usize;
    (0..n)
        .map(|i| {
            let x = i as f64 * ratio;
            let (a, f) = (x as usize, (x - x.floor()) as f32);
            // The last output sample can land on the final input sample, whose
            // right-hand neighbour does not exist; hold it instead of reading past.
            let b = (a + 1).min(pcm.len() - 1);
            pcm[a] * (1.0 - f) + pcm[b] * f
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::resample_linear;

    /// A ramp: every sample is its own index, so an interpolated output sample
    /// is exactly its position in input coordinates.
    fn ramp(n: usize) -> Vec<f32> {
        (0..n).map(|i| i as f32).collect()
    }

    #[test]
    fn matching_rates_are_a_copy() {
        let pcm = ramp(64);
        assert_eq!(resample_linear(&pcm, 32000, 32000), pcm);
    }

    #[test]
    fn downsampling_halves_the_length() {
        let out = resample_linear(&ramp(100), 32000, 16000);
        assert_eq!(out.len(), 50);
        // Output i reads input 2i.
        for (i, &s) in out.iter().enumerate() {
            assert!((s - (2 * i) as f32).abs() < 1e-3, "{i}: {s}");
        }
    }

    #[test]
    fn upsampling_triples_the_length_and_holds_the_last_sample() {
        let out = resample_linear(&ramp(10), 16000, 48000);
        assert_eq!(out.len(), 30);
        // Output i reads input i/3, up to the last sample that has a right-hand
        // neighbour to interpolate towards.
        for (i, &s) in out.iter().take(28).enumerate() {
            assert!((s - i as f32 / 3.0).abs() < 1e-3, "{i}: {s}");
        }
        // Past that the final input sample is held rather than read past.
        assert_eq!(&out[28..], &[9.0, 9.0]);
    }

    #[test]
    fn a_ratio_that_does_not_divide_interpolates_between_samples() {
        let out = resample_linear(&ramp(100), 48000, 32000);
        assert_eq!(out.len(), 66);
        assert_eq!(out[0], 0.0);
        assert!((out[1] - 1.5).abs() < 1e-3, "{}", out[1]);
        assert!(out.iter().all(|s| s.is_finite()));
        // The tail reaches the input's tail rather than stopping short of it.
        let last = *out.last().unwrap();
        assert!(last > 97.0 && last < 99.0, "{last}");
    }

    #[test]
    fn degenerate_inputs_are_empty_not_a_panic() {
        assert!(resample_linear(&[], 32000, 16000).is_empty());
        assert!(resample_linear(&ramp(8), 0, 16000).is_empty());
        assert!(resample_linear(&ramp(8), 32000, 0).is_empty());
    }
}
