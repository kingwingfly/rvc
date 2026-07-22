//! Decode arbitrary audio containers (mp3, wav, ...) into a stream of mono
//! `f32` PCM chunks resampled to a target sample rate.
//!
//! ffmpeg is fundamentally a blocking, pull-based API, so the actual decode
//! runs on a [`tokio::task::spawn_blocking`] worker that pushes decoded chunks
//! through a bounded channel. The public surface is a [`futures::Stream`], so
//! callers wire it into the async pipeline like any other stream.

use std::path::{Path, PathBuf};
use std::sync::Once;

use ffmpeg_next as ffmpeg;
use ffmpeg::format::sample::{Sample, Type as SampleType};
use ffmpeg::util::channel_layout::ChannelLayout;
use ffmpeg::util::frame::audio::Audio as AudioFrame;
use futures::Stream;
use tokio::sync::mpsc;

use crate::error::{AudioError, Result};
use crate::Samples;

static FFMPEG_INIT: Once = Once::new();

/// Ensure ffmpeg's global state is initialised exactly once per process.
fn ensure_ffmpeg() {
    FFMPEG_INIT.call_once(|| {
        // Only fails if ffmpeg itself is broken; nothing sane to do but proceed.
        let _ = ffmpeg::init();
    });
}

/// Options controlling decode output. Output is always mono `f32` PCM.
#[derive(Debug, Clone, Copy)]
pub struct DecodeOptions {
    /// Target sample rate in Hz (e.g. 16_000 for content encoders, 48_000 for
    /// the synthesizer).
    pub sample_rate: u32,
    /// Channel capacity (in chunks) for back-pressure between the blocking
    /// decoder and the async consumer.
    pub channel_capacity: usize,
}

impl DecodeOptions {
    /// Options for a given sample rate with a sensible channel capacity.
    pub fn new(sample_rate: u32) -> Self {
        Self { sample_rate, channel_capacity: 32 }
    }
}

/// Decode a single file into a stream of mono `f32` chunks at `opts.sample_rate`.
pub fn decode_path(
    path: impl AsRef<Path>,
    opts: DecodeOptions,
) -> impl Stream<Item = Result<Samples>> {
    decode_paths(vec![path.as_ref().to_path_buf()], opts)
}

/// Decode a vector of files **sequentially**, concatenating their audio into a
/// single stream of mono `f32` chunks at `opts.sample_rate`.
pub fn decode_paths(
    paths: Vec<PathBuf>,
    opts: DecodeOptions,
) -> impl Stream<Item = Result<Samples>> {
    let (tx, mut rx) = mpsc::channel::<Result<Samples>>(opts.channel_capacity.max(1));

    tokio::task::spawn_blocking(move || {
        ensure_ffmpeg();
        for path in paths {
            if let Err(e) = decode_one_blocking(&path, opts.sample_rate, &tx) {
                // Best-effort: report the error, then stop this file.
                let _ = tx.blocking_send(Err(e));
                // Continue to the next path so one bad file does not kill a batch.
            }
        }
    });

    async_stream::stream! {
        while let Some(item) = rx.recv().await {
            yield item;
        }
    }
}

/// Blocking decode of one file. Sends decoded mono `f32` chunks over `tx`.
fn decode_one_blocking(
    path: &Path,
    target_rate: u32,
    tx: &mpsc::Sender<Result<Samples>>,
) -> Result<()> {
    let mut ictx = ffmpeg::format::input(&path)?;

    let stream = ictx
        .streams()
        .best(ffmpeg::media::Type::Audio)
        .ok_or(AudioError::NoAudioStream)?;
    let stream_index = stream.index();

    let mut decoder = ffmpeg::codec::context::Context::from_parameters(stream.parameters())?
        .decoder()
        .audio()?;
    decoder.set_parameters(stream.parameters())?;

    // Resample whatever the source is into packed mono f32 at the target rate.
    let mut resampler = ffmpeg::software::resampling::context::Context::get(
        decoder.format(),
        decoder.channel_layout(),
        decoder.rate(),
        Sample::F32(SampleType::Packed),
        ChannelLayout::MONO,
        target_rate,
    )?;

    let send = |frame: &AudioFrame| -> Result<()> {
        let n = frame.samples();
        if n == 0 {
            return Ok(());
        }
        // Packed mono => plane 0 holds exactly `samples()` f32 values.
        let data: &[f32] = frame.plane(0);
        let chunk: Samples = data[..n].to_vec();
        // If the receiver is gone the consumer dropped the stream; stop quietly.
        tx.blocking_send(Ok(chunk)).map_err(|_| AudioError::WorkerGone)
    };

    let mut decoded = AudioFrame::empty();
    for (s, packet) in ictx.packets() {
        if s.index() != stream_index {
            continue;
        }
        decoder.send_packet(&packet)?;
        while decoder.receive_frame(&mut decoded).is_ok() {
            let mut resampled = AudioFrame::empty();
            resampler.run(&decoded, &mut resampled)?;
            send(&resampled)?;
        }
    }

    // Flush the decoder.
    decoder.send_eof()?;
    while decoder.receive_frame(&mut decoded).is_ok() {
        let mut resampled = AudioFrame::empty();
        resampler.run(&decoded, &mut resampled)?;
        send(&resampled)?;
    }

    // Flush any samples still buffered inside the resampler.
    loop {
        let mut resampled = AudioFrame::empty();
        match resampler.flush(&mut resampled)? {
            Some(_) => send(&resampled)?,
            None => {
                // A final partial buffer may still be present.
                send(&resampled)?;
                break;
            }
        }
    }

    Ok(())
}
