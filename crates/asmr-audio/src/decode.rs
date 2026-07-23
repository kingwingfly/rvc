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

    // Decode every frame to mono f32 at the source rate, then resample once.
    // We avoid libswresample's frame API entirely: its stereo→mono / rate
    // conversion is unreliable in this ffmpeg build (spurious "Output changed").
    let mut mono_src: Vec<f32> = Vec::new();
    let mut src_rate: u32 = 0;

    let take = |frame: &AudioFrame, mono_src: &mut Vec<f32>, src_rate: &mut u32| -> Result<()> {
        if frame.samples() == 0 {
            return Ok(());
        }
        if *src_rate == 0 {
            *src_rate = frame.rate();
        }
        mono_src.extend(frame_to_mono_f32(frame)?);
        Ok(())
    };

    let mut decoded = AudioFrame::empty();
    for (s, packet) in ictx.packets() {
        if s.index() != stream_index {
            continue;
        }
        decoder.send_packet(&packet)?;
        while decoder.receive_frame(&mut decoded).is_ok() {
            take(&decoded, &mut mono_src, &mut src_rate)?;
        }
    }
    decoder.send_eof()?;
    while decoder.receive_frame(&mut decoded).is_ok() {
        take(&decoded, &mut mono_src, &mut src_rate)?;
    }

    if mono_src.is_empty() {
        return Ok(());
    }
    let mono = resample(&mono_src, src_rate, target_rate);

    // Emit in chunks so downstream back-pressure still works.
    const CHUNK: usize = 16_384;
    for chunk in mono.chunks(CHUNK) {
        tx.blocking_send(Ok(chunk.to_vec())).map_err(|_| AudioError::WorkerGone)?;
    }
    Ok(())
}

/// Downmix one decoded frame to mono `f32`, handling the common sample formats
/// in both planar and packed layouts.
fn frame_to_mono_f32(frame: &AudioFrame) -> Result<Vec<f32>> {
    use ffmpeg::format::Sample as S;

    let n = frame.samples();
    let ch = frame.channels().max(1) as usize;
    let fmt = frame.format();

    let (width, conv): (usize, fn(&[u8]) -> f32) = match fmt {
        S::U8(_) => (1, |b| (b[0] as f32 - 128.0) / 128.0),
        S::I16(_) => (2, |b| i16::from_ne_bytes([b[0], b[1]]) as f32 / 32768.0),
        S::I32(_) => (4, |b| i32::from_ne_bytes([b[0], b[1], b[2], b[3]]) as f32 / 2_147_483_648.0),
        S::F32(_) => (4, |b| f32::from_ne_bytes([b[0], b[1], b[2], b[3]])),
        S::F64(_) => {
            (8, |b| f64::from_ne_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]) as f32)
        }
        other => return Err(AudioError::UnsupportedFormat(other)),
    };

    let inv = 1.0 / ch as f32;
    let mut out = Vec::with_capacity(n);
    if fmt.is_planar() {
        // Planar audio only fills `linesize[0]`, so each plane's slice length is
        // unreliable; read `n*width` bytes from each plane pointer directly.
        let planes: Vec<&[u8]> = (0..ch)
            .map(|c| unsafe { std::slice::from_raw_parts(frame.data(c).as_ptr(), n * width) })
            .collect();
        for i in 0..n {
            let off = i * width;
            let sum: f32 = planes.iter().map(|p| conv(&p[off..off + width])).sum();
            out.push(sum * inv);
        }
    } else {
        let d: &[u8] =
            unsafe { std::slice::from_raw_parts(frame.data(0).as_ptr(), n * ch * width) };
        for i in 0..n {
            let sum: f32 = (0..ch)
                .map(|c| {
                    let off = (i * ch + c) * width;
                    conv(&d[off..off + width])
                })
                .sum();
            out.push(sum * inv);
        }
    }
    Ok(out)
}

/// Resample a mono signal from `src` to `dst` Hz. Downsampling uses area
/// averaging (a cheap anti-alias); upsampling uses linear interpolation.
fn resample(input: &[f32], src: u32, dst: u32) -> Vec<f32> {
    if src == dst || src == 0 || input.len() < 2 {
        return input.to_vec();
    }
    let ratio = dst as f64 / src as f64;
    let out_len = ((input.len() as f64) * ratio).round().max(1.0) as usize;
    let step = 1.0 / ratio; // input samples per output sample
    let mut out = Vec::with_capacity(out_len);

    if dst < src {
        // Area average over each output sample's input window.
        for j in 0..out_len {
            let start = j as f64 * step;
            let end = start + step;
            let a = (start.floor() as usize).min(input.len() - 1);
            let b = (end.ceil() as usize).clamp(a + 1, input.len());
            let win = &input[a..b];
            out.push(win.iter().sum::<f32>() / win.len() as f32);
        }
    } else {
        for j in 0..out_len {
            let pos = j as f64 * step;
            let i0 = pos.floor() as usize;
            let i1 = (i0 + 1).min(input.len() - 1);
            let frac = (pos - i0 as f64) as f32;
            out.push(input[i0] * (1.0 - frac) + input[i1] * frac);
        }
    }
    out
}
