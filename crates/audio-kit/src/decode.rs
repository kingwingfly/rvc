//! Decode arbitrary audio containers (mp3, wav, ...) into a stream of mono
//! `f32` PCM chunks resampled to a target sample rate.
//!
//! ffmpeg is fundamentally a blocking, pull-based API, so the actual decode
//! runs on a [`tokio::task::spawn_blocking`] worker that pushes decoded chunks
//! through a bounded channel. The public surface is a [`futures::Stream`], so
//! callers wire it into the async pipeline like any other stream.
//!
//! [`decode_path_stereo`] is the same machinery with the downmix left out, for
//! the one consumer that needs two channels — see [`StereoSamples`] for why
//! that is a separate function rather than an option, and the crate docs for
//! why the stereo path stops at this module and [`crate::encode`].

use std::path::{Path, PathBuf};
use std::sync::Once;

use ffmpeg::util::frame::audio::Audio as AudioFrame;
use ffmpeg_next as ffmpeg;
use futures::Stream;
use tokio::sync::mpsc;

use crate::error::{AudioError, Result};
use crate::{Samples, StereoSamples};

static FFMPEG_INIT: Once = Once::new();

/// Samples per emitted chunk on the mono path, and **frames** per chunk on the
/// stereo one — so a chunk covers the same wall-clock either way and the
/// channel's back-pressure means the same thing on both.
const CHUNK: usize = 16_384;

/// Ensure ffmpeg's global state is initialised exactly once per process.
pub(crate) fn ensure_ffmpeg() {
    FFMPEG_INIT.call_once(|| {
        // Only fails if ffmpeg itself is broken; nothing sane to do but proceed.
        let _ = ffmpeg::init();
    });
}

/// Options controlling decode output: the rate and the back-pressure, and
/// deliberately **not** the channel count — the function chosen decides that,
/// so `decode_paths` always yields mono `f32` and `decode_paths_stereo` always
/// yields two channels.
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
        Self {
            sample_rate,
            channel_capacity: 32,
        }
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

/// Decode a single file into a stream of two-channel `f32` chunks at
/// `opts.sample_rate`.
pub fn decode_path_stereo(
    path: impl AsRef<Path>,
    opts: DecodeOptions,
) -> impl Stream<Item = Result<StereoSamples>> {
    decode_paths_stereo(vec![path.as_ref().to_path_buf()], opts)
}

/// Decode a vector of files **sequentially**, concatenating their audio into a
/// single stream of two-channel `f32` chunks at `opts.sample_rate`.
///
/// This is a second function rather than a `channels` field on
/// [`DecodeOptions`] on purpose. The channel count then lives in the *return
/// type*, so it cannot disagree with what the caller receives: an option set to
/// `2` beside a signature still promising [`Samples`] would be a lie no
/// compiler could catch, and an option nobody sets is one more thing every
/// existing mono call site has to be read past. Mono stays the default and
/// [`decode_paths`] stays the path every engine takes.
pub fn decode_paths_stereo(
    paths: Vec<PathBuf>,
    opts: DecodeOptions,
) -> impl Stream<Item = Result<StereoSamples>> {
    let (tx, mut rx) = mpsc::channel::<Result<StereoSamples>>(opts.channel_capacity.max(1));

    tokio::task::spawn_blocking(move || {
        ensure_ffmpeg();
        for path in paths {
            if let Err(e) = decode_one_stereo_blocking(&path, opts.sample_rate, &tx) {
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
    for chunk in mono.chunks(CHUNK) {
        tx.blocking_send(Ok(chunk.to_vec()))
            .map_err(|_| AudioError::WorkerGone)?;
    }
    Ok(())
}

/// Blocking decode of one file. Sends decoded two-channel `f32` chunks over
/// `tx`. Mirrors [`decode_one_blocking`] exactly but for the fold, which keeps
/// the two channels apart instead of averaging them away.
fn decode_one_stereo_blocking(
    path: &Path,
    target_rate: u32,
    tx: &mpsc::Sender<Result<StereoSamples>>,
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

    // Decode every frame to a left/right pair at the source rate, then resample
    // each channel once. Same reason the mono path avoids libswresample's frame
    // API: its conversion is unreliable in this ffmpeg build.
    let mut src = StereoSamples::default();
    let mut src_rate: u32 = 0;

    let take = |frame: &AudioFrame, src: &mut StereoSamples, src_rate: &mut u32| -> Result<()> {
        if frame.samples() == 0 {
            return Ok(());
        }
        if *src_rate == 0 {
            *src_rate = frame.rate();
        }
        // Both channels are extended *after* the fallible step, never one
        // before it: a half-extended pair breaks `StereoSamples`'s equal-length
        // invariant, and nothing downstream re-checks it.
        let (left, right) = frame_to_stereo_f32(frame)?;
        src.left.extend(left);
        src.right.extend(right);
        Ok(())
    };

    let mut decoded = AudioFrame::empty();
    for (s, packet) in ictx.packets() {
        if s.index() != stream_index {
            continue;
        }
        decoder.send_packet(&packet)?;
        while decoder.receive_frame(&mut decoded).is_ok() {
            take(&decoded, &mut src, &mut src_rate)?;
        }
    }
    decoder.send_eof()?;
    while decoder.receive_frame(&mut decoded).is_ok() {
        take(&decoded, &mut src, &mut src_rate)?;
    }

    if src.is_empty() {
        return Ok(());
    }
    // `resample`'s output length is a function of the input length alone, so
    // two channels that went in equal come out equal.
    let left = resample(&src.left, src_rate, target_rate);
    let right = resample(&src.right, src_rate, target_rate);

    for (l, r) in left.chunks(CHUNK).zip(right.chunks(CHUNK)) {
        tx.blocking_send(Ok(StereoSamples {
            left: l.to_vec(),
            right: r.to_vec(),
        }))
        .map_err(|_| AudioError::WorkerGone)?;
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
        S::I32(_) => (4, |b| {
            i32::from_ne_bytes([b[0], b[1], b[2], b[3]]) as f32 / 2_147_483_648.0
        }),
        S::F32(_) => (4, |b| f32::from_ne_bytes([b[0], b[1], b[2], b[3]])),
        S::F64(_) => (8, |b| {
            f64::from_ne_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]) as f32
        }),
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

/// Fold one decoded frame to a left/right pair, handling the same sample
/// formats [`frame_to_mono_f32`] does in both planar and packed layouts.
///
/// The decoder hands over whatever the file holds, so three cases need a
/// defined answer:
///
/// - **One channel** is duplicated into both. A separator handed that finds no
///   stereo cue — which is the information not existing in the recording rather
///   than a defect here, and is why asking for stereo cannot manufacture it.
/// - **Two channels** pass through untouched. This is the case that matters and
///   the one this function is exact for.
/// - **More than two** fold to the front pair: channel 0 leads the left,
///   channel 1 the right, and every remaining channel is added to **both** at
///   equal weight before dividing by `n - 1`, so a signal identical in every
///   channel comes back at unity — the same property [`frame_to_mono_f32`]'s
///   `1/n` has, which is what keeps the two paths comparable. Keeping only the
///   front pair would be simpler and is wrong for exactly the material this
///   exists for: 5.1's centre channel is where a centred vocal lives, so
///   discarding it removes what a separator is hunting for.
///
/// The fold is decided **per frame**, which is what makes a file that changes
/// channel count mid-stream harmless: every input frame still yields exactly
/// one left and one right sample, so the two channels cannot drift apart.
fn frame_to_stereo_f32(frame: &AudioFrame) -> Result<(Vec<f32>, Vec<f32>)> {
    use ffmpeg::format::Sample as S;

    let n = frame.samples();
    let ch = frame.channels().max(1) as usize;
    let fmt = frame.format();

    let (width, conv): (usize, fn(&[u8]) -> f32) = match fmt {
        S::U8(_) => (1, |b| (b[0] as f32 - 128.0) / 128.0),
        S::I16(_) => (2, |b| i16::from_ne_bytes([b[0], b[1]]) as f32 / 32768.0),
        S::I32(_) => (4, |b| {
            i32::from_ne_bytes([b[0], b[1], b[2], b[3]]) as f32 / 2_147_483_648.0
        }),
        S::F32(_) => (4, |b| f32::from_ne_bytes([b[0], b[1], b[2], b[3]])),
        S::F64(_) => (8, |b| {
            f64::from_ne_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]) as f32
        }),
        other => return Err(AudioError::UnsupportedFormat(other)),
    };

    // Planar audio only fills `linesize[0]`, so each plane's slice length is
    // unreliable; read `n*width` bytes from each plane pointer directly. Packed
    // audio is one plane holding every channel.
    let planar = fmt.is_planar();
    let planes: Vec<&[u8]> = if planar {
        (0..ch)
            .map(|c| unsafe { std::slice::from_raw_parts(frame.data(c).as_ptr(), n * width) })
            .collect()
    } else {
        vec![unsafe { std::slice::from_raw_parts(frame.data(0).as_ptr(), n * ch * width) }]
    };
    let at = |i: usize, c: usize| -> f32 {
        let (plane, off) = if planar {
            (c, i * width)
        } else {
            (0, (i * ch + c) * width)
        };
        conv(&planes[plane][off..off + width])
    };

    let mut left = Vec::with_capacity(n);
    let mut right = Vec::with_capacity(n);
    for i in 0..n {
        let (l, r) = match ch {
            1 => {
                let x = at(i, 0);
                (x, x)
            }
            2 => (at(i, 0), at(i, 1)),
            _ => {
                let rest: f32 = (2..ch).map(|c| at(i, c)).sum();
                let inv = 1.0 / (ch - 1) as f32;
                ((at(i, 0) + rest) * inv, (at(i, 1) + rest) * inv)
            }
        };
        left.push(l);
        right.push(r);
    }
    Ok((left, right))
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

#[cfg(test)]
mod tests {
    use super::*;
    use futures::StreamExt;

    /// A 32-bit float WAV built by hand from the RIFF spec, **not** through
    /// [`crate::encode`]. The independence is the whole point: a round trip
    /// through our own writer passes even when the writer and the decoder swap
    /// left for right in the same direction, which is precisely the mistake
    /// this path can make. A file written to the spec is an outside authority
    /// on which sample is the left one.
    fn float_wav(sample_rate: u32, channels: u16, interleaved: &[f32]) -> Vec<u8> {
        let data_bytes = (interleaved.len() * 4) as u32;
        let mut out = Vec::with_capacity(44 + interleaved.len() * 4);
        out.extend_from_slice(b"RIFF");
        out.extend_from_slice(&(36 + data_bytes).to_le_bytes());
        out.extend_from_slice(b"WAVE");
        out.extend_from_slice(b"fmt ");
        out.extend_from_slice(&16u32.to_le_bytes());
        out.extend_from_slice(&3u16.to_le_bytes()); // IEEE float
        out.extend_from_slice(&channels.to_le_bytes());
        out.extend_from_slice(&sample_rate.to_le_bytes());
        out.extend_from_slice(&(sample_rate * u32::from(channels) * 4).to_le_bytes());
        out.extend_from_slice(&(channels * 4).to_le_bytes());
        out.extend_from_slice(&32u16.to_le_bytes());
        out.extend_from_slice(b"data");
        out.extend_from_slice(&data_bytes.to_le_bytes());
        for s in interleaved {
            out.extend_from_slice(&s.to_le_bytes());
        }
        out
    }

    /// ffmpeg opens a container by path, so a decode test needs a real file.
    /// `name` and the pid keep concurrent tests — and concurrent checkouts —
    /// from writing over each other.
    fn temp_wav(name: &str, bytes: &[u8]) -> PathBuf {
        let path =
            std::env::temp_dir().join(format!("audio-kit-{}-{name}.wav", std::process::id()));
        std::fs::write(&path, bytes).expect("write temp wav");
        path
    }

    fn tone(freq: f32, sample_rate: u32, frames: usize) -> Vec<f32> {
        (0..frames)
            .map(|i| 0.5 * (std::f32::consts::TAU * freq * i as f32 / sample_rate as f32).sin())
            .collect()
    }

    fn interleave(channels: &[Vec<f32>]) -> Vec<f32> {
        let frames = channels[0].len();
        let mut out = Vec::with_capacity(frames * channels.len());
        for i in 0..frames {
            for c in channels {
                out.push(c[i]);
            }
        }
        out
    }

    async fn collect_stereo(path: &Path, sr: u32) -> StereoSamples {
        let mut stream = Box::pin(decode_path_stereo(path, DecodeOptions::new(sr)));
        let mut out = StereoSamples::default();
        while let Some(item) = stream.next().await {
            let chunk = item.expect("decode chunk");
            out.left.extend(chunk.left);
            out.right.extend(chunk.right);
        }
        out
    }

    async fn collect_mono(path: &Path, sr: u32) -> Samples {
        let mut stream = Box::pin(decode_path(path, DecodeOptions::new(sr)));
        let mut out = Samples::new();
        while let Some(item) = stream.next().await {
            out.extend(item.expect("decode chunk"));
        }
        out
    }

    /// The decisive test: the two channels carry *different* tones, so an order
    /// swap and an average are both failures. Give both channels the same
    /// signal and every one of these assertions passes while left and right are
    /// reversed. Decoded at the file's own rate, so no resampling stands
    /// between what was written and what comes back and the comparison can be
    /// exact.
    #[tokio::test]
    async fn stereo_decode_keeps_the_channels_apart_and_in_order() {
        const SR: u32 = 8_000;
        // Past `CHUNK`, so the channels have to stay aligned across a chunk
        // boundary as well as within one.
        const FRAMES: usize = 20_000;
        let l = tone(220.0, SR, FRAMES);
        let r = tone(440.0, SR, FRAMES);
        let path = temp_wav(
            "stereo-order",
            &float_wav(SR, 2, &interleave(&[l.clone(), r.clone()])),
        );

        let got = collect_stereo(&path, SR).await;
        let _ = std::fs::remove_file(&path);

        assert_eq!(got.frames(), FRAMES);
        for i in 0..FRAMES {
            assert!(
                (got.left[i] - l[i]).abs() < 1e-6,
                "left {i}: {} != {}",
                got.left[i],
                l[i]
            );
            assert!(
                (got.right[i] - r[i]).abs() < 1e-6,
                "right {i}: {} != {}",
                got.right[i],
                r[i]
            );
        }
        // ...and the two tones really are distinguishable, so "unmixed" means
        // something rather than being true of any pair.
        let apart = got
            .left
            .iter()
            .zip(&got.right)
            .map(|(a, b)| (a - b).abs())
            .sum::<f32>()
            / FRAMES as f32;
        assert!(apart > 0.1, "channels too alike to prove anything: {apart}");
    }

    /// A mono source asked for stereo yields the same signal twice — the cue a
    /// separator wants simply is not in the recording, and asking for two
    /// channels cannot manufacture it.
    #[tokio::test]
    async fn a_mono_file_asked_for_stereo_is_duplicated() {
        const SR: u32 = 8_000;
        let m = tone(330.0, SR, 4_000);
        let path = temp_wav("mono-to-stereo", &float_wav(SR, 1, &m));

        let got = collect_stereo(&path, SR).await;
        let _ = std::fs::remove_file(&path);

        assert_eq!(got.frames(), m.len());
        assert_eq!(got.left, got.right);
        for (i, want) in m.iter().enumerate() {
            assert!(
                (got.left[i] - want).abs() < 1e-6,
                "{i}: {} != {want}",
                got.left[i]
            );
        }
    }

    /// More than two channels fold to the front pair with the rest shared
    /// equally, rather than the extra channels being dropped. Constant levels
    /// make the arithmetic exact: `rest` is 0.4, so left is `(0.5 + 0.4) / 5`
    /// and right `(0.25 + 0.4) / 5`, and a centre channel that had been
    /// discarded would leave 0.5 and 0.25 untouched.
    #[tokio::test]
    async fn six_channels_fold_the_rest_into_both_sides() {
        const SR: u32 = 8_000;
        const FRAMES: usize = 2_000;
        let level = |v: f32| vec![v; FRAMES];
        let channels = vec![
            level(0.5),
            level(0.25),
            level(0.1),
            level(0.1),
            level(0.1),
            level(0.1),
        ];
        let path = temp_wav("six-channel", &float_wav(SR, 6, &interleave(&channels)));

        let got = collect_stereo(&path, SR).await;
        let _ = std::fs::remove_file(&path);

        assert_eq!(got.frames(), FRAMES);
        assert!(
            (got.left[FRAMES / 2] - 0.18).abs() < 1e-5,
            "{}",
            got.left[FRAMES / 2]
        );
        assert!(
            (got.right[FRAMES / 2] - 0.13).abs() < 1e-5,
            "{}",
            got.right[FRAMES / 2]
        );
    }

    /// A uniform signal survives the fold at unity, which is the property that
    /// keeps the stereo fold comparable with the mono downmix's `1/n` rather
    /// than quietly rescaling everything that is not two-channel.
    #[tokio::test]
    async fn a_uniform_multichannel_signal_folds_to_unity() {
        const SR: u32 = 8_000;
        const FRAMES: usize = 2_000;
        let channels = vec![vec![0.3f32; FRAMES]; 6];
        let path = temp_wav("uniform-six", &float_wav(SR, 6, &interleave(&channels)));

        let got = collect_stereo(&path, SR).await;
        let _ = std::fs::remove_file(&path);

        assert!(
            (got.left[FRAMES / 2] - 0.3).abs() < 1e-5,
            "{}",
            got.left[FRAMES / 2]
        );
        assert!(
            (got.right[FRAMES / 2] - 0.3).abs() < 1e-5,
            "{}",
            got.right[FRAMES / 2]
        );
    }

    /// The mono path is untouched by any of this: the same stereo file through
    /// [`decode_path`] still averages the channels, which is what every engine
    /// in the toolkit depends on.
    #[tokio::test]
    async fn the_mono_path_still_downmixes() {
        const SR: u32 = 8_000;
        const FRAMES: usize = 4_000;
        let l = tone(220.0, SR, FRAMES);
        let r = tone(440.0, SR, FRAMES);
        let path = temp_wav(
            "mono-downmix",
            &float_wav(SR, 2, &interleave(&[l.clone(), r.clone()])),
        );

        let got = collect_mono(&path, SR).await;
        let _ = std::fs::remove_file(&path);

        assert_eq!(got.len(), FRAMES);
        for i in 0..FRAMES {
            let want = (l[i] + r[i]) / 2.0;
            assert!((got[i] - want).abs() < 1e-6, "{i}: {} != {want}", got[i]);
        }
    }
}
