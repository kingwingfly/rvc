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
        // Warnings and above, so a real problem is still reported and ffmpeg's
        // own commentary is not.
        //
        // **The `av_log` level is the only control there is over a filter that
        // reports through the log**, which is not a tidiness preference: a
        // measuring filter such as `ebur128` prints a multi-line summary block
        // when its graph is dropped, and it does so *at info level, past its own
        // `framelog=quiet`* — that option governs the per-frame lines and not the
        // summary. Two of those blocks landed on stderr for every file
        // `normalize --lufs` touched, saying the same thing the stage's own
        // one-line report says. Nothing in the filter string can stop them.
        //
        // This is process-global, which is exactly why it belongs here rather
        // than beside any one caller: the level is not a property of one graph,
        // and setting it per call site would mean every future one remembering.
        ffmpeg::util::log::set_level(ffmpeg::util::log::Level::Warning);
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
///   channel comes back at unity, exactly as [`frame_to_mono_f32`]'s `1/n`
///   does. Keeping only the front pair would be simpler and is wrong for
///   exactly the material this exists for: 5.1's centre channel is where a
///   centred vocal lives, so discarding it removes what a separator is hunting
///   for.
///
/// **A uniform signal is the only input on which the two paths agree, so do not
/// read `(L + R) / 2` as the mono downmix.** Beyond two channels the fold
/// weights the front pair against the rest and the downmix does not: 5.1
/// carrying `v` in channel 0 alone gives `v/6` through [`frame_to_mono_f32`]
/// and `v/10` through the averaged pair. Whoever wants the downmix wants
/// [`decode_paths`], which is the default anyway.
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

/// Half-width of the interpolation kernel, counted in zero crossings of its
/// sinc rather than in samples. Expressed that way it is a statement about the
/// *filter* and stays one at any ratio: the width in input samples is derived
/// from it below, so a large decimation gets a proportionally longer kernel
/// instead of silently losing its cutoff. A count fixed in samples is the trap
/// here — it looks rate-independent and is not.
const KERNEL_ZEROS: usize = 24;

/// Cutoff as a fraction of the **lower** rate's Nyquist.
///
/// It is under 1.0 because a filter has a transition band, and the whole point
/// of this one is that the transition has closed before the output's Nyquist —
/// anything still passing there folds back into the audible band and cannot be
/// removed afterwards. 0.95 with [`KERNEL_ZEROS`] taps puts the stopband inside
/// Nyquist while leaving the response flat to 7 kHz at a 16 kHz output; the
/// price is measured by `the_top_of_the_passband_is_flat_and_the_edge_rolls_off`
/// and is 4 dB at 7.5 kHz.
const KERNEL_ROLLOFF: f64 = 0.95;

/// Kaiser shape parameter, ≈90 dB of sidelobe rejection. What it buys against
/// a plain truncated sinc is stopband depth, which is exactly the quantity
/// `a_tone_above_the_output_nyquist_does_not_fold_back` measures.
const KERNEL_BETA: f64 = 8.6;

/// Kernel table entries per zero crossing. The table is read with linear
/// interpolation between entries, so this sets how much of the stopband depth
/// survives the lookup; at 512 the quantisation sits far below
/// [`KERNEL_BETA`]'s sidelobes and is not what limits the measured figures.
const KERNEL_DENSITY: usize = 512;

/// Modified Bessel function of the first kind, order zero — the Kaiser window's
/// defining term, and the only reason this file does arithmetic `std` does not
/// provide. The series converges fast for the arguments a window uses.
fn bessel_i0(x: f64) -> f64 {
    let mut sum = 1.0f64;
    let mut term = 1.0f64;
    let half = x / 2.0;
    for k in 1..80 {
        let f = half / k as f64;
        term *= f * f;
        sum += term;
        if term < sum * 1e-17 {
            break;
        }
    }
    sum
}

/// A Kaiser-windowed sinc, tabulated once per [`resample`] call and read by
/// distance in input samples.
///
/// It is a table rather than a closure because the window costs a Bessel
/// evaluation per point: computed per tap that would dominate the resample,
/// where computed once it is a few thousand evaluations for the whole file.
struct Kernel {
    /// `h(x)` for `x` in `[0, half]`. Symmetric, so only one side is stored.
    table: Vec<f32>,
    /// Support half-width, in input samples.
    half: f64,
    /// `table.len() - 1`, kept as `f64` for the lookup.
    steps: f64,
}

impl Kernel {
    /// `cutoff` is in cycles per input sample; `half` is the support half-width
    /// in input samples.
    fn new(cutoff: f64, half: f64) -> Self {
        let steps = KERNEL_ZEROS * KERNEL_DENSITY;
        let norm = bessel_i0(KERNEL_BETA);
        let table = (0..=steps)
            .map(|k| {
                let x = k as f64 / steps as f64 * half;
                let t = x / half;
                let window = bessel_i0(KERNEL_BETA * (1.0 - t * t).max(0.0).sqrt()) / norm;
                let a = std::f64::consts::PI * 2.0 * cutoff * x;
                let sinc = if a.abs() < 1e-12 { 1.0 } else { a.sin() / a };
                (sinc * window) as f32
            })
            .collect();
        Self {
            table,
            half,
            steps: steps as f64,
        }
    }

    /// The kernel's value `x` input samples from its centre.
    #[inline]
    fn at(&self, x: f64) -> f32 {
        let p = x.abs() / self.half * self.steps;
        let i = p as usize;
        if i >= self.table.len() - 1 {
            return 0.0;
        }
        let f = (p - i as f64) as f32;
        self.table[i] + (self.table[i + 1] - self.table[i]) * f
    }
}

/// Resample a mono signal from `src` to `dst` Hz.
///
/// Every decode in the toolkit passes through here, in both directions and at
/// ratios that are rarely integers, so this is a band-limited interpolation
/// rather than the two cheap approximations it replaces.
///
/// **What it replaced, and why that mattered**, because "a cheap anti-alias" is
/// what the old comment called it: downsampling was an unweighted average over
/// each output sample's input window, which is a box filter whose stopband
/// starts at −13 dB, and upsampling was bare linear interpolation, whose images
/// sit at about −24 dB. Neither is a small error, and both baselines are
/// computed rather than recalled — see
/// `the_box_average_this_replaced_aliased_where_this_does_not` and
/// `an_upsample_leaves_no_image_above_the_source_nyquist`, which run the
/// replaced arithmetic beside this one. A 12 kHz tone taken from
/// 48 kHz to 16 kHz came back as a 4 kHz tone **9.5 dB** below the input —
/// arithmetically exact, since the window is three samples wide at that ratio
/// and the tone is `0, 1, 0, -1, …`, so a third of it survives at the fold. That
/// is a full-strength artefact in the middle of the speech band, on material
/// where nothing downstream can tell it from signal. The tests named on the
/// constants above are what pin the replacement, and the number to read first is
/// `a_tone_above_the_output_nyquist_does_not_fold_back`.
///
/// **One branch serves both directions**, which is not tidiness: the cutoff is
/// half the *lower* of the two rates either way, so an upsample gets the same
/// image rejection a downsample gets alias rejection, and there is no second
/// path to keep in step.
///
/// **Each output sample is normalised by the weights it actually used.** That is
/// what makes unity gain exact rather than approximate — a truncated kernel's
/// taps sum to slightly different totals at different fractional phases, and
/// dividing it out removes that variation instead of leaving it as a wobble on
/// the envelope. It also gives the first and last few samples, where the kernel
/// hangs off the end of the buffer, the right level rather than a fade.
///
/// The rates matching is still a byte-for-byte identity, which is the case every
/// native-rate decode in the workspace takes.
fn resample(input: &[f32], src: u32, dst: u32) -> Vec<f32> {
    if src == dst || src == 0 || input.len() < 2 {
        return input.to_vec();
    }
    let ratio = dst as f64 / src as f64;
    let out_len = ((input.len() as f64) * ratio).round().max(1.0) as usize;
    let step = 1.0 / ratio; // input samples per output sample

    // Cutoff in cycles per input sample: half the lower rate, pulled inside
    // Nyquist by the rolloff so the transition band has somewhere to close.
    let cutoff = 0.5 * KERNEL_ROLLOFF * src.min(dst) as f64 / src as f64;
    let half = KERNEL_ZEROS as f64 / (2.0 * cutoff);
    let kernel = Kernel::new(cutoff, half);

    let n = input.len() as isize;
    let mut out = Vec::with_capacity(out_len);
    for j in 0..out_len {
        let center = j as f64 * step;
        let lo = ((center - half).ceil() as isize).max(0);
        let hi = ((center + half).floor() as isize).min(n - 1);
        let mut acc = 0.0f32;
        let mut weight = 0.0f32;
        for i in lo..=hi {
            let w = kernel.at(center - i as f64);
            acc += w * input[i as usize];
            weight += w;
        }
        out.push(if weight.abs() > 1e-9 {
            acc / weight
        } else {
            0.0
        });
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
    ///
    /// Above two channels this header is deliberately the *simple* one and not
    /// a conformant one: the spec wants `WAVE_FORMAT_EXTENSIBLE` with a channel
    /// mask, where this keeps tag 3 and a 16-byte `fmt`. ffmpeg parses it
    /// leniently and assigns a default layout, which is all the fold tests
    /// need — they care which channel index carries what, not what the layout
    /// is called.
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

    /// Amplitude of `freq` in `x`, by least squares against a cosine/sine pair
    /// at that frequency.
    ///
    /// **Not a Goertzel**, and the difference is the reason this helper exists
    /// rather than the obvious ten-line one. Goertzel assumes the two basis
    /// vectors are orthogonal and equally long over the window, which holds only
    /// when the window spans a whole number of periods. 3 kHz at a 16 kHz output
    /// is 5.33 samples per period, so no short window does — and the leakage
    /// that follows reads as *amplitude modulation that is not in the signal*.
    /// Measuring this resampler's envelope with a Goertzel reports 2.1 dB of
    /// ripple for the box filter and the same 2.1 dB for a 200-tap sinc, which
    /// is the metric talking rather than the code. A least-squares projection is
    /// exact for a pure tone at any window length.
    fn amplitude_at(x: &[f32], sr: f64, freq: f64) -> f64 {
        let w = std::f64::consts::TAU * freq / sr;
        let (mut cc, mut cs, mut ss, mut xc, mut xs) = (0.0f64, 0.0, 0.0, 0.0, 0.0);
        for (i, &v) in x.iter().enumerate() {
            let (c, s) = ((w * i as f64).cos(), (w * i as f64).sin());
            cc += c * c;
            cs += c * s;
            ss += s * s;
            xc += v as f64 * c;
            xs += v as f64 * s;
        }
        let det = cc * ss - cs * cs;
        if det.abs() < 1e-12 {
            return 0.0;
        }
        let a = (xc * ss - xs * cs) / det;
        let b = (xs * cc - xc * cs) / det;
        (a * a + b * b).sqrt()
    }

    /// A full-scale tone, as `f64` phase so the source itself is not the thing
    /// under test.
    fn full_scale_tone(freq: f64, sr: u32, samples: usize) -> Vec<f32> {
        (0..samples)
            .map(|i| (std::f64::consts::TAU * freq * i as f64 / sr as f64).sin() as f32)
            .collect()
    }

    fn db(x: f64) -> f64 {
        20.0 * x.max(1e-30).log10()
    }

    /// Enough of the ends dropped that the kernel hanging off the buffer is not
    /// what any of these readings measure.
    const GUARD: usize = 800;

    /// The check the old resampler failed outright, and the one worth reading
    /// first. Content above the output's Nyquist has to be filtered away
    /// *before* the rate drops, because afterwards it is indistinguishable from
    /// signal at the frequency it folded to.
    ///
    /// The box average scored −9.54 dB on the first case — a 12 kHz tone at
    /// 48 kHz is `0, 1, 0, -1, …`, and averaging every three samples leaves
    /// exactly a third of it at 4 kHz, in the middle of the speech band. The
    /// thresholds below are loose against what this kernel actually measures
    /// (−113, −90, −103 dB) so that the test reports a regression rather than
    /// tracking arithmetic noise.
    #[test]
    fn a_tone_above_the_output_nyquist_does_not_fold_back() {
        // (tone, src, dst, where it would fold to, ceiling)
        let cases = [
            (12_000.0, 48_000u32, 16_000u32, 4_000.0, -80.0),
            (8_500.0, 44_100, 16_000, 7_500.0, -60.0),
            (10_000.0, 44_100, 16_000, 6_000.0, -80.0),
        ];
        for (freq, src, dst, image, ceiling) in cases {
            let x = full_scale_tone(freq, src, src as usize * 2);
            let y = resample(&x, src, dst);
            let level = db(amplitude_at(&y[GUARD..y.len() - GUARD], dst as f64, image));
            assert!(
                level < ceiling,
                "{freq} Hz at {src}->{dst} left {level:.2} dB at {image} Hz (want < {ceiling})"
            );
        }
    }

    /// The mirror of the alias test, in the other direction, and with the
    /// arithmetic it replaced computed beside it for the same reason
    /// [`box_average`] exists.
    ///
    /// Linear interpolation is a triangular kernel, so its response is a
    /// `sinc²` and the images it leaves are shallow: a 3 kHz tone taken from
    /// 16 kHz to 48 kHz comes back with **−24.3 dB** at 13 kHz and −28.4 at
    /// 19 kHz, against **−98.7** and −106.4 here. The windowed sinc removes
    /// them for free because its cutoff is half the *lower* of the two rates
    /// either way, so an upsample is filtered by the same kernel a downsample
    /// is.
    #[test]
    fn an_upsample_leaves_no_image_above_the_source_nyquist() {
        // The old upsampling branch verbatim.
        let linear = |input: &[f32], src: u32, dst: u32| -> Vec<f32> {
            let ratio = dst as f64 / src as f64;
            let out_len = ((input.len() as f64) * ratio).round().max(1.0) as usize;
            let step = 1.0 / ratio;
            (0..out_len)
                .map(|j| {
                    let pos = j as f64 * step;
                    let i0 = pos.floor() as usize;
                    let i1 = (i0 + 1).min(input.len() - 1);
                    let frac = (pos - i0 as f64) as f32;
                    input[i0] * (1.0 - frac) + input[i1] * frac
                })
                .collect()
        };

        let x = full_scale_tone(3_000.0, 16_000, 32_000);
        let y = resample(&x, 16_000, 48_000);
        let old = linear(&x, 16_000, 48_000);
        let body = &y[GUARD..y.len() - GUARD];
        assert!(
            db(amplitude_at(body, 48_000.0, 3_000.0)) > -0.1,
            "the tone itself did not survive"
        );
        for image in [13_000.0, 19_000.0] {
            let was = db(amplitude_at(
                &old[GUARD..old.len() - GUARD],
                48_000.0,
                image,
            ));
            let now = db(amplitude_at(body, 48_000.0, image));
            assert!(
                was > -35.0,
                "linear interpolation should image here: {was:.2} dB"
            );
            assert!(
                now < -70.0,
                "image at {image} Hz: {now:.2} dB (was {was:.2})"
            );
        }
    }

    /// A steady tone must come out steady. A resampler whose window length
    /// alternates between ratios modulates the amplitude at the beat frequency,
    /// which is a defect no spectrum taken over the whole file would show.
    ///
    /// Read [`amplitude_at`]'s doc before changing this: measured with the
    /// obvious Goertzel it reports ripple that is not there.
    #[test]
    fn the_envelope_of_a_resampled_tone_stays_flat() {
        for freq in [1_000.0, 3_000.0, 5_000.0, 6_000.0, 7_000.0] {
            let x = full_scale_tone(freq, 44_100, 88_200);
            let y = resample(&x, 44_100, 16_000);
            let body = &y[GUARD..y.len() - GUARD];
            let window = ((8.0 * 16_000.0 / freq).ceil() as usize).max(64);
            let (mut lo, mut hi) = (f64::MAX, 0.0f64);
            let mut k = 0;
            while k + window <= body.len() {
                let a = amplitude_at(&body[k..k + window], 16_000.0, freq);
                lo = lo.min(a);
                hi = hi.max(a);
                k += window / 4;
            }
            let ripple = db(hi) - db(lo);
            assert!(ripple < 0.05, "{freq} Hz wobbles by {ripple:.3} dB");
        }
    }

    /// Everything the toolkit cares about at a 16 kHz output lives under 7 kHz,
    /// and it has to arrive at the level it left. The box average was down
    /// **5.6 dB** at 7 kHz through 44.1 → 16 kHz — a treble loss applied to
    /// every corpus that was ever decoded off-rate, and a figure
    /// `the_box_average_this_replaced_aliased_where_this_does_not` recomputes
    /// rather than quoting.
    ///
    /// The last row is the price of [`KERNEL_ROLLOFF`] and is asserted as a
    /// *range*: the edge is supposed to roll off, so a flat reading there would
    /// mean the transition band had moved out past Nyquist and the alias test
    /// above is the one that would start failing.
    #[test]
    fn the_top_of_the_passband_is_flat_and_the_edge_rolls_off() {
        for (src, dst) in [(48_000u32, 16_000u32), (44_100, 16_000)] {
            for freq in [100.0, 1_000.0, 3_000.0, 5_000.0, 6_000.0, 7_000.0] {
                let x = full_scale_tone(freq, src, src as usize * 2);
                let y = resample(&x, src, dst);
                let level = db(amplitude_at(&y[GUARD..y.len() - GUARD], dst as f64, freq));
                assert!(level > -0.5, "{freq} Hz at {src}->{dst} lost {level:.3} dB");
            }
            let x = full_scale_tone(7_500.0, src, src as usize * 2);
            let y = resample(&x, src, dst);
            let edge = db(amplitude_at(
                &y[GUARD..y.len() - GUARD],
                dst as f64,
                7_500.0,
            ));
            assert!(
                (-8.0..-1.0).contains(&edge),
                "7500 Hz at {src}->{dst} is {edge:.3} dB, outside the transition band"
            );
        }
    }

    /// Unity gain, checked over the **whole** buffer rather than its middle, so
    /// the ends — where the kernel hangs off the buffer and the weight sum is
    /// the thing holding the level up — are part of the claim.
    #[test]
    fn a_constant_comes_back_as_the_same_constant() {
        for (src, dst) in [
            (44_100u32, 16_000u32),
            (16_000, 48_000),
            (48_000, 16_000),
            (22_050, 44_100),
            (44_100, 22_050),
        ] {
            // Long enough to visit every fractional phase of the ratio.
            let y = resample(&vec![0.7f32; src as usize], src, dst);
            for (i, v) in y.iter().enumerate() {
                assert!(
                    (v - 0.7).abs() < 1e-5,
                    "{src}->{dst} sample {i}: {v} != 0.7"
                );
            }
        }
    }

    /// The output length is a function of the input length and the ratio alone,
    /// which is what keeps two channels resampled separately the same length and
    /// what `preprocess`'s `the_rate_changes_and_the_duration_does_not` rests on.
    #[test]
    fn the_output_length_follows_the_rate_ratio() {
        for (len, src, dst) in [
            (44_100usize, 44_100u32, 16_000u32),
            (44_100, 48_000, 16_000),
            (16_000, 16_000, 48_000),
            (1_000, 22_050, 44_100),
            (3, 44_100, 16_000),
        ] {
            let want = ((len as f64) * dst as f64 / src as f64).round().max(1.0) as usize;
            assert_eq!(
                resample(&vec![0.0; len], src, dst).len(),
                want,
                "{len} @ {src}->{dst}"
            );
        }
    }

    /// The three inputs that take the early return come back untouched — the
    /// matching-rate case especially, since it is the path every native-rate
    /// decode in the workspace takes and the reason the other tests in this
    /// module can compare exactly.
    #[test]
    fn a_resample_that_has_nothing_to_do_returns_its_input() {
        let x = vec![0.25f32, -0.5, 0.75];
        assert_eq!(resample(&x, 16_000, 16_000), x);
        assert_eq!(resample(&x, 0, 16_000), x);
        assert_eq!(resample(&[0.4], 44_100, 16_000), vec![0.4]);
        assert!(resample(&[], 44_100, 16_000).is_empty());
    }

    /// The defect this kernel replaced, kept runnable so the numbers quoted for
    /// it are computed rather than remembered.
    ///
    /// This is the old downsampling branch verbatim: an unweighted average over
    /// `[j·step, j·step + step)`, described at the time as "a cheap
    /// anti-alias". A box filter's first sidelobe is 13 dB down and its
    /// stopband never gets deeper, so "cheap" understated it — at 48 → 16 kHz
    /// the window is exactly three samples, a 12 kHz tone is `0, 1, 0, -1, …`,
    /// and a third of it survives at 4 kHz.
    fn box_average(input: &[f32], src: u32, dst: u32) -> Vec<f32> {
        let ratio = dst as f64 / src as f64;
        let out_len = ((input.len() as f64) * ratio).round().max(1.0) as usize;
        let step = 1.0 / ratio;
        (0..out_len)
            .map(|j| {
                let start = j as f64 * step;
                let a = (start.floor() as usize).min(input.len() - 1);
                let b = ((start + step).ceil() as usize).clamp(a + 1, input.len());
                input[a..b].iter().sum::<f32>() / (b - a) as f32
            })
            .collect()
    }

    /// The comparison that says the replacement was worth making, rather than
    /// leaving the reader to take the new figures on trust. Both columns are
    /// computed here; the thresholds are the *gap*, which is the quantity that
    /// must not regress.
    ///
    /// What it reports, in dB of the image relative to a full-scale input:
    ///
    /// | tone | rates | folds to | box average | this kernel |
    /// |---|---|---|---|---|
    /// | 12 kHz | 48 → 16 k | 4 kHz | −9.5 | −113.2 |
    /// | 10 kHz | 44.1 → 16 k | 6 kHz | −14.6 | −95.1 |
    /// | 8.5 kHz | 44.1 → 16 k | 7.5 kHz | −9.1 | −89.8 |
    ///
    /// …and, on the passband it was supposed to be leaving alone, 44.1 → 16 kHz:
    /// −0.9 dB at 3 kHz, −2.7 at 5 kHz and −5.6 at 7 kHz against this kernel's
    /// −0.000, −0.000 and −0.138.
    #[test]
    fn the_box_average_this_replaced_aliased_where_this_does_not() {
        for (freq, src, dst, image) in [
            (12_000.0, 48_000u32, 16_000u32, 4_000.0),
            (10_000.0, 44_100, 16_000, 6_000.0),
            (8_500.0, 44_100, 16_000, 7_500.0),
        ] {
            let x = full_scale_tone(freq, src, src as usize * 2);
            let read = |y: &[f32]| db(amplitude_at(&y[GUARD..y.len() - GUARD], dst as f64, image));
            let old = read(&box_average(&x, src, dst));
            let new = read(&resample(&x, src, dst));
            assert!(
                old > -25.0,
                "{freq} Hz: the box average was supposed to alias here, got {old:.2} dB"
            );
            assert!(
                new < old - 60.0,
                "{freq} Hz: box {old:.2} dB vs kernel {new:.2} dB — under 60 dB of gain"
            );
        }

        // The other half of the trade: the box filter is a low-pass, so it was
        // also eating the passband it was supposed to be passing.
        for (freq, floor) in [(3_000.0, -0.5), (5_000.0, -2.0), (7_000.0, -4.0)] {
            let x = full_scale_tone(freq, 44_100, 88_200);
            let read = |y: &[f32]| db(amplitude_at(&y[GUARD..y.len() - GUARD], 16_000.0, freq));
            let old = read(&box_average(&x, 44_100, 16_000));
            let new = read(&resample(&x, 44_100, 16_000));
            assert!(old < floor, "{freq} Hz: box average only lost {old:.3} dB");
            assert!(new > -0.5, "{freq} Hz: this kernel lost {new:.3} dB");
        }
    }

    /// The same claims, but through the real call site rather than the private
    /// function: a 44.1 kHz file asked for at 16 kHz. This is what every engine
    /// in the toolkit actually does, and none of the five tests around it
    /// exercised it — they all decode at the file's own rate, which takes the
    /// early return.
    #[tokio::test]
    async fn a_decode_at_another_rate_keeps_the_tone_and_adds_no_image() {
        const SRC: u32 = 44_100;
        const DST: u32 = 16_000;
        // 3 kHz survives; 12 kHz is above the 8 kHz output Nyquist and must not
        // reappear at 4 kHz.
        let mixed: Vec<f32> = full_scale_tone(3_000.0, SRC, SRC as usize * 2)
            .iter()
            .zip(full_scale_tone(12_000.0, SRC, SRC as usize * 2).iter())
            .map(|(a, b)| 0.4 * a + 0.4 * b)
            .collect();
        let path = temp_wav("offrate", &float_wav(SRC, 1, &mixed));

        let got = collect_mono(&path, DST).await;
        let _ = std::fs::remove_file(&path);

        let want_len = (mixed.len() as f64 * DST as f64 / SRC as f64).round() as usize;
        assert_eq!(got.len(), want_len);

        let body = &got[GUARD..got.len() - GUARD];
        let kept = db(amplitude_at(body, DST as f64, 3_000.0) / 0.4);
        assert!(kept > -0.5, "the 3 kHz tone lost {kept:.3} dB");
        let folded = db(amplitude_at(body, DST as f64, 4_000.0) / 0.4);
        assert!(folded < -70.0, "12 kHz folded back at {folded:.2} dB");
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
