//! Prove the two pieces Seed-VC borrows actually fit, on a real clip.
//!
//! Neither is a port, so neither has a coverage number of its own — what has to
//! be checked instead is that the shapes and the arithmetic come out right:
//!
//! 1. `openai/whisper-small`'s encoder loads and emits **768-wide** features at
//!    50 Hz, which is the width `length_regulator`'s input projection expects.
//!    768 is the load-bearing number: it is why this preset was the first target.
//! 2. [`burn_vits::Spectral`] at the preset's 22 050 / 80 / 1024 / 256 gives
//!    exactly `samples / 256` frames — the grid the whole model downstream of the
//!    length regulator lives on.
//!
//! Finiteness is asserted separately from the shapes on purpose. A wrong mask or
//! a log of zero yields `NaN`, which propagates silently through every later
//! layer, and Burn's `assert_approx_eq` compares `NaN` against `NaN` without
//! complaint — so a shape check alone would pass a front end producing nothing.
//!
//! Usage:
//! `cargo run -p burn-seedvc --example content -- [--backend ndarray|cuda|tch] <whisper-small/model.safetensors> <clip.wav>`

#[path = "common/mod.rs"]
mod common;

use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, TensorData};
use burn_seedvc::config::{CONTENT_SR, SeedVcConfig};
use burn_seedvc::content::{CONTENT_STRIDE, ContentEncoder, mel_config};
use burn_vits::Spectral;

struct Content {
    weights: String,
    clip: String,
}

impl common::Job for Content {
    fn run<B: Backend>(self, device: &B::Device) {
        let cfg = SeedVcConfig::uvit_whisper_small_wavenet();
        let (source, source_sr) = read_wav(&self.clip);
        println!(
            "clip    : {} — {} samples at {} Hz ({:.2} s)",
            self.clip,
            source.len(),
            source_sr,
            source.len() as f32 / source_sr as f32
        );

        // The same recording, resampled twice to two different rates. That is
        // not redundancy: 16 kHz is Whisper's analysis rate and 22 050 Hz is the
        // preset's, and no single rate serves both.
        let content_wav = resample(&source, source_sr, CONTENT_SR);
        let mel_wav = resample(&source, source_sr, cfg.sample_rate);

        // ---- 1. the content encoder ----------------------------------------
        let mut encoder = ContentEncoder::<B>::new(device);
        let res = encoder
            .load_safetensors(&self.weights)
            .expect("failed to read the whisper checkpoint");
        println!(
            "\nencoder : applied {}, missing {}, errors {}",
            res.applied.len(),
            res.missing.len(),
            res.errors.len()
        );
        for (name, why) in &res.missing {
            println!("    MISSING {name}  ({why})");
        }
        println!(
            "          unused  {} — the decoder half, deleted upstream too",
            res.unused.len()
        );
        assert!(res.missing.is_empty(), "the encoder is not fully covered");
        assert!(res.errors.is_empty(), "{:?}", res.errors);

        let audio: Tensor<B, 2> = Tensor::from_data(
            TensorData::new(content_wav.clone(), [1, content_wav.len()]),
            device,
        );
        let features = encoder.forward(audio);
        let [batch, frames, width] = features.dims();
        println!(
            "content : [{batch}, {frames}, {width}] at {} Hz — expected {} frames \
             ({} samples / {CONTENT_STRIDE} + 1)",
            CONTENT_SR as usize / CONTENT_STRIDE,
            content_wav.len() / CONTENT_STRIDE + 1,
            content_wav.len()
        );
        assert_eq!(
            width, cfg.content_dim,
            "the length regulator projects from {}, not {width}",
            cfg.content_dim
        );
        assert_eq!(frames, content_wav.len() / CONTENT_STRIDE + 1);

        let v: Vec<f32> = features.into_data().to_vec().unwrap();
        assert!(
            v.iter().all(|x| x.is_finite()),
            "content features are not finite"
        );
        let mean = v.iter().sum::<f32>() / v.len() as f32;
        let peak = v.iter().fold(0f32, |a, x| a.max(x.abs()));
        println!("          finite, mean {mean:.4}, peak |x| {peak:.4}");

        // ---- 2. the mel front end ------------------------------------------
        let spectral = Spectral::<B>::new(&mel_config(&cfg), device);
        let wav: Tensor<B, 2> =
            Tensor::from_data(TensorData::new(mel_wav.clone(), [1, mel_wav.len()]), device);
        let mel = spectral.mel(wav);
        let [_, bands, mel_frames] = mel.dims();
        println!(
            "\nmel     : [1, {bands}, {mel_frames}] at {:.2} Hz — expected {} frames \
             ({} samples / {})",
            cfg.frame_rate(),
            mel_wav.len() / cfg.hop_length,
            mel_wav.len(),
            cfg.hop_length
        );
        assert_eq!(bands, cfg.n_mels);
        assert_eq!(mel_frames, mel_wav.len() / cfg.hop_length);

        let v: Vec<f32> = mel.into_data().to_vec().unwrap();
        assert!(v.iter().all(|x| x.is_finite()), "the mel is not finite");
        let lo = v.iter().copied().fold(f32::INFINITY, f32::min);
        let hi = v.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        println!("          finite, range {lo:.3}..{hi:.3}");

        println!(
            "\nratio   : {mel_frames} mel frames to {frames} content frames \
             ({:.3}x) — what the length regulator has to close",
            mel_frames as f32 / frames as f32
        );
    }
}

/// Minimal RIFF/WAVE reader: 16-bit PCM or 32-bit float, any channel count,
/// downmixed to mono `f32`.
///
/// Hand-rolled because `burn-*` crates carry no app dependencies, and decoding
/// anything richer than a WAV is `audio-kit`'s job in the real pipeline.
fn read_wav(path: &str) -> (Vec<f32>, u32) {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{path}: {e}"));
    assert!(
        bytes.len() > 12 && &bytes[0..4] == b"RIFF" && &bytes[8..12] == b"WAVE",
        "{path} is not a RIFF/WAVE file"
    );

    let u16at = |i: usize| u16::from_le_bytes([bytes[i], bytes[i + 1]]);
    let u32at = |i: usize| u32::from_le_bytes([bytes[i], bytes[i + 1], bytes[i + 2], bytes[i + 3]]);

    let (mut format, mut channels, mut rate, mut bits) = (0u16, 0u16, 0u32, 0u16);
    let mut pos = 12;
    while pos + 8 <= bytes.len() {
        let id = &bytes[pos..pos + 4];
        let size = u32at(pos + 4) as usize;
        let body = pos + 8;
        match id {
            b"fmt " => {
                format = u16at(body);
                channels = u16at(body + 2);
                rate = u32at(body + 4);
                bits = u16at(body + 14);
            }
            b"data" => {
                let end = (body + size).min(bytes.len());
                let ch = channels.max(1) as usize;
                let frames: Vec<f32> = match (format, bits) {
                    (3, 32) => bytes[body..end]
                        .chunks_exact(4)
                        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                        .collect(),
                    (1, 16) => bytes[body..end]
                        .chunks_exact(2)
                        .map(|c| i16::from_le_bytes([c[0], c[1]]) as f32 / 32768.0)
                        .collect(),
                    _ => panic!("{path}: unsupported WAV format {format} at {bits} bits"),
                };
                // Downmix: the whole toolkit passes mono f32 across every boundary.
                let mono = frames
                    .chunks(ch)
                    .map(|f| f.iter().sum::<f32>() / ch as f32)
                    .collect();
                return (mono, rate);
            }
            _ => {}
        }
        pos = body + size + (size & 1); // chunks are word-aligned
    }
    panic!("{path}: no data chunk");
}

/// Windowed-sinc resample, any ratio.
///
/// Linear interpolation would be enough for a shape check and wrong for this
/// one: 48 kHz down to 16 kHz folds everything above 8 kHz back into the speech
/// band, and the encoder would then be judged on audio it never saw. The cutoff
/// tracks the *lower* of the two rates, which is what makes it an anti-alias
/// filter on the way down and a plain interpolator on the way up.
fn resample(input: &[f32], from: u32, to: u32) -> Vec<f32> {
    if from == to {
        return input.to_vec();
    }
    const HALF_WIDTH: isize = 16;
    let step = from as f64 / to as f64;
    let cutoff = (to as f64 / from as f64).min(1.0);
    let out_len = (input.len() as f64 / step).floor() as usize;

    (0..out_len)
        .map(|i| {
            let centre = i as f64 * step;
            let base = centre.floor() as isize;
            let (mut acc, mut norm) = (0.0f64, 0.0f64);
            for k in -HALF_WIDTH..=HALF_WIDTH {
                let idx = base + k;
                if idx < 0 || idx as usize >= input.len() {
                    continue;
                }
                let d = centre - idx as f64;
                // Blackman window over the sinc, which is what keeps the
                // stopband deep enough for the fold-back to stay inaudible.
                let t = (d / (HALF_WIDTH as f64 + 1.0) + 1.0) / 2.0;
                let w = 0.42 - 0.5 * (std::f64::consts::TAU * t).cos()
                    + 0.08 * (2.0 * std::f64::consts::TAU * t).cos();
                let x = std::f64::consts::PI * d * cutoff;
                let sinc = if x.abs() < 1e-9 { 1.0 } else { x.sin() / x };
                acc += input[idx as usize] as f64 * sinc * w;
                norm += sinc * w;
            }
            (if norm.abs() > 1e-9 { acc / norm } else { acc }) as f32
        })
        .collect()
}

fn main() {
    let (backend, args) = common::parse_args();
    let (Some(weights), Some(clip)) = (args.first().cloned(), args.get(1).cloned()) else {
        eprintln!(
            "usage: content [--backend ndarray|cuda|tch] <whisper-small/model.safetensors> <clip.wav>"
        );
        std::process::exit(2);
    };
    println!("backend : {backend}");
    common::run_on(backend, Content { weights, clip });
}
