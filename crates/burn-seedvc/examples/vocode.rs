//! Run a real mel through BigVGAN and see whether the waveform tracks it.
//!
//! The check weight coverage cannot make. Every tensor of the vocoder lands at 0
//! missing, and that says the module tree matches NVIDIA's checkpoint — it says
//! nothing about the snake activation's β being read where β belongs, or the
//! anti-aliasing trims being off by a sample. Both load perfectly and produce
//! something that is not the signal.
//!
//! audio → `burn_vits::Spectral::mel` → BigVGAN → audio, then the output's
//! **energy envelope** is correlated against the input's. Envelope rather than
//! samples on purpose: a vocoder reconstructs phase from scratch, so a
//! sample-wise difference is large between two runs of the *same* model and
//! proves nothing either way. A shuffled control is printed beside the
//! correlation, because a number is only meaningful against the value chance
//! would give it, and spectral flatness beside that, because noise is flat where
//! speech is not.
//!
//! The input is raw mono f32le at 22.05 kHz — the rate BigVGAN was trained at,
//! and not a rate this crate is allowed to resample to itself, since a `burn-*`
//! crate has no app dependencies and ffmpeg is an app dependency:
//!
//! ```sh
//! ffmpeg -i clip.wav -f f32le -ac 1 -ar 22050 clip.raw
//! cargo run -p burn-seedvc --example vocode -- bigvgan_generator.pt clip.raw out.raw
//! ffmpeg -f f32le -ac 1 -ar 22050 -i out.raw out.wav
//! ```
//!
//! # What it reads on this repository's own corpus
//!
//! 3 s of close-mic speech from `dataset/`, `nvidia/bigvgan_v2_22khz_80band_256x`
//! on the `ndarray` backend:
//!
//! ```text
//! weights : 783 applied, 0 missing, 0 unused
//! r vs source      : 0.9792
//! r vs shuffled    : 0.0957   (chance baseline)
//! spectral flatness: 0.2761   (1.0 would be white noise)
//! ```
//!
//! **The chance baseline is what makes the first number mean anything**, and it
//! is why the shuffle is printed rather than assumed: an envelope correlation
//! taken against a signal of the same overall shape can be high for reasons that
//! have nothing to do with the vocoder. 0.98 against 0.10 is the port working.
//!
//! Expect this to be slow off a GPU — 3 s of audio is a couple of minutes on
//! `ndarray`, which is fine for a check that is run when the model changes and
//! not in a loop.

#[path = "common/mod.rs"]
mod common;

use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, TensorData};
use burn_seedvc::{BigVgan, BigVganConfig};
use burn_vits::{Spectral, SpectralConfig};

/// The rate the released weights work at, and the STFT they were trained
/// against. `config.json` of the weight repo: `n_fft` 1024, `hop_size` 256,
/// `win_size` 1024, `num_mels` 80, `fmin` 0, `fmax` null — and a null `fmax` is
/// librosa's default of half the sample rate, not "no limit".
const SAMPLE_RATE: usize = 22_050;
const HOP: usize = 256;

struct Vocode {
    weights: String,
    input: String,
    output: Option<String>,
}

/// Per-frame RMS: the signal's loudness contour, at the mel's own frame rate.
fn envelope(samples: &[f32]) -> Vec<f64> {
    samples
        .chunks_exact(HOP)
        .map(|frame| {
            let power: f64 = frame.iter().map(|s| (*s as f64).powi(2)).sum();
            (power / HOP as f64).sqrt()
        })
        .collect()
}

fn correlation(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    let (a, b) = (&a[..n], &b[..n]);
    let mean = |v: &[f64]| v.iter().sum::<f64>() / n as f64;
    let (ma, mb) = (mean(a), mean(b));
    let mut cov = 0.0;
    let (mut va, mut vb) = (0.0, 0.0);
    for (x, y) in a.iter().zip(b) {
        cov += (x - ma) * (y - mb);
        va += (x - ma).powi(2);
        vb += (y - mb).powi(2);
    }
    cov / (va * vb).sqrt().max(f64::MIN_POSITIVE)
}

/// Fisher-Yates against a fixed seed, so the chance baseline is the same number
/// on every run and a reader can tell a real change from a reshuffle.
fn shuffled(values: &[f64]) -> Vec<f64> {
    let mut out = values.to_vec();
    let mut state = 0x2545_F491_4F6C_DD1Du64;
    for i in (1..out.len()).rev() {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        out.swap(i, (state >> 33) as usize % (i + 1));
    }
    out
}

/// Geometric mean over arithmetic mean of the magnitude spectrum, averaged over
/// frames. White noise sits at 1.0; speech, which is all harmonics and formants,
/// sits far below it.
fn spectral_flatness(magnitudes: &[f32], bins: usize) -> f64 {
    let frames = magnitudes.len() / bins;
    let mut total = 0.0;
    for f in 0..frames {
        let (mut log_sum, mut sum) = (0.0f64, 0.0f64);
        for b in 0..bins {
            let m = (magnitudes[b * frames + f] as f64).max(1e-10);
            log_sum += m.ln();
            sum += m;
        }
        total += (log_sum / bins as f64).exp() / (sum / bins as f64);
    }
    total / frames.max(1) as f64
}

impl common::Job for Vocode {
    fn run<B: Backend>(self, device: &B::Device) {
        let raw = std::fs::read(&self.input).expect("read input");
        let pcm: Vec<f32> = raw
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        println!(
            "input   : {} samples, {:.2} s at {SAMPLE_RATE} Hz",
            pcm.len(),
            pcm.len() as f32 / SAMPLE_RATE as f32
        );

        let spectral = Spectral::<B>::new(
            &SpectralConfig {
                sample_rate: SAMPLE_RATE,
                n_fft: 1024,
                hop: HOP,
                win_length: 1024,
                n_mels: 80,
                fmin: 0.0,
                fmax: SAMPLE_RATE as f32 / 2.0,
            },
            device,
        );
        let wav = Tensor::<B, 2>::from_data(TensorData::new(pcm.clone(), [1, pcm.len()]), device);
        let mel = spectral.mel(wav);
        println!("mel     : {:?}", mel.dims());

        let mut model = BigVgan::<B>::new(&BigVganConfig::v2_22khz_80band_256x(), device);
        let res = model.load_pytorch(&self.weights).expect("load bigvgan");
        println!(
            "weights : {} applied, {} missing, {} unused",
            res.applied.len(),
            res.missing.len(),
            res.unused.len()
        );
        assert!(res.missing.is_empty(), "{:?}", res.missing);

        let audio = model.forward(mel);
        let out: Vec<f32> = audio.into_data().to_vec().unwrap();
        let peak = out.iter().fold(0.0f32, |a, b| a.max(b.abs()));
        println!(
            "output  : {} samples, peak {peak:.4}, finite {}",
            out.len(),
            out.iter().all(|s| s.is_finite())
        );

        // Both envelopes are cut on the same 256-sample grid from sample 0. The
        // mel's frames are *centred* on that grid rather than starting on it, so
        // the two are offset by half a hop — 5.8 ms, against an envelope that
        // moves on the scale of a syllable, which is why it is not corrected for.
        let source = envelope(&pcm);
        let vocoded = envelope(&out);
        let r = correlation(&source, &vocoded);
        let chance = correlation(&source, &shuffled(&vocoded));
        println!(
            "\nenergy envelope ({} frames)",
            source.len().min(vocoded.len())
        );
        println!("  r vs source      : {r:.4}");
        println!("  r vs shuffled    : {chance:.4}   (chance baseline)");

        let mag = spectral.linear(Tensor::from_data(
            TensorData::new(out.clone(), [1, out.len()]),
            device,
        ));
        let bins = mag.dims()[1];
        let flatness = spectral_flatness(&mag.into_data().to_vec::<f32>().unwrap(), bins);
        println!("  spectral flatness: {flatness:.4}   (1.0 would be white noise)");

        if let Some(path) = &self.output {
            let bytes: Vec<u8> = out.iter().flat_map(|s| s.to_le_bytes()).collect();
            std::fs::write(path, bytes).expect("write output");
            println!(
                "\nwrote {path} ({} samples @ {SAMPLE_RATE} Hz f32le)",
                out.len()
            );
        }
    }
}

fn main() {
    let (backend, args) = common::parse_args();
    if args.len() < 2 {
        eprintln!(
            "usage: vocode [--backend ndarray|cuda|tch] <bigvgan_generator.pt> \
             <in.f32le@22050> [out.f32le@22050]"
        );
        std::process::exit(2);
    }
    println!("backend : {backend}");
    common::run_on(
        backend,
        Vocode {
            weights: args[0].clone(),
            input: args[1].clone(),
            output: args.get(2).cloned(),
        },
    );
}
