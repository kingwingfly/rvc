//! Mix a known voice with a known instrumental bed, separate, and report the
//! **gap** between how well each stem tracks what it should and what it should
//! not.
//!
//! This is the check weight coverage cannot make. `examples/load` reports
//! 319/0/0, and that says the module tree matches the checkpoint — it says
//! nothing about the U-net running with time as the image height, or about the
//! norms being instance norms rather than batch norms in evaluation mode. Both
//! would load perfectly and separate nothing.
//!
//! # Why a gap and not a correlation
//!
//! `burn-gptsovits`'s `reconstruct` correlates its output against its input,
//! which is right for a round trip and **wrong here**: a network that returned
//! its input unchanged would score beautifully. So each estimated stem is
//! scored against *both* sources, and the claim is the ordering:
//!
//! ```text
//! corr(est_vocals,       voice) >> corr(est_vocals,       bed)
//! corr(est_instrumental, bed)   >> corr(est_instrumental, voice)
//! ```
//!
//! # Why log-spectra and not RMS
//!
//! CLAUDE.md records the lesson at length: sample-wise RMS on a waveform is
//! phase-sensitive, and two runs of the *same* model can differ hugely by it.
//! The primary number here is the correlation of **log-magnitude
//! spectrograms**, computed by a second, small STFT that has nothing to do with
//! the model's front end. The energy envelope is reported beside it as the
//! coarser reading, and a frame-shuffled log-spectrum gives the chance baseline
//! — without one, "0.6" means nothing.
//!
//! # Expected, on `MDX23C-8KFFT-InstVoc_HQ.ckpt` at 0 dB mix
//!
//! | | vs voice | vs bed |
//! |---|---|---|
//! | `est_vocals` | high | low |
//! | `est_instrumental` | low | high |
//!
//! A working port puts each stem's own source **well above** both the other
//! source and the shuffled baseline; the mixture's own correlation against each
//! source is printed as the do-nothing reference, and a stem that does not beat
//! it has not separated anything. The measured numbers are in this crate's
//! commit message and in the PR — they are hardware- and clip-dependent enough
//! that pinning an exact figure in a doc comment would rot.
//!
//! # Cost
//!
//! One chunk: 261,120 samples, 5.92 s, one forward pass over `[1, 16, 1024,
//! 256]` — around 1 TFLOP and 134 MB per activation at the widest. That is why
//! this example carries `required-features = ["tch"]` and why `--backend
//! tch-gpu` is the sensible way to run it.
//!
//! Usage:
//! `cargo run -p burn-mdx --example separate --features tch -- \
//!      [--backend tch|tch-gpu|cuda] [--out <dir>] <MDX23C-*.ckpt> <speech.wav>`

#[path = "common/mod.rs"]
mod common;

use burn::tensor::backend::Backend;
use burn_mdx::{MdxConfig, STEMS_8K_INSTVOC, Stft, TfcTdfNet};
use futures::StreamExt;

struct Separate {
    weights: String,
    clip: String,
    out: Option<String>,
}

/// Pearson correlation. `f64` throughout: a log-spectrum of a 6 s clip is
/// ~100k terms and the naive `f32` accumulation of their squares loses the
/// third digit, which is the digit this example is read for.
fn corr(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len());
    let n = a.len() as f64;
    let (ma, mb) = (
        a.iter().map(|x| *x as f64).sum::<f64>() / n,
        b.iter().map(|x| *x as f64).sum::<f64>() / n,
    );
    let (mut num, mut da, mut db) = (0.0, 0.0, 0.0);
    for (x, y) in a.iter().zip(b) {
        let (x, y) = (*x as f64 - ma, *y as f64 - mb);
        num += x * y;
        da += x * x;
        db += y * y;
    }
    num / (da.sqrt() * db.sqrt()).max(1e-12)
}

/// `[bins × frames]` log-magnitude, from a small analysis STFT that is
/// deliberately *not* the model's — measuring a model with its own front end
/// would hide a mistake in that front end.
fn log_spectrum(analysis: &Stft, x: &[f32]) -> Vec<f32> {
    let (spec, frames) = analysis.analyze(&[x.to_vec()]);
    let plane = analysis.dim_f() * frames;
    (0..plane)
        .map(|i| {
            let (re, im) = (spec[i], spec[plane + i]);
            (re * re + im * im).sqrt().max(1e-8).ln()
        })
        .collect()
}

/// Frame-shuffled log-spectrum: the same values, in an order that carries no
/// information about the signal. What "uncorrelated" actually scores.
fn shuffled(spectrum: &[f32], bins: usize, frames: usize) -> Vec<f32> {
    // A fixed odd stride is a permutation of the frames whenever it is coprime
    // with the count, and needs no RNG to be reproducible.
    let stride = 37;
    let mut out = vec![0.0f32; spectrum.len()];
    for bin in 0..bins {
        for frame in 0..frames {
            out[bin * frames + frame] = spectrum[bin * frames + (frame * stride) % frames];
        }
    }
    out
}

/// Short-term energy, the coarser reading `reconstruct` uses.
fn envelope(x: &[f32], window: usize) -> Vec<f32> {
    x.chunks(window)
        .map(|c| (c.iter().map(|v| v * v).sum::<f32>() / c.len() as f32).sqrt())
        .collect()
}

fn rms(x: &[f32]) -> f32 {
    (x.iter().map(|v| v * v).sum::<f32>() / x.len().max(1) as f32).sqrt()
}

/// A deterministic instrumental bed: a sustained triad with vibrato over a
/// pulsed bass. Synthesised rather than taken from a second file so the
/// "correct" answer is known exactly and the example needs one input.
fn instrumental_bed(samples: usize, sample_rate: f32) -> Vec<f32> {
    // A minor triad well below and around speech, so the two overlap in band
    // rather than being trivially separable by a low-pass.
    let voices = [110.0f32, 164.81, 220.0, 261.63, 329.63];
    (0..samples)
        .map(|i| {
            let t = i as f32 / sample_rate;
            // 2 Hz tremolo and a 4 Hz bass pulse: enough temporal structure
            // that an energy envelope can tell the bed from the speech.
            let tremolo = 0.6 + 0.4 * (2.0 * std::f32::consts::PI * 2.0 * t).sin();
            let pulse = ((4.0 * t).fract() * -6.0).exp();
            let chord: f32 = voices
                .iter()
                .enumerate()
                .map(|(k, f)| {
                    let vibrato = 1.0 + 0.004 * (2.0 * std::f32::consts::PI * 5.5 * t).sin();
                    (2.0 * std::f32::consts::PI * f * vibrato * t).sin() / (k as f32 + 2.0)
                })
                .sum();
            let bass = (2.0 * std::f32::consts::PI * 55.0 * t).sin() * pulse;
            0.7 * chord * tremolo + 0.5 * bass
        })
        .collect()
}

impl common::Job for Separate {
    fn run<B: Backend>(self, device: &B::Device) {
        let cfg = MdxConfig::mdx23c_8k_instvoc_hq();
        let chunk = cfg.chunk_size();

        // --- the two known sources -----------------------------------------
        let mut voice = decode(&self.clip, cfg.sample_rate);
        println!(
            "clip    : {:.2} s at {} Hz",
            voice.len() as f32 / cfg.sample_rate as f32,
            cfg.sample_rate
        );
        if voice.len() < chunk {
            // Looping rather than zero-padding: silence would let a stem score
            // well on the silent tail alone.
            let source = voice.clone();
            while voice.len() < chunk {
                voice.extend_from_slice(&source);
            }
        }
        voice.truncate(chunk);
        let bed = instrumental_bed(chunk, cfg.sample_rate as f32);

        // Equal RMS, so "0 dB" means what it says and neither source can win by
        // being louder.
        let voice: Vec<f32> = voice.iter().map(|v| v / rms(&voice).max(1e-9)).collect();
        let bed: Vec<f32> = bed.iter().map(|v| v / rms(&bed).max(1e-9)).collect();
        let mix: Vec<f32> = voice
            .iter()
            .zip(&bed)
            .map(|(v, b)| 0.25 * (v + b))
            .collect();
        println!(
            "mix     : {chunk} samples, 0 dB voice/bed, peak {:.3}",
            mix.iter().fold(0.0f32, |a, b| a.max(b.abs()))
        );

        // --- one forward pass ----------------------------------------------
        let mut model = TfcTdfNet::<B>::new(&cfg, device);
        let res = model.load_pytorch(&self.weights).expect("load checkpoint");
        println!(
            "weights : {} applied / {} missing / {} unused",
            res.applied.len(),
            res.missing.len(),
            res.unused.len()
        );
        assert!(res.missing.is_empty() && !res.applied.is_empty(), "bad load");

        let stft = cfg.stft();
        // Both channels identical: the network's front end is stereo, and a
        // mono source duplicated is what every caller will feed it.
        let spec = stft.forward::<B>(&[mix.clone(), mix.clone()], device);
        println!("spectrum: {:?}", spec.dims());

        let started = std::time::Instant::now();
        let out = model.forward(spec);
        let [_, stems, channels, bins, frames] = out.dims();
        println!(
            "forward : {:?} in {:.1} s",
            out.dims(),
            started.elapsed().as_secs_f32()
        );

        let waves = stft.inverse(out.reshape([stems, channels, bins, frames]));
        // Fold the duplicated stereo back to one channel for scoring.
        let estimates: Vec<Vec<f32>> = waves
            .iter()
            .map(|stem| {
                stem[0]
                    .iter()
                    .zip(&stem[1])
                    .map(|(l, r)| 0.5 * (l + r))
                    .collect()
            })
            .collect();

        // --- the gap --------------------------------------------------------
        let analysis = Stft::new(1024, 256, 513);
        let frames_a = analysis.frames(chunk);
        let (ls_voice, ls_bed) = (
            log_spectrum(&analysis, &voice),
            log_spectrum(&analysis, &bed),
        );
        let chance = corr(
            &ls_voice,
            &shuffled(&ls_bed, analysis.dim_f(), frames_a),
        );

        println!("\nlog-spectrum correlation (the number to read)");
        println!("  {:<22} {:>8} {:>8}", "", "vs voice", "vs bed");
        let ls_mix = log_spectrum(&analysis, &mix);
        println!(
            "  {:<22} {:>8.3} {:>8.3}   <- do nothing",
            "mixture",
            corr(&ls_mix, &ls_voice),
            corr(&ls_mix, &ls_bed)
        );
        for (i, est) in estimates.iter().enumerate() {
            let ls = log_spectrum(&analysis, est);
            let name = STEMS_8K_INSTVOC.get(i).copied().unwrap_or("stem");
            println!(
                "  {:<22} {:>8.3} {:>8.3}",
                format!("est_{name}"),
                corr(&ls, &ls_voice),
                corr(&ls, &ls_bed)
            );
        }
        println!("  {:<22} {:>8.3}          <- chance (frame-shuffled)", "baseline", chance);

        println!("\nenergy envelope correlation (32 ms windows)");
        let window = cfg.sample_rate as usize / 32;
        let (ev_voice, ev_bed) = (envelope(&voice, window), envelope(&bed, window));
        for (i, est) in estimates.iter().enumerate() {
            let ev = envelope(est, window);
            let name = STEMS_8K_INSTVOC.get(i).copied().unwrap_or("stem");
            println!(
                "  {:<22} {:>8.3} {:>8.3}   rms {:.4}",
                format!("est_{name}"),
                corr(&ev, &ev_voice),
                corr(&ev, &ev_bed),
                rms(est)
            );
            assert!(
                est.iter().all(|x| x.is_finite()),
                "stem {name} is not finite"
            );
        }

        if let Some(dir) = &self.out {
            std::fs::create_dir_all(dir).expect("create output directory");
            write_wav(&format!("{dir}/mixture.wav"), cfg.sample_rate, &mix);
            for (i, est) in estimates.iter().enumerate() {
                let name = STEMS_8K_INSTVOC.get(i).copied().unwrap_or("stem");
                write_wav(&format!("{dir}/{name}.wav"), cfg.sample_rate, est);
            }
            println!("\nwrote mixture and {} stems to {dir}", estimates.len());
        }
    }
}

/// Decode any container to mono `f32` at `sample_rate`.
///
/// A blocking wrapper around `audio-kit`'s stream, because these examples have
/// no async main and one clip fits in memory by construction — the model's unit
/// of work is 5.92 s.
fn decode(path: &str, sample_rate: u32) -> Vec<f32> {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("tokio runtime");
    runtime.block_on(async {
        let mut stream = Box::pin(audio_kit::decode_path(
            path,
            audio_kit::DecodeOptions::new(sample_rate),
        ));
        let mut out = Vec::new();
        while let Some(chunk) = stream.next().await {
            out.extend_from_slice(&chunk.expect("decode"));
        }
        out
    })
}

fn write_wav(path: &str, sample_rate: u32, samples: &[f32]) {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("tokio runtime");
    let owned = samples.to_vec();
    runtime.block_on(async move {
        let stream = futures::stream::once(async move { Ok(owned) });
        audio_kit::write_wav_file(path, sample_rate, Box::pin(stream))
            .await
            .expect("write wav");
    });
}

fn main() {
    let (backend, mut args) = common::parse_args();
    let out = match args.iter().position(|a| a == "--out") {
        Some(i) => {
            args.remove(i);
            if i >= args.len() {
                common::fail("--out needs a directory");
            }
            Some(args.remove(i))
        }
        None => None,
    };
    if args.len() < 2 {
        eprintln!(
            "usage: separate [--backend tch|tch-gpu|cuda] [--out <dir>] \
             <MDX23C-*.ckpt> <speech.wav>"
        );
        std::process::exit(2);
    }
    println!("backend : {backend}");
    common::run_on(
        backend,
        Separate {
            weights: args[0].clone(),
            clip: args[1].clone(),
            out,
        },
    );
}
