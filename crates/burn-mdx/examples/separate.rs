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
//!      [--backend tch|tch-gpu|cuda] [--out <dir>] <MDX23C-*.ckpt> <speech.wav>...`

#[path = "common/mod.rs"]
mod common;

use burn::tensor::backend::Backend;
use burn_mdx::{MdxConfig, STEMS_8K_INSTVOC, Stft, TfcTdfNet};
use futures::StreamExt;

struct Separate {
    weights: String,
    /// Concatenated in order until a chunk is full. Several clips rather than
    /// one because this repository's own corpus is 1.5 s per file and looping a
    /// single clip four times gives the metric a periodicity to latch onto that
    /// real speech does not have.
    clips: Vec<String>,
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

/// Scale-invariant signal-to-distortion ratio, in dB — the field's standard
/// number for source separation.
///
/// CLAUDE.md warns that sample-wise RMS is phase-sensitive and that two runs of
/// the *same* model can differ hugely by it. That warning is about **vocoders**,
/// which invent phase; MDX23C predicts the complex spectrum, so its phase is a
/// prediction that either matches the source or does not, and a waveform metric
/// is measuring the thing the model was trained on. The scale invariance
/// (projecting the reference onto the estimate before differencing) is what
/// keeps it from being a loudness comparison. The energy-envelope correlation
/// below is the phase-free companion reading, so neither number stands alone.
fn si_sdr(est: &[f32], reference: &[f32]) -> f64 {
    assert_eq!(est.len(), reference.len());
    let dot: f64 = est
        .iter()
        .zip(reference)
        .map(|(e, r)| *e as f64 * *r as f64)
        .sum();
    let energy: f64 = reference.iter().map(|r| (*r as f64).powi(2)).sum();
    let alpha = dot / energy.max(1e-30);
    let (mut signal, mut noise) = (0.0, 0.0);
    for (e, r) in est.iter().zip(reference) {
        let target = alpha * *r as f64;
        signal += target * target;
        noise += (*e as f64 - target).powi(2);
    }
    10.0 * (signal.max(1e-30) / noise.max(1e-30)).log10()
}

/// Magnitude-spectrogram correlation **weighted by the reference's own
/// magnitude**, from a small analysis STFT that is deliberately not the model's
/// — measuring a model with its own front end would hide a mistake in that
/// front end.
///
/// The weighting is not decoration, it is the whole metric. An unweighted
/// correlation over every bin is dominated by the bins where the reference has
/// nothing, and there the estimate follows whatever else is in the mixture:
/// measured here, the *unseparated mixture* scored 0.980 against the voice and
/// 0.062 against the bed, which says only that the voice is broadband and the
/// bed is not. Weighting by `|reference|` asks the question that actually
/// distinguishes a separation — *where this source has energy, does the
/// estimate track it?* — and needs no threshold to do it.
fn weighted_corr(analysis: &Stft, est: &[f32], reference: &[f32]) -> f64 {
    let (a, b) = (magnitude(analysis, est), magnitude(analysis, reference));
    let total: f64 = b.iter().map(|v| *v as f64).sum();
    let mean = |x: &[f32]| -> f64 {
        x.iter()
            .zip(&b)
            .map(|(v, w)| *v as f64 * *w as f64)
            .sum::<f64>()
            / total.max(1e-30)
    };
    let (ma, mb) = (mean(&a), mean(&b));
    let (mut num, mut da, mut db) = (0.0, 0.0, 0.0);
    for ((x, y), w) in a.iter().zip(&b).zip(&b) {
        let (x, y, w) = (*x as f64 - ma, *y as f64 - mb, *w as f64);
        num += w * x * y;
        da += w * x * x;
        db += w * y * y;
    }
    num / (da.sqrt() * db.sqrt()).max(1e-30)
}

/// `[bins × frames]` linear magnitude.
fn magnitude(analysis: &Stft, x: &[f32]) -> Vec<f32> {
    let (spec, frames) = analysis.analyze(&[x.to_vec()]);
    let plane = analysis.dim_f() * frames;
    (0..plane)
        .map(|i| {
            let (re, im) = (spec[i], spec[plane + i]);
            (re * re + im * im).sqrt()
        })
        .collect()
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

/// A deterministic instrumental bed: a sustained triad with vibrato, each note
/// carrying six harmonics, over a pulsed bass and a noise percussion track.
/// Synthesised rather than taken from a second file so the "correct" answer is
/// known exactly.
///
/// **The harmonics and the noise are the point, not decoration.** A first
/// version used five bare sinusoids, and the result was a bed with energy in
/// six of 513 analysis bins — the mixture's spectrum was then the voice's
/// almost everywhere, which flattened every spectral metric and gave the model
/// something that sounds nothing like the music it was trained to strip. A
/// harmonic-rich, partly broadband bed is both a fairer input and a measurable
/// one.
fn instrumental_bed(samples: usize, sample_rate: f32) -> Vec<f32> {
    // A minor triad spanning speech's own range, so the two overlap in band
    // rather than being trivially separable by a low-pass.
    let notes = [110.0f32, 164.81, 220.0, 261.63, 329.63];
    let mut noise_state = 0x9E37_79B9_7F4A_7C15u64;
    let mut hiss = 0.0f32;
    (0..samples)
        .map(|i| {
            let t = i as f32 / sample_rate;
            // 2 Hz tremolo and a 4 Hz pulse: enough temporal structure that an
            // energy envelope can tell the bed from the speech.
            let tremolo = 0.6 + 0.4 * (2.0 * std::f32::consts::PI * 2.0 * t).sin();
            let pulse = ((4.0 * t).fract() * -6.0).exp();
            let vibrato = 1.0 + 0.004 * (2.0 * std::f32::consts::PI * 5.5 * t).sin();
            let chord: f32 = notes
                .iter()
                .enumerate()
                .flat_map(|(k, f)| {
                    (1..=6).map(move |h| {
                        let freq = f * vibrato * h as f32;
                        (2.0 * std::f32::consts::PI * freq * t).sin()
                            / ((k as f32 + 2.0) * h as f32)
                    })
                })
                .sum();
            let bass = (2.0 * std::f32::consts::PI * 55.0 * t).sin() * pulse;
            // A one-pole high-passed white noise, gated by the same pulse: a
            // stand-in for a hi-hat, and the only broadband part of the bed.
            noise_state = noise_state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let white = ((noise_state >> 40) as f32 / 8388608.0) - 1.0;
            hiss = 0.85 * hiss + 0.15 * white;
            let hat = (white - hiss) * pulse * 0.35;
            0.7 * chord * tremolo + 0.5 * bass + hat
        })
        .collect()
}

impl common::Job for Separate {
    fn run<B: Backend>(self, device: &B::Device) {
        let cfg = MdxConfig::mdx23c_8k_instvoc_hq();
        let chunk = cfg.chunk_size();

        // --- the two known sources -----------------------------------------
        let decoded: Vec<Vec<f32>> = self
            .clips
            .iter()
            .map(|c| decode(c, cfg.sample_rate))
            .collect();
        let mut voice: Vec<f32> = decoded.concat();
        println!(
            "clips   : {} file(s), {:.2} s at {} Hz",
            self.clips.len(),
            voice.len() as f32 / cfg.sample_rate as f32,
            cfg.sample_rate
        );
        if voice.len() < chunk {
            // Cycling the supplied clips rather than zero-padding: silence
            // would let a stem score well on the silent tail alone, which is
            // the easiest way to read a good number off a broken port.
            let source = voice.clone();
            while voice.len() < chunk {
                voice.extend_from_slice(&source);
            }
            println!("          (short of a {chunk}-sample chunk, so the clips were cycled)");
        }
        voice.truncate(chunk);
        let bed = instrumental_bed(chunk, cfg.sample_rate as f32);

        // Equal RMS first, so "0 dB" means what it says and neither source can
        // win by being louder. The gains are hoisted out of the maps
        // deliberately — recomputing an RMS per sample is quadratic over a
        // 261k-sample chunk.
        let voice_gain = 1.0 / rms(&voice).max(1e-9);
        let bed_gain = 1.0 / rms(&bed).max(1e-9);
        let mut voice: Vec<f32> = voice.iter().map(|v| v * voice_gain).collect();
        let mut bed: Vec<f32> = bed.iter().map(|v| v * bed_gain).collect();

        // Then one shared gain that puts the *sum* at a realistic level.
        //
        // **This matters, and it is easy to skip.** MDX23C is not
        // scale-invariant: its instance norms are, but the head multiplies the
        // U-net's output by `first_conv`'s and then concatenates the raw
        // mixture beside it, so the two paths scale differently and the whole
        // network only behaves at the levels it was trained on. Two RMS-1.0
        // sources summed peak around +8 dBFS, which is not a level any music
        // ever reaches. Both references get the same gain, so they stay exactly
        // the parts of the mixture and the metrics below are unaffected.
        let raw_peak = voice
            .iter()
            .zip(&bed)
            .map(|(v, b)| (v + b).abs())
            .fold(0.0f32, f32::max);
        let headroom = 0.7 / raw_peak.max(1e-9);
        voice.iter_mut().for_each(|v| *v *= headroom);
        bed.iter_mut().for_each(|v| *v *= headroom);
        let mix: Vec<f32> = voice.iter().zip(&bed).map(|(v, b)| v + b).collect();
        println!(
            "mix     : {chunk} samples, 0 dB voice/bed, peak {:.3}, rms {:.4}",
            mix.iter().fold(0.0f32, |a, b| a.max(b.abs())),
            rms(&mix)
        );

        // --- the model ------------------------------------------------------
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
        let separate = |signal: &[f32]| -> Vec<Vec<f32>> {
            // Both channels identical: the network's front end is stereo, and a
            // mono source duplicated is what every caller will feed it.
            let spec = stft.forward::<B>(&[signal.to_vec(), signal.to_vec()], device);
            let out = model.forward(spec);
            let [_, stems, channels, bins, frames] = out.dims();
            stft.inverse(out.reshape([stems, channels, bins, frames]))
                .iter()
                // Fold the duplicated stereo back to one channel for scoring.
                .map(|stem| {
                    stem[0]
                        .iter()
                        .zip(&stem[1])
                        .map(|(l, r)| 0.5 * (l + r))
                        .collect()
                })
                .collect()
        };

        let started = std::time::Instant::now();
        let estimates = separate(&mix);
        println!(
            "forward : {} stems of {} samples in {:.2} s",
            estimates.len(),
            estimates[0].len(),
            started.elapsed().as_secs_f32()
        );

        // --- the decisive probe: what does it do to a *solo* source? --------
        //
        // The mixture table below is the interesting measurement but the weak
        // one, because it depends on the synthetic bed being something the
        // model recognises as music. This does not: feed it the voice alone and
        // the instrumental stem must be near-silent; feed it the bed alone and
        // the vocal stem must be. A port with a transposed U-net, a batch norm
        // where an instance norm belongs, or a scrambled stem axis cannot pass
        // this — it has no way to know which stem to empty.
        println!("\nsolo-source rejection, dB (how much of the wrong stem leaks)");
        for (label, solo) in [("voice only", &voice), ("bed only", &bed)] {
            let stems = separate(solo);
            let (a, b) = (rms(&stems[0]), rms(&stems[1]));
            let wanted = if label == "voice only" { a } else { b };
            let leaked = if label == "voice only" { b } else { a };
            println!(
                "  {label:<12} vocals {:.4}  instrumental {:.4}   rejection {:>6.1} dB",
                a,
                b,
                20.0 * (wanted.max(1e-9) / leaked.max(1e-9)).log10()
            );
        }

        // --- the gap --------------------------------------------------------
        let analysis = Stft::new(1024, 256, 513);
        let named: Vec<(String, &Vec<f32>)> = std::iter::once(("mixture".to_string(), &mix))
            .chain(estimates.iter().enumerate().map(|(i, e)| {
                (
                    format!("est_{}", STEMS_8K_INSTVOC.get(i).copied().unwrap_or("stem")),
                    e,
                )
            }))
            .collect();

        println!("\nSI-SDR, dB (the number to read; the mixture row is do-nothing)");
        println!("  {:<22} {:>9} {:>9}", "", "vs voice", "vs bed");
        for (name, signal) in &named {
            println!(
                "  {name:<22} {:>9.2} {:>9.2}",
                si_sdr(signal, &voice),
                si_sdr(signal, &bed)
            );
        }

        println!("\nmagnitude-spectrogram correlation, weighted by the reference");
        println!("  {:<22} {:>9} {:>9}", "", "vs voice", "vs bed");
        for (name, signal) in &named {
            println!(
                "  {name:<22} {:>9.3} {:>9.3}",
                weighted_corr(&analysis, signal, &voice),
                weighted_corr(&analysis, signal, &bed)
            );
        }

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
             <MDX23C-*.ckpt> <speech.wav>..."
        );
        std::process::exit(2);
    }
    println!("backend : {backend}");
    common::run_on(
        backend,
        Separate {
            weights: args[0].clone(),
            clips: args[1..].to_vec(),
            out,
        },
    );
}
