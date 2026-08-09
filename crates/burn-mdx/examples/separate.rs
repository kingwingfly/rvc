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
//! # The baseline is the mixture, not chance
//!
//! `burn-gptsovits`'s `reconstruct` correlates its output against its input,
//! which is right for a round trip and **wrong here**: a network that returned
//! its input unchanged would score beautifully. For separation the meaningful
//! do-nothing is the *unprocessed mixture*, so it is printed as a row of every
//! table and the number that means anything is the delta from it. Chance is the
//! wrong reference; a chance-level result and an identity result are miles
//! apart and only one of them is a bug.
//!
//! Three readings, because no single one of them is sufficient:
//!
//! 1. **Partition** — do the stems sum back to the mixture? Independent of how
//!    well the model separates, and the check that actually proves the port.
//! 2. **Solo rejection** — feed one source alone; the other stem must go quiet.
//!    Independent of the mixture being realistic.
//! 3. **SI-SDR and correlations against both sources** — the quality question,
//!    and the only one that depends on the bed being convincing.
//!
//! SI-SDR leads the third because it is the field's standard measure and, unlike
//! a correlation, is not saturated near 1.0 — a correlation has no room left to
//! show an improvement when the mixture already scores 0.9. CLAUDE.md's warning
//! that sample-wise metrics are phase-sensitive is about *vocoders*, which
//! invent phase; this model predicts the complex spectrum. The phase-free
//! energy-envelope correlation is reported beside it so neither stands alone.
//!
//! # What this actually established, and what it did not
//!
//! Measured on `MDX23C-8KFFT-InstVoc_HQ.ckpt`, LibTorch/CUDA, six clips of this
//! repository's own corpus against the bed below at a 0 dB mix:
//!
//! | reading | value |
//! |---|---|
//! | **partition** — `est_vocals + est_instrumental` vs the mixture | **41.5 dB** SI-SDR |
//! | solo voice in — vocals / instrumental rms | 0.0516 / 0.0299 (**4.7 dB** rejection) |
//! | solo bed in — vocals / instrumental rms | 0.0226 / 0.0526 (**7.3 dB** rejection) |
//! | SI-SDR of `est_vocals` vs voice / vs bed | −0.25 / −0.54 dB |
//! | envelope corr, `est_vocals` vs voice / vs bed | 0.678 / 0.645 |
//! | envelope corr, `est_instrumental` vs voice / vs bed | 0.508 / 0.583 |
//!
//! **Read the first row first.** The two stems reconstruct their input to
//! 41.5 dB, and the solo probes split *in opposite directions* depending on
//! which source went in. Together those say the forward pass is coherent and
//! content-dependent: a transposed U-net, a batch norm where an instance norm
//! belongs, a mis-scaled block or a scrambled stem axis destroys one or both,
//! because nothing downstream re-imposes either property.
//!
//! **What it does not establish is separation quality**, and the numbers are
//! honest about that: 4.7 dB of rejection on a solo source is far below the
//! 20 dB-plus a vocal separator manages on the material it was trained for, and
//! every mixture row sits within a decibel of doing nothing. The most likely
//! reason is the input rather than the port — MDX23C was trained on *sung*
//! vocals inside real productions, and this feeds it dry, close-mic Chinese
//! speech over a synthesised organ chord, which is out of distribution on both
//! sides. **Closing that gap needs a real music mixture, which this repository
//! does not contain**, so it is stated as an open question rather than
//! explained away. The unit tests are what pin the two components this example
//! cannot isolate: `net::tests` compares the norm against Burn's own
//! `InstanceNorm` and GELU against hand-computed erf values.
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
    /// `--mixture <file>`: a **real** recording whose sources are unknown, which
    /// switches every reading below to a reference-free one. Empty in the
    /// synthetic mode, and the two are mutually exclusive — a mixture has no
    /// known voice to score against, so there is nothing for the clips to be.
    mixture: Option<String>,
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
        if let Some(path) = &self.mixture {
            real_mixture::<B>(&cfg, &self.weights, path, self.out.as_deref(), device);
            return;
        }
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
        assert!(
            res.missing.is_empty() && !res.applied.is_empty(),
            "bad load"
        );

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

        // --- the partition check --------------------------------------------
        //
        // The one number here that does not depend on the model being *good*.
        // MDX23C's two stems are trained to partition their input, so a
        // coherent forward pass must satisfy
        // `est_vocals + est_instrumental ≈ mixture` no matter how well it
        // actually separates — and a port with a transposed U-net or a
        // mis-scaled block cannot produce that by accident, because nothing
        // downstream re-imposes it. This is what separates "the port is wrong"
        // from "this input is nothing like the music the weights were trained
        // on", and those two hypotheses are otherwise indistinguishable from
        // the table below.
        let partition: Vec<f32> = estimates[0]
            .iter()
            .zip(&estimates[1])
            .map(|(a, b)| a + b)
            .collect();
        println!(
            "\npartition: est_vocals + est_instrumental vs the mixture, SI-SDR {:.2} dB",
            si_sdr(&partition, &mix)
        );

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

/// Separate a **real** recording, whose sources are unknown, and report only
/// what can be measured without them.
///
/// No SI-SDR against a source appears here and none can: nobody holds the stems
/// of somebody else's stream. Every number below is either a property of the
/// output pair (do they partition, do they differ) or a *contrast* the mixture
/// is measured under the same way, so the mixture row stays the do-nothing
/// baseline the synthetic mode established.
fn real_mixture<B: Backend>(
    cfg: &MdxConfig,
    weights: &str,
    path: &str,
    out: Option<&str>,
    device: &B::Device,
) {
    let sr = cfg.sample_rate;
    let (left, right) = decode_stereo(path, sr);
    let frames = left.len();
    assert!(
        frames > cfg.chunk_size(),
        "a mixture shorter than one chunk"
    );

    // Mid/side, because how much of either there is decides what the model has
    // to work with. MDX23C is stereo-native — its front end takes two channels
    // and a real production gives it a wide bed against a centred vocal — so a
    // near-mono input removes the spatial cue and leaves it separating
    // spectrally. That is a property of the recording, and it is printed rather
    // than assumed so a poor result is not misread as a broken port.
    let mid: Vec<f32> = left
        .iter()
        .zip(&right)
        .map(|(l, r)| 0.5 * (l + r))
        .collect();
    let side: Vec<f32> = left
        .iter()
        .zip(&right)
        .map(|(l, r)| 0.5 * (l - r))
        .collect();
    println!(
        "mixture : {path}\n          {:.1} s at {sr} Hz, mid rms {:.5}, side rms {:.5} \
         ({:.1} dB below mid)",
        frames as f32 / sr as f32,
        rms(&mid),
        rms(&side),
        20.0 * (rms(&mid).max(1e-12) / rms(&side).max(1e-12)).log10(),
    );

    let mut model = TfcTdfNet::<B>::new(cfg, device);
    let res = model.load_pytorch(weights).expect("load checkpoint");
    println!(
        "weights : {} applied / {} missing / {} unused",
        res.applied.len(),
        res.missing.len(),
        res.unused.len()
    );
    assert!(
        res.missing.is_empty() && !res.applied.is_empty(),
        "bad load"
    );

    // --- windowed overlap-add ------------------------------------------------
    //
    // The network's unit is one chunk and nothing in the crate is aware of a
    // longer recording, so the caller owns the seams. A periodic Hann at 50%
    // hop is COLA, but the accumulated window is divided out explicitly rather
    // than relied on: that makes the first and last half-chunk — covered by one
    // pass instead of two — an *exact* reconstruction rather than one faded to
    // zero, which is the difference between an edge that is merely less
    // averaged and an edge that would drag every metric down.
    //
    // Upstream runs `inference.num_overlap: 8`. Two is what a 20-minute stream
    // can afford here at ~3 s a forward, and the seam it leaves is measurable:
    // the partition row below is computed over the whole interior and would
    // show it.
    let chunk = cfg.chunk_size();
    let hop = chunk / 2;
    let window: Vec<f32> = (0..chunk)
        .map(|i| 0.5 - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / chunk as f32).cos())
        .collect();
    let stft = cfg.stft();
    let stems = STEMS_8K_INSTVOC.len();
    let mut acc: Vec<[Vec<f32>; 2]> = (0..stems)
        .map(|_| [vec![0.0f32; frames], vec![0.0f32; frames]])
        .collect();
    let mut norm = vec![0.0f32; frames];

    let started = std::time::Instant::now();
    let mut passes = 0usize;
    let mut start = 0usize;
    loop {
        let end = (start + chunk).min(frames);
        let pad = |src: &[f32]| {
            let mut v = vec![0.0f32; chunk];
            v[..end - start].copy_from_slice(&src[start..end]);
            v
        };
        let spec = stft.forward::<B>(&[pad(&left), pad(&right)], device);
        let output = model.forward(spec);
        let [_, n_stems, n_channels, bins, n_frames] = output.dims();
        let waves = stft.inverse(output.reshape([n_stems, n_channels, bins, n_frames]));
        for (si, stem) in waves.iter().enumerate() {
            for ch in 0..2 {
                for i in 0..end - start {
                    acc[si][ch][start + i] += window[i] * stem[ch][i];
                }
            }
        }
        for i in 0..end - start {
            norm[start + i] += window[i];
        }
        passes += 1;
        if start + chunk >= frames {
            break;
        }
        start += hop;
    }
    for stem in &mut acc {
        for ch in stem {
            for (v, w) in ch.iter_mut().zip(&norm) {
                *v /= w.max(1e-6);
            }
        }
    }
    println!(
        "forward : {passes} chunks at 50% overlap in {:.1} s ({:.2} s per chunk)",
        started.elapsed().as_secs_f32(),
        started.elapsed().as_secs_f32() / passes as f32,
    );
    for stem in &acc {
        assert!(
            stem[0].iter().chain(&stem[1]).all(|x| x.is_finite()),
            "a stem is not finite"
        );
    }

    // Everything below is scored on the mono downmix of the **interior**, one
    // hop in from each end: those two regions are single-pass, and while the
    // window division makes them exact they are not the same measurement as the
    // rest, so they are not averaged into it.
    let trim = hop;
    let interior = |x: &[f32]| x[trim..frames - trim].to_vec();
    let mono = |stem: &[Vec<f32>; 2]| -> Vec<f32> {
        stem[0]
            .iter()
            .zip(&stem[1])
            .map(|(l, r)| 0.5 * (l + r))
            .collect()
    };
    let mix_i = interior(&mid);
    let stem_i: Vec<Vec<f32>> = acc.iter().map(|s| interior(&mono(s))).collect();
    let sum: Vec<f32> = stem_i[0]
        .iter()
        .zip(&stem_i[1])
        .map(|(a, b)| a + b)
        .collect();
    println!(
        "\npartition: est_vocals + est_instrumental vs the mixture, SI-SDR {:.2} dB \
         (over the interior, {:.1} s)",
        si_sdr(&sum, &mix_i),
        mix_i.len() as f32 / sr as f32,
    );

    println!(
        "\nhow the energy was split (mixture rms {:.5})",
        rms(&mix_i)
    );
    for (i, est) in stem_i.iter().enumerate() {
        println!(
            "  est_{:<18} rms {:.5}   {:>6.1} dB relative to the mixture",
            STEMS_8K_INSTVOC[i],
            rms(est),
            20.0 * (rms(est).max(1e-12) / rms(&mix_i).max(1e-12)).log10(),
        );
    }
    // Partition means the stems trade energy, so a *negative* correlation is
    // what a working separation gives and ≈ +1 would say both stems are half
    // the mixture — the shape a model that separated nothing produces.
    println!(
        "  {:<22} {:>6.3} sample-wise, {:>6.3} on the 32 ms envelope",
        "vocals vs instrumental",
        corr(&stem_i[0], &stem_i[1]),
        corr(
            &envelope(&stem_i[0], sr as usize / 32),
            &envelope(&stem_i[1], sr as usize / 32)
        ),
    );

    // --- the reading that says whether it separated *this* material ----------
    //
    // With no stems, the discriminating question is what happens in the frames
    // where the mixture is quietest. A continuous music bed under intermittent
    // speech puts the mixture's minima at the speech gaps, so a vocals stem
    // that found the speech must be *quieter there than the mixture is*, and an
    // instrumental stem that found the bed must be *flatter than the mixture
    // is*. Both are contrasts within one signal, so neither needs a reference
    // and neither is fooled by a stem that is simply scaled down.
    //
    // **250 ms, not one second, and the difference is not cosmetic.** The gap
    // between two sentences is a few hundred milliseconds, so at a one-second
    // window every "quiet" frame still contains speech and the reading
    // collapses toward the mixture's own dynamics. Measured both ways on the
    // same 60 s of the same file: at one second the vocals stem sat **2.1 dB**
    // under the mixture in the quiet frames and its contrast came out *below*
    // the mixture's (11.5 against 12.0 dB), which reads as a model that
    // separated nothing; at 250 ms the same run gives **6.5 dB** and a contrast
    // *above* the mixture's (24.4 against 19.9). Same audio, same stems — the
    // window was measuring the speech's duty cycle rather than the gaps.
    let win = sr as usize / 4;
    let per_frame: Vec<f32> = mix_i.chunks(win).map(rms).collect();
    let mut order: Vec<usize> = (0..per_frame.len()).collect();
    order.sort_by(|a, b| per_frame[*a].total_cmp(&per_frame[*b]));
    let decile = (order.len() / 10).max(1);
    let quiet = &order[..decile];
    let loud = &order[order.len() - decile..];
    let over = |signal: &[f32], frames: &[usize]| -> f32 {
        let mut gathered = Vec::new();
        for f in frames {
            let (a, b) = (f * win, ((f + 1) * win).min(signal.len()));
            gathered.extend_from_slice(&signal[a..b]);
        }
        rms(&gathered)
    };
    println!(
        "\nspeech-gap contrast: the {decile} quietest and {decile} loudest of \
         {} frames of 250 ms,\nchosen by the *mixture* — so the quiet ones are \
         speech gaps and the bed runs through both",
        per_frame.len()
    );
    println!(
        "  {:<22} {:>10} {:>10} {:>10}",
        "", "quiet rms", "loud rms", "contrast"
    );
    let mix_quiet = over(&mix_i, quiet);
    for (name, signal) in std::iter::once(("mixture".to_string(), &mix_i)).chain(
        stem_i
            .iter()
            .enumerate()
            .map(|(i, e)| (format!("est_{}", STEMS_8K_INSTVOC[i]), e)),
    ) {
        let (q, l) = (over(signal, quiet), over(signal, loud));
        println!(
            "  {name:<22} {q:>10.5} {l:>10.5} {:>7.1} dB",
            20.0 * (l.max(1e-12) / q.max(1e-12)).log10()
        );
    }
    // The one number a caller cleaning a corpus is actually buying. In a speech
    // gap the mixture is the bed and nothing else, so whatever the vocals stem
    // still holds there *is* bed — and how far below the mixture it sits is how
    // much of the music the stem got rid of.
    println!(
        "  → in the speech gaps, est_vocals sits {:.1} dB under the mixture: \
         that is how\n    much of the bed came out of it",
        20.0 * (over(&stem_i[0], quiet).max(1e-12) / mix_quiet.max(1e-12)).log10()
    );

    if let Some(dir) = out {
        std::fs::create_dir_all(dir).expect("create output directory");
        write_wav_stereo(&format!("{dir}/mixture.wav"), sr, &left, &right);
        for (i, stem) in acc.iter().enumerate() {
            let name = STEMS_8K_INSTVOC[i];
            write_wav_stereo(&format!("{dir}/{name}.wav"), sr, &stem[0], &stem[1]);
        }
        println!("\nwrote the mixture and {stems} stems to {dir} — listen to them");
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

/// Decode a container to **two** channels at `sample_rate`.
///
/// Not the mono path folded twice, which is what the synthetic mode feeds the
/// network: a real mixture's channel difference is one of the two cues the
/// model has, so throwing it away before the forward pass would make a
/// near-mono result something this example imposed rather than something it
/// measured.
fn decode_stereo(path: &str, sample_rate: u32) -> (Vec<f32>, Vec<f32>) {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("tokio runtime");
    runtime.block_on(async {
        let mut stream = Box::pin(audio_kit::decode_path_stereo(
            path,
            audio_kit::DecodeOptions::new(sample_rate),
        ));
        let (mut left, mut right) = (Vec::new(), Vec::new());
        while let Some(chunk) = stream.next().await {
            let chunk = chunk.expect("decode");
            left.extend_from_slice(&chunk.left);
            right.extend_from_slice(&chunk.right);
        }
        (left, right)
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

fn write_wav_stereo(path: &str, sample_rate: u32, left: &[f32], right: &[f32]) {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("tokio runtime");
    let samples = audio_kit::StereoSamples {
        left: left.to_vec(),
        right: right.to_vec(),
    };
    runtime.block_on(async move {
        let stream = futures::stream::once(async move { Ok(samples) });
        audio_kit::write_wav_stereo_file(path, sample_rate, Box::pin(stream))
            .await
            .expect("write wav");
    });
}

/// Pull `--<flag> <value>` out of the arguments.
fn take_value(args: &mut Vec<String>, flag: &str) -> Option<String> {
    let i = args.iter().position(|a| a == flag)?;
    args.remove(i);
    if i >= args.len() {
        common::fail(&format!("{flag} needs a value"));
    }
    Some(args.remove(i))
}

fn main() {
    let (backend, mut args) = common::parse_args();
    let out = take_value(&mut args, "--out");
    let mixture = take_value(&mut args, "--mixture");
    // A mixture has no known voice, so there is nothing a clip could be scored
    // against and passing both is a request for two different measurements.
    let malformed = match &mixture {
        Some(_) => args.len() != 1,
        None => args.len() < 2,
    };
    if malformed {
        eprintln!(
            "usage: separate [--backend tch|tch-gpu|cuda] [--out <dir>] \\\n         \
             <MDX23C-*.ckpt> <speech.wav>...            # synthetic, with references\n   \
             or: separate [--backend ...] [--out <dir>] --mixture <song.wav> \\\n         \
             <MDX23C-*.ckpt>                            # real, reference-free"
        );
        std::process::exit(2);
    }
    println!("backend : {backend}");
    common::run_on(
        backend,
        Separate {
            weights: args[0].clone(),
            clips: args[1..].to_vec(),
            mixture,
            out,
        },
    );
}
