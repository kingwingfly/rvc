//! Round-trip real audio through `s2` and see whether it comes back.
//!
//! The check weight coverage cannot make. Every component of `s2` loads at 0
//! missing, and that says the module tree matches the checkpoint — it says
//! nothing about the MRTE being wired the right way round, or the style
//! attention scaling by the right constant. Both would load perfectly and
//! produce noise.
//!
//! audio → cnhubert → quantise → `s2.decode` → audio, with the original
//! spectrogram as the speaker reference. This is not a fair test of *quality*
//! (the text is not the transcript, and the prior is sampled), but a port that
//! is wired wrongly returns noise, and speech is unmistakable against it.
//!
//! Usage: `cargo run -p burn-gptsovits --example reconstruct -- <hubert.bin> <s2G.pth> <audio.f32le@16k> <out.f32le@32k>`
//!
//! # The number, and the length it needs
//!
//! The output's **energy envelope** is correlated against the source's, with a
//! shuffled control beside it. Envelope rather than samples: the prior is
//! sampled and the phonemes are arbitrary, so a sample-wise difference is large
//! between two runs of the *same* model and proves nothing — but a mis-wired
//! MRTE or a mis-scaled style attention returns something whose loudness stops
//! tracking the source at all.
//!
//! **Read the frame count first.** On `s2G2333k.pth` and this repository's own
//! corpus, LibTorch:
//!
//! | clip | frames | r vs source | chance |
//! |---|---|---|---|
//! | 1.75 s | 86 | 0.4675 | -0.1196 |
//! | 10 s | 498 | **0.8562** | -0.0600 |
//!
//! Both are the same model on the same weights, and the short one is not a
//! worse port — 86 frames of one breathy phrase is too little for a correlation
//! to settle, exactly as `rvc-core`'s `f0_runtimes` is meaningless below a few
//! dozen jointly voiced frames. **Give it ten seconds or more**, and treat
//! anything under ~200 frames as no measurement at all.
//!
//! This correlation is what CLAUDE.md has long cited for `s2` — at r=0.91
//! against a 0.30 baseline — but **the example did not compute it**, printing
//! RMS and finiteness instead. The number was real and the harness credited
//! with it was not, which is the same shape of error as a fabricated
//! calibration table: it can only be caught by running the thing. It is
//! computed here now, so the claim and the check are the same object.

#[path = "common/mod.rs"]
mod common;

use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor, TensorData};
use burn_gptsovits::{Hubert, HubertConfig, SovitsConfig, SovitsPartial};
use burn_vits::{Spectral, SpectralConfig};

struct Reconstruct {
    hubert: String,
    sovits: String,
    input: String,
    output: String,
}

/// Loudness and peak, so two signals can be compared without listening.
fn describe(name: &str, v: &[f32]) {
    let n = v.len().max(1) as f64;
    let rms = (v.iter().map(|x| (*x as f64).powi(2)).sum::<f64>() / n).sqrt();
    let peak = v.iter().fold(0.0f32, |a, b| a.max(b.abs()));
    let finite = v.iter().all(|x| x.is_finite());
    println!(
        "  {name:22} {:>8} samples  rms {rms:.5} ({:+.1} dBFS)  peak {peak:.4}  finite {finite}",
        v.len(),
        20.0 * rms.max(1e-12).log10()
    );
}

/// RMS per `hop` samples. The two signals are at different rates — 16 kHz in,
/// 32 kHz out — so each gets a hop covering the same wall-clock and the two
/// envelopes land on one grid.
fn envelope(samples: &[f32], hop: usize) -> Vec<f64> {
    samples
        .chunks_exact(hop)
        .map(|frame| {
            let power: f64 = frame.iter().map(|s| (*s as f64).powi(2)).sum();
            (power / hop as f64).sqrt()
        })
        .collect()
}

fn correlation(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }
    let (a, b) = (&a[..n], &b[..n]);
    let mean = |v: &[f64]| v.iter().sum::<f64>() / n as f64;
    let (ma, mb) = (mean(a), mean(b));
    let (mut cov, mut va, mut vb) = (0.0, 0.0, 0.0);
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

impl common::Job for Reconstruct {
    fn run<B: Backend>(self, device: &B::Device) {
        let raw = std::fs::read(&self.input).expect("read input");
        let pcm16k: Vec<f32> = raw
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        println!("input: {:.2} s at 16 kHz", pcm16k.len() as f32 / 16_000.0);
        describe("source (16k)", &pcm16k);

        // --- semantic tokens -------------------------------------------------
        let mut hubert = Hubert::<B>::new(&HubertConfig::chinese_base(), device);
        hubert.load_pytorch(&self.hubert).expect("load hubert");

        let mut sovits = SovitsPartial::<B>::new(&SovitsConfig::default(), device);
        sovits.load_pytorch(&self.sovits).expect("load s2");

        let wav =
            Tensor::<B, 2>::from_data(TensorData::new(pcm16k.clone(), [1, pcm16k.len()]), device);
        let ssl = hubert.forward(wav).swap_dims(1, 2);
        println!("  cnhubert           {:?}", ssl.dims());

        let codes = sovits.quantizer.encode(ssl);
        let ids: Vec<i64> = codes.clone().into_data().to_vec().unwrap();
        let distinct: std::collections::HashSet<_> = ids.iter().collect();
        println!(
            "  semantic tokens    {:?}  {} distinct of {}",
            codes.dims(),
            distinct.len(),
            ids.len()
        );

        // --- speaker reference ----------------------------------------------
        // The reference is a spectrogram of the audio itself, resampled to the
        // 32 kHz the synthesizer works in. Nearest-neighbour is crude but this
        // only has to be speech-shaped, not faithful.
        let pcm32k: Vec<f32> = (0..pcm16k.len() * 2).map(|i| pcm16k[i / 2]).collect();
        let spectral = Spectral::<B>::new(&SpectralConfig::gptsovits_v2_32k(), device);
        let refer = spectral.linear(Tensor::from_data(
            TensorData::new(pcm32k, [1, pcm16k.len() * 2]),
            device,
        ));
        println!("  reference spec     {:?}", refer.dims());
        let g = sovits.speaker(refer);
        let gv: Vec<f32> = g.clone().into_data().to_vec().unwrap();
        describe("speaker vector", &gv);

        // --- synthesise -------------------------------------------------------
        // Arbitrary phonemes: this checks the path runs and produces speech-like
        // audio, not that it says anything in particular.
        let phones: Vec<i32> = (0..12).map(|i| 100 + i).collect();
        let text = Tensor::<B, 2, Int>::from_data(
            TensorData::new(phones.clone(), [1, phones.len()]),
            device,
        );
        let audio = sovits.decode(codes, text, g, 0.5);
        println!("  decoder out        {:?}", audio.dims());

        let out: Vec<f32> = audio.into_data().to_vec().unwrap();
        describe("reconstructed (32k)", &out);

        // The number that says the stage is *wired* right rather than merely
        // loaded, and the reason this example exists beyond a coverage count.
        // Envelope rather than samples: the prior is sampled and the phonemes
        // are arbitrary, so a sample-wise difference between input and output
        // is large and means nothing — but a mis-wired MRTE or a mis-scaled
        // style attention returns something whose loudness stops tracking the
        // source at all.
        //
        // 20 ms frames at each rate, so the two envelopes share a grid. The
        // shuffled control is printed beside it because a correlation is only
        // meaningful against what chance would give: speech envelopes are
        // autocorrelated, so even unrelated ones agree somewhat.
        let src_env = envelope(&pcm16k, 16_000 / 50);
        let out_env = envelope(&out, 32_000 / 50);
        println!(
            "\nenergy envelope ({} frames)",
            src_env.len().min(out_env.len())
        );
        println!(
            "  r vs source      : {:.4}",
            correlation(&src_env, &out_env)
        );
        println!(
            "  r vs shuffled    : {:.4}   (chance baseline)",
            correlation(&src_env, &shuffled(&out_env))
        );

        let bytes: Vec<u8> = out.iter().flat_map(|s| s.to_le_bytes()).collect();
        std::fs::write(&self.output, bytes).expect("write output");
        println!("\nwrote {} ({} samples @32 kHz)", self.output, out.len());
    }
}

fn main() {
    let (backend, args) = common::parse_args();
    if args.len() < 4 {
        eprintln!(
            "usage: reconstruct [--backend ..] <hubert.bin> <s2G.pth> <in.f32le@16k> <out.f32le@32k>"
        );
        std::process::exit(2);
    }
    println!("backend : {backend}");
    common::run_on(
        backend,
        Reconstruct {
            hubert: args[0].clone(),
            sovits: args[1].clone(),
            input: args[2].clone(),
            output: args[3].clone(),
        },
    );
}
