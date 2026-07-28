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
