//! Load a pretrained generator and run one inference forward pass, printing the
//! output waveform shape (should be `frames · hop_length` samples).
//!
//! Complements `load`: that one proves the *weights* map on, this one proves the
//! graph actually runs on the selected backend.
//!
//! Usage: `cargo run -p burn-rvc --example infer -- [--backend ndarray|cuda|tch] <checkpoint.pth>`

#[path = "common/mod.rs"]
mod common;

use burn::tensor::backend::Backend;
use burn::tensor::{Distribution, Int, Tensor};
use burn_rvc::{Synthesizer, SynthesizerConfig};

struct Infer {
    path: String,
}

impl common::Job for Infer {
    fn run<B: Backend>(self, device: &B::Device) {
        let cfg = SynthesizerConfig::v2_48k();
        let hop = cfg.hop_length();
        let mut model = Synthesizer::<B>::new(&cfg, device);
        let res = model
            .load_weights(&self.path)
            .expect("failed to load checkpoint");
        println!(
            "loaded {} params ({} missing)",
            res.applied.len(),
            res.missing.len()
        );

        // Dummy analysis features for T frames.
        let t = 120usize;
        let phone = Tensor::<B, 3>::random([1, t, 768], Distribution::Normal(0.0, 1.0), device);
        let pitch = Tensor::<B, 2, Int>::full([1, t], 128, device);
        let nsff0 = Tensor::<B, 2>::full([1, t], 220.0, device);

        let audio = model.infer(phone, pitch, nsff0, 0);
        let dims = audio.dims();
        println!("output waveform dims: {dims:?}");
        println!("expected samples: {} (= {t} frames · {hop} hop)", t * hop);

        let flat = audio.into_data().to_vec::<f32>().unwrap();
        let (min, max) = flat
            .iter()
            .fold((f32::MAX, f32::MIN), |(lo, hi), &v| (lo.min(v), hi.max(v)));
        println!(
            "sample range: [{min:.4}, {max:.4}] over {} samples",
            flat.len()
        );
    }
}

fn main() {
    let (backend, args) = common::parse_args();
    let Some(path) = args.first().cloned() else {
        eprintln!("usage: infer [--backend ndarray|cuda|tch] <checkpoint.pth>");
        std::process::exit(2);
    };
    println!("backend : {backend}");
    common::run_on(backend, Infer { path });
}
