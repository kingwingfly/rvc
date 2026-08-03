//! Embed several clips and print their pairwise cosine matrix — **the timbre
//! oracle for the whole engine.**
//!
//! Weight coverage says [`burn_seedvc::campplus`]'s module tree matches
//! `campplus_cn_common.bin`, and [`burn_seedvc::fbank`] has no checkpoint to
//! cover at all, so neither has anything to report about whether the two
//! together compute a *speaker*. This is the check that does, and it needs no
//! reference implementation: run clips through the pair and read the cosine
//! matrix. **The number that matters is the gap** — same-speaker pairs clearly
//! above cross-speaker ones. A filterbank that is subtly wrong still clusters
//! somewhat, so a high same-speaker score on its own proves nothing; a *narrow*
//! gap is the failure this example exists to make visible.
//!
//! Zero-shot conversion is conditioned on nothing but this vector, so a weak
//! separation here is an upper bound on everything downstream: the transformer
//! cannot render a timbre the encoder did not distinguish.
//!
//! The input is raw mono f32le at 16 kHz — CAM++'s analysis rate, and not one
//! this crate is allowed to resample to itself, since a `burn-*` crate has no
//! app dependencies and ffmpeg is an app dependency:
//!
//! ```sh
//! ffmpeg -i clip.wav -f f32le -ac 1 -ar 16000 clip.raw
//! cargo run -p burn-seedvc --example speaker -- campplus_cn_common.bin a.raw b.raw c.raw
//! ```
//!
//! Two clips are enough to run it and tell you nothing. Give it at least two of
//! one speaker and one of another, or there is no gap to read.
//!
//! # Choosing the clips is most of the work
//!
//! **A weak-looking matrix is far more often a weak clip set than a broken
//! port**, and two ways of building one were tried here and thrown away:
//!
//! - **A voice converted through a trained `rvc` model is only a second speaker
//!   if that model was trained on somebody else's voice.** Converting a corpus
//!   clip through the voice fine-tuned *on that corpus* is close to an identity,
//!   so the two halves of the matrix are one speaker and the gap collapses.
//! - **Do not infer ground truth from a filename.** Two clips whose names differ
//!   only in a trailing index need not be one person; on the set this was first
//!   run against they were not, and the "same-speaker" row that looked broken was
//!   reporting correctly.
//!
//! What works is either genuinely different recordings of a speaker you know, or
//! — as a lower bound that needs no second speaker at all — the two halves of one
//! recording, which should sit near 0.85.

#[path = "common/mod.rs"]
mod common;

use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, TensorData};
use burn_seedvc::campplus::{CamPPlus, CamPPlusConfig};
use burn_seedvc::fbank::{Fbank, FbankConfig};

struct Speaker {
    weights: String,
    clips: Vec<String>,
}

/// Cosine similarity. The embedding is not unit norm — CAM++'s last layer is a
/// non-affine batch norm, not a normalisation over the vector — so the divisor
/// is doing real work and is not a formality to drop.
fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let dot: f64 = a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum();
    let norm = |v: &[f32]| v.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    dot / (norm(a) * norm(b)).max(f64::MIN_POSITIVE)
}

impl common::Job for Speaker {
    fn run<B: Backend>(self, device: &B::Device) {
        let mut model = CamPPlus::<B>::new(&CamPPlusConfig::default(), device);
        let res = model.load_pytorch(&self.weights).expect("load campplus");
        println!(
            "weights : {} applied, {} missing, {} unused",
            res.applied.len(),
            res.missing.len(),
            res.unused.len()
        );
        assert!(res.missing.is_empty(), "{:?}", res.missing);

        let cfg = FbankConfig::default();
        let fbank = Fbank::<B>::new(&cfg, device);
        let mut embeddings = Vec::new();

        println!();
        for (i, path) in self.clips.iter().enumerate() {
            let raw = std::fs::read(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
            let pcm: Vec<f32> = raw
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect();
            let wav =
                Tensor::<B, 2>::from_data(TensorData::new(pcm.clone(), [1, pcm.len()]), device);

            let feats = fbank.forward(wav);
            let [_, frames, bins] = feats.dims();
            let embedding: Vec<f32> = model.forward(feats).into_data().to_vec().unwrap();
            assert!(
                embedding.iter().all(|x| x.is_finite()),
                "{path} embedded to a non-finite vector"
            );
            let norm = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
            println!(
                "[{i}] {path}\n    {:.2} s, {frames} frames x {bins} bins, |e| = {norm:.3}",
                pcm.len() as f64 / cfg.sample_rate as f64,
            );
            embeddings.push(embedding);
        }

        println!("\npairwise cosine");
        print!("     ");
        for i in 0..embeddings.len() {
            print!("{:>8}", format!("[{i}]"));
        }
        println!();
        for (i, a) in embeddings.iter().enumerate() {
            print!("[{i}] ");
            for b in &embeddings {
                print!("{:>8.3}", cosine(a, b));
            }
            println!();
        }
    }
}

fn main() {
    let (backend, args) = common::parse_args();
    if args.len() < 2 {
        eprintln!(
            "usage: speaker [--backend ndarray|cuda|tch] <campplus_cn_common.bin> \
             <clip.f32le@16000>..."
        );
        std::process::exit(2);
    }
    println!("backend : {backend}");
    common::run_on(
        backend,
        Speaker {
            weights: args[0].clone(),
            clips: args[1..].to_vec(),
        },
    );
}
