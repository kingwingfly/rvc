//! Load the Seed-VC checkpoint and report weight coverage, module by module.
//!
//! This is the crate's stand-in for unit tests: the network has none, so the
//! check is that real released weights map onto the module tree with nothing
//! missing. The counts must come out identical on every backend — a backend that
//! changes them is a bug.
//!
//! **While the port is in progress this also serves as the map.** Run it against
//! the checkpoint and it prints how many tensors sit under each top-level
//! prefix, which is how each module's owner knows what their subtree has to
//! account for. As a module lands, its arm goes in `report` below and its
//! prefix moves from "not yet ported" to a real applied/missing/unused triple.
//!
//! Measured against
//! `DiT_seed_v2_uvit_whisper_small_wavenet_bigvgan_pruned.pth` (440 MB), the
//! whole of which is **302 tensors / 110,035,232 parameters**:
//!
//! | prefix | tensors | module |
//! |---|---|---|
//! | `net.cfm.module.estimator.*` | 255 | [`dit`] and [`wavenet`] |
//! | `net.length_regulator.module.*` | 22 | [`length_regulator`] |
//! | `net.style_encoder.module.*` | 18 | [`style_encoder`] |
//! | `net.vq.module.quantizers.*` | 7 | [`vq`] |
//!
//! Two of the six networks are **not in this file at all** — the content encoder
//! is `openai/whisper-small` and the vocoder is
//! `nvidia/bigvgan_v2_22khz_80band_256x`, each from its own release. Hunting for
//! their tensors here is a way to lose an afternoon.
//!
//! Usage: `cargo run -p burn-seedvc --example load -- [--backend ndarray|cuda|tch] <ckpt.pth>`

#[path = "common/mod.rs"]
mod common;

use std::collections::BTreeMap;

use burn::tensor::backend::Backend;
use burn::tensor::{Distribution, Int, Tensor};
use burn_seedvc::style_encoder::{StyleEncoder, StyleEncoderConfig};

struct Load {
    checkpoint: String,
}

impl common::Job for Load {
    fn run<B: Backend>(self, device: &B::Device) {
        let tensors = burn_kit::store::pytorch_keys(self.checkpoint.as_ref(), None)
            .expect("failed to read checkpoint");

        // Group by the first path component: upstream's module boundaries are
        // exactly this crate's, so a prefix is one unit's territory.
        let mut by_prefix: BTreeMap<&str, (usize, usize)> = BTreeMap::new();
        for (name, _dtype, shape) in &tensors {
            let prefix = name.split('.').next().unwrap_or(name);
            let entry = by_prefix.entry(prefix).or_default();
            entry.0 += 1;
            entry.1 += shape.iter().product::<usize>();
        }

        println!("{:<28} {:>8}  {:>14}", "prefix", "tensors", "parameters");
        for (prefix, (count, params)) in &by_prefix {
            println!("{prefix:<28} {count:>8}  {params:>14}");
        }
        println!(
            "\n{} tensors, {} parameters across {} prefixes",
            tensors.len(),
            by_prefix.values().map(|(_, p)| p).sum::<usize>(),
            by_prefix.len()
        );
        // Per-module coverage. One block per module as it lands; a prefix that
        // is never claimed is a piece of the model nobody ported.
        //
        // Every loader is handed the whole 302-tensor checkpoint, so `unused`
        // arrives full of other modules' weights. Each block filters it down to
        // its own subtree — the only number that says anything about the port.
        let cfg = StyleEncoderConfig::default();
        let mut style = StyleEncoder::<B>::new(&cfg, device);
        let res = style
            .load_pytorch(&self.checkpoint)
            .expect("failed to load net.style_encoder");
        // `load_pytorch` strips `net.style_encoder.module.` off the keys it
        // claims, so anything still wearing a `net.` prefix belongs to someone
        // else and is not this module's business.
        let unused: Vec<_> = res
            .unused
            .iter()
            .filter(|key| !key.starts_with("net."))
            .collect();
        println!("\nstyle_encoder (net.style_encoder.module.*)");
        println!("  applied : {}", res.applied.len());
        println!("  missing : {}", res.missing.len());
        for (name, why) in &res.missing {
            println!("      MISSING {name}  ({why})");
        }
        println!("  unused  : {} (in this subtree)", unused.len());
        for name in &unused {
            println!("      UNUSED {name}");
        }
        println!("  errors  : {}", res.errors.len());
        for e in &res.errors {
            println!("      ERROR {e:?}");
        }

        // Coverage proves the layout, never the arithmetic — this repo has
        // shipped a port that loaded at 100% and produced garbage. The cheapest
        // check that exercises the forward pass: a timbre encoder that ignores
        // its input fails in the way that looks healthiest, converting every
        // clip into the same voice. Synthetic mels are enough to catch it, and
        // they keep this harness free of an audio dependency.
        //
        // The two references differ in *spectral tilt*, not just in their
        // samples: two white-noise mels are the same signal twice as far as any
        // timbre encoder is concerned, so they would agree closely however the
        // arithmetic were wired, and the comparison would prove nothing.
        let normal = Distribution::Normal(0.0, 1.0);
        let a = Tensor::<B, 3>::random([1, cfg.n_mels, 128], normal, device);
        let tilt = Tensor::<B, 1, Int>::arange(0..cfg.n_mels as i64, device)
            .float()
            .reshape([1, cfg.n_mels, 1]);
        let b = Tensor::<B, 3>::random([1, cfg.n_mels, 128], normal, device) * tilt;
        let embed = |mel| -> Vec<f32> { style.forward(mel).into_data().to_vec().unwrap() };
        let (va, va_again, vb) = (embed(a.clone()), embed(a), embed(b));
        let dot = |x: &[f32], y: &[f32]| x.iter().zip(y).map(|(p, q)| p * q).sum::<f32>();
        let cosine = dot(&va, &vb) / (dot(&va, &va) * dot(&vb, &vb)).sqrt();
        println!(
            "  forward : finite={}, repeatable={}, cos(two references)={cosine:.4}",
            va.iter().all(|x| x.is_finite()),
            va == va_again,
        );
    }
}

fn main() {
    let (backend, args) = common::parse_args();
    let Some(checkpoint) = args.first() else {
        eprintln!("usage: load [--backend ndarray|cuda|tch] <checkpoint>");
        std::process::exit(2);
    };
    println!("backend : {backend}");
    common::run_on(
        backend,
        Load {
            checkpoint: checkpoint.clone(),
        },
    );
}
