//! Load a reference RVC checkpoint and report weight coverage.
//!
//! This is the crate's stand-in for unit tests: the network has none, so the
//! check is that real pretrained weights map onto the module tree with nothing
//! missing. The counts must come out identical on every backend — a backend that
//! changes them is a bug.
//!
//! Two checkpoints, because RVC is two published files rather than one. The
//! default is the generator (and optionally its discriminator); `contentvec`
//! is the content encoder on the input side, `hubert_base/` from the same
//! `lj1995/VoiceConversionWebUI` repository.
//!
//! **Coverage cannot catch a wrong readout.** ContentVec's checkpoint carries
//! `final_proj`, which is RVC *v1*'s ninth-layer head; v2 reads the 12th
//! encoder layer directly, so those two tensors are expected to land in
//! `unused` and taking them as a signal to wire them up is exactly the wrong
//! move. Comparing features against `vec-768-layer-12.onnx` on real audio is
//! the check for everything coverage cannot see.
//!
//! Usage:
//! - `cargo run -p burn-rvc --example load -- [--backend ndarray|cuda|tch] <G.pth> [D.pth]`
//! - `cargo run -p burn-rvc --example load -- [--backend …] contentvec <hubert_base dir|pytorch_model.bin>`

#[path = "common/mod.rs"]
mod common;

use burn::tensor::backend::Backend;
use burn_rvc::{ContentVec, MultiPeriodDiscriminator, Synthesizer, SynthesizerConfig};

struct Load {
    generator: String,
    discriminator: Option<String>,
}

impl common::Job for Load {
    fn run<B: Backend>(self, device: &B::Device) {
        let cfg = SynthesizerConfig::v2_48k();
        let mut model = Synthesizer::<B>::new(&cfg, device);

        let res = model
            .load_pytorch(&self.generator)
            .expect("failed to read checkpoint");

        println!("applied : {}", res.applied.len());
        println!(
            "missing : {}  (model params with no checkpoint tensor)",
            res.missing.len()
        );
        for (name, why) in &res.missing {
            println!("    MISSING {name}  ({why})");
        }
        println!(
            "unused  : {}  (checkpoint tensors not yet ported)",
            res.unused.len()
        );
        let mut prefixes: Vec<_> = res
            .unused
            .iter()
            .map(|k| k.split('.').next().unwrap_or(k).to_string())
            .collect();
        prefixes.sort();
        prefixes.dedup();
        println!("    unused top-level modules: {prefixes:?}");
        println!("errors  : {}", res.errors.len());
        for e in &res.errors {
            println!("    ERROR {e:?}");
        }

        if let Some(dpath) = &self.discriminator {
            println!("\n== discriminator {dpath} ==");
            let mut disc = MultiPeriodDiscriminator::<B>::new(&burn_rvc::RVC_V2_PERIODS, device);
            let dres = disc
                .load_pytorch(dpath, Some("model"))
                .expect("failed to read D checkpoint");
            println!("applied : {}", dres.applied.len());
            println!("missing : {}", dres.missing.len());
            for (name, why) in &dres.missing {
                println!("    MISSING {name}  ({why})");
            }
            println!("unused  : {}", dres.unused.len());
            println!("errors  : {}", dres.errors.len());
        }
    }
}

struct LoadContentVec {
    weights: String,
}

impl common::Job for LoadContentVec {
    fn run<B: Backend>(self, device: &B::Device) {
        let (_, res) = ContentVec::<B>::load(std::path::Path::new(&self.weights), device)
            .expect("failed to read checkpoint");

        println!("applied : {}", res.applied.len());
        println!(
            "missing : {}  (model params with no checkpoint tensor)",
            res.missing.len()
        );
        for (name, why) in &res.missing {
            println!("    MISSING {name}  ({why})");
        }

        // `final_proj` is v1's readout and `masked_spec_embed` is SpecAugment's
        // mask token; both are in the file and read by nothing at inference. The
        // LayerNorms are a reporting artefact — `burn-store` applies them under
        // Burn's `gamma`/`beta` names and still counts the originals unconsumed.
        let expected = |k: &str| {
            k.starts_with("final_proj.")
                || k == "masked_spec_embed"
                || k.ends_with("layer_norm.weight")
                || k.ends_with("layer_norm.bias")
        };
        let (known, real): (Vec<_>, Vec<_>) = res.unused.iter().partition(|k| expected(k));
        println!(
            "unused  : {} ({} expected — v1's final_proj, the mask token, the \
             renamed LayerNorms; {} genuinely unused)",
            res.unused.len(),
            known.len(),
            real.len()
        );
        for name in &real {
            println!("    UNUSED {name}");
        }
        println!("errors  : {}", res.errors.len());
        for e in &res.errors {
            println!("    ERROR {e:?}");
        }
    }
}

fn main() {
    let (backend, mut args) = common::parse_args();
    let contentvec = args.first().is_some_and(|a| a == "contentvec");
    if contentvec {
        args.remove(0);
    }

    let Some(first) = args.first().cloned() else {
        eprintln!(
            "usage: load [--backend ndarray|cuda|tch] <G.pth> [D.pth]\n   \
             or: load [--backend ndarray|cuda|tch] contentvec <hubert_base dir|pytorch_model.bin>"
        );
        std::process::exit(2);
    };
    println!("backend : {backend}");

    if contentvec {
        common::run_on(backend, LoadContentVec { weights: first });
    } else {
        common::run_on(
            backend,
            Load {
                generator: first,
                discriminator: args.get(1).cloned(),
            },
        );
    }
}
