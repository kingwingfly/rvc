//! Load a reference RVC generator checkpoint and report weight coverage.
//!
//! This is the crate's stand-in for unit tests: the network has none, so the
//! check is that real pretrained weights map onto the module tree with nothing
//! missing. The counts must come out identical on every backend — a backend that
//! changes them is a bug.
//!
//! Usage: `cargo run -p burn-rvc --example load -- [--backend ndarray|cuda|tch] <G.pth> [D.pth]`

#[path = "common/mod.rs"]
mod common;

use burn::tensor::backend::Backend;
use burn_rvc::{MultiPeriodDiscriminator, Synthesizer, SynthesizerConfig};

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

fn main() {
    let (backend, args) = common::parse_args();
    let Some(generator) = args.first().cloned() else {
        eprintln!("usage: load [--backend ndarray|cuda|tch] <G.pth> [D.pth]");
        std::process::exit(2);
    };
    println!("backend : {backend}");
    common::run_on(
        backend,
        Load {
            generator,
            discriminator: args.get(1).cloned(),
        },
    );
}
