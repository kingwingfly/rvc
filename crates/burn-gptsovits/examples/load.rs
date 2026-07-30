//! Load a GPT-SoVITS component's checkpoint and report weight coverage.
//!
//! The crate's stand-in for unit tests, as in `burn-rvc` and `burn-whisper`: the
//! networks have none, so correctness starts with real published weights mapping
//! onto the module tree with nothing missing. It catches a wrong module *layout*.
//! It cannot catch a wrong *formula* — post-norm where the reference is pre-norm
//! loads at 100% and is wrong — which is what end-to-end listening is for.
//!
//! Usage: `cargo run -p burn-gptsovits --example load -- [--backend ndarray|cuda|tch] <hubert|quantizer|sovits|t2s|disc> <checkpoint>`

#[path = "common/mod.rs"]
mod common;

use burn::tensor::backend::Backend;
use burn_gptsovits::{
    GPTSOVITS_V2_PERIODS, Hubert, HubertConfig, Quantizer, QuantizerConfig, SovitsConfig,
    SovitsPartial, T2s, T2sConfig,
};
use burn_vits::MultiPeriodDiscriminator;

struct Load {
    component: String,
    weights: String,
}

impl common::Job for Load {
    fn run<B: Backend>(self, device: &B::Device) {
        let res = match self.component.as_str() {
            "hubert" => {
                let mut model = Hubert::<B>::new(&HubertConfig::chinese_base(), device);
                model.load_pytorch(&self.weights)
            }
            "quantizer" => {
                let mut model = Quantizer::<B>::new(&QuantizerConfig::default(), 1, device);
                model.load_pytorch(&self.weights)
            }
            "sovits" => {
                let mut model = SovitsPartial::<B>::new(&SovitsConfig::default(), device);
                model.load_pytorch(&self.weights)
            }
            "t2s" => {
                let mut model = T2s::<B>::new(&T2sConfig::default(), device);
                model.load_pytorch(&self.weights)
            }
            // The adversary `s2` fine-tuning warm-starts from. Its state dict
            // sits under `weight`, where RVC's sits under `model` — the one
            // difference between the two projects' discriminator checkpoints,
            // and the reason the key is a parameter.
            "disc" => {
                let mut model = MultiPeriodDiscriminator::<B>::new(&GPTSOVITS_V2_PERIODS, device);
                model.load_pytorch(&self.weights, Some("weight"))
            }
            other => {
                eprintln!(
                    "unknown component `{other}` (known: hubert, quantizer, sovits, t2s, disc)"
                );
                std::process::exit(2);
            }
        }
        .expect("failed to read checkpoint");

        println!("applied : {}", res.applied.len());
        println!(
            "missing : {}  (model params with no checkpoint tensor)",
            res.missing.len()
        );
        for (name, why) in &res.missing {
            println!("    MISSING {name}  ({why})");
        }
        // Every norm's weight/bias shows up here even though it applied: the
        // adapter consumes them as Burn's gamma/beta and the store still counts
        // the original key as unconsumed. Anything else in this list is real.
        let (norms, real): (Vec<_>, Vec<_>) = res.unused.iter().partition(|k| {
            let stem = k.rsplit_once('.').map(|(s, _)| s).unwrap_or(k);
            (k.ends_with(".weight") || k.ends_with(".bias"))
                && (stem.contains("norm") || stem.ends_with("_ln"))
        });
        println!(
            "unused  : {} ({} norm gamma/beta, reported but applied; {} genuinely unused)",
            res.unused.len(),
            norms.len(),
            real.len()
        );
        for name in real.iter().take(20) {
            println!("    UNUSED {name}");
        }
        println!("errors  : {}", res.errors.len());
        for e in &res.errors {
            println!("    ERROR {e:?}");
        }
    }
}

fn main() {
    let (backend, args) = common::parse_args();
    let (Some(component), Some(weights)) = (args.first().cloned(), args.get(1).cloned()) else {
        eprintln!("usage: load [--backend ndarray|cuda|tch] <component> <checkpoint>");
        std::process::exit(2);
    };
    println!("backend : {backend}");
    common::run_on(backend, Load { component, weights });
}
