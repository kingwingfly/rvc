//! Load a Hugging Face Whisper checkpoint and report weight coverage.
//!
//! This is the crate's stand-in for unit tests: the network has none, so the
//! check is that real published weights map onto the module tree with nothing
//! missing and nothing unused. For `openai/whisper-large-v3-turbo` the answer
//! must be **587 / 0 / 0**, and it must be identical on every backend — a
//! backend that changes the counts is a bug.
//!
//! It catches a wrong module *layout*. It cannot catch a wrong *formula*; that
//! is what comparing transcripts against `whisper.cpp` is for.
//!
//! Usage: `cargo run -p burn-whisper --example load -- [--backend ndarray|cuda|tch] <model.safetensors>`

#[path = "common/mod.rs"]
mod common;

use burn::tensor::backend::Backend;
use burn_store::ModuleSnapshot;
use burn_whisper::{Whisper, WhisperConfig};

struct Load {
    weights: String,
}

impl common::Job for Load {
    fn run<B: Backend>(self, device: &B::Device) {
        let cfg = WhisperConfig::large_v3_turbo();
        let mut model = Whisper::<B>::new(&cfg, device);

        let res = model
            .load_safetensors(&self.weights)
            .expect("failed to read checkpoint");

        println!("applied : {}", res.applied.len());
        println!(
            "missing : {}  (model params with no checkpoint tensor)",
            res.missing.len()
        );
        for (name, why) in &res.missing {
            println!("    MISSING {name}  ({why})");
        }
        // Every LayerNorm `weight`/`bias` shows up here even though it applied:
        // the adapter consumes them under Burn's `gamma`/`beta` names, and the
        // store records the original key as unconsumed. `applied` and the spot
        // checks below are the ground truth. Anything else in this list is real.
        let (norms, real): (Vec<_>, Vec<_>) = res
            .unused
            .iter()
            .partition(|k| k.ends_with("_norm.weight") || k.ends_with("_norm.bias"));
        println!(
            "unused  : {} ({} LayerNorm gamma/beta, reported but applied; {} genuinely unused)",
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

        // Coverage counts say a tensor was *matched*, not that its numbers landed
        // in the right parameter. Print a few so they can be diffed against the
        // checkpoint itself — LayerNorm especially, whose `weight`/`bias` reach
        // Burn's `gamma`/`beta` through the adapter's alternative-name path, and
        // which are exactly the tensors `unused` above misreports.
        println!("\nspot checks — diff these against the .safetensors:");
        let want = [
            "encoder.layer_norm.gamma",
            "encoder.layer_norm.beta",
            "decoder.layers.0.self_attn_layer_norm.gamma",
            "encoder.layers.0.self_attn.q_proj.bias",
        ];
        for snap in model.collect(None, None, false) {
            let path = snap.full_path();
            if !want.contains(&path.as_str()) {
                continue;
            }
            let data = snap.to_data().expect("materialize parameter");
            let v: Vec<f32> = data.to_vec().expect("f32 parameter");
            let head: Vec<String> = v.iter().take(5).map(|x| format!("{x:.5}")).collect();
            println!(
                "  {path:44} first5=[{}]  sum={:.4}",
                head.join(", "),
                v.iter().map(|x| *x as f64).sum::<f64>()
            );
        }
    }
}

fn main() {
    let (backend, args) = common::parse_args();
    let Some(weights) = args.first().cloned() else {
        eprintln!("usage: load [--backend ndarray|cuda|tch] <model.safetensors>");
        std::process::exit(2);
    };
    println!("backend : {backend}");
    common::run_on(backend, Load { weights });
}
