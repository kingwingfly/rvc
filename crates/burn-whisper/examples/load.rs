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
//! `--size` picks which [`WhisperConfig`] preset the checkpoint is measured
//! against, because the counts are a property of the pair. `whisper-small` is
//! **479 / 0 / 0** — fewer than turbo's 587 only because it has 12 encoder
//! layers against 32, and a proportionally larger decoder against turbo's 4.
//!
//! Usage: `cargo run -p burn-whisper --example load -- [--backend ndarray|cuda|tch] [--size large-v3-turbo|large-v3|small] <model.safetensors>`

#[path = "common/mod.rs"]
mod common;

use burn::tensor::backend::Backend;
use burn_store::ModuleSnapshot;
use burn_whisper::{Whisper, WhisperConfig};

struct Load {
    weights: String,
    cfg: WhisperConfig,
}

impl common::Job for Load {
    fn run<B: Backend>(self, device: &B::Device) {
        let mut model = Whisper::<B>::new(&self.cfg, device);

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
    let (backend, mut args) = common::parse_args();

    // Hand-rolled rather than a flag parser: this crate has no app dependencies,
    // and `--backend` is already pulled out the same way.
    let mut size = "large-v3-turbo".to_string();
    if let Some(i) = args.iter().position(|a| a == "--size") {
        args.remove(i);
        match args.get(i) {
            Some(_) => size = args.remove(i),
            None => {
                eprintln!("error: --size needs a value");
                std::process::exit(2);
            }
        }
    }
    let cfg = match size.as_str() {
        "large-v3-turbo" => WhisperConfig::large_v3_turbo(),
        "large-v3" => WhisperConfig::large_v3(),
        "small" => WhisperConfig::small(),
        other => {
            eprintln!("error: unknown size `{other}` (expected large-v3-turbo, large-v3 or small)");
            std::process::exit(2);
        }
    };

    let Some(weights) = args.first().cloned() else {
        eprintln!(
            "usage: load [--backend ndarray|cuda|tch] [--size large-v3-turbo|large-v3|small] <model.safetensors>"
        );
        std::process::exit(2);
    };
    println!("backend : {backend}");
    println!("size    : {size}");
    common::run_on(backend, Load { weights, cfg });
}
