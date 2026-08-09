//! Load a Hugging Face Whisper checkpoint and report weight coverage.
//!
//! This is the crate's stand-in for unit tests: the network has none, so the
//! check is that real published weights map onto the module tree with nothing
//! missing and nothing unused. For `openai/whisper-large-v3-turbo` the answer
//! must be **587 / 0 / 0**, and it must be identical on every backend — a
//! backend that changes the counts is a bug.
//!
//! **The third number is the *genuinely* unused count, not the raw one.** Every
//! LayerNorm `weight`/`bias` is reported unconsumed because the adapter applies
//! it under Burn's `gamma`/`beta` names, so the raw `unused` is 156 for turbo
//! and 124 for small; both partition to zero real leftovers, which is what
//! `587 / 0 / 0` and `479 / 0 / 0` mean here.
//!
//! It catches a wrong module *layout*. It cannot catch a wrong *formula*; that
//! is what comparing transcripts against `whisper.cpp` is for.
//!
//! `--size` picks which [`WhisperConfig`] preset the checkpoint is measured
//! against, because the counts are a property of the pair. `whisper-small` is
//! **479 / 0 / 0** — fewer than turbo's 587 only because it has 12 encoder
//! layers against 32, and a proportionally larger decoder against turbo's 4.
//! `--size large-v3` has no triple recorded here because no run of it has been
//! measured; it is the preset, not a number to be assumed from turbo's.
//!
//! `--strict` turns the report into a **gate**, exiting 1 on anything but the
//! load above. The allowance is a predicate over the leftover *names* — every
//! unused key must be a LayerNorm `weight`/`bias` — rather than the count, which
//! would pass just as happily on 156 wrong ones. Pointing `--size` at the other
//! preset's weights is the cheapest way to watch it fail: the shapes disagree,
//! which lands in `errors`, and `errors` is why this checks more than `missing`.
//!
//! Usage: `cargo run -p burn-whisper --example load -- [--backend ndarray|cuda|tch] [--size large-v3-turbo|large-v3|small] [--strict] <model.safetensors>`

#[path = "common/mod.rs"]
mod common;

use burn::tensor::backend::Backend;
use burn_store::ModuleSnapshot;
use burn_whisper::{Whisper, WhisperConfig};

struct Load {
    weights: String,
    cfg: WhisperConfig,
    size: String,
    strict: bool,
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

        // The order of the four questions is not arbitrary. `burn-store` derives
        // `missing` as `visited && !applied && !skipped && !errored`, so a tensor
        // whose shape did not match is dropped from *both* lists and a checkpoint
        // that failed to apply reads as full coverage — `errors` has to be asked
        // first. An empty `applied` is the same failure one step further along: a
        // rename that matched nothing leaves every parameter at its initialised
        // value, which is a model that transcribes fluent nonsense.
        if self.strict {
            let mut failures: Vec<String> = Vec::new();
            if !res.errors.is_empty() {
                failures.push(format!(
                    "{} checkpoint tensor(s) could not be applied — a shape mismatch is dropped \
                     from `missing` too, so this is the question that has to come first",
                    res.errors.len()
                ));
            }
            if !res.missing.is_empty() {
                failures.push(format!(
                    "{} model parameter(s) had no checkpoint tensor",
                    res.missing.len()
                ));
            }
            if res.applied.is_empty() {
                failures.push(
                    "nothing applied — every parameter is still at its initialised value".into(),
                );
            }
            if !real.is_empty() {
                // Printed in full rather than `take(20)`: this is the list a
                // reader has to talk themselves out of, and a truncated one is
                // how a predicate gets written that does not cover the tail.
                failures.push(format!(
                    "{} checkpoint tensor(s) are unused and are not a LayerNorm gamma/beta: {}",
                    real.len(),
                    real.iter()
                        .map(|k| k.as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                ));
            }
            if !failures.is_empty() {
                eprintln!(
                    "\n--strict: not the load this crate records for `--size {}`. \
                     A checkpoint of another size is the usual cause — the counts are a \
                     property of the config/weights *pair*, not of either alone.",
                    self.size
                );
                for why in &failures {
                    eprintln!("  - {why}");
                }
                std::process::exit(1);
            }
            println!("\n--strict: ok (every unused tensor is a LayerNorm gamma/beta)");
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
    let strict = args.iter().any(|a| a == "--strict");
    args.retain(|a| a != "--strict");
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

    // `parse_args` hands anything it does not recognise back as a positional, and
    // this example reads only the *first* one — so a typo'd `--stict` after the
    // path would be silently ignored, which is a gate that never fires. Refusing
    // an unknown flag is the point of `--strict` applied to its own arguments.
    if let Some(flag) = args.iter().find(|a| a.starts_with("--")) {
        eprintln!("error: unknown flag `{flag}`");
        std::process::exit(2);
    }

    let Some(weights) = args.first().cloned() else {
        eprintln!(
            "usage: load [--backend ndarray|cuda|tch] [--size large-v3-turbo|large-v3|small] [--strict] <model.safetensors>"
        );
        std::process::exit(2);
    };
    println!("backend : {backend}");
    println!("size    : {size}");
    common::run_on(
        backend,
        Load {
            weights,
            cfg,
            size,
            strict,
        },
    );
}
