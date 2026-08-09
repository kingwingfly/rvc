//! Load `rmvpe.pt` and report weight coverage.
//!
//! The crate's stand-in for a unit test on the module tree: real published
//! weights have to map onto it with nothing missing. For `rmvpe.pt` the answer
//! is **623 applied / 0 missing / 118 unused**, identical on every backend — a
//! backend that changes the counts is a bug.
//!
//! The 118 are one `num_batches_tracked` per `BatchNorm`, a training counter
//! with no inference role, which `unet::Norm` deliberately does not model.
//!
//! **Coverage cannot catch a wrong formula**, and this network has a specific
//! way of being wrong that coverage will never see: a bidirectional GRU whose
//! gates are in the wrong order, or whose two biases are fused, loads perfectly
//! and predicts nonsense. `gru`'s unit tests are the check for that, and
//! comparing decoded F0 against `rmvpe.onnx` on real audio is the check for
//! everything else.
//!
//! `--strict` turns the report into a **gate**: without it this example prints
//! 118 `MISSING` lines and still exits 0, which is a report nobody has to read.
//! With it, anything but the load above exits 1. The allowance is written as a
//! predicate over the leftover names rather than as the number 118, because a
//! count passes when 118 *different* tensors are the ones left over.
//!
//! Usage: `cargo run -p burn-rmvpe --example load -- [--backend ndarray|cuda|tch] [--strict] <rmvpe.pt>`

#[path = "common/mod.rs"]
mod common;

use burn::tensor::backend::Backend;
use burn_rmvpe::{Rmvpe, RmvpeConfig};

struct Load {
    weights: String,
    strict: bool,
}

impl common::Job for Load {
    fn run<B: Backend>(self, device: &B::Device) {
        let cfg = RmvpeConfig::default();
        let mut model = Rmvpe::<B>::new(&cfg, device);

        let res = model
            .load_pytorch(&self.weights)
            .expect("failed to read checkpoint");

        println!("applied : {}", res.applied.len());
        println!(
            "missing : {}  (model params with no checkpoint tensor)",
            res.missing.len()
        );
        for (name, why) in &res.missing {
            println!("    MISSING {name}  ({why})");
        }

        let (counters, real): (Vec<_>, Vec<_>) = res
            .unused
            .iter()
            .partition(|k| k.ends_with("num_batches_tracked"));
        println!(
            "unused  : {} ({} num_batches_tracked, expected; {} genuinely unused)",
            res.unused.len(),
            counters.len(),
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
        // value, and this network then predicts pitch confidently and wrongly.
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
                // Printed in full rather than `take(20)`: this is the list the
                // reader has to talk themselves out of, and a truncated one is
                // how a predicate gets written that does not cover the tail.
                failures.push(format!(
                    "{} checkpoint tensor(s) are unused and are not `num_batches_tracked`: {}",
                    real.len(),
                    real.iter()
                        .map(|k| k.as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                ));
            }
            if !failures.is_empty() {
                eprintln!("\n--strict: not the load this crate records (623 / 0 / 118).");
                for why in &failures {
                    eprintln!("  - {why}");
                }
                std::process::exit(1);
            }
            println!("\n--strict: ok (every unused tensor is a `num_batches_tracked` counter)");
        }

        // A loaded model that cannot run is a port that only looks finished, so
        // push one buffer through. The frame count is deliberately *not* a
        // multiple of 32: alignment is `forward`'s job, and this is where a
        // caller would otherwise have found that out.
        let frames = 100;
        let mel = burn::tensor::Tensor::<B, 3>::zeros([1, cfg.n_mels, frames], device);
        let salience = model.forward(mel);
        assert_eq!(salience.dims(), [1, frames, cfg.n_class]);
        let v: Vec<f32> = salience.into_data().to_vec().expect("f32 salience");
        // Checked separately from any approximate comparison: Burn's
        // `assert_approx_eq` treats NaN as equal to NaN, so a NaN would sail
        // through a value check.
        assert!(v.iter().all(|x| x.is_finite()), "salience must be finite");
        let mean = v.iter().map(|x| *x as f64).sum::<f64>() / v.len() as f64;
        let peak = v.iter().fold(0.0f32, |a, b| a.max(*b));
        println!(
            "\nforward : [1, {frames}, {}] mean={mean:.5} peak={peak:.5}",
            cfg.n_class
        );
        println!("          (silence in, so a low mean is what a working model gives)");
    }
}

fn main() {
    let (backend, mut args) = common::parse_args();

    // Pulled out before anything positional is read, and hand-rolled for the
    // reason `--backend` is: this crate has no app dependencies and should not
    // gain `clap` for one flag.
    let strict = args.iter().any(|a| a == "--strict");
    args.retain(|a| a != "--strict");
    // `parse_args` hands anything it does not recognise back as a positional, so
    // a typo'd `--stict` would otherwise be read as the checkpoint path — or, on
    // an example that takes fewer positionals than it was given, ignored. A gate
    // that silently never fires is the exact failure `--strict` exists to
    // prevent, so an unknown flag is refused rather than absorbed.
    if let Some(flag) = args.iter().find(|a| a.starts_with("--")) {
        eprintln!("error: unknown flag `{flag}`");
        std::process::exit(2);
    }

    let Some(weights) = args.first().cloned() else {
        eprintln!("usage: load [--backend ndarray|cuda|tch] [--strict] <rmvpe.pt>");
        std::process::exit(2);
    };
    println!("backend : {backend}");
    common::run_on(backend, Load { weights, strict });
}
