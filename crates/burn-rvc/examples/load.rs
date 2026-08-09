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
//! `--strict` turns the report into a **gate**, exiting 1 on anything but the
//! loads this crate records: `f0G48k.pth` at **560 / 0 / 0**, `f0D48k.pth` at
//! **165 / 0 / 0**, and `hubert_base` at **210 / 0 / 57** where all 57 are named
//! below. Without it the example prints its `MISSING` lines and still exits 0,
//! so nothing but a reader stands between a renamed tensor and a model running
//! on freshly-initialised weights.
//!
//! Usage:
//! - `cargo run -p burn-rvc --example load -- [--backend ndarray|cuda|tch] [--strict] <G.pth> [D.pth]`
//! - `cargo run -p burn-rvc --example load -- [--backend …] [--strict] contentvec <hubert_base dir|pytorch_model.bin>`

#[path = "common/mod.rs"]
mod common;

use burn::tensor::backend::Backend;
use burn_kit::ApplyResult;
use burn_rvc::{ContentVec, MultiPeriodDiscriminator, Synthesizer, SynthesizerConfig};

/// Turn one coverage report into a verdict, printing every way it falls short.
///
/// The order of the four questions is not arbitrary. `burn-store` derives
/// `missing` as `visited && !applied && !skipped && !errored`, so a tensor whose
/// shape did not match is dropped from *both* lists and a checkpoint that failed
/// to apply reads as full coverage — `errors` has to be asked first. An empty
/// `applied` is the same failure one step further along: a rename that matched
/// nothing leaves every parameter at its initialised value, and this network
/// then converts speech into noise with nothing said.
///
/// `unaccounted` is the caller's allowance already applied — the `unused` keys
/// this particular checkpoint is *not* expected to leave. A name rather than a
/// count, because a count of 57 passes just as happily on 57 wrong tensors.
fn verdict(what: &str, recorded: &str, res: &ApplyResult, unaccounted: &[&String]) -> bool {
    let mut failures: Vec<String> = Vec::new();
    if !res.errors.is_empty() {
        failures.push(format!(
            "{} checkpoint tensor(s) could not be applied — a shape mismatch is dropped from \
             `missing` too, so this is the question that has to come first",
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
        failures.push("nothing applied — every parameter is still at its initialised value".into());
    }
    if !unaccounted.is_empty() {
        // Printed in full rather than truncated: this is the list a reader has to
        // talk themselves out of, and a truncated one is how an allowance gets
        // written that does not cover the tail.
        failures.push(format!(
            "{} checkpoint tensor(s) are unused with no recorded reason: {}",
            unaccounted.len(),
            unaccounted
                .iter()
                .map(|k| k.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        ));
    }
    if failures.is_empty() {
        println!("--strict: {what} ok ({recorded})");
        return true;
    }
    eprintln!("\n--strict: {what} is not the load this crate records ({recorded}).");
    for why in &failures {
        eprintln!("  - {why}");
    }
    false
}

struct Load {
    generator: String,
    discriminator: Option<String>,
    strict: bool,
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

        // `f0G48k.pth` is the synthesizer's own file and nothing else's, so the
        // allowance here is the strongest one there is: **nothing** may be left
        // over. Anything unused is a tensor upstream ships and this port does not
        // read, which is exactly what a reader should be made to look at.
        let mut ok = !self.strict
            || verdict(
                "the generator",
                "560 / 0 / 0",
                &res,
                &res.unused.iter().collect::<Vec<_>>(),
            );

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
            if self.strict {
                ok &= verdict(
                    "the discriminator",
                    "165 / 0 / 0",
                    &dres,
                    &dres.unused.iter().collect::<Vec<_>>(),
                );
            }
        }

        // Both are reported before either verdict decides the exit code: a run
        // that fails on the generator is exactly the run whose discriminator
        // numbers are worth seeing.
        if !ok {
            std::process::exit(1);
        }
    }
}

struct LoadContentVec {
    weights: String,
    strict: bool,
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

        // Every unused tensor is accounted for by name rather than by a total,
        // because the whole value of this check is that a *new* one stands out.
        // `final_proj` is RVC v1's ninth-layer readout and `masked_spec_embed` is
        // SpecAugment's mask token — both in the file, both read by nothing at
        // inference. The LayerNorms are a reporting artefact: `burn-store`
        // applies them under Burn's `gamma`/`beta` names and still counts the
        // originals unconsumed.
        let class = |k: &str| {
            if k.starts_with("final_proj.") {
                "final_proj (v1's readout)"
            } else if k == "masked_spec_embed" {
                "masked_spec_embed (SpecAugment)"
            } else if k.ends_with("layer_norm.weight") || k.ends_with("layer_norm.bias") {
                "LayerNorm (applied as gamma/beta)"
            } else {
                "UNACCOUNTED"
            }
        };
        println!("unused  : {}", res.unused.len());
        let mut counts: std::collections::BTreeMap<&str, Vec<&String>> = Default::default();
        for name in &res.unused {
            counts.entry(class(name)).or_default().push(name);
        }
        for (why, names) in &counts {
            println!("    {:3}  {why}", names.len());
            if *why == "UNACCOUNTED" {
                for name in names {
                    println!("         {name}");
                }
            }
        }
        println!("errors  : {}", res.errors.len());
        for e in &res.errors {
            println!("    ERROR {e:?}");
        }

        // The classifier above is already the allowance, so `--strict` is the
        // same question the reader was being asked: is anything `UNACCOUNTED`?
        // Written that way round rather than as `unused == 57` on purpose —
        // fifty-seven *different* leftovers would satisfy the count.
        if self.strict
            && !verdict(
                "contentvec",
                "210 / 0 / 57",
                &res,
                counts.get("UNACCOUNTED").map(Vec::as_slice).unwrap_or(&[]),
            )
        {
            std::process::exit(1);
        }
    }
}

fn main() {
    let (backend, mut args) = common::parse_args();

    // Pulled out before anything positional is read: the arms below index
    // `args` by position (`contentvec` at 0, `D.pth` at 1), so a `--strict` left
    // in the list would shift them.
    let strict = args.iter().any(|a| a == "--strict");
    args.retain(|a| a != "--strict");
    // `parse_args` hands anything it does not recognise back as a positional, so
    // a typo'd `--stict` would be read as a checkpoint path — or, past the two
    // this example reads, ignored outright. A gate that silently never fires is
    // the exact failure `--strict` exists to prevent.
    if let Some(flag) = args.iter().find(|a| a.starts_with("--")) {
        eprintln!("error: unknown flag `{flag}`");
        std::process::exit(2);
    }

    let contentvec = args.first().is_some_and(|a| a == "contentvec");
    if contentvec {
        args.remove(0);
    }

    let Some(first) = args.first().cloned() else {
        eprintln!(
            "usage: load [--backend ndarray|cuda|tch] [--strict] <G.pth> [D.pth]\n   \
             or: load [--backend ndarray|cuda|tch] [--strict] contentvec <hubert_base dir|pytorch_model.bin>"
        );
        std::process::exit(2);
    };
    println!("backend : {backend}");

    if contentvec {
        common::run_on(
            backend,
            LoadContentVec {
                weights: first,
                strict,
            },
        );
    } else {
        common::run_on(
            backend,
            Load {
                generator: first,
                discriminator: args.get(1).cloned(),
                strict,
            },
        );
    }
}
