//! Load a GPT-SoVITS component's checkpoint and report weight coverage.
//!
//! The crate's stand-in for unit tests, as in `burn-rvc` and `burn-whisper`: the
//! networks have none, so correctness starts with real published weights mapping
//! onto the module tree with nothing missing. It catches a wrong module *layout*.
//! It cannot catch a wrong *formula* — post-norm where the reference is pre-norm
//! loads at 100% and is wrong — which is what end-to-end listening is for.
//!
//! `--strict` turns the report into a **gate**, exiting 1 on anything but the
//! load that component records. What it gates on differs per component, and the
//! split is [`Allowance`]'s: a checkpoint that holds the component and nothing
//! else can be gated on `unused` *by name*, one that holds a whole other model
//! cannot be gated on `unused` at all.
//!
//! Usage: `cargo run -p burn-gptsovits --example load -- [--backend ndarray|cuda|tch] [--strict] <hubert|quantizer|sovits|t2s|disc> <checkpoint>`

#[path = "common/mod.rs"]
mod common;

use burn::tensor::backend::Backend;
use burn_gptsovits::{
    GPTSOVITS_V2_PERIODS, Hubert, HubertConfig, Quantizer, QuantizerConfig, SovitsConfig,
    SovitsPartial, T2s, T2sConfig,
};
use burn_vits::MultiPeriodDiscriminator;

/// What a component's checkpoint is allowed to leave `unused`, and why the two
/// shapes are not interchangeable.
enum Allowance {
    /// The file holds this component and (as far as the port is concerned)
    /// nothing else, so `unused` is a closed set and can be gated **by name**.
    /// The listed strings are final path segments allowed *beyond* the norm
    /// `weight`/`bias` every load here misreports as unconsumed.
    ///
    /// A name and not a count: `hubert`'s 55 leftovers would satisfy `== 55`
    /// just as happily if all 55 were different tensors.
    ByName(&'static [&'static str]),
    /// The file holds a whole other model too, so what is left over is unbounded
    /// and says nothing — `quantizer` is 3 tensors out of `s2G2333k.pth`'s 776,
    /// and the same three are 773 `unused` from its own point of view. There is
    /// no predicate to write, so `applied` is pinned to the recorded count
    /// instead; without it this component has no gate at all, since dropping a
    /// submodule would lower `applied`, leave `missing` at 0, and be absorbed
    /// silently by `unused`.
    ApplyCount(usize),
}

struct Load {
    component: String,
    weights: String,
    strict: bool,
}

impl common::Job for Load {
    fn run<B: Backend>(self, device: &B::Device) {
        // The recorded triple and the allowance travel with the arm that
        // produced the result, so adding a component cannot leave either behind.
        let (res, recorded, allowance) = match self.component.as_str() {
            "hubert" => {
                let mut model = Hubert::<B>::new(&HubertConfig::chinese_base(), device);
                (
                    model.load_pytorch(&self.weights),
                    "210 / 0 / 55",
                    // SpecAugment's mask token: in the file, read by nothing at
                    // inference. RVC's ContentVec leaves the same one.
                    Allowance::ByName(&["masked_spec_embed"]),
                )
            }
            "quantizer" => {
                let mut model = Quantizer::<B>::new(&QuantizerConfig::default(), 1, device);
                (
                    model.load_pytorch(&self.weights),
                    "3 / 0 / 773",
                    Allowance::ApplyCount(3),
                )
            }
            // `load_weights`, not `load_pytorch`: these are the two components
            // fine-tuning also *writes*, so the example has to cover the Burn
            // `.safetensors` path as well as the original `.pth`/`.ckpt` — it is
            // the harness to reach for when `tts` refuses a tuned checkpoint.
            "sovits" => {
                let mut model = SovitsPartial::<B>::new(&SovitsConfig::default(), device);
                (
                    model.load_weights(&self.weights),
                    "773 / 0 / 3",
                    // The codebook's EMA training statistics. Matched on the leaf
                    // rather than the whole path so a second quantiser layer
                    // brings three more of the same kind rather than a failure.
                    Allowance::ByName(&["cluster_size", "embed_avg", "inited"]),
                )
            }
            "t2s" => {
                let mut model = T2s::<B>::new(&T2sConfig::default(), device);
                (
                    model.load_weights(&self.weights),
                    "295 / 0 / 96",
                    Allowance::ByName(&[]),
                )
            }
            // The adversary `s2` fine-tuning warm-starts from. Its state dict
            // sits under `weight`, where RVC's sits under `model` — the one
            // difference between the two projects' discriminator checkpoints,
            // and the reason the key is a parameter.
            "disc" => {
                let mut model = MultiPeriodDiscriminator::<B>::new(&GPTSOVITS_V2_PERIODS, device);
                (
                    model.load_pytorch(&self.weights, Some("weight")),
                    "111 / 0 / 0",
                    // Recorded rather than measured here: `s2D2333k.pth` was not
                    // on the machine this gate was written on. RVC's own
                    // discriminator, the same `burn-vits` module under a different
                    // state-dict key, leaves nothing over — so nothing is the
                    // allowance until somebody runs this and finds otherwise.
                    Allowance::ByName(&[]),
                )
            }
            other => {
                eprintln!(
                    "unknown component `{other}` (known: hubert, quantizer, sovits, t2s, disc)"
                );
                std::process::exit(2);
            }
        };
        let res = res.expect("failed to read checkpoint");

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

        if !self.strict {
            return;
        }

        // The order of the questions is not arbitrary. `burn-store` derives
        // `missing` as `visited && !applied && !skipped && !errored`, so a tensor
        // whose shape did not match is dropped from *both* lists and a checkpoint
        // that failed to apply reads as full coverage — `errors` has to be asked
        // first. An empty `applied` is the same failure one step further along: a
        // rename that matched nothing leaves every parameter at its initialised
        // value, and the stage then runs and synthesises noise.
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
            failures
                .push("nothing applied — every parameter is still at its initialised value".into());
        }
        match allowance {
            Allowance::ByName(allowed) => {
                let unaccounted: Vec<&&String> = real
                    .iter()
                    .filter(|k| {
                        let leaf = k.rsplit('.').next().unwrap_or(k);
                        !allowed.contains(&leaf)
                    })
                    .collect();
                if !unaccounted.is_empty() {
                    // In full rather than `take(20)`: this is the list a reader
                    // has to talk themselves out of, and a truncated one is how
                    // an allowance gets written that does not cover the tail.
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
            }
            Allowance::ApplyCount(want) => {
                if res.applied.len() != want {
                    failures.push(format!(
                        "applied {} where {want} is recorded — `unused` cannot be checked for \
                         this component, so this count is the whole gate",
                        res.applied.len()
                    ));
                }
            }
        }

        if !failures.is_empty() {
            eprintln!(
                "\n--strict: not the load `{}` records ({recorded}).",
                self.component
            );
            for why in &failures {
                eprintln!("  - {why}");
            }
            std::process::exit(1);
        }
        println!("\n--strict: ok ({recorded})");
    }
}

fn main() {
    let (backend, mut args) = common::parse_args();

    // Pulled out before anything positional is read: the two arguments below are
    // taken by position, so a `--strict` left in the list would become the
    // component name. Hand-rolled for the reason `--backend` is — this crate has
    // no app dependencies and should not gain `clap` for one flag.
    let strict = args.iter().any(|a| a == "--strict");
    args.retain(|a| a != "--strict");
    // `parse_args` hands anything it does not recognise back as a positional, and
    // this example reads only the first two — so a typo'd `--stict` at the end
    // would be ignored outright, which is a gate that silently never fires.
    if let Some(flag) = args.iter().find(|a| a.starts_with("--")) {
        eprintln!("error: unknown flag `{flag}`");
        std::process::exit(2);
    }

    let (Some(component), Some(weights)) = (args.first().cloned(), args.get(1).cloned()) else {
        eprintln!("usage: load [--backend ndarray|cuda|tch] [--strict] <component> <checkpoint>");
        std::process::exit(2);
    };
    println!("backend : {backend}");
    common::run_on(
        backend,
        Load {
            component,
            weights,
            strict,
        },
    );
}
