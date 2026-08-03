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
//! | `net.style_encoder.module.*` | 18 | [`campplus`] |
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
use burn_seedvc::{InterpolateRegulator, ResidualVq, SeedVcConfig, VqConfig};

struct Load {
    checkpoint: String,
}

/// Applied/missing/unused for one module, in the form every `burn-*` crate's
/// `load` example reports it.
///
/// Two things make a raw `unused` count meaningless here, and both are subtracted
/// rather than hidden:
///
/// - **The checkpoint is one file holding every module**, so loading any one of
///   them leaves the other ~280 tensors unconsumed. Those still carry their
///   `net.<module>.` prefix, because each loader's remaps strip only its own —
///   which is exactly what separates "belongs to somebody else" from "belongs
///   here and did not land".
/// - A normalisation layer's `weight`/`bias` are consumed under Burn's
///   `gamma`/`beta` names and the store still counts the original keys as
///   unconsumed, so they appear here *having been applied*.
///
/// **What is left after both is real, and 0 is the only acceptable number.**
fn report(label: &str, result: &burn_store::ApplyResult) {
    let mine: Vec<&String> = result
        .unused
        .iter()
        .filter(|k| !k.starts_with("net."))
        .collect();
    let (norms, real): (Vec<&String>, Vec<&String>) = mine.iter().copied().partition(|k| {
        let stem = k.rsplit_once('.').map(|(s, _)| s).unwrap_or(k);
        (k.ends_with(".weight") || k.ends_with(".bias")) && stem.contains("norm")
    });
    println!(
        "\n{label}\n  applied : {}\n  missing : {}\n  unused  : {} in this subtree ({} norm \
         gamma/beta, reported but applied; {} genuinely unused) + {} belonging to other \
         modules\n  errors  : {}",
        result.applied.len(),
        result.missing.len(),
        mine.len(),
        norms.len(),
        real.len(),
        result.unused.len() - mine.len(),
        result.errors.len(),
    );
    for (name, why) in &result.missing {
        println!("    MISSING {name}  ({why})");
    }
    for name in &real {
        println!("    UNUSED {name}");
    }
    for e in &result.errors {
        println!("    ERROR {e:?}");
    }
}

impl common::Job for Load {
    fn run<B: Backend>(self, device: &B::Device) {
        let tensors = burn_kit::store::pytorch_keys(self.checkpoint.as_ref(), None)
            .expect("failed to read checkpoint");

        // Group by the *second* path component: upstream's module boundaries are
        // exactly this crate's, but every key in this checkpoint starts `net.`,
        // so the first component alone puts all 302 tensors in one bucket and
        // says nothing.
        let mut by_prefix: BTreeMap<&str, (usize, usize)> = BTreeMap::new();
        for (name, _dtype, shape) in &tensors {
            let depth = if name.starts_with("net.") { 2 } else { 1 };
            let prefix = name
                .match_indices('.')
                .nth(depth - 1)
                .map_or(name.as_str(), |(i, _)| &name[..i]);
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
        println!(
            "\nA prefix with no coverage block below is a piece of the model nobody \
             has ported yet."
        );

        let cfg = SeedVcConfig::uvit_whisper_small_wavenet();

        let mut regulator = InterpolateRegulator::<B>::new(&cfg, device);
        let res = regulator
            .load_pytorch(&self.checkpoint)
            .expect("failed to read checkpoint");
        report("net.length_regulator.module.*", &res);

        // Reported for completeness rather than because anything runs it:
        // upstream's `build_model` does not construct this subtree at all, so it
        // is a residue of the training script. See `burn_seedvc::vq`.
        let mut vq = ResidualVq::<B>::new(&VqConfig::default(), device);
        let res = vq
            .load_pytorch(&self.checkpoint)
            .expect("failed to read checkpoint");
        report("net.vq.module.quantizers.*", &res);
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
