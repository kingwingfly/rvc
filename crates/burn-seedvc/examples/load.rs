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
//! | `net.style_encoder.module.*` | 18 | [`style_encoder`] |
//! | `net.vq.module.quantizers.*` | 7 | [`vq`] |
//!
//! Two of the six networks are **not in this file at all** — the content encoder
//! is `openai/whisper-small` and the vocoder is
//! `nvidia/bigvgan_v2_22khz_80band_256x`, each from its own release. Hunting for
//! their tensors here is a way to lose an afternoon. The vocoder's own
//! `bigvgan_generator.pt` can be named as a second argument, and then it gets a
//! coverage block of its own; leaving it off skips that block rather than
//! failing, so the Seed-VC checkpoint alone is still a complete run.
//!
//! Usage: `cargo run -p burn-seedvc --example load -- [--backend ndarray|cuda|tch] <ckpt.pth> [bigvgan_generator.pt]`

#[path = "common/mod.rs"]
mod common;

use std::collections::BTreeMap;

use burn::tensor::backend::Backend;
use burn::tensor::{Distribution, Int, Tensor};
use burn_seedvc::style_encoder::{StyleEncoder, StyleEncoderConfig};
use burn_seedvc::{
    BigVgan, BigVganConfig, InterpolateRegulator, ResidualVq, SeedVcConfig, VqConfig,
};

struct Load {
    checkpoint: String,
    bigvgan: Option<String>,
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
        // Per-module coverage. One block per module as it lands; a prefix that
        // is never claimed is a piece of the model nobody ported.
        //
        // Every loader is handed the whole 302-tensor checkpoint, so `unused`
        // arrives full of other modules' weights. Each block filters it down to
        // its own subtree — the only number that says anything about the port.
        let cfg = StyleEncoderConfig::default();
        let mut style = StyleEncoder::<B>::new(&cfg, device);
        let res = style
            .load_pytorch(&self.checkpoint)
            .expect("failed to load net.style_encoder");
        // `load_pytorch` strips `net.style_encoder.module.` off the keys it
        // claims, so anything still wearing a `net.` prefix belongs to someone
        // else and is not this module's business.
        let unused: Vec<_> = res
            .unused
            .iter()
            .filter(|key| !key.starts_with("net."))
            .collect();
        println!("\nstyle_encoder (net.style_encoder.module.*)");
        println!("  applied : {}", res.applied.len());
        println!("  missing : {}", res.missing.len());
        for (name, why) in &res.missing {
            println!("      MISSING {name}  ({why})");
        }
        println!("  unused  : {} (in this subtree)", unused.len());
        for name in &unused {
            println!("      UNUSED {name}");
        }
        println!("  errors  : {}", res.errors.len());
        for e in &res.errors {
            println!("      ERROR {e:?}");
        }

        // Coverage proves the layout, never the arithmetic — this repo has
        // shipped a port that loaded at 100% and produced garbage. The cheapest
        // check that exercises the forward pass: a timbre encoder that ignores
        // its input fails in the way that looks healthiest, converting every
        // clip into the same voice. Synthetic mels are enough to catch it, and
        // they keep this harness free of an audio dependency.
        //
        // The two references differ in *spectral tilt*, not just in their
        // samples: two white-noise mels are the same signal twice as far as any
        // timbre encoder is concerned, so they would agree closely however the
        // arithmetic were wired, and the comparison would prove nothing.
        let normal = Distribution::Normal(0.0, 1.0);
        let a = Tensor::<B, 3>::random([1, cfg.n_mels, 128], normal, device);
        let tilt = Tensor::<B, 1, Int>::arange(0..cfg.n_mels as i64, device)
            .float()
            .reshape([1, cfg.n_mels, 1]);
        let b = Tensor::<B, 3>::random([1, cfg.n_mels, 128], normal, device) * tilt;
        let embed = |mel| -> Vec<f32> { style.forward(mel).into_data().to_vec().unwrap() };
        let (va, va_again, vb) = (embed(a.clone()), embed(a), embed(b));
        let dot = |x: &[f32], y: &[f32]| x.iter().zip(y).map(|(p, q)| p * q).sum::<f32>();
        let cosine = dot(&va, &vb) / (dot(&va, &va) * dot(&vb, &vb)).sqrt();
        println!(
            "  forward : finite={}, repeatable={}, cos(two references)={cosine:.4}",
            va.iter().all(|x| x.is_finite()),
            va == va_again,
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

        // The vocoder is the one module whose weights are **not** in the file
        // above, so it is the one block that can be given a whole checkpoint of
        // its own. Nothing else lives in `bigvgan_generator.pt`, which is why
        // `report`'s "belonging to other modules" tally comes out at zero here
        // and would be a real finding if it did not.
        //
        // An `if let` rather than a `let … else … return`, because blocks are
        // appended to this function as modules land and an early return here
        // would silently skip every one of them whenever the optional second
        // argument is left off.
        if let Some(path) = &self.bigvgan {
            let mut vocoder = BigVgan::<B>::new(&BigVganConfig::v2_22khz_80band_256x(), device);
            // Taken before the load, because the checkpoint is about to
            // overwrite it.
            let derived = vocoder.derived_filter();
            let res = vocoder.load_pytorch(path).expect("failed to load bigvgan");
            report("bigvgan (its own checkpoint)", &res);

            // The anti-aliasing kernels are a deterministic function of the
            // filter design, and upstream stores them anyway because
            // `register_buffer` is persistent. That redundancy is free evidence:
            // the copy the file carries is an independent answer to the same
            // arithmetic, so the two agreeing says the Kaiser window, the sinc
            // grid and the normalisation are all right. A disagreement here is a
            // real defect that no coverage count and no shape check would show.
            let loaded = vocoder.derived_filter();
            let worst = derived
                .iter()
                .zip(&loaded)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            println!(
                "  filter  : derived vs stored, max |Δ| = {worst:.3e} over {} taps",
                derived.len()
            );
        } else {
            println!(
                "\nbigvgan: skipped — pass `nvidia/bigvgan_v2_22khz_80band_256x`'s \
                 `bigvgan_generator.pt` as a second argument to cover it"
            );
        }
    }
}

fn main() {
    let (backend, args) = common::parse_args();
    let Some(checkpoint) = args.first() else {
        eprintln!("usage: load [--backend ndarray|cuda|tch] <checkpoint> [bigvgan_generator.pt]");
        std::process::exit(2);
    };
    println!("backend : {backend}");
    common::run_on(
        backend,
        Load {
            checkpoint: checkpoint.clone(),
            bigvgan: args.get(1).cloned(),
        },
    );
}
