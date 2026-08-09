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
//! | prefix | tensors | module | applied/missing/unused |
//! |---|---|---|---|
//! | `net.cfm.module.estimator.*` | 255 | [`dit`] and [`wavenet`] | 255/0/0 |
//! | `net.length_regulator.module.*` | 22 | [`length_regulator`] | 22/0/0 (+8 norm aliases) |
//! | `net.style_encoder.module.*` | 18 | [`style_encoder`] | 18/0/0 |
//! | `net.vq.module.quantizers.*` | 7 | [`vq`] | 7/0/0 |
//!
//! and the two networks that arrive in their own releases:
//!
//! | file | module | applied/missing/unused |
//! |---|---|---|
//! | `campplus_cn_common.bin` | [`campplus`] | **815/0/122** |
//! | `bigvgan_generator.pt` | [`bigvgan`] | 783/0/0 |
//!
//! The 122 are one `num_batches_tracked` per norm — a training counter PyTorch
//! stores as a buffer and inference never reads. The 8 beside the length
//! regulator are norm `weight`/`bias` pairs consumed under Burn's `gamma`/`beta`
//! names, which `burn-store` counts as unconsumed *while having applied them*.
//! Both are subtracted rather than hidden, and what is left is 0 everywhere.
//!
//! **`Σ applied over the four blocks above == the checkpoint's own tensor
//! count`** is the assertion that makes this a test rather than a printout, and
//! it is computed from the file rather than written down: a subtree nobody
//! ported cannot be seen in any single block's `unused`, because each loader's
//! remap strips only its own prefix and everything else is filed as "somebody
//! else's". It shows up in the sum, and nowhere else.
//!
//! Two of the six networks are **not in this file at all** — the content encoder
//! is `openai/whisper-small` and the vocoder is
//! `nvidia/bigvgan_v2_22khz_80band_256x`, each from its own release. Hunting for
//! their tensors here is a way to lose an afternoon. The vocoder's own
//! `bigvgan_generator.pt` can be named as a second argument, and then it gets a
//! coverage block of its own; leaving it off skips that block rather than
//! failing, so the Seed-VC checkpoint alone is still a complete run.
//!
//! The **speaker encoder is a third file again** — `campplus_cn_common.bin` from
//! `funasr/campplus`, which is what upstream conditions the transformer on.
//!
//! So three released files between them hold this model, and each extra one is
//! named rather than positional: passing only the Seed-VC checkpoint is still a
//! complete run, and each flag adds its own block instead of being required to
//! reach the next.
//!
//! `--strict` turns every one of those numbers into an exit code, which is what
//! makes this runnable from a script. Without it the example prints a fault and
//! exits 0, which is how a load that reported 90 `MISSING` lines could look like
//! a clean run to anything that was not reading the output.
//!
//! ```text
//! cargo run -p burn-seedvc --example load -- [--backend ndarray|cuda|tch] [--strict] \
//!     <ckpt.pth> [--campplus campplus_cn_common.bin] [--bigvgan bigvgan_generator.pt]
//! ```

#[path = "common/mod.rs"]
mod common;

use std::collections::BTreeMap;

use burn::tensor::backend::Backend;
use burn::tensor::{Distribution, Int, Tensor};
use burn_seedvc::campplus::{CamPPlus, CamPPlusConfig};
use burn_seedvc::style_encoder::{StyleEncoder, StyleEncoderConfig};
use burn_seedvc::{
    BigVgan, BigVganConfig, Dit, InterpolateRegulator, ResidualVq, SeedVcConfig, VqConfig,
};

struct Load {
    checkpoint: String,
    /// `campplus_cn_common.bin`, which is a **different file** — see the block
    /// that uses it.
    campplus: Option<String>,
    /// `bigvgan_generator.pt`, a third file again.
    bigvgan: Option<String>,
    /// Exit non-zero on any fault, rather than printing it and exiting 0.
    strict: bool,
}

/// What one module's load came to: how much of the checkpoint it claimed, and
/// everything wrong with it.
struct Coverage {
    /// Tensors this module claimed. Summed over the modules that read the
    /// Seed-VC checkpoint, this is what has to equal the file's own tensor
    /// count — see the module docs for why no single block's `unused` can
    /// substitute for it.
    applied: usize,
    /// Already-phrased complaints, empty when the module loaded cleanly. What
    /// `--strict` exits on.
    faults: Vec<String>,
}

/// Is `key` a norm's `weight`/`bias`, reported unused but in fact applied under
/// Burn's `gamma`/`beta` names?
///
/// **The test is the last path segment, not a substring of the parent path**,
/// and that is a correction rather than a nicety. `stem.contains("norm")` files
/// every orphaned key that merely *sits under* a norm-ish parent as a benign
/// alias: `AdaLayerNorm` holds a plain `Linear` called `project_layer`, so a
/// genuinely unclaimed `blocks.0.attention_norm.project_layer.weight` matched
/// the substring and vanished into the count nobody reads. So did anything under
/// `ffn_norm.*` or `transformer.norm.*`.
///
/// A compound name has to keep matching, which is why this is not simply
/// `== "norm"`: the checkpoint's own aliases arrive as `blocks.N.norm.weight`
/// here and as `attention_norm.weight` elsewhere, and both are real norms.
fn is_norm_alias(key: &str) -> bool {
    let Some((stem, leaf)) = key.rsplit_once('.') else {
        return false;
    };
    if leaf != "weight" && leaf != "bias" {
        return false;
    }
    let field = stem.rsplit('.').next().unwrap_or(stem);
    field == "norm" || field.ends_with("_norm")
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
///   here and did not land". It is a *presumption*, though, and a weak one: a
///   loader strips one prefix, so a tensor at `net.cfm.module.<something-else>.*`
///   keeps its `net.` and is filed under somebody else's name while belonging to
///   nobody. Only the sum over every block catches that, which is why this
///   returns [`Coverage::applied`] rather than printing it and forgetting it.
/// - A normalisation layer's `weight`/`bias` are consumed under Burn's
///   `gamma`/`beta` names and the store still counts the original keys as
///   unconsumed, so they appear here *having been applied* — [`is_norm_alias`].
///
/// **What is left after both is real, and 0 is the only acceptable number.**
fn report(label: &str, result: &burn_store::ApplyResult) -> Coverage {
    let mine: Vec<&String> = result
        .unused
        .iter()
        .filter(|k| !k.starts_with("net."))
        .collect();
    let (norms, real): (Vec<&String>, Vec<&String>) = mine
        .iter()
        .copied()
        .partition(|k| is_norm_alias(k.as_str()));
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
    Coverage {
        applied: result.applied.len(),
        faults: faults_for(label, result, real.len()),
    }
}

/// Everything about one load that `--strict` refuses to exit 0 on.
///
/// **`errors` is tested first, and not only for tidiness.** `burn-store`'s
/// applier computes `missing` as `visited && !applied && !skipped && !errored`,
/// so a tensor that failed on a shape mismatch is excluded from `missing` — read
/// the two counts in the other order and a module whose every weight was the
/// wrong shape reports full coverage.
///
/// `applied == 0` is called out separately because it is the observed shape of
/// pointing a loader at the wrong file: the read succeeds, every key misses, and
/// the `Ok` is what makes it look like a load rather than a mistake.
fn faults_for(label: &str, result: &burn_store::ApplyResult, real_unused: usize) -> Vec<String> {
    let mut faults = Vec::new();
    if !result.errors.is_empty() {
        faults.push(format!("{label}: {} tensors errored", result.errors.len()));
    }
    if !result.missing.is_empty() {
        faults.push(format!("{label}: {} tensors missing", result.missing.len()));
    }
    if real_unused > 0 {
        faults.push(format!(
            "{label}: {real_unused} checkpoint tensors nothing claimed"
        ));
    }
    if result.applied.is_empty() {
        faults.push(format!(
            "{label}: nothing applied at all — the usual cause is the wrong file, \
             which reads cleanly and matches no key"
        ));
    }
    faults
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
        // Every block that reads the Seed-VC checkpoint pushes its `applied`
        // here. The sum is the only check that sees a subtree nobody ported —
        // see the module docs.
        let mut claimed = Vec::new();
        let mut faults = Vec::new();

        let cfg = StyleEncoderConfig::default();
        let mut style = StyleEncoder::<B>::new(&cfg, device);
        let res = style
            .load_pytorch(&self.checkpoint)
            .expect("failed to load net.style_encoder");
        let cov = report("style_encoder (net.style_encoder.module.*)", &res);
        claimed.push(cov.applied);
        faults.extend(cov.faults);

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

        let mut dit = Dit::<B>::new(&cfg, device);
        let res = dit
            .load_pytorch(&self.checkpoint)
            .expect("failed to read checkpoint");
        let cov = report("net.cfm.module.estimator.* (dit + wavenet)", &res);
        claimed.push(cov.applied);
        faults.extend(cov.faults);

        // The same reasoning as the timbre encoder's check below, applied to the
        // network that has the most ways to load perfectly and compute nonsense:
        // a wrong rotary convention, a swapped scale/shift or a mask that should
        // not be there all leave the tensor shapes intact. Two seconds of mel is
        // enough, and it keeps this harness free of an audio dependency.
        let frames = cfg.frame_rate() as usize * 2;
        let velocity = dit.forward(
            Tensor::<B, 3>::random([1, cfg.n_mels, frames], normal, device),
            Tensor::zeros([1, cfg.n_mels, frames], device),
            Tensor::from_floats([0.5], device),
            Tensor::<B, 2>::random([1, cfg.style_dim], normal, device),
            Tensor::<B, 3>::random([1, frames, cfg.hidden_dim], normal, device),
        );
        let values: Vec<f32> = velocity.into_data().to_vec().unwrap();
        // Finiteness is asserted on its own: a NaN out of a mis-built mask
        // reaches every mel band, and Burn's approximate comparisons treat NaN
        // as equal to NaN.
        println!(
            "  forward : {:?} finite={}, rms={:.4}",
            [1, cfg.n_mels, frames],
            values.iter().all(|v| v.is_finite()),
            (values.iter().map(|v| v * v).sum::<f32>() / values.len() as f32).sqrt(),
        );

        let mut regulator = InterpolateRegulator::<B>::new(&cfg, device);
        let res = regulator
            .load_pytorch(&self.checkpoint)
            .expect("failed to read checkpoint");
        let cov = report("net.length_regulator.module.*", &res);
        claimed.push(cov.applied);
        faults.extend(cov.faults);

        // CAMPPlus, if its checkpoint was named. It is deliberately optional:
        // `campplus_cn_common.bin` is a 28 MB file from somebody else's release
        // (`funasr/campplus`), so requiring it would make the map above
        // unreachable for anyone who only has the Seed-VC weights.
        if let Some(path) = &self.campplus {
            let cfg = CamPPlusConfig::default();
            let mut model = CamPPlus::<B>::new(&cfg, device);
            let res = model.load_pytorch(path).expect("failed to load CAMPPlus");
            // A separate file, so `report`'s `net.` filter says nothing here —
            // every key in it is this module's. What is left over instead is one
            // `num_batches_tracked` per norm: a training counter PyTorch stores
            // as a buffer and inference never reads.
            let (counters, real): (Vec<&String>, Vec<&String>) = res
                .unused
                .iter()
                .partition(|k| k.ends_with("num_batches_tracked"));
            println!(
                "\ncampplus_cn_common.bin\n  applied : {}\n  missing : {}\n  unused  : {} ({} \
                 num_batches_tracked, expected; {} genuinely unused)\n  errors  : {}",
                res.applied.len(),
                res.missing.len(),
                res.unused.len(),
                counters.len(),
                real.len(),
                res.errors.len(),
            );
            for (name, why) in &res.missing {
                println!("      MISSING {name}  ({why})");
            }
            for name in &real {
                println!("      UNUSED {name}");
            }
            for e in &res.errors {
                println!("      ERROR {e:?}");
            }
            faults.extend(faults_for("campplus_cn_common.bin", &res, real.len()));

            // Same forward check as the style encoder's, for the same reason and
            // against the same failure: an encoder that ignores its input
            // converts every clip into one voice and looks perfectly healthy.
            // Note the axis order — CAMPPlus takes frames before bins.
            let a = Tensor::<B, 3>::random([1, 240, cfg.feat_dim], normal, device);
            let tilt = Tensor::<B, 1, Int>::arange(0..cfg.feat_dim as i64, device)
                .float()
                .reshape([1, 1, cfg.feat_dim]);
            let b = Tensor::<B, 3>::random([1, 240, cfg.feat_dim], normal, device) * tilt;
            let embed = |x| -> Vec<f32> { model.forward(x).into_data().to_vec().unwrap() };
            let (va, va_again, vb) = (embed(a.clone()), embed(a), embed(b));
            let cosine = dot(&va, &vb) / (dot(&va, &va) * dot(&vb, &vb)).sqrt();
            println!(
                "  forward : finite={}, repeatable={}, cos(two references)={cosine:.4}",
                va.iter().all(|x| x.is_finite()),
                va == va_again,
            );

            // A stronger claim than "the output moves": a *speaker* encoder has
            // to key on the spectral envelope and ignore what is under it. Two
            // independent noise draws shaped by the same envelope must land
            // closer together than either does to a third under a different
            // envelope — which no amount of weight coverage can tell you, and
            // which a transposed axis or a softmax over the wrong dimension
            // would destroy.
            //
            // Evidence rather than proof: these are not real filterbanks, so a
            // narrow margin here would be as likely to be the input distribution
            // as the port.
            let bins = Tensor::<B, 1, Int>::arange(0..cfg.feat_dim as i64, device).float();
            let shape = [1, 1, cfg.feat_dim];
            let rising = bins.clone().div_scalar(40.0).add_scalar(1.0).reshape(shape);
            let falling = bins.div_scalar(-40.0).add_scalar(3.0).reshape(shape);
            let noise = || Tensor::<B, 3>::random([1, 240, cfg.feat_dim], normal, device);
            let same_a = embed(noise() * rising.clone());
            let same_b = embed(noise() * rising);
            let other = embed(noise() * falling);
            let cos = |x: &[f32], y: &[f32]| dot(x, y) / (dot(x, x) * dot(y, y)).sqrt();
            println!(
                "  envelope: cos(same envelope, different noise)={:.4} vs cos(different \
                 envelope)={:.4}",
                cos(&same_a, &same_b),
                cos(&same_a, &other),
            );
        } else {
            println!(
                "\ncampplus_cn_common.bin\n  not checked — pass it as a second argument. It is \
                 what the transformer is really conditioned on, so a run without it says \
                 nothing about the timbre path."
            );
        }

        // Reported for completeness rather than because anything runs it:
        // upstream's `build_model` does not construct this subtree at all, so it
        // is a residue of the training script. See `burn_seedvc::vq`.
        let mut vq = ResidualVq::<B>::new(&VqConfig::default(), device);
        let res = vq
            .load_pytorch(&self.checkpoint)
            .expect("failed to read checkpoint");
        let cov = report("net.vq.module.quantizers.*", &res);
        claimed.push(cov.applied);
        faults.extend(cov.faults);

        // **The check no single block can make.** Each loader's remap strips
        // only its own prefix, so every block files the other ~280 tensors as
        // "belonging to other modules" without ever asking whether some module
        // actually claimed them. A subtree nobody ported is invisible in all
        // four `unused` counts and shows up only here, as a shortfall of exactly
        // its own size. Computed from the file rather than written down, so a
        // different checkpoint is checked against itself.
        let total: usize = claimed.iter().sum();
        println!(
            "\nΣ applied over the four blocks reading this checkpoint : {total} of {} tensors",
            tensors.len()
        );
        if total != tensors.len() {
            let short = tensors.len() - total;
            println!(
                "    UNCLAIMED {short} tensors are in this checkpoint and no module took them — \
                 the prefix table above says which"
            );
            faults.push(format!(
                "{short} of {} checkpoint tensors nothing claimed",
                tensors.len()
            ));
        }

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
            // Not added to `claimed`: this is a different file, so its tensors
            // are not part of the Seed-VC checkpoint's sum.
            faults.extend(report("bigvgan (its own checkpoint)", &res).faults);

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

        // The verdict. Printing a fault and exiting 0 is what let a load that
        // reported 90 `MISSING` lines pass for a clean run, so `--strict` is the
        // form anything automated should use; without it the numbers above are
        // still all there, and reading them is the caller's job.
        if faults.is_empty() {
            println!("\nall blocks clean");
        } else {
            println!("\n{} fault(s):", faults.len());
            for f in &faults {
                println!("  - {f}");
            }
            if self.strict {
                std::process::exit(1);
            }
            println!("note: exiting 0 anyway — pass --strict to make this an exit code");
        }
    }
}

/// Pull `--<flag> <value>` (or `--<flag>=<value>`) out of the arguments.
///
/// The two extra checkpoints are named rather than positional because they are
/// independent: either can be given without the other, which a second and third
/// position could not express.
fn take_value(args: &mut Vec<String>, flag: &str) -> Option<String> {
    let eq = format!("--{flag}=");
    if let Some(i) = args.iter().position(|a| a.starts_with(&eq)) {
        return Some(args.remove(i)[eq.len()..].to_string());
    }
    let name = format!("--{flag}");
    let i = args.iter().position(|a| *a == name)?;
    args.remove(i);
    if i >= args.len() {
        eprintln!("error: --{flag} needs a path");
        std::process::exit(2);
    }
    Some(args.remove(i))
}

fn main() {
    let (backend, mut args) = common::parse_args();
    let campplus = take_value(&mut args, "campplus");
    let bigvgan = take_value(&mut args, "bigvgan");
    // A bare switch, so it cannot go through `take_value` — that one consumes
    // the following argument, which here is the checkpoint.
    let strict = args.iter().any(|a| a == "--strict");
    args.retain(|a| a != "--strict");
    let Some(checkpoint) = args.first() else {
        eprintln!(
            "usage: load [--backend ndarray|cuda|tch] [--strict] <checkpoint> \
             [--campplus campplus_cn_common.bin] [--bigvgan bigvgan_generator.pt]"
        );
        std::process::exit(2);
    };
    println!("backend : {backend}");
    common::run_on(
        backend,
        Load {
            checkpoint: checkpoint.clone(),
            campplus,
            bigvgan,
            strict,
        },
    );
}
