//! Load `MDX23C-8KFFT-InstVoc_HQ.ckpt` and report weight coverage.
//!
//! The crate's stand-in for a unit test on the module tree: real published
//! weights have to map onto it with nothing missing. For that checkpoint the
//! answer is **319 applied / 0 missing / 0 unused**, identical on every
//! backend — a backend that changes the counts is a bug.
//!
//! Zero unused is worth stating rather than glossing: the norms are
//! `InstanceNorm2d`, so unlike every `BatchNorm`-based port in this workspace
//! there are no running statistics and no `num_batches_tracked` counters, and
//! there is therefore nothing in this file a reader has to talk themselves out
//! of. A non-zero `unused` here means the checkpoint is a different MDX23C
//! variant, not that a training tensor turned up.
//!
//! **Coverage cannot catch a wrong formula.** This network has two specific
//! ways of being wrong that a perfect triple will never see: the U-net runs
//! with *time* as the image height and *frequency* as its width, and the norms
//! are instance norms rather than batch norms in evaluation mode. Either
//! mistake loads at 100%, produces finite output, and separates nothing.
//! `examples/separate` is the check for that, and `stft`'s round-trip tests are
//! the check for the transform around it.
//!
//! Usage:
//! `cargo run -p burn-mdx --example load -- [--backend ndarray|cuda|tch] [--strict] <ckpt>`

#[path = "common/mod.rs"]
mod common;

use burn::tensor::backend::Backend;
use burn_mdx::{MdxConfig, TfcTdfNet};

struct Load {
    weights: String,
    strict: bool,
}

impl common::Job for Load {
    fn run<B: Backend>(self, device: &B::Device) {
        let cfg = MdxConfig::mdx23c_8k_instvoc_hq();
        let mut model = TfcTdfNet::<B>::new(&cfg, device);

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
        println!(
            "unused  : {}  (checkpoint tensors no param claimed)",
            res.unused.len()
        );
        for name in res.unused.iter().take(20) {
            println!("    UNUSED {name}");
        }
        println!("errors  : {}", res.errors.len());
        for e in &res.errors {
            println!("    ERROR {e:?}");
        }

        if self.strict {
            // Ordered so the first failure is the informative one. `errors`
            // before `missing` because a shape mismatch also shows up as a
            // missing parameter and the shapes say *why*; `applied.is_empty()`
            // separately from `missing`, because a checkpoint whose every name
            // was remapped into nothing returns a clean `Ok` with zero applied
            // and zero missing — that has actually happened twice in this
            // workspace and is invisible to any threshold on the other two.
            assert!(res.errors.is_empty(), "{} load errors", res.errors.len());
            assert!(
                res.missing.is_empty(),
                "{} parameters had no checkpoint tensor",
                res.missing.len()
            );
            assert!(
                !res.applied.is_empty(),
                "nothing was applied — every checkpoint name missed the module tree"
            );
            // A predicate rather than a count: this checkpoint has no
            // training-only tensors at all, so *any* leftover name means the
            // file is a different variant and the config above is wrong for it.
            // Naming the leftovers is what makes that diagnosable.
            assert!(
                res.unused.is_empty(),
                "{} checkpoint tensors were not claimed, starting with {:?} — \
                 this is not `MDX23C-8KFFT-InstVoc_HQ`",
                res.unused.len(),
                res.unused.iter().take(5).collect::<Vec<_>>()
            );
        }

        // A loaded model that cannot run is a port that only looks finished, so
        // push one buffer through. Deliberately a short chunk rather than the
        // released 256 frames: this example defaults to `ndarray`, where even
        // 32 frames is several minutes, and the point here is that the shapes
        // compose — `separate` is where the arithmetic is checked.
        //
        // **Noise rather than silence.** Zeros make this vacuous: the head
        // multiplies by `first_conv`'s output, and a bias-free 1×1 convolution
        // of zeros is zero, so a silent input gives an exactly zero output
        // whatever the 319 tensors in between contain.
        let frames = 32;
        let spec = burn::tensor::Tensor::<B, 4>::random(
            [1, 2 * cfg.audio_channels, cfg.dim_f, frames],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            device,
        );
        let out = model.forward(spec);
        assert_eq!(
            out.dims(),
            [1, cfg.stems, 2 * cfg.audio_channels, cfg.dim_f, frames]
        );
        let v: Vec<f32> = out.into_data().to_vec().expect("f32 spectrum");
        // Checked separately from any approximate comparison: Burn's
        // `assert_approx_eq` treats NaN as equal to NaN, so a NaN would sail
        // through a value check.
        assert!(v.iter().all(|x| x.is_finite()), "output must be finite");
        let peak = v.iter().fold(0.0f32, |a, b| a.max(b.abs()));
        assert!(peak > 0.0, "noise in, silence out — the network did nothing");
        println!(
            "\nforward : {:?} peak={peak:.6}",
            [1, cfg.stems, 2 * cfg.audio_channels, cfg.dim_f, frames]
        );
    }
}

fn main() {
    let (backend, mut args) = common::parse_args();
    let strict = common::take_flag(&mut args, "--strict");
    let Some(weights) = args.first().cloned() else {
        eprintln!("usage: load [--backend ndarray|cuda|tch] [--strict] <MDX23C-*.ckpt>");
        std::process::exit(2);
    };
    println!("backend : {backend}");
    common::run_on(backend, Load { weights, strict });
}
