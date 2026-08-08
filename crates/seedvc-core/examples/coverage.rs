//! Load Seed-VC's checkpoints and report coverage, all four or any subset.
//!
//! `burn-seedvc`'s own `load` example checks each module against the file it
//! comes from; this one checks the thing above them — that
//! [`seedvc_core::model::BurnModel::load`] finds all four files, applies every
//! one, and refuses a bad one. It is the engine's stand-in for a test that cannot
//! be written: none of these weights can be committed.
//!
//! Measured on `--backend tch --device cpu`, and the numbers to expect:
//!
//! | module | flag | applied | missing | unused |
//! |---|---|---|---|---|
//! | the transformer | `--dit` | 255 | 0 | 47 (the checkpoint's other modules) |
//! | the length regulator | `--dit` | 22 | 0 | 288 (likewise) |
//! | the timbre encoder | `--campplus` | 815 | 0 | 122 (`num_batches_tracked`, a training counter) |
//! | the vocoder | `--bigvgan` | 783 | 0 | 0 |
//! | the content encoder | `--content` | 187 | 0 | 342 (Whisper's decoder, plus norm aliases) |
//!
//! **Only `missing` is a verdict, and `applied` is the gate `unused` cannot be.**
//! Every `unused` above is expected and none of them is checkable from inside the
//! loader — the Seed-VC `.pth` holds five modules, so loading any one leaves the
//! others over, and Whisper's decoder is deleted by upstream for the same reason.
//! `errors` is asked *before* `missing`, because `burn-store` derives `missing` as
//! `visited && !applied && !skipped && !errored`: a tensor whose shape did not
//! match is dropped from both lists, so a checkpoint that failed to apply
//! otherwise reads as full coverage. See `covered` in `model.rs`, which is the
//! same rule. Pinning `applied` to the recorded count is what closes the hole
//! `unused` leaves: drop a submodule and `applied` falls while `missing` stays 0
//! and the extra tensors are absorbed silently by `unused`.
//!
//! **Every flag is independently optional.** A machine that holds three of the
//! four checkpoints can check three; whatever is left out is reported as skipped
//! rather than refusing the whole run, which is what the two-flag form below used
//! to do. Giving none of them is the one error, since there is then nothing to
//! check.
//!
//! ```text
//! cargo run -p seedvc-core --features tch --example coverage -- \
//!     --dit dit.pth --campplus campplus_cn_common.bin \
//!     --bigvgan bigvgan_generator.pt --content whisper-small/model.safetensors
//! ```
//!
//! With **all four**, the modules are additionally loaded a second time through
//! [`seedvc_core::load`] itself — the only way to check the engine's own loader,
//! which returns a model rather than the five reports. That is deliberate
//! duplication of the five loads and the one thing here that can drift: a module
//! added to `BurnModel::load` and not to this file would go unchecked *by the
//! per-module pass*, while the whole-loader pass would still exercise it.
//!
//! Build it with an isolated `CARGO_TARGET_DIR` if anything else is building in
//! a sibling worktree — example binaries are not hashed per checkout, and the
//! one that runs is whichever landed last.

use std::error::Error;
use std::path::{Path, PathBuf};

use burn::tensor::backend::Backend as BurnBackend;
use burn_kit::{ApplyResult, DeviceSpec};
use burn_seedvc::campplus::{CamPPlus, CamPPlusConfig};
use burn_seedvc::content::ContentEncoder;
use burn_seedvc::{BigVgan, BigVganConfig, Dit, InterpolateRegulator, SeedVcConfig};
use cli_kit::Backend;
use seedvc_core::{ModelPaths, load};

fn flag(args: &mut Vec<String>, name: &str) -> Option<String> {
    let i = args.iter().position(|a| a == &format!("--{name}"))?;
    args.remove(i);
    Some(args.remove(i))
}

/// The checkpoints the caller named, in the order [`BurnModel::load`] reads them.
///
/// Four `Option`s rather than a [`ModelPaths`]: that type is four `&Path`s and
/// the loader behind it reads all of them, which is exactly why a subset cannot
/// go through it.
struct Given {
    dit: Option<PathBuf>,
    campplus: Option<PathBuf>,
    bigvgan: Option<PathBuf>,
    content: Option<PathBuf>,
}

/// Print one module's coverage and say whether it is the load recorded above.
///
/// `dead_code` because a build with **no** compute-backend feature keeps only
/// the fallback arm of [`run_piecewise`], so nothing reaches this — which is the
/// same reason `seedvc-core`'s own `load` has unused parameters there. It is
/// live on every build that can actually run a model.
#[allow(dead_code)]
fn report(what: &str, recorded: usize, result: Result<ApplyResult, Box<dyn Error>>) -> bool {
    let res = match result {
        Ok(res) => res,
        Err(e) => {
            eprintln!("{what:24} FAILED TO READ: {e}");
            return false;
        }
    };
    println!(
        "{what:24} applied {:4}  missing {:3}  unused {:4}  errors {}",
        res.applied.len(),
        res.missing.len(),
        res.unused.len(),
        res.errors.len(),
    );

    let mut failures: Vec<String> = Vec::new();
    if !res.errors.is_empty() {
        failures.push(format!(
            "{} checkpoint tensor(s) could not be applied ({:?}) — a shape mismatch is dropped \
             from `missing` too, so this is the question that has to come first",
            res.errors.len(),
            res.errors[0],
        ));
    }
    if !res.missing.is_empty() {
        failures.push(format!(
            "{} model parameter(s) had no checkpoint tensor",
            res.missing.len()
        ));
    }
    if res.applied.len() != recorded {
        failures.push(format!(
            "applied {} where {recorded} is recorded — `unused` is unbounded for this module, so \
             this count is the whole gate",
            res.applied.len()
        ));
    }
    for why in &failures {
        eprintln!("    - {why}");
    }
    failures.is_empty()
}

/// Load each module the caller named, on one concrete Burn backend.
///
/// `dead_code` for the reason [`report`] is: with no backend feature on, every
/// arm that instantiates it is `#[cfg]`ed away.
#[allow(dead_code)]
fn piecewise<B: BurnBackend>(given: &Given, device: &B::Device) -> bool {
    let cfg = SeedVcConfig::uvit_whisper_small_wavenet();
    let mut ok = true;

    match &given.dit {
        // One file, two modules — the transformer and the length regulator are
        // the pair that share a checkpoint, which is why `--dit` checks both.
        Some(path) => {
            let mut dit = Dit::<B>::new(&cfg, device);
            ok &= report("the transformer", 255, dit.load_pytorch(path));
            let mut regulator = InterpolateRegulator::<B>::new(&cfg, device);
            ok &= report("the length regulator", 22, regulator.load_pytorch(path));
        }
        None => println!(
            "the transformer          skipped (no --dit)\nthe length regulator     skipped (no --dit)"
        ),
    }

    match &given.campplus {
        Some(path) => {
            let mut campplus = CamPPlus::<B>::new(&CamPPlusConfig::default(), device);
            ok &= report("the timbre encoder", 815, campplus.load_pytorch(path));
        }
        None => println!("the timbre encoder       skipped (no --campplus)"),
    }

    match &given.bigvgan {
        Some(path) => {
            let mut vocoder = BigVgan::<B>::new(&BigVganConfig::v2_22khz_80band_256x(), device);
            ok &= report("the vocoder", 783, vocoder.load_pytorch(path));
        }
        None => println!("the vocoder              skipped (no --bigvgan)"),
    }

    match &given.content {
        Some(path) => {
            let mut content = ContentEncoder::<B>::new(device);
            ok &= report("the content encoder", 187, content.load_safetensors(path));
        }
        None => println!("the content encoder      skipped (no --content)"),
    }

    ok
}

/// Erase the compute backend, exactly as `seedvc_core::backend::load` does.
///
/// The engine's own loader cannot be reused here because it takes all four paths
/// at once, so the dispatch has to be repeated — and it is the *only* part that
/// is repeated, since each arm just names a concrete Burn backend.
fn run_piecewise(given: &Given, backend: Backend, device: DeviceSpec) -> bool {
    match backend {
        #[cfg(feature = "tch")]
        Backend::Tch => piecewise::<burn::backend::LibTorch<f32>>(
            given,
            &burn_kit::libtorch_device(device).expect("--device"),
        ),
        #[cfg(feature = "cuda")]
        Backend::Cuda => piecewise::<burn::backend::Cuda>(
            given,
            &burn_kit::cuda_device(device).expect("--device"),
        ),
        #[cfg(feature = "wgpu")]
        Backend::Wgpu => piecewise::<burn::backend::Wgpu>(
            given,
            &burn_kit::wgpu_device(device).expect("--device"),
        ),
        // Reachable two ways: a `--no-default-features` build whose arm above was
        // `#[cfg]`ed away, and `auto`/`onnx`, which `resolve` has already settled
        // or refused before this is called.
        #[allow(unreachable_patterns)]
        other => {
            let _ = (given, device);
            eprintln!("{}", other.unavailable());
            false
        }
    }
}

fn main() {
    cli_kit::init_logging(None);
    let mut args: Vec<String> = std::env::args().skip(1).collect();

    let backend = flag(&mut args, "backend").unwrap_or_else(|| "tch".into());
    let device = flag(&mut args, "device").unwrap_or_else(|| "cpu".into());
    let given = Given {
        dit: flag(&mut args, "dit").map(PathBuf::from),
        campplus: flag(&mut args, "campplus").map(PathBuf::from),
        bigvgan: flag(&mut args, "bigvgan").map(PathBuf::from),
        content: flag(&mut args, "content").map(PathBuf::from),
    };
    if given.dit.is_none()
        && given.campplus.is_none()
        && given.bigvgan.is_none()
        && given.content.is_none()
    {
        eprintln!(
            "error: nothing to check — name at least one checkpoint.\n\
             usage: coverage [--backend tch] [--device cpu] [--dit <ckpt.pth>] \
             [--campplus <campplus_cn_common.bin>] [--bigvgan <bigvgan_generator.pt>] \
             [--content <whisper-small/model.safetensors>]"
        );
        std::process::exit(2);
    }

    // `--backend onnx` first, and before anything is read: without an export
    // directory it must fail on the argument rather than on a missing file, and
    // it must say why — the graphs are on disk only where `--onnx` points. The
    // paths handed to it are deliberately nonexistent, which is what makes the
    // "before any file is opened" half of that checkable at all.
    let nowhere = Path::new("/nonexistent");
    let probe = ModelPaths {
        dit: nowhere,
        campplus: nowhere,
        bigvgan: nowhere,
        content: nowhere,
    };
    match load(&probe, Backend::Onnx, DeviceSpec::Auto) {
        Ok(_) => panic!("--backend onnx was accepted without --onnx"),
        Err(e) => println!("--backend onnx (no --onnx): {e}\n"),
    }

    // With an export directory the same request is accepted — that is what
    // `--onnx <dir>` does. The coverage example cannot open a real export (no
    // weights can be committed), so the resolver accepting the pair is the part
    // that is checkable here.
    match seedvc_core::backend::resolve(Backend::Onnx, true) {
        Ok(Backend::Onnx) => println!("--backend onnx + --onnx <dir>: resolves to onnx\n"),
        Ok(other) => panic!("--backend onnx + --onnx resolved to {other}"),
        Err(e) => panic!("--backend onnx + --onnx refused: {e}"),
    }

    let backend = <Backend as clap::ValueEnum>::from_str(&backend, true).expect("--backend");
    let device = cli_kit::parse_device(&device).expect("--device");
    // `auto` is settled once, here, so the per-module pass and the whole-loader
    // pass below cannot land on two different backends.
    let backend = seedvc_core::backend::resolve(backend, false).expect("--backend");
    println!("== per module, on {backend}/{device} ==");
    let mut ok = run_piecewise(&given, backend, device);

    // The engine's own loader, which is the thing this example exists to check.
    // Only reachable with all four, because `ModelPaths` is four `&Path`s.
    match (&given.dit, &given.campplus, &given.bigvgan, &given.content) {
        (Some(dit), Some(campplus), Some(bigvgan), Some(content)) => {
            let paths = ModelPaths {
                dit,
                campplus,
                bigvgan,
                content,
            };
            match load(&paths, backend, device) {
                Ok(model) => println!(
                    "\nloaded on {backend}/{device}: {} Hz, {} mels, hop {}",
                    model.config().sample_rate,
                    model.config().n_mels,
                    model.config().hop_length,
                ),
                Err(e) => {
                    eprintln!("\nfailed: {e}");
                    ok = false;
                }
            }
        }
        _ => println!("\nseedvc_core::load: skipped (it takes all four checkpoints at once)"),
    }

    if !ok {
        std::process::exit(1);
    }
}
