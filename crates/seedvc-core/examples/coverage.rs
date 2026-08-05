//! Load all four checkpoints through [`seedvc_core::load`] and report coverage.
//!
//! `burn-seedvc`'s own `load` example checks each module against the file it
//! comes from; this one checks the thing above them — that
//! [`seedvc_core::model::BurnModel::load`] finds all four files, applies every
//! one, and refuses a bad one. It is the engine's stand-in for a test that cannot
//! be written: none of these weights can be committed.
//!
//! Measured on `--backend tch --device cpu`, and the numbers to expect:
//!
//! | module | applied | missing | unused |
//! |---|---|---|---|
//! | the transformer | 255 | 0 | 47 (the checkpoint's other modules) |
//! | the length regulator | 22 | 0 | 288 (likewise) |
//! | the timbre encoder | 815 | 0 | 122 (`num_batches_tracked`, a training counter) |
//! | the vocoder | 783 | 0 | 0 |
//! | the content encoder | 187 | 0 | 342 (Whisper's decoder, plus norm aliases) |
//!
//! **Only `missing` is a verdict.** Every `unused` above is expected and none of
//! them is checkable from inside the loader — the Seed-VC `.pth` holds five
//! modules, so loading any one leaves the others over, and Whisper's decoder is
//! deleted by upstream for the same reason. See `covered` in `model.rs`.
//!
//! ```text
//! cargo run -p seedvc-core --features tch --example coverage -- \
//!     --dit dit.pth --campplus campplus_cn_common.bin \
//!     --bigvgan bigvgan_generator.pt --content whisper-small/model.safetensors
//! ```
//!
//! Build it with an isolated `CARGO_TARGET_DIR` if anything else is building in
//! a sibling worktree — example binaries are not hashed per checkout, and the
//! one that runs is whichever landed last.

use std::path::PathBuf;

use cli_kit::Backend;
use seedvc_core::{ModelPaths, load};

fn flag(args: &mut Vec<String>, name: &str) -> Option<String> {
    let i = args.iter().position(|a| a == &format!("--{name}"))?;
    args.remove(i);
    Some(args.remove(i))
}

fn main() {
    cli_kit::init_logging(None);
    let mut args: Vec<String> = std::env::args().skip(1).collect();

    let backend = flag(&mut args, "backend").unwrap_or_else(|| "tch".into());
    let device = flag(&mut args, "device").unwrap_or_else(|| "cpu".into());
    let paths: Vec<Option<PathBuf>> = ["dit", "campplus", "bigvgan", "content"]
        .iter()
        .map(|n| flag(&mut args, n).map(PathBuf::from))
        .collect();
    let [dit, campplus, bigvgan, content] = <[_; 4]>::try_from(paths).unwrap();
    let (Some(dit), Some(campplus), Some(bigvgan), Some(content)) =
        (dit, campplus, bigvgan, content)
    else {
        eprintln!(
            "usage: coverage [--backend tch] [--device cpu] --dit <ckpt.pth> \
             --campplus <campplus_cn_common.bin> --bigvgan <bigvgan_generator.pt> \
             --content <whisper-small/model.safetensors>"
        );
        std::process::exit(2);
    };

    // `--backend onnx` first, and before anything is read: without an export
    // directory it must fail on the argument rather than on a missing file, and
    // it must say why — the graphs are on disk only where `--onnx` points.
    let paths = ModelPaths {
        dit: &dit,
        campplus: &campplus,
        bigvgan: &bigvgan,
        content: &content,
    };
    match load(&paths, Backend::Onnx, burn_kit::DeviceSpec::Auto) {
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
    match load(&paths, backend, device) {
        Ok(model) => println!(
            "\nloaded on {backend}/{device}: {} Hz, {} mels, hop {}",
            model.config().sample_rate,
            model.config().n_mels,
            model.config().hop_length,
        ),
        Err(e) => {
            eprintln!("\nfailed: {e}");
            std::process::exit(1);
        }
    }
}
