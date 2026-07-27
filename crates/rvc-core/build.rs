//! Fail the `tch` build early when LibTorch isn't configured.
//!
//! `burn-tch` pulls `tch` with `download-libtorch` on, and cargo features are
//! additive, so we can't switch it off. Left alone it silently downloads a
//! **CPU-only** LibTorch and every conversion quietly runs on the CPU. Better to
//! stop and name the variable.

use std::path::{Path, PathBuf};

/// The version `tch 0.22` generates bindings for. Anything else is likely to
/// fail deep in the C++ build: PyTorch 2.13 dropped `torch::align_tensors`,
/// which those bindings call.
const WANT_VERSION: &str = "2.9.0";

fn main() {
    for var in ["LIBTORCH", "LIBTORCH_LIB", "LIBTORCH_USE_PYTORCH"] {
        println!("cargo:rerun-if-env-changed={var}");
    }
    if std::env::var_os("CARGO_FEATURE_TCH").is_none()
        // Defers to python; torch-sys checks that path itself.
        || std::env::var_os("LIBTORCH_USE_PYTORCH").is_some()
    {
        return;
    }

    let Some(root) = std::env::var_os("LIBTORCH").map(PathBuf::from).or_else(|| {
        Path::new("/usr/lib/libtorch.so")
            .exists()
            .then(|| "/usr".into())
    }) else {
        panic!("{MISSING}");
    };

    // Advisory only: torch-sys is the authority on what it can use, and an
    // unfamiliar layout isn't proof of a broken install.
    if let Ok(v) = std::fs::read_to_string(root.join("build-version")) {
        let v = v.trim();
        if v.split_once('+').map_or(v, |(v, _)| v) != WANT_VERSION {
            println!(
                "cargo:warning=LIBTORCH is LibTorch {v}, but tch 0.22 needs {WANT_VERSION} \
                 — expect the C++ build to fail on removed APIs"
            );
        }
    }
    let lib = root.join("lib");
    if lib.join("libtorch.so").exists() && !lib.join("libtorch_cuda.so").exists() {
        println!(
            "cargo:warning=LIBTORCH is a CPU-only build: `--backend tch` will refuse CUDA \
             devices. Re-download a cuNNN build if you have an NVIDIA GPU"
        );
    }
}

const MISSING: &str = "\
LibTorch not found, and the `tch` backend needs it. Download it once (~3 GB,
CUDA-specific, so the choice is yours) and point LIBTORCH at it:

  wget https://download.pytorch.org/libtorch/cu126/libtorch-shared-with-deps-2.9.0%2Bcu126.zip
  unzip libtorch-shared-with-deps-2.9.0+cu126.zip     # -> ./libtorch
  export LIBTORCH=$PWD/libtorch

Swap cu126 for the CUDA build matching your driver (cu118/cu121/cu124/cu126/cu128).
The version must be 2.9.0. Or build without it:

  cargo build --no-default-features --features cuda,wgpu
";
