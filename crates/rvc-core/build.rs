//! Fail the `tch` build early, and helpfully, when LibTorch isn't configured.
//!
//! `burn-tch` pulls `tch` with `download-libtorch` on, and cargo features are
//! additive, so we can't switch it off. Left alone it silently downloads a
//! **CPU-only** LibTorch and every conversion quietly runs on the CPU.

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

    // `crates/rvc-core` -> the workspace root, where a downloaded ./libtorch lands.
    let project_root = PathBuf::from(std::env::var_os("CARGO_MANIFEST_DIR").unwrap())
        .join("../..")
        .canonicalize()
        .unwrap_or_default();
    let in_project = project_root.join("libtorch");

    let Some(root) = std::env::var_os("LIBTORCH")
        .map(PathBuf::from)
        // torch-sys only looks at LIBTORCH and /usr, so a project-root install
        // can't be picked up silently — but we can say so precisely.
        .or_else(|| in_project.join("lib").is_dir().then(|| in_project.clone()))
        .or_else(|| {
            Path::new("/usr/lib/libtorch.so")
                .exists()
                .then(|| "/usr".into())
        })
    else {
        panic!("{}", not_found());
    };

    if std::env::var_os("LIBTORCH").is_none() && root == in_project {
        panic!(
            "Found LibTorch at {}, but the build cannot use it until you say so:\n\n  \
             export LIBTORCH={}\n",
            root.display(),
            root.display()
        );
    }

    // Advisory from here down: torch-sys is the authority on what it can use.
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

fn not_found() -> String {
    format!(
        "LibTorch not found, and the `tch` backend needs it. It is ~3 GB and
CUDA-specific, so rvc never downloads it for you:

  wget https://download.pytorch.org/libtorch/cu126/libtorch-shared-with-deps-{WANT_VERSION}%2Bcu126.zip
  unzip libtorch-shared-with-deps-{WANT_VERSION}+cu126.zip     # -> ./libtorch
  export LIBTORCH=$PWD/libtorch

Match cuNNN to your driver (cu118/cu121/cu124/cu126/cu128); the version must be
{WANT_VERSION}. Searched: $LIBTORCH, ./libtorch at the project root, /usr.

Or build without the backend:

  cargo build --no-default-features --features cuda,wgpu
"
    )
}
