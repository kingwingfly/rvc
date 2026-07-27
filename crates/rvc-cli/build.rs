//! Make the built `rvc` find LibTorch by itself at run time.
//!
//! Unlike ONNX Runtime (dlopened via `ORT_DYLIB_PATH`), LibTorch is linked, so
//! the loader — not us — resolves it, and it does that *before* `main`. Baking
//! the search paths in at link time is what keeps `LD_LIBRARY_PATH` out of the
//! user's way. `torch-sys` only emits `rustc-link-search`, which the linker uses
//! and the loader never sees, so this is on us.

use std::path::PathBuf;

fn main() {
    for var in ["LIBTORCH", "LIBTORCH_LIB"] {
        println!("cargo:rerun-if-env-changed={var}");
    }
    if std::env::var_os("CARGO_FEATURE_TCH").is_none() {
        return;
    }

    // Tried in order at load time. `$ORIGIN` is the binary's own directory; a
    // plain relative path is resolved against the working directory, which is
    // what makes `./libtorch` next to your project just work.
    let mut rpaths = vec![
        "$ORIGIN/libtorch/lib".to_string(),
        "$ORIGIN/../libtorch/lib".to_string(),
        "libtorch/lib".to_string(),
    ];

    // The build machine's own install, so `cargo run` works from anywhere.
    let lib = std::env::var_os("LIBTORCH_LIB")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("LIBTORCH").map(|r| PathBuf::from(r).join("lib")));
    if let Some(lib) = lib.filter(|l| l.is_dir()) {
        rpaths.insert(0, lib.display().to_string());
    }

    println!("cargo:rustc-link-arg-bins=-Wl,-rpath,{}", rpaths.join(":"));
}
