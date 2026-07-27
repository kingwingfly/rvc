//! Bake LibTorch's search path into `rvc` so running it needs no environment.
//!
//! ONNX Runtime is dlopened, so `ORT_DYLIB_PATH` can be a run-time choice.
//! LibTorch is linked, so the loader resolves it before `main` and only a
//! compiled-in rpath can help; `torch-sys` emits just a link search path, which
//! the loader never reads.

use std::path::PathBuf;

fn main() {
    for var in ["LIBTORCH", "LIBTORCH_LIB"] {
        println!("cargo:rerun-if-env-changed={var}");
    }
    if std::env::var_os("CARGO_FEATURE_TCH").is_none() {
        return;
    }

    // Search order, first match wins. A bare relative path resolves against the
    // working directory; `$ORIGIN` is the binary's own directory. The system
    // paths come last and for free, from ld.so.cache.
    let mut rpaths = vec![
        "libtorch/lib".to_string(),            // ./libtorch where you run
        "$ORIGIN/libtorch/lib".to_string(),    // ./libtorch beside the binary
        "$ORIGIN/../libtorch/lib".to_string(), // ../libtorch, for bin/ layouts
    ];
    // The build machine's own install goes first: it is the most specific answer
    // and the one the person who compiled this asked for.
    let built_against = std::env::var_os("LIBTORCH_LIB")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("LIBTORCH").map(|r| PathBuf::from(r).join("lib")));
    if let Some(lib) = built_against.filter(|l| l.is_dir()) {
        rpaths.insert(0, lib.display().to_string());
    }

    println!("cargo:rustc-link-arg-bins=-Wl,-rpath,{}", rpaths.join(":"));
}
