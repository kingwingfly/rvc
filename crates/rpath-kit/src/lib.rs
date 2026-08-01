//! Bake the linked libraries' search paths into an executable, so running one
//! of this toolkit's binaries needs no environment.
//!
//! ffmpeg and LibTorch are **linked**, not dlopened, so the loader resolves them
//! before `main` runs. A missing `libavcodec.so` or `libtorch.so` therefore
//! aborts inside `ld.so` on *every* invocation, including ones that touch
//! neither — `stt completions` fails the same way a transcription does. Nothing
//! inside the program can help, and neither can the link *search* path that
//! `ffmpeg-sys-next` and `torch-sys` emit: that is consumed by the linker at
//! build time and never read at run time. Only a path compiled into the
//! executable is.
//!
//! (ONNX Runtime is the other case and needs none of this: it is dlopened on
//! first use, so `ORT_DYLIB_PATH` is a run-time choice and a run that never
//! reaches it never looks for it.)
//!
//! Called from each `*-cli` crate's `build.rs`. The *emission* is still
//! per-executable — an rpath is a property of one linked binary — but the
//! search order is decided here once. Four copies of it had already been
//! written by hand, and a search order that differs between `rvc` and `stt` is
//! the kind of difference nobody decides on and everybody has to learn.

use std::path::PathBuf;

/// Emit the `-Wl,-rpath` link argument for the binaries of the calling crate.
///
/// Reads the environment of the build script, so the `tch` feature it consults
/// is the calling crate's.
pub fn emit() {
    for var in ["LIBTORCH", "LIBTORCH_LIB", "FFMPEG_DIR"] {
        println!("cargo:rerun-if-env-changed={var}");
    }

    let env = |k: &str| std::env::var_os(k).map(PathBuf::from);

    let mut rpaths = Vec::new();
    // Only when the backend that links it is compiled in. Naming a directory
    // nothing in the binary needs is harmless, but it reads as a dependency.
    if std::env::var_os("CARGO_FEATURE_TCH").is_some() {
        let built = env("LIBTORCH_LIB").or_else(|| env("LIBTORCH").map(|r| r.join("lib")));
        rpaths.extend(search_path("libtorch", built));
    }
    // ffmpeg is never optional: every engine decodes, resamples and writes audio
    // through it, so this applies to every binary and every feature set.
    rpaths.extend(search_path(
        "ffmpeg",
        env("FFMPEG_DIR").map(|r| r.join("lib")),
    ));

    println!("cargo:rustc-link-arg-bins=-Wl,-rpath,{}", rpaths.join(":"));
}

/// Where the loader should look for one library's shared objects, most specific
/// first: what this was built against, then `./<name>/lib` where the binary is
/// *run*, then `<name>/lib` beside the binary itself, then one level up for a
/// `bin/` layout. System directories come last and for free, out of
/// `ld.so.cache`, so a distribution's own package keeps working untouched.
///
/// A bare relative entry resolves against the working directory and `$ORIGIN`
/// against the executable's own, which is what lets a `./ffmpeg` in a project
/// directory keep working after the binary moves — and lets a self-contained
/// tree be shipped beside the binary with no environment at all.
fn search_path(name: &str, built_against: Option<PathBuf>) -> Vec<String> {
    built_against
        // The build machine's own install is the most specific answer and the
        // one whoever compiled this asked for. Dropped when it has since gone,
        // so a stale absolute path does not sit in front of the others.
        .filter(|lib| lib.is_dir())
        .map(|lib| lib.display().to_string())
        .into_iter()
        .chain([
            format!("{name}/lib"),
            format!("$ORIGIN/{name}/lib"),
            format!("$ORIGIN/../{name}/lib"),
        ])
        .collect()
}
