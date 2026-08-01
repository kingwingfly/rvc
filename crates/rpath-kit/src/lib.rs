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
//! **Every entry is relative**, to the working directory or to `$ORIGIN`. The
//! build machine's `LIBTORCH` and `FFMPEG_DIR` say where to *link* against and
//! stop there: an absolute path from the machine that compiled a binary means
//! nothing on the machine that runs it, and baking one in makes `ldd` report a
//! stranger's directory layout. Where the loader honours a variable at run time
//! it is `LD_LIBRARY_PATH`, which glibc consults *before* any of this.
//!
//! Called from each `*-cli` crate's `build.rs`. The *emission* is still
//! per-executable — an rpath is a property of one linked binary — but the
//! search order is decided here once. Four copies of it had already been
//! written by hand, and a search order that differs between `rvc` and `stt` is
//! the kind of difference nobody decides on and everybody has to learn.

/// Emit the `-Wl,-rpath` link argument for the binaries of the calling crate.
///
/// Reads the environment of the build script, so the `tch` feature it consults
/// is the calling crate's.
pub fn emit() {
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_TCH");

    let mut rpaths = Vec::new();
    // Only when the backend that links it is compiled in. Naming a directory
    // nothing in the binary needs is harmless, but it reads as a dependency.
    if std::env::var_os("CARGO_FEATURE_TCH").is_some() {
        rpaths.extend(search_path("libtorch"));
    }
    // ffmpeg is never optional: every engine decodes, resamples and writes audio
    // through it, so this applies to every binary and every feature set.
    rpaths.extend(search_path("ffmpeg"));

    println!("cargo:rustc-link-arg-bins=-Wl,-rpath,{}", rpaths.join(":"));
}

/// Where the loader should look for one library's shared objects: `<name>/lib`
/// in the directory the binary is *run* from, then beside the binary itself,
/// then one level up for a `bin/` layout. System directories come after, for
/// free, out of `ld.so.cache` — so a distribution's own package keeps working
/// untouched and is what most machines will use.
///
/// A bare relative entry resolves against the working directory and `$ORIGIN`
/// against the executable's own, which is what lets a `./ffmpeg` in a project
/// directory keep working after the binary moves, and lets a self-contained
/// tree ship beside the binary with no environment at all.
fn search_path(name: &str) -> [String; 3] {
    [
        format!("{name}/lib"),
        format!("$ORIGIN/{name}/lib"),
        format!("$ORIGIN/../{name}/lib"),
    ]
}
