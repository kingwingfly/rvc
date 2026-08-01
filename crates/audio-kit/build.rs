//! Say something useful when ffmpeg is unpacked in the project but not declared.
//!
//! The counterpart of `crates/rvc-core/build.rs`, which does this for LibTorch.
//! `ffmpeg-sys-next` reads `FFMPEG_DIR` and otherwise asks `pkg-config`; it does
//! not look in the project directory, so a `./ffmpeg` sitting right there is
//! invisible to it and the build fails with a pkg-config error naming half a
//! dozen `.pc` files. That is a long way from "you have it, say so".
//!
//! Deliberately narrow: this fires **only** when an unpacked tree is found and
//! `FFMPEG_DIR` is unset. Every other case — a system install, vcpkg, a
//! cross-compile — is left to `ffmpeg-sys-next`, which is the authority on what
//! it can actually use. Probing pkg-config here to second-guess it would risk
//! failing a build that was about to work.

use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-env-changed=FFMPEG_DIR");
    if std::env::var_os("FFMPEG_DIR").is_some() {
        return;
    }

    // `crates/audio-kit` -> the workspace root, where an unpacked ./ffmpeg lands.
    let root = PathBuf::from(std::env::var_os("CARGO_MANIFEST_DIR").unwrap())
        .join("../..")
        .canonicalize()
        .unwrap_or_default()
        .join("ffmpeg");

    if root.join("lib").is_dir() && root.join("include").is_dir() {
        panic!(
            "Found ffmpeg at {}, but the build cannot use it until you say so:\n\n  \
             export FFMPEG_DIR={}\n\n\
             Rename or remove that directory to build against the system \
             libraries instead.\n",
            root.display(),
            root.display(),
        );
    }
}
