//! Bake the linked libraries' search paths into `tts` so running it needs no
//! environment. All of it is `rpath-kit`, which explains why.
//!
//! Every binary that links them needs its own call: an rpath is a property of
//! one linked executable, not of the workspace.

fn main() {
    rpath_kit::emit();
}
