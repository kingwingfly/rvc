//! Corpus preparation: the stages that run *before* an engine ever sees the
//! audio.
//!
//! A trainer eats a directory of clean per-utterance clips, and a recording is
//! almost never that. Getting from one to the other is a sequence of
//! independent steps — slice on silence, strip hiss, and more to come — each of
//! which reads audio files and writes audio files, and none of which knows what
//! will be trained on the result.
//!
//! That is why this is its own crate rather than a subcommand each engine
//! grows. It used to be one shared `preprocess` subcommand (`preprocess-kit`)
//! that `rvc` and `tts` both hosted, which was already the right instinct — one
//! definition, so the slicing could not differ between the two engines — but it
//! could only ever hold *one* stage, because the subcommand and the stage were
//! the same name. A second stage had nowhere to go but a second subcommand on
//! two engines that neither of them is about.
//!
//! # Shape
//!
//! [`plan`] turns whatever paths the user named into the batch to process:
//! directories expanded to their audio files, sorted, de-duplicated, and each
//! given an output base name that is **unique across the batch** — so
//! `a/take.wav` and `b/take.wav` do not overwrite each other's output. Every
//! stage takes that list and writes into one output directory, which keeps the
//! stages composable: the output of one is a legitimate input to the next.
//!
//! Each stage is one module with one per-file function ([`clip::file`],
//! [`denoise::file`], [`separate::file`]). Driving the batch — tolerating a file
//! that will not decode, tallying what happened — belongs to the caller, because
//! what is worth reporting differs per stage: `clip` counts clips against input
//! duration, `denoise` writes exactly one file per input.
//!
//! # The first stage that runs a model
//!
//! [`separate`] is it, and it is why this crate now has backend features at
//! all. The stage itself carries none of them — the
//! [`Separator`](separate::Separator) boundary, the overlap-add and the file
//! driver compile and are tested with no backend enabled — and inside
//! [`backend`] only the arms that name a Burn backend are behind
//! `cuda`/`tch`/`wgpu`. That split is what keeps the seam arithmetic under
//! `cargo test` on a machine with no checkpoint and no GPU, and it is what lets
//! a build with no backend at all still *refuse* a request with a reason
//! instead of failing to have the function.

pub mod clip;
pub mod denoise;
mod input;
pub mod separate;

// Only the loader names a compute backend. Keeping it here rather than in the
// CLI is `seedvc-core`'s trade, made for the same reason: the erasure has to
// name a backend, so the module that performs it depends on `cli-kit` for the
// one `--backend` enum the workspace shares.
pub mod backend;

pub use input::{AUDIO_EXTS, InputFile, decode_mono, plan};
