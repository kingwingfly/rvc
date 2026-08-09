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
//! [`denoise::file`]). Driving the batch — tolerating a file that will not
//! decode, tallying what happened — belongs to the caller, because what is
//! worth reporting differs per stage: `clip` counts clips against input
//! duration, `denoise` writes exactly one file per input.

pub mod clip;
pub mod denoise;
mod input;

pub use input::{AUDIO_EXTS, InputFile, decode_mono, plan};
