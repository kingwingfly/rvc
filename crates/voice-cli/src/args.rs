//! `voice` argument definitions.
//!
//! No engine's flags are defined here. Each `*-cli` crate exports the whole of
//! its own command tree — [`rvc_cli::args::RvcCli`], [`stt_cli::SttCli`],
//! [`tts_cli::TtsCli`], [`seedvc_cli::SeedVcCli`] — and this file only nests
//! them, so `voice rvc convert` and `rvc convert` are one definition worn two
//! ways.
//!
//! There is deliberately no top-level asset command. There used to be a `voice
//! models`, which announced itself as fetching "shared model assets" and in
//! fact fetched only voice conversion's two. Every engine has a `download` of
//! its own now, and nesting them is what says which engine's weights a
//! gigabyte is about to be spent on.

use clap::{Parser, Subcommand};
use rvc_cli::args::{CompletionsArgs, RvcCli};

use seedvc_cli::SeedVcCli;
use stt_cli::SttCli;
use tts_cli::TtsCli;

/// voice — speech toolkit: recognition, synthesis and voice conversion, each a
/// filter that composes in a pipe.
#[derive(Debug, Parser)]
#[command(name = "voice", version, about)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Debug, Subcommand)]
pub enum Command {
    /// Voice conversion: f32le mono PCM @16 kHz on stdin, converted PCM on
    /// stdout; `convert`, `train`, `preprocess` and `download` beside it.
    // Every engine's tree is boxed because each carries its trainer's arguments,
    // and an enum is as large as its biggest variant.
    Rvc(Box<RvcCli>),
    /// Speech recognition: f32le mono PCM @16 kHz on stdin, text on stdout;
    /// `download` beside it.
    Stt(Box<SttCli>),
    /// Speech synthesis: text on stdin, f32le mono PCM on stdout; `train`,
    /// `preprocess` and `download` beside it.
    Tts(Box<TtsCli>),
    /// Zero-shot voice conversion: f32le mono PCM @16 kHz on stdin, converted
    /// PCM @22.05 kHz on stdout; `convert` and `download` beside it. The voice
    /// comes from a reference clip, so there is no `train`.
    // Named explicitly because clap's kebab-case default would spell this
    // variant `seed-vc`, and the subcommand has to be the binary's own name.
    #[command(name = "seedvc")]
    SeedVc(Box<SeedVcCli>),
    /// Print a shell completion script (bash, zsh, fish, powershell, elvish).
    Completions(CompletionsArgs),
}
