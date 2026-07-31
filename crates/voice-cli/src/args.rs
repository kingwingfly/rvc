//! `voice` argument definitions.
//!
//! No engine's flags are defined here. Each `*-cli` crate exports the whole of
//! its own command tree — [`rvc_cli::args::RvcCli`], [`stt_cli::SttCli`],
//! [`tts_cli::TtsCli`] — and this file only nests them, so `voice rvc convert`
//! and `rvc convert` are one definition worn two ways.

use clap::{Parser, Subcommand};
use rvc_cli::args::{CompletionsArgs, ModelsArgs, RvcCli};

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
    /// stdout; `convert`, `train` and `preprocess` beside it.
    // Every engine's tree is boxed because each carries its trainer's arguments,
    // and an enum is as large as its biggest variant.
    Rvc(Box<RvcCli>),
    /// Speech recognition: f32le mono PCM @16 kHz on stdin, text on stdout.
    Stt(Box<SttCli>),
    /// Speech synthesis: text on stdin, f32le mono PCM on stdout; `train`
    /// beside it.
    Tts(Box<TtsCli>),
    /// Download/prefetch shared model assets from Hugging Face.
    Models(ModelsArgs),
    /// Print a shell completion script (bash, zsh, fish, powershell, elvish).
    Completions(CompletionsArgs),
}
