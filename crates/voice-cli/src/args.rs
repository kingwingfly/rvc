//! `voice` argument definitions.
//!
//! The voice-conversion flags are not defined here — [`rvc_cli::args::RvcCommand`]
//! is the single definition, worn nested (`voice rvc convert`) instead of flat
//! (`rvc convert`). This file only adds the nesting and the engines the `rvc`
//! binary does not have.

use clap::{Parser, Subcommand};
use rvc_cli::args::{CompletionsArgs, ModelsArgs, RvcCommand};

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
    /// Voice conversion: retimbre audio into a trained target voice.
    // Boxed because `TrainArgs` alone dwarfs every other variant; see the same
    // note on `rvc_cli::args::Command`.
    Rvc {
        #[command(subcommand)]
        command: Box<RvcCommand>,
    },
    /// Download/prefetch shared model assets from Hugging Face.
    Models(ModelsArgs),
    /// Print a shell completion script (bash, zsh, fish, powershell, elvish).
    Completions(CompletionsArgs),
}
