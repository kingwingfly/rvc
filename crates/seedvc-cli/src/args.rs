//! Command-line argument definitions (clap derive).
//!
//! Filled in by the unit that owns this file. The shape is fixed by the house
//! rule every engine follows: the bare invocation is the stdin→stdout filter,
//! and everything that is not streaming is a subcommand beside it.

use clap::{Args, Subcommand};

pub use cli_kit::{Backend, CompletionsArgs};

/// The whole of the `seedvc` command tree, defined once and worn two ways: the
/// `seedvc` binary flattens it at its top level, `voice` nests it under a
/// `seedvc` subcommand.
#[derive(Debug, Args)]
pub struct SeedVcCli {
    #[command(flatten)]
    pub filter: FilterArgs,
    #[command(subcommand)]
    pub command: Option<SeedVcCommand>,
}

#[derive(Debug, Subcommand)]
pub enum SeedVcCommand {
    /// Convert audio files into the reference's voice (WAV output).
    Convert(ConvertArgs),
    /// Prefetch the weights a conversion needs, so the first run is offline.
    Download(DownloadArgs),
    /// Print a shell completion script (bash, zsh, fish, powershell, elvish).
    Completions(CompletionsArgs),
}

/// Options for the bare invocation — the streaming filter.
#[derive(Debug, Args)]
pub struct FilterArgs {}

/// Options for batch conversion.
#[derive(Debug, Args)]
pub struct ConvertArgs {}

/// What a default run would fetch on demand, fetched up front instead.
#[derive(Debug, Args)]
pub struct DownloadArgs {}
