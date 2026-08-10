//! `preprocess` — turn recordings into a corpus a trainer can eat.
//!
//! Each subcommand is one stage, and they compose because every one of them
//! reads audio files and writes audio files:
//!
//! ```sh
//! preprocess denoise raw/ -o denoised/
//! preprocess clip denoised/ -o dataset/
//! ```
//!
//! There is deliberately no bare invocation — see the crate docs for why that
//! is not the streaming rule being broken.
//!
//! `voice preprocess …` hosts the same command tree, from this crate as a
//! library.

use anyhow::Result;
use clap::{CommandFactory, Parser, Subcommand};
use preprocess_cli::args::{CompletionsArgs, PreprocessCli, PreprocessCommand};

/// preprocess — corpus preparation: slice recordings on silence, remove hiss.
#[derive(Debug, Parser)]
#[command(name = "preprocess", version, about)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

/// The stages, plus the one subcommand that belongs to this binary.
///
/// `completions` is here rather than in [`PreprocessCommand`] because a
/// completion script describes an executable: under `voice` the executable is
/// `voice`, and a nested `voice preprocess completions` could only emit
/// `voice`'s script while appearing to offer this one's.
#[derive(Debug, Subcommand)]
enum Command {
    #[command(flatten)]
    Stage(PreprocessCommand),
    /// Print a shell completion script (bash, zsh, fish, powershell, elvish).
    Completions(CompletionsArgs),
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    preprocess_cli::init_logging();
    let command = match cli.command {
        Command::Completions(a) => return cli_kit::completions(a, Cli::command()),
        Command::Stage(c) => c,
    };
    preprocess_cli::run(PreprocessCli { command }).await
}
