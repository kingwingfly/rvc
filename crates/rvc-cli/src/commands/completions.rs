//! `rvc completions <shell>` — print a shell completion script to stdout.
//!
//! `clap_complete` derives the script from the same `clap` command tree the
//! parser uses, so completions never drift from the actual flags. Logs go to
//! stderr, so stdout carries only the script:
//!
//! ```sh
//! rvc completions zsh  > ~/.zfunc/_rvc
//! rvc completions bash > /etc/bash_completion.d/rvc
//! rvc completions fish > ~/.config/fish/completions/rvc.fish
//! ```

use anyhow::Result;
use clap::CommandFactory;

use crate::args::{Cli, CompletionsArgs};

pub async fn run(args: CompletionsArgs) -> Result<()> {
    let mut cmd = Cli::command();
    clap_complete::generate(args.shell, &mut cmd, "rvc", &mut std::io::stdout());
    Ok(())
}
