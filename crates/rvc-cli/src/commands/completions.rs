//! `completions <shell>` — print a shell completion script to stdout.
//!
//! `clap_complete` derives the script from the same `clap` command tree the
//! parser uses, so completions never drift from the actual flags. The caller
//! passes its own tree, which is how one implementation serves both binaries.
//! Logs go to stderr, so stdout carries only the script:
//!
//! ```sh
//! rvc   completions zsh  > ~/.zfunc/_rvc
//! voice completions fish > ~/.config/fish/completions/voice.fish
//! ```

use anyhow::Result;

use crate::args::CompletionsArgs;

pub async fn run(args: CompletionsArgs, mut cmd: clap::Command) -> Result<()> {
    // Take the binary name off the command tree rather than as a separate
    // argument, so the script and the thing it completes cannot disagree.
    let bin = cmd.get_name().to_string();
    clap_complete::generate(args.shell, &mut cmd, bin, &mut std::io::stdout());
    Ok(())
}
