//! `voice` — the full speech toolkit: recognition, synthesis and voice
//! conversion, each one a Unix filter so they compose in a pipe.
//!
//! ```sh
//! ffmpeg -i take.mp3 -f f32le -ar 16000 -ac 1 - \
//!   | voice rvc -m voice.safetensors --model-sr 48000 \
//!   | ffplay -f f32le -ar 48000 -ac 1 -
//! ```
//!
//! Every engine has the same shape: the bare invocation is the filter, and
//! everything else is a subcommand beside it (`voice rvc convert`,
//! `voice tts train`). `voice rvc …` hosts exactly the command tree of the
//! standalone `rvc` binary, from the same code — install `rvc` on its own if
//! voice conversion is all you need, and `voice` if you want the rest of the
//! toolkit with it.
//!
//! Logs go to **stderr** so a filter's stdout carries only data.

mod args;

use anyhow::Result;
use clap::{CommandFactory, Parser};
use rvc_cli::args::RvcCommand;
use tts_cli::TtsCommand;

use args::{Cli, Command};

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    // Both trainers put a TUI on the terminal, so both need their tracing sent
    // to a file instead of stderr. Each engine owns the decision for its own
    // subcommand — `voice` only routes to the right one.
    match &cli.command {
        Command::Rvc(c) => rvc_cli::init_logging(match &c.command {
            Some(RvcCommand::Train(a)) => Some(a.as_ref()),
            _ => None,
        }),
        Command::Tts(c) => tts_cli::init_logging(match &c.command {
            Some(TtsCommand::Train(a)) => Some(a.as_ref()),
            _ => None,
        }),
        _ => rvc_cli::init_logging(None),
    }
    match cli.command {
        Command::Rvc(c) => rvc_cli::run::<Cli>(*c).await,
        Command::Stt(c) => stt_cli::run::<Cli>(*c).await,
        Command::Tts(c) => tts_cli::run::<Cli>(*c).await,
        Command::Completions(a) => cli_kit::completions(a, Cli::command()),
    }
}
