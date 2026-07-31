//! `stt` as a library, so the `voice` binary can host the same subcommand
//! without duplicating a single flag definition.
//!
//! `stt` is the whole tool on its own; `voice stt` is the same [`args::SttArgs`]
//! and the same [`run`] behind an extra level of subcommand.

pub mod args;
pub mod backend;

pub use args::{Format, SttArgs, run};
pub use cli_kit::Backend;
