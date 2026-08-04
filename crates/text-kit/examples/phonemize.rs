//! Phonemize a line of text, the way `tts` would.
//!
//! `text-kit` is pure logic with no weights, so its unit tests cover everything
//! except the one path that needs a file on disk: **Japanese, whose dictionary
//! is a 28.7 MB run-time asset.** The label parsing and the prosody rules are
//! tested against hand-written HTS labels, but nothing in `cargo test` proves
//! that a real NAIST-JDic directory analyses a real sentence. This is that
//! check, and it is why the example exists rather than a test.
//!
//! ```sh
//! cargo run -p text-kit --example phonemize -- zh "今天天气很好"
//! cargo run -p text-kit --example phonemize -- ja "私は東京にいます" <naist-jdic-dir>
//! ```
//!
//! The dictionary directory is whatever `hub_kit::fetch_naist_jdic` unpacked
//! into the shared cache; `tts --language ja` fetches it on first use.

use std::path::PathBuf;

use text_kit::{JapaneseDict, Language};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let (lang, text) = match (args.next(), args.next()) {
        (Some(l), Some(t)) => (l, t),
        _ => {
            return Err("usage: phonemize <zh|en|ja> <text> [naist-jdic-dir]".into());
        }
    };
    let language: Language = lang.parse()?;
    let dict = match args.next().map(PathBuf::from) {
        Some(dir) => Some(JapaneseDict::open(&dir)?),
        None => None,
    };

    // `phonemize_mixed`, not `phonemize`: it is the entry point `tts` uses, so
    // a mixed line routes each run to its own front end and this reports what
    // the model would actually be fed.
    let out = text_kit::phonemize_mixed(&text, language, dict.as_ref())?;
    println!("normalized : {}", out.normalized);
    println!("phones ({:>3}): {}", out.phones.len(), out.phones.join(" "));
    println!("ids        : {:?}", out.ids());
    match &out.word2ph {
        Some(w) => println!("word2ph    : {w:?}"),
        // Not a gap: upstream's cleaner returns `None` for anything that is not
        // Chinese, so the prosody encoder is fed zeros. Reported so a reader
        // does not go looking for the counts.
        None => println!("word2ph    : none (as upstream, for this language)"),
    }

    // An `UNK` is how an out-of-table symbol reaches the model, and it is silent
    // otherwise — the id is simply the unknown one. Japanese emits `#` for an
    // accent-phrase border, which is deliberately not in the 732-symbol table.
    let unknown = out
        .phones
        .iter()
        .filter(|p| text_kit::id(p) == text_kit::id("UNK"));
    let count = unknown.count();
    if count > 0 {
        println!("note       : {count} phone(s) fell back to UNK");
    }
    Ok(())
}
