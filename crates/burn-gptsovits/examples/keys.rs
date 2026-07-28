//! List a checkpoint's tensors, so a module tree can be built to match.
//!
//! The first thing to run against any new checkpoint. Guessing names from a
//! reference implementation's source is how a port acquires a silent mismatch —
//! the weights load, the counts look plausible, and the model is wrong.
//!
//! `--group` collapses `layers.0`, `layers.1`, … into `layers.N` so a 24-layer
//! transformer prints as a couple of dozen lines instead of several hundred.
//!
//! Usage: `cargo run -p burn-gptsovits --example keys -- [--top-level-key model] [--group] <ckpt>`

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let group = take_flag(&mut args, "--group");
    let top = take_value(&mut args, "--top-level-key");
    let Some(path) = args.first() else {
        eprintln!("usage: keys [--top-level-key <k>] [--group] <checkpoint>");
        std::process::exit(2);
    };

    let tensors = burn_kit::store::pytorch_keys(path.as_ref(), top.as_deref())?;
    let mut seen = std::collections::HashSet::new();
    let mut shown = 0;
    for (name, dtype, shape) in &tensors {
        let label = if group { collapse(name) } else { name.clone() };
        if group && !seen.insert(label.clone()) {
            continue;
        }
        println!("{label:<66} {dtype:?} {shape:?}");
        shown += 1;
    }
    println!("\n{} tensors ({shown} shown)", tensors.len());
    Ok(())
}

/// `encoder.layers.7.attn.weight` -> `encoder.layers.N.attn.weight`.
fn collapse(name: &str) -> String {
    name.split('.')
        .map(|part| {
            if !part.is_empty() && part.chars().all(|c| c.is_ascii_digit()) {
                "N"
            } else {
                part
            }
        })
        .collect::<Vec<_>>()
        .join(".")
}

fn take_flag(args: &mut Vec<String>, flag: &str) -> bool {
    match args.iter().position(|a| a == flag) {
        Some(i) => {
            args.remove(i);
            true
        }
        None => false,
    }
}

fn take_value(args: &mut Vec<String>, flag: &str) -> Option<String> {
    let i = args.iter().position(|a| a == flag)?;
    args.remove(i);
    Some(args.remove(i))
}
