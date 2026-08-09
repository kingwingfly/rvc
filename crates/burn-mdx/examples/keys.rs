//! List a checkpoint's tensors, so a module tree can be built to match.
//!
//! The first thing to run against any new checkpoint. Guessing names from a
//! reference implementation's source is how a port acquires a silent mismatch —
//! the weights load, the counts look plausible, and the model is wrong.
//!
//! It earns its place here beyond the other five copies for one reason
//! specific to this family: **the MDX config decides the architecture**, and
//! the checkpoint is the only thing that confirms which config a file was
//! trained with. `MDX23C-8KFFT-InstVoc_HQ.ckpt` and `MDX23C_D1581.ckpt` sit in
//! the same directory under the same name-prefix and do not have the same
//! shape. Read `first_conv.weight`, one `tdf.2.weight` and `final_conv.2.weight`
//! and every field of [`burn_mdx::MdxConfig`] follows.
//!
//! `--group` collapses `blocks.0`, `blocks.1`, … into `blocks.N` so a 319-tensor
//! file prints as 27 lines.
//!
//! Usage: `cargo run -p burn-mdx --example keys -- [--top-level-key model] [--group] <ckpt>`

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

/// `encoder_blocks.3.tfc_tdf.blocks.1.tdf.0.weight` ->
/// `encoder_blocks.N.tfc_tdf.blocks.N.tdf.N.weight`.
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
