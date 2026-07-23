//! Load a reference RVC generator checkpoint and report weight coverage.
//!
//! Usage: `cargo run -p burn-rvc --example load -- /path/to/f0G48k.pth`

use burn_ndarray::NdArray;
use burn_rvc::{MultiPeriodDiscriminator, Synthesizer, SynthesizerConfig};

fn main() {
    type B = NdArray;
    let device = Default::default();

    let path = std::env::args().nth(1).expect("usage: load <G.pth> [D.pth]");
    let cfg = SynthesizerConfig::v2_48k();
    let mut model = Synthesizer::<B>::new(&cfg, &device);

    let res = model.load_pytorch(&path).expect("failed to read checkpoint");

    println!("applied : {}", res.applied.len());
    println!("missing : {}  (model params with no checkpoint tensor)", res.missing.len());
    for (name, why) in &res.missing {
        println!("    MISSING {name}  ({why})");
    }
    println!("unused  : {}  (checkpoint tensors not yet ported)", res.unused.len());
    let mut prefixes: Vec<_> = res
        .unused
        .iter()
        .map(|k| k.split('.').next().unwrap_or(k).to_string())
        .collect();
    prefixes.sort();
    prefixes.dedup();
    println!("    unused top-level modules: {prefixes:?}");
    println!("errors  : {}", res.errors.len());
    for e in &res.errors {
        println!("    ERROR {e:?}");
    }

    if let Some(dpath) = std::env::args().nth(2) {
        println!("\n== discriminator {dpath} ==");
        let mut disc = MultiPeriodDiscriminator::<B>::new(&device);
        let dres = disc.load_pytorch(&dpath).expect("failed to read D checkpoint");
        println!("applied : {}", dres.applied.len());
        println!("missing : {}", dres.missing.len());
        for (name, why) in &dres.missing {
            println!("    MISSING {name}  ({why})");
        }
        println!("unused  : {}", dres.unused.len());
        println!("errors  : {}", dres.errors.len());
    }
}
