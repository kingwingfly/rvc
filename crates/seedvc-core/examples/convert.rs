//! Convert one recording into another speaker's voice — **the first thing in
//! this crate that produces audio.**
//!
//! `examples/coverage` proves the four checkpoints load; this proves they
//! compute. It is the engine's end-to-end check until `seedvc-cli` exists, and
//! it stays useful afterwards, because it is the shortest path from a weights
//! directory to a WAV with no argument parsing of its own in the way.
//!
//! ```text
//! cargo run -p seedvc-core --features tch --example convert -- \
//!     --dit dit.pth --campplus campplus_cn_common.bin \
//!     --bigvgan bigvgan.pt --content whisper-small/model.safetensors \
//!     --reference target-speaker.wav --input said-by-someone-else.wav -o out.wav
//! ```
//!
//! # Reading the result
//!
//! Two checks, and neither is a listen:
//!
//! - **Content survived** — transcribe the input and the output with `stt` and
//!   compare. Words in, same words out.
//! - **Timbre moved** — `burn-seedvc`'s `speaker` example embeds raw 16 kHz
//!   clips and prints their cosine matrix. `cos(output, reference)` has to beat
//!   `cos(output, input)`. It is a *relative* comparison and says nothing about
//!   quality; a conversion can pass it and still sound rough.
//!
//! Build it with an isolated `CARGO_TARGET_DIR` if anything else is building in
//! a sibling worktree — example binaries are not hashed per checkout, and the
//! one that runs is whichever landed last.

use std::path::PathBuf;

use burn_seedvc::flow::Sampler;
use cli_kit::Backend;
use seedvc_core::{ConvertOptions, ModelPaths, load};

/// Either spelling, so the output flag can be `-o` like every other binary's.
fn flag(args: &mut Vec<String>, name: &str) -> Option<String> {
    let (long, short) = (format!("--{name}"), format!("-{name}"));
    let i = args.iter().position(|a| *a == long || *a == short)?;
    args.remove(i);
    Some(args.remove(i))
}

fn main() {
    cli_kit::init_logging(None);
    let mut args: Vec<String> = std::env::args().skip(1).collect();

    let backend = flag(&mut args, "backend").unwrap_or_else(|| "tch".into());
    let device = flag(&mut args, "device").unwrap_or_else(|| "cpu".into());
    let opts = ConvertOptions {
        sampler: Sampler {
            steps: parse(flag(&mut args, "steps"), Sampler::default().steps),
            guidance: parse(flag(&mut args, "guidance"), Sampler::default().guidance),
        },
        length_adjust: parse(flag(&mut args, "length-adjust"), 1.0),
        seed: parse(flag(&mut args, "seed"), 0),
    };

    let names = [
        "dit",
        "campplus",
        "bigvgan",
        "content",
        "reference",
        "input",
        "o",
    ];
    let given: Vec<Option<PathBuf>> = names
        .iter()
        .map(|n| flag(&mut args, n).map(PathBuf::from))
        .collect();
    let [dit, campplus, bigvgan, content, reference, input, out] =
        <[_; 7]>::try_from(given).unwrap();
    let (
        Some(dit),
        Some(campplus),
        Some(bigvgan),
        Some(content),
        Some(reference),
        Some(input),
        Some(out),
    ) = (dit, campplus, bigvgan, content, reference, input, out)
    else {
        eprintln!(
            "usage: convert [--backend tch] [--device cpu] [--steps 30] [--guidance 0.7] \
             [--length-adjust 1.0] [--seed 0] --dit <ckpt.pth> --campplus <campplus_cn_common.bin> \
             --bigvgan <bigvgan_generator.pt> --content <whisper-small/model.safetensors> \
             --reference <voice-to-become.wav> --input <audio-to-convert.wav> -o <out.wav>"
        );
        std::process::exit(2);
    };

    let paths = ModelPaths {
        dit: &dit,
        campplus: &campplus,
        bigvgan: &bigvgan,
        content: &content,
    };
    let backend = <Backend as clap::ValueEnum>::from_str(&backend, true).expect("--backend");
    let device = cli_kit::parse_device(&device).expect("--device");

    // A current-thread runtime is enough and is the honest shape: every decode
    // is a `spawn_blocking` inside `audio-kit`, and the conversion itself is a
    // long synchronous stretch of tensor work that nothing else overlaps with.
    let runtime = tokio::runtime::Builder::new_current_thread()
        .build()
        .expect("tokio runtime");
    runtime.block_on(async {
        let model = load(&paths, backend, device).expect("loading the model");

        let started = std::time::Instant::now();
        let reference = seedvc_core::reference::analyse(model.as_ref(), &reference)
            .await
            .expect("analysing the reference");
        println!(
            "reference: {} mel frames ({:.2} s), {:.1?}",
            reference.frames,
            reference.frames as f32 / model.config().frame_rate(),
            started.elapsed(),
        );

        let started = std::time::Instant::now();
        let audio = seedvc_core::convert_path(model.as_ref(), &reference, &input, &opts)
            .await
            .expect("converting");
        let seconds = audio.len() as f32 / model.config().sample_rate as f32;
        println!(
            "converted: {} samples ({seconds:.2} s at {} Hz), {:.1?}",
            audio.len(),
            model.config().sample_rate,
            started.elapsed(),
        );

        audio_kit::write_wav_file(
            &out,
            model.config().sample_rate,
            futures::stream::iter([Ok(audio)]),
        )
        .await
        .expect("writing the output");
        println!("wrote {}", out.display());
    });
}

fn parse<T: std::str::FromStr>(given: Option<String>, fallback: T) -> T {
    given.map_or(fallback, |v| match v.parse() {
        Ok(parsed) => parsed,
        Err(_) => panic!("could not parse {v:?}"),
    })
}
