//! Convert a recording **through the streaming filter**, and hold the result
//! against the batch path.
//!
//! `examples/convert` is the file-in, file-out path; this is the same engine
//! driven as a `futures::Stream`, which is what the bare `seedvc` invocation
//! will be. It is the crate's end-to-end check that a chunked, crossfaded
//! conversion is the same conversion.
//!
//! ```text
//! cargo run -p seedvc-core --features tch --example stream -- \
//!     --dit dit.pth --campplus campplus_cn_common.bin \
//!     --bigvgan bigvgan.pt --content whisper-small/model.safetensors \
//!     --reference target-speaker.wav --input said-by-someone-else.wav \
//!     --compare -o out.wav
//! ```
//!
//! # Reading the result
//!
//! `--compare` runs the batch path over the same source, against the same
//! reference and the same seed, and prints how the two agree. **The metric is an
//! energy-envelope correlation and an RMS ratio, never a sample-wise
//! difference**: every chunk is an independent integration from its own noise,
//! so two runs of the same model agree on content and timbre and on nothing
//! about phase — a sample-wise number would be large and would mean nothing.
//!
//! The latency line is what this example exists to report, and it is worth
//! reading carefully: the source is fed as fast as the converter accepts it, so
//! what is timed is the **model** term alone. A real pipe adds the buffering
//! term on top — one `block + crossfade` of source, 2.2 s at the realtime
//! preset — because those samples have to exist before the chunk can run.
//!
//! Build it with an isolated `CARGO_TARGET_DIR` if anything else is building in
//! a sibling worktree — example binaries are not hashed per checkout, and the
//! one that runs is whichever landed last.

use std::path::PathBuf;
use std::time::Instant;

use burn_seedvc::flow::Sampler;
use cli_kit::Backend;
use futures::StreamExt;
use seedvc_core::{ConvertOptions, Converter, ModelPaths, StreamParams, convert_stream, load};

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

    let compare = args.iter().any(|a| a == "--compare");
    args.retain(|a| a != "--compare");
    let backend = flag(&mut args, "backend").unwrap_or_else(|| "tch".into());
    let device = flag(&mut args, "device").unwrap_or_else(|| "cpu".into());
    // 100 ms of 16 kHz mono, which is what the filter reads from stdin.
    let chunk: usize = parse(flag(&mut args, "chunk"), 1600);
    let block: usize = parse(flag(&mut args, "block"), StreamParams::realtime().block);
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
            "usage: stream [--backend tch] [--device cpu] [--chunk 1600] [--block 172] \
             [--steps 30] [--guidance 0.7] [--length-adjust 1.0] [--seed 0] [--compare] \
             --dit <ckpt.pth> --campplus <campplus_cn_common.bin> \
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
    let params = StreamParams {
        block,
        ..StreamParams::realtime()
    };

    // A current-thread runtime is enough, as in `examples/convert`: the model
    // already runs on its own blocking thread — that is what `convert_stream`
    // is for — so nothing here needs a second runtime worker.
    let runtime = tokio::runtime::Builder::new_current_thread()
        .build()
        .expect("tokio runtime");
    runtime.block_on(async {
        let model = load(&paths, backend, device).expect("loading the model");
        let sample_rate = model.config().sample_rate;
        let frame_rate = model.config().frame_rate();

        let started = Instant::now();
        let reference = seedvc_core::reference::analyse(model.as_ref(), &reference)
            .await
            .expect("analysing the reference");
        println!(
            "reference: {} mel frames ({:.2} s), {:.1?}",
            reference.frames,
            reference.frames as f32 / frame_rate,
            started.elapsed(),
        );

        let source = seedvc_core::CONTENT_SR;
        let mut decoded = Vec::new();
        let mut stream = Box::pin(audio_kit::decode_path(
            input.clone(),
            audio_kit::DecodeOptions::new(source),
        ));
        while let Some(piece) = stream.next().await {
            decoded.extend_from_slice(&piece.expect("decoding the source"));
        }
        println!(
            "source: {:.2} s at {source} Hz, fed {chunk} samples at a time",
            decoded.len() as f32 / source as f32,
        );

        // The batch path first, while the model is still borrowable — the
        // converter takes ownership of it.
        let batched = compare.then(|| {
            let started = Instant::now();
            let audio = seedvc_core::convert(model.as_ref(), &reference, &decoded, &opts)
                .expect("converting in one piece");
            println!(
                "batch: {:.2} s in {:.1?}",
                seconds(&audio, sample_rate),
                started.elapsed()
            );
            audio
        });

        let converter = Converter::new(model, reference, params, opts).expect("the geometry");
        println!(
            "stream: {} frames of new audio per chunk plus {} of crossfade — {:.2} s of source \
             buffered before the first chunk can run",
            params.block,
            params.crossfade,
            (params.block + params.crossfade) as f32 / frame_rate,
        );

        let pieces = futures::stream::iter(
            decoded
                .chunks(chunk)
                .map(|c| Ok(c.to_vec()))
                .collect::<Vec<_>>(),
        );
        let started = Instant::now();
        let mut first = None;
        let mut streamed = Vec::new();
        let mut output = Box::pin(convert_stream(converter, pieces));
        while let Some(item) = output.next().await {
            let piece = item.expect("converting");
            first.get_or_insert_with(|| started.elapsed());
            streamed.extend_from_slice(&piece);
        }
        println!(
            "stream: {:.2} s in {:.1?}, first chunk out after {:.1?} of model time",
            seconds(&streamed, sample_rate),
            started.elapsed(),
            first.expect("the stream produced nothing"),
        );

        if let Some(batched) = batched {
            // Envelope over the model's own hop, so the comparison is on the
            // grid the two paths chunk on.
            let hop = 256;
            let (a, b) = (envelope(&streamed, hop), envelope(&batched, hop));
            println!(
                "agreement: energy-envelope correlation {:.4}, RMS ratio {:.4} \
                 (streamed {:.5} against batched {:.5})",
                correlation(&a, &b),
                rms(&streamed) / rms(&batched),
                rms(&streamed),
                rms(&batched),
            );
        }

        audio_kit::write_wav_file(&out, sample_rate, futures::stream::iter([Ok(streamed)]))
            .await
            .expect("writing the output");
        println!("wrote {}", out.display());
    });
}

fn seconds(audio: &[f32], sample_rate: u32) -> f32 {
    audio.len() as f32 / sample_rate as f32
}

fn rms(audio: &[f32]) -> f32 {
    (audio.iter().map(|s| s * s).sum::<f32>() / audio.len() as f32).sqrt()
}

/// Per-hop RMS — the shape of the audio, with the phase two independent
/// generations were never going to share thrown away.
fn envelope(audio: &[f32], hop: usize) -> Vec<f32> {
    audio.chunks(hop).map(rms).collect()
}

fn correlation(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let (a, b) = (&a[..n], &b[..n]);
    let mean = |v: &[f32]| v.iter().sum::<f32>() / n as f32;
    let (ma, mb) = (mean(a), mean(b));
    let cov: f32 = a.iter().zip(b).map(|(x, y)| (x - ma) * (y - mb)).sum();
    let va: f32 = a.iter().map(|x| (x - ma).powi(2)).sum();
    let vb: f32 = b.iter().map(|y| (y - mb).powi(2)).sum();
    cov / (va * vb).sqrt()
}

fn parse<T: std::str::FromStr>(given: Option<String>, fallback: T) -> T {
    given.map_or(fallback, |v| match v.parse() {
        Ok(parsed) => parsed,
        Err(_) => panic!("could not parse {v:?}"),
    })
}
