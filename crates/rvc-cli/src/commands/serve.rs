//! `rvc serve` — realtime Unix filter (raw f32le PCM stdin -> stdout).
//!
//! Input must be **mono f32le @ 16 kHz** (the RVC analysis rate); output is mono
//! f32le at the generator's sample rate. Wire it up with ffmpeg on both ends:
//!
//! ```sh
//! ffmpeg -i in.mp3 -f f32le -ar 16000 -ac 1 - \
//!   | rvc serve -m voice.onnx --model-sr 48000 \
//!   | ffplay -f f32le -ar 48000 -ac 1 -
//! ```
//!
//! Pass `--denoise` to strip steady background hiss from the output — ffmpeg's
//! `anlmdn` non-local-means de-noiser applied in-process (via libavfilter),
//! tuned to preserve the soft broadband texture of ASMR/breathy content. For a
//! different chain, denoise downstream with the `ffmpeg` binary instead, e.g.:
//!
//! ```sh
//! ... | rvc serve -m voice.onnx --model-sr 48000 \
//!   | ffmpeg -f f32le -ar 48000 -ac 1 -i - -af afftdn=nf=-25,highpass=f=60 \
//!       -f f32le -ar 48000 -ac 1 - \
//!   | ffplay -f f32le -ar 48000 -ac 1 -
//! ```

use anyhow::{Context, Result};
use futures::StreamExt;
use rvc_core::{StreamParams, convert_stream};
use tokio::io::{AsyncWriteExt, BufWriter};

use crate::args::ServeArgs;
use crate::commands::common::build_converter;

pub async fn run(args: ServeArgs) -> Result<()> {
    let converter = build_converter(
        &args.models,
        args.backend,
        args.device,
        args.transpose,
        StreamParams::realtime(),
        args.denoise.params(),
    )
    .await?;
    let out_sr = converter.output_sr();
    tracing::info!(
        "serving: stdin f32le mono @16000 Hz -> stdout f32le mono @{} Hz",
        out_sr
    );

    // stdin -> stream of f32 chunks -> converter -> stdout. Box::pin so the
    // async-generator streams satisfy the Unpin + Send + 'static bounds.
    let input = Box::pin(rvc_audio::read_f32le(tokio::io::stdin(), args.chunk));
    let mut output = Box::pin(convert_stream(converter, input));

    let mut stdout = BufWriter::new(tokio::io::stdout());
    let mut bytes: Vec<u8> = Vec::new();
    while let Some(item) = output.next().await {
        let chunk = item.context("conversion failed")?;
        bytes.clear();
        bytes.reserve(chunk.len() * 4);
        for s in chunk {
            bytes.extend_from_slice(&s.to_le_bytes());
        }
        stdout.write_all(&bytes).await.context("writing stdout")?;
        // Flush eagerly so downstream players get low latency.
        stdout.flush().await.ok();
    }
    stdout.flush().await.context("final flush")?;
    Ok(())
}
