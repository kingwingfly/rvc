//! The bare invocation — a Unix filter: raw f32le PCM stdin -> stdout.
//!
//! Input must be **mono f32le @ 16 kHz** (the RVC analysis rate); output is mono
//! f32le at the generator's sample rate. Wire it up with ffmpeg on both ends:
//!
//! ```sh
//! ffmpeg -i in.mp3 -f f32le -ar 16000 -ac 1 - \
//!   | rvc -m voice.onnx --model-sr 48000 \
//!   | ffplay -f f32le -ar 48000 -ac 1 -
//! ```
//!
//! Pass `--denoise` to strip steady background hiss from the output — ffmpeg's
//! `anlmdn` non-local-means de-noiser applied in-process (via libavfilter),
//! tuned to preserve the soft broadband texture of quiet, breathy content. For
//! a different chain, denoise downstream with the `ffmpeg` binary instead,
//! e.g.:
//!
//! ```sh
//! ... | rvc -m voice.onnx --model-sr 48000 \
//!   | ffmpeg -f f32le -ar 48000 -ac 1 -i - -af afftdn=nf=-25,highpass=f=60 \
//!       -f f32le -ar 48000 -ac 1 - \
//!   | ffplay -f f32le -ar 48000 -ac 1 -
//! ```

use anyhow::{Context, Result};
use futures::StreamExt;
use rvc_core::{StreamParams, convert_stream};
use tokio::io::{AsyncWriteExt, BufWriter};

use crate::args::FilterArgs;
use crate::commands::common::build_converter;

pub async fn run(args: FilterArgs) -> Result<()> {
    args.verify()?;
    // Clap cannot mark `-m` required — these options sit beside the subcommands,
    // which would then demand it too — so it is checked here, before a download
    // or a model load can happen.
    let model = args.models.model()?;

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
        "{}: stdin f32le mono @16000 Hz -> stdout f32le mono @{} Hz",
        model.display(),
        out_sr
    );

    // stdin -> stream of f32 chunks -> converter -> stdout. Box::pin so the
    // async-generator streams satisfy the Unpin + Send + 'static bounds.
    let input = Box::pin(audio_kit::read_f32le(tokio::io::stdin(), args.chunk));
    let mut output = Box::pin(convert_stream(converter, input));

    let mut stdout = BufWriter::new(tokio::io::stdout());
    while let Some(item) = output.next().await {
        let chunk = item.context("conversion failed")?;
        audio_kit::write_f32le_chunk(&mut stdout, &chunk)
            .await
            .context("writing stdout")?;
        // Flush eagerly so downstream players get low latency.
        stdout.flush().await.ok();
    }
    stdout.flush().await.context("final flush")?;
    Ok(())
}
