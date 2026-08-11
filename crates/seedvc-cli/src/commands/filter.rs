//! The bare invocation — a Unix filter: raw f32le PCM stdin -> stdout.
//!
//! Input must be **mono f32le @ 16 kHz**, the content encoder's rate; output is
//! mono f32le at **22.05 kHz**, which is BigVGAN's rate and not `rvc`'s 48 kHz —
//! so anything downstream in the pipe has to be told which one it is getting:
//!
//! ```sh
//! ffmpeg -v quiet -i take.mp3 -f f32le -ar 16000 -ac 1 - \
//!   | seedvc -r target-voice.wav \
//!   | ffplay -f f32le -ar 22050 -ac 1 -
//! ```
//!
//! `-r` alone decides who the output sounds like. There is no model to train, so
//! the reference clip is the whole speaker specification, and it is analysed
//! once — before a single sample is read from stdin — rather than per chunk.
//!
//! **The latency is per block, not per sample.** Every stage the audio passes
//! through is a window rather than a filter, so nothing can be emitted until a
//! whole chunk exists: at [`StreamParams::realtime`] that is 2.2 s of buffering
//! plus one chunk of model time. `--chunk` is only how much stdin is read at a
//! time and does not move it — **`--block-frames` is the flag that does**, and
//! the arithmetic behind it is [`seedvc_core::stream`]'s. On the hardware this
//! was developed against the model is slower than realtime, so a live pipe falls
//! behind — what the preset buys is that the audio which does come out comes out
//! in steps rather than after the whole recording.

use anyhow::{Context, Result};
use futures::StreamExt;
use seedvc_core::{CONTENT_SR, Converter, convert_stream};
use tokio::io::{AsyncWriteExt, BufWriter};

use crate::args::FilterArgs;
use crate::commands::common::load_model;

pub async fn run(args: FilterArgs) -> Result<()> {
    args.verify()?;
    // Clap cannot mark `-r` required — these options sit beside the subcommands,
    // which would then demand one too — so it is checked here, before a download
    // or a model load can happen.
    let reference = args.models.reference()?;

    let (model, analysed) = load_model(&args.models, args.backend, args.device).await?;
    // `Converter::new` clamps the block to what the reference left of the shared
    // window, so a long `--reference-secs` quietly lowers whatever was asked for
    // here — which is the arithmetic both flags are dials on.
    let converter = Converter::new(model, analysed, args.params(), args.sampler.options())?;
    let out_sr = converter.output_sr();
    tracing::info!(
        "{}: stdin f32le mono @{CONTENT_SR} Hz -> stdout f32le mono @{out_sr} Hz",
        reference.display(),
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
        // Flush eagerly, and the reason is sharper than "low latency": tokio's
        // `BufWriter` passes any single write at or above its 8 KiB capacity
        // straight through, so without this the bulk of each chunk would stream
        // and only its sub-8-KiB tail would be stranded. That failure looks like
        // a working filter, which is exactly how it went unnoticed in `tts`.
        stdout.flush().await.ok();
    }
    stdout.flush().await.context("final flush")?;
    Ok(())
}
