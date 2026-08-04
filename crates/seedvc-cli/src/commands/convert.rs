//! `seedvc convert` — batch files in, one `<stem>.wav` per input out.
//!
//! The same engine the bare invocation runs, with files at both ends instead of
//! a pipe. The model is loaded and the reference analysed **once** for the whole
//! batch, which is what this buys over a shell loop over the filter.
//!
//! It drives [`seedvc_core::convert`] rather than the streaming
//! [`Converter`](seedvc_core::Converter), and that is the one real difference
//! between the two paths: a file's length is known before the first chunk, so
//! the last chunk can be balanced against the one before it instead of being
//! whatever is left over. A stream cannot do that — it has already emitted the
//! audio it would need to give back.

use anyhow::{Context, Result};
use audio_kit::{DecodeOptions, decode_paths, write_wav_file};
use futures::{StreamExt, stream};
use seedvc_core::CONTENT_SR;

use crate::args::ConvertArgs;
use crate::commands::common::load_model;

pub async fn run(args: Box<ConvertArgs>) -> Result<()> {
    args.verify()?;
    // Checked before anything is fetched or loaded, for the reason `ModelOpts`
    // gives: clap cannot mark it required where these options sit.
    args.models.reference()?;

    let (model, reference) = load_model(&args.models, args.backend, args.device).await?;
    let out_sr = model.config().sample_rate;
    let opts = args.sampler.options();

    tokio::fs::create_dir_all(&args.output_dir)
        .await
        .with_context(|| format!("creating output dir {}", args.output_dir.display()))?;

    for input in &args.input {
        tracing::info!("converting {}", input.display());
        let source = decode(input).await?;
        // Seconds of uninterrupted tensor work per chunk, so it goes on a
        // blocking thread rather than stalling the runtime. Each call draws its
        // noise from a generator seeded afresh, so a file is reproducible on its
        // own rather than only as the nth of a run.
        let out = tokio::task::block_in_place(|| {
            seedvc_core::convert(model.as_ref(), &reference, &source, &opts)
        })
        .with_context(|| format!("converting {}", input.display()))?;

        // A source under one content frame has nothing for the encoder to read,
        // so the converter correctly returns nothing. Writing that would leave a
        // 44-byte WAV — a header and no samples — beside the real outputs, and
        // the run would report having written it. Refuse instead: an empty file
        // is indistinguishable from a conversion that went wrong.
        anyhow::ensure!(
            !out.is_empty(),
            "{} converted to no audio at all: it is shorter than one 20 ms frame of the content \
             encoder, so there is nothing in it to convert",
            input.display(),
        );

        // `<stem>.wav`, unlike `rvc`'s `<stem>_<model>.wav`: there is no trained
        // model to name here, and naming the reference clip instead would put a
        // path fragment nobody chose into every filename.
        let stem = input.file_stem().and_then(|s| s.to_str()).unwrap_or("out");
        let out_path = args.output_dir.join(format!("{stem}.wav"));
        let samples = stream::iter([Ok::<_, audio_kit::AudioError>(out)]);
        write_wav_file(&out_path, out_sr, samples)
            .await
            .with_context(|| format!("writing {}", out_path.display()))?;
        tracing::info!("wrote {}", out_path.display());
    }
    Ok(())
}

/// Decode one source file to mono `f32` at the content encoder's rate.
///
/// Only 16 kHz, unlike the reference's two decodes: the source's waveform is
/// never needed at the vocoder's rate, only what was said in it.
async fn decode(input: &std::path::Path) -> Result<Vec<f32>> {
    let decoded = decode_paths(vec![input.to_path_buf()], DecodeOptions::new(CONTENT_SR));
    let mut decoded = std::pin::pin!(decoded);
    let mut pcm = Vec::new();
    while let Some(chunk) = decoded.next().await {
        pcm.extend(chunk.with_context(|| format!("decoding {}", input.display()))?);
    }
    Ok(pcm)
}
