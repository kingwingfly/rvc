//! `rvc convert` — batch file conversion to WAV (Burn or ONNX generator).

use anyhow::{Context, Result};
use audio_kit::{DecodeOptions, decode_paths, write_wav_file};
use futures::{StreamExt, stream};
use rvc_core::{ANALYSIS_SR, StreamParams};

use crate::args::ConvertArgs;
use crate::commands::common::{build_converter, resolve_backend};

pub async fn run(args: ConvertArgs) -> Result<()> {
    let backend = resolve_backend(args.backend, &args.models.model);

    // One loaded model (GPU/ORT init is expensive), reused across files.
    let mut converter = build_converter(
        &args.models,
        args.backend,
        args.device,
        args.transpose,
        StreamParams::batch(),
        args.denoise.params(),
    )
    .await?;
    let model_sr = converter.output_sr();

    tokio::fs::create_dir_all(&args.output_dir)
        .await
        .with_context(|| format!("creating output dir {}", args.output_dir.display()))?;

    for input in &args.input {
        tracing::info!("converting {} ({backend})", input.display());
        let wav16k = decode_16k(input).await?;
        converter.reset();
        let out = tokio::task::block_in_place(|| converter.convert_all(&wav16k))
            .with_context(|| format!("converting {}", input.display()))?;
        write_out(&args.output_dir, input, &args.models.model, model_sr, out).await?;
    }
    Ok(())
}

/// Decode a file to mono f32 @ 16 kHz (the RVC analysis rate).
async fn decode_16k(input: &std::path::Path) -> Result<Vec<f32>> {
    let decode = decode_paths(vec![input.to_path_buf()], DecodeOptions::new(ANALYSIS_SR));
    let mut decode = std::pin::pin!(decode);
    let mut wav16k = Vec::new();
    while let Some(chunk) = decode.next().await {
        wav16k.extend(chunk.with_context(|| format!("decoding {}", input.display()))?);
    }
    Ok(wav16k)
}

async fn write_out(
    dir: &std::path::Path,
    input: &std::path::Path,
    model: &std::path::Path,
    sr: u32,
    out: Vec<f32>,
) -> Result<()> {
    // `{original_name}_{model_name}.wav`.
    let stem = input.file_stem().and_then(|s| s.to_str()).unwrap_or("out");
    let model_name = model
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("model");
    let out_path = dir.join(format!("{stem}_{model_name}.wav"));
    let out_stream = stream::iter([Ok::<_, audio_kit::AudioError>(out)]);
    write_wav_file(&out_path, sr, out_stream)
        .await
        .with_context(|| format!("writing {}", out_path.display()))?;
    tracing::info!("wrote {}", out_path.display());
    Ok(())
}
