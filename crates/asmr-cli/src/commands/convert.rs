//! `asmr convert` — batch file conversion to WAV (Burn or ONNX generator).

use anyhow::{Context, Result};
use asmr_audio::{DecodeOptions, decode_paths, write_wav_file};
use asmr_vc::{ANALYSIS_SR, ConvertParams, Converter, RvcModel, StreamParams};
use futures::{StreamExt, stream};

use crate::args::{ConvertArgs, InferBackend};
use crate::commands::common::{build_rvc_config, resolve_feature_models};
use crate::commands::convert_burn::BurnConverter;

pub async fn run(args: ConvertArgs) -> Result<()> {
    let onnx_model = args.models.model.extension().and_then(|e| e.to_str()) == Some("onnx");
    let use_burn = match args.backend {
        InferBackend::Auto => !onnx_model,
        InferBackend::Burn => true,
        InferBackend::Onnx => false,
    };
    if use_burn {
        run_burn(args).await
    } else {
        run_onnx(args).await
    }
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
    sr: u32,
    out: Vec<f32>,
) -> Result<()> {
    let stem = input.file_stem().and_then(|s| s.to_str()).unwrap_or("out");
    let out_path = dir.join(format!("{stem}.wav"));
    let out_stream = stream::iter([Ok::<_, asmr_audio::AudioError>(out)]);
    write_wav_file(&out_path, sr, out_stream)
        .await
        .with_context(|| format!("writing {}", out_path.display()))?;
    tracing::info!("wrote {}", out_path.display());
    Ok(())
}

/// Native Burn generator path (`.pth`/`.safetensors` weights).
async fn run_burn(args: ConvertArgs) -> Result<()> {
    let (content, rmvpe) = resolve_feature_models(&args.models).await?;
    let weights = args.models.model.clone();
    anyhow::ensure!(
        weights.exists(),
        "generator weights not found: {} (train one with `asmr train`)",
        weights.display()
    );

    tracing::info!("loading Burn generator from {}", weights.display());
    let mut conv = BurnConverter::load(
        &content,
        &rmvpe,
        &weights,
        args.models.model_sr,
        args.transpose,
        args.models.speaker_id,
    )
    .context("failed to load Burn generator")?;
    let model_sr = conv.output_sr();

    tokio::fs::create_dir_all(&args.output_dir)
        .await
        .with_context(|| format!("creating output dir {}", args.output_dir.display()))?;

    for input in &args.input {
        tracing::info!("converting {} (burn)", input.display());
        let wav16k = decode_16k(input).await?;
        let out = tokio::task::block_in_place(|| conv.convert(&wav16k))
            .with_context(|| format!("converting {}", input.display()))?;
        write_out(&args.output_dir, input, model_sr, out).await?;
    }
    Ok(())
}

/// ONNX Runtime generator path (`.onnx` weights).
async fn run_onnx(args: ConvertArgs) -> Result<()> {
    let cfg = build_rvc_config(&args.models).await?;
    let model_sr = cfg.model_sr;

    // Load the model once (GPU/ORT init is expensive), reuse across files.
    let model = RvcModel::load(cfg).context("failed to load RVC models")?;
    let mut converter = Converter::new(
        model,
        StreamParams::batch(),
        ConvertParams {
            transpose: args.transpose,
        },
    );

    tokio::fs::create_dir_all(&args.output_dir)
        .await
        .with_context(|| format!("creating output dir {}", args.output_dir.display()))?;

    for input in &args.input {
        tracing::info!("converting {} (onnx)", input.display());
        let wav16k = decode_16k(input).await?;
        converter.reset();
        let out = tokio::task::block_in_place(|| converter.convert_all(&wav16k))
            .with_context(|| format!("converting {}", input.display()))?;
        write_out(&args.output_dir, input, model_sr, out).await?;
    }
    Ok(())
}
