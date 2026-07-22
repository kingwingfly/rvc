//! `asmr convert` — batch file conversion to WAV.

use anyhow::{Context, Result};
use asmr_audio::{decode_paths, write_wav_file, DecodeOptions};
use asmr_vc::{ConvertParams, Converter, RvcModel, StreamParams, ANALYSIS_SR};
use futures::{stream, StreamExt};

use crate::args::ConvertArgs;
use crate::commands::common::build_config;

pub async fn run(args: ConvertArgs) -> Result<()> {
    let cfg = build_config(&args.models).await?;
    let model_sr = cfg.model_sr;

    // Load the model once (GPU/ORT init is expensive), reuse across files.
    let model = RvcModel::load(cfg).context("failed to load RVC models")?;
    let mut converter = Converter::new(
        model,
        StreamParams::batch(),
        ConvertParams { transpose: args.transpose },
    );

    tokio::fs::create_dir_all(&args.output_dir)
        .await
        .with_context(|| format!("creating output dir {}", args.output_dir.display()))?;

    for input in &args.input {
        tracing::info!("converting {}", input.display());

        // Decode to mono f32 @ 16 kHz (the RVC analysis rate).
        let decode = decode_paths(vec![input.clone()], DecodeOptions::new(ANALYSIS_SR));
        let mut decode = std::pin::pin!(decode);
        let mut wav16k: Vec<f32> = Vec::new();
        while let Some(chunk) = decode.next().await {
            wav16k.extend(chunk.with_context(|| format!("decoding {}", input.display()))?);
        }

        // Conversion (blocking ORT work) on a worker thread.
        converter.reset();
        let out = tokio::task::block_in_place(|| converter.convert_all(&wav16k))
            .with_context(|| format!("converting {}", input.display()))?;

        let stem = input.file_stem().and_then(|s| s.to_str()).unwrap_or("out");
        let out_path = args.output_dir.join(format!("{stem}.wav"));
        let out_stream = stream::iter([Ok::<_, asmr_audio::AudioError>(out)]);
        write_wav_file(&out_path, model_sr, out_stream)
            .await
            .with_context(|| format!("writing {}", out_path.display()))?;
        tracing::info!("wrote {}", out_path.display());
    }

    Ok(())
}
