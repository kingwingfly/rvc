//! `asmr train` — native (Rust/burn) RVC generator training.
//!
//! Corpus audio is decoded and resampled with `asmr-audio`, content/F0 features
//! come from the same ONNX extractors the inference path uses (`asmr-vc`), and
//! the generator is trained with `burn`. Only the final weight -> ONNX step is
//! left to a small Python helper (RVC's own exporter), per project policy.

use anyhow::{Context, Result};
use asmr_train::{TrainRequest, TrainSettings};

use crate::args::TrainArgs;

pub async fn run(args: TrainArgs) -> Result<()> {
    let cache = args.cache_dir.as_deref();

    let content = match &args.content {
        Some(p) => p.clone(),
        None => {
            tracing::info!("resolving ContentVec ONNX from Hugging Face...");
            asmr_hub::fetch(&asmr_hub::default_contentvec(), cache)
                .await
                .context("failed to fetch ContentVec ONNX (override with --content)")?
        }
    };
    let rmvpe = match &args.rmvpe {
        Some(p) => p.clone(),
        None => {
            tracing::info!("resolving RMVPE ONNX from Hugging Face...");
            asmr_hub::fetch(&asmr_hub::default_rmvpe(), cache)
                .await
                .context("failed to fetch RMVPE ONNX (override with --rmvpe)")?
        }
    };

    let req = TrainRequest {
        data: args.data,
        out: args.out.clone(),
        work_dir: args.work_dir,
        content,
        rmvpe,
        pretrained_g: args.pretrained_g,
        pretrained_d: args.pretrained_d,
        settings: TrainSettings {
            sample_rate: args.model_sr,
            epochs: args.epochs,
            batch_size: args.batch_size,
            speaker_id: args.speaker_id,
        },
    };

    // Training is blocking (GPU/CPU compute); keep it off the async runtime.
    let out = tokio::task::spawn_blocking(move || asmr_train::train(req))
        .await
        .context("training task panicked")??;

    tracing::info!("training complete; generator written to {}", out.display());
    Ok(())
}
