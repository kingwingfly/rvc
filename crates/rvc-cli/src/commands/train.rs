//! `rvc train` — native (Rust/burn) RVC generator training.
//!
//! Corpus audio is decoded and resampled with `rvc-audio`, content/F0 features
//! come from the same ONNX extractors the inference path uses (`rvc-core`), and
//! the generator is trained with `burn`, writing a `.safetensors` file. Deploy
//! it directly (`rvc convert --backend burn`), or run the standalone `export/`
//! uv script to convert it to ONNX — the only Python the toolkit uses.

use std::io::IsTerminal;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::{Context, Result};
use rvc_train::{TrainRequest, TrainSettings};

use crate::args::TrainArgs;

pub async fn run(args: TrainArgs) -> Result<()> {
    // The dashboard runs only on a real terminal; otherwise plain logs. This
    // must match main.rs's decision to route logs off stderr.
    let use_tui = !args.no_tui && std::io::stdout().is_terminal();

    // Early stop: Ctrl-C flips this flag; the trainer saves the model and exits.
    // (With the TUI active, Ctrl-C is captured as a key — stop with `q` there.)
    let stop = Arc::new(AtomicBool::new(false));
    {
        let stop = stop.clone();
        tokio::spawn(async move {
            if tokio::signal::ctrl_c().await.is_ok() {
                stop.store(true, Ordering::Relaxed);
            }
        });
    }

    let cache = args.cache_dir.as_deref();

    let content = match &args.content {
        Some(p) => p.clone(),
        None => {
            tracing::info!("resolving ContentVec ONNX from Hugging Face...");
            rvc_hub::fetch(&rvc_hub::default_contentvec(), cache)
                .await
                .context("failed to fetch ContentVec ONNX (override with --content)")?
        }
    };
    let rmvpe = match &args.rmvpe {
        Some(p) => p.clone(),
        None => {
            tracing::info!("resolving RMVPE ONNX from Hugging Face...");
            rvc_hub::fetch(&rvc_hub::default_rmvpe(), cache)
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
            use_tui,
        },
        stop,
    };

    // Training is blocking (GPU/CPU compute); keep it off the async runtime.
    let out = tokio::task::spawn_blocking(move || rvc_train::train(req))
        .await
        .context("training task panicked")??;

    // Print to stderr so it's visible even in TUI mode (logs went to a file).
    tracing::info!("training complete; generator written to {}", out.display());
    eprintln!("training complete; generator written to {}", out.display());
    Ok(())
}
