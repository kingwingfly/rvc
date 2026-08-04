//! `rvc train` — native (Rust/burn) RVC generator training.
//!
//! Corpus audio is decoded and resampled with `audio-kit`, content/F0 features
//! come from the same ONNX extractors the inference path uses (`rvc-core`), and
//! the generator is trained with `burn`, writing a `.safetensors` file. Deploy
//! it directly (`rvc convert --backend burn`), or run the standalone `export/`
//! uv script to convert it to ONNX — the only Python the toolkit uses.

use anyhow::{Context, Result};
use rvc_train::{TrainRequest, TrainSettings};

use crate::args::{Backend, TrainArgs, weight_format};

/// The Burn backend a `--backend` choice names, rejecting the one that cannot
/// train.
///
/// `--backend` is one enum across the whole toolkit, so `onnx` parses here too;
/// it just has nowhere to go. Saying that beats the generic "unsupported", since
/// ONNX Runtime has no training path at all and the graphs it runs are what a
/// finished model is *exported to* afterwards.
fn train_backend(backend: Backend) -> Result<rvc_train::TrainBackend> {
    Ok(match backend {
        Backend::Auto => rvc_train::TrainBackend::Auto,
        Backend::Cuda => rvc_train::TrainBackend::Cuda,
        Backend::Tch => rvc_train::TrainBackend::LibTorch,
        Backend::Wgpu => rvc_train::TrainBackend::Wgpu,
        Backend::Onnx => anyhow::bail!(
            "ONNX Runtime cannot train — train on a Burn backend \
             (`--backend auto|cuda|tch|wgpu`), then convert the result with \
             `export/export_onnx.py` to run it on ONNX Runtime"
        ),
    })
}

pub async fn run(args: TrainArgs) -> Result<()> {
    args.verify()?;
    // Before anything is fetched: a backend that cannot train should not cost a
    // model download first.
    let backend = train_backend(args.backend)?;
    // ONNX unless `--content-vec-backend`/`--rmvpe-backend` say otherwise, and
    // they may only say ONNX for now — `TrainArgs::feature_backends` explains
    // why the trainer is the one command whose feature default is not
    // `--backend`. Resolved here so the fetches below name a format rather than
    // hard-coding one.
    let features = args.feature_backends()?;

    // The dashboard runs only on a real terminal; otherwise plain logs. This
    // must match main.rs's decision to route logs off stderr.
    let use_tui = cli_kit::use_tui(args.no_tui);
    let stop = cli_kit::stop_on_ctrl_c();

    // Before anything is fetched, decoded or loaded: a run that would replace an
    // earlier voice must cost a second to refuse, not an hour of GPU. `--resume`
    // is the case where overwriting is the whole point.
    let family = train_kit::Checkpoint::new(&args.out);
    let ema = args.ema_frac > 0.0;
    let mut planned = family.members(ema, true);
    if !args.no_save_best {
        planned.extend(family.best().members(ema, true));
    }
    train_kit::ensure_absent(planned, args.yes || args.resume.is_some())?;

    let cache = args.cache_dir.as_path();

    let content = match &args.content {
        Some(p) => p.clone(),
        None => {
            tracing::info!(
                "resolving ContentVec ({}) from Hugging Face...",
                features.content
            );
            hub_kit::fetch_contentvec(weight_format(features.content), cache)
                .await
                .context("failed to fetch ContentVec (override the path with --content)")?
        }
    };
    let rmvpe = match &args.rmvpe {
        Some(p) => p.clone(),
        None => {
            tracing::info!("resolving RMVPE ({}) from Hugging Face...", features.rmvpe);
            hub_kit::fetch_rmvpe(weight_format(features.rmvpe), cache)
                .await
                .context("failed to fetch RMVPE (override the path with --rmvpe)")?
        }
    };

    // Both short-circuits matter: `--no-pretrained` wants no warm start, and
    // `--resume` continues from weights that already exist — a base would be
    // loaded and immediately overwritten. Neither may cost a download.
    let auto_pretrained = !args.no_pretrained && args.resume.is_none();
    let base_dir = hub_kit::pretrained_dir(cache);
    let pretrained_g = match &args.pretrained_g {
        Some(p) => Some(p.clone()),
        None if auto_pretrained => Some(
            hub_kit::fetch_pretrained(&hub_kit::default_pretrained_g(), &base_dir)
                .await
                .context("failed to fetch the generator base (override with --pretrained-g, or pass --no-pretrained to train from scratch)")?,
        ),
        None => None,
    };
    let pretrained_d = match &args.pretrained_d {
        Some(p) => Some(p.clone()),
        None if auto_pretrained => Some(
            hub_kit::fetch_pretrained(&hub_kit::default_pretrained_d(), &base_dir)
                .await
                .context("failed to fetch the discriminator base (override with --pretrained-d, or pass --no-pretrained to train from scratch)")?,
        ),
        None => None,
    };

    let req = TrainRequest {
        data: args.data,
        out: args.out.clone(),
        content,
        rmvpe,
        resume: args.resume,
        pretrained_g,
        pretrained_d,
        settings: TrainSettings {
            sample_rate: args.model_sr,
            epochs: args.epochs,
            batch_size: args.batch_size,
            speaker_id: args.speaker_id,
            lr: args.lr,
            lr_final: args.lr_final,
            ema_frac: args.ema_frac,
            grad_accum: args.grad_accum,
            d_lr_ratio: args.d_lr_ratio,
            d_interval: args.d_interval,
            snr_weight: args.snr_weight,
            save_best: !args.no_save_best,
            use_tui,
        },
        backend,
        devices: args.device,
        stop,
    };

    // Training is blocking (GPU/CPU compute); keep it off the async runtime.
    let out = tokio::task::spawn_blocking(move || rvc_train::train(req))
        .await
        .context("training task panicked")??;

    // Print to stderr so it's visible even in TUI mode (logs went to a file).
    // stdout, not the log: this is the one line a caller may want to pipe.
    println!("{}", out.display());
    Ok(())
}
