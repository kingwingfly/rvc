//! Shared helpers: resolve ONNX assets, build an [`RvcConfig`], and construct a
//! backend-agnostic [`Converter`] for both `convert` and `serve`.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use rvc_hub::ModelRef;
use rvc_core::{
    BurnGenerator, ConvertParams, Converter, ModelPaths, RvcConfig, RvcModel, StreamParams,
};

use crate::args::{InferBackend, ModelOpts};

/// Decide whether the Burn (native) generator runs, given the `--backend` flag
/// and the weights extension (`auto` → `.onnx` uses ONNX Runtime, else Burn).
pub fn use_burn_backend(backend: InferBackend, model: &Path) -> bool {
    let onnx_model = model.extension().and_then(|e| e.to_str()) == Some("onnx");
    match backend {
        InferBackend::Auto => !onnx_model,
        InferBackend::Burn => true,
        InferBackend::Onnx => false,
    }
}

/// Build a streaming [`Converter`] over the selected backend. Shared by
/// `convert` (batch preset) and `serve` (realtime preset); the same
/// block/overlap/crossfade code drives either the ONNX or the Burn generator.
pub async fn build_converter(
    opts: &ModelOpts,
    backend: InferBackend,
    transpose: i32,
    params: StreamParams,
) -> Result<Converter> {
    let conv_params = ConvertParams { transpose };
    if use_burn_backend(backend, &opts.model) {
        let (content, rmvpe) = resolve_feature_models(opts).await?;
        anyhow::ensure!(
            opts.model.exists(),
            "generator weights not found: {} (train one with `rvc train`)",
            opts.model.display()
        );
        tracing::info!("loading Burn generator from {}", opts.model.display());
        let generator = tokio::task::block_in_place(|| {
            BurnGenerator::load(&content, &rmvpe, &opts.model, opts.model_sr, opts.speaker_id)
        })
        .context("failed to load Burn generator")?;
        Ok(Converter::new(generator, params, conv_params))
    } else {
        let cfg = build_rvc_config(opts).await?;
        let model = RvcModel::load(cfg).context("failed to load RVC models")?;
        Ok(Converter::new(model, params, conv_params))
    }
}

/// Resolve the ContentVec and RMVPE ONNX paths (downloading when not provided).
/// Used by the Burn inference path, which needs the feature extractors but not
/// the ORT generator session.
pub async fn resolve_feature_models(opts: &ModelOpts) -> Result<(PathBuf, PathBuf)> {
    let cache = opts.cache_dir.as_deref();
    let content = match &opts.content {
        Some(p) => p.clone(),
        None => rvc_hub::fetch(&rvc_hub::default_contentvec(), cache)
            .await
            .context("failed to fetch ContentVec ONNX (override with --content)")?,
    };
    let rmvpe = match &opts.rmvpe {
        Some(p) => p.clone(),
        None => rvc_hub::fetch(&rvc_hub::default_rmvpe(), cache)
            .await
            .context("failed to fetch RMVPE ONNX (override with --rmvpe)")?,
    };
    Ok((content, rmvpe))
}

/// Resolve the ContentVec and RMVPE ONNX paths, downloading from Hugging Face
/// when not provided explicitly, then assemble the full pipeline config.
pub async fn build_rvc_config(opts: &ModelOpts) -> Result<RvcConfig> {
    let cache = opts.cache_dir.as_deref();

    let content = match &opts.content {
        Some(p) => p.clone(),
        None => {
            tracing::info!("resolving ContentVec ONNX from Hugging Face...");
            rvc_hub::fetch(&rvc_hub::default_contentvec(), cache)
                .await
                .context("failed to fetch ContentVec ONNX (override with --content)")?
        }
    };

    let rmvpe = match &opts.rmvpe {
        Some(p) => p.clone(),
        None => {
            tracing::info!("resolving RMVPE ONNX from Hugging Face...");
            rvc_hub::fetch(&rvc_hub::default_rmvpe(), cache)
                .await
                .context("failed to fetch RMVPE ONNX (override with --rmvpe)")?
        }
    };

    anyhow::ensure!(
        opts.model.exists(),
        "generator model not found: {} (train one with `rvc train`)",
        opts.model.display()
    );

    let mut cfg = RvcConfig::new(
        ModelPaths {
            content,
            rmvpe,
            generator: opts.model.clone(),
        },
        opts.model_sr,
    );
    cfg.speaker_id = opts.speaker_id;
    Ok(cfg)
}

/// Parse an `owner/name:file` override string into a [`ModelRef`].
pub fn parse_model_ref(s: &str) -> Result<ModelRef> {
    let (repo, file) = s
        .rsplit_once(':')
        .context("expected format `owner/name:file`")?;
    let (owner, name) = repo
        .split_once('/')
        .context("expected format `owner/name:file`")?;
    Ok(ModelRef::new(owner, name, file))
}
