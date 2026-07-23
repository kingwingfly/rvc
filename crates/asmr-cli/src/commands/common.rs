//! Shared helpers: resolve ONNX assets and build an [`RvcConfig`].

use std::path::PathBuf;

use anyhow::{Context, Result};
use asmr_hub::ModelRef;
use asmr_vc::{ModelPaths, RvcConfig};

use crate::args::ModelOpts;

/// Resolve the ContentVec and RMVPE ONNX paths (downloading when not provided).
/// Used by the Burn inference path, which needs the feature extractors but not
/// the ORT generator session.
pub async fn resolve_feature_models(opts: &ModelOpts) -> Result<(PathBuf, PathBuf)> {
    let cache = opts.cache_dir.as_deref();
    let content = match &opts.content {
        Some(p) => p.clone(),
        None => asmr_hub::fetch(&asmr_hub::default_contentvec(), cache)
            .await
            .context("failed to fetch ContentVec ONNX (override with --content)")?,
    };
    let rmvpe = match &opts.rmvpe {
        Some(p) => p.clone(),
        None => asmr_hub::fetch(&asmr_hub::default_rmvpe(), cache)
            .await
            .context("failed to fetch RMVPE ONNX (override with --rmvpe)")?,
    };
    Ok((content, rmvpe))
}

/// Resolve the ContentVec and RMVPE ONNX paths, downloading from Hugging Face
/// when not provided explicitly, then assemble the full pipeline config.
pub async fn build_config(opts: &ModelOpts) -> Result<RvcConfig> {
    let cache = opts.cache_dir.as_deref();

    let content = match &opts.content {
        Some(p) => p.clone(),
        None => {
            tracing::info!("resolving ContentVec ONNX from Hugging Face...");
            asmr_hub::fetch(&asmr_hub::default_contentvec(), cache)
                .await
                .context("failed to fetch ContentVec ONNX (override with --content)")?
        }
    };

    let rmvpe = match &opts.rmvpe {
        Some(p) => p.clone(),
        None => {
            tracing::info!("resolving RMVPE ONNX from Hugging Face...");
            asmr_hub::fetch(&asmr_hub::default_rmvpe(), cache)
                .await
                .context("failed to fetch RMVPE ONNX (override with --rmvpe)")?
        }
    };

    anyhow::ensure!(
        opts.model.exists(),
        "generator model not found: {} (train one with `asmr train`)",
        opts.model.display()
    );

    let mut cfg = RvcConfig::new(
        ModelPaths { content, rmvpe, generator: opts.model.clone() },
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
