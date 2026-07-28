//! Shared helpers: resolve ONNX assets, build an [`RvcConfig`], and construct a
//! backend-agnostic [`Converter`] for both `convert` and `serve`.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use burn_kit::DeviceSpec;
use hub_kit::ModelRef;
use rvc_core::{
    ConvertParams, Converter, DenoiseParams, ModelPaths, RvcConfig, RvcModel, StreamParams,
};

use crate::args::{InferBackend, ModelOpts};

/// The generator runtime a `--backend` choice resolves to for given weights.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Runtime {
    Onnx,
    Cuda,
    Tch,
    Wgpu,
}

impl Runtime {
    /// The name to log, so a run says which of the three actually ran.
    pub fn label(self) -> &'static str {
        match self {
            Self::Onnx => "onnx",
            Self::Cuda => "burn-cuda",
            Self::Tch => "burn-tch",
            Self::Wgpu => "burn-wgpu",
        }
    }
}

/// Resolve `--backend` against the weights extension and the hardware present.
///
/// `auto` reads the extension first (`.onnx` → ONNX Runtime), then asks
/// [`burn_kit::auto_backend`]. An explicit choice is never substituted: if it
/// can't run, loading it reports why.
pub fn resolve_runtime(backend: InferBackend, model: &Path) -> Runtime {
    match backend {
        InferBackend::Onnx => Runtime::Onnx,
        InferBackend::Cuda => Runtime::Cuda,
        InferBackend::Tch => Runtime::Tch,
        InferBackend::Wgpu => Runtime::Wgpu,
        InferBackend::Auto => {
            if model.extension().and_then(|e| e.to_str()) == Some("onnx") {
                return Runtime::Onnx;
            }
            match burn_kit::auto_backend() {
                burn_kit::AutoBackend::LibTorch => Runtime::Tch,
                burn_kit::AutoBackend::Cuda => Runtime::Cuda,
                burn_kit::AutoBackend::Wgpu => Runtime::Wgpu,
            }
        }
    }
}

/// Build a streaming [`Converter`] over the selected backend. Shared by
/// `convert` (batch preset) and `serve` (realtime preset); the same
/// block/overlap/crossfade code drives either the ONNX or the Burn generator.
pub async fn build_converter(
    opts: &ModelOpts,
    backend: InferBackend,
    device: DeviceSpec,
    transpose: i32,
    params: StreamParams,
    denoise: Option<DenoiseParams>,
) -> Result<Converter> {
    let conv_params = ConvertParams { transpose };
    let runtime = resolve_runtime(backend, &opts.model);

    if runtime == Runtime::Onnx {
        let cfg = build_rvc_config(opts).await?;
        let model = RvcModel::load(cfg).context("failed to load RVC models")?;
        return Ok(Converter::new(model, params, conv_params).with_denoise(denoise));
    }

    let (content, rmvpe) = resolve_feature_models(opts).await?;
    anyhow::ensure!(
        opts.model.exists(),
        "generator weights not found: {} (train one with the `train` subcommand)",
        opts.model.display()
    );
    tracing::info!(
        "loading Burn generator from {} ({}, device {device})",
        opts.model.display(),
        runtime.label()
    );

    // Each arm erases into the same non-generic `Converter` — that type erasure
    // is what makes the backend a run-time choice.
    let converter = match runtime {
        #[cfg(feature = "cuda")]
        Runtime::Cuda => {
            let g = tokio::task::block_in_place(|| {
                rvc_core::cuda_generator(
                    &content,
                    &rmvpe,
                    &opts.model,
                    opts.model_sr,
                    opts.speaker_id,
                    device,
                )
            })
            .context("failed to load the Burn generator on the CubeCL/CUDA backend")?;
            Converter::new(g, params, conv_params)
        }
        #[cfg(feature = "wgpu")]
        Runtime::Wgpu => {
            let g = tokio::task::block_in_place(|| {
                rvc_core::wgpu_generator(
                    &content,
                    &rmvpe,
                    &opts.model,
                    opts.model_sr,
                    opts.speaker_id,
                    device,
                )
            })
            .context("failed to load the Burn generator on the WebGPU backend")?;
            Converter::new(g, params, conv_params)
        }
        #[cfg(feature = "tch")]
        Runtime::Tch => {
            let g = tokio::task::block_in_place(|| {
                rvc_core::libtorch_generator(
                    &content,
                    &rmvpe,
                    &opts.model,
                    opts.model_sr,
                    opts.speaker_id,
                    device,
                )
            })
            .context("failed to load the Burn generator on the LibTorch backend")?;
            Converter::new(g, params, conv_params)
        }
        Runtime::Onnx => unreachable!("handled above"),
        // Only reachable on a `--no-default-features` build.
        #[allow(unreachable_patterns)]
        other => anyhow::bail!(
            "this binary was built without the {} backend \
             (rebuild with `--features {}`)",
            other.label(),
            match other {
                Runtime::Tch => "tch",
                Runtime::Wgpu => "wgpu",
                _ => "cuda",
            }
        ),
    };
    Ok(converter.with_denoise(denoise))
}

/// Resolve the ContentVec and RMVPE ONNX paths (downloading when not provided).
/// Used by the Burn inference path, which needs the feature extractors but not
/// the ORT generator session.
pub async fn resolve_feature_models(opts: &ModelOpts) -> Result<(PathBuf, PathBuf)> {
    let cache = opts.cache_dir.as_deref();
    let content = match &opts.content {
        Some(p) => p.clone(),
        None => hub_kit::fetch(&hub_kit::default_contentvec(), cache)
            .await
            .context("failed to fetch ContentVec ONNX (override with --content)")?,
    };
    let rmvpe = match &opts.rmvpe {
        Some(p) => p.clone(),
        None => hub_kit::fetch(&hub_kit::default_rmvpe(), cache)
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
            hub_kit::fetch(&hub_kit::default_contentvec(), cache)
                .await
                .context("failed to fetch ContentVec ONNX (override with --content)")?
        }
    };

    let rmvpe = match &opts.rmvpe {
        Some(p) => p.clone(),
        None => {
            tracing::info!("resolving RMVPE ONNX from Hugging Face...");
            hub_kit::fetch(&hub_kit::default_rmvpe(), cache)
                .await
                .context("failed to fetch RMVPE ONNX (override with --rmvpe)")?
        }
    };

    anyhow::ensure!(
        opts.model.exists(),
        "generator model not found: {} (train one with the `train` subcommand)",
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
