//! Shared helpers: resolve the feature-model assets, build an [`RvcConfig`], and
//! construct a backend-agnostic [`Converter`] for both the streaming filter and
//! `convert`.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use burn_kit::DeviceSpec;
use hub_kit::ModelRef;
use rvc_core::{
    ConvertParams, Converter, DenoiseParams, ModelPaths, RvcConfig, RvcModel, StreamParams,
};

use crate::args::{Backend, FeatureBackendOpts, FeatureBackends, ModelOpts, weight_format};

/// Resolve `--backend` against the weights and the hardware present.
///
/// The generator's weights are one file, so "is this an ONNX artefact" is the
/// extension — which is the engine-specific half [`Backend::resolve`] leaves to
/// the caller.
pub fn resolve_backend(backend: Backend, model: &Path) -> Backend {
    backend.resolve(model.extension().and_then(|e| e.to_str()) == Some("onnx"))
}

/// Build a streaming [`Converter`] over the selected backend. Shared by
/// `convert` (batch preset) and the bare filter (realtime preset); the same
/// block/overlap/crossfade code drives either the ONNX or the Burn generator.
pub async fn build_converter(
    opts: &ModelOpts,
    feature_opts: FeatureBackendOpts,
    backend: Backend,
    device: DeviceSpec,
    transpose: i32,
    params: StreamParams,
    denoise: Option<DenoiseParams>,
) -> Result<Converter> {
    let conv_params = ConvertParams { transpose };
    let model = opts.model()?;
    let backend = resolve_backend(backend, model);
    let features = feature_opts.resolve(backend);

    // The pure-ORT path is **one fused pipeline, not three independent models**:
    // [`RvcModel`] is built from a single [`RvcConfig`]/[`ModelPaths`] naming all
    // three graphs at once, and there is no way to describe an ONNX generator
    // whose RMVPE runs on LibTorch to it. That is why this is an early return
    // and not a fourth arm of the match below — and why it demands that all
    // three agree. Any mixed configuration goes the generic way instead, which
    // composes a `FeatureExtractor` out of separately chosen parts and hands it
    // to a boxed `Generator`.
    //
    // Do not "simplify" this back to `if backend == Backend::Onnx`: that reads
    // as the same thing and quietly forces both feature models onto the
    // generator's runtime, discarding whatever the two override flags said.
    if backend == Backend::Onnx
        && features.content == Backend::Onnx
        && features.rmvpe == Backend::Onnx
    {
        let (content, rmvpe) = resolve_feature_models(opts, features).await?;
        let cfg = build_rvc_config(opts, content, rmvpe)?;
        let onnx = RvcModel::load(cfg).context("failed to load RVC models")?;
        return Ok(Converter::new(onnx, params, conv_params).with_denoise(denoise));
    }

    let (content, rmvpe) = resolve_feature_models(opts, features).await?;
    anyhow::ensure!(
        model.exists(),
        "generator weights not found: {} (train one with the `train` subcommand)",
        model.display()
    );
    tracing::info!(
        "loading Burn generator from {} ({backend}, device {device})",
        model.display(),
    );

    // The two feature models are built here, not inside each generator
    // constructor, because that is the only place that knows both backends —
    // and they need not agree with each other or with the generator. Each
    // constructor is `#[cfg]`-gated, so a `--no-default-features` build simply
    // has no arm to take and says so rather than failing to link.
    let features_extractor = tokio::task::block_in_place(|| {
        let content_model = build_content_encoder(features.content, &content, device)?;
        let rmvpe_model = build_pitch_estimator(features.rmvpe, &rmvpe, device)?;
        anyhow::Ok(rvc_core::FeatureExtractor::from_parts(
            content_model,
            rmvpe_model,
        ))
    })?;

    // Each arm erases into the same non-generic `Converter` — that type erasure
    // is what makes the backend a run-time choice. Annotated because a
    // `--no-default-features` build compiles every arm away and leaves nothing
    // to infer from.
    let converter: Converter = match backend {
        #[cfg(feature = "cuda")]
        Backend::Cuda => {
            let g = tokio::task::block_in_place(|| {
                rvc_core::cuda_generator(
                    features_extractor,
                    model,
                    opts.model_sr,
                    opts.speaker_id,
                    device,
                )
            })
            .context("failed to load the Burn generator on the CubeCL/CUDA backend")?;
            Converter::new(g, params, conv_params)
        }
        #[cfg(feature = "wgpu")]
        Backend::Wgpu => {
            let g = tokio::task::block_in_place(|| {
                rvc_core::wgpu_generator(
                    features_extractor,
                    model,
                    opts.model_sr,
                    opts.speaker_id,
                    device,
                )
            })
            .context("failed to load the Burn generator on the WebGPU backend")?;
            Converter::new(g, params, conv_params)
        }
        #[cfg(feature = "tch")]
        Backend::Tch => {
            let g = tokio::task::block_in_place(|| {
                rvc_core::libtorch_generator(
                    features_extractor,
                    model,
                    opts.model_sr,
                    opts.speaker_id,
                    device,
                )
            })
            .context("failed to load the Burn generator on the LibTorch backend")?;
            Converter::new(g, params, conv_params)
        }
        Backend::Onnx => unreachable!("handled above"),
        Backend::Auto => unreachable!("resolved above"),
        // Only reachable on a `--no-default-features` build.
        #[allow(unreachable_patterns)]
        other => return Err(other.unavailable()),
    };
    Ok(converter.with_denoise(denoise))
}

/// Build the content encoder on whichever backend `--content-vec-backend`
/// resolved to.
///
/// Boxed rather than `impl ContentEncoder` because these arms are alternatives:
/// two `impl Trait` returns are different opaque types, so a `match` over them
/// needs one type, and `FeatureExtractor::from_parts` takes boxes anyway.
fn build_content_encoder(
    backend: Backend,
    path: &std::path::Path,
    device: DeviceSpec,
) -> Result<Box<dyn rvc_core::ContentEncoder>> {
    let encoder = match backend {
        Backend::Onnx => rvc_core::onnx_content_encoder(path)
            .context("failed to load ContentVec on ONNX Runtime")?,
        #[cfg(feature = "cuda")]
        Backend::Cuda => rvc_core::cuda_content_encoder(path, device)
            .context("failed to load ContentVec on the CubeCL/CUDA backend")?,
        #[cfg(feature = "tch")]
        Backend::Tch => rvc_core::libtorch_content_encoder(path, device)
            .context("failed to load ContentVec on the LibTorch backend")?,
        #[cfg(feature = "wgpu")]
        Backend::Wgpu => rvc_core::wgpu_content_encoder(path, device)
            .context("failed to load ContentVec on the WebGPU backend")?,
        Backend::Auto => unreachable!("resolved before this point"),
        // Only reachable on a `--no-default-features` build.
        #[allow(unreachable_patterns)]
        other => return Err(other.unavailable()),
    };
    Ok(encoder)
}

/// Build the pitch estimator on whichever backend `--rmvpe-backend` resolved to.
///
/// Boxed for the same reason as [`build_content_encoder`].
fn build_pitch_estimator(
    backend: Backend,
    path: &std::path::Path,
    device: DeviceSpec,
) -> Result<Box<dyn rvc_core::PitchEstimator>> {
    let estimator = match backend {
        Backend::Onnx => {
            rvc_core::onnx_pitch_estimator(path).context("failed to load RMVPE on ONNX Runtime")?
        }
        #[cfg(feature = "cuda")]
        Backend::Cuda => {
            rvc_core::cuda_pitch_estimator(path, rvc_core::FeatureExtractor::F0_THRESHOLD, device)
                .context("failed to load RMVPE on the CubeCL/CUDA backend")?
        }
        #[cfg(feature = "tch")]
        Backend::Tch => rvc_core::libtorch_pitch_estimator(
            path,
            rvc_core::FeatureExtractor::F0_THRESHOLD,
            device,
        )
        .context("failed to load RMVPE on the LibTorch backend")?,
        #[cfg(feature = "wgpu")]
        Backend::Wgpu => {
            rvc_core::wgpu_pitch_estimator(path, rvc_core::FeatureExtractor::F0_THRESHOLD, device)
                .context("failed to load RMVPE on the WebGPU backend")?
        }
        Backend::Auto => unreachable!("resolved before this point"),
        // Only reachable on a `--no-default-features` build.
        #[allow(unreachable_patterns)]
        other => return Err(other.unavailable()),
    };
    Ok(estimator)
}

/// Resolve the ContentVec and RMVPE paths, downloading each in the weight
/// format its chosen backend can read.
///
/// The two formats are not interchangeable and not even the same shape: an ONNX
/// ContentVec is a single `.onnx` file where the PyTorch one is a *directory* of
/// weights, config and preprocessor. `--content`/`--rmvpe` bypass the choice
/// entirely, which is the escape hatch for a mirror this doesn't know about.
pub async fn resolve_feature_models(
    opts: &ModelOpts,
    features: FeatureBackends,
) -> Result<(PathBuf, PathBuf)> {
    let cache = opts.cache_dir.as_path();
    let content = match &opts.content {
        Some(p) => p.clone(),
        None => {
            tracing::info!(
                "resolving ContentVec ({}) from Hugging Face...",
                features.content
            );
            hub_kit::fetch_contentvec(weight_format(features.content), cache)
                .await
                .with_context(|| {
                    format!(
                        "failed to fetch ContentVec for {} (override the path with --content)",
                        features.content
                    )
                })?
        }
    };
    let rmvpe = match &opts.rmvpe {
        Some(p) => p.clone(),
        None => {
            tracing::info!("resolving RMVPE ({}) from Hugging Face...", features.rmvpe);
            hub_kit::fetch_rmvpe(weight_format(features.rmvpe), cache)
                .await
                .with_context(|| {
                    format!(
                        "failed to fetch RMVPE for {} (override the path with --rmvpe)",
                        features.rmvpe
                    )
                })?
        }
    };
    Ok((content, rmvpe))
}

/// Assemble the fused ORT pipeline config from paths already resolved by
/// [`resolve_feature_models`].
///
/// Takes the two paths rather than fetching them again: this is only reachable
/// when generator, ContentVec and RMVPE all run on ONNX Runtime, and fetching
/// inside would have to re-derive that fact from nothing.
pub fn build_rvc_config(opts: &ModelOpts, content: PathBuf, rmvpe: PathBuf) -> Result<RvcConfig> {
    let generator = opts.model()?;
    anyhow::ensure!(
        generator.exists(),
        "generator model not found: {} (train one with the `train` subcommand)",
        generator.display()
    );

    let mut cfg = RvcConfig::new(
        ModelPaths {
            content,
            rmvpe,
            generator: generator.to_path_buf(),
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
