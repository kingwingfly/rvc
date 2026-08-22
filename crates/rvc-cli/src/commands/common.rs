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
/// `-m` names a file that is not there: a message, not two downloads.
///
/// `ModelOpts::model()` says only that the flag was *given*. This used to be
/// checked after `resolve_feature_models`, so a typo'd path fetched ContentVec
/// and RMVPE first — around 200 MB to report a typo.
fn ensure_generator_exists(model: &std::path::Path) -> Result<()> {
    anyhow::ensure!(
        model.exists(),
        "generator weights not found: {} (train one with the `train` subcommand)",
        model.display()
    );
    Ok(())
}

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
    // whose RMVPE runs on LibTorch to it. That is why it is an early return and
    // not a fourth arm of the match below.
    //
    // It is also why an ONNX generator is the **one** combination the generic
    // path cannot host either: every Burn generator constructor takes a prebuilt
    // `FeatureExtractor` and no ONNX one does, so there is no third place for a
    // mixed configuration to go. Hence the `ensure!` rather than a condition on
    // the `if`: falling through on disagreement reached `Backend::Onnx =>
    // unreachable!()` below — a panic, two downloads and two model loads after
    // the flags that caused it were parsed.
    //
    // **Do not turn the refusal back into a silent narrowing** by folding
    // `features.content == Onnx && features.rmvpe == Onnx` into the `if`. That
    // sends the mix down the generic path, which cannot build an ONNX generator,
    // so it only moves the panic; and dropping the check entirely forces both
    // feature models onto ORT, discarding whatever the two override flags said.
    if backend == Backend::Onnx {
        let mixed: Vec<&str> = [
            ("--content-vec-backend", features.content),
            ("--rmvpe-backend", features.rmvpe),
        ]
        .into_iter()
        .filter(|(_, chosen)| *chosen != Backend::Onnx)
        .map(|(flag, _)| flag)
        .collect();
        anyhow::ensure!(
            mixed.is_empty(),
            "{} cannot name a Burn backend while the generator is an `.onnx` export: ONNX \
             Runtime runs all three models as one fused pipeline, so there is nowhere to put \
             a feature model built elsewhere. Either drop the flag (or pass `onnx`) and \
             convert entirely on ONNX Runtime, or point `-m` at a `.safetensors` generator \
             and convert entirely on Burn.",
            mixed.join(" and "),
        );

        // Before any fetch, for the same reason the refusal above is: whether
        // `-m` names a file that exists is knowable from the command line
        // alone. It used to be checked after `resolve_feature_models`, so a
        // typo'd path downloaded ContentVec and RMVPE first — around 200 MB
        // spent to report a typo. It sits *after* the mixing check because a
        // command that is wrong in both ways should hear about the flags it
        // can fix rather than about a path it may have meant to create.
        ensure_generator_exists(model)?;
        let (content, rmvpe) = resolve_feature_models(opts, features).await?;
        let cfg = build_rvc_config(opts, content, rmvpe)?;
        let onnx = RvcModel::load(cfg).context("failed to load RVC models")?;
        return Ok(Converter::new(onnx, params, conv_params).with_denoise(denoise));
    }

    ensure_generator_exists(model)?;
    let (content, rmvpe) = resolve_feature_models(opts, features).await?;
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
        let rmvpe_model = build_pitch_estimator(features.rmvpe, &rmvpe, device, opts.f0_threshold)?;
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
        // Genuinely unreachable: the block above either returned the fused
        // pipeline or refused the mix that would have arrived here.
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
///
/// `threshold` is `--f0-threshold`, and every arm gets it: the voicing floor is
/// a property of the conversion, not of the runtime that computes it, and the
/// 7–87 frames the two runtimes already disagree about are frames sitting on
/// exactly this boundary. Handing one backend the flag and another the constant
/// would turn a tuning knob into a second reason the runtimes differ.
fn build_pitch_estimator(
    backend: Backend,
    path: &std::path::Path,
    device: DeviceSpec,
    threshold: f32,
) -> Result<Box<dyn rvc_core::PitchEstimator>> {
    let estimator = match backend {
        Backend::Onnx => rvc_core::onnx_pitch_estimator(path, threshold)
            .context("failed to load RMVPE on ONNX Runtime")?,
        #[cfg(feature = "cuda")]
        Backend::Cuda => rvc_core::cuda_pitch_estimator(path, threshold, device)
            .context("failed to load RMVPE on the CubeCL/CUDA backend")?,
        #[cfg(feature = "tch")]
        Backend::Tch => rvc_core::libtorch_pitch_estimator(path, threshold, device)
            .context("failed to load RMVPE on the LibTorch backend")?,
        #[cfg(feature = "wgpu")]
        Backend::Wgpu => rvc_core::wgpu_pitch_estimator(path, threshold, device)
            .context("failed to load RMVPE on the WebGPU backend")?,
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
    // The fused pipeline builds its own RMVPE from this config rather than from
    // `build_pitch_estimator`, so `--f0-threshold` has to be carried here too —
    // miss it and the flag parses, validates and does nothing whenever `-m`
    // names an `.onnx` export.
    cfg.f0_threshold = opts.f0_threshold;
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::args::CrossfadeShape;

    /// A scratch directory of this test's own. Every test here that names a
    /// cache wants one nothing has ever downloaded into, because "is it still
    /// empty afterwards" is the assertion.
    fn scratch(what: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "rvc-cli-{what}-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("scratch dir");
        dir
    }

    fn model_opts(model: &Path, cache: &Path) -> ModelOpts {
        ModelOpts {
            model: Some(model.to_path_buf()),
            model_sr: 48_000,
            content: None,
            rmvpe: None,
            cache_dir: cache.to_path_buf(),
            speaker_id: 0,
            f0_threshold: rvc_core::FeatureExtractor::F0_THRESHOLD,
        }
    }

    fn feature_opts(content: Option<Backend>, rmvpe: Option<Backend>) -> FeatureBackendOpts {
        FeatureBackendOpts {
            content_vec_backend: content,
            rmvpe_backend: rmvpe,
        }
    }

    /// Whether anything at all landed in the cache. The refusals below are only
    /// worth having if they cost a message rather than two downloads, and this
    /// is the only way to say that from inside a test.
    fn is_empty(dir: &Path) -> bool {
        std::fs::read_dir(dir)
            .map(|mut d| d.next().is_none())
            .unwrap_or(true)
    }

    /// The engine-specific half of backend resolution: the generator's weights
    /// are one file, so "is this an ONNX artefact" is its extension. Everything
    /// else `Backend::resolve` decides, and a named backend is never
    /// substituted — including the case that reads backwards, an explicit Burn
    /// backend against an `.onnx` path, which must survive to be *refused*
    /// downstream rather than being quietly rewritten to `onnx` here.
    #[test]
    fn the_extension_decides_auto_and_nothing_else() {
        assert_eq!(
            resolve_backend(Backend::Auto, Path::new("voice.onnx")),
            Backend::Onnx
        );
        for named in [Backend::Onnx, Backend::Cuda, Backend::Tch, Backend::Wgpu] {
            assert_eq!(
                resolve_backend(named, Path::new("voice.onnx")),
                named,
                "{named} against .onnx"
            );
            assert_eq!(
                resolve_backend(named, Path::new("voice.safetensors")),
                named,
                "{named} against .safetensors"
            );
        }
        // `auto` on Burn weights resolves on hardware alone, so the only thing
        // this can assert is what it must never become.
        assert_ne!(
            resolve_backend(Backend::Auto, Path::new("voice.safetensors")),
            Backend::Onnx
        );
        assert_ne!(
            resolve_backend(Backend::Auto, Path::new("voice.safetensors")),
            Backend::Auto
        );
    }

    /// The one combination that cannot be mixed, and the one property that
    /// makes refusing it worth anything: **it costs a message, not two
    /// downloads**. `RvcModel` is one fused pipeline built from a single
    /// `RvcConfig` naming all three graphs at once, so there is nowhere to put
    /// a feature model built elsewhere; and the generic path below cannot host
    /// an ONNX generator either, since every constructor that takes a prebuilt
    /// `FeatureExtractor` is a Burn one.
    ///
    /// The empty cache is the ordering assertion. Widen the condition back into
    /// the `if` and this falls through to `Backend::Onnx => unreachable!()` —
    /// two downloads and two model loads after the flag that caused it was
    /// parsed, which is the panic this `ensure!` replaced.
    #[tokio::test(flavor = "multi_thread")]
    async fn an_onnx_generator_refuses_a_burn_feature_model_before_fetching_anything() {
        for (flag, opts) in [
            (
                "--content-vec-backend",
                feature_opts(Some(Backend::Tch), None),
            ),
            ("--rmvpe-backend", feature_opts(None, Some(Backend::Cuda))),
        ] {
            let cache = scratch("mix");
            // The generator is not even on disk: the refusal must come first,
            // so *which* error arrives says which check ran first.
            let err = build_converter(
                &model_opts(Path::new("voice.onnx"), &cache),
                opts,
                Backend::Onnx,
                burn_kit::DeviceSpec::Auto,
                0,
                StreamParams::batch(),
                None,
            )
            .await
            .err()
            .expect("a Burn feature model under an .onnx generator must be refused")
            .to_string();
            assert!(err.contains(flag), "{err}");
            assert!(err.contains(".onnx"), "{err}");
            assert!(
                is_empty(&cache),
                "{flag}: the refusal fetched something into {}",
                cache.display()
            );
            let _ = std::fs::remove_dir_all(&cache);
        }
    }

    /// Both flags mixed names both, rather than the first one found — a message
    /// that names one of two offending flags sends the user round twice.
    #[tokio::test(flavor = "multi_thread")]
    async fn both_offending_flags_are_named_at_once() {
        let cache = scratch("mix-both");
        let err = build_converter(
            &model_opts(Path::new("voice.onnx"), &cache),
            feature_opts(Some(Backend::Tch), Some(Backend::Tch)),
            Backend::Auto,
            burn_kit::DeviceSpec::Auto,
            0,
            StreamParams::batch(),
            None,
        )
        .await
        .err()
        .expect("two Burn feature models under an .onnx generator must be refused")
        .to_string();
        assert!(err.contains("--content-vec-backend"), "{err}");
        assert!(err.contains("--rmvpe-backend"), "{err}");
        assert!(is_empty(&cache), "the refusal fetched something");
        let _ = std::fs::remove_dir_all(&cache);
    }

    /// `--content-vec-backend onnx` beside an `.onnx` generator is not a mix,
    /// so it must *not* be refused — the check has to be about disagreement,
    /// not about the flag having been typed. Reached by letting the fetch fail
    /// against an unusable cache: whatever error comes back, it is not the
    /// mixing refusal.
    #[tokio::test(flavor = "multi_thread")]
    async fn naming_onnx_out_loud_is_not_a_mix() {
        let blocker = std::env::temp_dir().join(format!(
            "rvc-cli-notdir-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        std::fs::write(&blocker, b"a file, not a directory").expect("blocker");
        // A cache path *under a regular file*: nothing can be written there, so
        // the fetch fails immediately and without a network round trip.
        let cache = blocker.join("cache");

        let err = build_converter(
            &model_opts(Path::new("voice.onnx"), &cache),
            feature_opts(Some(Backend::Onnx), Some(Backend::Auto)),
            Backend::Onnx,
            burn_kit::DeviceSpec::Auto,
            0,
            StreamParams::batch(),
            None,
        )
        .await
        .err()
        .expect("an unusable cache cannot produce a converter")
        .to_string();
        assert!(
            !err.contains("cannot name a Burn backend"),
            "onnx named out loud was read as a mix: {err}"
        );
        let _ = std::fs::remove_file(&blocker);
    }

    /// Every field of the fused pipeline's config, value by value, each set to
    /// something that is not its default — so a swapped pair or a forgotten
    /// line fails rather than coincidentally agreeing.
    ///
    /// `f0_threshold` is the one with history: the fused pipeline builds its own
    /// RMVPE from this config rather than from `build_pitch_estimator`, so miss
    /// it here and `--f0-threshold` parses, validates and does nothing whenever
    /// `-m` names an `.onnx` export.
    #[test]
    fn every_model_option_reaches_the_fused_config() {
        let dir = scratch("cfg");
        let generator = dir.join("voice.onnx");
        std::fs::write(&generator, b"").unwrap();

        let mut opts = model_opts(&generator, &dir);
        opts.model_sr = 40_000;
        opts.speaker_id = 7;
        opts.f0_threshold = 0.42;

        let content = PathBuf::from("/content/vec.onnx");
        let rmvpe = PathBuf::from("/rmvpe/rmvpe.onnx");
        let cfg = build_rvc_config(&opts, content.clone(), rmvpe.clone()).expect("config");

        assert_eq!(cfg.model_sr, 40_000);
        assert_eq!(cfg.speaker_id, 7);
        assert_eq!(cfg.f0_threshold, 0.42);
        assert_eq!(cfg.models.generator, generator);
        assert_eq!(cfg.models.content, content, "content path");
        assert_eq!(cfg.models.rmvpe, rmvpe, "rmvpe path");
        // The two feature paths are the pair a swap would hide: both are
        // `PathBuf`s of resolved weights, so nothing downstream disagrees about
        // the types and ORT would load each graph into the other's slot.
        assert_ne!(cfg.models.content, cfg.models.rmvpe);

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A `-m` that names nothing must cost a message, not two downloads.
    ///
    /// `models.model()` only says the flag was *given*; whether the file is
    /// there was checked after `resolve_feature_models`, so a typo'd path
    /// fetched ContentVec and RMVPE first — around 200 MB to report a typo. The
    /// empty cache afterwards is the whole assertion; the message is the easy
    /// half.
    #[tokio::test(flavor = "multi_thread")]
    async fn a_generator_that_is_not_there_is_reported_before_anything_is_fetched() {
        for name in ["missing.onnx", "missing.safetensors"] {
            let cache = scratch("missing");
            let err = build_converter(
                &model_opts(&cache.join(name), &cache),
                feature_opts(None, None),
                Backend::Auto,
                burn_kit::DeviceSpec::Auto,
                0,
                StreamParams::batch(),
                None,
            )
            .await
            .err()
            .expect("a generator that is not on disk cannot be loaded");
            // The whole chain, not just the top: the refusal is raised where
            // the path is known and may be wrapped by the loader above it.
            let err = format!("{err:#}");
            assert!(err.contains(name), "{err}");
            assert!(
                is_empty(&cache),
                "{name}: reporting a missing generator fetched into {}",
                cache.display()
            );
            let _ = std::fs::remove_dir_all(&cache);
        }
    }

    /// The `owner/name:file` escape hatch, which is how a user points at a
    /// mirror this crate has never heard of. `rsplit_once(':')` and
    /// `split_once('/')` in that order is what lets a *file* carry slashes —
    /// which the PyTorch ContentVec's `hubert_base/…` names do.
    #[test]
    fn a_model_override_splits_on_the_last_colon_and_the_first_slash() {
        let r = parse_model_ref("lj1995/VoiceConversionWebUI:rmvpe.pt").unwrap();
        assert_eq!(r.owner, "lj1995");
        assert_eq!(r.name, "VoiceConversionWebUI");
        assert_eq!(r.file, "rmvpe.pt");

        let nested =
            parse_model_ref("lj1995/VoiceConversionWebUI:hubert_base/config.json").unwrap();
        assert_eq!(nested.name, "VoiceConversionWebUI");
        assert_eq!(nested.file, "hubert_base/config.json");

        for bad in ["no-colon-here", "nocolon/or/slash", "owner:file"] {
            assert!(
                parse_model_ref(bad).is_err(),
                "{bad} is not `owner/name:file`"
            );
        }
    }

    /// The de-hiss switch is this engine's and the tuning is shared, so the two
    /// halves have to meet correctly: off means `None` whatever the knobs say,
    /// and on carries all three through. `patch_secs` and `research_secs` are
    /// both small floats a millisecond apart — a swapped pair is invisible
    /// everywhere except here.
    #[test]
    fn the_denoise_flags_reach_the_filter_only_when_the_stage_is_on() {
        let tuning = cli_kit::DenoiseOpts {
            denoise_strength: 0.05,
            denoise_patch: 0.003,
            denoise_research: 0.011,
        };
        assert!(
            crate::args::DenoiseOpts {
                denoise: false,
                tuning,
            }
            .params()
            .is_none(),
            "the tuning flags must not turn the stage on by themselves"
        );

        let params = crate::args::DenoiseOpts {
            denoise: true,
            tuning,
        }
        .params()
        .expect("--denoise turns the stage on");
        assert_eq!(params.strength, 0.05);
        assert_eq!(params.patch_secs, 0.003);
        assert_eq!(params.research_secs, 0.011);
    }

    /// The crossfade curve is the one geometry value that is not a length, so
    /// the three `resolve` tests in `args.rs` cannot catch a mis-mapped variant.
    /// `linear` is the default and stays the default: every voice this toolkit
    /// has converted was joined with it.
    #[test]
    fn the_crossfade_curve_maps_variant_for_variant() {
        assert_eq!(
            rvc_core::CrossfadeShape::from(CrossfadeShape::Linear),
            rvc_core::CrossfadeShape::Linear
        );
        assert_eq!(
            rvc_core::CrossfadeShape::from(CrossfadeShape::EqualPower),
            rvc_core::CrossfadeShape::EqualPower
        );
    }
}
