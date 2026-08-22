//! Shared setup: resolve the four checkpoints, load them, and analyse the
//! reference against them.
//!
//! Everything the two conversion paths share, and no more — they diverge after
//! this. The filter has to stream, so it drives
//! [`Converter`](seedvc_core::Converter); `convert` has whole files and so drives
//! [`seedvc_core::convert`], which knows each source's length up front and can
//! therefore balance its last chunk instead of leaving a fragment.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use burn_kit::DeviceSpec;
use seedvc_core::{Model, ModelPaths, Reference};

use crate::args::{Backend, ModelOpts};

/// Whisper's weights inside the repo directory.
///
/// `--content` and [`hub_kit::SeedVcPaths::whisper`] are both **directories**,
/// because Whisper's `config.json` and BigVGAN's share a name and a flat layout
/// would hand each loader the other's; [`ModelPaths::content`] wants the weights
/// file itself. Joining is safe because this is the one name Whisper fixes
/// across a Hub snapshot, a git clone and a hand-made copy alike.
const WHISPER_WEIGHTS: &str = "model.safetensors";

/// Where the four networks are, after the override flags and the cache have both
/// had their say.
pub struct Paths {
    pub checkpoint: PathBuf,
    pub campplus: PathBuf,
    pub bigvgan: PathBuf,
    /// Whisper's weights *file*, not the directory `--content` names.
    pub content: PathBuf,
}

/// Resolve the four checkpoints, downloading only what no flag supplied.
///
/// [`hub_kit::seedvc_paths`] is deliberately **not** used here. It locates all
/// four inside one hand-assembled directory, and no flag on this engine names
/// such a directory: the four overrides are independent files from four separate
/// repos, and `--cache-dir` is the cache rather than a bundle. Someone who
/// already holds the weights in one folder points the four flags at them, which
/// is also the only way to be explicit about *which* file is which — the reason
/// `seedvc_paths` has to guess from names at all.
pub async fn resolve_paths(opts: &ModelOpts) -> Result<Paths> {
    // Each resolved on its own, so a flag suppresses exactly the download it
    // replaces and no more. Naming all four therefore touches no network and
    // needs no cache directory to exist — a user who was handed the weights
    // should not have a download attempted behind their back — and naming one of
    // them saves that one file rather than nothing.
    let context = "(name weights you already hold with --checkpoint, --campplus, --bigvgan \
                   and --content)";
    let checkpoint = match &opts.checkpoint {
        Some(p) => p.clone(),
        None => hub_kit::fetch_seedvc_checkpoint(&opts.cache_dir)
            .await
            .with_context(|| format!("failed to fetch the Seed-VC checkpoint {context}"))?,
    };
    let campplus = match &opts.campplus {
        Some(p) => p.clone(),
        None => hub_kit::fetch_campplus(&opts.cache_dir)
            .await
            .with_context(|| format!("failed to fetch the CAMPPlus timbre encoder {context}"))?,
    };
    let bigvgan = match &opts.bigvgan {
        Some(p) => p.clone(),
        // The config half is dropped: nothing reads it, and it is fetched at all
        // only so a hand-assembled directory can be identified by `hub-kit`.
        None => {
            hub_kit::fetch_bigvgan(&opts.cache_dir)
                .await
                .with_context(|| format!("failed to fetch the BigVGAN vocoder {context}"))?
                .0
        }
    };
    let content = match &opts.content {
        Some(p) => p.clone(),
        None => {
            hub_kit::fetch_whisper(Some(hub_kit::SEEDVC_WHISPER), &opts.cache_dir)
                .await
                .with_context(|| format!("failed to fetch the Whisper content encoder {context}"))?
                .dir
        }
    };

    Ok(Paths {
        checkpoint,
        campplus,
        bigvgan,
        content: content.join(WHISPER_WEIGHTS),
    })
}

/// Load the model — the four checkpoints, or the six ONNX graphs when `--onnx`
/// names an export — and analyse the reference against it.
///
/// Hands back both rather than a built converter, because the two commands want
/// different things from them: the filter wraps them in a
/// [`Converter`](seedvc_core::Converter), while `convert` drives
/// [`seedvc_core::convert`], which balances a file's last chunk in a way a
/// stream cannot.
///
/// The reference is analysed **once**, here, rather than per file or per chunk:
/// it costs a Whisper encode, a CAMPPlus pass and a mel, and none of that
/// depends on the source. That is what makes a batch of files cheaper than the
/// same files through the filter one at a time.
pub async fn load_model(
    opts: &ModelOpts,
    backend: Backend,
    device: DeviceSpec,
) -> Result<(Box<dyn Model>, Reference)> {
    let reference = opts.reference()?;
    // Asked before the fetch, not after: `load` checks this again, but by then a
    // cold cache has already spent a gigabyte on a backend that was never going
    // to run. The two calls are the same function, so they cannot disagree. The
    // export directory is what lets `auto` settle on ONNX Runtime at all.
    let backend = seedvc_core::backend::resolve(backend, opts.onnx.is_some())?;

    // `--onnx` short-circuits the four checkpoints entirely: a graph carries its
    // weights, so there is nothing to fetch and nothing to override. Reaching
    // here with a directory means `resolve` settled on ONNX Runtime — it refuses
    // `--onnx` beside a named Burn backend rather than letting one win silently,
    // which is what this branch used to do.
    let model: Box<dyn Model> = if let Some(dir) = &opts.onnx {
        load_onnx(dir)?
    } else {
        let paths = resolve_paths(opts).await?;
        // Four checkpoints and roughly a gigabyte of tensors, all of it
        // synchronous: `block_in_place` keeps it off the runtime's worker
        // threads, as the other engines' loaders do. No `context` here —
        // `seedvc_core::Error` already names the file that failed.
        tokio::task::block_in_place(|| {
            seedvc_core::load(
                &ModelPaths {
                    dit: &paths.checkpoint,
                    campplus: &paths.campplus,
                    bigvgan: &paths.bigvgan,
                    content: &paths.content,
                },
                backend,
                device,
            )
        })?
    };

    let analysed = seedvc_core::reference::analyse(model.as_ref(), reference, opts.reference_secs)
        .await
        .with_context(|| format!("analysing the reference {}", reference.display()))?;
    // The frame count is worth logging because it is the number that decides how
    // much source fits in a chunk — a long reference silently buys short chunks,
    // and this is where that becomes visible. It is also the reading that says
    // whether `--reference-secs` bit: a clip under the cap is unchanged by it.
    tracing::info!(
        "reference {}: {} mel frames ({:.2} s, capped at {} s)",
        reference.display(),
        analysed.frames,
        analysed.frames as f32 / model.config().frame_rate(),
        opts.reference_secs,
    );

    Ok((model, analysed))
}

/// Open the six ONNX graphs from the directory `--onnx` names.
///
/// Compiled only with the feature that can open them, so a build without ONNX
/// Runtime fails with a rebuild hint rather than a type error about a module
/// that is not there — the same shape `tts-cli`'s loader uses. The four
/// checkpoint flags are deliberately ignored: the graphs carry their weights.
#[cfg(feature = "onnx")]
fn load_onnx(dir: &Path) -> Result<Box<dyn Model>> {
    // The Burn path logs the same line inside `seedvc_core::load`; here there is
    // no backend or Burn device to name (ORT picks its own execution provider),
    // so the useful fact is which export directory the graphs come from.
    tracing::info!("loading Seed-VC (onnx, from {})", dir.display());
    let model = tokio::task::block_in_place(|| seedvc_core::onnx_model::OnnxModel::load(dir))?;
    Ok(Box::new(model))
}

#[cfg(not(feature = "onnx"))]
fn load_onnx(_dir: &Path) -> Result<Box<dyn Model>> {
    Err(anyhow::anyhow!(
        "this binary was built without the `onnx` feature — rebuild with `--features onnx` to \
         run the six graphs `export/export_seedvc.py` writes"
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A cache directory nothing can be written into, so a fetch that is
    /// attempted fails with *its own* message rather than quietly succeeding
    /// against a warm cache on the machine this happens to run on. That is what
    /// makes the assertions below about *ordering* rather than about wording: a
    /// `load_model` that resolved paths first would come back saying it failed
    /// to fetch a checkpoint, not saying which flag was wrong.
    const UNUSABLE_CACHE: &str = "/dev/null/seedvc-cli-must-not-fetch";

    fn opts(backend_dir: Option<&str>) -> ModelOpts {
        ModelOpts {
            reference: Some(PathBuf::from("me.wav")),
            reference_secs: seedvc_core::reference::REFERENCE_SECONDS,
            checkpoint: None,
            campplus: None,
            bigvgan: None,
            content: None,
            onnx: backend_dir.map(PathBuf::from),
            cache_dir: PathBuf::from(UNUSABLE_CACHE),
        }
    }

    /// The message `load_model` refused with.
    ///
    /// Matched rather than `expect_err`ed: `Box<dyn Model>` is not `Debug`, and
    /// making it one would be a bound on every implementation for the sake of
    /// three tests.
    async fn refuse(opts: &ModelOpts, backend: Backend) -> String {
        match load_model(opts, backend, DeviceSpec::Auto).await {
            Ok(_) => panic!("`--backend {backend}` was accepted"),
            Err(e) => format!("{e:#}"),
        }
    }

    /// `--backend onnx` with no `--onnx` has to be refused **before** a byte is
    /// fetched. `seedvc_core::backend::resolve` already pins the wording; what
    /// this pins is that `load_model` asks it first, which is the whole reason
    /// `resolve` is split out of `load` at all — a cold cache would otherwise
    /// spend a gigabyte on four checkpoints in order to reject the backend
    /// afterwards.
    #[tokio::test]
    async fn onnx_without_an_export_dir_is_refused_before_anything_is_fetched() {
        let err = refuse(&opts(None), Backend::Onnx).await;
        assert!(err.contains("--onnx"), "{err}");
        assert!(
            !err.contains("failed to fetch"),
            "the refusal arrived after a download was attempted: {err}"
        );
    }

    /// The mirror, and the one that used to be silent: `--onnx` beside a Burn
    /// backend asks for two runtimes at once, and the loader used to take the
    /// graphs regardless — discarding a backend the user had named out loud.
    ///
    /// Nothing is fetched down this branch either way (a graph carries its
    /// weights), so what this pins is the *asking*: restoring the old shape —
    /// `if opts.onnx.is_some() { Backend::Onnx } else { resolve(..) }` — makes
    /// this test fail with `load_onnx`'s message instead of the refusal, which
    /// is exactly the silent substitution.
    #[tokio::test]
    async fn a_burn_backend_beside_an_export_dir_is_refused_before_anything_is_fetched() {
        for backend in [Backend::Tch, Backend::Cuda, Backend::Wgpu] {
            let err = refuse(&opts(Some("/nonexistent")), backend).await;
            assert!(err.contains("--onnx"), "{backend}: {err}");
            assert!(
                !err.contains("failed to fetch"),
                "{backend}: the refusal arrived after a download was attempted: {err}"
            );
        }
    }

    /// The reference is the whole speaker specification, so its absence is
    /// reported before either of the two above — and before a fetch, which is
    /// what `ModelOpts::reference` being called at the top of `load_model` buys.
    #[tokio::test]
    async fn a_missing_reference_is_reported_before_anything_is_fetched() {
        let mut opts = opts(None);
        opts.reference = None;
        let err = refuse(&opts, Backend::Auto).await;
        assert!(err.contains("--reference"), "{err}");
        assert!(!err.contains("failed to fetch"), "{err}");
    }
}
