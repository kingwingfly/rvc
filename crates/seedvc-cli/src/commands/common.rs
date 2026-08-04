//! Shared setup: resolve the four checkpoints, load them, analyse the reference,
//! and hand back the [`Converter`] both conversion paths drive.
//!
//! The two paths differ only in their [`StreamParams`] — the filter takes
//! [`StreamParams::realtime`] and `convert` takes [`StreamParams::batch`] — so
//! everything up to that point lives here once, exactly as `rvc-cli`'s
//! `build_converter` does.

use std::path::PathBuf;

use anyhow::{Context, Result};
use burn_kit::DeviceSpec;
use seedvc_core::{ConvertOptions, Converter, ModelPaths, StreamParams};

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
    // Matched rather than fetched-then-overridden so that a fully specified run
    // touches no network and needs no cache directory to exist: a user who was
    // handed the weights should not have a download attempted behind their back.
    Ok(
        match (
            &opts.checkpoint,
            &opts.campplus,
            &opts.bigvgan,
            &opts.content,
        ) {
            (Some(checkpoint), Some(campplus), Some(bigvgan), Some(content)) => Paths {
                checkpoint: checkpoint.clone(),
                campplus: campplus.clone(),
                bigvgan: bigvgan.clone(),
                content: content.join(WHISPER_WEIGHTS),
            },
            _ => {
                let fetched = hub_kit::fetch_seedvc(&opts.cache_dir).await.context(
                    "failed to fetch the Seed-VC models (name weights you already hold with \
                     --checkpoint, --campplus, --bigvgan and --content)",
                )?;
                Paths {
                    checkpoint: opts.checkpoint.clone().unwrap_or(fetched.checkpoint),
                    campplus: opts.campplus.clone().unwrap_or(fetched.campplus),
                    bigvgan: opts.bigvgan.clone().unwrap_or(fetched.bigvgan),
                    content: opts
                        .content
                        .clone()
                        .unwrap_or(fetched.whisper)
                        .join(WHISPER_WEIGHTS),
                }
            }
        },
    )
}

/// Load the model, analyse the reference, and build the converter around both.
///
/// The reference is analysed **once**, here, rather than per file or per chunk:
/// it costs a Whisper encode, a CAMPPlus pass and a mel, and none of that
/// depends on the source. That is what makes a batch of files cheaper than the
/// same files through the filter one at a time.
pub async fn build_converter(
    opts: &ModelOpts,
    backend: Backend,
    device: DeviceSpec,
    params: StreamParams,
    convert: ConvertOptions,
) -> Result<Converter> {
    let reference = opts.reference()?;
    let paths = resolve_paths(opts).await?;

    // Four checkpoints and roughly a gigabyte of tensors, all of it synchronous:
    // `block_in_place` keeps it off the runtime's worker threads, as the other
    // engines' loaders do. No `context` here — `seedvc_core::Error` already
    // names both the file that failed and, for `--backend onnx`, why no rebuild
    // would help.
    let model = tokio::task::block_in_place(|| {
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
    })?;

    let analysed = seedvc_core::reference::analyse(model.as_ref(), reference)
        .await
        .with_context(|| format!("analysing the reference {}", reference.display()))?;
    // The frame count is worth logging because it is the number that decides how
    // much source fits in a chunk — a long reference silently buys short chunks,
    // and this is where that becomes visible.
    tracing::info!(
        "reference {}: {} mel frames ({:.2} s)",
        reference.display(),
        analysed.frames,
        analysed.frames as f32 / model.config().frame_rate(),
    );

    Ok(Converter::new(model, analysed, params, convert)?)
}
