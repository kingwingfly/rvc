//! `rvc download` — prefetch the weights a conversion would otherwise fetch on
//! its first run.

use anyhow::Result;

use crate::args::{DownloadArgs, weight_format};
use crate::commands::common::parse_model_ref;

pub async fn run(a: DownloadArgs) -> Result<()> {
    // Before the first fetch, so a stalled transfer is bounded rather than
    // discovered — and before `parse_model_ref` too, since a bad
    // `--download-timeout` should not wait for a bad `--content` to be reported.
    a.download.install()?;
    let content = a.content.as_deref().map(parse_model_ref).transpose()?;
    let rmvpe = a.rmvpe.as_deref().map(parse_model_ref).transpose()?;

    // Prefetching is only worth anything if it fetches the files the next run
    // will actually load, so this resolves the backends exactly as a conversion
    // does. There is no `-m` here, so `auto` has no weights to inspect and
    // resolves on hardware alone — `--backend onnx` is how a user with an
    // `.onnx` generator says so.
    let features = a.features.resolve(a.backend.resolve(false));
    tracing::info!(
        "fetching ContentVec for {} and RMVPE for {}",
        features.content,
        features.rmvpe
    );

    let assets = hub_kit::fetch_shared(
        content,
        weight_format(features.content),
        rmvpe,
        weight_format(features.rmvpe),
        &a.cache_dir,
    )
    .await?;
    // Where they landed is the point of the command, so it goes to stdout.
    println!("contentvec: {}", assets.contentvec.display());
    println!("rmvpe:      {}", assets.rmvpe.display());
    Ok(())
}
