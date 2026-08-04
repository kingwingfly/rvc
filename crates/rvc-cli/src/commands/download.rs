//! `rvc download` — prefetch the weights a conversion would otherwise fetch on
//! its first run.

use anyhow::Result;

use crate::args::DownloadArgs;
use crate::commands::common::parse_model_ref;

pub async fn run(a: DownloadArgs) -> Result<()> {
    let content = a.content.as_deref().map(parse_model_ref).transpose()?;
    let rmvpe = a.rmvpe.as_deref().map(parse_model_ref).transpose()?;

    // `rvc download` has no `--content-vec-backend`/`--rmvpe-backend` of its
    // own yet, so it prefetches what a default bare invocation would: ONNX.
    let assets = hub_kit::fetch_shared(
        content,
        hub_kit::WeightFormat::Onnx,
        rmvpe,
        hub_kit::WeightFormat::Onnx,
        &a.cache_dir,
    )
    .await?;
    // Where they landed is the point of the command, so it goes to stdout.
    println!("contentvec: {}", assets.contentvec.display());
    println!("rmvpe:      {}", assets.rmvpe.display());
    Ok(())
}
