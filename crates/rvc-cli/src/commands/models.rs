//! `rvc models download` — prefetch the shared ONNX assets.

use anyhow::Result;

use crate::args::{ModelsArgs, ModelsCommand, ModelsDownloadArgs};
use crate::commands::common::parse_model_ref;

pub async fn run(args: ModelsArgs) -> Result<()> {
    match args.command {
        ModelsCommand::Download(a) => download(a).await,
    }
}

async fn download(a: ModelsDownloadArgs) -> Result<()> {
    let content = a.content.as_deref().map(parse_model_ref).transpose()?;
    let rmvpe = a.rmvpe.as_deref().map(parse_model_ref).transpose()?;

    let assets = rvc_hub::fetch_shared(content, rmvpe, a.cache_dir.as_deref()).await?;
    // These paths are useful to the user, so print them on stdout.
    println!("contentvec: {}", assets.contentvec.display());
    println!("rmvpe:      {}", assets.rmvpe.display());
    Ok(())
}
