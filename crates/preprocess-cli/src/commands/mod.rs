//! Stage implementations: one module per subcommand.

pub mod clip;
pub mod denoise;
pub mod separate;

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use preprocess_core::InputFile;

/// Create the output directory, then resolve the batch to process.
///
/// **In that order, and it matters**: [`preprocess_core::plan`] skips the
/// output directory so a re-run never re-ingests its own output, and it can
/// only recognise a directory that exists. Creating it afterwards would leave
/// `preprocess clip corpus/ -o corpus/clips` quietly doubling its input on the
/// second run.
///
/// Shared by every stage rather than written per command, so a stage added
/// later cannot get that ordering wrong by copying the wrong half.
pub async fn prepare(input: &[PathBuf], output_dir: &Path) -> Result<Vec<InputFile>> {
    tokio::fs::create_dir_all(output_dir)
        .await
        .with_context(|| format!("creating output dir {}", output_dir.display()))?;

    let files = preprocess_core::plan(input, output_dir)?;
    anyhow::ensure!(
        !files.is_empty(),
        "no audio files found in the given input paths: {}",
        input
            .iter()
            .map(|p| p.display().to_string())
            .collect::<Vec<_>>()
            .join(", ")
    );
    Ok(files)
}
