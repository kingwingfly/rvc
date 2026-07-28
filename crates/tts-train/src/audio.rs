//! Reading corpus audio.
//!
//! `audio-kit` decodes as a `futures::Stream`, which is right for the streaming
//! CLIs and wrong here: preparation is a blocking loop over files, and a clip
//! has to be whole before cnhubert sees it.

use std::path::Path;

use crate::error::{Result, TrainError};

/// Decode a file to mono `f32` at `sample_rate`.
pub fn read(path: &Path, sample_rate: u32) -> Result<Vec<f32>> {
    use futures::StreamExt;

    let opts = audio_kit::DecodeOptions::new(sample_rate);
    let path = path.to_path_buf();
    futures::executor::block_on(async move {
        let mut stream = Box::pin(audio_kit::decode_path(path.clone(), opts));
        let mut out = Vec::new();
        while let Some(chunk) = stream.next().await {
            out.extend_from_slice(
                &chunk.map_err(|e| TrainError::Corpus(format!("{}: {e}", path.display())))?,
            );
        }
        Ok(out)
    })
}
