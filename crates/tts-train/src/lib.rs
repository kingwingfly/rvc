//! Fine-tuning GPT-SoVITS, natively.
//!
//! Two stages and they adapt different things. [`s1`] is next-token prediction
//! over semantic tokens and teaches *delivery* — pacing, emphasis, where a
//! speaker breathes. [`s2`] is the VITS adversarial loop and teaches *timbre*.
//! Few-shot cloning from a reference clip already works without either; this is
//! for when a particular voice is worth more than a few seconds of prompt.
//!
//! Both consume a corpus of audio beside transcripts, which is what `stt` is for.

pub mod audio;
pub mod dataset;
mod error;
pub mod s1;
pub mod s2;

pub use dataset::{Clip, SAMPLES_PER_FRAME, encoders, pairs, prepare};
pub use error::{Result, TrainError};
pub use s1::S1Settings;
pub use s2::S2Settings;

use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor};

/// Read an integer tensor as ids, whichever width the backend stores.
///
/// `Int` is i64 on LibTorch and i32 elsewhere, so the width is asked for rather
/// than assumed.
fn ids<B: Backend>(t: Tensor<B, 2, Int>) -> Result<Vec<u32>> {
    let data = t.into_data();
    if let Ok(v) = data.to_vec::<i64>() {
        return Ok(v.into_iter().map(|x| x as u32).collect());
    }
    data.to_vec::<i32>()
        .map(|v| v.into_iter().map(|x| x as u32).collect())
        .map_err(|e| TrainError::Weights(format!("token ids: {e:?}")))
}
