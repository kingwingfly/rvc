//! ContentVec / HuBERT content-feature extraction.

use ort::session::Session;

use crate::analysis::ContentEncoder;
use crate::config::CONTENT_DIM;
use crate::dsp::{locate_axis, to_time_major};
use crate::error::Result;
use crate::session::run_single_f32;

/// Wraps the ContentVec ONNX session and yields time-major `[T, 768]` features.
pub struct OnnxContentEncoder {
    session: Session,
}

impl OnnxContentEncoder {
    /// Wrap an already-built session.
    pub fn new(session: Session) -> Self {
        Self { session }
    }
}

impl ContentEncoder for OnnxContentEncoder {
    /// Extract content features from a mono 16 kHz `f32` buffer.
    ///
    /// The audio is fed as a `[1, 1, L]` tensor (the shape used by the standard
    /// `content-vec-best` / `vec-768-layer-12` exports). The output is located
    /// by its 768-wide axis so `[1, T, 768]` and `[1, 768, T]` are both handled.
    fn extract(&mut self, wav16k: &[f32]) -> Result<Vec<Vec<f32>>> {
        let shape = vec![1i64, 1, wav16k.len() as i64];
        let (out_shape, out_data) = run_single_f32(&mut self.session, shape, wav16k.to_vec())?;
        let (axis, time) = locate_axis(&out_shape, CONTENT_DIM, "content features")?;
        Ok(to_time_major(
            &out_data,
            &out_shape,
            axis,
            CONTENT_DIM,
            time,
        ))
    }
}
