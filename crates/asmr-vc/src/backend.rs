//! Generator backend abstraction shared by the ONNX and Burn runtimes.
//!
//! Both the ONNX Runtime path ([`crate::RvcModel`]) and the native Burn path
//! ([`crate::BurnGenerator`], `burn` feature) implement [`Generator`], so the
//! streaming [`crate::Converter`] — with all its block/overlap/crossfade logic —
//! and the CLI drive either runtime through one code path.

use crate::config::ConvertParams;
use crate::error::Result;
use crate::rvc::RvcModel;

/// A loaded RVC generator: converts a mono 16 kHz analysis segment into audio at
/// [`Generator::output_sr`].
pub trait Generator: Send {
    /// The generator's output sample rate.
    fn output_sr(&self) -> u32;
    /// Convert one mono 16 kHz segment, returning audio at `output_sr()`.
    fn convert_segment(&mut self, wav16k: &[f32], params: ConvertParams) -> Result<Vec<f32>>;
}

impl Generator for RvcModel {
    fn output_sr(&self) -> u32 {
        RvcModel::output_sr(self)
    }

    fn convert_segment(&mut self, wav16k: &[f32], params: ConvertParams) -> Result<Vec<f32>> {
        RvcModel::convert_segment(self, wav16k, params)
    }
}
