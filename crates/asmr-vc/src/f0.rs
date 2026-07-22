//! RMVPE fundamental-frequency (F0) extraction.

use ort::session::Session;

use crate::config::RMVPE_BINS;
use crate::dsp::{locate_axis, rmvpe_decode, to_time_major};
use crate::error::Result;
use crate::mel::RmvpeMel;
use crate::session::run_single_f32;

/// Wraps the RMVPE ONNX session and yields an F0 (Hz) contour at 100 Hz.
pub struct F0Estimator {
    session: Session,
    mel: RmvpeMel,
    threshold: f32,
}

impl F0Estimator {
    /// Wrap an already-built session with the given voicing threshold.
    pub fn new(session: Session, threshold: f32) -> Self {
        Self { session, mel: RmvpeMel::new(), threshold }
    }

    /// Estimate the F0 (Hz) contour from a mono 16 kHz `f32` buffer.
    ///
    /// RMVPE consumes a `[1, 128, T]` log-mel magnitude spectrogram and emits a
    /// `[1, T, 360]` cents-bin salience, which we decode to Hz. The UNet halves
    /// time several times, so the frame count is reflect-padded to a multiple of
    /// 32 before inference and the salience is trimmed back afterwards.
    pub fn extract(&mut self, wav16k: &[f32]) -> Result<Vec<f32>> {
        let (n_mels, time, data) = self.mel.compute(wav16k);
        if time == 0 {
            return Ok(Vec::new());
        }
        let time_pad = round_up_32(time);
        let padded = pad_time_reflect(&data, n_mels, time, time_pad);

        let shape = vec![1i64, n_mels as i64, time_pad as i64];
        let (out_shape, out_data) = run_single_f32(&mut self.session, shape, padded)?;
        let (axis, out_time) = locate_axis(&out_shape, RMVPE_BINS, "rmvpe salience")?;
        let mut salience = to_time_major(&out_data, &out_shape, axis, RMVPE_BINS, out_time);
        salience.truncate(time); // drop the padded frames
        Ok(rmvpe_decode(&salience, self.threshold))
    }
}

/// Round up to the next multiple of 32 (RMVPE UNet alignment).
fn round_up_32(t: usize) -> usize {
    32 * ((t - 1) / 32 + 1)
}

/// Reflect-pad a mel-major `[n_mels * time]` buffer along time to `time_pad`.
fn pad_time_reflect(data: &[f32], n_mels: usize, time: usize, time_pad: usize) -> Vec<f32> {
    if time_pad == time {
        return data.to_vec();
    }
    let mut out = vec![0.0f32; n_mels * time_pad];
    for d in 0..n_mels {
        let src = &data[d * time..d * time + time];
        let dst = &mut out[d * time_pad..d * time_pad + time_pad];
        dst[..time].copy_from_slice(src);
        for j in 0..(time_pad - time) {
            // numpy 'reflect': mirror without repeating the edge sample.
            let idx = (time as isize - 2 - j as isize).max(0) as usize;
            dst[time + j] = src[idx];
        }
    }
    out
}
