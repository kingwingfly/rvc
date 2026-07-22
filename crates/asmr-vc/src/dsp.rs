//! Pure DSP helpers for the RVC pipeline: coarse-pitch quantization, feature
//! upsampling, RMVPE cents decoding, and tensor-axis location.
//!
//! These mirror the reference RVC (`infer/lib`) numerics so a generator trained
//! by the Python pipeline sees inputs shaped exactly as it expects.

use crate::config::RMVPE_BINS;
use crate::error::{Result, VcError};

/// Locate the axis of a 3-D tensor whose length equals `size`, returning
/// `(axis_index, time_len)`. Used to accept ContentVec/RMVPE outputs regardless
/// of whether they come out as `[1, T, D]` or `[1, D, T]`.
pub fn locate_axis(shape: &[i64], size: usize, what: &'static str) -> Result<(usize, usize)> {
    let dims: Vec<usize> = shape.iter().map(|&d| d.max(0) as usize).collect();
    if let Some(axis) = dims.iter().position(|&d| d == size) {
        // Time is the product of the remaining (non-batch, non-feature) dims.
        let time: usize = dims
            .iter()
            .enumerate()
            .filter(|&(i, &d)| i != axis && d != 1)
            .map(|(_, &d)| d)
            .product::<usize>()
            .max(1);
        Ok((axis, time))
    } else {
        Err(VcError::Shape { what, expected: size, got: shape.to_vec() })
    }
}

/// Convert a flat `[1, T, D]`-or-`[1, D, T]` buffer into a row-major `[T, D]`
/// matrix (as `Vec<Vec<f32>>`), given the located feature axis.
pub fn to_time_major(data: &[f32], shape: &[i64], feat_axis: usize, dim: usize, time: usize) -> Vec<Vec<f32>> {
    let dims: Vec<usize> = shape.iter().map(|&d| d.max(0) as usize).collect();
    let mut out = vec![vec![0.0f32; dim]; time];
    // Two supported layouts on a batch-1 tensor.
    if feat_axis == dims.len() - 1 {
        // [.., T, D] contiguous: row t is data[t*D .. t*D+D].
        for t in 0..time {
            out[t].copy_from_slice(&data[t * dim..t * dim + dim]);
        }
    } else {
        // [.., D, T]: element (d, t) at d*T + t.
        for t in 0..time {
            for d in 0..dim {
                out[t][d] = data[d * time + t];
            }
        }
    }
    out
}

/// Upsample time-major features by an integer factor using nearest-neighbour
/// (each frame repeated `factor` times), matching RVC's `F.interpolate` x2.
pub fn upsample_rows(rows: &[Vec<f32>], factor: usize) -> Vec<Vec<f32>> {
    let mut out = Vec::with_capacity(rows.len() * factor);
    for r in rows {
        for _ in 0..factor {
            out.push(r.clone());
        }
    }
    out
}

/// Quantize an F0 (Hz) contour to RVC coarse pitch indices in `[1, 255]`.
/// Unvoiced frames (`f0 <= 0`) map to `1`.
pub fn f0_to_coarse(f0: &[f32]) -> Vec<i64> {
    const F0_MIN: f32 = 50.0;
    const F0_MAX: f32 = 1100.0;
    let mel_min = 1127.0 * (1.0 + F0_MIN / 700.0).ln();
    let mel_max = 1127.0 * (1.0 + F0_MAX / 700.0).ln();
    f0.iter()
        .map(|&f| {
            let mut mel = 1127.0 * (1.0 + f / 700.0).ln();
            if mel > 0.0 {
                mel = (mel - mel_min) * 254.0 / (mel_max - mel_min) + 1.0;
            }
            let v = mel.round();
            (v.clamp(1.0, 255.0)) as i64
        })
        .collect()
}

/// Decode an RMVPE salience matrix into an F0 (Hz) contour.
///
/// `salience` is `time` rows of [`RMVPE_BINS`] cents-bin activations. For each
/// frame we take a ±4-bin weighted average of the cents mapping around the peak;
/// frames whose peak is below `threshold` are unvoiced (0 Hz).
pub fn rmvpe_decode(salience: &[Vec<f32>], threshold: f32) -> Vec<f32> {
    // cents_mapping[i] = 20*i + 1997.3794084376191 (RMVPE reference constant).
    let cents = |i: usize| 20.0f32 * i as f32 + 1997.3794;
    salience
        .iter()
        .map(|frame| {
            debug_assert_eq!(frame.len(), RMVPE_BINS);
            let (peak, &peak_val) = frame
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .unwrap_or((0, &0.0));
            if peak_val < threshold {
                return 0.0;
            }
            let lo = peak.saturating_sub(4);
            let hi = (peak + 5).min(RMVPE_BINS);
            let mut num = 0.0f32;
            let mut den = 0.0f32;
            for (i, &w) in frame.iter().enumerate().take(hi).skip(lo) {
                num += w * cents(i);
                den += w;
            }
            if den <= 0.0 {
                return 0.0;
            }
            let c = num / den;
            10.0 * 2.0f32.powf(c / 1200.0)
        })
        .collect()
}

/// Apply a semitone pitch shift to an F0 (Hz) contour in place, leaving
/// unvoiced (0) frames untouched.
pub fn shift_pitch(f0: &mut [f32], semitones: i32) {
    if semitones == 0 {
        return;
    }
    let factor = 2.0f32.powf(semitones as f32 / 12.0);
    for f in f0.iter_mut() {
        if *f > 0.0 {
            *f *= factor;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coarse_pitch_bounds_and_unvoiced() {
        let coarse = f0_to_coarse(&[0.0, 50.0, 300.0, 1100.0, 5000.0]);
        // Unvoiced maps to 1; everything stays within [1, 255].
        assert_eq!(coarse[0], 1);
        assert!(coarse.iter().all(|&c| (1..=255).contains(&c)));
        // Higher pitch => higher (monotonic non-decreasing) coarse index.
        assert!(coarse[1] <= coarse[2] && coarse[2] <= coarse[3]);
    }

    #[test]
    fn upsample_repeats_rows() {
        let rows = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let up = upsample_rows(&rows, 2);
        assert_eq!(up.len(), 4);
        assert_eq!(up[0], rows[0]);
        assert_eq!(up[1], rows[0]);
        assert_eq!(up[2], rows[1]);
    }

    #[test]
    fn locate_axis_handles_both_layouts() {
        // [1, T, D]
        assert_eq!(locate_axis(&[1, 40, 768], 768, "x").unwrap(), (2, 40));
        // [1, D, T]
        assert_eq!(locate_axis(&[1, 768, 40], 768, "x").unwrap(), (1, 40));
        assert!(locate_axis(&[1, 40, 100], 768, "x").is_err());
    }

    #[test]
    fn to_time_major_transposes_when_needed() {
        // [1, D=2, T=3] laid out as d-major: d0=[1,2,3], d1=[4,5,6].
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let rows = to_time_major(&data, &[1, 2, 3], 1, 2, 3);
        assert_eq!(rows, vec![vec![1.0, 4.0], vec![2.0, 5.0], vec![3.0, 6.0]]);
    }

    #[test]
    fn rmvpe_threshold_silences_low_salience() {
        let mut quiet = vec![0.0f32; RMVPE_BINS];
        quiet[10] = 0.01; // below default threshold
        let mut loud = vec![0.0f32; RMVPE_BINS];
        loud[200] = 0.9;
        let f0 = rmvpe_decode(&[quiet, loud], 0.03);
        assert_eq!(f0[0], 0.0);
        assert!(f0[1] > 0.0);
    }
}
