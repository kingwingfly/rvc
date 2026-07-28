//! Odds and ends the training loops share.

use burn::tensor::Tensor;
use burn::tensor::backend::Backend;

/// `1h02m`, `7m30s`, `45s` — short enough to sit inside a progress line.
pub fn human(d: std::time::Duration) -> String {
    let s = d.as_secs();
    match (s / 3600, (s % 3600) / 60, s % 60) {
        (0, 0, s) => format!("{s}s"),
        (0, m, s) => format!("{m}m{s:02}s"),
        (h, m, _) => format!("{h}h{m:02}m"),
    }
}

/// Extract a scalar loss value to `f32` for logging. Syncs the device.
pub fn scalar<B: Backend>(t: Tensor<B, 1>) -> f32 {
    t.into_data()
        .to_vec::<f32>()
        .map(|v| v.first().copied().unwrap_or(f32::NAN))
        .unwrap_or(f32::NAN)
}
