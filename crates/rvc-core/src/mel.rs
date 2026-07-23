//! RMVPE mel-spectrogram frontend.
//!
//! The `rmvpe.onnx` model consumes a `[1, 128, T]` log-mel magnitude
//! spectrogram (not raw audio). This reproduces RMVPE's `MelSpectrogram`:
//! librosa Slaney mel basis (n_fft=1024, hop=160, 128 mels, fmin=30, fmax=8000),
//! a periodic Hann window, center/reflect padding, magnitude (not power) spectra,
//! and `log(clamp(mel, 1e-5))`.

use rustfft::FftPlanner;
use rustfft::num_complex::Complex;

const SR: f32 = 16_000.0;
const N_FFT: usize = 1024;
const HOP: usize = 160;
const N_MELS: usize = 128;
const F_MIN: f32 = 30.0;
const F_MAX: f32 = 8_000.0;
const CLAMP: f32 = 1e-5;

fn hz_to_mel_slaney(f: f32) -> f32 {
    let f_sp = 200.0 / 3.0;
    let min_log_hz = 1000.0;
    let min_log_mel = min_log_hz / f_sp;
    let logstep = (6.4f32).ln() / 27.0;
    if f < min_log_hz {
        f / f_sp
    } else {
        min_log_mel + (f / min_log_hz).ln() / logstep
    }
}

fn mel_to_hz_slaney(m: f32) -> f32 {
    let f_sp = 200.0 / 3.0;
    let min_log_hz = 1000.0;
    let min_log_mel = min_log_hz / f_sp;
    let logstep = (6.4f32).ln() / 27.0;
    if m < min_log_mel {
        m * f_sp
    } else {
        min_log_hz * (logstep * (m - min_log_mel)).exp()
    }
}

/// Precomputed RMVPE mel extractor.
pub struct RmvpeMel {
    window: Vec<f32>,
    mel_basis: Vec<Vec<f32>>, // [N_MELS][N_FFT/2 + 1]
    fft: std::sync::Arc<dyn rustfft::Fft<f32>>,
}

impl RmvpeMel {
    /// Build the extractor (window + Slaney mel basis + FFT plan).
    pub fn new() -> Self {
        // Periodic Hann window (torch.hann_window default).
        let window: Vec<f32> = (0..N_FFT)
            .map(|n| 0.5 - 0.5 * (std::f32::consts::TAU * n as f32 / N_FFT as f32).cos())
            .collect();
        let mel_basis = build_slaney_mel();
        let fft = FftPlanner::<f32>::new().plan_fft_forward(N_FFT);
        Self {
            window,
            mel_basis,
            fft,
        }
    }

    /// Compute the log-mel spectrogram, returned as `(n_mels, time, data)` where
    /// `data` is `[n_mels * time]` in mel-major (`d*time + t`) order, ready to
    /// feed as a `[1, n_mels, time]` tensor.
    pub fn compute(&self, wav16k: &[f32]) -> (usize, usize, Vec<f32>) {
        // Center padding: reflect by N_FFT/2 on both sides.
        let pad = N_FFT / 2;
        let padded = reflect_pad(wav16k, pad);
        if padded.len() < N_FFT {
            return (N_MELS, 0, Vec::new());
        }
        let n_frames = 1 + (padded.len() - N_FFT) / HOP;
        let half = N_FFT / 2 + 1;

        let mut scratch: Vec<Complex<f32>> = vec![Complex::new(0.0, 0.0); N_FFT];
        // mel-major output: data[d * n_frames + t].
        let mut data = vec![0.0f32; N_MELS * n_frames];

        for t in 0..n_frames {
            let start = t * HOP;
            for (i, s) in scratch.iter_mut().enumerate() {
                *s = Complex::new(padded[start + i] * self.window[i], 0.0);
            }
            self.fft.process(&mut scratch);

            // Magnitude spectrum (not power).
            let mag: Vec<f32> = scratch[..half].iter().map(|c| c.norm()).collect();

            for (d, basis) in self.mel_basis.iter().enumerate() {
                let mut e = 0.0f32;
                for (k, &w) in basis.iter().enumerate() {
                    e += w * mag[k];
                }
                data[d * n_frames + t] = e.max(CLAMP).ln();
            }
        }
        (N_MELS, n_frames, data)
    }
}

impl Default for RmvpeMel {
    fn default() -> Self {
        Self::new()
    }
}

/// Reflect-pad (numpy `mode='reflect'`): mirror without repeating the edge.
fn reflect_pad(x: &[f32], pad: usize) -> Vec<f32> {
    let n = x.len();
    let mut out = Vec::with_capacity(n + 2 * pad);
    for j in 0..pad {
        out.push(x[(pad - j).min(n - 1)]);
    }
    out.extend_from_slice(x);
    for j in 0..pad {
        out.push(x[n.saturating_sub(2 + j)]);
    }
    out
}

/// librosa Slaney mel filterbank with Slaney area normalization.
fn build_slaney_mel() -> Vec<Vec<f32>> {
    let half = N_FFT / 2 + 1;
    let fft_freqs: Vec<f32> = (0..half).map(|k| k as f32 * SR / N_FFT as f32).collect();

    // n_mels + 2 mel band edges (in Hz).
    let mel_min = hz_to_mel_slaney(F_MIN);
    let mel_max = hz_to_mel_slaney(F_MAX);
    let edges: Vec<f32> = (0..N_MELS + 2)
        .map(|i| mel_to_hz_slaney(mel_min + (mel_max - mel_min) * i as f32 / (N_MELS as f32 + 1.0)))
        .collect();

    let mut basis = vec![vec![0.0f32; half]; N_MELS];
    for (m, row) in basis.iter_mut().enumerate() {
        let (lo, ctr, hi) = (edges[m], edges[m + 1], edges[m + 2]);
        let enorm = 2.0 / (hi - lo);
        for (k, slot) in row.iter_mut().enumerate() {
            let f = fft_freqs[k];
            let lower = (f - lo) / (ctr - lo);
            let upper = (hi - f) / (hi - ctr);
            let w = lower.min(upper).max(0.0);
            *slot = w * enorm;
        }
    }
    basis
}
