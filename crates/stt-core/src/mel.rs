//! Whisper's log-mel front-end.
//!
//! Reproduces `whisper/audio.py::log_mel_spectrogram` exactly, because the
//! encoder was trained on precisely these numbers: a 400-point periodic Hann
//! window every 160 samples, centred with reflect padding, **power** spectra, a
//! 128-band Slaney mel filterbank, then `log10`, a floor 8 decades below the
//! window's own peak, and a `(x + 4) / 4` rescale. Note what that last pair does
//! and does not guarantee: the output always spans exactly 2, but where it sits
//! follows the window's loudest bin, so it is not bounded to [-1, 1].
//!
//! The floor is taken against the maximum of the whole window, so this is not a
//! streaming transform: a 30 s window has to be complete before any of its
//! frames are final.
//!
//! (`rvc-core` has a similar file for RMVPE. They share a shape and no numbers —
//! different FFT size, magnitude rather than power, natural log, no rescale —
//! and engines do not depend on each other, so the Slaney helpers are written
//! twice on purpose.)

use rustfft::FftPlanner;
use rustfft::num_complex::Complex;

/// Whisper's analysis rate. Audio must arrive resampled to this.
pub const SAMPLE_RATE: u32 = 16_000;
const N_FFT: usize = 400;
/// One mel frame per 10 ms.
pub const HOP: usize = 160;
/// Samples in one 30 s encoder window.
pub const WINDOW_SAMPLES: usize = 30 * SAMPLE_RATE as usize;
/// Mel frames in one window — `WINDOW_SAMPLES / HOP`.
pub const WINDOW_FRAMES: usize = WINDOW_SAMPLES / HOP;

const F_MIN: f32 = 0.0;
const F_MAX: f32 = SAMPLE_RATE as f32 / 2.0;

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

/// Precomputed window, filterbank and FFT plan.
pub struct LogMel {
    n_mels: usize,
    window: Vec<f32>,
    basis: Vec<Vec<f32>>, // [n_mels][N_FFT/2 + 1]
    fft: std::sync::Arc<dyn rustfft::Fft<f32>>,
}

impl LogMel {
    /// `n_mels` is 128 for large-v3 and its derivatives, 80 for everything older.
    pub fn new(n_mels: usize) -> Self {
        // Periodic Hann, matching `torch.hann_window`'s default.
        let window = (0..N_FFT)
            .map(|n| 0.5 - 0.5 * (std::f32::consts::TAU * n as f32 / N_FFT as f32).cos())
            .collect();
        Self {
            n_mels,
            window,
            basis: build_slaney_mel(n_mels),
            fft: FftPlanner::<f32>::new().plan_fft_forward(N_FFT),
        }
    }

    /// Log-mel for one window of 16 kHz mono audio, returned mel-major
    /// (`data[mel * frames + frame]`) ready to become a `[1, n_mels, frames]`
    /// tensor, alongside the frame count.
    ///
    /// Short input is zero-padded to a full window; the encoder's positional
    /// table is 1500 frames wide and it was trained on padded windows, so this
    /// is what it expects rather than an accommodation.
    pub fn compute(&self, audio: &[f32]) -> (usize, Vec<f32>) {
        let mut padded_input;
        let audio = if audio.len() < WINDOW_SAMPLES {
            padded_input = Vec::with_capacity(WINDOW_SAMPLES);
            padded_input.extend_from_slice(audio);
            padded_input.resize(WINDOW_SAMPLES, 0.0);
            &padded_input[..]
        } else {
            &audio[..WINDOW_SAMPLES]
        };

        let pad = N_FFT / 2;
        let padded = reflect_pad(audio, pad);
        // `torch.stft(center=True)` yields `1 + len/hop` frames and Whisper drops
        // the last, which is the one reaching past the end of the audio.
        let n_frames = audio.len() / HOP;
        let half = N_FFT / 2 + 1;

        let mut scratch = vec![Complex::new(0.0, 0.0); N_FFT];
        let mut data = vec![0.0f32; self.n_mels * n_frames];
        let mut power = vec![0.0f32; half];

        for t in 0..n_frames {
            let start = t * HOP;
            for (i, s) in scratch.iter_mut().enumerate() {
                *s = Complex::new(padded[start + i] * self.window[i], 0.0);
            }
            self.fft.process(&mut scratch);
            for (p, c) in power.iter_mut().zip(&scratch[..half]) {
                *p = c.norm_sqr();
            }
            for (d, row) in self.basis.iter().enumerate() {
                let e: f32 = row.iter().zip(&power).map(|(w, p)| w * p).sum();
                data[d * n_frames + t] = e.max(1e-10).log10();
            }
        }

        // Floor 8 decades below this window's peak, then rescale. Both are over
        // the whole window, which is why a window must be complete first.
        let peak = data.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let floor = peak - 8.0;
        for v in &mut data {
            *v = (v.max(floor) + 4.0) / 4.0;
        }
        (n_frames, data)
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

/// librosa's Slaney mel filterbank with Slaney area normalization —
/// `librosa.filters.mel(sr=16000, n_fft=400, n_mels=n_mels)`, which is how
/// Whisper's shipped `mel_filters.npz` was generated.
fn build_slaney_mel(n_mels: usize) -> Vec<Vec<f32>> {
    let half = N_FFT / 2 + 1;
    let fft_freqs: Vec<f32> = (0..half)
        .map(|k| k as f32 * SAMPLE_RATE as f32 / N_FFT as f32)
        .collect();

    let mel_min = hz_to_mel_slaney(F_MIN);
    let mel_max = hz_to_mel_slaney(F_MAX);
    let edges: Vec<f32> = (0..n_mels + 2)
        .map(|i| mel_to_hz_slaney(mel_min + (mel_max - mel_min) * i as f32 / (n_mels as f32 + 1.0)))
        .collect();

    let mut basis = vec![vec![0.0f32; half]; n_mels];
    for (m, row) in basis.iter_mut().enumerate() {
        let (lo, ctr, hi) = (edges[m], edges[m + 1], edges[m + 2]);
        let enorm = 2.0 / (hi - lo);
        for (k, slot) in row.iter_mut().enumerate() {
            let f = fft_freqs[k];
            let lower = (f - lo) / (ctr - lo);
            let upper = (hi - f) / (hi - ctr);
            *slot = lower.min(upper).max(0.0) * enorm;
        }
    }
    basis
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_full_window_is_exactly_the_encoder_width() {
        // 3000 frames in, 1500 out after the stride-2 conv — the number the
        // encoder's positional table is sized for.
        let mel = LogMel::new(128);
        let (frames, data) = mel.compute(&vec![0.0; WINDOW_SAMPLES]);
        assert_eq!(frames, WINDOW_FRAMES);
        assert_eq!(data.len(), 128 * WINDOW_FRAMES);
    }

    #[test]
    fn short_input_is_padded_not_truncated() {
        // Every window is full width whatever came in, because that is what the
        // model saw in training; a short clip must not produce a short window.
        let mel = LogMel::new(128);
        let (frames, _) = mel.compute(&vec![0.1; SAMPLE_RATE as usize]);
        assert_eq!(frames, WINDOW_FRAMES);
    }

    #[test]
    fn the_dynamic_range_is_always_eight_decades() {
        // The floor is 8 decades below the window's own peak and the rescale
        // divides by 4, so the span is exactly 2 whatever the input. The
        // *position* is not fixed — it tracks the loudest bin — so an absolute
        // [-1, 1] bound would be a claim about loudness, not about this code.
        let mel = LogMel::new(128);
        for amplitude in [1.0, 0.05] {
            let audio: Vec<f32> = (0..WINDOW_SAMPLES)
                .map(|i| (i as f32 * 0.05).sin() * amplitude)
                .collect();
            let (_, data) = mel.compute(&audio);
            let lo = data.iter().copied().fold(f32::INFINITY, f32::min);
            let hi = data.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            assert!(lo.is_finite() && hi.is_finite(), "non-finite mel");
            assert!(
                (hi - lo - 2.0).abs() < 1e-3,
                "amplitude {amplitude}: {lo}..{hi} should span 2"
            );
        }
    }

    #[test]
    fn reflect_padding_mirrors_without_repeating_the_edge() {
        assert_eq!(
            reflect_pad(&[1.0, 2.0, 3.0, 4.0], 2),
            [3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 2.0]
        );
    }
}
