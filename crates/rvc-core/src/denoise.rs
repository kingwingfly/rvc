//! Optional streaming spectral de-hiss for the generator's output.
//!
//! A conservative STFT noise-suppressor: track the per-bin noise floor by
//! minimum-statistics, then apply a spectral-subtraction gain with a floor. Bins
//! at the tracked floor (steady hiss) are pulled down; bins above it (voice, soft
//! breaths) stay ~unity gain, so quiet content is preserved. Gains are smoothed
//! over frequency and time to avoid musical noise.
//!
//! Streaming: [`Denoiser::process`] returns the finalized output so far (an
//! `N_FFT` look-ahead is buffered); [`Denoiser::flush`] drains the tail.

use std::sync::Arc;

use rustfft::num_complex::Complex;
use rustfft::{Fft, FftPlanner};

/// STFT window length (samples). ~21 ms at 48 kHz.
const N_FFT: usize = 1024;
/// Hop between frames (75% overlap → constant-overlap-add with a Hann window).
const HOP: usize = 256;

// --- noise-tracking / suppression constants (conservative for soft content) --
/// Per-bin power smoothing (higher = steadier).
const P_SMOOTH: f32 = 0.7;
/// Slow upward leak of the tracked minimum, per frame (lets the floor recover).
const MIN_LEAK: f32 = 1.002;
/// Min-statistics under-estimates the floor; scale it back up.
const NOISE_BIAS: f32 = 2.0;
/// Frames to observe before suppressing anything (let the floor settle).
const WARMUP_FRAMES: usize = 8;
/// Time smoothing of the per-bin gain.
const GAIN_TIME_SMOOTH: f32 = 0.5;
const EPS: f32 = 1e-10;

/// Tunable strength; defaults are conservative.
#[derive(Debug, Clone, Copy)]
pub struct DenoiseParams {
    /// Spectral over-subtraction factor β (higher = more aggressive).
    pub oversub: f32,
    /// Gain floor: quietest a bin is ever attenuated to (linear; 0.06 ≈ −24 dB).
    /// Above zero leaves a natural noise bed; `1.0` disables suppression.
    pub floor: f32,
}

impl Default for DenoiseParams {
    fn default() -> Self {
        Self {
            oversub: 2.0,
            floor: 0.06,
        }
    }
}

/// Streaming STFT de-hiss processor. One per stream; not `Send`-shared.
pub struct Denoiser {
    fft: Arc<dyn Fft<f32>>,
    ifft: Arc<dyn Fft<f32>>,
    window: Vec<f32>,
    params: DenoiseParams,
    /// Pending input samples not yet framed.
    inbuf: Vec<f32>,
    /// Overlap-add accumulator (length `N_FFT`) for the synthesized signal.
    ola: Vec<f32>,
    /// Constant COLA normalization (Σ window² over overlapping frames). Dividing
    /// by this constant rather than the running per-sample overlap is exact in
    /// steady state and only fades the edges — the running overlap is ~0 at the
    /// edges and would blow those samples up.
    wsum: f32,
    /// Per-bin state over `N_FFT/2 + 1` bins.
    p_smooth: Vec<f32>,
    noise: Vec<f32>,
    prev_gain: Vec<f32>,
    frames: usize,
    /// Reused FFT scratch (length `N_FFT`).
    spec: Vec<Complex<f32>>,
}

impl Denoiser {
    /// Build a denoiser at the given strength. `_sample_rate` is unused today
    /// (the window is fixed in samples) but kept for a future rate-aware window.
    pub fn new(_sample_rate: u32, params: DenoiseParams) -> Self {
        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(N_FFT);
        let ifft = planner.plan_fft_inverse(N_FFT);
        let window: Vec<f32> = (0..N_FFT)
            .map(|i| 0.5 - 0.5 * (std::f32::consts::TAU * i as f32 / N_FFT as f32).cos())
            .collect();
        let bins = N_FFT / 2 + 1;
        // Σ window² across the overlapping frames (constant for Hann at 75%).
        let wsum: f32 = (0..N_FFT)
            .step_by(HOP)
            .map(|k| window[k] * window[k])
            .sum::<f32>()
            .max(EPS);
        Self {
            fft,
            ifft,
            window,
            params,
            inbuf: Vec::new(),
            ola: vec![0.0; N_FFT],
            wsum,
            p_smooth: vec![0.0; bins],
            noise: vec![f32::MAX; bins],
            prev_gain: vec![1.0; bins],
            frames: 0,
            spec: vec![Complex::new(0.0, 0.0); N_FFT],
        }
    }

    /// Clear all state so the denoiser can start fresh on an independent input.
    pub fn reset(&mut self) {
        self.inbuf.clear();
        self.ola.iter_mut().for_each(|v| *v = 0.0);
        self.p_smooth.iter_mut().for_each(|v| *v = 0.0);
        self.noise.iter_mut().for_each(|v| *v = f32::MAX);
        self.prev_gain.iter_mut().for_each(|v| *v = 1.0);
        self.frames = 0;
    }

    /// Feed samples; returns the finalized output produced so far (may be empty
    /// until the first `N_FFT` samples have arrived).
    pub fn process(&mut self, input: &[f32]) -> Vec<f32> {
        self.inbuf.extend_from_slice(input);
        let mut out = Vec::new();
        while self.inbuf.len() >= N_FFT {
            self.process_frame(&mut out);
            self.inbuf.drain(..HOP);
        }
        out
    }

    /// Drain the tail: zero-pad the last partial frame, then emit the pending
    /// overlap-add buffer (the trailing edge fades out).
    pub fn flush(&mut self) -> Vec<f32> {
        let mut out = Vec::new();
        if self.inbuf.is_empty() && self.frames == 0 {
            return out;
        }
        if !self.inbuf.is_empty() {
            let pad = N_FFT.saturating_sub(self.inbuf.len());
            self.inbuf.extend(std::iter::repeat_n(0.0, pad));
            while self.inbuf.len() >= N_FFT {
                self.process_frame(&mut out);
                let drain = HOP.min(self.inbuf.len());
                self.inbuf.drain(..drain);
            }
            self.inbuf.clear();
        }
        // The remaining N_FFT - HOP samples the emit loop hasn't reached yet.
        out.extend_from_slice(&self.ola[..N_FFT - HOP]);
        out
    }

    /// Analyze + suppress + synthesize one frame from `inbuf[..N_FFT]`, appending
    /// the `HOP` newly-finalized output samples to `out`.
    fn process_frame(&mut self, out: &mut Vec<f32>) {
        // --- analysis: windowed FFT ---
        for i in 0..N_FFT {
            self.spec[i] = Complex::new(self.inbuf[i] * self.window[i], 0.0);
        }
        self.fft.process(&mut self.spec);

        // --- per-bin noise tracking + suppression gain ---
        let bins = N_FFT / 2 + 1;
        let mut gain = vec![1.0f32; bins];
        let floor2 = self.params.floor * self.params.floor;
        for (k, g) in gain.iter_mut().enumerate() {
            let p = self.spec[k].norm_sqr();
            self.p_smooth[k] = P_SMOOTH * self.p_smooth[k] + (1.0 - P_SMOOTH) * p;
            // Min-statistics: running min of the smoothed power, with a slow leak.
            let ps = self.p_smooth[k];
            self.noise[k] = if ps < self.noise[k] {
                ps
            } else {
                (self.noise[k] * MIN_LEAK).min(ps)
            };
            if self.frames < WARMUP_FRAMES {
                continue; // leave gain at 1.0 until the floor settles
            }
            let nz = self.noise[k] * NOISE_BIAS;
            // Power spectral subtraction: g^2 = 1 - β·N/P, floored.
            let g2 = (1.0 - self.params.oversub * nz / (self.p_smooth[k] + EPS)).max(floor2);
            *g = g2.sqrt();
        }

        // Smooth the gain over frequency (3-bin box) then time, to suppress
        // musical noise.
        if self.frames >= WARMUP_FRAMES {
            let raw = gain.clone();
            for k in 0..bins {
                let lo = k.saturating_sub(1);
                let hi = (k + 1).min(bins - 1);
                let sm = (raw[lo] + raw[k] + raw[hi]) / 3.0;
                let g = GAIN_TIME_SMOOTH * self.prev_gain[k] + (1.0 - GAIN_TIME_SMOOTH) * sm;
                gain[k] = g;
                self.prev_gain[k] = g;
            }
        }

        // --- apply gain (Hermitian-symmetric) ---
        for (s, &g) in self.spec.iter_mut().zip(gain.iter()) {
            *s *= g;
        }
        for k in 1..(N_FFT / 2) {
            self.spec[N_FFT - k] = self.spec[k].conj();
        }

        // --- synthesis: IFFT, window, overlap-add ---
        self.ifft.process(&mut self.spec);
        let norm = 1.0 / (N_FFT as f32 * self.wsum);
        for i in 0..N_FFT {
            self.ola[i] += self.spec[i].re * norm * self.window[i];
        }

        // Emit the first HOP samples (now fully overlapped), then advance by HOP.
        out.extend_from_slice(&self.ola[..HOP]);
        self.ola.drain(..HOP);
        self.ola.extend(std::iter::repeat_n(0.0, HOP));

        self.frames += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rms(x: &[f32]) -> f32 {
        if x.is_empty() {
            return 0.0;
        }
        (x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32).sqrt()
    }

    /// A tiny deterministic white-ish noise source (no `rand` dep).
    struct Noise(u64);
    impl Noise {
        fn next(&mut self) -> f32 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 >> 40) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0
        }
    }

    /// With `floor = 1.0` the gain is identity, so the STFT round-trips: output
    /// should reconstruct the input (up to the N_FFT edge transient).
    #[test]
    fn identity_reconstructs() {
        let mut d = Denoiser::new(
            48_000,
            DenoiseParams {
                oversub: 1.0,
                floor: 1.0,
            },
        );
        let sr = 48_000.0;
        let input: Vec<f32> = (0..48_000)
            .map(|i| 0.3 * (std::f32::consts::TAU * 220.0 * i as f32 / sr).sin())
            .collect();
        let mut out = d.process(&input);
        out.extend(d.flush());
        // Compare a middle stretch (avoid the leading/trailing edge frames).
        let a = &input[N_FFT..input.len() - N_FFT];
        let b = &out[N_FFT..N_FFT + a.len()];
        let err = rms(&a.iter().zip(b).map(|(x, y)| x - y).collect::<Vec<_>>());
        assert!(err < 1e-3, "identity reconstruction error too high: {err}");
    }

    /// The denoiser must never amplify — no output sample may exceed the input
    /// peak, edges included. (Regression: normalizing by the running per-sample
    /// overlap divided by ~0 at the edges and blew them up.)
    #[test]
    fn never_amplifies() {
        let mut d = Denoiser::new(48_000, DenoiseParams::default());
        let mut n = Noise(0xdead_beef);
        let sr = 48_000.0;
        let input: Vec<f32> = (0..96_000)
            .map(|i| 0.3 * (std::f32::consts::TAU * 200.0 * i as f32 / sr).sin() + 0.03 * n.next())
            .collect();
        let mut out = d.process(&input);
        out.extend(d.flush());
        let in_peak = input.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let out_peak = out.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        assert!(
            out_peak <= in_peak * 1.05,
            "denoiser amplified: in_peak={in_peak} out_peak={out_peak}"
        );
    }

    /// Steady hiss (low-level white noise) is substantially reduced once the
    /// noise floor has settled.
    #[test]
    fn reduces_hiss() {
        let mut d = Denoiser::new(48_000, DenoiseParams::default());
        let mut n = Noise(0x1234_5678);
        let input: Vec<f32> = (0..96_000).map(|_| 0.02 * n.next()).collect();
        let mut out = d.process(&input);
        out.extend(d.flush());
        // Measure a settled 1 s region well past warmup.
        let tail_in = rms(&input[64_000..80_000]);
        let tail_out = rms(&out[64_000..80_000]);
        assert!(
            tail_out < 0.6 * tail_in,
            "hiss not reduced enough: in={tail_in} out={tail_out} ({:.2}x)",
            tail_out / tail_in
        );
    }

    /// Loud content is preserved: after learning the floor during an initial
    /// quiet passage, a following loud tone comes through at ~unity gain. (A pure
    /// tone from t=0, with no pause to learn the floor, is the pathological case
    /// for min-statistics and is out of scope by design.)
    #[test]
    fn preserves_loud_signal() {
        let mut d = Denoiser::new(48_000, DenoiseParams::default());
        let mut n = Noise(0x9abc_def0);
        let sr = 48_000.0;
        let input: Vec<f32> = (0..96_000)
            .map(|i| {
                let hiss = 0.02 * n.next();
                // First 1 s: hiss only (learn the floor). Then add a loud tone.
                if i < 48_000 {
                    hiss
                } else {
                    0.4 * (std::f32::consts::TAU * 300.0 * i as f32 / sr).sin() + hiss
                }
            })
            .collect();
        let mut out = d.process(&input);
        out.extend(d.flush());
        let a = rms(&input[64_000..80_000]);
        let b = rms(&out[64_000..80_000]);
        assert!(
            b > 0.85 * a,
            "loud signal over-attenuated: in={a} out={b} ({:.2}x)",
            b / a
        );
    }
}
