//! Kaldi's filterbank — **the only input CAM++ has ever been shown.**
//!
//! `torchaudio.compliance.kaldi.fbank` in Burn, because nothing else in the
//! workspace computes it and [`crate::campplus`] cannot run without it. Seed-VC's
//! `inference.py` writes the speaker encoder's input in two lines:
//!
//! ```text
//! feat2 = kaldi.fbank(wave_16k, num_mel_bins=80, dither=0, sample_frequency=16000)
//! feat2 = feat2 - feat2.mean(dim=0, keepdim=True)
//! ```
//!
//! **Both lines are [`Fbank::forward`].** The subtraction is not a nicety —
//! CAM++ has no input normalisation of its own — and leaving it out is invisible:
//! the embedding stays finite, stays repeatable, and quietly stops separating
//! speakers. Splitting the two would buy a caller nothing and would put the
//! omission one forgotten line away, so the raw filterbank is not offered.
//!
//! # Not the mel the rest of this crate speaks
//!
//! [`burn_vits::Spectral`] is 22.05 kHz, 80 band, log — and so is this, which is
//! exactly the trap [`crate::campplus`]'s module docs name. They are different
//! transforms for different consumers, and every difference below is a place
//! where substituting one for the other would run, produce a plausible tensor,
//! and feed the network a distribution it was never trained on:
//!
//! - **Per-frame DC removal**, then a **preemphasis of 0.97** with a replicate
//!   pad at the frame's left edge — so the first tap keeps `1 - 0.97` of itself
//!   rather than being high-passed against a neighbour it does not have.
//! - The **Povey window**, `(0.5 - 0.5·cos(2πn/(N-1)))^0.85`: a *symmetric* Hann
//!   raised to 0.85, which reaches zero at both ends where a periodic Hann does
//!   not.
//! - The 400-sample window is **zero-padded on the right** to the next power of
//!   two, 512, before the transform. Right, not centred — a centred pad would
//!   rotate every frame's phase and is what `burn_vits::Spectral`'s `hann` does.
//! - `snip_edges = True` framing: no padding at either end, so the frame count is
//!   `1 + (samples - 400) / 160` and the final partial window is dropped.
//! - A **power** spectrum, and **Kaldi's own mel scale** `1127·ln(1 + f/700)`
//!   with triangles laid out evenly in that domain between `low_freq = 20` and
//!   the Nyquist rate — not Slaney, and not area-normalised.
//! - The **Nyquist FFT bin carries no weight at all**: Kaldi builds the bank over
//!   `padded_window / 2` bins and torchaudio pads a zero column onto the right to
//!   reach the 257 an `rfft` returns.
//! - The log is floored at `f32::EPSILON`, which is `torch.finfo(torch.float).eps`
//!   and the value `_get_epsilon` hands `torch.max`.
//!
//! # Waveform scale does not matter, and that is worth knowing
//!
//! Kaldi proper expects samples scaled like int16; upstream hands it the
//! `[-1, 1]` float waveform. That is not a bug to fix here, because a gain of `a`
//! adds `2·ln a` to *every* bin of *every* frame and the mean subtraction on the
//! next line removes exactly that. The one place the scale is visible is the
//! epsilon floor, so this follows upstream's `[-1, 1]` convention rather than
//! Kaldi's — the floor has to sit where the released model saw it.
//!
//! # Dither is not a knob here
//!
//! There is no `dither` field pinned to zero: the only value any Seed-VC path
//! passes is `0`, and a filterbank with no RNG in it is a pure function of its
//! input — which is what lets `examples/speaker` print the same cosine matrix
//! twice and lets the tests below assert equalities rather than tolerances.
//!
//! # What is established and what is not
//!
//! **There is no numerical diff against torchaudio, and there cannot be one
//! here** — this workspace runs no Python, and `torchaudio` is not among the
//! repositories cloned under `.reference`. So the semantics above are a careful
//! reading of `torchaudio.compliance.kaldi` rather than something a test pins
//! against the original. Two consequences, both named rather than hidden:
//!
//! - The unit tests check *properties* — the frame grid, DC rejection, and that
//!   a tone lands in the mel bin Kaldi's own scale puts it in. That catches a
//!   wrong frequency axis or a missing high-pass, not a window exponent of 0.8.
//! - The check that actually matters is end to end, through the network that
//!   consumes this: `examples/speaker` embeds several clips and prints their
//!   pairwise cosine matrix. A filterbank that is subtly wrong still clusters
//!   *somewhat*, so the number to read is the **gap** between same-speaker and
//!   cross-speaker pairs, not either one alone.

use std::f64::consts::PI;

use burn::tensor::backend::Backend;
use burn::tensor::module::conv1d;
use burn::tensor::ops::ConvOptions;
use burn::tensor::{Tensor, TensorData};

/// `torch.finfo(torch.float).eps`, which is what torchaudio floors the log at.
const EPSILON: f32 = f32::EPSILON;

/// The exponent that turns a symmetric Hann window into Kaldi's Povey window.
const POVEY_EXPONENT: f64 = 0.85;

/// Everything `kaldi.fbank` is called with, resolved.
///
/// The defaults are the call in `inference.py` plus torchaudio's own defaults for
/// everything it leaves out, so [`Default::default`] is the only configuration
/// Seed-VC ever uses.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FbankConfig {
    /// 16 kHz. Equal to [`crate::config::CONTENT_SR`] and not the same
    /// constraint — Whisper and CAM++ were each trained at 16 kHz independently,
    /// so the two rates agreeing is a coincidence to keep rather than to share.
    pub sample_rate: usize,
    /// Filterbank bins, 80 — `CamPPlusConfig::feat_dim`.
    pub num_mel_bins: usize,
    /// Window length in milliseconds, 25.
    pub frame_length_ms: f64,
    /// Hop in milliseconds, 10 — so 100 frames per second, which the speaker
    /// encoder's first strided TDNN layer halves to 50.
    pub frame_shift_ms: f64,
    /// Bottom of the mel bank, 20 Hz.
    pub low_freq: f64,
    /// Top of the mel bank. Kaldi spells this `0.0` and means "the Nyquist rate";
    /// it is resolved here so nothing downstream has to know that convention.
    pub high_freq: f64,
    /// First-order high-pass coefficient, 0.97.
    pub preemphasis: f64,
}

impl Default for FbankConfig {
    fn default() -> Self {
        Self {
            sample_rate: 16_000,
            num_mel_bins: 80,
            frame_length_ms: 25.0,
            frame_shift_ms: 10.0,
            low_freq: 20.0,
            high_freq: 8_000.0,
            preemphasis: 0.97,
        }
    }
}

impl FbankConfig {
    /// Samples in one analysis window, 400.
    ///
    /// **Computed in `f64` on purpose.** Kaldi's arithmetic is
    /// `int(sample_frequency * frame_length_ms * 0.001)`, and in `f32` the hop
    /// comes out `159.99999` and truncates to **159** — a frame grid that drifts
    /// a sample every window and fails no assertion anywhere.
    pub fn window(&self) -> usize {
        (self.sample_rate as f64 * self.frame_length_ms * 1e-3) as usize
    }

    /// Samples between consecutive windows, 160. See [`Self::window`].
    pub fn shift(&self) -> usize {
        (self.sample_rate as f64 * self.frame_shift_ms * 1e-3) as usize
    }

    /// The FFT length, 512: [`Self::window`] rounded up to a power of two.
    pub fn padded_window(&self) -> usize {
        self.window().next_power_of_two()
    }
}

/// Kaldi's mel scale, `1127·ln(1 + f/700)`.
///
/// Not the Slaney scale [`burn_vits::Spectral`] and Whisper both use: that one is
/// linear below 1 kHz and logarithmic above, where this is one expression over
/// the whole range. They disagree by tens of hertz in the middle of the band.
fn mel(hz: f64) -> f64 {
    1127.0 * (1.0 + hz / 700.0).ln()
}

/// The precomputed transform, on a device.
pub struct Fbank<B: Backend> {
    /// `[bins, 1, window]` — the real half of the analysis, with framing,
    /// DC removal, preemphasis, the window and the DFT all folded together.
    cos_kernel: Tensor<B, 3>,
    /// `[bins, 1, window]`, the imaginary half.
    sin_kernel: Tensor<B, 3>,
    /// `[num_mel_bins, bins]`.
    mel_fb: Tensor<B, 2>,
    shift: usize,
}

impl<B: Backend> Fbank<B> {
    /// Precompute the analysis kernels and the mel bank.
    pub fn new(cfg: &FbankConfig, device: &B::Device) -> Self {
        let (window, padded) = (cfg.window(), cfg.padded_window());
        let bins = padded / 2 + 1;

        // Everything Kaldi does between a frame and its spectrum is *linear in
        // that frame* — subtract the mean, high-pass, multiply by a window,
        // zero-pad, transform — so the whole chain collapses into one row of
        // taps per output bin, and framing becomes a strided `conv1d`. That is
        // an exact rewrite, not an approximation, and it is what keeps this a
        // pair of convolutions on whatever backend the caller chose.
        let povey: Vec<f64> = (0..window)
            .map(|n| {
                let hann = 0.5 - 0.5 * (2.0 * PI * n as f64 / (window - 1) as f64).cos();
                hann.powf(POVEY_EXPONENT)
            })
            .collect();

        let row = |bin: usize, imaginary: bool| -> Vec<f32> {
            // The windowed DFT basis. The frame is zero-padded on the right to
            // `padded`, so only its first `window` taps can meet a sample and
            // the padding never has to be materialised.
            let basis: Vec<f64> = (0..window)
                .map(|k| {
                    let angle = 2.0 * PI * bin as f64 * k as f64 / padded as f64;
                    povey[k] * if imaginary { -angle.sin() } else { angle.cos() }
                })
                .collect();

            // Preemphasis, folded in from the right: sample `j` reaches outputs
            // `j` and `j + 1`, so a tap on input `j` picks up `basis[j]` less
            // `0.97 · basis[j + 1]`. The replicate pad at the left edge is why
            // tap 0 keeps only `1 - 0.97` of itself.
            let taps: Vec<f64> = (0..window)
                .map(|j| {
                    let own = if j == 0 { 1.0 - cfg.preemphasis } else { 1.0 };
                    let next = basis.get(j + 1).copied().unwrap_or(0.0);
                    own * basis[j] - cfg.preemphasis * next
                })
                .collect();

            // DC removal, likewise: subtracting a frame's own mean from each of
            // its samples subtracts the row's mean from each of its taps.
            let mean = taps.iter().sum::<f64>() / window as f64;
            taps.iter().map(|t| (t - mean) as f32).collect()
        };

        let kernel = |imaginary: bool| {
            let taps: Vec<f32> = (0..bins).flat_map(|b| row(b, imaginary)).collect();
            Tensor::from_data(TensorData::new(taps, [bins, 1, window]), device)
        };

        // Kaldi lays `num_mel_bins` triangles over `[low_freq, high_freq]` at
        // even spacing *in the mel domain*, each spanning two spacings and
        // peaking in the middle — hence the `+ 1` in the denominator, which is
        // the end effect of the first and last triangles hanging off the edges.
        let num_fft_bins = padded / 2;
        let fft_bin_width = cfg.sample_rate as f64 / padded as f64;
        let (mel_low, mel_high) = (mel(cfg.low_freq), mel(cfg.high_freq));
        let delta = (mel_high - mel_low) / (cfg.num_mel_bins + 1) as f64;

        // Left at `bins` wide with the Nyquist column zero, which is exactly the
        // zero column torchaudio pads on: Kaldi's bank stops one bin short of
        // what an `rfft` returns.
        let mut fb = vec![0f32; cfg.num_mel_bins * bins];
        for m in 0..cfg.num_mel_bins {
            let left = mel_low + m as f64 * delta;
            let right = left + 2.0 * delta;
            for k in 0..num_fft_bins {
                let f = mel(fft_bin_width * k as f64);
                let up = (f - left) / delta;
                let down = (right - f) / delta;
                fb[m * bins + k] = up.min(down).max(0.0) as f32;
            }
        }

        Self {
            cos_kernel: kernel(false),
            sin_kernel: kernel(true),
            mel_fb: Tensor::from_data(TensorData::new(fb, [cfg.num_mel_bins, bins]), device),
            shift: cfg.shift(),
        }
    }

    /// `wav [batch, samples]` at the configured rate → `[batch, frames, bins]`.
    ///
    /// **Frames before bins**, which is what [`crate::campplus::CamPPlus::forward`]
    /// takes, and **the per-clip mean is already subtracted** — see the module
    /// docs for why that is not the caller's job.
    ///
    /// A batch is a batch of *whole* clips of the same length. Padding one out to
    /// match another would be normalised against its own silence and pooled over
    /// it afterwards, which is wrong rather than merely different; a reference is
    /// one clip, so nothing here needs it.
    pub fn forward(&self, wav: Tensor<B, 2>) -> Tensor<B, 3> {
        let [batch, samples] = wav.dims();
        let window = self.cos_kernel.dims()[2];
        assert!(
            samples >= window,
            "a clip of {samples} samples is shorter than one {window}-sample analysis window, \
             and `snip_edges` framing pads neither end — so there is no frame to compute"
        );

        let x = wav.reshape([batch, 1, samples]);
        let opts = ConvOptions::new([self.shift], [0], [1], 1);
        let real = conv1d(x.clone(), self.cos_kernel.clone(), None, opts.clone());
        let imaginary = conv1d(x, self.sin_kernel.clone(), None, opts);
        let power = real.powf_scalar(2.0) + imaginary.powf_scalar(2.0);

        let bins = power.dims()[1];
        let bank =
            self.mel_fb
                .clone()
                .unsqueeze::<3>()
                .expand([batch, self.mel_fb.dims()[0], bins]);
        let feats = bank.matmul(power).clamp_min(EPSILON).log().swap_dims(1, 2);
        feats.clone() - feats.mean_dim(1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    type B = burn_ndarray::NdArray;

    /// A tone of `hz` for the first half of a second and `other` for the second,
    /// at unit amplitude. Two tones rather than one because [`Fbank::forward`]
    /// subtracts each bin's mean over time: a *stationary* signal has the same
    /// spectrum in every frame and comes out identically zero, which would make
    /// every spectral assertion below vacuously true.
    fn two_tones(cfg: &FbankConfig, hz: f64, other: f64) -> Vec<f32> {
        let rate = cfg.sample_rate as f64;
        (0..cfg.sample_rate * 2)
            .map(|n| {
                let f = if n < cfg.sample_rate { hz } else { other };
                (2.0 * PI * f * n as f64 / rate).sin() as f32
            })
            .collect()
    }

    fn features(cfg: &FbankConfig, pcm: &[f32]) -> Vec<f32> {
        let device = Default::default();
        let wav = Tensor::<B, 2>::from_data(TensorData::new(pcm.to_vec(), [1, pcm.len()]), &device);
        Fbank::new(cfg, &device)
            .forward(wav)
            .into_data()
            .to_vec()
            .unwrap()
    }

    /// The frame grid is the whole contract with the network downstream, and
    /// `snip_edges` is the half of it that is easy to get wrong — a transform
    /// that centre-pads gains two frames and still runs.
    #[test]
    fn snip_edges_pads_neither_end_and_drops_the_partial_window() {
        let cfg = FbankConfig::default();
        assert_eq!(
            (cfg.window(), cfg.shift(), cfg.padded_window()),
            (400, 160, 512)
        );

        let device = Default::default();
        let fbank = Fbank::<B>::new(&cfg, &device);
        // A second, plus 90 samples that cannot start a whole window.
        let samples = cfg.sample_rate + 90;
        let wav = Tensor::zeros([1, samples], &device);
        let expected = 1 + (samples - cfg.window()) / cfg.shift();
        assert_eq!(fbank.forward(wav).dims(), [1, expected, cfg.num_mel_bins]);
        assert_eq!(expected, 99);
    }

    /// A tone must land in the triangle Kaldi's own mel scale puts it in.
    ///
    /// This is the arithmetic check: it fails on a wrong FFT length, a wrong
    /// `fft_bin_width`, a Slaney bank in place of Kaldi's, or an off-by-one in
    /// the triangle layout — none of which changes a single tensor shape. The
    /// expected index comes from the mel formula rather than from the code under
    /// test, so what is being pinned is the *frequency axis*.
    #[test]
    fn a_tone_peaks_in_the_bin_kaldis_mel_scale_puts_it_in() {
        let cfg = FbankConfig::default();
        let delta = (mel(cfg.high_freq) - mel(cfg.low_freq)) / (cfg.num_mel_bins + 1) as f64;
        // Triangle `m` peaks at `mel_low + (m + 1)·delta`.
        let expected = |hz: f64| ((mel(hz) - mel(cfg.low_freq)) / delta - 1.0).round() as usize;

        let (low, high) = (1_000.0, 3_000.0);
        let feats = features(&cfg, &two_tones(&cfg, low, high));
        assert!(feats.iter().all(|f| f.is_finite()), "non-finite features");

        // Frame 30 is well inside the first tone, frame 170 inside the second;
        // both are clear of the discontinuity at the join.
        for (frame, hz) in [(30, low), (170, high)] {
            let bins = &feats[frame * cfg.num_mel_bins..(frame + 1) * cfg.num_mel_bins];
            let peak = bins
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .unwrap()
                .0;
            assert!(
                peak.abs_diff(expected(hz)) <= 1,
                "{hz} Hz peaked at bin {peak}, expected {}",
                expected(hz)
            );
        }
    }

    /// Kaldi removes each frame's mean before anything else, and a front end that
    /// skips it differs only in the lowest bins — which is where a speaker
    /// encoder's cheapest cues live.
    ///
    /// A full-scale offset against a full-scale tone: without the subtraction the
    /// preemphasis still leaves 3% of it, and 3% of full scale is far above the
    /// window's own sidelobes down there.
    #[test]
    fn a_constant_offset_changes_nothing() {
        let cfg = FbankConfig::default();
        let pcm = two_tones(&cfg, 1_000.0, 3_000.0);
        let offset: Vec<f32> = pcm.iter().map(|s| s + 1.0).collect();

        let (clean, shifted) = (features(&cfg, &pcm), features(&cfg, &offset));
        let worst = clean
            .iter()
            .zip(&shifted)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(worst < 1e-2, "a DC offset moved a bin by {worst}");
    }

    /// The mean subtraction is half the contract with CAM++ and the half that
    /// fails silently, so it gets an assertion of its own.
    #[test]
    fn every_bin_averages_to_zero_over_time() {
        let cfg = FbankConfig::default();
        let feats = features(&cfg, &two_tones(&cfg, 1_000.0, 3_000.0));
        let frames = feats.len() / cfg.num_mel_bins;
        for bin in 0..cfg.num_mel_bins {
            let mean: f32 = (0..frames).map(|f| feats[f * cfg.num_mel_bins + bin]).sum();
            assert!(
                (mean / frames as f32).abs() < 1e-3,
                "bin {bin} averages {} over time",
                mean / frames as f32
            );
        }
    }
}
