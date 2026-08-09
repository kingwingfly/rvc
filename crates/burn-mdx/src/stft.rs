//! The complex STFT the separation network eats, and its inverse.
//!
//! MDX23C does not consume a spectrogram *magnitude*: it consumes the complex
//! spectrum with real and imaginary parts laid out as separate image channels
//! (upstream calls this "cac", complex-as-channels), predicts a complex
//! spectrum per stem, and inverts it. Phase therefore travels through the
//! network rather than being reused from the mixture, which is why the inverse
//! belongs here beside the forward transform.
//!
//! # This is deliberately not `burn_vits::Spectral`
//!
//! That transform is a *mel* front end built for the VITS training objective:
//! `center = false`, magnitudes only, 128 Slaney mel bands, and a floor that
//! CLAUDE.md records as a two-engine decision nobody may move quietly. Every
//! one of those is wrong here. This one is `center = true` with **reflect**
//! padding (torch's `stft` default), keeps re and im, and truncates the bin
//! axis rather than warping it. The two would run happily on each other's
//! input and compute something else, so they stay separate.
//!
//! # Why it is host-side rather than a Burn graph
//!
//! A DFT-basis matmul or a conv1d filterbank is how a *differentiable* STFT is
//! written, and at `n_fft = 8192` the basis is `8192 × 4097 × 2` floats — 268 MB
//! of constants for a transform that costs microseconds on a radix-2 FFT. This
//! crate is inference-only, so it takes the FFT. That also follows the
//! precedent CLAUDE.md sets for RMVPE, whose mel front end is host-side
//! `rustfft` in `rvc-core` and shared by both runtimes rather than reimplemented
//! per backend.
//!
//! [`Stft::analyze`] and [`Stft::synthesize`] work on plain slices for exactly
//! that reason: the same buffer feeds the Burn model and an ONNX Runtime
//! session, so the two runtimes cannot disagree about a transform neither of
//! them computes.
//!
//! # Frame arithmetic
//!
//! `center = true` pads by `n_fft / 2` on each side, so `samples` yields
//! `samples / hop + 1` frames and the inverse returns `hop * (frames - 1)`
//! samples. That is what makes the checkpoint's `chunk_size = 261120` come out
//! at exactly `dim_t = 256` frames for `hop = 1024`.

use std::sync::Arc;

use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, TensorData};
use rustfft::num_complex::Complex;
use rustfft::{Fft, FftPlanner};

/// Below this the overlap-add window envelope is treated as zero rather than
/// divided by. torch's `istft` uses the same guard for the same reason: a frame
/// the window never covered has no information to normalise.
const ENVELOPE_FLOOR: f32 = 1e-11;

/// A complex STFT/ISTFT pair, parameterised so one type serves every MDX
/// variant.
///
/// `n_fft`, `hop` and `dim_f` are the three numbers that differ across
/// checkpoints — MDX23C reads them from its config YAML, and the older MDX-Net
/// v2 ONNX models read them from UVR's hash-keyed `model_data.json`. Neither
/// set is hardcoded here, so an ONNX path can share this front end with the
/// Burn one (see the crate docs).
pub struct Stft {
    n_fft: usize,
    hop: usize,
    dim_f: usize,
    /// Periodic Hann, matching `torch.hann_window(n_fft, periodic=True)`.
    window: Vec<f32>,
    forward: Arc<dyn Fft<f32>>,
    inverse: Arc<dyn Fft<f32>>,
}

impl std::fmt::Debug for Stft {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Stft")
            .field("n_fft", &self.n_fft)
            .field("hop", &self.hop)
            .field("dim_f", &self.dim_f)
            .finish()
    }
}

impl Stft {
    /// `dim_f` is how many of the `n_fft / 2 + 1` bins the network sees; the
    /// rest are dropped on the way in and zero-filled on the way out, which is
    /// upstream's behaviour and is why a round trip through this pair is lossy
    /// by construction whenever `dim_f < n_bins`.
    pub fn new(n_fft: usize, hop: usize, dim_f: usize) -> Self {
        assert!(n_fft.is_multiple_of(2), "n_fft must be even");
        assert!(hop > 0 && hop <= n_fft, "hop must be in 1..=n_fft");
        assert!(
            dim_f <= n_fft / 2 + 1,
            "dim_f {dim_f} exceeds the {} bins an n_fft of {n_fft} has",
            n_fft / 2 + 1
        );
        let mut planner = FftPlanner::new();
        Self {
            n_fft,
            hop,
            dim_f,
            // Periodic, not symmetric: the denominator is `n_fft`, not
            // `n_fft - 1`. Getting that wrong shifts every window sample by
            // half a bin's worth of taper and shows up as a quiet, broadband
            // reconstruction error rather than as anything that looks like a
            // bug.
            window: (0..n_fft)
                .map(|i| {
                    let phase = 2.0 * std::f64::consts::PI * i as f64 / n_fft as f64;
                    (0.5 - 0.5 * phase.cos()) as f32
                })
                .collect(),
            forward: planner.plan_fft_forward(n_fft),
            inverse: planner.plan_fft_inverse(n_fft),
        }
    }

    /// Bins a full one-sided spectrum has, `n_fft / 2 + 1`.
    pub fn n_bins(&self) -> usize {
        self.n_fft / 2 + 1
    }

    /// Bins the network actually sees.
    pub fn dim_f(&self) -> usize {
        self.dim_f
    }

    /// Frames [`Self::analyze`] produces from `samples`.
    pub fn frames(&self, samples: usize) -> usize {
        samples / self.hop + 1
    }

    /// Samples [`Self::synthesize`] returns for `frames`.
    pub fn samples(&self, frames: usize) -> usize {
        self.hop * frames.saturating_sub(1)
    }

    /// `[channels][samples]` → a flat `[2 * channels, dim_f, frames]` buffer.
    ///
    /// The channel axis is **interleaved re/im per audio channel**:
    /// `2 * ch` is channel `ch`'s real part and `2 * ch + 1` its imaginary
    /// part. That is upstream's `permute([0, 3, 1, 2])` followed by the
    /// `c, 2 → c * 2` reshape, and it is not the layout a reader would guess —
    /// the natural "all real parts then all imaginary parts" runs happily and
    /// swaps two of the four channels.
    pub fn analyze(&self, audio: &[Vec<f32>]) -> (Vec<f32>, usize) {
        assert!(!audio.is_empty(), "analyze needs at least one channel");
        let samples = audio[0].len();
        assert!(
            audio.iter().all(|c| c.len() == samples),
            "channels must be the same length"
        );
        let frames = self.frames(samples);
        let pad = self.n_fft / 2;
        let mut out = vec![0.0f32; 2 * audio.len() * self.dim_f * frames];
        let mut scratch = vec![Complex::new(0.0f32, 0.0); self.n_fft];

        for (ch, samples_in) in audio.iter().enumerate() {
            let padded = reflect_pad(samples_in, pad);
            let re_base = 2 * ch * self.dim_f * frames;
            let im_base = (2 * ch + 1) * self.dim_f * frames;
            for frame in 0..frames {
                let start = frame * self.hop;
                for (j, slot) in scratch.iter_mut().enumerate() {
                    *slot = Complex::new(padded[start + j] * self.window[j], 0.0);
                }
                self.forward.process(&mut scratch);
                for (bin, value) in scratch.iter().take(self.dim_f).enumerate() {
                    out[re_base + bin * frames + frame] = value.re;
                    out[im_base + bin * frames + frame] = value.im;
                }
            }
        }
        (out, frames)
    }

    /// The inverse of [`Self::analyze`]: a flat `[2 * channels, dim_f, frames]`
    /// buffer → `[channels][samples]`.
    ///
    /// Bins above `dim_f` are zero-filled, which is what makes this the inverse
    /// of the *truncating* forward transform rather than of a full one.
    pub fn synthesize(&self, spec: &[f32], channels: usize, frames: usize) -> Vec<Vec<f32>> {
        let plane = self.dim_f * frames;
        assert_eq!(
            spec.len(),
            2 * channels * plane,
            "spectrum is not [2 * {channels}, {}, {frames}]",
            self.dim_f
        );
        let pad = self.n_fft / 2;
        let full = self.n_fft + self.hop * frames.saturating_sub(1);
        let kept = self.samples(frames);

        // The overlap-add of the squared window. It depends only on the frame
        // count, so it is computed once and shared by every channel.
        let mut envelope = vec![0.0f32; full];
        for frame in 0..frames {
            let start = frame * self.hop;
            for (j, w) in self.window.iter().enumerate() {
                envelope[start + j] += w * w;
            }
        }

        let mut scratch = vec![Complex::new(0.0f32, 0.0); self.n_fft];
        (0..channels)
            .map(|ch| {
                let re_base = 2 * ch * plane;
                let im_base = (2 * ch + 1) * plane;
                let mut acc = vec![0.0f32; full];
                for frame in 0..frames {
                    scratch.fill(Complex::new(0.0, 0.0));
                    for bin in 0..self.dim_f {
                        let value = Complex::new(
                            spec[re_base + bin * frames + frame],
                            spec[im_base + bin * frames + frame],
                        );
                        scratch[bin] = value;
                        // Hermitian mirror. Bin 0 and Nyquist are their own
                        // conjugates and must not be written twice — doubling
                        // either adds a DC or an alternating-sign term that
                        // survives every shape and finiteness check there is.
                        if bin > 0 && bin < self.n_fft - bin {
                            scratch[self.n_fft - bin] = value.conj();
                        }
                    }
                    self.inverse.process(&mut scratch);
                    let norm = 1.0 / self.n_fft as f32;
                    let start = frame * self.hop;
                    for (j, w) in self.window.iter().enumerate() {
                        // Taking the real part after a full inverse FFT is
                        // exactly `irfft`: bin 0's and Nyquist's imaginary
                        // parts contribute only to the imaginary result, which
                        // is discarded either way.
                        acc[start + j] += scratch[j].re * norm * w;
                    }
                }
                for (value, env) in acc.iter_mut().zip(&envelope) {
                    if *env > ENVELOPE_FLOOR {
                        *value /= env;
                    }
                }
                acc[pad..pad + kept].to_vec()
            })
            .collect()
    }

    /// [`Self::analyze`] as a `[1, 2 * channels, dim_f, frames]` tensor — the
    /// shape [`crate::TfcTdfNet::forward`] takes.
    pub fn forward<B: Backend>(&self, audio: &[Vec<f32>], device: &B::Device) -> Tensor<B, 4> {
        let (data, frames) = self.analyze(audio);
        let shape = [1, 2 * audio.len(), self.dim_f, frames];
        Tensor::from_data(TensorData::new(data, shape), device)
    }

    /// [`Self::synthesize`] over a `[n, 2 * channels, dim_f, frames]` tensor,
    /// returning `[n][channels][samples]`.
    ///
    /// `n` is the stem axis on the way out of the network, which is why this
    /// takes a rank-4 tensor rather than the rank-5 one `forward` produces —
    /// the caller flattens `[batch, stems, …]` first and knows which is which.
    pub fn inverse<B: Backend>(&self, spec: Tensor<B, 4>) -> Vec<Vec<Vec<f32>>> {
        let [n, channels2, bins, frames] = spec.dims();
        assert_eq!(bins, self.dim_f, "spectrum has {bins} bins, expected dim_f");
        assert!(channels2.is_multiple_of(2), "re/im channels must pair up");
        let data: Vec<f32> = spec.into_data().to_vec().expect("f32 spectrum");
        let stride = channels2 * bins * frames;
        (0..n)
            .map(|i| self.synthesize(&data[i * stride..(i + 1) * stride], channels2 / 2, frames))
            .collect()
    }
}

/// `torch.nn.functional.pad(x, (p, p), mode="reflect")`.
///
/// The reflection **excludes** the edge sample: the first padded value is
/// `x[p]`, not `x[p - 1]`. Off by that one index and the transform still runs,
/// still round-trips, and disagrees with every published checkpoint's front end
/// over the first and last `n_fft / 2` samples of every chunk — which is
/// exactly where a chunked separator puts its seams.
fn reflect_pad(x: &[f32], pad: usize) -> Vec<f32> {
    assert!(
        pad < x.len(),
        "reflect padding of {pad} needs more than {} samples",
        x.len()
    );
    let mut out = Vec::with_capacity(x.len() + 2 * pad);
    out.extend((1..=pad).rev().map(|i| x[i]));
    out.extend_from_slice(x);
    out.extend((1..=pad).map(|i| x[x.len() - 1 - i]));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A deterministic pseudo-random signal — a fixed LCG rather than a
    /// dependency, since the only property needed is "not band-limited".
    fn noise(n: usize) -> Vec<f32> {
        let mut state = 0x2545_F491_4F6C_DD1Du64;
        (0..n)
            .map(|_| {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((state >> 40) as f32 / 8388608.0) - 1.0
            })
            .collect()
    }

    fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len());
        a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
    }

    /// Periodic Hann starts at exactly zero and never reaches 1 at the end;
    /// the symmetric variant is the one that is 0 at *both* ends, and swapping
    /// them is silent.
    #[test]
    fn the_window_is_periodic_not_symmetric() {
        let stft = Stft::new(8, 2, 5);
        assert_eq!(stft.window[0], 0.0);
        // w[n/2] is the peak for a periodic window of even length.
        assert!((stft.window[4] - 1.0).abs() < 1e-6);
        // A symmetric window would put a zero here; a periodic one does not.
        assert!(stft.window[7] > 0.0);
    }

    /// The check no coverage report can make and no weights are needed for: a
    /// wrong window, a wrong normalisation or a missing envelope division all
    /// fail here.
    ///
    /// **Untruncated on purpose.** With `dim_f < n_bins` the pair is lossy by
    /// construction, so asserting exactness against the truncating transform
    /// would fail for a reason that is not a bug — see the next test for what
    /// is true in that case.
    #[test]
    fn the_untruncated_round_trip_returns_the_input() {
        let n_fft = 64;
        let hop = 8;
        let stft = Stft::new(n_fft, hop, n_fft / 2 + 1);
        let samples = hop * 31;
        let audio = vec![noise(samples), noise(samples).iter().rev().copied().collect()];

        let (spec, frames) = stft.analyze(&audio);
        assert_eq!(frames, 32);
        let back = stft.synthesize(&spec, 2, frames);

        assert_eq!(back.len(), 2);
        for (channel, (original, restored)) in audio.iter().zip(&back).enumerate() {
            assert_eq!(restored.len(), original.len());
            let diff = max_abs_diff(original, restored);
            assert!(diff < 1e-4, "channel {channel} round-tripped off by {diff}");
        }
    }

    /// What *is* true of the truncating pair: dropping the bins above `dim_f`
    /// is idempotent, so re-analysing a synthesis reproduces the spectrum it
    /// came from. A window or normalisation error breaks this too — it is the
    /// same check, made where exactness is unavailable.
    #[test]
    fn the_truncated_round_trip_preserves_the_retained_band() {
        let n_fft = 64;
        let hop = 8;
        let stft = Stft::new(n_fft, hop, 12);
        let samples = hop * 31;
        let audio = vec![noise(samples)];

        let (spec, frames) = stft.analyze(&audio);
        let back = stft.synthesize(&spec, 1, frames);
        let (again, frames_again) = stft.analyze(&back);

        assert_eq!(frames_again, frames);
        // The first and last frame see the reflect padding of a signal that is
        // no longer the original, so they are excluded; every interior frame
        // must agree.
        let interior = |s: &[f32]| -> Vec<f32> {
            (0..2 * 12)
                .flat_map(|plane| {
                    (1..frames - 1).map(move |f| s[plane * frames + f])
                })
                .collect()
        };
        let diff = max_abs_diff(&interior(&spec), &interior(&again));
        let scale = spec.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        assert!(
            diff < 1e-3 * scale.max(1e-6),
            "retained band moved by {diff} against a peak of {scale}"
        );
    }

    /// The frame arithmetic the checkpoint's `chunk_size` depends on. A
    /// separator that gets this wrong feeds the network a time axis that is not
    /// a multiple of 32 and fails deep inside the decoder's skip concatenation.
    #[test]
    fn a_chunk_yields_exactly_dim_t_frames() {
        let stft = Stft::new(8192, 1024, 4096);
        let chunk = 1024 * (256 - 1);
        assert_eq!(chunk, 261_120);
        assert_eq!(stft.frames(chunk), 256);
        assert_eq!(stft.samples(256), chunk);
    }

    /// Reflection excludes the edge sample. Pinned against a hand-written
    /// expectation rather than a property, because the off-by-one version
    /// satisfies every property a reflection has.
    #[test]
    fn reflection_excludes_the_edge_sample() {
        let x = [1.0f32, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(reflect_pad(&x, 2), vec![3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0]);
    }
}
