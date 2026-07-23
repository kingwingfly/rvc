//! Differentiable STFT → linear + mel spectrograms, in Burn.
//!
//! Framing, windowing and the DFT are fused into two `conv1d` kernels
//! (`cos`/`sin`), so the whole transform is a couple of convolutions plus a
//! matmul — fully differentiable, so the mel-L1 loss backprops through the
//! generated waveform. Mel filterbank is librosa-compatible (Slaney).

use std::f32::consts::PI;

use burn::tensor::backend::Backend;
use burn::tensor::module::conv1d;
use burn::tensor::ops::ConvOptions;
use burn::tensor::{Tensor, TensorData};

/// STFT/mel parameters (RVC v2 48 kHz).
pub struct SpectralConfig {
    pub sample_rate: usize,
    pub n_fft: usize,
    pub hop: usize,
    pub win_length: usize,
    pub n_mels: usize,
    pub fmin: f32,
    pub fmax: f32,
}

impl SpectralConfig {
    /// The 48 kHz config used by RVC v2.
    pub fn v2_48k() -> Self {
        Self {
            sample_rate: 48_000,
            n_fft: 2048,
            hop: 480,
            win_length: 2048,
            n_mels: 128,
            fmin: 0.0,
            fmax: 24_000.0,
        }
    }

    /// Number of linear-spectrogram bins.
    pub fn n_bins(&self) -> usize {
        self.n_fft / 2 + 1
    }
}

/// Precomputed STFT/mel transform on a device.
pub struct Spectral<B: Backend> {
    cos_kernel: Tensor<B, 3>, // [n_bins, 1, n_fft]
    sin_kernel: Tensor<B, 3>, // [n_bins, 1, n_fft]
    mel_fb: Tensor<B, 2>,     // [n_mels, n_bins]
    hop: usize,
    pad: usize,
}

impl<B: Backend> Spectral<B> {
    /// Precompute the fused DFT kernels and mel filterbank.
    pub fn new(cfg: &SpectralConfig, device: &B::Device) -> Self {
        let n_fft = cfg.n_fft;
        let n_bins = cfg.n_bins();
        let window = hann(cfg.win_length, n_fft);

        // cos[b,k] = w[k]·cos(2πbk/N); sin[b,k] = -w[k]·sin(2πbk/N).
        let mut cos = vec![0f32; n_bins * n_fft];
        let mut sin = vec![0f32; n_bins * n_fft];
        for b in 0..n_bins {
            for k in 0..n_fft {
                let angle = 2.0 * PI * (b as f32) * (k as f32) / n_fft as f32;
                cos[b * n_fft + k] = window[k] * angle.cos();
                sin[b * n_fft + k] = -window[k] * angle.sin();
            }
        }
        let cos_kernel = Tensor::from_data(TensorData::new(cos, [n_bins, 1, n_fft]), device);
        let sin_kernel = Tensor::from_data(TensorData::new(sin, [n_bins, 1, n_fft]), device);

        let fb = mel_filterbank(cfg);
        let mel_fb = Tensor::from_data(TensorData::new(fb, [cfg.n_mels, n_bins]), device);

        // RVC uses center=False with (n_fft-hop)/2 reflect padding, so the frame
        // count is exactly L/hop (matching the content/F0 frame grid).
        let pad = (n_fft - cfg.hop) / 2;
        Self {
            cos_kernel,
            sin_kernel,
            mel_fb,
            hop: cfg.hop,
            pad,
        }
    }

    /// Magnitude linear spectrogram of `wav [b, L]` → `[b, n_bins, frames]`.
    pub fn linear(&self, wav: Tensor<B, 2>) -> Tensor<B, 3> {
        let [batch, _] = wav.dims();
        let padded = reflect_pad(wav, self.pad);
        let plen = padded.dims()[1];
        let x = padded.reshape([batch, 1, plen]);
        let opts = ConvOptions::new([self.hop], [0], [1], 1);
        let real = conv1d(x.clone(), self.cos_kernel.clone(), None, opts.clone());
        let imag = conv1d(x, self.sin_kernel.clone(), None, opts);
        (real.powf_scalar(2.0) + imag.powf_scalar(2.0) + 1e-9).sqrt()
    }

    /// Log-mel spectrogram of `wav [b, L]` → `[b, n_mels, frames]`.
    pub fn mel(&self, wav: Tensor<B, 2>) -> Tensor<B, 3> {
        let mag = self.linear(wav); // [b, n_bins, frames]
        let [b, n_bins, frames] = mag.dims();
        // mel_fb [n_mels, n_bins] · mag [b, n_bins, frames]
        let n_mels = self.mel_fb.dims()[0];
        let fb = self.mel_fb.clone().unsqueeze::<3>(); // [1, n_mels, n_bins]
        let fb = fb.expand([b, n_mels, n_bins]);
        let mel = fb.matmul(mag); // [b, n_mels, frames]
        let _ = frames;
        mel.clamp_min(1e-5).log()
    }
}

/// A periodic Hann window of `win` samples, zero-padded/centered to `n_fft`.
fn hann(win: usize, n_fft: usize) -> Vec<f32> {
    let mut w = vec![0f32; n_fft];
    let pad = (n_fft - win) / 2;
    for i in 0..win {
        w[pad + i] = 0.5 - 0.5 * (2.0 * PI * i as f32 / win as f32).cos();
    }
    w
}

/// Reflect-pad both ends of `wav [b, L]` by `p` (numpy `reflect`).
fn reflect_pad<B: Backend>(wav: Tensor<B, 2>, p: usize) -> Tensor<B, 2> {
    if p == 0 {
        return wav;
    }
    let [b, l] = wav.dims();
    let left = wav.clone().slice([0..b, 1..(p + 1)]).flip([1]);
    let right = wav.clone().slice([0..b, (l - 1 - p)..(l - 1)]).flip([1]);
    Tensor::cat(vec![left, wav, right], 1)
}

// ---- librosa-compatible Slaney mel filterbank -------------------------------

fn hz_to_mel(f: f32) -> f32 {
    let f_sp = 200.0 / 3.0;
    let min_log_hz = 1000.0;
    let min_log_mel = min_log_hz / f_sp;
    let logstep = (6.4f32).ln() / 27.0;
    if f >= min_log_hz {
        min_log_mel + (f / min_log_hz).ln() / logstep
    } else {
        f / f_sp
    }
}

fn mel_to_hz(m: f32) -> f32 {
    let f_sp = 200.0 / 3.0;
    let min_log_hz = 1000.0;
    let min_log_mel = min_log_hz / f_sp;
    let logstep = (6.4f32).ln() / 27.0;
    if m >= min_log_mel {
        min_log_hz * (logstep * (m - min_log_mel)).exp()
    } else {
        f_sp * m
    }
}

/// `[n_mels, n_bins]` filterbank, row-major.
fn mel_filterbank(cfg: &SpectralConfig) -> Vec<f32> {
    let n_bins = cfg.n_bins();
    let fft_freqs: Vec<f32> = (0..n_bins)
        .map(|i| i as f32 * cfg.sample_rate as f32 / cfg.n_fft as f32)
        .collect();

    let mel_min = hz_to_mel(cfg.fmin);
    let mel_max = hz_to_mel(cfg.fmax);
    let n = cfg.n_mels;
    let mel_points: Vec<f32> = (0..n + 2)
        .map(|i| mel_min + (mel_max - mel_min) * i as f32 / (n + 1) as f32)
        .collect();
    let hz_points: Vec<f32> = mel_points.iter().map(|&m| mel_to_hz(m)).collect();

    let mut fb = vec![0f32; n * n_bins];
    for m in 0..n {
        let (lower, center, upper) = (hz_points[m], hz_points[m + 1], hz_points[m + 2]);
        let norm = 2.0 / (upper - lower); // Slaney
        for (bin, &f) in fft_freqs.iter().enumerate() {
            let up = (f - lower) / (center - lower);
            let down = (upper - f) / (upper - center);
            let w = up.min(down).max(0.0);
            fb[m * n_bins + bin] = w * norm;
        }
    }
    fb
}
