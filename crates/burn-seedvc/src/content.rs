//! The frozen Whisper content encoder and the mel front end, both reused.
//!
//! **No network is ported in this file.** Seed-VC's content encoder *is*
//! `openai/whisper-small`'s encoder, which [`burn_whisper::AudioEncoder`] already
//! implements, and its mel is the HiFi-GAN transform [`burn_vits::Spectral`]
//! already implements. What is here is the adapter that makes those two claims
//! checkable, plus the one thing neither crate supplies: Whisper's own log-mel
//! front end, which is *not* the same transform as the mel the rest of the model
//! runs on.
//!
//! # Two sample rates meet here, and only here
//!
//! | stage | rate | frame rate | transform |
//! |---|---|---|---|
//! | [`ContentEncoder`] | 16 kHz ([`CONTENT_SR`]) | 50 Hz | Whisper log-mel, then 12 encoder layers |
//! | everything downstream | 22 050 Hz | ≈86 Hz | [`mel_config`] through [`burn_vits::Spectral`] |
//!
//! The 16 kHz is not a choice — it is Whisper's analysis rate, baked into the
//! weights. The 22 050 Hz is the preset's. So one utterance is resampled
//! **twice, to different rates, for different purposes**, and the two frame grids
//! that come out do not line up: 50 Hz against ≈86 Hz is exactly the gap the
//! length regulator exists to close. Conflating the two rates would error
//! nowhere and simply stretch the output, which is why the table is written down.
//!
//! # Why Whisper's mel is written out again here
//!
//! [`burn_vits::Spectral`] is close enough to look interchangeable and is not.
//! Whisper centres its STFT (reflect padding of `n_fft / 2`) where `Spectral`
//! pads `(n_fft - hop) / 2`; it takes **power** where `Spectral` takes magnitude;
//! and it finishes with `log10`, a floor eight decades below the window's own
//! peak and a `(x + 4) / 4` rescale where `Spectral` takes a natural log. The
//! frame *counts* agree, so substituting one for the other shifts every feature
//! by half a hop and rescales it — silently. `stt-core` carries the same
//! transform for the same reason; a network crate cannot depend on an engine, so
//! this is a third copy on purpose.

use std::error::Error;
use std::f32::consts::{LN_10, PI};
use std::path::Path;

use burn::tensor::backend::Backend;
use burn::tensor::module::conv1d;
use burn::tensor::ops::ConvOptions;
use burn::tensor::{Tensor, TensorData};
use burn_store::ApplyResult;
use burn_vits::SpectralConfig;
use burn_whisper::{AudioEncoder, WhisperConfig};

use crate::config::{CONTENT_SR, SeedVcConfig};

/// Samples of 16 kHz audio behind one content frame.
///
/// Two 160-sample mel hops, because the encoder's second convolution has stride
/// 2 — so 20 ms, so content arrives at **50 Hz**. Every content-side length in
/// the model is derived from this number rather than from the mel's hop.
pub const CONTENT_STRIDE: usize = 320;

/// Samples in the 30 s window Whisper's encoder is fixed at.
pub const WINDOW_SAMPLES: usize = 30 * CONTENT_SR as usize;

/// Mel frames in one window, before the encoder's stride-2 convolution halves
/// them to the 1500 its positional table is sized for.
const WINDOW_FRAMES: usize = WINDOW_SAMPLES / MEL_HOP;

const MEL_N_FFT: usize = 400;
const MEL_HOP: usize = 160;

/// Whisper's encoder as Seed-VC uses it: a frozen feature extractor.
///
/// Holds the log-mel front end beside the network because the two are one
/// interface — the encoder is only correct on mel computed exactly this way, and
/// separating them is how a caller ends up feeding it the *other* mel in this
/// file.
pub struct ContentEncoder<B: Backend> {
    encoder: AudioEncoder<B>,
    /// Fused DFT kernels, `[n_bins, 1, n_fft]`. Framing, windowing and the
    /// transform are one `conv1d` each, which keeps the front end on whatever
    /// device the model is on instead of round-tripping through the host.
    cos: Tensor<B, 3>,
    sin: Tensor<B, 3>,
    /// Slaney mel filterbank, `[n_mels, n_bins]`.
    filters: Tensor<B, 2>,
}

impl<B: Backend> ContentEncoder<B> {
    pub fn new(device: &B::Device) -> Self {
        let cfg = WhisperConfig::small();
        let n_bins = MEL_N_FFT / 2 + 1;

        // cos[b,k] = w[k]·cos(2πbk/N); sin[b,k] = -w[k]·sin(2πbk/N), against a
        // periodic Hann window — `torch.hann_window`'s default, and what Whisper
        // was trained on.
        let mut cos = vec![0f32; n_bins * MEL_N_FFT];
        let mut sin = vec![0f32; n_bins * MEL_N_FFT];
        for b in 0..n_bins {
            for k in 0..MEL_N_FFT {
                let w = 0.5 - 0.5 * (2.0 * PI * k as f32 / MEL_N_FFT as f32).cos();
                let angle = 2.0 * PI * (b as f32) * (k as f32) / MEL_N_FFT as f32;
                cos[b * MEL_N_FFT + k] = w * angle.cos();
                sin[b * MEL_N_FFT + k] = -w * angle.sin();
            }
        }

        Self {
            encoder: AudioEncoder::new(&cfg, device),
            cos: Tensor::from_data(TensorData::new(cos, [n_bins, 1, MEL_N_FFT]), device),
            sin: Tensor::from_data(TensorData::new(sin, [n_bins, 1, MEL_N_FFT]), device),
            filters: Tensor::from_data(
                TensorData::new(
                    slaney_filterbank(cfg.num_mel_bins),
                    [cfg.num_mel_bins, n_bins],
                ),
                device,
            ),
        }
    }

    /// Load `openai/whisper-small`'s `model.safetensors`, keeping the encoder
    /// half and nothing else.
    ///
    /// The remap strips `model.encoder.`, so **the whole decoder lands in
    /// `unused`, and that is the expected result rather than a coverage gap**:
    /// upstream does `del whisper_model.decoder` for the same reason, because
    /// Seed-VC wants acoustic features and never generates text.
    ///
    /// Measured against `openai/whisper-small`: **187 applied, 0 missing, 0
    /// errors, 342 unused**. Those 342 are the decoder's 292 tensors plus the 50
    /// `weight`/`bias` entries of the encoder's own 25 LayerNorms, which the
    /// adapter consumes under Burn's `gamma`/`beta` names and the store then
    /// records as unconsumed — so `unused` overcounts by construction and
    /// `applied` is the number to read.
    ///
    /// `burn-whisper`'s own `load` example with `--size small` accounts for the
    /// entire file at 479 / 0, which is the check that the *checkpoint* matches
    /// the port; this one checks that the encoder subtree is addressable alone.
    pub fn load_safetensors(
        &mut self,
        path: impl AsRef<Path>,
    ) -> Result<ApplyResult, Box<dyn Error>> {
        burn_kit::store::load_safetensors_into::<B, _>(
            &mut self.encoder,
            path.as_ref(),
            &[(r"^model\.encoder\.", "")],
        )
    }

    /// 16 kHz mono `[batch, samples]` → content features `[batch, frames, 768]`.
    ///
    /// `frames` is `samples / 320 + 1`, matching upstream: the encoder always
    /// runs over a padded 30 s window and returns 1500 frames whatever came in,
    /// so all but the leading `samples / 320 + 1` describe the padding.
    ///
    /// **Audio longer than 30 s is truncated here.** Upstream chunks it with a
    /// 5 s overlap and stitches the features; that path is not implemented, and
    /// wiring it belongs with the inference code rather than with this adapter. A
    /// clip that hits the limit is silently shortened, so the caller must check.
    pub fn forward(&self, audio: Tensor<B, 2>) -> Tensor<B, 3> {
        let [batch, samples] = audio.dims();
        let features = self.encoder.forward(self.log_mel(audio));
        let [_, encoded, width] = features.dims();
        let frames = (samples / CONTENT_STRIDE + 1).min(encoded);
        features.slice([0..batch, 0..frames, 0..width])
    }

    /// Whisper's `log_mel_spectrogram`, `[batch, samples]` → `[batch, 80, 3000]`.
    fn log_mel(&self, audio: Tensor<B, 2>) -> Tensor<B, 3> {
        let [batch, samples] = audio.dims();
        let device = audio.device();

        // A window is always the full 30 s. The encoder's positional table is
        // 1500 frames wide and it was trained on padded windows, so zero padding
        // is what it expects rather than an accommodation for short input.
        let audio = match samples.cmp(&WINDOW_SAMPLES) {
            std::cmp::Ordering::Less => Tensor::cat(
                vec![
                    audio,
                    Tensor::zeros([batch, WINDOW_SAMPLES - samples], &device),
                ],
                1,
            ),
            std::cmp::Ordering::Equal => audio,
            std::cmp::Ordering::Greater => audio.slice([0..batch, 0..WINDOW_SAMPLES]),
        };

        // `torch.stft(center=True)`: reflect padding of half the FFT size, so
        // frame `t` is centred on sample `t * hop` rather than half a hop later.
        let padded = reflect_pad(audio, MEL_N_FFT / 2);
        let plen = padded.dims()[1];
        let opts = ConvOptions::new([MEL_HOP], [0], [1], 1);
        let x = padded.reshape([batch, 1, plen]);
        let re = conv1d(x.clone(), self.cos.clone(), None, opts.clone());
        let im = conv1d(x, self.sin.clone(), None, opts);

        // `center=True` yields `1 + L/hop` frames and Whisper drops the last —
        // the one whose window reaches past the end of the audio.
        let n_bins = re.dims()[1];
        let power = (re.powf_scalar(2.0) + im.powf_scalar(2.0)).slice([
            0..batch,
            0..n_bins,
            0..WINDOW_FRAMES,
        ]);

        let n_mels = self.filters.dims()[0];
        let bank = self
            .filters
            .clone()
            .unsqueeze::<3>()
            .expand([batch, n_mels, n_bins]);
        let log = bank.matmul(power).clamp_min(1e-10).log().div_scalar(LN_10);

        // Floor eight decades below *this window's* peak, then rescale. Both are
        // taken over the whole window, which is why this is not a streaming
        // transform: a window must be complete before any of its frames are
        // final. The span is therefore always exactly 2, but where it sits
        // follows the loudest bin, so the output is not bounded to [-1, 1].
        let floor = log
            .clone()
            .reshape([batch, n_mels * WINDOW_FRAMES])
            .max_dim(1)
            .sub_scalar(8.0)
            .reshape([batch, 1, 1])
            .expand([batch, n_mels, WINDOW_FRAMES]);
        log.max_pair(floor).add_scalar(4.0).div_scalar(4.0)
    }
}

/// The 22 050 Hz mel every module after the content encoder speaks in.
///
/// Upstream's `mel_spectrogram` (`modules/audio.py`) is [`burn_vits::Spectral`]
/// step for step: reflect padding of `(n_fft - hop) / 2` with `center=False`,
/// `sqrt(re² + im² + 1e-9)`, a Slaney-normalised librosa filterbank, then
/// `log(clamp(x, 1e-5))`. So this is a config and not a module — the transform
/// RVC and GPT-SoVITS already share is the one Seed-VC wants, at different
/// numbers — and the frame count is `samples / hop` exactly.
///
/// The preset writes `fmax: "None"`, which librosa reads as `sample_rate / 2`,
/// and `win_length` equal to `n_fft`.
pub fn mel_config(cfg: &SeedVcConfig) -> SpectralConfig {
    SpectralConfig {
        sample_rate: cfg.sample_rate as usize,
        n_fft: cfg.n_fft,
        hop: cfg.hop_length,
        win_length: cfg.n_fft,
        n_mels: cfg.n_mels,
        fmin: 0.0,
        fmax: cfg.sample_rate as f32 / 2.0,
    }
}

/// Reflect-pad both ends of `wav [b, L]` by `p` (numpy `reflect`): mirror
/// without repeating the edge sample.
fn reflect_pad<B: Backend>(wav: Tensor<B, 2>, p: usize) -> Tensor<B, 2> {
    let [b, l] = wav.dims();
    let left = wav.clone().slice([0..b, 1..(p + 1)]).flip([1]);
    let right = wav.clone().slice([0..b, (l - 1 - p)..(l - 1)]).flip([1]);
    Tensor::cat(vec![left, wav, right], 1)
}

/// `librosa.filters.mel(sr=16000, n_fft=400, n_mels=n_mels)` row-major, which is
/// how Whisper's shipped `mel_filters.npz` was generated — Slaney scale with
/// Slaney area normalisation.
fn slaney_filterbank(n_mels: usize) -> Vec<f32> {
    let f_sp = 200.0 / 3.0;
    let min_log_hz = 1000.0f32;
    let min_log_mel = min_log_hz / f_sp;
    let logstep = (6.4f32).ln() / 27.0;
    let hz_to_mel = |f: f32| {
        if f < min_log_hz {
            f / f_sp
        } else {
            min_log_mel + (f / min_log_hz).ln() / logstep
        }
    };
    let mel_to_hz = |m: f32| {
        if m < min_log_mel {
            m * f_sp
        } else {
            min_log_hz * (logstep * (m - min_log_mel)).exp()
        }
    };

    let n_bins = MEL_N_FFT / 2 + 1;
    let fft_freqs: Vec<f32> = (0..n_bins)
        .map(|k| k as f32 * CONTENT_SR as f32 / MEL_N_FFT as f32)
        .collect();

    let (mel_min, mel_max) = (hz_to_mel(0.0), hz_to_mel(CONTENT_SR as f32 / 2.0));
    let edges: Vec<f32> = (0..n_mels + 2)
        .map(|i| mel_to_hz(mel_min + (mel_max - mel_min) * i as f32 / (n_mels as f32 + 1.0)))
        .collect();

    let mut fb = vec![0f32; n_mels * n_bins];
    for m in 0..n_mels {
        let (lo, ctr, hi) = (edges[m], edges[m + 1], edges[m + 2]);
        let enorm = 2.0 / (hi - lo);
        for (bin, &f) in fft_freqs.iter().enumerate() {
            let lower = (f - lo) / (ctr - lo);
            let upper = (hi - f) / (hi - ctr);
            fb[m * n_bins + bin] = lower.min(upper).max(0.0) * enorm;
        }
    }
    fb
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_vits::Spectral;

    type B = burn_ndarray::NdArray;

    fn tone(samples: usize, amplitude: f32) -> Tensor<B, 2> {
        Tensor::from_data(
            TensorData::new(
                (0..samples)
                    .map(|i| (i as f32 * 0.05).sin() * amplitude)
                    .collect::<Vec<f32>>(),
                [1, samples],
            ),
            &Default::default(),
        )
    }

    #[test]
    fn the_preset_mel_gives_one_frame_per_hop() {
        // The claim `burn-vits` has to satisfy for Seed-VC to reuse it: at the
        // preset's numbers the transform is `center=False` with `(n_fft-hop)/2`
        // padding, so a clip of `n` samples is exactly `n / hop` frames and the
        // mel grid is the grid the length regulator resamples onto. One frame
        // either way would misalign content against mel for the whole clip.
        let cfg = SeedVcConfig::uvit_whisper_small_wavenet();
        let spectral = Spectral::<B>::new(&mel_config(&cfg), &Default::default());

        let samples = cfg.hop_length * 173; // ≈2 s at 22 050 Hz, a dataset clip
        let mel = spectral.mel(tone(samples, 0.5));
        assert_eq!(mel.dims(), [1, cfg.n_mels, samples / cfg.hop_length]);

        let v: Vec<f32> = mel.into_data().to_vec().unwrap();
        assert!(v.iter().all(|x| x.is_finite()), "mel is not finite");
    }

    #[test]
    fn a_short_clip_still_fills_the_whole_encoder_window() {
        // Two things at once, because computing a 30 s window is the expensive
        // part: that short input is padded rather than truncated, and that the
        // floor and rescale land where Whisper's do. The span is exactly 2
        // whatever the amplitude — that is what `max(peak - 8)` then `/ 4` means
        // — while its *position* follows the loudest bin, so asserting a fixed
        // range would be a claim about loudness instead.
        let encoder = ContentEncoder::<B>::new(&Default::default());

        for amplitude in [1.0f32, 0.02] {
            let mel = encoder.log_mel(tone(CONTENT_SR as usize, amplitude));
            assert_eq!(mel.dims(), [1, 80, WINDOW_FRAMES]);

            let v: Vec<f32> = mel.into_data().to_vec().unwrap();
            assert!(
                v.iter().all(|x| x.is_finite()),
                "amplitude {amplitude}: mel is not finite"
            );
            let lo = v.iter().copied().fold(f32::INFINITY, f32::min);
            let hi = v.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            assert!(
                (hi - lo - 2.0).abs() < 1e-3,
                "amplitude {amplitude}: {lo}..{hi} should span 2"
            );
        }
    }

    #[test]
    fn reflect_padding_mirrors_without_repeating_the_edge() {
        let wav: Tensor<B, 2> = Tensor::from_data(
            TensorData::new(vec![1.0f32, 2.0, 3.0, 4.0], [1, 4]),
            &Default::default(),
        );
        let v: Vec<f32> = reflect_pad(wav, 2).into_data().to_vec().unwrap();
        assert_eq!(v, [3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 2.0]);
    }
}
