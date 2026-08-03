//! The four checkpoints, loaded and held behind one non-generic type.
//!
//! [`Model`] is the whole boundary between the pipeline and a compute backend,
//! the same arrangement `tts-core`'s `Engine` and `stt-core`'s use: everything
//! crossing it is plain `f32`, so [`crate::convert`] and [`crate::reference`]
//! never name a Burn type and nothing above here is generic. [`BurnModel<B>`] is
//! the only implementation today — there is no ONNX export of Seed-VC — and
//! [`crate::backend::load`] is what erases it.
//!
//! # What is inside, and why it cannot be outside
//!
//! Five networks and one transform, from **five different files**:
//!
//! | held | from |
//! |---|---|
//! | [`Dit`] + [`InterpolateRegulator`] | Seed-VC's own `.pth` |
//! | [`CamPPlus`] | `campplus_cn_common.bin` (`funasr/campplus`) |
//! | [`BigVgan`] | `bigvgan_generator.pt` (`nvidia/bigvgan_v2_22khz_80band_256x`) |
//! | [`ContentEncoder`] | `openai/whisper-small`'s `model.safetensors` |
//! | [`Spectral`] | nothing — derived from the preset |
//!
//! [`burn_seedvc::style_encoder`] and [`burn_seedvc::vq`] are **deliberately not
//! here**. Both are in the checkpoint and neither is in upstream's inference
//! path; wiring the style encoder would feed the transformer a timbre vector the
//! released weights were never conditioned on.
//!
//! The sampler is inside for a narrower reason: [`Sampler::sample`] is generic
//! over `E: Estimator<B>` and [`Dit`] is what implements it, so the integration
//! cannot be lifted above the point where `B` is still known.
//!
//! # Three rates, and the caller supplies two of them
//!
//! [`Model::analyse`] takes the reference clip **twice** — once at
//! [`CONTENT_SR`] for Whisper and CAMPPlus, once at the preset's 22 050 Hz for
//! the mel — because resampling is ffmpeg's job and belongs in the pipeline, and
//! because a single rate would silently be wrong for two of the three consumers.
//! [`Model::convert`] takes the source at [`CONTENT_SR`] alone: the source's
//! waveform is never needed, only what was said in it.

use std::f32::consts::PI;
use std::path::Path;

use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor, TensorData};
use burn_seedvc::campplus::{CamPPlus, CamPPlusConfig};
use burn_seedvc::content::{CONTENT_STRIDE, ContentEncoder, WINDOW_SAMPLES, mel_config};
use burn_seedvc::flow::Sampler;
use burn_seedvc::{BigVgan, BigVganConfig, Dit, InterpolateRegulator, SeedVcConfig};
use burn_vits::Spectral;

use crate::error::{Error, Result};

pub use burn_seedvc::config::CONTENT_SR;

/// Where the four checkpoints are.
///
/// Named rather than positional because they are four independent releases —
/// only `dit` holds two of the five networks, and the other three each come from
/// somebody else's repository. `hub-kit` is what fetches them; this crate only
/// reads what it is pointed at.
pub struct ModelPaths<'a> {
    /// `DiT_seed_v2_uvit_whisper_small_wavenet_bigvgan_pruned.pth` — the
    /// transformer **and** the length regulator, which is the one file that
    /// holds two of the modules.
    pub dit: &'a Path,
    /// `campplus_cn_common.bin`.
    pub campplus: &'a Path,
    /// `bigvgan_generator.pt`.
    pub bigvgan: &'a Path,
    /// `openai/whisper-small`'s `model.safetensors`. Only its encoder is read.
    pub content: &'a Path,
}

/// Everything one reference clip specifies, as plain floats.
///
/// A value rather than a handle, for the reason `tts_core::Reference` is one: it
/// is small, so it can outlive the model that produced it, be cached beside the
/// clip, or be reused across a whole streaming session without pinning a device.
/// The transformer is conditioned on all three of these at once, and on nothing
/// else about the speaker.
#[derive(Debug, Clone)]
pub struct Reference {
    /// Mel frames in the clip — `mel` is `n_mels × frames` and `cond` is
    /// `frames × hidden_dim`, so this is the one length both are read with.
    pub frames: usize,
    /// `[n_mels, frames]` row-major: the mel the generation is asked to
    /// **continue**, not to imitate. It occupies the leading frames of the
    /// sampler's window and is sliced back off afterwards.
    pub mel: Vec<f32>,
    /// `[frames, hidden_dim]` row-major: the clip's own length-regulated content,
    /// which is prepended to the source's so the transformer sees a prompt whose
    /// text and audio agree.
    pub cond: Vec<f32>,
    /// `[style_dim]`: CAMPPlus's timbre vector.
    pub style: Vec<f32>,
}

/// One loaded Seed-VC, backend erased.
///
/// `&self` throughout: unlike `s1`'s key/value cache there is no state between
/// calls, so one model converts any number of sources against any number of
/// references.
pub trait Model: Send {
    /// The preset every shape here is derived from — sample rate, hop, `n_mels`,
    /// `style_dim`, `hidden_dim` and the transformer's `block_size`.
    ///
    /// Exposed so the pipeline can size its chunks and its noise without
    /// repeating a single constant.
    fn config(&self) -> &SeedVcConfig;

    /// Analyse a reference clip into the whole speaker specification.
    ///
    /// - `content` — mono `f32` at [`CONTENT_SR`], at most 30 s. Feeds both
    ///   Whisper and CAMPPlus.
    /// - `mel` — the **same clip** at [`SeedVcConfig::sample_rate`]. Its length
    ///   fixes `Reference::frames` at `len / hop_length`.
    ///
    /// The two are not interchangeable and nothing downstream can tell them
    /// apart, which is why they are separate arguments rather than one clip and a
    /// rate.
    fn analyse(&self, content: &[f32], mel: &[f32]) -> Result<Reference>;

    /// Convert one chunk of source audio into the reference's voice.
    ///
    /// - `source` — mono `f32` at [`CONTENT_SR`], at most 30 s (Whisper's window;
    ///   longer input is refused rather than truncated, because truncation here
    ///   is silent).
    /// - `frames` — mel frames to generate. The pipeline's, not the model's:
    ///   `source_samples / CONTENT_SR * sample_rate / hop_length` reproduces the
    ///   source's duration, and scaling it is upstream's `--length-adjust`.
    /// - `noise` — `[n_mels, reference.frames + frames]` row-major, drawn by the
    ///   caller. The prompt region is overwritten by the sampler, so its values
    ///   there are ignored, but its **width includes it**.
    /// - `sampler` — Euler steps and guidance scale, which are user flags.
    ///
    /// Returns mono `f32` at [`SeedVcConfig::sample_rate`], `frames * hop_length`
    /// samples long.
    ///
    /// The caller draws the noise for the reason [`burn_seedvc::flow`] gives: a
    /// sampler that is a pure function of its inputs can be compared run to run,
    /// and a seeded stream is the only way two backends can be held against each
    /// other at all.
    fn convert(
        &self,
        source: &[f32],
        frames: usize,
        reference: &Reference,
        noise: &[f32],
        sampler: Sampler,
    ) -> Result<Vec<f32>>;
}

/// The native Burn implementation, generic over the compute backend and erased
/// behind [`Model`] at the constructor.
pub struct BurnModel<B: Backend> {
    content: ContentEncoder<B>,
    regulator: InterpolateRegulator<B>,
    dit: Dit<B>,
    campplus: CamPPlus<B>,
    vocoder: BigVgan<B>,
    spectral: Spectral<B>,
    cfg: SeedVcConfig,
    device: B::Device,
}

/// Reject a load that left parameters at their initialised values.
///
/// Every loader under `burn-kit` allows a partial apply so that a coverage report
/// can be *inspected* rather than a single mismatch aborting the load. The cost
/// of that is that **an empty apply is a success unless somebody looks**: a
/// checkpoint whose names no longer match would leave every parameter freshly
/// initialised and the model would convert speech into noise with nothing said.
///
/// `missing` is the number that matters. `unused` is not checkable here and is
/// deliberately not checked — the Seed-VC `.pth` holds five modules, so loading
/// any one leaves the others' tensors over; CAMPPlus leaves 122
/// `num_batches_tracked` counters; and Whisper leaves its whole decoder, which
/// upstream deletes for the same reason. `errors` is separate from `missing`
/// because the applier drops a failed path from *both* lists, so a shape
/// mismatch otherwise reads as full coverage.
fn covered(what: &'static str, result: &burn_kit::ApplyResult) -> Result<()> {
    if let Some(first) = result.errors.first() {
        return Err(Error::Load {
            what,
            why: format!(
                "{} of the checkpoint's tensors could not be applied ({first}) — \
                 the file does not match the model",
                result.errors.len(),
            ),
        });
    }
    if result.applied.is_empty() || !result.missing.is_empty() {
        return Err(Error::Load {
            what,
            why: format!(
                "{} of {} parameters had no weights in the checkpoint (applied {}) — \
                 the file's tensor names do not match the model",
                result.missing.len(),
                result.missing.len() + result.applied.len(),
                result.applied.len(),
            ),
        });
    }
    // At `info` rather than `debug` because this is the only place the numbers
    // exist: `examples/coverage` reads them off the log rather than duplicating
    // the five loads, so the check and the thing checked cannot drift.
    tracing::info!(
        "{what}: applied {}, missing 0, unused {}",
        result.applied.len(),
        result.unused.len()
    );
    Ok(())
}

impl<B: Backend> BurnModel<B> {
    /// Load all four files onto one device.
    ///
    /// Order is the order they are cheapest to fail on: the Seed-VC checkpoint
    /// twice, then the two foreign releases, then Whisper's 967 MB.
    pub fn load(paths: &ModelPaths<'_>, device: &B::Device) -> Result<Self> {
        let cfg = SeedVcConfig::uvit_whisper_small_wavenet();
        let load = |what: &'static str, e: Box<dyn std::error::Error>| Error::Load {
            what,
            why: e.to_string(),
        };

        let mut dit = Dit::<B>::new(&cfg, device);
        covered(
            "the transformer",
            &dit.load_pytorch(paths.dit)
                .map_err(|e| load("the transformer", e))?,
        )?;

        let mut regulator = InterpolateRegulator::<B>::new(&cfg, device);
        covered(
            "the length regulator",
            &regulator
                .load_pytorch(paths.dit)
                .map_err(|e| load("the length regulator", e))?,
        )?;

        let mut campplus = CamPPlus::<B>::new(&CamPPlusConfig::default(), device);
        covered(
            "the timbre encoder",
            &campplus
                .load_pytorch(paths.campplus)
                .map_err(|e| load("the timbre encoder", e))?,
        )?;

        let mut vocoder = BigVgan::<B>::new(&BigVganConfig::v2_22khz_80band_256x(), device);
        covered(
            "the vocoder",
            &vocoder
                .load_pytorch(paths.bigvgan)
                .map_err(|e| load("the vocoder", e))?,
        )?;

        let mut content = ContentEncoder::<B>::new(device);
        covered(
            "the content encoder",
            &content
                .load_safetensors(paths.content)
                .map_err(|e| load("the content encoder", e))?,
        )?;

        Ok(Self {
            spectral: Spectral::new(&mel_config(&cfg), device),
            content,
            regulator,
            dit,
            campplus,
            vocoder,
            cfg,
            device: device.clone(),
        })
    }

    fn audio(&self, samples: &[f32]) -> Tensor<B, 2> {
        Tensor::from_data(
            TensorData::new(samples.to_vec(), [1, samples.len()]),
            &self.device,
        )
    }

    fn floats<const D: usize>(&self, values: &[f32], shape: [usize; D]) -> Tensor<B, D> {
        Tensor::from_data(TensorData::new(values.to_vec(), shape), &self.device)
    }
}

impl<B: Backend> Model for BurnModel<B> {
    fn config(&self) -> &SeedVcConfig {
        &self.cfg
    }

    fn analyse(&self, content: &[f32], mel: &[f32]) -> Result<Reference> {
        if content.len() > WINDOW_SAMPLES {
            return Err(Error::Reference(format!(
                "the reference clip is {:.1} s at {CONTENT_SR} Hz and the content encoder's \
                 window is 30 s — trim it, because what runs past the window is dropped \
                 without a sound",
                content.len() as f32 / CONTENT_SR as f32,
            )));
        }
        let frames = mel.len() / self.cfg.hop_length;
        if frames == 0 {
            return Err(Error::Reference(
                "the reference clip is shorter than one mel frame".into(),
            ));
        }
        if frames >= self.cfg.block_size {
            return Err(Error::Reference(format!(
                "the reference clip is {frames} mel frames and the transformer's block_size is \
                 {} — a reference that fills the window leaves nothing to generate",
                self.cfg.block_size,
            )));
        }

        // CAMPPlus wants frames before bins, which is the opposite of everything
        // else here and will not fail loudly: both axes are 80 wide.
        let (fbank, fbank_frames) = kaldi_fbank(content);
        if fbank_frames == 0 {
            return Err(Error::Reference(format!(
                "the reference clip is {} samples at {CONTENT_SR} Hz, shorter than the timbre \
                 encoder's own 25 ms analysis window",
                content.len(),
            )));
        }
        let style = self
            .campplus
            .forward(self.floats(&fbank, [1, fbank_frames, FBANK_BINS]));

        // `Spectral` is `center=False` with `(n_fft - hop)/2` padding, so this is
        // exactly `frames` and lines up with the grid the regulator resamples to.
        let mel = self.spectral.mel(self.audio(mel));
        let cond = self
            .regulator
            .forward(self.content.forward(self.audio(content)), frames);

        Ok(Reference {
            frames,
            mel: vector("the reference mel", mel)?,
            cond: vector("the reference conditioning", cond)?,
            style: vector("the timbre vector", style)?,
        })
    }

    fn convert(
        &self,
        source: &[f32],
        frames: usize,
        reference: &Reference,
        noise: &[f32],
        sampler: Sampler,
    ) -> Result<Vec<f32>> {
        let total = reference.frames + frames;
        if source.len() > WINDOW_SAMPLES {
            return Err(Error::Input(format!(
                "this chunk is {:.1} s at {CONTENT_SR} Hz and the content encoder's window is \
                 30 s — split it before calling, because what runs past the window is dropped \
                 without a sound",
                source.len() as f32 / CONTENT_SR as f32,
            )));
        }
        if frames == 0 || source.len() < CONTENT_STRIDE {
            return Err(Error::Input(
                "nothing to convert: the chunk is under one content frame long".into(),
            ));
        }
        if total > self.cfg.block_size {
            return Err(Error::Input(format!(
                "{} reference frames plus {frames} generated frames is {total}, past the \
                 transformer's block_size of {} — chunk the source against the reference's \
                 length, not against the window alone",
                reference.frames, self.cfg.block_size,
            )));
        }
        if noise.len() != self.cfg.n_mels * total {
            return Err(Error::Input(format!(
                "the noise is {} values and the sampler wants {} ({} mels × {total} frames, \
                 the prompt region included even though it is overwritten)",
                noise.len(),
                self.cfg.n_mels * total,
                self.cfg.n_mels,
            )));
        }
        // `reference.frames == 0` is checked separately from the lengths below,
        // which it would satisfy with three empty vectors: the sampler asserts a
        // non-empty prompt, and a panic there says nothing about where the
        // reference came from.
        if reference.frames == 0 {
            return Err(Error::Input(
                "the reference specifies no frames — it conditions on nothing".into(),
            ));
        }
        if reference.mel.len() != self.cfg.n_mels * reference.frames
            || reference.cond.len() != reference.frames * self.cfg.hidden_dim
            || reference.style.len() != self.cfg.style_dim
        {
            return Err(Error::Input(
                "the reference's tensors do not match its frame count — it was built for a \
                 different preset"
                    .into(),
            ));
        }

        let cond = self
            .regulator
            .forward(self.content.forward(self.audio(source)), frames);
        // The reference's conditioning leads, so what the transformer is asked
        // for is a continuation rather than an imitation.
        let cond = Tensor::cat(
            vec![
                self.floats(&reference.cond, [1, reference.frames, self.cfg.hidden_dim]),
                cond,
            ],
            1,
        );

        let mel = sampler.sample(
            &self.dit,
            self.floats(noise, [1, self.cfg.n_mels, total]),
            self.floats(&reference.mel, [1, self.cfg.n_mels, reference.frames]),
            cond,
            self.floats(&reference.style, [1, self.cfg.style_dim]),
            Tensor::<B, 1, Int>::from_ints([total as i32], &self.device),
        );

        // The prompt region comes back still pinned at zero — the sampler returns
        // the whole window and slicing it off is documented as the caller's job.
        let mel = mel.narrow(2, reference.frames, frames);
        vector("the converted audio", self.vocoder.forward(mel))
    }
}

/// A tensor's values on the host, with the failure named.
///
/// Burn's `to_vec` fails only on a dtype mismatch, which at this point would mean
/// a backend whose float is not `f32` — worth a sentence rather than an `unwrap`,
/// because the `wgpu` backend is exactly where that would first show up.
fn vector<B: Backend, const D: usize>(what: &'static str, t: Tensor<B, D>) -> Result<Vec<f32>> {
    t.into_data().to_vec().map_err(|e| Error::Load {
        what,
        why: format!("was not f32: {e:?}"),
    })
}

// --- CAMPPlus's front end ----------------------------------------------------
//
// `torchaudio.compliance.kaldi.fbank(wave_16k, num_mel_bins=80, dither=0,
// sample_frequency=16000)`, which is what upstream feeds the timbre encoder and
// what its weights were trained against. It lives here rather than in the
// pipeline for the same reason Whisper's log-mel lives inside
// `burn_seedvc::content::ContentEncoder`: **the transform is part of the
// network's interface**, fixed by the checkpoint, not a choice a caller makes.
//
// It is *not* the mel anything else in this crate uses, and none of the
// differences would fail loudly. Kaldi frames with `snip_edges` (no padding at
// either end), removes each frame's DC offset, pre-emphasises, windows with
// Povey (`hann^0.85`, non-periodic), pads 400 samples to 512, takes **power**,
// and runs it through *Kaldi's* triangular filterbank — which is neither Slaney-
// normalised nor the same shape as librosa's — before a natural log floored at
// `f32::EPSILON`, which is `torch.finfo(torch.float).eps` exactly.
//
// **Untested numerically against torchaudio**, because checking it would need
// Python. The tests below pin the framing, the window and the filterbank's
// triangles, and `examples/coverage` runs the whole path into the real CAMPPlus
// and reports whether two spectral envelopes separate — which catches a front end
// that has stopped carrying timbre, not one that is subtly mis-scaled.

/// Kaldi's `num_mel_bins`, and CAMPPlus's `feat_dim`.
const FBANK_BINS: usize = 80;
/// 25 ms at [`CONTENT_SR`].
const FBANK_WINDOW: usize = 400;
/// 10 ms at [`CONTENT_SR`], so the filterbank arrives at 100 Hz.
const FBANK_SHIFT: usize = 160;
/// `round_to_power_of_two`: 400 padded up.
const FBANK_FFT: usize = 512;
/// Kaldi's `preemphasis_coefficient` default, which upstream does not override.
const FBANK_PREEMPH: f32 = 0.97;
/// Kaldi's `low_freq` default. `high_freq` is 0, which it reads as the Nyquist.
const FBANK_LOW_HZ: f32 = 20.0;

/// `[frames, 80]` row-major, and the frame count.
fn kaldi_fbank(wave: &[f32]) -> (Vec<f32>, usize) {
    if wave.len() < FBANK_WINDOW {
        return (Vec::new(), 0);
    }
    // `snip_edges=True`: only whole windows, and no padding at either end.
    let frames = 1 + (wave.len() - FBANK_WINDOW) / FBANK_SHIFT;
    let window = povey_window();
    let bank = kaldi_mel_bank();

    let mut out = vec![0f32; frames * FBANK_BINS];
    let mut re = vec![0f32; FBANK_FFT];
    let mut im = vec![0f32; FBANK_FFT];
    for f in 0..frames {
        let frame = &wave[f * FBANK_SHIFT..][..FBANK_WINDOW];
        let mean = frame.iter().sum::<f32>() / FBANK_WINDOW as f32;

        re.fill(0.0);
        im.fill(0.0);
        for k in 0..FBANK_WINDOW {
            // Pre-emphasis after the DC removal, with the first sample its own
            // predecessor — PyTorch's `pad(mode='replicate')`.
            let previous = frame[k.saturating_sub(1)] - mean;
            re[k] = ((frame[k] - mean) - FBANK_PREEMPH * previous) * window[k];
        }
        fft(&mut re, &mut im);

        let row = &mut out[f * FBANK_BINS..][..FBANK_BINS];
        for (b, slot) in row.iter_mut().enumerate() {
            let weights = &bank[b * (FBANK_FFT / 2)..][..FBANK_FFT / 2];
            let energy: f32 = weights
                .iter()
                .enumerate()
                .map(|(k, w)| w * (re[k] * re[k] + im[k] * im[k]))
                .sum();
            *slot = energy.max(f32::EPSILON).ln();
        }
    }
    (out, frames)
}

/// Kaldi's Povey window: a non-periodic Hann raised to 0.85.
fn povey_window() -> Vec<f32> {
    (0..FBANK_WINDOW)
        .map(|k| {
            let hann = 0.5 - 0.5 * (2.0 * PI * k as f32 / (FBANK_WINDOW - 1) as f32).cos();
            hann.powf(0.85)
        })
        .collect()
}

/// Kaldi's triangular filterbank, `[80, 256]` row-major.
///
/// 256 columns rather than 257: Kaldi builds its triangles over
/// `padded_window_size / 2` bins and pads a zero column for the Nyquist, so the
/// last bin contributes to nothing. Dropping it instead of padding it is the same
/// arithmetic with one fewer trap.
///
/// The triangles are placed on a **uniform mel grid between the two edge
/// frequencies** and normalised by nothing at all — librosa's Slaney area
/// normalisation would scale every band by its own width, which is the difference
/// that loads perfectly and shifts every embedding.
fn kaldi_mel_bank() -> Vec<f32> {
    let mel = |hz: f32| 1127.0 * (1.0 + hz / 700.0).ln();
    let bin_width = CONTENT_SR as f32 / FBANK_FFT as f32;
    let (low, high) = (mel(FBANK_LOW_HZ), mel(CONTENT_SR as f32 / 2.0));
    let delta = (high - low) / (FBANK_BINS + 1) as f32;

    let bins = FBANK_FFT / 2;
    let mut fb = vec![0f32; FBANK_BINS * bins];
    for b in 0..FBANK_BINS {
        let (left, centre, right) = (
            low + b as f32 * delta,
            low + (b + 1) as f32 * delta,
            low + (b + 2) as f32 * delta,
        );
        for k in 0..bins {
            let m = mel(bin_width * k as f32);
            let up = (m - left) / (centre - left);
            let down = (right - m) / (right - centre);
            fb[b * bins + k] = up.min(down).max(0.0);
        }
    }
    fb
}

/// In-place radix-2 Cooley–Tukey, forward transform.
///
/// Written out rather than reached for because the only alternative in the
/// workspace is `burn_vits::Spectral`'s fused-DFT convolution, which cannot
/// express this front end: Kaldi's per-frame DC removal and pre-emphasis are
/// frame-local, so they do not factor out into a filter over the whole waveform.
fn fft(re: &mut [f32], im: &mut [f32]) {
    let n = re.len();
    let mut j = 0;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j |= bit;
        if i < j {
            re.swap(i, j);
            im.swap(i, j);
        }
    }

    let mut len = 2;
    while len <= n {
        let step = -2.0 * PI / len as f32;
        for start in (0..n).step_by(len) {
            for k in 0..len / 2 {
                let (wr, wi) = ((step * k as f32).cos(), (step * k as f32).sin());
                let (a, b) = (start + k, start + k + len / 2);
                let (vr, vi) = (re[b] * wr - im[b] * wi, re[b] * wi + im[b] * wr);
                let (ur, ui) = (re[a], im[a]);
                re[a] = ur + vr;
                im[a] = ui + vi;
                re[b] = ur - vr;
                im[b] = ui - vi;
            }
        }
        len <<= 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tone(hz: f32, seconds: f32) -> Vec<f32> {
        let n = (CONTENT_SR as f32 * seconds) as usize;
        (0..n)
            .map(|i| (2.0 * PI * hz * i as f32 / CONTENT_SR as f32).sin() * 0.5)
            .collect()
    }

    /// The transform this file exists to reproduce is checked against a reference
    /// implementation nowhere, so the least that has to hold is that a pure tone
    /// lands in the band that contains it and nowhere else.
    #[test]
    fn a_tone_peaks_in_the_band_that_holds_it() {
        let (fbank, frames) = kaldi_fbank(&tone(1000.0, 0.5));
        assert_eq!(frames, 1 + (8000 - FBANK_WINDOW) / FBANK_SHIFT);
        assert!(fbank.iter().all(|v| v.is_finite()), "fbank is not finite");

        // Which band 1 kHz falls in follows from the mel grid rather than from a
        // constant, so it is derived the same way the bank is.
        let mel = |hz: f32| 1127.0 * (1.0 + hz / 700.0f32).ln();
        let (low, high) = (mel(FBANK_LOW_HZ), mel(CONTENT_SR as f32 / 2.0));
        let delta = (high - low) / (FBANK_BINS + 1) as f32;
        let want = ((mel(1000.0) - low) / delta - 1.0).round() as usize;

        // A middle frame, so the tone is steady across the whole window.
        let row = &fbank[(frames / 2) * FBANK_BINS..][..FBANK_BINS];
        let peak = row
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        assert!(
            peak.abs_diff(want) <= 1,
            "1 kHz peaked in band {peak}, expected {want}"
        );
    }

    /// Framing is `snip_edges`, so the last partial window is dropped and a clip
    /// shorter than one window yields nothing at all rather than a padded frame.
    #[test]
    fn framing_snips_the_edges() {
        assert_eq!(kaldi_fbank(&tone(440.0, 0.01)).1, 0);
        for (samples, want) in [(400, 1), (559, 1), (560, 2), (16_000, 98)] {
            let wave = vec![0.1f32; samples];
            assert_eq!(kaldi_fbank(&wave).1, want, "{samples} samples");
        }
    }

    /// Kaldi's triangles rise towards 1 at their centre and are scaled by
    /// **nothing** — librosa's Slaney normalisation would divide each band by its
    /// own width, which is the difference that loads perfectly and shifts every
    /// embedding.
    ///
    /// So the claim is that no tap exceeds 1 and the wide bands reach it. The low
    /// bands do not, and that is Kaldi's behaviour rather than a defect: at 20 Hz
    /// a band is narrower than the 31.25 Hz bin spacing, so no bin lands on the
    /// apex — band 0 peaks at 0.50. A normalised bank would instead have peaks
    /// spread over two orders of magnitude.
    #[test]
    fn the_filterbank_is_unnormalised_triangles() {
        let bank = kaldi_mel_bank();
        let bins = FBANK_FFT / 2;
        assert_eq!(bank.len(), FBANK_BINS * bins);

        let peak = |b: usize| {
            bank[b * bins..][..bins]
                .iter()
                .copied()
                .fold(0.0f32, f32::max)
        };
        for b in 0..FBANK_BINS {
            assert!(
                (0.0..=1.0).contains(&peak(b)),
                "band {b} peaks at {}, outside an unnormalised triangle",
                peak(b)
            );
            assert!(
                bank[b * bins..][..bins].iter().all(|w| *w >= 0.0),
                "band {b} has a negative tap"
            );
        }
        // The tallest tap all but reaches the apex — 0.99, since no bin lands on
        // one exactly — which is what pins the scale and rules out any
        // normalisation: Slaney's would put every peak near 0.04.
        let highest = (0..FBANK_BINS).map(peak).fold(0.0f32, f32::max);
        assert!(highest > 0.98, "the tallest tap is {highest}");
        // And from half way up, where a band spans at least two bins either side
        // of its centre, the apex can only be missed by a fraction of a bin.
        for b in FBANK_BINS / 2..FBANK_BINS {
            assert!(peak(b) > 0.75, "band {b} peaks at {}", peak(b));
        }
        // The first bin is DC, which is below `low_freq` and so belongs to no band.
        assert!(bank.iter().step_by(bins).all(|w| *w == 0.0));
    }

    /// A constant transforms to a single non-zero bin, which is the cheapest
    /// statement that the butterflies and the bit reversal agree.
    #[test]
    fn the_transform_puts_a_constant_at_dc() {
        let mut re = vec![1.0f32; FBANK_FFT];
        let mut im = vec![0.0f32; FBANK_FFT];
        fft(&mut re, &mut im);
        assert!((re[0] - FBANK_FFT as f32).abs() < 1e-3, "{}", re[0]);
        assert!(re[1..].iter().all(|v| v.abs() < 1e-2));
        assert!(im.iter().all(|v| v.abs() < 1e-2));
    }
}
