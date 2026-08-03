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
//! | [`Spectral`] and [`Fbank`] | nothing — both derived from the preset |
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

use std::path::Path;

use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor, TensorData};
use burn_seedvc::campplus::{CamPPlus, CamPPlusConfig};
use burn_seedvc::content::{CONTENT_STRIDE, ContentEncoder, WINDOW_SAMPLES, mel_config};
use burn_seedvc::fbank::{Fbank, FbankConfig};
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
    /// The mel every module after the content encoder speaks, at 22.05 kHz.
    spectral: Spectral<B>,
    /// CAMPPlus's own front end, at 16 kHz — a different transform for a
    /// different consumer, and both are 80 band, so substituting one for the
    /// other would run and compute something else.
    fbank: Fbank<B>,
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
            fbank: Fbank::new(&FbankConfig::default(), device),
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

        // `Fbank::forward` asserts on a clip too short to frame, and an assert
        // would name the transform's invariant rather than the clip that broke it.
        let window = FbankConfig::default().window();
        if content.len() < window {
            return Err(Error::Reference(format!(
                "the reference clip is {} samples at {CONTENT_SR} Hz, shorter than the timbre \
                 encoder's own {window}-sample analysis window",
                content.len(),
            )));
        }
        // `Fbank` hands back `[batch, frames, bins]` — the axis order CAMPPlus
        // takes, and the opposite of everything else here — with the per-bin mean
        // over time **already subtracted**. That subtraction living inside
        // `forward` rather than at this call site is the whole reason this went
        // back to `burn-seedvc`: upstream spells it as a separate line, and a
        // separate line is one that gets dropped, silently, leaving the embedding
        // carrying the recording's channel alongside the speaker.
        let style = self
            .campplus
            .forward(self.fbank.forward(self.audio(content)));

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
