//! The ONNX Runtime implementation of [`Model`].
//!
//! Consumes the six graphs `export/export_seedvc.py` writes — `content`,
//! `style`, `mel`, `regulator`, `dit` and `bigvgan` — whose contracts are
//! documented in `export/README.md`. Everything above this file is shared with
//! the Burn path, including the reference analysis and the flow-matching Euler
//! loop, so the two runtimes differ only in the arithmetic.
//!
//! Three things happen on the host, because no graph can express them:
//!
//! - **Whisper's fixed 30 s window.** `content.onnx` is static — it always
//!   takes `[1, 480000]` and always returns `[1, 1500, 768]` — so the host
//!   pads a shorter clip with zeros and slices `samples / 320 + 1` frames off
//!   the output, the same bound `content.rs` folds into `forward`.
//! - **The length regulator's gather indices.** `picks[i] =
//!   min(floor(i · source / target), source − 1)` is data-dependent control
//!   flow over a length the graph has no way to know, so the host computes the
//!   `[T] i64` tensor and the graph is a plain gather followed by the conv
//!   stack.
//! - **The Euler loop.** `dit.onnx` is one velocity evaluation; [`euler`] is
//!   the integration, written over plain `f32` so it can be pinned against
//!   `burn_seedvc::flow::Sampler` without any graph.

use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard};

use burn_seedvc::SeedVcConfig;
use burn_seedvc::content::{CONTENT_STRIDE, WINDOW_SAMPLES};
use burn_seedvc::fbank::FbankConfig;
use burn_seedvc::flow::Sampler;
use ort::session::Session;
use ort::value::Tensor;

use crate::error::{Error, Result};
use crate::model::{CONTENT_SR, Model, Reference};

/// The six graphs, and the names [`OnnxModel::find`] looks for.
const GRAPHS: [&str; 6] = [
    "content.onnx",
    "style.onnx",
    "mel.onnx",
    "regulator.onnx",
    "dit.onnx",
    "bigvgan.onnx",
];

/// One loaded Seed-VC, running on ONNX Runtime.
///
/// Holds the six sessions the exporter writes and the preset config, and
/// nothing else — there is no state between calls, so one model converts any
/// number of sources against any number of references, exactly as
/// [`BurnModel`](crate::model::BurnModel) does. Each session sits behind a
/// `Mutex` only because `Session::run` asks for `&mut self`; see [`lock`].
pub struct OnnxModel {
    content: Mutex<Session>,
    style: Mutex<Session>,
    mel: Mutex<Session>,
    regulator: Mutex<Session>,
    dit: Mutex<Session>,
    bigvgan: Mutex<Session>,
    cfg: SeedVcConfig,
}

impl OnnxModel {
    /// The directory holding an export, if it has one.
    ///
    /// Both layouts the exporter's README suggests: the graphs directly in
    /// `dir`, or under `dir/onnx`. Used by `auto` backend selection, so it must
    /// answer without loading anything.
    pub fn find(dir: &Path) -> Option<PathBuf> {
        [dir.join("onnx"), dir.to_path_buf()]
            .into_iter()
            .find(|d| GRAPHS.iter().all(|name| d.join(name).exists()))
    }

    /// Load all six graphs.
    pub fn load(dir: &Path) -> Result<Self> {
        let dir = Self::find(dir).ok_or_else(|| Error::Load {
            what: "a Seed-VC ONNX export",
            why: format!(
                "no Seed-VC ONNX export under {} or {}/onnx — expected {} \
                 (write them with `export/export_seedvc.py`)",
                dir.display(),
                dir.display(),
                GRAPHS.join(", "),
            ),
        })?;
        Ok(Self {
            content: Mutex::new(session(&dir.join("content.onnx"))?),
            style: Mutex::new(session(&dir.join("style.onnx"))?),
            mel: Mutex::new(session(&dir.join("mel.onnx"))?),
            regulator: Mutex::new(session(&dir.join("regulator.onnx"))?),
            dit: Mutex::new(session(&dir.join("dit.onnx"))?),
            bigvgan: Mutex::new(session(&dir.join("bigvgan.onnx"))?),
            cfg: SeedVcConfig::uvit_whisper_small_wavenet(),
        })
    }

    /// Whisper's encoder over one clip, the host-side window arithmetic applied.
    ///
    /// Pads `samples` to the graph's fixed 30 s window, runs it, then keeps the
    /// leading `samples / 320 + 1` frames — everything after that describes
    /// nothing but the padding. Returns the features flat as `[1, frames, 768]`
    /// and the kept frame count, which is what the length regulator resamples.
    fn run_content(&self, samples: &[f32]) -> Result<(Vec<f32>, usize)> {
        let mut audio = vec![0.0f32; WINDOW_SAMPLES];
        audio[..samples.len()].copy_from_slice(samples);
        let mut session = lock(&self.content);
        let outputs = session
            .run(ort::inputs!["audio" => Tensor::from_array((
                vec![1i64, WINDOW_SAMPLES as i64],
                audio,
            )).map_err(onnx)?])
            .map_err(onnx)?;
        let (shape, features) = outputs["content"]
            .try_extract_tensor::<f32>()
            .map_err(onnx)?;
        let frames = (samples.len() / CONTENT_STRIDE + 1).min(shape[1] as usize);
        Ok((features[..frames * self.cfg.content_dim].to_vec(), frames))
    }

    /// CAMPPlus's timbre vector for one clip — the whole speaker specification.
    fn run_style(&self, samples: &[f32]) -> Result<Vec<f32>> {
        let mut session = lock(&self.style);
        let outputs = session
            .run(ort::inputs!["audio" => Tensor::from_array((
                vec![1i64, samples.len() as i64],
                samples.to_vec(),
            )).map_err(onnx)?])
            .map_err(onnx)?;
        let (_, style) = outputs["style"].try_extract_tensor::<f32>().map_err(onnx)?;
        Ok(style.to_vec())
    }

    /// The 22 050 Hz log-mel of one clip, `[1, 80, samples / 256]`.
    fn run_mel(&self, samples: &[f32]) -> Result<(Vec<f32>, usize)> {
        let mut session = lock(&self.mel);
        let outputs = session
            .run(ort::inputs!["audio" => Tensor::from_array((
                vec![1i64, samples.len() as i64],
                samples.to_vec(),
            )).map_err(onnx)?])
            .map_err(onnx)?;
        let (shape, mel) = outputs["mel"].try_extract_tensor::<f32>().map_err(onnx)?;
        Ok((mel.to_vec(), shape[2] as usize))
    }

    /// The length regulator: content at 50 Hz onto the mel's grid.
    ///
    /// `content` is flat `[1, content_frames, 768]` and `picks` the host-side
    /// gather indices [`pick_indices`] computed for the target frame count;
    /// what comes back is `[1, picks.len(), hidden_dim]`, the transformer's
    /// conditioning.
    fn run_regulator(
        &self,
        content: &[f32],
        content_frames: usize,
        picks: &[i64],
    ) -> Result<Vec<f32>> {
        let mut session = lock(&self.regulator);
        let outputs = session
            .run(ort::inputs![
                "content" => Tensor::from_array((
                    vec![1i64, content_frames as i64, self.cfg.content_dim as i64],
                    content.to_vec(),
                )).map_err(onnx)?,
                "picks" => Tensor::from_array((vec![picks.len() as i64], picks.to_vec()))
                    .map_err(onnx)?,
            ])
            .map_err(onnx)?;
        let (_, cond) = outputs["cond"].try_extract_tensor::<f32>().map_err(onnx)?;
        Ok(cond.to_vec())
    }

    /// One velocity evaluation — the whole of what `dit.onnx` computes.
    fn run_dit(
        &self,
        x: &[f32],
        prompt_x: &[f32],
        t: f64,
        style: &[f32],
        cond: &[f32],
        frames: usize,
    ) -> Result<Vec<f32>> {
        let cfg = &self.cfg;
        let mut session = lock(&self.dit);
        let outputs = session
            .run(ort::inputs![
                "x" => Tensor::from_array((
                    vec![1i64, cfg.n_mels as i64, frames as i64],
                    x.to_vec(),
                )).map_err(onnx)?,
                "prompt_x" => Tensor::from_array((
                    vec![1i64, cfg.n_mels as i64, frames as i64],
                    prompt_x.to_vec(),
                )).map_err(onnx)?,
                "t" => Tensor::from_array((vec![1i64], vec![t as f32])).map_err(onnx)?,
                "style" => Tensor::from_array((
                    vec![1i64, cfg.style_dim as i64],
                    style.to_vec(),
                )).map_err(onnx)?,
                "cond" => Tensor::from_array((
                    vec![1i64, frames as i64, cfg.hidden_dim as i64],
                    cond.to_vec(),
                )).map_err(onnx)?,
            ])
            .map_err(onnx)?;
        let (_, v) = outputs["v"].try_extract_tensor::<f32>().map_err(onnx)?;
        Ok(v.to_vec())
    }

    /// The vocoder: a mel `[1, 80, frames]` to audio at `frames × hop_length`.
    fn run_bigvgan(&self, mel: &[f32], frames: usize) -> Result<Vec<f32>> {
        let mut session = lock(&self.bigvgan);
        let outputs = session
            .run(ort::inputs!["mel" => Tensor::from_array((
                vec![1i64, self.cfg.n_mels as i64, frames as i64],
                mel.to_vec(),
            )).map_err(onnx)?])
            .map_err(onnx)?;
        let (_, audio) = outputs["audio"].try_extract_tensor::<f32>().map_err(onnx)?;
        Ok(audio.to_vec())
    }
}

impl Model for OnnxModel {
    fn config(&self) -> &SeedVcConfig {
        &self.cfg
    }

    fn analyse(&self, content: &[f32], mel: &[f32]) -> Result<Reference> {
        // The same bounds `BurnModel::analyse` checks, in the same words: a
        // reference that fills the content window, is shorter than one mel
        // frame, fills the transformer's block, or is too short for CAMPPlus's
        // statistics pooling is unusable no matter which runtime runs it.
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
        let fbank = FbankConfig::default();
        let minimum = fbank.window() + 2 * fbank.shift();
        if content.len() < minimum {
            return Err(Error::Reference(format!(
                "the reference clip is {} samples at {CONTENT_SR} Hz ({:.0} ms) and the timbre \
                 encoder needs at least {minimum} ({:.0} ms) — its statistics pooling takes a \
                 variance over time, which needs two frames to be defined at all. A clip this \
                 short cannot specify a speaker anyway; 1-30 s is the useful range",
                content.len(),
                content.len() as f64 * 1e3 / CONTENT_SR as f64,
                minimum as f64 * 1e3 / CONTENT_SR as f64,
            )));
        }

        let style = self.run_style(content)?;
        let (mel_values, mel_graph_frames) = self.run_mel(mel)?;
        if mel_graph_frames != frames {
            return Err(Error::Reference(format!(
                "the mel graph returned {mel_graph_frames} frames for {} samples at the \
                 {}-sample hop — the export does not match the preset",
                mel.len(),
                self.cfg.hop_length,
            )));
        }
        let (content, content_frames) = self.run_content(content)?;
        // The regulator resamples the clip's own content onto the clip's mel
        // grid, so the prompt the transformer is asked to continue is one whose
        // audio and content agree.
        let cond = self.run_regulator(
            &content,
            content_frames,
            &pick_indices(content_frames, frames),
        )?;

        Ok(Reference {
            frames,
            mel: mel_values,
            cond,
            style,
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
        let cfg = &self.cfg;
        let total = reference.frames + frames;
        // The same bounds `BurnModel::convert` checks, in the same words.
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
        if total > cfg.block_size {
            return Err(Error::Input(format!(
                "{} reference frames plus {frames} generated frames is {total}, past the \
                 transformer's block_size of {} — chunk the source against the reference's \
                 length, not against the window alone",
                reference.frames, cfg.block_size,
            )));
        }
        if noise.len() != cfg.n_mels * total {
            return Err(Error::Input(format!(
                "the noise is {} values and the sampler wants {} ({} mels × {total} frames, \
                 the prompt region included even though it is overwritten)",
                noise.len(),
                cfg.n_mels * total,
                cfg.n_mels,
            )));
        }
        if reference.frames == 0 {
            return Err(Error::Input(
                "the reference specifies no frames — it conditions on nothing".into(),
            ));
        }
        if reference.mel.len() != cfg.n_mels * reference.frames
            || reference.cond.len() != reference.frames * cfg.hidden_dim
            || reference.style.len() != cfg.style_dim
        {
            return Err(Error::Input(
                "the reference's tensors do not match its frame count — it was built for a \
                 different preset"
                    .into(),
            ));
        }

        let (content, content_frames) = self.run_content(source)?;
        let cond_source = self.run_regulator(
            &content,
            content_frames,
            &pick_indices(content_frames, frames),
        )?;
        // The reference's conditioning leads, so what the transformer is asked
        // for is a continuation rather than an imitation.
        let mut cond = Vec::with_capacity(total * cfg.hidden_dim);
        cond.extend_from_slice(&reference.cond);
        cond.extend_from_slice(&cond_source);

        let mel = euler(
            |x, prompt_x, t, style, cond| self.run_dit(x, prompt_x, t, style, cond, total),
            noise,
            &reference.mel,
            cfg.n_mels,
            total,
            reference.frames,
            &reference.style,
            &cond,
            sampler,
        )?;

        // The prompt region comes back still pinned at zero — [`euler`] returns
        // the whole window and slicing it off is documented as the caller's job.
        let mut generated = Vec::with_capacity(cfg.n_mels * frames);
        for c in 0..cfg.n_mels {
            generated.extend_from_slice(&mel[c * total + reference.frames..(c + 1) * total]);
        }
        let audio = self.run_bigvgan(&generated, frames)?;
        if audio.len() != frames * cfg.hop_length {
            return Err(Error::Input(format!(
                "the vocoder returned {} samples for {frames} mel frames — the export does not \
                 match the preset's {}-sample hop",
                audio.len(),
                cfg.hop_length,
            )));
        }
        Ok(audio)
    }
}

/// The nearest-neighbour source index per output frame — the `picks` input.
///
/// Upstream's `F.interpolate(mode='nearest')` arithmetic, and it lives on the
/// host because it is data-dependent control flow over a length the graph is
/// not told. The float multiply-then-truncate is reproduced rather than
/// simplified: `floor(i · source / frames)` in integers disagrees with PyTorch
/// wherever the division is inexact, which is every frame of a 50 Hz → 86 Hz
/// resample.
fn pick_indices(source: usize, frames: usize) -> Vec<i64> {
    let scale = source as f64 / frames as f64;
    (0..frames)
        .map(|i| ((i as f64 * scale) as i64).min(source as i64 - 1))
        .collect()
}

/// Integrate `dx/dt = v(x, t)` from noise at `t = 0` to a mel at `t = 1`.
///
/// This is `flow.rs`'s `Sampler::sample` over plain `f32`, written out so the
/// ONNX runtime can drive it without a Burn backend — the sampler is generic
/// over `Estimator<B>` and therefore inaccessible here. `velocity` is one
/// velocity evaluation, the closure that wraps `dit.onnx`; the loop calls it
/// once per step, and twice when guidance is on, exactly as the Burn side
/// stacks the conditioned and unconditional inputs into one call.
///
/// Layout is row-major `[channels, frames]` throughout. `prompt` is the
/// leading `prompt_frames` frames of the window, and `prompt_x` is it padded
/// to full width with zeros; the matching frames of the state are pinned at
/// zero for the whole integration and come back still zero, as they do in
/// `flow.rs`.
#[allow(clippy::too_many_arguments)]
fn euler(
    velocity: impl Fn(&[f32], &[f32], f64, &[f32], &[f32]) -> Result<Vec<f32>>,
    noise: &[f32],
    prompt: &[f32],
    channels: usize,
    frames: usize,
    prompt_frames: usize,
    style: &[f32],
    cond: &[f32],
    sampler: Sampler,
) -> Result<Vec<f32>> {
    assert!(
        prompt_frames > 0 && prompt_frames < frames,
        "the reference clip is the whole speaker specification, so a zero-length prompt \
         conditions on nothing, and one filling the whole {frames}-frame window leaves \
         nothing to generate (got {prompt_frames})"
    );
    assert_eq!(
        noise.len(),
        channels * frames,
        "the noise must span the whole window"
    );
    assert_eq!(
        prompt.len(),
        channels * prompt_frames,
        "the prompt is a slice of the window"
    );

    // The reference sits in the leading frames of a full-width tensor, and the
    // matching frames of the state stay zero from here to the end.
    let mut prompt_x = vec![0.0; channels * frames];
    for c in 0..channels {
        prompt_x[c * frames..c * frames + prompt_frames]
            .copy_from_slice(&prompt[c * prompt_frames..(c + 1) * prompt_frames]);
    }
    let mut x = noise.to_vec();
    for frame in 0..prompt_frames {
        for c in 0..channels {
            x[c * frames + frame] = 0.0;
        }
    }

    // The unconditional branch zeroes the same three signals; built once so the
    // loop does not rebuild them every step.
    let blank_prompt = vec![0.0; prompt_x.len()];
    let blank_style = vec![0.0; style.len()];
    let blank_cond = vec![0.0; cond.len()];

    let dt = 1.0 / sampler.steps as f64;
    for step in 0..sampler.steps {
        let t = step as f64 * dt;
        let v_cond = velocity(&x, &prompt_x, t, style, cond)?;
        if v_cond.len() != channels * frames {
            return Err(Error::Input(format!(
                "the velocity graph returned {} values for a {channels}×{frames} state — the \
                 export does not match the preset",
                v_cond.len(),
            )));
        }
        let v = if sampler.guidance > 0.0 {
            // `(1 + w)·v_cond − w·v_uncond`: the scale measures how far *past*
            // the conditioned prediction to extrapolate, away from the
            // unconditional one, exactly as `flow.rs` blends them.
            let v_uncond = velocity(&x, &blank_prompt, t, &blank_style, &blank_cond)?;
            let g = sampler.guidance as f32;
            v_cond
                .iter()
                .zip(v_uncond)
                .map(|(c, u)| (1.0 + g) * c - g * u)
                .collect()
        } else {
            v_cond
        };
        for (xi, vi) in x.iter_mut().zip(v) {
            *xi += vi * dt as f32;
        }
        for frame in 0..prompt_frames {
            for c in 0..channels {
                x[c * frames + frame] = 0.0;
            }
        }
    }
    Ok(x)
}

/// A session behind its lock, for the duration of one graph run.
///
/// `Session::run` takes `&mut self`, while [`Model`] is `&self` — the whole
/// point of that boundary is that a conversion holds no state between calls, so
/// each session mutex is locked for one run and released. The lock is never
/// contended in practice; the `expect` covers a panic happening while a graph
/// ran, which this runtime cannot recover from.
fn lock(session: &Mutex<Session>) -> MutexGuard<'_, Session> {
    session.lock().expect("an ONNX session poisoned its lock")
}

/// Build a session with CUDA-then-CPU execution providers.
///
/// The fourth copy of these six lines — `stt-core`, `rvc-core` and `tts-core`
/// each carry them, and they are not shared because engines do not depend on
/// each other. `tts-core`'s comment marks the third copy as the moment to
/// consider a kit crate rather than the moment to reach across — four is past
/// that bar, the count at which lifting the six lines into `cli-kit` becomes
/// worthwhile.
fn session(path: &Path) -> Result<Session> {
    use ort::execution_providers::{CPU, CUDA};

    Session::builder()
        .map_err(onnx)?
        .with_execution_providers([CUDA::default().build(), CPU::default().build()])
        .map_err(|e| Error::Onnx(e.to_string()))?
        .commit_from_file(path)
        .map_err(|e| Error::Onnx(format!("{}: {e}", path.display())))
}

fn onnx(e: ort::Error) -> Error {
    Error::Onnx(e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::backend::Backend;
    use burn::tensor::{Int, Tensor, TensorData};
    use burn_seedvc::flow::Estimator;

    type B = burn_ndarray::NdArray;

    const CHANNELS: usize = 2;
    const FRAMES: usize = 4;
    const PROMPT: usize = 1;
    const STYLE: usize = 3;
    const HIDDEN: usize = 5;

    /// `dx/dt = x` — the field `flow.rs` integrates to pin its own solver.
    struct Linear;

    impl<B: Backend> Estimator<B> for Linear {
        fn velocity(
            &self,
            x: Tensor<B, 3>,
            _prompt_x: Tensor<B, 3>,
            _x_lens: Tensor<B, 1, Int>,
            _t: Tensor<B, 1>,
            _style: Tensor<B, 2>,
            _cond: Tensor<B, 3>,
        ) -> Tensor<B, 3> {
            x
        }
    }

    /// The whole point of writing the loop out: on the same inputs it has to
    /// give the Burn sampler's answer, and the closed-form error `flow.rs`
    /// measures has to be the one this loop makes. Both solvers start from
    /// noise of ones and integrate `dx/dt = x`, so the generated frames should
    /// approach `e` from below at first order.
    #[test]
    fn the_euler_loop_matches_flow_rs_on_dx_dt_is_x() {
        let device = Default::default();
        // The same flat data feeds both solvers: noise and prompt all ones, so
        // the generated frames integrate from 1 towards e; style and cond are
        // arbitrary but finite (the field ignores them).
        let noise = vec![1.0f32; CHANNELS * FRAMES];
        let prompt = vec![1.0f32; CHANNELS * PROMPT];
        let style = vec![1.0f32; STYLE];
        let cond = vec![1.0f32; FRAMES * HIDDEN];

        let burn = |steps: usize| -> Vec<f32> {
            let out = Sampler {
                steps,
                guidance: 0.0,
            }
            .sample(
                &Linear,
                Tensor::<B, 3>::from_data(
                    TensorData::new(noise.clone(), [1, CHANNELS, FRAMES]),
                    &device,
                ),
                Tensor::<B, 3>::from_data(
                    TensorData::new(prompt.clone(), [1, CHANNELS, PROMPT]),
                    &device,
                ),
                Tensor::<B, 3>::from_data(
                    TensorData::new(cond.clone(), [1, FRAMES, HIDDEN]),
                    &device,
                ),
                Tensor::<B, 2>::from_data(TensorData::new(style.clone(), [1, STYLE]), &device),
                Tensor::<B, 1, Int>::from_ints([FRAMES as i32], &device),
            );
            out.into_data().to_vec().unwrap()
        };

        let ours = |steps: usize| -> Vec<f32> {
            euler(
                |x, _prompt_x, _t, _style, _cond| Ok(x.to_vec()),
                &noise,
                &prompt,
                CHANNELS,
                FRAMES,
                PROMPT,
                &style,
                &cond,
                Sampler {
                    steps,
                    guidance: 0.0,
                },
            )
            .unwrap()
        };

        for steps in [4, 10, 50] {
            let expected = burn(steps);
            let got = ours(steps);
            for (i, (a, b)) in expected.iter().zip(&got).enumerate() {
                assert!(
                    (a - b).abs() < 1e-4,
                    "{steps} steps, element {i}: Burn {a}, ours {b}"
                );
            }
            let error = (got[PROMPT] as f64 - std::f64::consts::E).abs();
            let expected_error = match steps {
                4 => 0.276876,
                10 => 0.124539,
                50 => 0.026694,
                _ => unreachable!(),
            };
            assert!(
                (error - expected_error).abs() < 1e-4,
                "{steps} steps: error {error} != {expected_error}"
            );
        }
    }

    /// Guidance zero must be the **conditioned** velocity and a positive scale
    /// the exact extrapolation away from the unconditional one — the closed-form
    /// check `flow.rs`'s own guidance test makes, repeated here because this is
    /// the second implementation of the blend.
    #[test]
    fn guidance_extrapolates_from_the_conditioned_velocity() {
        // `velocity = 1 + Σstyle + Σprompt_x + Σcond`, the flow.rs toy: with
        // ones everywhere the conditioned value is 26 and the unconditional one
        // (all three signals zeroed) is 1. One step of Δt = 1 from a zero
        // state, so the generated frame *is* the blended velocity.
        let conditioned = 26.0;
        let unconditional = 1.0;

        let velocity = |guidance: f64| {
            let out = euler(
                |x, prompt_x, _t, style, cond| {
                    let total = style.iter().sum::<f32>()
                        + prompt_x.iter().sum::<f32>()
                        + cond.iter().sum::<f32>();
                    Ok(x.iter().map(|_| 1.0 + total).collect())
                },
                &[0.0; CHANNELS * FRAMES],
                &[1.0; CHANNELS * PROMPT],
                CHANNELS,
                FRAMES,
                PROMPT,
                &[1.0; STYLE],
                &[1.0; FRAMES * HIDDEN],
                Sampler { steps: 1, guidance },
            )
            .unwrap();
            assert!(
                out[..PROMPT].iter().all(|v| *v == 0.0),
                "the prompt region must stay pinned at zero"
            );
            out[PROMPT] as f64
        };

        let unguided = velocity(0.0);
        assert!(
            (unguided - conditioned).abs() < 1e-4,
            "guidance 0 must reduce to the conditioned velocity, got {unguided}"
        );
        for guidance in [0.5, 0.7, 3.0] {
            let expected = (1.0 + guidance) * conditioned - guidance * unconditional;
            let got = velocity(guidance);
            assert!(
                (got - expected).abs() < 1e-3,
                "guidance {guidance}: {got} != {expected}"
            );
        }
    }

    /// `find` answers from the filesystem alone, in both layouts the exporter
    /// suggests — the graphs directly in a directory or under its `onnx/`.
    #[test]
    fn find_spots_both_export_layouts() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let root = std::env::temp_dir().join(format!(
            "seedvc-onnx-find-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed),
        ));
        std::fs::create_dir_all(&root).unwrap();

        // Graphs nested under `onnx/` first.
        let nested = root.join("onnx");
        std::fs::create_dir_all(&nested).unwrap();
        for name in GRAPHS {
            std::fs::write(nested.join(name), b"").unwrap();
        }
        assert_eq!(OnnxModel::find(&root), Some(nested.clone()));

        // Moved up a level, found there instead — and a missing graph is not
        // an export at all.
        for name in GRAPHS {
            std::fs::rename(nested.join(name), root.join(name)).unwrap();
        }
        std::fs::remove_dir_all(&nested).unwrap();
        assert_eq!(OnnxModel::find(&root), Some(root.clone()));
        std::fs::remove_file(root.join("dit.onnx")).unwrap();
        assert_eq!(OnnxModel::find(&root), None);

        std::fs::remove_dir_all(&root).unwrap();
    }

    /// The gather indices are PyTorch's own `F.interpolate(mode='nearest')`
    /// arithmetic — the check `length_regulator.py`'s `pick_indices` makes, so
    /// the two implementations cannot drift.
    #[test]
    fn pick_indices_reproduce_nearest_interpolation() {
        assert_eq!(pick_indices(50, 86)[..3], [0, 0, 1]);
        assert_eq!(pick_indices(50, 86).last(), Some(&49));
        // A 1:1 resample is the identity, and the final index is clamped so a
        // boundary rounding cannot point past the last source frame.
        assert_eq!(
            pick_indices(86, 86),
            (0..86).map(|i| i as i64).collect::<Vec<_>>()
        );
        let picks = pick_indices(50, 86);
        assert!(
            picks.windows(2).all(|w| w[0] <= w[1]),
            "picks must be non-decreasing"
        );
        assert!(
            picks.iter().all(|&p| p < 50),
            "a pick must name an existing source frame"
        );
    }
}
