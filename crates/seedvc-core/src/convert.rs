//! The batch conversion pipeline.
//!
//! A file of somebody's speech in, the same words in the reference's voice out.
//! [`Model::convert`] does one chunk of that; everything here is the arithmetic
//! that decides what a chunk is, plus the crossfade that joins them.
//!
//! # One 30 s window, shared
//!
//! The transformer is shown the reference's mel and the source's conditioning in
//! **one** sequence, so the two compete for the same room. Upstream calls it
//! `max_context_window = sr // hop_length * 30` — integer division first, so it
//! is **2580** frames rather than 2584 — and takes the source's share as what is
//! left after the reference:
//!
//! ```text
//! max_source_window = max_context_window − reference.frames
//! ```
//!
//! That is the fact worth carrying away from this module: **a longer reference
//! buys a better speaker vector and costs source per chunk.** At the 25 s cap
//! [`crate::reference`] applies, the reference is 2153 frames and a chunk is
//! under 5 s; at 5 s of reference a chunk is nearly 25 s. Neither is wrong, and
//! nothing warns.
//!
//! # Where this departs from upstream, and why it has to
//!
//! Upstream runs Whisper over the **whole** source first — sliding a 30 s window
//! with 5 s of overlap and discarding the overlapping 250 content frames — then
//! length-regulates the result and chunks *that*. [`Model::convert`] takes audio
//! rather than conditioning, so a chunk here is a Whisper window: the content
//! encoding and the conversion are chunked together, on the same boundaries, and
//! the 5 s content overlap collapses into the 16-frame conversion overlap.
//!
//! **The cost is context, and it is worth naming as a guess rather than a
//! measurement**: each chunk's content is encoded with 0.19 s of left context
//! where upstream gives it 5 s, so a chunk boundary is a place Whisper sees less
//! than it otherwise would. It is not audible on the clips this was developed
//! against, but those are single-chunk — under 25 s of source with a short
//! reference never reaches a second chunk at all, which is most real use. If a
//! seam ever *is* audible, this is the first thing to suspect, and the fix is a
//! `Model` method that returns conditioning instead of audio, not a wider
//! crossfade.
//!
//! # The chunks are independent generations
//!
//! Every chunk is a fresh integration from fresh noise against the same prompt,
//! so consecutive chunks agree on timbre and content but on nothing about phase.
//! That is upstream's design and the reason the join is a **cos² equal-power**
//! crossfade rather than a linear one: the two fades sum to 1 exactly, so a
//! steady signal crosses a seam unchanged, which a linear pair would dip.

use std::path::Path;

use audio_kit::DecodeOptions;
use burn_seedvc::SeedVcConfig;
use burn_seedvc::content::WINDOW_SAMPLES;
use burn_seedvc::flow::Sampler;
use futures::StreamExt;

use crate::error::{Error, Result};
use crate::model::{CONTENT_SR, Model, Reference};

/// Frames two consecutive chunks share — upstream's `overlap_frame_len`. At the
/// 256-sample hop that is 4096 samples, or 0.19 s.
pub const OVERLAP_FRAMES: usize = 16;

/// The transformer's context window as a duration; the frame count it becomes
/// depends on the hop, so it is derived rather than written down.
pub const CONTEXT_SECONDS: usize = 30;

/// Smallest chunk of *new* audio worth asking for, on top of the crossfade.
///
/// Not upstream's — upstream chunks conditioning it has already computed, so its
/// last chunk can be one frame wide and still make sense. Here a chunk is audio
/// fed to Whisper, and a chunk under a content frame is refused by
/// [`Model::convert`] rather than merely being small. So the *previous* chunk
/// gives back a few frames to keep the last one above this, which is 0.37 s.
const MIN_CHUNK_FRAMES: usize = 32;

/// Frames the source is left after the reference has taken its share of the
/// shared window, or the refusal when there are not enough of them.
///
/// Both paths ask this before anything is converted — [`convert`] once per file
/// and [`Converter::new`](crate::stream::Converter::new) once per session — so a
/// reference that fills the window is refused in the same words either way, and
/// in terms of the only thing the user can do about it.
pub(crate) fn room(cfg: &SeedVcConfig, reference: &Reference) -> Result<usize> {
    let hop = cfg.hop_length;
    let context = (cfg.sample_rate as usize / hop) * CONTEXT_SECONDS;

    // Saturating, because this is the subtraction that underflows: `analyse`
    // admits a reference up to the content encoder's own 30 s, which is
    // 2584 frames — four more than the window they then have to share.
    let room = context.saturating_sub(reference.frames);
    if room <= OVERLAP_FRAMES + MIN_CHUNK_FRAMES {
        return Err(Error::Input(format!(
            "the reference is {} of the {context} mel frames the transformer sees at once \
             ({CONTEXT_SECONDS} s at {} Hz over a hop of {hop}), leaving {room} for the source — \
             a chunk carries {OVERLAP_FRAMES} frames of crossfade and at least \
             {MIN_CHUNK_FRAMES} of new audio, so trim the reference",
            reference.frames, cfg.sample_rate,
        )));
    }
    Ok(room)
}

/// Blend a chunk's head over the previous chunk's held-back tail, in place.
///
/// **cos² equal-power**, not linear: the two fades sum to 1 exactly, so a steady
/// signal crosses the seam unchanged where a linear pair would dip. Every chunk
/// is an independent generation from fresh noise against the same prompt, which
/// is what makes the choice matter — the two sides agree on timbre and content
/// and on nothing about phase.
///
/// The fade is as long as `previous`, so an empty tail — the first chunk of a
/// file or of a stream — is a no-op.
pub(crate) fn crossfade(wave: &mut [f32], previous: &[f32]) {
    // The saturation guards the empty tail — every first chunk — and the `max`
    // the one-sample tail no caller produces; either would otherwise divide by
    // zero and write `NaN` into the output rather than failing.
    let last = previous.len().saturating_sub(1).max(1) as f32;
    for (i, (sample, held)) in wave.iter_mut().zip(previous).enumerate() {
        let theta = std::f32::consts::FRAC_PI_2 * i as f32 / last;
        *sample = *sample * theta.sin().powi(2) + held * theta.cos().powi(2);
    }
}

/// How to convert.
#[derive(Debug, Clone, Copy)]
pub struct ConvertOptions {
    /// Euler steps and classifier-free guidance — see [`Sampler`], whose
    /// [`Default`] is upstream's own 30 and 0.7.
    pub sampler: Sampler,
    /// Scales the output's duration against the source's. Above 1 is slower,
    /// below is faster; the content is resampled onto the new frame count by the
    /// length regulator, so this stretches delivery without touching pitch.
    pub length_adjust: f64,
    /// Seed for the sampler's noise, so a conversion can be repeated exactly.
    pub seed: u64,
}

impl Default for ConvertOptions {
    fn default() -> Self {
        Self {
            sampler: Sampler::default(),
            length_adjust: 1.0,
            seed: 0,
        }
    }
}

/// Decode a source recording and convert it into the reference's voice.
///
/// Returns mono `f32` at
/// [`SeedVcConfig::sample_rate`](burn_seedvc::SeedVcConfig::sample_rate).
pub async fn convert_path(
    model: &dyn Model,
    reference: &Reference,
    path: impl AsRef<Path>,
    opts: &ConvertOptions,
) -> Result<Vec<f32>> {
    let path = path.as_ref();
    let mut stream = Box::pin(audio_kit::decode_path(
        path.to_path_buf(),
        DecodeOptions::new(CONTENT_SR),
    ));
    let mut source = Vec::new();
    while let Some(chunk) = stream.next().await {
        source.extend_from_slice(&chunk?);
    }
    convert(model, reference, &source, opts)
}

/// Convert source audio already decoded to mono `f32` at [`CONTENT_SR`].
///
/// Of any length: the 30 s limit [`Model::convert`] enforces is a property of one
/// chunk, and chunking is this function's job.
pub fn convert(
    model: &dyn Model,
    reference: &Reference,
    source: &[f32],
    opts: &ConvertOptions,
) -> Result<Vec<f32>> {
    let cfg = model.config();
    let hop = cfg.hop_length;
    let room = room(cfg, reference)?;

    // The source's mel frame count, which upstream reads off a 22.05 kHz decode
    // of the same file. Derived from the 16 kHz length instead: the source's
    // waveform is never used at 22.05 kHz, only its duration, and a second
    // decode is a real cost for a number the two agree on to within one 11.6 ms
    // frame.
    let natural = source.len() * cfg.sample_rate as usize / CONTENT_SR as usize / hop;
    // `int(mel.size(2) * length_adjust)` — truncating, as upstream's does.
    let total = (natural as f64 * opts.length_adjust) as usize;
    if total == 0 {
        return Err(Error::Input(format!(
            "{:.2} s of source at a length adjustment of {} is under one mel frame",
            source.len() as f64 / CONTENT_SR as f64,
            opts.length_adjust,
        )));
    }

    // Source samples behind one output frame — the model's own 256/22050 moved
    // by `length_adjust`, and the map from a frame boundary to a cut in the
    // waveform.
    let per_frame = source.len() as f64 / total as f64;
    let at = |frame: usize| ((frame as f64 * per_frame).round() as usize).min(source.len());

    // A chunk is bounded twice: by the room left in the transformer's context,
    // and by the content encoder's own 30 s window. The second binds only when
    // `length_adjust` compresses time, since a frame then stands for more than
    // its usual 11.6 ms of source. One frame of slack, because `at` rounds.
    let by_audio = (WINDOW_SAMPLES as f64 / per_frame) as usize;
    let window = room.min(by_audio.saturating_sub(1));
    if total > window && window <= OVERLAP_FRAMES + MIN_CHUNK_FRAMES {
        return Err(Error::Input(format!(
            "a length adjustment of {} puts {:.1} s of source behind each {window}-frame chunk, \
             past the content encoder's {CONTEXT_SECONDS} s window — convert at a milder \
             adjustment, or resample the source first",
            opts.length_adjust,
            window as f64 * per_frame / CONTENT_SR as f64,
        )));
    }

    let overlap = OVERLAP_FRAMES * hop;
    let mut rng = Rng::new(opts.seed);
    let mut out: Vec<f32> = Vec::with_capacity(total * hop);
    let mut tail: Vec<f32> = Vec::new();
    let mut done = 0;
    while done < total {
        let mut frames = window.min(total - done);
        let last = done + window >= total;
        if !last {
            // What would be left after stepping on. Handing a few frames back
            // now is the cheap way to keep the final chunk convertible at all —
            // see `MIN_CHUNK_FRAMES`.
            let rest = total - (done + frames - OVERLAP_FRAMES);
            frames -= MIN_CHUNK_FRAMES.saturating_sub(rest);
        }

        // The prompt region's noise is overwritten by the sampler and drawn
        // anyway, because its width is part of the window the solver integrates.
        let noise: Vec<f32> = (0..cfg.n_mels * (reference.frames + frames))
            .map(|_| rng.next_normal())
            .collect();
        let mut wave = model.convert(
            &source[at(done)..at(done + frames)],
            frames,
            reference,
            &noise,
            opts.sampler,
        )?;

        crossfade(&mut wave, &tail);

        if last {
            out.extend_from_slice(&wave);
            break;
        }
        let keep = wave.len() - overlap;
        out.extend_from_slice(&wave[..keep]);
        tail.clear();
        tail.extend_from_slice(&wave[keep..]);
        done += frames - OVERLAP_FRAMES;
    }
    Ok(out)
}

/// A small deterministic generator.
///
/// The same xorshift64\* `tts-core` samples with, written again rather than
/// shared: no engine depends on another engine, and fifteen lines of arithmetic
/// is not enough model-free plumbing to earn a `*-kit` of its own. If a third
/// engine wants it, that is the point to move it.
///
/// [`crate::stream`] draws from one of these too, seeded the same way, which is
/// what lets a streamed conversion and a batch one be held against each other.
pub(crate) struct Rng(u64);

impl Rng {
    /// Seeded through splitmix64's finalizer, **not** by using the seed as state.
    ///
    /// `seed | 1` was the obvious thing and it makes seeds 0 and 1 the same run:
    /// the low bit is all that separates them and setting it erases the
    /// difference. That matters more here than anywhere else in the workspace —
    /// the noise is the *only* thing that varies a take, so `--seed 0` and
    /// `--seed 1` produced byte-identical audio, which is precisely the nudge a
    /// user reaches for. `| 1` afterwards only keeps the state non-zero, which
    /// the xorshift below requires.
    pub(crate) fn new(seed: u64) -> Self {
        let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        Self((z ^ (z >> 31)) | 1)
    }

    fn next_f32(&mut self) -> f32 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        let x = self.0.wrapping_mul(0x2545_F491_4F6C_DD1D);
        (x >> 40) as f32 / (1u32 << 24) as f32
    }

    /// A standard normal draw, which is what the solver starts from.
    ///
    /// Box–Muller, keeping one of the two values it produces: holding the spare
    /// would make the state depend on how many draws came before, and the one
    /// property that has to hold is that the same seed gives the same noise.
    pub(crate) fn next_normal(&mut self) -> f32 {
        let u = self.next_f32().max(f32::MIN_POSITIVE);
        let v = self.next_f32();
        (-2.0 * u.ln()).sqrt() * (2.0 * std::f32::consts::PI * v).cos()
    }
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;

    use burn_seedvc::SeedVcConfig;

    use super::*;

    /// A [`Model`] that converts nothing and records how it was asked to.
    ///
    /// The chunk arithmetic is what these tests are about, and it needs no
    /// weights — but it does need a `Model`, because the loop's shape *is* the
    /// sequence of calls it makes. Returning a constant is what makes the
    /// crossfade checkable: cos² and sin² sum to 1, so a steady signal has to
    /// cross every seam unchanged.
    struct Recorder {
        cfg: SeedVcConfig,
        calls: RefCell<Vec<(usize, usize)>>,
    }

    impl Recorder {
        fn new() -> Self {
            Self {
                cfg: SeedVcConfig::uvit_whisper_small_wavenet(),
                calls: RefCell::new(Vec::new()),
            }
        }
    }

    impl Model for Recorder {
        fn config(&self) -> &SeedVcConfig {
            &self.cfg
        }

        fn analyse(&self, _content: &[f32], _mel: &[f32]) -> Result<Reference> {
            unreachable!("these tests build a `Reference` by hand")
        }

        fn convert(
            &self,
            source: &[f32],
            frames: usize,
            reference: &Reference,
            noise: &[f32],
            _sampler: Sampler,
        ) -> Result<Vec<f32>> {
            assert!(
                source.len() <= WINDOW_SAMPLES,
                "{} samples is past the content encoder's window",
                source.len()
            );
            assert_eq!(noise.len(), self.cfg.n_mels * (reference.frames + frames));
            self.calls.borrow_mut().push((source.len(), frames));
            Ok(vec![1.0; frames * self.cfg.hop_length])
        }
    }

    fn reference(frames: usize) -> Reference {
        let cfg = SeedVcConfig::uvit_whisper_small_wavenet();
        Reference {
            frames,
            mel: vec![0.0; cfg.n_mels * frames],
            cond: vec![0.0; frames * cfg.hidden_dim],
            style: vec![0.0; cfg.style_dim],
        }
    }

    /// Seconds of 16 kHz source.
    fn source(seconds: f64) -> Vec<f32> {
        vec![0.0; (CONTENT_SR as f64 * seconds) as usize]
    }

    /// The whole point of the loop: the chunks have to reassemble into exactly
    /// the frame count that was asked for, whatever the reference's length does
    /// to the chunk size, and the seams must not dip.
    #[test]
    fn the_chunks_reassemble_into_one_signal() {
        let cfg = SeedVcConfig::uvit_whisper_small_wavenet();
        // 5 s of reference leaves nearly a 25 s chunk, so 90 s needs four of
        // them; 25 s of reference leaves under 5 s and needs twenty.
        for (reference_seconds, seconds) in [(5.0, 90.0), (25.0, 90.0), (5.0, 10.0)] {
            let model = Recorder::new();
            let reference = reference((reference_seconds * cfg.frame_rate() as f64) as usize);
            let out = convert(
                &model,
                &reference,
                &source(seconds),
                &ConvertOptions::default(),
            )
            .unwrap();

            let total = (CONTENT_SR as f64 * seconds) as usize * cfg.sample_rate as usize
                / CONTENT_SR as usize
                / cfg.hop_length;
            assert_eq!(out.len(), total * cfg.hop_length, "{reference_seconds} s");

            // A constant through an equal-power crossfade is the same constant.
            let worst = out.iter().fold(0.0f32, |w, s| w.max((s - 1.0).abs()));
            assert!(worst < 1e-5, "the seams dip by {worst}");

            // Every chunk hands `OVERLAP_FRAMES` back to the next, so the frames
            // generated exceed the frames kept by exactly that, once per seam.
            let calls = model.calls.borrow();
            let generated: usize = calls.iter().map(|(_, f)| f).sum();
            assert_eq!(generated - OVERLAP_FRAMES * (calls.len() - 1), total);
            assert!(
                calls.iter().all(|(_, f)| *f >= MIN_CHUNK_FRAMES),
                "a chunk came out under the floor: {calls:?}"
            );
        }
    }

    /// A short source is one chunk and no crossfade at all, which is the case
    /// nearly every real conversion takes.
    #[test]
    fn a_short_source_is_one_chunk() {
        let model = Recorder::new();
        let out = convert(
            &model,
            &reference(430),
            &source(7.8),
            &ConvertOptions::default(),
        )
        .unwrap();
        assert_eq!(model.calls.borrow().len(), 1);
        assert_eq!(out.len(), model.calls.borrow()[0].1 * 256);
    }

    /// The same seed has to give the same audio, since that is the only way two
    /// backends — or two versions of this loop — can be held against each other.
    #[test]
    fn the_noise_is_a_function_of_the_seed() {
        let draw = |seed| {
            let mut rng = Rng::new(seed);
            (0..64).map(|_| rng.next_normal()).collect::<Vec<_>>()
        };
        assert_eq!(draw(7), draw(7));
        assert_ne!(draw(7), draw(8));
        assert!(draw(0).iter().all(|v| v.is_finite()));

        // 0 and 1 specifically: `seed | 1` mapped both to the same state, and
        // 7/8 above did not catch it because only an even seed collides with its
        // successor. Every adjacent pair has to differ, since incrementing the
        // seed is what a user does when a take comes out wrong.
        for seed in 0..8 {
            assert_ne!(draw(seed), draw(seed + 1), "seeds {seed} and {}", seed + 1);
        }

        // Standard normal, loosely: the check is that it is not uniform and not
        // scaled, since either would still integrate to *something*.
        let sample = draw(1);
        let mean = sample.iter().sum::<f32>() / sample.len() as f32;
        let variance = sample.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / sample.len() as f32;
        assert!(mean.abs() < 0.4, "mean {mean}");
        assert!((0.5..2.0).contains(&variance), "variance {variance}");
    }

    /// A reference that fills the shared window has to say so in terms of the
    /// window, because trimming it is the only thing the user can do about it —
    /// and the subtraction underflows rather than erroring if nobody checks.
    #[test]
    fn a_reference_that_fills_the_window_names_the_arithmetic() {
        let model = Recorder::new();
        let err = convert(
            &model,
            &reference(2584),
            &source(5.0),
            &ConvertOptions::default(),
        )
        .expect_err("2584 frames of reference is past the 2580-frame window");
        let err = err.to_string();
        assert!(err.contains("2580"), "{err}");
        assert!(err.contains("trim the reference"), "{err}");
        assert!(model.calls.borrow().is_empty(), "it ran anyway");
    }

    /// `length_adjust` moves the source-samples-per-frame ratio, so at a small
    /// enough value a chunk's audio outgrows Whisper's window before the
    /// transformer's context is anywhere near full. The model would refuse it;
    /// this refuses it first, in terms of the flag that caused it.
    #[test]
    fn a_compressing_length_adjust_is_bounded_by_the_content_window() {
        let model = Recorder::new();
        let opts = ConvertOptions {
            length_adjust: 0.01,
            ..ConvertOptions::default()
        };
        let err = convert(&model, &reference(430), &source(600.0), &opts)
            .expect_err("100x compression puts an hour of source in one chunk")
            .to_string();
        assert!(err.contains("content encoder"), "{err}");

        // Milder compression stays inside it, and still lands on the frame count
        // the adjustment asks for.
        let opts = ConvertOptions {
            length_adjust: 0.5,
            ..ConvertOptions::default()
        };
        let out = convert(&model, &reference(430), &source(600.0), &opts).unwrap();
        let natural = 600 * 22_050 / 256;
        assert_eq!(out.len(), (natural / 2) * 256);
    }
}
