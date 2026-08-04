//! The streaming filter: source PCM in, converted PCM out.
//!
//! [`Converter`] is [`crate::convert`]'s loop turned inside out — the same
//! chunks on the same frame grid, the same seeded noise, the same equal-power
//! crossfade — driven by audio arriving rather than by a file whose length is
//! already known. Only the source streams: **the reference is analysed once, at
//! construction**, because a reference is a whole speaker specification and
//! costs a Whisper encode, a CAMPPlus pass and a mel, none of which depend on
//! the source.
//!
//! # What the latency actually is
//!
//! **Block latency, not sample latency.** Nothing can be emitted until a whole
//! chunk exists, because every stage the audio passes through is a window rather
//! than a filter: the content encoder reads a window, the length regulator
//! resamples one onto the mel grid, the flow-matching sampler integrates one, and
//! the vocoder renders one. The first output sample therefore lags the first
//! input sample by
//!
//! ```text
//! (block + crossfade) / 86.13 s   of buffering
//!   + one chunk of model time
//! ```
//!
//! and after that each chunk arrives one `block` of buffering plus one chunk of
//! model time behind the last. At [`StreamParams::realtime`]'s 172 + 16 frames
//! the buffering term is **2.2 s**. The model term is the one that decides
//! whether a stream keeps up, and it is the larger of the two.
//!
//! ## Model time barely falls when the block does
//!
//! Two of the three costs per chunk are nearly independent of how much *new*
//! audio the chunk carries, which is why shrinking the block is not the lever it
//! looks like:
//!
//! - the content encoder **zero-pads every input to a full 30 s window**
//!   (`burn_seedvc::content`, because Whisper's positional table is 1500 frames
//!   wide and was trained that way), so a 2 s chunk costs what a 25 s one does;
//! - the sampler integrates the reference prefix *and* the new frames together,
//!   every chunk, `--steps` times over. At the 25 s cap [`crate::reference`]
//!   applies that prefix is 2153 of the ~2341 frames in a realtime-preset
//!   window — so ~92% of the transformer's work in such a chunk is
//!   re-integrating the prompt.
//!
//! Only the vocoder scales with the block. So halving the block halves the
//! buffering term, leaves the model term roughly where it was, and close to
//! doubles the total work over a whole stream.
//!
//! ## Measured
//!
//! On the maintainer's RTX 2060 with `--backend tch --device gpu` at the default
//! 30 Euler steps, the batch path converted 7.79 s of source in **10.9 s** of
//! model time (unit 4's end-to-end run). **The model alone is ~1.4× slower than
//! realtime on that hardware**, before any per-chunk overhead, and chunking
//! multiplies the fixed costs above rather than amortising them.
//! [`StreamParams::realtime`] is therefore named for being the lowest-latency
//! preset and not for a throughput promise it cannot keep here. What hardware
//! would keep up is unmeasured; the guess is that it takes both a much faster
//! device and a lower step count.
//!
//! # The reference and the block compete for one window
//!
//! The reference's mel and the source's chunk share the same 2580-frame context,
//! so `block + crossfade` is clamped at construction to what the reference
//! leaves: a 25 s reference leaves 427 frames — under 5 s — where a 5 s one
//! leaves nearly 25 s. [`StreamParams::realtime`]'s 188 frames fit under any
//! reference the 25 s cap admits; [`StreamParams::batch`] asks for the whole
//! window and takes whatever is left. A reference that leaves too little is
//! refused by [`Converter::new`] in the same words the batch path uses, because
//! both call one implementation of that check.
//!
//! # Where a streamed conversion differs from a batch one
//!
//! Same model, same crossfade, same grid — but the cuts land differently, and
//! two of the reasons are worth knowing when the two are compared:
//!
//! - **The frame grid is nominal here.** The batch path knows the source's
//!   length, so it spreads its frames across exactly that many samples; a stream
//!   uses the preset's own 185.76 samples per frame. The two drift by well under
//!   a frame per chunk (≈7 ms over 30 s) and neither is wrong, but it means the
//!   two paths cut in *almost* the same places rather than the same ones.
//! - **The last chunk is not balanced**, and a stream can end up to `crossfade`
//!   frames long. The batch path hands frames back from the penultimate chunk to
//!   keep the last one usable; a stream has already emitted that audio. If the
//!   input stops inside the region the last chunk generated, [`Converter::flush`]
//!   releases that held tail rather than re-generating a fragment — up to 0.19 s
//!   of audio past where the source ended.
//!
//! So the honest comparison between the two is a log-spectrogram or
//! energy-envelope correlation, never a sample-wise difference.

use audio_kit::Samples;
use burn_seedvc::content::{CONTENT_STRIDE, WINDOW_SAMPLES};
use futures::{Stream, StreamExt};
use tokio::sync::mpsc;

use crate::convert::{CONTEXT_SECONDS, ConvertOptions, OVERLAP_FRAMES, Rng, crossfade, room};
use crate::error::{Error, Result};
use crate::model::{CONTENT_SR, Model, Reference};

/// Upstream's `max_context_window` at this preset — 30 s at 86.13 Hz, the same
/// 2580 [`crate::convert`] derives from the config.
///
/// Written down rather than derived because a preset is an associated function
/// with no model in hand. [`Converter::new`] computes the real number from the
/// config it is handed and clamps to it, so this going stale would cost a clamp
/// rather than a wrong window.
const CONTEXT_FRAMES: usize = 2580;

/// Chunk geometry, in **mel frames** at the preset's 86.13 Hz.
///
/// `rvc-core`'s equivalent counts 16 kHz input samples and this one deliberately
/// does not: every window Seed-VC has is measured in frames — the 2580 the
/// reference and the source share, the transformer's `block_size`,
/// [`OVERLAP_FRAMES`] — and the map from a frame to a count of source samples
/// moves with `length_adjust`. A block in samples would change size with a flag
/// that has nothing to do with latency, and would need rounding back onto the
/// frame grid at every use.
#[derive(Debug, Clone, Copy)]
pub struct StreamParams {
    /// New frames each chunk contributes to the output — the advance per model
    /// call, and the block the latency above is measured in.
    pub block: usize,
    /// Frames each chunk re-generates over the previous one and blends across.
    ///
    /// There is no separate `context` field as `rvc-core` has, because here the
    /// two are one region: a chunk's leading `crossfade` frames are generated
    /// from audio the previous chunk already saw, and they are blended into it
    /// rather than dropped. Widening this gives the content encoder more left
    /// context at a seam *and* a longer blend, and re-generates more per chunk.
    pub crossfade: usize,
}

impl StreamParams {
    /// The lowest-latency preset — 172-frame blocks, so 2.0 s of new audio per
    /// chunk and 2.2 s of buffering with the crossfade on top.
    ///
    /// **The name is about latency and not about throughput.** On the hardware
    /// this was developed against the model does not run at realtime speed at
    /// all (see the module docs), so a live pipe falls behind; what the preset
    /// buys is that the audio which *does* come out comes out in 2 s steps
    /// rather than after the whole recording. Much below 2 s the buffering term
    /// stops dominating and the per-chunk fixed costs — a padded 30 s Whisper
    /// window, and the reference prefix through the sampler — make smaller
    /// blocks cost more in total for very little more responsiveness.
    pub fn realtime() -> Self {
        Self {
            block: 172,
            crossfade: OVERLAP_FRAMES,
        }
    }

    /// One chunk as wide as the shared window admits, which is what the batch
    /// path uses.
    ///
    /// [`Converter::new`] clamps this to the room the reference leaves, so the
    /// chunks land where [`crate::convert`] would put them for the same source:
    /// the highest-quality, highest-latency end of the same mechanism, for a
    /// caller that wants file behaviour out of a stream.
    pub fn batch() -> Self {
        Self {
            block: CONTEXT_FRAMES,
            crossfade: OVERLAP_FRAMES,
        }
    }
}

/// A loaded model, one analysed reference, and the state between chunks.
///
/// One of these converts one source: [`reset`](Self::reset) is what starts the
/// next, and it re-seeds the noise so the second input is as reproducible as the
/// first. Like `rvc-core`'s it is meant to live on a single blocking worker,
/// which is what [`convert_stream`] gives it.
pub struct Converter {
    model: Box<dyn Model>,
    reference: Reference,
    opts: ConvertOptions,
    /// Frames per model call: the block plus the crossfade handed to the next.
    chunk: usize,
    /// Frames of that which are new — the advance, and what is emitted.
    block: usize,
    /// Source samples behind one generated frame, `length_adjust` included.
    /// Every cut in the waveform is `round(frame * per_frame)`, which is the
    /// grid [`crate::convert`] cuts on too.
    per_frame: f64,
    /// Source samples not yet converted. Starts at frame `done`, so its leading
    /// `chunk - block` frames were generated by the previous call and are what
    /// the next chunk's head will be blended over.
    pending: Vec<f32>,
    /// Frames converted and advanced past.
    done: usize,
    /// Output samples withheld from the last chunk, awaiting that blend.
    tail: Vec<f32>,
    rng: Rng,
}

impl Converter {
    /// Build a converter around a loaded model and an analysed reference.
    ///
    /// Fallible, and deliberately so: the reference's share of the shared window
    /// decides how much source a chunk can hold, so a reference that leaves too
    /// little is a construction-time error rather than one the user meets in the
    /// middle of a pipe.
    pub fn new(
        model: Box<dyn Model>,
        reference: Reference,
        params: StreamParams,
        opts: ConvertOptions,
    ) -> Result<Self> {
        let cfg = model.config();
        if opts.length_adjust <= 0.0 || !opts.length_adjust.is_finite() {
            return Err(Error::Input(format!(
                "the length adjustment is {}; it scales the output's duration against the \
                 source's, so it has to be a positive number (1.0 keeps the source's)",
                opts.length_adjust,
            )));
        }
        // Source samples behind one output frame. The batch path reads this
        // ratio off the source's own length, which a stream does not have — the
        // two agree to well under a frame, and the module docs say where that
        // leaves a comparison between them.
        let per_frame = CONTENT_SR as f64 * cfg.hop_length as f64
            / (cfg.sample_rate as f64 * opts.length_adjust);

        let room = room(cfg, &reference)?;
        // A chunk is bounded twice, exactly as in the batch path: by the room
        // left in the transformer's context, and by the content encoder's own
        // 30 s window, which binds only when `length_adjust` compresses time.
        // One frame of slack, because the cuts round.
        let by_audio = (WINDOW_SAMPLES as f64 / per_frame) as usize;
        let chunk = params
            .block
            .saturating_add(params.crossfade)
            .min(room)
            .min(by_audio.saturating_sub(1));
        let block = chunk.saturating_sub(params.crossfade);
        if block == 0 {
            return Err(Error::Input(format!(
                "a {}-frame crossfade leaves the chunk no new audio to carry: the reference \
                 leaves {room} frames of the transformer's window, and a length adjustment of {} \
                 puts {} frames behind the content encoder's {CONTEXT_SECONDS} s — shorten the \
                 crossfade, trim the reference, or convert at a milder adjustment",
                params.crossfade,
                opts.length_adjust,
                by_audio.saturating_sub(1),
            )));
        }

        Ok(Self {
            model,
            reference,
            chunk,
            block,
            per_frame,
            pending: Vec::new(),
            done: 0,
            tail: Vec::new(),
            rng: Rng::new(opts.seed),
            opts,
        })
    }

    /// The rate the converted audio comes out at — the vocoder's 22.05 kHz, not
    /// the 16 kHz the source goes in at.
    pub fn output_sr(&self) -> u32 {
        self.model.config().sample_rate
    }

    /// Clear every trace of the current input, so the same loaded model and the
    /// same reference can convert an independent one next.
    ///
    /// The noise is re-seeded as well as the buffers cleared, because the batch
    /// path draws from a fresh generator per file: without this the second file
    /// of a run would not be reproducible on its own.
    pub fn reset(&mut self) {
        self.pending.clear();
        self.tail.clear();
        self.done = 0;
        self.rng = Rng::new(self.opts.seed);
    }

    /// Feed source samples — mono `f32` at [`CONTENT_SR`] — and take whatever
    /// chunks are now complete.
    ///
    /// Usually none. A chunk needs `block + crossfade` frames of source before
    /// the model sees any of it, so a caller pushing 100 ms at a time gets
    /// nothing back for the first twenty or so pushes and then a whole block.
    pub fn push(&mut self, input: &[f32]) -> Result<Vec<Samples>> {
        self.pending.extend_from_slice(input);
        let mut outs = Vec::new();
        loop {
            let start = self.at(self.done);
            let width = self.at(self.done + self.chunk) - start;
            if self.pending.len() < width {
                return Ok(outs);
            }

            let mut wave = self.generate(width, self.chunk)?;
            // Withhold the frames the next chunk will re-generate, so the two
            // are blended over the same output-time region rather than merely
            // being adjacent — the join adds audio, it never deletes any.
            let keep = wave.len() - (self.chunk - self.block) * self.model.config().hop_length;
            self.tail = wave.split_off(keep);
            if !wave.is_empty() {
                outs.push(wave);
            }

            self.pending
                .drain(..self.at(self.done + self.block) - start);
            self.done += self.block;
        }
    }

    /// Convert what is left and release the withheld crossfade tail.
    ///
    /// The remainder is normally shorter than a block, so this is where a stream
    /// gets its one undersized chunk. Audio past the last whole frame is dropped,
    /// and so is an entire stream under one content frame (20 ms) — there is
    /// nothing for the content encoder to read in either case.
    pub fn flush(&mut self) -> Result<Vec<Samples>> {
        let mut outs = Vec::new();
        let start = self.at(self.done);
        // The inverse of `at`, so the last cut lands on the same grid as every
        // other one instead of on a second, slightly different, rounding.
        let end = (((start + self.pending.len()) as f64 / self.per_frame).round() as usize)
            .max(self.done);
        let frames = end - self.done;
        let width = (self.at(end) - start).min(self.pending.len());

        // Only worth generating if it reaches past what the last chunk already
        // covered; anything shorter is already in `tail`.
        if frames > self.chunk - self.block && width >= CONTENT_STRIDE {
            let wave = self.generate(width, frames)?;
            self.tail.clear();
            outs.push(wave);
        } else if !self.tail.is_empty() {
            outs.push(std::mem::take(&mut self.tail));
        }

        self.pending.clear();
        self.done = end;
        Ok(outs)
    }

    /// Convert a whole 16 kHz buffer to one vector at
    /// [`output_sr`](Self::output_sr).
    ///
    /// The streaming path driven to completion in one call, which is what makes
    /// it directly comparable against [`crate::convert`] on the same input.
    pub fn convert_all(&mut self, source: &[f32]) -> Result<Vec<f32>> {
        let mut out = Vec::new();
        for chunk in self.push(source)? {
            out.extend(chunk);
        }
        for chunk in self.flush()? {
            out.extend(chunk);
        }
        Ok(out)
    }

    /// Where a frame boundary falls in the source waveform — the batch path's
    /// `at`, and the reason a push boundary never becomes a cut.
    fn at(&self, frame: usize) -> usize {
        (frame as f64 * self.per_frame).round() as usize
    }

    /// Convert the head of `pending` into `frames` of audio, blended over the
    /// tail the last chunk withheld.
    fn generate(&mut self, samples: usize, frames: usize) -> Result<Samples> {
        let width = self.model.config().n_mels * (self.reference.frames + frames);
        // Drawn for the prompt region too, whose values the sampler overwrites:
        // its *width* is part of the window the solver integrates.
        let noise: Vec<f32> = (0..width).map(|_| self.rng.next_normal()).collect();
        let mut wave = self.model.convert(
            &self.pending[..samples],
            frames,
            &self.reference,
            &noise,
            self.opts.sampler,
        )?;
        crossfade(&mut wave, &self.tail);
        Ok(wave)
    }
}

/// Drive a [`Converter`] over an async input stream, yielding an async output
/// stream.
///
/// The model runs on a dedicated blocking worker, because a chunk is seconds of
/// uninterrupted tensor work and running that on a runtime thread would stall
/// every other task in the process; back-pressure flows through bounded
/// channels. The same arrangement `rvc-core` uses, for the same reason.
pub fn convert_stream<S>(
    mut converter: Converter,
    mut input: S,
) -> impl Stream<Item = Result<Samples>>
where
    S: Stream<Item = std::result::Result<Samples, audio_kit::AudioError>> + Unpin + Send + 'static,
{
    let (in_tx, mut in_rx) = mpsc::channel::<Samples>(32);
    let (out_tx, mut out_rx) = mpsc::channel::<Result<Samples>>(32);

    // Forward the async input stream into the worker's input channel, surfacing
    // decode errors on the output side.
    let err_tx = out_tx.clone();
    tokio::spawn(async move {
        while let Some(item) = input.next().await {
            match item {
                Ok(s) => {
                    if in_tx.send(s).await.is_err() {
                        break;
                    }
                }
                Err(e) => {
                    let _ = err_tx.send(Err(e.into())).await;
                    break;
                }
            }
        }
        // Dropping `in_tx` is what tells the worker to flush.
    });

    let worker_out = out_tx;
    tokio::task::spawn_blocking(move || {
        while let Some(samples) = in_rx.blocking_recv() {
            match converter.push(&samples) {
                Ok(chunks) => {
                    for c in chunks {
                        if worker_out.blocking_send(Ok(c)).is_err() {
                            return;
                        }
                    }
                }
                Err(e) => {
                    let _ = worker_out.blocking_send(Err(e));
                    return;
                }
            }
        }
        match converter.flush() {
            Ok(chunks) => {
                for c in chunks {
                    if worker_out.blocking_send(Ok(c)).is_err() {
                        return;
                    }
                }
            }
            Err(e) => {
                let _ = worker_out.blocking_send(Err(e));
            }
        }
    });

    async_stream::stream! {
        while let Some(item) = out_rx.recv().await {
            yield item;
        }
    }
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;

    use burn_seedvc::SeedVcConfig;
    use burn_seedvc::flow::Sampler;

    use super::*;
    use crate::convert::convert;

    /// A [`Model`] that renders each chunk as its own mean, and records how it
    /// was asked to.
    ///
    /// The chunk arithmetic is what these tests are about and it needs no
    /// weights. The mean is what makes the *cuts* observable: an output that
    /// depends on the chunk's content moves the moment a boundary does, which a
    /// constant would hide — while a constant source still comes back constant,
    /// so the crossfade's equal-power property stays checkable too.
    struct Mean {
        cfg: SeedVcConfig,
        calls: RefCell<Vec<(usize, usize)>>,
    }

    impl Mean {
        fn new() -> Box<Self> {
            Box::new(Self {
                cfg: SeedVcConfig::uvit_whisper_small_wavenet(),
                calls: RefCell::new(Vec::new()),
            })
        }
    }

    impl Model for Mean {
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
            assert!(
                source.len() >= CONTENT_STRIDE,
                "{} samples is under one content frame",
                source.len()
            );
            assert_eq!(noise.len(), self.cfg.n_mels * (reference.frames + frames));
            self.calls.borrow_mut().push((source.len(), frames));
            let mean = source.iter().sum::<f32>() / source.len() as f32;
            Ok(vec![mean; frames * self.cfg.hop_length])
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

    fn converter(reference_frames: usize, params: StreamParams) -> Converter {
        Converter::new(
            Mean::new(),
            reference(reference_frames),
            params,
            ConvertOptions::default(),
        )
        .unwrap()
    }

    /// Seconds of 16 kHz source, rising slowly, so a chunk's mean says where its
    /// boundaries were.
    fn ramp(seconds: f64) -> Vec<f32> {
        let n = (CONTENT_SR as f64 * seconds) as usize;
        (0..n).map(|i| i as f32 / n as f32).collect()
    }

    /// Frames a source of this length converts to, which is what both paths owe
    /// the caller.
    fn frames(source: &[f32]) -> usize {
        let cfg = SeedVcConfig::uvit_whisper_small_wavenet();
        source.len() * cfg.sample_rate as usize / CONTENT_SR as usize / cfg.hop_length
    }

    /// Pearson correlation over the shorter of the two, which is how outputs
    /// from two independent generations are compared here and end to end — a
    /// sample-wise difference is phase-sensitive and says nothing.
    fn correlation(a: &[f32], b: &[f32]) -> f32 {
        let n = a.len().min(b.len());
        let (a, b) = (&a[..n], &b[..n]);
        let mean = |v: &[f32]| v.iter().sum::<f32>() / n as f32;
        let (ma, mb) = (mean(a), mean(b));
        let cov: f32 = a.iter().zip(b).map(|(x, y)| (x - ma) * (y - mb)).sum();
        let va: f32 = a.iter().map(|x| (x - ma).powi(2)).sum();
        let vb: f32 = b.iter().map(|y| (y - mb).powi(2)).sum();
        cov / (va * vb).sqrt()
    }

    /// The chunks have to reassemble into one signal of the source's duration,
    /// whatever the reference's length does to the chunk size — and the seams
    /// must not dip, which is the whole reason the crossfade is equal-power.
    #[test]
    fn the_chunks_reassemble_into_one_signal() {
        let cfg = SeedVcConfig::uvit_whisper_small_wavenet();
        // 25 s of reference leaves 427 frames, under 5 s, and the realtime
        // preset's 188-frame chunk still fits under it; 5 s of reference leaves
        // nearly the whole window, which is what `batch()` then takes.
        for (reference_frames, params) in [
            (2153, StreamParams::realtime()),
            (430, StreamParams::realtime()),
            (430, StreamParams::batch()),
        ] {
            let mut converter = converter(reference_frames, params);
            let source = vec![1.0f32; CONTENT_SR as usize * 40];
            let out = converter.convert_all(&source).unwrap();

            let expected = frames(&source) * cfg.hop_length;
            let drift = (out.len() as isize - expected as isize).abs();
            assert!(
                drift <= (OVERLAP_FRAMES * cfg.hop_length) as isize,
                "{} samples against the source's {expected}",
                out.len(),
            );
            // A constant through an equal-power crossfade is the same constant.
            let worst = out.iter().fold(0.0f32, |w, s| w.max((s - 1.0).abs()));
            assert!(worst < 1e-5, "the seams dip by {worst}");
        }
    }

    /// How the audio arrives must not change what comes out. This is the one
    /// property a streaming filter owes above every other, and the cheapest way
    /// to break it is to cut on a buffer boundary instead of on the frame grid.
    #[test]
    fn the_arrival_pattern_does_not_move_the_cuts() {
        let source = ramp(20.0);
        let expected = converter(430, StreamParams::realtime())
            .convert_all(&source)
            .unwrap();

        for size in [1, 999, 16_000, 65_536] {
            let mut converter = converter(430, StreamParams::realtime());
            let mut out = Vec::new();
            for piece in source.chunks(size) {
                for chunk in converter.push(piece).unwrap() {
                    out.extend(chunk);
                }
            }
            for chunk in converter.flush().unwrap() {
                out.extend(chunk);
            }
            assert_eq!(out, expected, "pushed {size} samples at a time");
        }
    }

    /// Streaming and batch are one loop driven two ways, so with the same
    /// geometry they have to produce the same signal. Not the same *samples*:
    /// the batch path fits its frame grid to a length a stream does not know,
    /// and it balances its last chunk, so the two are compared the way two
    /// generations always are here.
    #[test]
    fn streaming_matches_the_batch_path() {
        let cfg = SeedVcConfig::uvit_whisper_small_wavenet();
        let source = ramp(30.0);
        let reference_frames = 2153;
        // What `convert` picks for its own window is everything the reference
        // leaves; `block` is the *new* audio in a chunk, so the crossfade comes
        // off it.
        let room = CONTEXT_FRAMES - reference_frames;
        let params = StreamParams {
            block: room - OVERLAP_FRAMES,
            crossfade: OVERLAP_FRAMES,
        };

        let streamed = converter(reference_frames, params)
            .convert_all(&source)
            .unwrap();
        let batched = convert(
            &*Mean::new(),
            &reference(reference_frames),
            &source,
            &ConvertOptions::default(),
        )
        .unwrap();

        assert!(
            (streamed.len() as isize - batched.len() as isize).abs() <= cfg.hop_length as isize,
            "{} samples streamed against {} batched",
            streamed.len(),
            batched.len(),
        );
        let r = correlation(&streamed, &batched);
        assert!(r > 0.999, "the two paths correlate at only {r}");
    }

    /// A reference that fills the shared window has to be refused before any
    /// audio is pushed, in the same words the batch path uses — both call one
    /// implementation of that check, and this is what pins it.
    #[test]
    fn a_reference_that_fills_the_window_is_refused_at_construction() {
        let err = Converter::new(
            Mean::new(),
            reference(2584),
            StreamParams::realtime(),
            ConvertOptions::default(),
        )
        .err()
        .expect("2584 frames of reference is past the 2580-frame window")
        .to_string();
        assert!(err.contains("2580"), "{err}");
        assert!(err.contains("trim the reference"), "{err}");
    }

    /// A crossfade wider than the room the reference left would advance zero
    /// frames per chunk — an output that never grows out of a loop that never
    /// ends. It is refused, and the message names every way out.
    #[test]
    fn a_crossfade_wider_than_the_chunk_is_refused() {
        let err = Converter::new(
            Mean::new(),
            reference(2153),
            StreamParams {
                block: 100,
                crossfade: 500,
            },
            ConvertOptions::default(),
        )
        .err()
        .expect("a 500-frame crossfade cannot fit a 427-frame window")
        .to_string();
        assert!(err.contains("no new audio"), "{err}");
        assert!(err.contains("shorten the crossfade"), "{err}");
    }

    /// `length_adjust` moves the frame grid, so the output's duration moves with
    /// it while every cut stays on a whole frame.
    #[test]
    fn a_length_adjustment_scales_the_output() {
        let cfg = SeedVcConfig::uvit_whisper_small_wavenet();
        let source = ramp(20.0);
        for adjust in [0.5, 2.0] {
            let mut converter = Converter::new(
                Mean::new(),
                reference(430),
                StreamParams::realtime(),
                ConvertOptions {
                    length_adjust: adjust,
                    ..ConvertOptions::default()
                },
            )
            .unwrap();
            let out = converter.convert_all(&source).unwrap();
            let expected = (frames(&source) as f64 * adjust) as usize * cfg.hop_length;
            let drift = (out.len() as isize - expected as isize).abs();
            assert!(
                drift <= (OVERLAP_FRAMES * cfg.hop_length) as isize,
                "{} samples at {adjust}× against {expected}",
                out.len(),
            );
        }
    }

    /// A stream shorter than one chunk is still one chunk, which is the case a
    /// short utterance down a live pipe takes.
    #[test]
    fn a_short_stream_is_one_chunk() {
        let mut converter = converter(430, StreamParams::realtime());
        assert!(converter.push(&ramp(1.0)).unwrap().is_empty());
        let out = converter.flush().unwrap();
        assert_eq!(out.len(), 1);
        assert!(!out[0].is_empty());
    }
}
