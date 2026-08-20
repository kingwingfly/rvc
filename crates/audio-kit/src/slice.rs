//! Sentence-safe RMS slicer.
//!
//! Splits a mono `f32` recording into voiced `[start, end)` sample ranges by
//! removing *between-sentence dead-air only*. This is a clean-room RMS slicer
//! written for quiet, breathy corpora, where soft content is low-energy but
//! wanted — so energy is used **solely** to locate long silent gaps between
//! sentences, never to gate quiet-but-present sound.
//!
//! The core invariant is hysteresis on the silence gap: two voiced runs are a
//! single sentence unless the silence between them is *both* below the energy
//! floor *and* longer than [`SliceOptions::min_silence`]. Short internal pauses
//! and soft tails therefore stay inside the clip, and every kept segment is
//! edge-padded by up to [`SliceOptions::pad`] seconds of the bordering quiet so
//! onsets and breathy tails are not clipped.
//!
//! Two entry points, one algorithm: [`slice`] takes a whole recording, and
//! [`Slicer`] takes it a chunk at a time and hands back each clip as soon as it
//! provably cannot change again. They share every step past run detection, so a
//! streamed pass and a batch one cut in exactly the same places.
//!
//! The floor itself can be measured instead of guessed: [`noise_floor`] reads a
//! recording's own quiet and loud percentiles off that same grid, and
//! [`SliceOptions::with_measured_floor`] turns the reading into a `silence_db`.

/// Tuning knobs for [`slice`]. All durations are in seconds.
#[derive(Debug, Clone, Copy)]
pub struct SliceOptions {
    /// Energy floor in dBFS. A frame quieter than this counts as silence.
    /// Lower it (e.g. `-50`) to keep the very softest passages, or let the
    /// recording decide with [`SliceOptions::with_measured_floor`].
    pub silence_db: f32,
    /// Minimum length of a silent gap (seconds) for it to be a cut point.
    /// Shorter gaps are treated as internal pauses and kept inside the clip, so
    /// a complete sentence is never split.
    pub min_silence: f32,
    /// Drop any kept segment shorter than this (seconds).
    ///
    /// A corpus slicer has a floor under this that it cannot see from here:
    /// `rvc-train` draws 0.48 s windows (48 frames on its 100 Hz grid), and a
    /// clip shorter than one window is decoded, feature-extracted and *then*
    /// discarded with a warning. Below 0.48 s this knob therefore buys nothing
    /// and costs the analysis of every fragment it lets through. The number is
    /// deliberately not a constant here — a `*-kit` crate that knew a trainer's
    /// window would be depending on an engine — so it is stated, not enforced.
    pub min_clip: f32,
    /// Hard cap on segment length (seconds); `0` means never split. Longer runs
    /// are split at their quietest interior frame until every piece fits.
    pub max_clip: f32,
    /// Edge-pad each kept segment by up to this many seconds of bordering quiet.
    pub pad: f32,
}

impl Default for SliceOptions {
    fn default() -> Self {
        Self {
            // Deliberately a fixed number and not a measured one. Every corpus
            // prepared so far was cut at -40 dBFS, and deriving the floor by
            // default would silently re-cut all of them — the same reason the
            // mel front end's clip floor has not moved. A caller that wants the
            // measurement asks for it, and the change gets re-baselined.
            silence_db: -40.0,
            min_silence: 0.3,
            min_clip: 1.0,
            max_clip: 0.0,
            pad: 0.15,
        }
    }
}

/// Frames quieter than this are digital silence rather than a noise floor, and
/// the ratio against them is unbounded — 1e-6 full scale, which is what
/// `rvc-train`'s per-clip SNR clamped its linear floor at before this function
/// existed.
const SILENT_FLOOR_DB: f32 = -120.0;

/// A recording needs at least this many frames (~80 ms) before its percentiles
/// mean anything.
const MIN_MEASURED_FRAMES: usize = 8;

/// How far above the measured floor a derived `silence_db` sits, when the
/// recording has room for it.
///
/// Room tone is a distribution, not a level: [`NoiseFloor::floor_db`] is the
/// middle of the dead-air frames, so a threshold sitting *on* it would call
/// half the dead air voiced and the dead air would survive into the corpus.
/// 8 dB clears the spread without approaching speech on any recording with a
/// normal dynamic range.
const FLOOR_MARGIN_DB: f32 = 8.0;

/// ...but never more than this far from the floor towards the speech.
///
/// The margin above cannot be the whole story, because a close-mic breathy take
/// has almost no range to spend it in: on this repository's own corpus,
/// rebuilt into a raw recording (ten sentences with a second of the corpus's
/// own room tone between them), the floor measures -51.4 dBFS and the speech
/// -41.6 — a **9.8 dB** gap, where a flat 8 dB margin lands 1.8 dB under the
/// voice. Taking a fraction of the measured gap instead makes the threshold
/// structurally unable to reach the speech, which is the guard, and the two
/// failures are then bounded from the same measurement rather than by two
/// unrelated constants.
///
/// 0.4 is where the sweep put it. Every value from 0.3 to 0.6 recovers all ten
/// sentences on that recording — against **five** for the fixed -40 dBFS — but
/// they differ in what they keep of the soft tails: 28.5 s at 0.3, 27.9 s at
/// 0.4, 27.0 s at 0.5, 25.9 s at 0.6, out of a 29.7 s ceiling (speech plus the
/// full 0.15 s pad each side). The fixed floor keeps 8.2 s. 0.4 keeps 94% of
/// that ceiling while still sitting 3.9 dB clear of the tone.
const MARGIN_FRACTION: f32 = 0.4;

impl SliceOptions {
    /// Replace `silence_db` with a floor measured from `samples`, leaving it
    /// untouched when there is nothing to measure ([`noise_floor`] returns
    /// `None`).
    ///
    /// **This is a whole-signal statistic, which is why it is the caller's step
    /// and not the slicer's.** [`Slicer`] never sees the whole signal — it
    /// exists precisely so a ten-minute file does not have to be buffered — so
    /// measuring inside it would give the streaming path a floor that moved as
    /// audio arrived, and the two paths would cut in different places. Instead
    /// the measurement happens once, before slicing, and hands both paths the
    /// same concrete number; `streaming_matches_batch` is untouched by it. The
    /// honest cost is that a true stdin filter has no whole signal to measure,
    /// so this is reachable only where the audio is already in memory.
    #[must_use]
    pub fn with_measured_floor(mut self, samples: &[f32], sample_rate: u32) -> Self {
        if let Some(measured) = noise_floor(samples, sample_rate) {
            self.silence_db = measured.silence_db();
        }
        self
    }
}

/// What a recording's own frames say about its noise floor, read off the grid
/// [`slice`] cuts on.
#[derive(Debug, Clone, Copy)]
pub struct NoiseFloor {
    /// dBFS of the quiet percentile: between-sentence dead air, room tone, and
    /// whatever the preamp contributes.
    pub floor_db: f32,
    /// dBFS of the loud percentile — a representative speech level, not a peak.
    pub signal_db: f32,
}

impl NoiseFloor {
    /// Signal over floor in dB. The floor is clamped at [`SILENT_FLOOR_DB`], so
    /// a digitally-silent recording reports a large ratio rather than an
    /// unbounded one.
    pub fn snr_db(&self) -> f32 {
        self.signal_db - self.floor_db.max(SILENT_FLOOR_DB)
    }

    /// [`NoiseFloor::snr_db`] as a linear amplitude ratio.
    pub fn snr(&self) -> f32 {
        10f32.powf(self.snr_db() / 20.0)
    }

    /// The energy floor to slice this recording at: [`FLOOR_MARGIN_DB`] above
    /// the measured floor, or [`MARGIN_FRACTION`] of the way up to the speech,
    /// whichever is nearer the floor. Never above 0 dBFS, which would call
    /// every sample silence.
    pub fn silence_db(&self) -> f32 {
        let gap = (self.signal_db - self.floor_db).max(0.0);
        (self.floor_db + FLOOR_MARGIN_DB.min(MARGIN_FRACTION * gap)).min(0.0)
    }
}

/// Measure `samples`' noise floor and speech level, or `None` when there is too
/// little to measure (empty input, a zero rate, under [`MIN_MEASURED_FRAMES`]).
///
/// The grid is [`slice`]'s — ~30 ms windows at a ~10 ms hop — and it derives
/// from `sample_rate` alone, so the measurement never depends on the very knob
/// it is there to decide. Percentiles are taken over dBFS directly rather than
/// over RMS, which is the same ordering: `20 * log10` is monotone.
pub fn noise_floor(samples: &[f32], sample_rate: u32) -> Option<NoiseFloor> {
    if sample_rate == 0 {
        return None;
    }
    let (hop, win) = frame_geometry(sample_rate);
    let mut db = frame_db(samples, hop, win);
    if db.len() < MIN_MEASURED_FRAMES {
        return None;
    }
    db.sort_by(f32::total_cmp);
    let pct = |p: f32| db[((db.len() - 1) as f32 * p) as usize];
    Some(NoiseFloor {
        floor_db: pct(0.10),
        signal_db: pct(0.75),
    })
}

/// A contiguous voiced run measured in frame indices, `[start, end)`.
#[derive(Debug, Clone, Copy)]
struct FrameRun {
    start: usize,
    end: usize,
}

/// A window of audio addressed by *absolute* sample index: the sample at index
/// `i` is `samples[i - offset]`. [`slice`] holds the whole recording at offset
/// zero; [`Slicer`] holds only what it can still be asked about.
#[derive(Clone, Copy)]
struct View<'a> {
    samples: &'a [f32],
    offset: usize,
}

impl View<'_> {
    /// One past the last sample index the view covers.
    fn end(&self) -> usize {
        self.offset + self.samples.len()
    }

    fn get(&self, start: usize, end: usize) -> &[f32] {
        &self.samples[start - self.offset..end - self.offset]
    }
}

/// Frame geometry and the sample counts both paths derive from
/// [`SliceOptions`], resolved once so streaming and batch cannot end up
/// disagreeing about what a frame is.
#[derive(Debug, Clone, Copy)]
struct Grid {
    hop: usize,
    win: usize,
    silence_db: f32,
    /// Frames; everything below is samples.
    min_silence: usize,
    pad: usize,
    min_clip: usize,
    /// `0` means never split — both `max_clip <= 0` and a cap that rounds away.
    max_clip: usize,
}

/// The analysis grid as `(hop, win)`: ~10 ms hop, ~30 ms window, at least one
/// sample each. It depends on nothing but the rate, which is what lets
/// [`noise_floor`] measure on the grid the cuts will be made on.
fn frame_geometry(sample_rate: u32) -> (usize, usize) {
    let hop = ((sample_rate as f32 * 0.010).round() as usize).max(1);
    let win = ((sample_rate as f32 * 0.030).round() as usize).max(hop);
    (hop, win)
}

impl Grid {
    fn new(sample_rate: u32, opts: &SliceOptions) -> Self {
        let (hop, win) = frame_geometry(sample_rate);
        let samples = |secs: f32| (secs * sample_rate as f32).round() as usize;
        Self {
            hop,
            win,
            silence_db: opts.silence_db,
            min_silence: seconds_to_frames(opts.min_silence, sample_rate, hop),
            pad: samples(opts.pad),
            min_clip: samples(opts.min_clip),
            max_clip: if opts.max_clip > 0.0 {
                samples(opts.max_clip)
            } else {
                0
            },
        }
    }

    /// Last sample the frames below `end_frame` cover, clamped to `n`: a voiced
    /// frame `f` covers samples `[f * hop, f * hop + win)`.
    fn raw_end(&self, end_frame: usize, n: usize) -> usize {
        (end_frame.saturating_sub(1) * self.hop + self.win).min(n)
    }

    /// Where a run's padded range starts: up to `pad` of the leading quiet,
    /// hard-bounded at the midpoint of the gap to the previous run so two
    /// neighbours can never both claim the same silence and overlap.
    fn padded_start(&self, raw_start: usize, prev_raw_end: Option<usize>) -> usize {
        let lo = match prev_raw_end {
            Some(prev) => midpoint(prev.min(raw_start), raw_start),
            None => 0,
        };
        raw_start.saturating_sub(self.pad).max(lo)
    }

    /// The mirror of [`Grid::padded_start`], additionally clamped to `n` — the
    /// end of the input, or `usize::MAX` while a stream is still open.
    fn padded_end(&self, raw_end: usize, next_raw_start: Option<usize>, n: usize) -> usize {
        let hi = match next_raw_start {
            Some(next) => midpoint(raw_end, next.max(raw_end)),
            None => n,
        };
        (raw_end + self.pad).min(hi).min(n)
    }
}

/// Halfway from `lo` to `hi`, rounding down.
fn midpoint(lo: usize, hi: usize) -> usize {
    lo + hi.saturating_sub(lo) / 2
}

/// The tail of the pipeline both paths share: the non-overlap clamp, the
/// `max_clip` split and the `min_clip` drop. Ranges have to arrive in order,
/// and `prev_end` is the whole of the state that keeps them so.
struct Emit {
    grid: Grid,
    prev_end: usize,
}

impl Emit {
    fn new(grid: Grid) -> Self {
        Self { grid, prev_end: 0 }
    }

    /// Clamp, split and filter one padded range, appending what survives.
    /// `view` has to cover the range — its interior is read only to find a
    /// split point.
    fn push(&mut self, range: (usize, usize), view: View, out: &mut Vec<(usize, usize)>) {
        // Enforce the documented non-overlap invariant. A voiced frame spans
        // `win` (~30 ms) but frames step by `hop` (~10 ms), so with a very
        // small `min_silence` two padded ranges could touch or overlap. Clamp
        // the start past the previous end, dropping a range swallowed whole.
        let (start, end) = (range.0.max(self.prev_end), range.1);
        if start >= end {
            return;
        }
        self.prev_end = end;

        let mut stack = vec![(start, end)];
        while let Some((s, e)) = stack.pop() {
            match self.cut(s, e, view) {
                // Right half first, so the left one is emitted before it.
                Some(cut) => {
                    stack.push((cut, e));
                    stack.push((s, cut));
                }
                None if e - s >= self.grid.min_clip => out.push((s, e)),
                None => {}
            }
        }
    }

    /// Where to split `[start, end)`, or `None` if it already fits inside
    /// `max_clip` — or is too short to split into two `>= min_clip` pieces, in
    /// which case keeping the audio beats manufacturing a fragment that the
    /// `min_clip` filter would then silently discard.
    fn cut(&self, start: usize, end: usize, view: View) -> Option<usize> {
        if self.grid.max_clip == 0 || end - start <= self.grid.max_clip {
            return None;
        }
        // Keep both halves >= min_clip (and away from the exact edges) so each
        // half makes progress.
        let margin = (self.grid.max_clip / 4)
            .max(self.grid.hop)
            .max(self.grid.min_clip);
        let (lo, hi) = (start + margin, end.checked_sub(margin)?);
        (lo < hi).then(|| quietest_cut(view, &self.grid, lo, hi).clamp(lo, hi))
    }
}

/// Slice `samples` into voiced `[start, end)` sample ranges.
///
/// Returns an empty vector for empty or all-silent input. Ranges are sorted,
/// non-overlapping, and clamped to `[0, samples.len())`.
pub fn slice(samples: &[f32], sample_rate: u32, opts: &SliceOptions) -> Vec<(usize, usize)> {
    if samples.is_empty() || sample_rate == 0 {
        return Vec::new();
    }
    let grid = Grid::new(sample_rate, opts);

    let db = frame_db(samples, grid.hop, grid.win);
    if db.is_empty() {
        return Vec::new();
    }

    // Voiced = not silent. Merge voiced runs separated by a gap shorter than
    // `min_silence` (the key sentence-preserving step).
    let runs = merged_voiced_runs(&db, grid.silence_db, grid.min_silence);

    // Frame runs -> padded, clamped, non-overlapping sample ranges.
    let n = samples.len();
    let view = View { samples, offset: 0 };
    let mut emit = Emit::new(grid);
    let mut ranges = Vec::with_capacity(runs.len());
    for (i, run) in runs.iter().enumerate() {
        // The previous run's raw end is deliberately unclamped: `padded_start`
        // bounds it by this run's start, which is already inside the input.
        let prev = (i > 0).then(|| grid.raw_end(runs[i - 1].end, usize::MAX));
        let next = runs.get(i + 1).map(|r| r.start * grid.hop);
        let range = (
            grid.padded_start(run.start * grid.hop, prev),
            grid.padded_end(grid.raw_end(run.end, n), next, n),
        );
        emit.push(range, view, &mut ranges);
    }
    ranges
}

/// A finished clip cut out of a stream: absolute sample offsets into the whole
/// input, plus the audio itself — so the caller can drop what it has fed in.
#[derive(Debug, Clone)]
pub struct Clip {
    pub start: usize,
    pub end: usize,
    pub samples: Vec<f32>,
}

/// [`slice`] over a stream: push audio as it arrives, take back the clips that
/// have closed.
///
/// A run is finalised only once `max(min_silence, 2 * pad)` of silence has
/// followed it — **the point past which no later sample can change its range**,
/// which is what makes these the clips [`slice`] would have cut from the whole
/// recording. Before then, either the silence could still turn out to be an
/// internal pause, so the run would grow, or the next run could start close
/// enough to bound this one's trailing pad at the midpoint between them.
///
/// The one divergence is a run that never falls silent: once it passes
/// `max_clip` it is force-cut at its quietest interior frame, whereas [`slice`]
/// knows the run's full length and halves it instead. Cutting late is the price
/// of not buffering the whole recording, which is the point of the type.
///
/// A measured floor ([`SliceOptions::with_measured_floor`]) is deliberately
/// *not* one of the differences: it is a whole-signal statistic taken by the
/// caller before slicing, so this type receives the same concrete `silence_db`
/// the batch path does and never measures anything of its own. Measuring here
/// — from a bounded prefix, say — would be the one thing that cannot be done
/// silently, because the two paths would then cut in different places.
pub struct Slicer {
    /// Kept only to reproduce [`slice`]'s tolerance of a zero rate, which has
    /// no frame geometry at all.
    sample_rate: u32,
    grid: Grid,
    /// Frames of silence after a run before its range is final.
    settle: usize,
    /// Absolute index of `buf[0]`; audio before it can no longer be reached.
    offset: usize,
    buf: Vec<f32>,
    /// Samples pushed, and frames whose energy is already decided.
    total: usize,
    frames: usize,
    /// Frame the voiced run in progress started at.
    cur: Option<usize>,
    /// The merged run accumulating, and the raw end of the one before it.
    acc: Option<FrameRun>,
    prev_raw_end: Option<usize>,
    emit: Emit,
}

impl Slicer {
    pub fn new(sample_rate: u32, opts: &SliceOptions) -> Self {
        let grid = Grid::new(sample_rate, opts);
        Self {
            sample_rate,
            // A run's trailing pad is bounded by the midpoint of the gap to the
            // next run, so that bound stops binding only once the next run
            // cannot start within `2 * pad` — and `k` frames of silence put it
            // at least `(k + 1) * hop - win` samples away.
            settle: grid
                .min_silence
                .max((2 * grid.pad + grid.win).div_ceil(grid.hop)),
            grid,
            offset: 0,
            buf: Vec::new(),
            total: 0,
            frames: 0,
            cur: None,
            acc: None,
            prev_raw_end: None,
            emit: Emit::new(grid),
        }
    }

    /// Feed the next chunk of audio, taking back any clip it completed.
    pub fn push(&mut self, chunk: &[f32]) -> Vec<Clip> {
        self.buf.extend_from_slice(chunk);
        self.total += chunk.len();
        self.advance(false)
    }

    /// Close the stream, taking back whatever was still open.
    pub fn finish(&mut self) -> Vec<Clip> {
        self.advance(true)
    }

    fn advance(&mut self, eof: bool) -> Vec<Clip> {
        if self.sample_rate == 0 {
            return Vec::new();
        }
        // While the stream is open the eventual length is unknown — but every
        // clamp against it is provably slack by the time a range is finalised,
        // so standing in for it with `usize::MAX` changes no answer.
        let n = if eof { self.total } else { usize::MAX };

        let mut ranges = Vec::new();
        loop {
            let start = self.frames * self.grid.hop;
            // A frame is decided only once its window is full; before that more
            // audio would change its energy. At the end of the input the
            // truncated windows are exactly the ones `frame_db` computes.
            let ready = if eof {
                start < self.total
            } else {
                start + self.grid.win <= self.total
            };
            if !ready {
                break;
            }
            let end = (start + self.grid.win).min(self.total);
            let db = window_db(&self.buf[start - self.offset..end - self.offset]);
            self.frames += 1;
            self.step(self.frames - 1, db >= self.grid.silence_db, n, &mut ranges);
        }
        if eof {
            if let Some(start) = self.cur.take() {
                self.merge(FrameRun {
                    start,
                    end: self.frames,
                });
            }
            if let Some(acc) = self.acc.take() {
                self.finalise(acc, None, n, &mut ranges);
            }
        }

        let clips = ranges
            .into_iter()
            .map(|(start, end)| Clip {
                start,
                end,
                samples: self.buf[start - self.offset..end - self.offset].to_vec(),
            })
            .collect();
        self.trim();
        clips
    }

    /// Fold one frame's verdict into the run in progress.
    fn step(&mut self, f: usize, voiced: bool, n: usize, ranges: &mut Vec<(usize, usize)>) {
        if voiced {
            if self.cur.is_none() {
                self.cur = Some(f);
                // A run starting after `min_silence` of quiet cannot merge into
                // the one before it, so that one's range is final now — and its
                // trailing pad is bounded by this run's start, which is exactly
                // what the batch path reads off the next entry in the list.
                if let Some(acc) = self.acc
                    && f - acc.end >= self.grid.min_silence
                {
                    self.acc = None;
                    self.finalise(acc, Some(f * self.grid.hop), n, ranges);
                }
            }
        } else {
            if let Some(start) = self.cur.take() {
                self.merge(FrameRun { start, end: f });
            }
            if let Some(acc) = self.acc
                && f + 1 - acc.end >= self.settle
            {
                self.acc = None;
                self.finalise(acc, None, n, ranges);
            }
        }
        self.force_cut(ranges);
    }

    /// Absorb a closed voiced run. Anything still accumulating is within
    /// `min_silence` of it — the non-merging case was already finalised in
    /// [`Slicer::step`], when this run started.
    fn merge(&mut self, run: FrameRun) {
        self.acc = Some(match self.acc {
            Some(acc) => FrameRun {
                start: acc.start,
                end: run.end,
            },
            None => run,
        });
    }

    /// Turn a finished merged run into ranges, through the same helpers
    /// [`slice`] uses.
    fn finalise(
        &mut self,
        run: FrameRun,
        next_raw_start: Option<usize>,
        n: usize,
        ranges: &mut Vec<(usize, usize)>,
    ) {
        let range = (
            self.grid
                .padded_start(run.start * self.grid.hop, self.prev_raw_end),
            self.grid
                .padded_end(self.grid.raw_end(run.end, n), next_raw_start, n),
        );
        self.prev_raw_end = Some(self.grid.raw_end(run.end, usize::MAX));

        let Self {
            emit, buf, offset, ..
        } = self;
        emit.push(
            range,
            View {
                samples: buf,
                offset: *offset,
            },
            ranges,
        );
    }

    /// Cut a run that has gone on longer than `max_clip` without a gap.
    ///
    /// Its *start* is already fixed — only the end is still moving — so the
    /// piece taken here is the one the batch path would also start with. Where
    /// the two differ is the cut point: this one is as late as `max_clip`
    /// allows, while [`slice`] knows the run's full length and halves it.
    fn force_cut(&mut self, ranges: &mut Vec<(usize, usize)>) {
        if self.grid.max_clip == 0 {
            return;
        }
        let Some(first) = self.acc.map(|acc| acc.start).or(self.cur) else {
            return;
        };
        let start = self
            .grid
            .padded_start(first * self.grid.hop, self.prev_raw_end)
            .max(self.emit.prev_end);
        if (self.frames * self.grid.hop).saturating_sub(start) <= self.grid.max_clip {
            return;
        }
        let margin = (self.grid.max_clip / 4)
            .max(self.grid.hop)
            .max(self.grid.min_clip);
        let (lo, hi) = (start + margin, start + self.grid.max_clip);
        if lo >= hi {
            return;
        }

        let Self {
            emit,
            buf,
            offset,
            grid,
            ..
        } = self;
        let view = View {
            samples: buf,
            offset: *offset,
        };
        let cut = quietest_cut(view, grid, lo, hi).clamp(lo, hi);
        emit.push((start, cut), view, ranges);
    }

    /// Drop audio no future clip can reach: everything before the padded start
    /// of the run in progress, or of one that could begin at the next frame.
    fn trim(&mut self) {
        let keep = match self.acc.map(|acc| acc.start).or(self.cur) {
            Some(f) => self.grid.padded_start(f * self.grid.hop, self.prev_raw_end),
            None => (self.frames * self.grid.hop).saturating_sub(self.grid.pad),
        }
        .max(self.emit.prev_end)
        // Never past the next frame's window, which is still to be measured —
        // nor past the audio itself, since the last frame of all is truncated
        // and so starts before `frames * hop` reaches.
        .min((self.frames * self.grid.hop).min(self.total));
        if keep > self.offset {
            self.buf.drain(..keep - self.offset);
            self.offset = keep;
        }
    }
}

/// dBFS of one window: `20 * log10(rms + 1e-9)`.
fn window_db(window: &[f32]) -> f32 {
    let mean_sq = window
        .iter()
        .map(|x| (*x as f64) * (*x as f64))
        .sum::<f64>()
        / window.len() as f64;
    (20.0 * (mean_sq.sqrt() + 1e-9).log10()) as f32
}

/// Per-frame dBFS over ~30 ms windows at a ~10 ms hop.
fn frame_db(samples: &[f32], hop: usize, win: usize) -> Vec<f32> {
    let n = samples.len();
    if n == 0 {
        return Vec::new();
    }
    let frames = n.div_ceil(hop);
    let mut db = Vec::with_capacity(frames);
    for f in 0..frames {
        let start = f * hop;
        if start >= n {
            break;
        }
        db.push(window_db(&samples[start..(start + win).min(n)]));
    }
    db
}

/// Convert a duration in seconds to a whole number of frames (hops).
fn seconds_to_frames(seconds: f32, sample_rate: u32, hop: usize) -> usize {
    let samples = seconds * sample_rate as f32;
    (samples / hop as f32).round() as usize
}

/// Contiguous non-silent frame runs, merging any two separated by a silent gap
/// **shorter than** `min_silence_frames` (that gap is an internal pause, not a
/// sentence boundary).
fn merged_voiced_runs(db: &[f32], silence_db: f32, min_silence_frames: usize) -> Vec<FrameRun> {
    // Raw voiced runs first.
    let mut runs: Vec<FrameRun> = Vec::new();
    let mut cur: Option<usize> = None;
    for (i, &d) in db.iter().enumerate() {
        let voiced = d >= silence_db;
        match (voiced, cur) {
            (true, None) => cur = Some(i),
            (false, Some(start)) => {
                runs.push(FrameRun { start, end: i });
                cur = None;
            }
            _ => {}
        }
    }
    if let Some(start) = cur {
        runs.push(FrameRun {
            start,
            end: db.len(),
        });
    }

    if runs.is_empty() {
        return runs;
    }

    // Merge runs whose separating gap is shorter than `min_silence_frames`.
    let mut merged: Vec<FrameRun> = Vec::with_capacity(runs.len());
    let mut acc = runs[0];
    for run in &runs[1..] {
        let gap = run.start - acc.end; // frames of silence between the runs
        if gap < min_silence_frames {
            acc.end = run.end;
        } else {
            merged.push(acc);
            acc = *run;
        }
    }
    merged.push(acc);
    merged
}

/// Sample index of the quietest frame boundary within `[lo, hi)`. Ties in
/// energy (e.g. a flat passage) break toward the centre so splits stay balanced.
fn quietest_cut(view: View, grid: &Grid, lo: usize, hi: usize) -> usize {
    let n = view.end();
    let mid = lo + (hi - lo) / 2;
    let mut best = mid;
    let mut best_energy = f64::INFINITY;
    let mut best_dist = usize::MAX;
    let mut pos = lo;
    while pos < hi {
        let end = (pos + grid.win).min(n).max(pos + 1).min(n);
        let slice = view.get(pos, end);
        let energy: f64 = slice.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>()
            / slice.len().max(1) as f64;
        // Relative tolerance so genuinely-tied frames are treated as equal.
        let tol = best_energy.abs() * 1e-6 + 1e-12;
        let dist = pos.abs_diff(mid);
        if energy < best_energy - tol || (energy <= best_energy + tol && dist < best_dist) {
            best_energy = energy;
            best_dist = dist;
            best = pos;
        }
        pos += grid.hop;
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    const SR: u32 = 48_000;

    /// Append `secs` seconds of a ±0.5 voiced tone, well above the -40 dB floor.
    fn push_voiced(buf: &mut Vec<f32>, secs: f32) {
        let n = (secs * SR as f32) as usize;
        for i in 0..n {
            buf.push(if i % 2 == 0 { 0.5 } else { -0.5 });
        }
    }

    /// Append `secs` seconds of pure silence.
    fn push_silence(buf: &mut Vec<f32>, secs: f32) {
        let n = (secs * SR as f32) as usize;
        buf.extend(std::iter::repeat_n(0.0, n));
    }

    /// Append `secs` seconds of white noise at a *known* RMS, which is what
    /// lets a measured floor be checked against a number instead of against
    /// itself. Uniform on `[-amp, amp]` has RMS `amp / sqrt(3)`.
    fn push_noise(buf: &mut Vec<f32>, rng: &mut Rng, secs: f32, rms: f32) {
        let amp = rms * 3f32.sqrt();
        let n = (secs * SR as f32) as usize;
        for _ in 0..n {
            buf.push((rng.unit() * 2.0 - 1.0) * amp);
        }
    }

    /// dBFS of a linear amplitude, for stating a test's expectation in the
    /// units the measurement reports.
    fn dbfs(amplitude: f32) -> f32 {
        20.0 * amplitude.log10()
    }

    /// Run a signal through [`Slicer`] in fixed-size chunks.
    fn stream(sig: &[f32], sample_rate: u32, opts: &SliceOptions, chunk: usize) -> Vec<Clip> {
        let mut slicer = Slicer::new(sample_rate, opts);
        let mut clips = Vec::new();
        for part in sig.chunks(chunk) {
            clips.extend(slicer.push(part));
        }
        clips.extend(slicer.finish());
        clips
    }

    #[test]
    fn short_gap_keeps_sentence_long_gap_cuts() {
        // [voiced 2s][gap 0.15s][voiced 2s][gap 0.6s][voiced 2s]
        let mut sig = Vec::new();
        push_voiced(&mut sig, 2.0);
        push_silence(&mut sig, 0.15);
        push_voiced(&mut sig, 2.0);
        push_silence(&mut sig, 0.6);
        push_voiced(&mut sig, 2.0);

        let segs = slice(&sig, SR, &SliceOptions::default());
        assert_eq!(
            segs.len(),
            2,
            "0.15s internal gap must merge (sentence kept whole); 0.6s gap must cut: got {segs:?}"
        );
    }

    #[test]
    fn all_silence_yields_nothing() {
        let mut sig = Vec::new();
        push_silence(&mut sig, 3.0);
        assert!(slice(&sig, SR, &SliceOptions::default()).is_empty());
    }

    #[test]
    fn empty_input_yields_nothing() {
        assert!(slice(&[], SR, &SliceOptions::default()).is_empty());
    }

    #[test]
    fn short_blip_is_dropped() {
        // A 0.2s voiced blip is below the default 1.0s min_clip.
        let mut sig = Vec::new();
        push_silence(&mut sig, 0.5);
        push_voiced(&mut sig, 0.2);
        push_silence(&mut sig, 0.5);
        assert!(
            slice(&sig, SR, &SliceOptions::default()).is_empty(),
            "a sub-min_clip blip must be dropped"
        );
    }

    #[test]
    fn padding_pushes_boundary_before_first_voiced_sample() {
        // One voiced run with quiet room on both sides so padding can grow.
        let mut sig = Vec::new();
        push_silence(&mut sig, 0.5);
        let first_voiced = sig.len();
        push_voiced(&mut sig, 2.0);
        push_silence(&mut sig, 0.5);

        let segs = slice(&sig, SR, &SliceOptions::default());
        assert_eq!(segs.len(), 1);
        let (start, end) = segs[0];
        assert!(
            start < first_voiced,
            "pad must push segment start ({start}) before the first voiced sample ({first_voiced})"
        );
        assert!(end <= sig.len());
    }

    #[test]
    fn ranges_never_overlap_with_tiny_min_silence() {
        // A small `--min-silence` with generous padding is the case where two
        // padded ranges could otherwise overlap (the ~30 ms energy window and
        // the 0.15 s pad both reach into the short gap between them).
        let mut sig = Vec::new();
        push_voiced(&mut sig, 1.5);
        push_silence(&mut sig, 0.08); // 80 ms > 30 ms window -> a detectable gap
        push_voiced(&mut sig, 1.5);
        let opts = SliceOptions {
            min_silence: 0.02, // 20 ms -> the 80 ms gap becomes a cut
            pad: 0.15,
            ..SliceOptions::default()
        };
        let segs = slice(&sig, SR, &opts);
        assert!(segs.len() >= 2, "tiny gap should cut: got {segs:?}");
        for w in segs.windows(2) {
            assert!(
                w[0].1 <= w[1].0,
                "ranges must stay sorted and non-overlapping: {:?} then {:?}",
                w[0],
                w[1]
            );
        }
    }

    #[test]
    fn max_clip_splits_long_runs() {
        // A single 5s voiced run with max_clip=2.0 must split into >=3 pieces.
        let mut sig = Vec::new();
        push_voiced(&mut sig, 5.0);
        let opts = SliceOptions {
            max_clip: 2.0,
            ..SliceOptions::default()
        };
        let segs = slice(&sig, SR, &opts);
        assert!(
            segs.len() >= 3,
            "5s run with max_clip 2s must split into >=3 pieces: got {}",
            segs.len()
        );
        let max_samples = (2.0 * SR as f32) as usize;
        for (s, e) in &segs {
            assert!(e - s <= max_samples + 1, "piece {s}..{e} exceeds max_clip");
        }
    }

    /// xorshift64, so the property test below is reproducible and pulls in no
    /// dependency.
    struct Rng(u64);

    impl Rng {
        fn bits(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            x
        }

        fn range(&mut self, lo: usize, hi: usize) -> usize {
            lo + (self.bits() % (hi - lo) as u64) as usize
        }

        /// A value in `[0, 1)`.
        fn unit(&mut self) -> f32 {
            (self.bits() % 10_000) as f32 / 10_000.0
        }
    }

    #[test]
    fn streaming_matches_batch() {
        let mut rng = Rng(0x5eed_1234_9abc_def0);
        for case in 0..200u32 {
            let mut sig = Vec::new();
            let mut voiced = rng.bits().is_multiple_of(2);
            for _ in 0..rng.range(2, 12) {
                let secs = 0.05 + rng.unit() * 0.85;
                if voiced {
                    push_voiced(&mut sig, secs);
                } else {
                    push_silence(&mut sig, secs);
                }
                voiced = !voiced;
            }
            let opts = SliceOptions {
                silence_db: -40.0,
                min_silence: 0.05 + rng.unit() * 0.5,
                min_clip: rng.unit() * 0.6,
                // Either off, or out of reach. A run long enough to be
                // force-cut is the one place `Slicer` is *meant* to diverge —
                // `force_cut_bounds_a_gapless_run` covers that instead.
                max_clip: if case.is_multiple_of(2) { 0.0 } else { 60.0 },
                pad: rng.unit() * 0.3,
            };

            let chunk = rng.range(1, 20_000);
            let clips = stream(&sig, SR, &opts, chunk);
            let got: Vec<_> = clips.iter().map(|c| (c.start, c.end)).collect();
            assert_eq!(
                slice(&sig, SR, &opts),
                got,
                "case {case}, chunk {chunk}, {opts:?}"
            );
            for c in &clips {
                assert_eq!(
                    c.samples,
                    sig[c.start..c.end],
                    "clip {}..{} carries the wrong audio",
                    c.start,
                    c.end
                );
            }
        }
    }

    #[test]
    fn force_cut_bounds_a_gapless_run() {
        // 12 s of unbroken voice: the batch path would halve it recursively,
        // the streaming one cuts as late as `max_clip` allows. Either way not
        // one sample may go missing.
        let mut sig = Vec::new();
        push_voiced(&mut sig, 12.0);
        let opts = SliceOptions {
            max_clip: 3.0,
            min_clip: 0.5,
            ..SliceOptions::default()
        };

        let clips = stream(&sig, SR, &opts, 4096);
        let cap = (3.0 * SR as f32) as usize;
        assert!(clips.len() >= 4, "12 s at max_clip 3 s must cut repeatedly");
        assert_eq!(clips[0].start, 0);
        assert_eq!(clips.last().unwrap().end, sig.len());
        for w in clips.windows(2) {
            assert_eq!(w[0].end, w[1].start, "a force cut must not delete audio");
        }
        for c in &clips {
            assert!(
                c.end - c.start <= cap,
                "piece {}..{} exceeds max_clip",
                c.start,
                c.end
            );
        }
    }

    #[test]
    fn measurement_recovers_a_known_noise_floor() {
        // Two decades apart, so a measurement that merely echoed the signal
        // level or a constant could not pass both.
        for floor_rms in [1e-3f32, 1e-4] {
            let mut rng = Rng(0x1234_5678_9abc_def0);
            let mut sig = Vec::new();
            // 60% floor / 40% voice puts the 10th percentile inside the noise
            // and the 75th inside the tone.
            push_noise(&mut sig, &mut rng, 3.0, floor_rms);
            push_voiced(&mut sig, 2.0);

            let m = noise_floor(&sig, SR).expect("5 s is plenty to measure");
            assert!(
                (m.floor_db - dbfs(floor_rms)).abs() < 1.5,
                "floor {:.2} dBFS should be within 1.5 dB of {:.2}",
                m.floor_db,
                dbfs(floor_rms)
            );
            assert!(
                (m.signal_db - dbfs(0.5)).abs() < 0.5,
                "signal {:.2} dBFS should be within 0.5 dB of {:.2} (a +/-0.5 square)",
                m.signal_db,
                dbfs(0.5)
            );
            // The ratio is the reading `rvc-train` weights its sampling by.
            assert!(
                (m.snr_db() - (dbfs(0.5) - dbfs(floor_rms))).abs() < 2.0,
                "snr {:.1} dB",
                m.snr_db()
            );
        }
    }

    #[test]
    fn a_derived_floor_clears_the_room_tone_without_reaching_the_speech() {
        let mut rng = Rng(0xdead_beef_0000_0001);
        let mut sig = Vec::new();
        push_noise(&mut sig, &mut rng, 3.0, 1e-3); // -60 dBFS room tone
        push_voiced(&mut sig, 2.0); // -6 dBFS speech

        let m = noise_floor(&sig, SR).unwrap();
        let derived = m.silence_db();
        assert!(
            derived > m.floor_db && derived < m.signal_db,
            "derived floor {derived:.2} must sit between {:.2} and {:.2}",
            m.floor_db,
            m.signal_db
        );
        // A 54 dB range is room enough for the flat margin, so that is the
        // branch that binds rather than the fraction.
        assert!((derived - (m.floor_db + FLOOR_MARGIN_DB)).abs() < 1e-4);
        // ...and the whole point: it is far below the fixed default, so the
        // soft content between -52 and -40 dBFS survives.
        assert!(derived < -40.0, "derived {derived:.2} should be under -40");

        let opts = SliceOptions::default().with_measured_floor(&sig, SR);
        assert_eq!(opts.silence_db, derived);
    }

    #[test]
    fn a_noisy_take_gets_a_floor_that_does_not_gate_its_speech() {
        // Room tone only ~14 dB under the voice, which is the close-mic case
        // this corpus is full of: a flat 8 dB margin would land 6 dB under the
        // speech and start cutting into it, so the fraction has to win.
        let mut rng = Rng(0x0bad_c0de_1111_2222);
        let mut sig = Vec::new();
        push_noise(&mut sig, &mut rng, 3.0, 0.1); // -20 dBFS
        push_voiced(&mut sig, 2.0); // -6 dBFS

        let m = noise_floor(&sig, SR).unwrap();
        let derived = m.silence_db();
        let gap = m.signal_db - m.floor_db;
        assert!(gap < FLOOR_MARGIN_DB / MARGIN_FRACTION, "gap {gap:.1} dB");
        assert!(
            (derived - (m.floor_db + MARGIN_FRACTION * gap)).abs() < 1e-4,
            "the fraction must bind: derived {derived:.2}, floor {:.2}, signal {:.2}",
            m.floor_db,
            m.signal_db
        );
        // The whole point of the fraction: the threshold cannot reach the
        // speech, whatever the recording's range.
        assert!(derived < m.floor_db + gap * 0.5);
    }

    #[test]
    fn a_loud_room_is_cut_where_the_fixed_floor_keeps_the_dead_air() {
        // Room tone at -35 dBFS under speech at -25: a recording whose noise
        // sits *above* the fixed -40, so nothing in it is ever silent and the
        // whole take comes back as one clip — dead air and all, which is what
        // collapsed a generator to silence. The derived floor lands between the
        // two and finds the sentences.
        let mut rng = Rng(0x9999_1111_2222_3333);
        let mut sig = Vec::new();
        for i in 0..3 {
            push_noise(&mut sig, &mut rng, 1.0, 10f32.powf(-35.0 / 20.0));
            if i < 2 {
                let n = (1.5 * SR as f32) as usize;
                let amp = 10f32.powf(-25.0 / 20.0);
                sig.extend((0..n).map(|j| if j.is_multiple_of(2) { amp } else { -amp }));
            }
        }

        let m = noise_floor(&sig, SR).unwrap();
        let derived = m.silence_db();
        assert!(
            derived > m.floor_db && derived < m.signal_db,
            "derived {derived:.1} must separate tone {:.1} from speech {:.1}",
            m.floor_db,
            m.signal_db
        );
        assert_eq!(
            slice(&sig, SR, &SliceOptions::default()).len(),
            1,
            "the fixed -40 floor cannot see this recording's dead air at all"
        );
        assert_eq!(
            slice(
                &sig,
                SR,
                &SliceOptions::default().with_measured_floor(&sig, SR)
            )
            .len(),
            2,
            "the derived floor must recover both sentences"
        );
    }

    #[test]
    fn a_derived_floor_is_never_positive() {
        // Full-scale noise: `floor + margin` is above 0 dBFS, which would call
        // every sample silence and produce no clips at all.
        let mut rng = Rng(0x4444_5555_6666_7777);
        let mut sig = Vec::new();
        push_noise(&mut sig, &mut rng, 2.0, 1.0);
        let m = noise_floor(&sig, SR).unwrap();
        assert!(m.silence_db() <= 0.0, "{:.2} dBFS", m.silence_db());
    }

    #[test]
    fn digital_silence_is_clamped_rather_than_unbounded() {
        // `window_db`'s +1e-9 puts true silence near -180 dBFS. Unclamped, the
        // ratio against it is ~1e9, and an SNR-weighted sampler would hand such
        // a clip the whole distribution.
        let mut sig = Vec::new();
        push_silence(&mut sig, 3.0);
        push_voiced(&mut sig, 2.0);
        let m = noise_floor(&sig, SR).unwrap();
        assert!(m.floor_db < SILENT_FLOOR_DB, "{:.1} dBFS", m.floor_db);
        // 0.5 full scale over the 1e-6 clamp is 5e5 — the same number the
        // linear `.max(1e-6)` this replaces produced.
        assert!(
            (m.snr() - 5e5).abs() < 5e4,
            "clamped snr {:.3e} should be ~5e5",
            m.snr()
        );
    }

    #[test]
    fn nothing_to_measure_leaves_the_floor_alone() {
        let short = vec![0.5f32; 100]; // 3 frames at 48 kHz, under the minimum
        assert!(noise_floor(&short, SR).is_none());
        assert!(noise_floor(&[], SR).is_none());
        assert!(noise_floor(&[0.5; 48_000], 0).is_none());

        let opts = SliceOptions::default();
        assert_eq!(opts.silence_db, -40.0, "the default floor must not move");
        assert_eq!(
            opts.with_measured_floor(&short, SR).silence_db,
            -40.0,
            "an unmeasurable signal must leave silence_db untouched"
        );
    }

    #[test]
    fn streaming_tolerates_a_zero_sample_rate() {
        let mut slicer = Slicer::new(0, &SliceOptions::default());
        assert!(slicer.push(&[0.5; 4096]).is_empty());
        assert!(slicer.finish().is_empty());
    }

    /// dBFS of a hand-computed RMS, in the exact form [`window_db`] reports it
    /// — the `+1e-9` included, so an expectation and a reading can be compared
    /// without a fudge factor standing in for it.
    fn expected_db(rms: f64) -> f64 {
        20.0 * (rms + 1e-9).log10()
    }

    /// The energy every cut in this file is decided by, against signals whose
    /// RMS is known from their algebra rather than from another run of the same
    /// code.
    ///
    /// `streaming_matches_batch` cannot see an error here and neither can any
    /// other test in this module: both paths call [`window_db`], so a wrong
    /// frame energy is wrong identically on each side and the equivalence still
    /// holds. That is the shape the separation bug had — an invariant that
    /// survives the answer being wrong — which is why the oracle has to come
    /// from outside the module.
    #[test]
    fn frame_energy_is_the_dbfs_of_a_hand_computed_rms() {
        // A full-scale square: every sample is +/-1, so the RMS is exactly 1
        // and the reading is exactly 0 dBFS.
        let square: Vec<f32> = (0..1440)
            .map(|i: usize| if i.is_multiple_of(2) { 1.0 } else { -1.0 })
            .collect();
        assert!(
            (f64::from(window_db(&square))).abs() < 1e-6,
            "a full-scale square is 0 dBFS, read {:.6}",
            window_db(&square)
        );

        // ...and at half scale, exactly 6.0206 dB under it.
        let half: Vec<f32> = square.iter().map(|x| x * 0.5).collect();
        assert!((f64::from(window_db(&half)) - expected_db(0.5)).abs() < 1e-4);

        // A sine of amplitude `a` has RMS `a / sqrt(2)` — over a whole number
        // of periods *exactly*, which is why this is 1 kHz at 48 kHz (48
        // samples to a period, 30 of them in the 1440-sample window) and not
        // the 220 Hz the stages' own fixtures use.
        let sine: Vec<f32> = (0..1440)
            .map(|i| (0.5 * (std::f64::consts::TAU * 1000.0 * i as f64 / SR as f64).sin()) as f32)
            .collect();
        let want = expected_db(0.5 / 2f64.sqrt());
        assert!(
            (f64::from(window_db(&sine)) - want).abs() < 1e-3,
            "a half-scale sine is {want:.4} dBFS, read {:.4}",
            window_db(&sine)
        );

        // Uniform noise on +/-a has RMS `a / sqrt(3)`. That identity is what
        // `push_noise` is built on, so every measured-floor test in this file
        // rests on it holding.
        let mut rng = Rng(0x1111_2222_3333_4444);
        let amp = 0.2f64;
        let noise: Vec<f32> = (0..480_000)
            .map(|_| ((f64::from(rng.unit()) * 2.0 - 1.0) * amp) as f32)
            .collect();
        let want = expected_db(amp / 3f64.sqrt());
        assert!(
            (f64::from(window_db(&noise)) - want).abs() < 0.05,
            "uniform noise on +/-{amp} is {want:.4} dBFS, read {:.4}",
            window_db(&noise)
        );

        // Digital silence is the `+1e-9` and nothing else: -180 dBFS.
        assert!((f64::from(window_db(&[0.0; 256])) - expected_db(0.0)).abs() < 1e-6);
    }

    /// Where a frame starts, how long its window is, and how many of them a
    /// signal yields — pinned with an impulse, since an off-by-one in the
    /// geometry moves every cut in the file by a hop and leaves both paths
    /// agreeing about the wrong place.
    #[test]
    fn the_frame_grid_is_ten_millisecond_hops_of_thirty_millisecond_windows() {
        assert_eq!(frame_geometry(48_000), (480, 1440));
        assert_eq!(frame_geometry(44_100), (441, 1323));
        assert_eq!(frame_geometry(16_000), (160, 480));
        // A rate at which neither duration rounds to a whole sample still has
        // to yield a usable grid rather than a division by zero.
        assert_eq!(frame_geometry(1), (1, 1));

        let (hop, win) = frame_geometry(SR);

        // Ten whole hops of silence with one full-scale sample at 2500. Frame
        // `f` covers `[f * hop, f * hop + win)`, so exactly frames 3, 4 and 5
        // can see it — 2 ends at 2400 and 6 starts at 2880.
        let mut sig = vec![0.0f32; 10 * hop];
        sig[2500] = 1.0;
        let db = frame_db(&sig, hop, win);
        assert_eq!(
            db.len(),
            10,
            "one frame per hop over a whole number of hops"
        );
        let loud: Vec<usize> = (0..db.len()).filter(|&f| db[f] > -100.0).collect();
        assert_eq!(loud, vec![3, 4, 5], "frame windows start at f * hop");
        for f in loud {
            let want = expected_db((1.0 / win as f64).sqrt());
            assert!(
                (f64::from(db[f]) - want).abs() < 1e-4,
                "frame {f} spreads one sample over {win}: {want:.4} dBFS, read {:.4}",
                db[f]
            );
        }

        // A trailing partial window is measured rather than dropped, and it is
        // divided by the samples actually there: 100 past the last whole hop is
        // an eleventh frame, and the two before it reach into the same impulse
        // over windows the end of the input truncates.
        let mut sig = vec![0.0f32; 10 * hop + 100];
        sig[10 * hop + 50] = 1.0;
        let db = frame_db(&sig, hop, win);
        assert_eq!(db.len(), 11, "the short final frame is included");
        for (f, len) in [(10usize, 100usize), (9, hop + 100), (8, 2 * hop + 100)] {
            let want = expected_db((1.0 / len as f64).sqrt());
            assert!(
                (f64::from(db[f]) - want).abs() < 1e-4,
                "frame {f} covers {len} samples: {want:.4} dBFS, read {:.4}",
                db[f]
            );
        }
    }

    /// [`noise_floor`] reads two *ranks* off the sorted frame levels, and this
    /// is what says so: a mean, a minimum, or a percentile transcribed one
    /// decimal out lands on a different step of a staircase.
    ///
    /// 101 blocks of 25 hops, each 1 dB louder than the last, from -100 to
    /// 0 dBFS. Every frame whose window lies inside a block reads that block's
    /// level exactly, and the two frames straddling each boundary read
    /// something strictly between its neighbours — so the sorted array *is* the
    /// staircase and rank `i` names block `i / 25`. 2525 frames put the 10th
    /// percentile at index 252 (block 10, -90 dBFS) and the 75th at 1893
    /// (block 75, -25 dBFS), neither of them within a block of an extreme.
    #[test]
    fn a_percentile_names_the_frame_level_at_that_rank() {
        // 8 kHz keeps the fixture small; the window is three hops there just as
        // it is at 48 kHz, which is the only property the construction needs.
        const RATE: u32 = 8_000;
        const HOPS: usize = 25;
        let (hop, win) = frame_geometry(RATE);
        assert_eq!(win, 3 * hop);

        let mut sig = Vec::with_capacity(101 * HOPS * hop);
        for k in 0..101 {
            let amp = 10f32.powf((k as f32 - 100.0) / 20.0);
            sig.extend((0..HOPS * hop).map(|i| if i.is_multiple_of(2) { amp } else { -amp }));
        }

        let m = noise_floor(&sig, RATE).expect("2525 frames is plenty to measure");
        assert!(
            (f64::from(m.floor_db) + 90.0).abs() < 0.01,
            "the 10th percentile is block 10's -90 dBFS, read {:.4}",
            m.floor_db
        );
        assert!(
            (f64::from(m.signal_db) + 25.0).abs() < 0.01,
            "the 75th percentile is block 75's -25 dBFS, read {:.4}",
            m.signal_db
        );
        // The quietest frame in the signal is 10 dB under the reported floor
        // and the loudest 25 dB over the reported signal, so neither reading is
        // an extreme dressed up as a percentile.
        assert!(
            (m.snr_db() - 65.0).abs() < 0.02,
            "-25 over -90 is 65 dB, read {:.2}",
            m.snr_db()
        );
    }
}
