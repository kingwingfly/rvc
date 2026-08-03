//! Sentence-safe RMS slicer.
//!
//! Splits a mono `f32` recording into voiced `[start, end)` sample ranges by
//! removing *between-sentence dead-air only*. This is a clean-room RMS slicer
//! written for ASMR corpora, where soft breathy content is low-energy but
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

/// Tuning knobs for [`slice`]. All durations are in seconds.
#[derive(Debug, Clone, Copy)]
pub struct SliceOptions {
    /// Energy floor in dBFS. A frame quieter than this counts as silence. ASMR
    /// users can lower it (e.g. `-50`) to keep the very softest passages.
    pub silence_db: f32,
    /// Minimum length of a silent gap (seconds) for it to be a cut point.
    /// Shorter gaps are treated as internal pauses and kept inside the clip, so
    /// a complete sentence is never split.
    pub min_silence: f32,
    /// Drop any kept segment shorter than this (seconds).
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
            silence_db: -40.0,
            min_silence: 0.3,
            min_clip: 1.0,
            max_clip: 0.0,
            pad: 0.15,
        }
    }
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

impl Grid {
    fn new(sample_rate: u32, opts: &SliceOptions) -> Self {
        // ~10 ms hop, ~30 ms window, at least one sample each.
        let hop = ((sample_rate as f32 * 0.010).round() as usize).max(1);
        let win = ((sample_rate as f32 * 0.030).round() as usize).max(hop);
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
    fn streaming_tolerates_a_zero_sample_rate() {
        let mut slicer = Slicer::new(0, &SliceOptions::default());
        assert!(slicer.push(&[0.5; 4096]).is_empty());
        assert!(slicer.finish().is_empty());
    }
}
