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

/// Slice `samples` into voiced `[start, end)` sample ranges.
///
/// Returns an empty vector for empty or all-silent input. Ranges are sorted,
/// non-overlapping, and clamped to `[0, samples.len())`.
pub fn slice(samples: &[f32], sample_rate: u32, opts: &SliceOptions) -> Vec<(usize, usize)> {
    if samples.is_empty() || sample_rate == 0 {
        return Vec::new();
    }

    // ~10 ms hop, ~30 ms window, at least one sample each.
    let hop = ((sample_rate as f32 * 0.010).round() as usize).max(1);
    let win = ((sample_rate as f32 * 0.030).round() as usize).max(hop);

    let db = frame_db(samples, hop, win);
    if db.is_empty() {
        return Vec::new();
    }

    // Voiced = not silent. Merge voiced runs separated by a gap shorter than
    // `min_silence` (the key sentence-preserving step).
    let min_silence_frames = seconds_to_frames(opts.min_silence, sample_rate, hop);
    let runs = merged_voiced_runs(&db, opts.silence_db, min_silence_frames);

    // Frame runs -> padded, clamped, non-overlapping sample ranges.
    let n = samples.len();
    let pad_samples = (opts.pad * sample_rate as f32).round() as usize;
    let mut ranges: Vec<(usize, usize)> = Vec::with_capacity(runs.len());
    for (i, run) in runs.iter().enumerate() {
        // A voiced frame `f` covers samples `[f*hop, f*hop + win)`; clamp to `n`.
        let raw_start = run.start * hop;
        let raw_end = ((run.end.saturating_sub(1)) * hop + win).min(n);

        // Hard-bound padding at the midpoint of the bordering silent gap so two
        // neighbours can never both claim the same quiet and overlap.
        let lo_bound = if i == 0 {
            0
        } else {
            let prev_raw_end = ((runs[i - 1].end.saturating_sub(1)) * hop + win).min(raw_start);
            prev_raw_end + (raw_start - prev_raw_end) / 2
        };
        let hi_bound = if i + 1 < runs.len() {
            let next_raw_start = (runs[i + 1].start * hop).max(raw_end);
            raw_end + (next_raw_start - raw_end) / 2
        } else {
            n
        };
        let start = raw_start.saturating_sub(pad_samples).max(lo_bound);
        let end = (raw_end + pad_samples).min(hi_bound).min(n);
        ranges.push((start, end));
    }

    // Enforce the documented non-overlap invariant. A voiced frame spans `win`
    // (~30 ms) but frames step by `hop` (~10 ms), so with a very small
    // `--min-silence` the padded ranges above could touch or overlap. Clamp each
    // start past the previous end (dropping any range fully swallowed).
    let mut prev_end = 0usize;
    ranges.retain_mut(|(s, e)| {
        *s = (*s).max(prev_end);
        if *s >= *e {
            return false;
        }
        prev_end = *e;
        true
    });

    let min_clip_samples = (opts.min_clip * sample_rate as f32).round() as usize;

    // Split over-long runs at their quietest interior frame.
    if opts.max_clip > 0.0 {
        let max_samples = (opts.max_clip * sample_rate as f32).round() as usize;
        if max_samples > 0 {
            ranges = split_long_ranges(ranges, samples, hop, win, max_samples, min_clip_samples);
        }
    }

    // Drop runs shorter than `min_clip`.
    ranges.retain(|(s, e)| e.saturating_sub(*s) >= min_clip_samples);

    ranges
}

/// Per-frame dBFS: `20*log10(rms + 1e-9)` over ~30 ms windows at a ~10 ms hop.
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
        let end = (start + win).min(n);
        let win_slice = &samples[start..end];
        let mean_sq = win_slice
            .iter()
            .map(|x| (*x as f64) * (*x as f64))
            .sum::<f64>()
            / win_slice.len() as f64;
        let rms = mean_sq.sqrt();
        db.push((20.0 * (rms + 1e-9).log10()) as f32);
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

/// Greedily split any sample range longer than `max_samples` at its quietest
/// interior frame, recursing until every piece fits.
///
/// The cut is constrained so both halves are at least `min_clip_samples` long,
/// so splitting never manufactures a sub-`min_clip` fragment that the later
/// filter would silently discard (losing real voiced audio). If a range is too
/// short to split into two `>= min_clip` pieces, it is kept whole even though it
/// slightly exceeds `max_samples` — keeping audio beats dropping it.
fn split_long_ranges(
    ranges: Vec<(usize, usize)>,
    samples: &[f32],
    hop: usize,
    win: usize,
    max_samples: usize,
    min_clip_samples: usize,
) -> Vec<(usize, usize)> {
    let mut out = Vec::with_capacity(ranges.len());
    let mut stack: Vec<(usize, usize)> = ranges.into_iter().rev().collect();
    while let Some((start, end)) = stack.pop() {
        if end.saturating_sub(start) <= max_samples {
            out.push((start, end));
            continue;
        }
        // Keep both halves >= min_clip (and away from the exact edges) so no
        // droppable fragment is created and each half makes progress.
        let margin = (max_samples / 4).max(hop).max(min_clip_samples);
        let cut_lo = start + margin;
        let cut_hi = end.saturating_sub(margin);
        if cut_lo >= cut_hi {
            // Can't split without a sub-min_clip fragment; keep the range whole.
            out.push((start, end));
            continue;
        }
        let cut = quietest_cut(samples, hop, win, cut_lo, cut_hi).clamp(cut_lo, cut_hi);
        // Push right first so the left half is emitted in order after sorting.
        stack.push((cut, end));
        stack.push((start, cut));
    }
    out.sort_by_key(|(s, _)| *s);
    out
}

/// Sample index of the quietest frame boundary within `[lo, hi)`. Ties in
/// energy (e.g. a flat passage) break toward the centre so splits stay balanced.
fn quietest_cut(samples: &[f32], hop: usize, win: usize, lo: usize, hi: usize) -> usize {
    let n = samples.len();
    let mid = lo + (hi - lo) / 2;
    let mut best = mid;
    let mut best_energy = f64::INFINITY;
    let mut best_dist = usize::MAX;
    let mut pos = lo;
    while pos < hi {
        let end = (pos + win).min(n).max(pos + 1).min(n);
        let slice = &samples[pos..end];
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
        pos += hop;
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
}
