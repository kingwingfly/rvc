//! Streaming de-hiss: non-local means over mono `f32`, computed here.
//!
//! Here rather than in an engine because two callers now want it and neither
//! owns it: voice conversion runs it as an optional stage on the generator's
//! output, and corpus preparation runs it over a whole recording before
//! anything is trained on it. It knows about no model and no engine — it is an
//! algorithm and a sample rate.
//!
//! Soft, close-mic speech is the hard case: the content itself (breaths,
//! whispers, mouth sounds) is soft and *broadband* — spectrally
//! indistinguishable from steady hiss — so a spectral-subtraction de-noiser
//! (`afftdn`) that keys off level and frequency muffles the very texture we
//! want to keep, and an RNN de-noiser (`arnndn`) keys off pitch and harmonic
//! structure, which whispered material does not have. **Non-local means keys
//! off neither.** It averages each short patch of audio with *self-similar*
//! patches found nearby in time: stationary hiss looks the same everywhere and
//! averages away, while ever-changing breath texture has few close matches and
//! survives intact. That property, not "denoising", is what this stage is for,
//! and it is the thing any replacement has to keep.
//!
//! On a breath-over-hiss signal typical of soft, close-mic speech this removes
//! ~75 % of the hiss while preserving content energy and high-frequency
//! "crispness" to within ~1 %, with none of the warbly musical-noise `afftdn`
//! leaves behind.
//!
//! ## Why the algorithm is ours and not ffmpeg's
//!
//! This was `AudioFilter::new(sr, "anlmdn=…")` until ffmpeg 9.0, where that
//! filter corrupts the heap — see **De-hiss is ours, because ffmpeg 9.0 broke
//! `anlmdn`** in `CLAUDE.md` for the measurement, and [`Nlm::frame`] for the
//! one-line defect
//! that causes it. Two things made porting the better answer over waiting.
//!
//! The bug is **unreported and unreachable from here**: nothing in this
//! workspace can fix a heap overflow inside libavfilter, a fail-open path
//! cannot catch it (the graph builds *successfully* and then corrupts memory,
//! so there is no error to fail open on), and a version guard would have to be
//! widened by hand whenever upstream moves. And the algorithm is small — a
//! patch SSD, an `exp(-d)` lookup and a cache that slides one sample at a time.
//! `libavfilter/af_anlmdn.c` is its whole specification, so this is a port
//! rather than an invention, and the defaults below are still the ones that
//! were tuned by ear against that filter.
//!
//! The arithmetic is deliberately upstream's, down to the 2²⁰-entry weight
//! table, because the point is to keep the sound identical rather than to
//! improve on it. [`AudioFilter`](crate::AudioFilter) is untouched and still
//! carries `ebur128` for `preprocess normalize`; only de-hiss stopped going
//! through libavfilter.
//!
//! One consequence worth stating: **de-hiss can no longer be "unavailable".**
//! The old wrapper failed *open* — passing audio through with a warning when
//! ffmpeg had been built without `anlmdn` — and that branch is gone along with
//! the dependency that needed it.

/// `exp()` is far too slow for the inner loop — it runs `2 * research` times
/// per output sample, ~28 M times per second of 48 kHz audio at the defaults —
/// so weights come from a table. Upstream's size, kept because a coarser table
/// would quietly change the sound of a corpus that was tuned against it.
const WEIGHT_LUT_NBITS: u32 = 20;
const WEIGHT_LUT_SIZE: usize = 1 << WEIGHT_LUT_NBITS;

/// `anlmdn`'s `m`, never exposed as a flag and so always upstream's default.
/// It bounds how far a patch may differ before it stops contributing at all.
const SMOOTH: f32 = 11.0;

/// Tunable de-hiss settings. Defaults are tuned for soft, breathy content
/// (`strength` is the one knob most worth touching: raise it for more hiss
/// removal, lower it if soft texture starts to smear).
#[derive(Debug, Clone, Copy)]
pub struct DenoiseParams {
    /// Denoising strength: higher removes more hiss but eventually smears the
    /// texture. ~0.008 removes most steady hiss with no audible content loss.
    pub strength: f32,
    /// Patch duration in seconds — the unit compared for self-similarity.
    /// Small keeps fine detail; 2 ms is upstream's default.
    pub patch_secs: f32,
    /// Research-window duration in seconds: how far in time the filter looks
    /// for similar patches. Must exceed `patch_secs`; also sets the latency.
    pub research_secs: f32,
}

impl Default for DenoiseParams {
    fn default() -> Self {
        Self {
            strength: 0.008,
            patch_secs: 0.002,
            research_secs: 0.006,
        }
    }
}

/// Non-local means over one mono `f32` stream at a fixed rate.
///
/// Ported from `libavfilter/af_anlmdn.c`; the names here are upstream's so the
/// two can be read side by side. `k` is the patch half-length, `s` the research
/// half-length, `h = 2k + 1` the hop, and the sliding `window` is
/// `h + 2 * (k + s)` samples so a patch centred anywhere in the output range
/// has its full research neighbourhood in memory.
struct Nlm {
    k: usize,
    s: usize,
    h: usize,
    window: Vec<f32>,
    /// Squared distance from the patch under the cursor to each of its `2 * s`
    /// neighbours, rebuilt at the start of every frame and then slid one sample
    /// at a time. Sliding it is the whole reason this is affordable: an
    /// `O(patch)` update replaces an `O(patch * research)` recomputation.
    cache: Vec<f32>,
    weight_lut: Vec<f32>,
    /// Scales a squared distance into the weight table's domain.
    sw: f32,
    pdiff_lut_scale: f32,
}

impl Nlm {
    fn new(sample_rate: u32, params: &DenoiseParams) -> Self {
        // Upstream's option ranges, applied here for the same reason it applies
        // them there: they bound the buffers and the inner loop, and a research
        // window of 0.3 s at 48 kHz is already 28 800 comparisons per sample.
        let sr = sample_rate.max(1) as f32;
        let patch = params.patch_secs.clamp(0.001, 0.1);
        let research = params.research_secs.clamp(0.002, 0.3).max(patch + 0.001);
        let strength = params.strength.clamp(0.00001, 10000.0);

        let k = ((patch * sr).round() as usize).max(1);
        let s = ((research * sr).round() as usize).max(1);
        let h = 2 * k + 1;

        let pdiff_lut_scale = 1.0 / SMOOTH * WEIGHT_LUT_SIZE as f32;
        Self {
            k,
            s,
            h,
            window: vec![0.0; h + (k + s) * 2],
            cache: vec![0.0; 2 * s],
            weight_lut: (0..WEIGHT_LUT_SIZE)
                .map(|i| (-(i as f32) / pdiff_lut_scale).exp())
                .collect(),
            sw: (65536.0 / (4 * k + 2) as f32) / strength.sqrt(),
            pdiff_lut_scale,
        }
    }

    /// Squared distance between the patches centred at window positions `a` and
    /// `b`. Both span `2k + 1` samples, so this reads `window[a ..= a + 2k]`
    /// against `window[b ..= b + 2k]`.
    fn ssd(&self, a: usize, b: usize) -> f32 {
        self.window[a..=a + 2 * self.k]
            .iter()
            .zip(&self.window[b..=b + 2 * self.k])
            .map(|(x, y)| (x - y) * (x - y))
            .sum()
    }

    /// Consume up to `h` samples and append exactly `input.len()` filtered
    /// samples to `out`.
    ///
    /// **`input.len()` is what bounds the output loop, and that is precisely
    /// the line ffmpeg gets wrong.** `filter_channel` there always writes `h`
    /// samples, but at EOF `ff_inlink_consume_samples` hands it the short final
    /// frame and `out = in` is only `nb_samples` long — so it writes past the
    /// end by `h - nb_samples` floats. Its own `memset` two lines earlier zeroes
    /// the *window* by that same shortfall, which is what makes it clear the
    /// partial frame was anticipated for the input and forgotten for the output.
    /// Measured: 100 × `h` samples through ffmpeg 9.0.1 is clean, one sample
    /// more aborts.
    fn frame(&mut self, input: &[f32], out: &mut Vec<f32>) {
        let (k, s, h) = (self.k, self.s, self.h);
        let count = input.len().min(h);

        // Slide the window left by one hop and append the new samples, zeroing
        // whatever a short final frame does not fill.
        self.window.copy_within(h.., 0);
        let offset = self.window.len() - h;
        self.window[offset..offset + count].copy_from_slice(&input[..count]);
        self.window[offset + count..].fill(0.0);

        for i in s..s + count {
            if i == s {
                // No previous cursor to slide from: measure all 2s neighbours.
                let mut v = 0;
                for j in i - s..=i + s {
                    if j != i {
                        self.cache[v] = self.ssd(i, j);
                        v += 1;
                    }
                }
            } else {
                // The cursor moved one sample, so every patch gained a sample at
                // its right edge and lost one at its left. Both halves of the
                // cache slide by the same difference of squared differences.
                let (lead, trail) = (self.window[i + 2 * k], self.window[i - 1]);
                for v in 0..s {
                    let before = i - s + v;
                    let after = i + 1 + v;
                    let d = |x: f32, y: f32| (x - y) * (x - y);
                    self.cache[v] +=
                        d(lead, self.window[before + 2 * k]) - d(trail, self.window[before - 1]);
                    self.cache[s + v] +=
                        d(lead, self.window[after + 2 * k]) - d(trail, self.window[after - 1]);
                }
            }

            // Weighted mean of the neighbours, plus the sample itself at weight
            // 1 so a patch with no close matches passes through untouched.
            let (mut p, mut q) = (0.0f32, 1.0f32);
            for v in 0..2 * s {
                // Sliding the cache accumulates rounding, which can drift a
                // distance below zero; upstream clamps in place and so do we.
                if self.cache[v] < 0.0 {
                    self.cache[v] = 0.0;
                }
                let w = self.cache[v] * self.sw;
                if w >= SMOOTH {
                    continue;
                }
                let weight = self.weight_lut[(w * self.pdiff_lut_scale) as usize];
                // Neighbour v is at i - s + v, skipping the cursor itself.
                p += weight * self.window[k + i - s + v + usize::from(v >= s)];
                q += weight;
            }
            p += self.window[k + i];
            out.push(p / q);
        }
    }
}

/// Streaming de-hiss stage. One per stream; not shared across threads.
pub struct Denoiser {
    sample_rate: u32,
    params: DenoiseParams,
    /// Built on first [`Self::process`], because the weight table is 4 MB and a
    /// `Denoiser` that is constructed and never fed should not pay for it.
    nlm: Option<Nlm>,
    /// Input not yet forming a whole hop.
    pending: Vec<f32>,
}

impl Denoiser {
    /// Build a de-hiss stage at the given sample rate and strength.
    pub fn new(sample_rate: u32, params: DenoiseParams) -> Self {
        Self {
            sample_rate,
            params,
            nlm: None,
            pending: Vec::new(),
        }
    }

    /// Clear all state so the stage can start fresh on an independent input.
    /// Drops the sliding window, because the research window would otherwise
    /// average the start of one recording against the end of another.
    pub fn reset(&mut self) {
        self.nlm = None;
        self.pending.clear();
    }

    /// Feed samples; returns the filtered output produced so far. Output lags
    /// input by up to one hop, so this returns fewer samples than it was given
    /// until [`Self::flush`] drains the tail.
    pub fn process(&mut self, input: &[f32]) -> Vec<f32> {
        let nlm = self
            .nlm
            .get_or_insert_with(|| Nlm::new(self.sample_rate, &self.params));

        self.pending.extend_from_slice(input);
        let mut out = Vec::with_capacity(self.pending.len());
        let mut taken = 0;
        while self.pending.len() - taken >= nlm.h {
            nlm.frame(&self.pending[taken..taken + nlm.h], &mut out);
            taken += nlm.h;
        }
        self.pending.drain(..taken);
        out
    }

    /// Drain the tail: whatever is left is one short final frame. Empty if
    /// de-hiss was never fed anything.
    pub fn flush(&mut self) -> Vec<f32> {
        let Some(nlm) = self.nlm.as_mut() else {
            return Vec::new();
        };
        let mut out = Vec::with_capacity(self.pending.len());
        if !self.pending.is_empty() {
            let tail = std::mem::take(&mut self.pending);
            nlm.frame(&tail, &mut out);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rms(x: &[f32]) -> f32 {
        if x.is_empty() {
            return 0.0;
        }
        (x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32).sqrt()
    }

    fn harmonics(sr: u32, secs: f32) -> Vec<f32> {
        (0..(sr as f32 * secs) as usize)
            .map(|i| {
                let t = i as f32 / sr as f32;
                use std::f32::consts::TAU;
                0.2 * ((TAU * 220.0 * t).sin()
                    + 0.5 * (TAU * 440.0 * t).sin()
                    + 0.3 * (TAU * 880.0 * t).sin())
            })
            .collect()
    }

    fn run(sr: u32, input: &[f32], chunk: usize) -> Vec<f32> {
        let mut d = Denoiser::new(sr, DenoiseParams::default());
        let mut out = Vec::new();
        for c in input.chunks(chunk) {
            out.extend(d.process(c));
        }
        out.extend(d.flush());
        out
    }

    /// Deterministic white-ish noise; a hash of the index rather than a
    /// generator, so a test can ask for the same samples without threading
    /// state through.
    fn noise(n: usize, seed: u32, amp: f32) -> Vec<f32> {
        (0..n)
            .map(|i| {
                let mut x = (i as u32).wrapping_add(seed).wrapping_mul(2_654_435_761);
                x ^= x >> 15;
                x = x.wrapping_mul(2_246_822_519);
                x ^= x >> 13;
                amp * ((x >> 8) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0)
            })
            .collect()
    }

    fn db(a: f32, b: f32) -> f32 {
        20.0 * (a / b).log10()
    }

    /// Non-local means written straight from the equations: every squared
    /// distance measured from scratch, every weight a real `exp`, one output
    /// sample at a time.
    ///
    /// This exists because [`Nlm`] is almost entirely optimisation. Its speed
    /// comes from an incremental cache slid one sample per output and a
    /// 2²⁰-entry `exp` table, and both are exactly the kind of arithmetic that
    /// stays plausible while computing something else — a patch index off by
    /// one, a cache half updated from the wrong edge, a `sw` scaled by the
    /// wrong power of `strength`. None of that changes the length, the
    /// chunk-invariance or the energy of a tone, so nothing else here sees it.
    ///
    /// Two things it deliberately does *not* borrow from the implementation.
    /// It recomputes `k`, `s` and `sw` from `params` rather than reading them
    /// off an [`Nlm`], so a change to any of those three formulas fails this
    /// test. And it takes the parameters unclamped, which is safe only because
    /// every caller below passes values already inside upstream's ranges.
    fn naive_nlm(sr: u32, params: &DenoiseParams, x: &[f32]) -> Vec<f32> {
        let k = ((params.patch_secs * sr as f32).round() as isize).max(1);
        let s = ((params.research_secs * sr as f32).round() as isize).max(1);
        let sw = (65536.0 / (4 * k + 2) as f32) / params.strength.sqrt();

        // Everything outside the recording is silence: the sliding window
        // starts zeroed and is zero-filled past a short final frame.
        let at = |i: isize| -> f32 {
            if i < 0 || i as usize >= x.len() {
                0.0
            } else {
                x[i as usize]
            }
        };
        let ssd = |a: isize, b: isize| -> f32 {
            (-k..=k)
                .map(|d| {
                    let e = at(a + d) - at(b + d);
                    e * e
                })
                .sum()
        };

        (0..x.len() as isize)
            .map(|o| {
                // The stage emits the sample centred at `c` in output position
                // `c + k + s`: the research window is a look-ahead, and neither
                // this port nor upstream compensates for it. `nlm_parity` is
                // what pins that alignment against ffmpeg itself.
                let c = o - (k + s);
                let (mut p, mut q) = (at(c), 1.0f32);
                for j in c - s..=c + s {
                    if j == c {
                        continue;
                    }
                    let w = ssd(c, j) * sw;
                    if w >= SMOOTH {
                        continue;
                    }
                    let weight = (-w).exp();
                    p += weight * at(j);
                    q += weight;
                }
                p / q
            })
            .collect()
    }

    /// The check the other five cannot make: that the filter computes
    /// non-local means at all, rather than something with the same shape.
    ///
    /// Run at 4 kHz so `k` is 8 and `s` 24 — the reference is `O(n · s · k)`
    /// and would take minutes at 48 kHz — over a signal chosen so the weights
    /// actually vary: a periodic tone puts near-identical patches inside every
    /// research window, and the noise on top keeps them from being *equal*, so
    /// a mis-scaled `sw` or a mis-slid cache moves the answer instead of
    /// cancelling out.
    ///
    /// Measured agreement is max |diff| 2.3e-7 against an output RMS of 0.11,
    /// i.e. ~2e-6 relative, which is the weight table's quantisation
    /// (`SMOOTH / 2²⁰` ≈ 1e-5 per weight, largely cancelling between numerator
    /// and denominator) plus f32 summation order. A wrong index or a wrong
    /// scale is orders of magnitude above that, not just outside it.
    #[test]
    fn the_filter_matches_a_scalar_reference_written_from_the_equations() {
        let sr = 4_000u32;
        let tone: Vec<f32> = (0..600)
            .map(|i| 0.15 * (std::f32::consts::TAU * 110.0 * i as f32 / sr as f32).sin())
            .collect();
        let input: Vec<f32> = tone
            .iter()
            .zip(noise(600, 7, 0.004))
            .map(|(t, n)| t + n)
            .collect();

        for params in [
            DenoiseParams::default(),
            // A second point in the parameter space, so the agreement cannot
            // come from a formula that happens to be right at one setting:
            // both radii and the strength move, and `sw` depends on `k` and
            // `strength` together.
            DenoiseParams {
                strength: 0.05,
                patch_secs: 0.0015,
                research_secs: 0.005,
            },
        ] {
            let reference = naive_nlm(sr, &params, &input);
            // Every chunking, because the cache is rebuilt from scratch at each
            // frame's first sample and slid for the rest: a bug in either half
            // hides behind the other at some chunk size.
            for chunk in [input.len(), 193, 17, 1] {
                let mut d = Denoiser::new(sr, params);
                let mut out = Vec::new();
                for c in input.chunks(chunk) {
                    out.extend(d.process(c));
                }
                out.extend(d.flush());

                assert_eq!(out.len(), reference.len(), "chunk={chunk}");
                let worst = out
                    .iter()
                    .zip(&reference)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f32, f32::max);
                assert!(
                    worst < 1e-5,
                    "{params:?} chunk={chunk}: diverged from the scalar \
                     reference by {worst:.3e} (rms {:.3})",
                    rms(&reference)
                );
            }
        }
    }

    /// Silence in, silence out — **exactly** zero, not merely small.
    ///
    /// It is exact by construction rather than by luck: every patch distance in
    /// a silent window is 0, so every weight is the table's first entry
    /// `exp(0) = 1`, and the weighted mean of `2s + 1` zeros is `0 / (2s + 1)`.
    /// That makes it the one property here that a tolerance would weaken —
    /// anything that corrupts an index, a length or a buffer past its end shows
    /// up as a non-zero sample rather than as a slightly worse ratio.
    #[test]
    fn silence_comes_out_exactly_silent() {
        for sr in [8_000u32, 48_000] {
            for n in [1, 193, 4801, 20_000] {
                let out = run(sr, &vec![0.0; n], 1024);
                assert_eq!(out.len(), n);
                assert!(
                    out.iter().all(|v| *v == 0.0),
                    "sr={sr} n={n}: silence acquired energy"
                );
            }
        }
    }

    /// The property the stage is actually sold on, in the two halves that
    /// matter and in dB: steady broadband hiss falls a long way, and content
    /// does not move.
    ///
    /// Hiss and content are measured through separate passes rather than by
    /// subtracting one mixed run from another, because non-local means is not
    /// linear and the difference of two runs is not "the noise that survived".
    /// The mixture is then checked too, and it is the interesting reading: with
    /// the tone present the hiss removal drops sharply, because a patch of
    /// tone-plus-hiss has far fewer near-identical neighbours than a patch of
    /// hiss alone. That is the algorithm being conservative, and it is why this
    /// filter is safe on breathy material — it declines to average what it
    /// cannot match.
    ///
    /// Measured at the defaults, 48 kHz, hiss at -54 dBFS:
    /// hiss alone **-11.4 dB**, tone alone **-0.00 dB**, and the tone's own
    /// energy inside the mixture **-0.03 dB**.
    #[test]
    fn hiss_falls_far_while_content_does_not_move() {
        let sr = 48_000u32;
        let hiss = noise(sr as usize, 11, 0.0035);
        let tone = harmonics(sr, 1.0);
        let mix: Vec<f32> = tone.iter().zip(&hiss).map(|(t, h)| t + h).collect();

        // Skip the first `k + s` samples of every output: that is the
        // look-ahead, and it is zero-padded rather than signal.
        let settle = 1_000;
        let after = |x: &[f32]| rms(&run(sr, x, 4096)[settle..]);

        let hiss_db = db(after(&hiss), rms(&hiss[..sr as usize - settle]));
        let tone_db = db(after(&tone), rms(&tone[..sr as usize - settle]));
        let mix_db = db(after(&mix), rms(&mix[..sr as usize - settle]));

        eprintln!("MEASURED hiss={hiss_db:.4} tone={tone_db:.4} mix={mix_db:.4}");
        assert!(
            hiss_db < -8.0,
            "hiss barely moved: {hiss_db:.2} dB — non-local means is not \
             averaging self-similar patches"
        );
        assert!(
            tone_db.abs() < 0.1,
            "content energy moved by {tone_db:.2} dB"
        );
        // The mixture is dominated by the tone, so its total energy must not
        // move either — the hiss coming off it is 30 dB down on the content.
        assert!(mix_db.abs() < 0.1, "mixture energy moved by {mix_db:.2} dB");
    }

    /// Non-local means must **preserve** wanted content, which is the whole
    /// reason it is chosen over spectral subtraction for soft, breathy
    /// material. Its hiss *removal* is signal-dependent — it keys off the
    /// broadband self-similarity of real breath texture, so a synthetic tone
    /// cannot exercise it — and is covered by listening; here we lock in that a
    /// structured signal passes through at ~unity energy.
    #[test]
    fn a_clean_signal_keeps_its_energy() {
        let sr = 48_000u32;
        let input = harmonics(sr, 2.0);
        let out = run(sr, &input, 4096);
        let half = sr as usize / 2;
        let a = rms(&input[sr as usize..sr as usize + half]);
        let b = rms(&out[sr as usize..sr as usize + half]);
        assert!(
            (b / a - 1.0).abs() < 0.1,
            "de-hiss altered clean-signal energy: in={a} out={b}"
        );
    }

    /// Every sample in, every sample out. The batch stage writes the result
    /// straight to a `.wav`, so a stage that quietly shortened a recording
    /// would desynchronise a corpus against its transcripts.
    #[test]
    fn length_is_preserved_exactly() {
        let sr = 48_000u32;
        for n in [0, 1, 100, 4801, 48_000] {
            let input = harmonics(sr, 1.0)[..n].to_vec();
            assert_eq!(run(sr, &input, 1024).len(), n, "length changed for n={n}");
        }
    }

    /// The chunking a caller happens to use must not reach the output: voice
    /// conversion feeds block-sized pieces and corpus preparation feeds one
    /// file, and they have to agree sample for sample.
    #[test]
    fn chunking_does_not_change_the_result() {
        let sr = 48_000u32;
        let input = harmonics(sr, 0.5);
        let reference = run(sr, &input, input.len());
        for chunk in [1, 7, 193, 1024, 4096] {
            let out = run(sr, &input, chunk);
            assert_eq!(out.len(), reference.len(), "chunk={chunk} changed length");
            for (i, (a, b)) in out.iter().zip(&reference).enumerate() {
                assert!(
                    (a - b).abs() < 1e-6,
                    "chunk={chunk} diverged at {i}: {a} vs {b}"
                );
            }
        }
    }

    /// The partial final frame is what ffmpeg's `anlmdn` writes past the end of
    /// — see [`Nlm::frame`]. `h` is 193 at 48 kHz, so a length one sample over a
    /// multiple of it is the exact shape that aborts there, and lengths below
    /// one hop never form a full frame at all.
    #[test]
    fn a_partial_final_frame_is_not_an_overflow() {
        let sr = 48_000u32;
        let input = harmonics(sr, 1.0);
        for n in [1, 192, 193, 194, 193 * 100, 193 * 100 + 1] {
            let out = run(sr, &input[..n], 8192);
            assert_eq!(out.len(), n, "n={n}");
            assert!(
                out.iter().all(|v| v.is_finite()),
                "n={n} produced non-finite"
            );
        }
    }

    /// A degenerate rate or window must not panic on an index. Upstream clamps
    /// these into range rather than refusing, and so do we.
    #[test]
    fn degenerate_settings_are_clamped_not_panics() {
        let input = harmonics(8_000, 0.2);
        for params in [
            DenoiseParams {
                strength: 0.0,
                patch_secs: 0.0,
                research_secs: 0.0,
            },
            DenoiseParams {
                strength: 1e9,
                patch_secs: 10.0,
                research_secs: 1.0,
            },
        ] {
            let mut d = Denoiser::new(8_000, params);
            let mut out = d.process(&input);
            out.extend(d.flush());
            assert_eq!(out.len(), input.len());
            assert!(out.iter().all(|v| v.is_finite()));
        }
    }
}
