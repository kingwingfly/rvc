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
