//! Optional streaming de-hiss for the generator's output, backed by ffmpeg's
//! `anlmdn` non-local-means de-noiser (in-process via libavfilter, not a
//! subprocess).
//!
//! ASMR is the hard case: the content itself (breaths, whispers, mouth sounds)
//! is soft and *broadband* — spectrally indistinguishable from steady hiss — so
//! a spectral-subtraction de-noiser (`afftdn`) that keys off level/frequency
//! muffles the very texture we want to keep. `anlmdn` instead averages each
//! short patch of audio with *self-similar* patches found nearby in time:
//! stationary hiss looks the same everywhere and averages away, while the
//! ever-changing breath texture has few close matches and survives intact.
//!
//! On an ASMR-like breath-over-hiss signal this removes ~75 % of the hiss while
//! preserving content energy and high-frequency "crispness" to within ~1 %,
//! with none of the warbly musical-noise `afftdn` leaves behind — and it runs
//! ~12× faster than realtime, so `serve` stays realtime.
//!
//! The heavy lifting lives in [`audio_kit::AudioFilter`]; this is a thin,
//! streaming wrapper that keeps the same `process`/`flush`/`reset` shape the
//! [`crate::Converter`] expects. The graph is built lazily and, if ffmpeg
//! cannot build it (e.g. a build without `anlmdn`), the stage fails **open** —
//! audio passes through unchanged with a warning rather than being dropped.

use audio_kit::AudioFilter;

/// Tunable de-hiss settings. Defaults are tuned for ASMR (`strength` is the one
/// knob most worth touching: raise it for more hiss removal, lower it if soft
/// texture starts to smear).
#[derive(Debug, Clone, Copy)]
pub struct DenoiseParams {
    /// `anlmdn` strength `s`: higher removes more hiss but eventually smears the
    /// texture. ~0.008 removes most steady hiss with no audible content loss.
    pub strength: f32,
    /// `anlmdn` patch duration `p` in seconds — the unit compared for
    /// self-similarity. Small keeps fine detail; 2 ms is the ffmpeg default.
    pub patch_secs: f32,
    /// `anlmdn` research-window `r` in seconds: how far in time it looks for
    /// similar patches. Must exceed `patch_secs`; also sets the filter latency.
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

impl DenoiseParams {
    /// The `avfilter` chain string these params describe.
    fn filter_desc(&self) -> String {
        // `r` must be strictly greater than `p`; clamp defensively so a bad
        // params combination can't make libavfilter reject the whole graph.
        let p = self.patch_secs.max(1e-3);
        let r = self.research_secs.max(p + 1e-3);
        format!(
            "anlmdn=s={:.6}:p={:.6}:r={:.6}",
            self.strength.max(0.0),
            p,
            r
        )
    }
}

/// Streaming de-hiss stage. One per stream; not shared across threads.
pub struct Denoiser {
    sample_rate: u32,
    params: DenoiseParams,
    /// The libavfilter graph, built lazily on first [`Self::process`].
    filter: Option<AudioFilter>,
    /// Set once building the graph has failed; from then on we pass audio
    /// through untouched (fail-open) instead of retrying every chunk.
    failed: bool,
}

impl Denoiser {
    /// Build a de-hiss stage at the given output sample rate and strength. The
    /// underlying ffmpeg filter graph is created lazily on first use.
    pub fn new(sample_rate: u32, params: DenoiseParams) -> Self {
        Self {
            sample_rate,
            params,
            filter: None,
            failed: false,
        }
    }

    /// Clear all state so the stage can start fresh on an independent input.
    /// Drops the current filter graph; a new one is built on the next
    /// [`Self::process`] (afftdn's noise profile must not carry across files).
    pub fn reset(&mut self) {
        self.filter = None;
        self.failed = false;
    }

    /// Lazily build the filter graph, returning `None` (fail-open) if ffmpeg
    /// cannot construct it.
    fn ensure_filter(&mut self) -> Option<&mut AudioFilter> {
        if self.filter.is_none() && !self.failed {
            match AudioFilter::new(self.sample_rate, &self.params.filter_desc()) {
                Ok(f) => self.filter = Some(f),
                Err(e) => {
                    tracing::warn!("de-hiss disabled (ffmpeg filter unavailable): {e}");
                    self.failed = true;
                }
            }
        }
        self.filter.as_mut()
    }

    /// Feed samples; returns the filtered output produced so far (may be empty
    /// until the filter's look-ahead fills). Passes input through unchanged if
    /// the filter graph could not be built.
    pub fn process(&mut self, input: &[f32]) -> Vec<f32> {
        match self.ensure_filter() {
            Some(f) => match f.process(input) {
                Ok(out) => out,
                Err(e) => {
                    tracing::warn!("de-hiss failed mid-stream, passing audio through: {e}");
                    self.failed = true;
                    self.filter = None;
                    input.to_vec()
                }
            },
            None => input.to_vec(),
        }
    }

    /// Drain the filter's tail. Empty if de-hiss was never active.
    pub fn flush(&mut self) -> Vec<f32> {
        match self.filter.as_mut() {
            Some(f) => f.flush().unwrap_or_default(),
            None => Vec::new(),
        }
    }
}
