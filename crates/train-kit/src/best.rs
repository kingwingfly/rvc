//! Keeping the best weights a run produced, not merely its last.
//!
//! The per-step mel loss is noisy enough that its minimum is mostly luck, so the
//! score judged is the **mean over a window** of steps. The window cap matters
//! more than the fraction: `total_steps` is the *scheduled* count, and a run
//! stopped by hand — the way these trainers are meant to be used — would
//! otherwise exit before its first window ever closed, leaving the in-loop save
//! dead code.
//!
//! Shared because both adversarial loops here want exactly this, down to the
//! rule for the ragged tail an early stop leaves behind.

use burn::module::AutodiffModule;
use burn::tensor::backend::AutodiffBackend;
use burn_store::ModuleSnapshot;

use crate::{BestMeta, Checkpoint};

/// Windowed best-so-far tracking over one scalar (the mel loss).
///
/// A disabled tracker (`enabled == false`) accepts every call and does nothing,
/// so a loop needs no branch of its own.
pub struct Best {
    /// `None` when best-checkpointing is off.
    ck: Option<Checkpoint>,
    window: usize,
    sum: f32,
    n: usize,
    /// The score to beat — inherited from a previous run when one left a
    /// sidecar, so a fresh process can only *improve* on what is on disk.
    /// Starting from infinity would make every run's first window a clobber.
    score: f32,
    /// The step of a best written *by this run*. An inherited score is not
    /// evidence that this run produced anything, and the closing report is the
    /// only place that distinction is visible.
    at: Option<usize>,
}

impl Best {
    /// Track the best-so-far family beside `out` (`out.best()`).
    /// `window` overrides the derived one; `None` keeps `total_steps / 20`
    /// clamped to `1..=50`.
    pub fn new(out: &Checkpoint, enabled: bool, total_steps: usize, window: Option<usize>) -> Self {
        let ck = enabled.then(|| out.best());
        // A sidecar whose weights are gone is ignored: it would veto every save.
        let prev = ck
            .as_ref()
            .filter(|c| c.generator().exists())
            .and_then(Checkpoint::load_meta);
        if let Some(m) = prev {
            tracing::info!(
                "best so far: mel {:.3} (step {}) from a prior run",
                m.mel,
                m.step
            );
        }
        Self {
            ck,
            // The derived value is a *fraction* of a scheduled run, so a short
            // one collapses it to the clamp's floor of 1 — and a "best" chosen
            // on a single step is the noise this whole mechanism exists to
            // average out. That is why the override exists rather than a wider
            // clamp: only the caller knows whether 200 steps is the whole run
            // or the part of it somebody sat through.
            window: window.unwrap_or((total_steps / 20).clamp(1, 50)),
            sum: 0.0,
            n: 0,
            score: prev.map_or(f32::INFINITY, |m| m.mel),
            at: None,
        }
    }

    /// Feed one step's score, saving when a closed window beats the best.
    pub fn observe<AB, G, D>(
        &mut self,
        step: usize,
        mel: f32,
        ema: Option<&G::InnerModule>,
        live: &G,
        disc: Option<&D>,
    ) where
        AB: AutodiffBackend,
        G: AutodiffModule<AB>,
        G::InnerModule: ModuleSnapshot<AB::InnerBackend>,
        D: AutodiffModule<AB>,
        D::InnerModule: ModuleSnapshot<AB::InnerBackend>,
    {
        if self.ck.is_none() {
            return;
        }
        self.sum += mel;
        self.n += 1;
        if self.n == self.window {
            let mean = self.sum / self.n as f32;
            (self.sum, self.n) = (0.0, 0);
            self.keep(mean, step, ema, live, disc);
        }
    }

    /// Judge the partial window an early stop (or a ragged tail) leaves behind.
    ///
    /// Only when there is enough of it to mean anything: a one-step mean carries
    /// many times the variance of a full window, so a lucky tail would otherwise
    /// unseat a genuinely better best. Under that bar it still stands when
    /// nothing is on disk — a stopped-early run must leave *something*.
    pub fn finish<AB, G, D>(
        &mut self,
        last_step: usize,
        ema: Option<&G::InnerModule>,
        live: &G,
        disc: Option<&D>,
    ) where
        AB: AutodiffBackend,
        G: AutodiffModule<AB>,
        G::InnerModule: ModuleSnapshot<AB::InnerBackend>,
        D: AutodiffModule<AB>,
        D::InnerModule: ModuleSnapshot<AB::InnerBackend>,
    {
        let trustworthy = self.n * 2 >= self.window || self.score.is_infinite();
        if self.ck.is_some() && self.n > 0 && trustworthy {
            let mean = self.sum / self.n as f32;
            self.keep(mean, last_step, ema, live, disc);
        }
    }

    /// Say what this run contributed. A run that contributes no best is
    /// invisible otherwise: state plainly that this one added nothing, and
    /// whether anything is there at all.
    pub fn report(&self) {
        let Some(ck) = &self.ck else {
            return;
        };
        match self.at {
            Some(step) => tracing::info!(
                "best mel {:.3} (step {step}) -> {}",
                self.score,
                ck.generator().display()
            ),
            None if self.score.is_finite() => tracing::info!(
                "no new best this run; kept mel {:.3} in {}",
                self.score,
                ck.generator().display()
            ),
            None => tracing::warn!(
                "no best checkpoint was written (the run was too short to complete a window)"
            ),
        }
    }

    /// Save when `mean` beats the score. The score advances only once the whole
    /// family is on disk, so a failed write cannot block a later minimum from
    /// being saved; the sidecar is written last for the same reason — a later
    /// run must not inherit a best whose weights never landed.
    fn keep<AB, G, D>(
        &mut self,
        mean: f32,
        step: usize,
        ema: Option<&G::InnerModule>,
        live: &G,
        disc: Option<&D>,
    ) where
        AB: AutodiffBackend,
        G: AutodiffModule<AB>,
        G::InnerModule: ModuleSnapshot<AB::InnerBackend>,
        D: AutodiffModule<AB>,
        D::InnerModule: ModuleSnapshot<AB::InnerBackend>,
    {
        let Some(ck) = &self.ck else {
            return;
        };
        if mean.is_nan() || mean >= self.score {
            return;
        }
        // `valid()` copies the weights off the autodiff graph, which is why it
        // happens here and not on every step.
        match ck.save(ema, &live.valid(), disc.map(D::valid).as_ref()) {
            Ok(()) => {
                tracing::info!("best mel {mean:.3} (step {step}) saved");
                self.score = mean;
                self.at = Some(step);
                if let Err(e) = ck.save_meta(BestMeta { mel: mean, step }) {
                    // The weights are the checkpoint; losing the score only
                    // costs the *next* run its memory of what to beat.
                    tracing::warn!("could not record the best score: {e:#}");
                }
            }
            Err(e) => tracing::warn!("could not save best checkpoint: {e:#}"),
        }
    }
}
