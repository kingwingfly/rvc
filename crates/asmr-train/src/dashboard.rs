//! Live training dashboard using Burn's TUI renderer.
//!
//! Our GAN loop doesn't fit Burn's `Learner`, so instead of adopting the whole
//! learner machinery we drive the same [`TuiMetricsRendererWrapper`] Burn uses
//! for its dashboard directly: register the loss metrics once, then push a
//! numeric value + progress each step. The renderer runs its own UI thread and
//! owns the terminal, so the caller must route logs to a file (not stderr) to
//! keep the display clean.
//!
//! The renderer shares a [`Interrupter`] with the dashboard — pressing `q`
//! (quit) in the TUI flips it, which [`Dashboard::interrupted`] reports so the
//! training loop can stop early and save. When the TUI is disabled (no TTY, or
//! `--no-tui`), everything here is a no-op and training logs to stderr as usual.

use std::sync::Arc;

use burn::data::dataloader::Progress;
use burn::train::Interrupter;
use burn::train::metric::{
    MetricAttributes, MetricDefinition, MetricEntry, MetricId, NumericAttributes, NumericEntry,
    SerializedEntry,
};
use burn::train::renderer::tui::TuiMetricsRendererWrapper;
use burn::train::renderer::{
    MetricState, MetricsRenderer, MetricsRendererTraining, TrainingProgress,
};

/// The three losses plotted in the dashboard.
struct Metric {
    id: MetricId,
    /// Display name (also the plot/group label).
    name: &'static str,
}

/// A handle over Burn's TUI renderer (or nothing, when the TUI is off).
pub struct Dashboard {
    renderer: Option<TuiMetricsRendererWrapper>,
    interrupter: Interrupter,
    metrics: Vec<Metric>,
    total_steps: usize,
    steps_per_epoch: usize,
    total_epochs: usize,
}

impl Dashboard {
    /// Build the dashboard. When `enabled`, spawn Burn's TUI (it takes over the
    /// terminal) and register the loss metrics; otherwise this is inert.
    pub fn new(
        enabled: bool,
        total_steps: usize,
        steps_per_epoch: usize,
        total_epochs: usize,
    ) -> Self {
        let interrupter = Interrupter::new();
        let metrics = vec![
            Metric { id: metric_id("g_loss"), name: "g_loss" },
            Metric { id: metric_id("d_loss"), name: "d_loss" },
            Metric { id: metric_id("mel_loss"), name: "mel_loss" },
        ];

        let renderer = if enabled {
            let mut r = TuiMetricsRendererWrapper::new(interrupter.clone(), None);
            for m in &metrics {
                // Registration must precede any update — the renderer looks the
                // definition up (and panics if missing) when a value arrives.
                r.register_metric(MetricDefinition {
                    metric_id: m.id.clone(),
                    name: m.name.to_string(),
                    description: None,
                    attributes: MetricAttributes::Numeric(NumericAttributes {
                        unit: None,
                        higher_is_better: false,
                    }),
                });
            }
            Some(r)
        } else {
            None
        };

        Self { renderer, interrupter, metrics, total_steps, steps_per_epoch, total_epochs }
    }

    /// Whether the TUI is driving the display (vs. plain stderr logging).
    pub fn is_active(&self) -> bool {
        self.renderer.is_some()
    }

    /// True once the user asked to stop early (pressed `q` in the TUI).
    pub fn interrupted(&self) -> bool {
        self.interrupter.should_stop()
    }

    /// Push the current step's losses and progress to the dashboard.
    pub fn update(&mut self, step: usize, g: f32, d: f32, mel: f32) {
        let Some(r) = self.renderer.as_mut() else {
            return;
        };
        let values = [g, d, mel];
        for (m, &v) in self.metrics.iter().zip(values.iter()) {
            let entry = MetricEntry::new(
                m.id.clone(),
                SerializedEntry::new(format!("{v:.3}"), v.to_string()),
            );
            r.update_train(MetricState::Numeric(entry, NumericEntry::Value(v as f64)));
        }

        let epoch = (step / self.steps_per_epoch.max(1)).min(self.total_epochs.saturating_sub(1));
        r.render_train(
            TrainingProgress {
                progress: Some(Progress {
                    items_processed: step + 1,
                    items_total: self.total_steps,
                }),
                global_progress: Progress {
                    items_processed: epoch,
                    items_total: self.total_epochs,
                },
                iteration: Some(step),
            },
            Vec::new(),
        );
    }

    /// Close the dashboard, restoring the terminal.
    pub fn finish(&mut self) {
        if let Some(mut r) = self.renderer.take() {
            let _ = r.on_train_end(None);
            // Dropping the wrapper joins its UI thread and restores the screen.
            drop(r);
        }
    }
}

/// Build a stable metric id from its name (Burn keys entries by this).
fn metric_id(name: &str) -> MetricId {
    MetricId::new(Arc::new(name.to_string()))
}
