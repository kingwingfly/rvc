//! Live training dashboard using Burn's TUI renderer.
//!
//! Our GAN loop doesn't fit Burn's `Learner`, so instead of adopting the whole
//! learner machinery we drive the same [`TuiMetricsRendererWrapper`] Burn uses
//! for its dashboard directly: register the metrics once, then push values +
//! progress each step. The renderer runs its own UI thread and owns the
//! terminal, so the caller must route logs to a file (not stderr) to keep the
//! display clean.
//!
//! Two kinds of metric are shown:
//! - **plotted** (`MetricState::Numeric`): the `g`/`d`/`mel` losses and the
//!   learning rate, so their curves are visible over time;
//! - **text-only** (`MetricState::Generic`): CPU and GPU usage, reusing Burn's
//!   own [`CpuUse`] (sysinfo) and [`CudaMetric`] (NVML) system probes. These are
//!   status readouts, not something worth a plot.
//!
//! The renderer shares a [`Interrupter`] with the dashboard — pressing `q`
//! (quit) in the TUI flips it, which [`Dashboard::interrupted`] reports so the
//! training loop can stop early and save. When the TUI is disabled (no TTY, or
//! `--no-tui`), everything here is a no-op and training logs to stderr as usual.

use std::sync::Arc;

use burn::data::dataloader::Progress;
use burn::train::Interrupter;
use burn::train::metric::{
    CpuUse, CudaMetric, Metric, MetricAttributes, MetricDefinition, MetricEntry, MetricId,
    MetricMetadata, NumericAttributes, NumericEntry, SerializedEntry,
};
use burn::train::renderer::tui::TuiMetricsRendererWrapper;
use burn::train::renderer::{
    MetricState, MetricsRenderer, MetricsRendererTraining, TrainingProgress,
};

/// A plotted (numeric) metric: the three losses and the learning rate.
struct PlotMetric {
    id: MetricId,
    /// Display name (also the plot/group label).
    name: &'static str,
    /// Format the value in scientific notation (for the tiny learning rate).
    sci: bool,
}

/// A handle over Burn's TUI renderer (or nothing, when the TUI is off).
pub struct Dashboard {
    renderer: Option<TuiMetricsRendererWrapper>,
    interrupter: Interrupter,
    /// Plotted numeric metrics, in the order [`Dashboard::update`] supplies them.
    plots: Vec<PlotMetric>,
    /// Burn's system probes, shown as text (`None` when the TUI is off).
    cpu: Option<(MetricId, CpuUse)>,
    gpu: Option<(MetricId, CudaMetric)>,
    total_steps: usize,
    steps_per_epoch: usize,
    total_epochs: usize,
}

impl Dashboard {
    /// Build the dashboard. When `enabled`, spawn Burn's TUI (it takes over the
    /// terminal) and register the metrics; otherwise this is inert.
    pub fn new(
        enabled: bool,
        total_steps: usize,
        steps_per_epoch: usize,
        total_epochs: usize,
    ) -> Self {
        let interrupter = Interrupter::new();
        let plots = vec![
            PlotMetric {
                id: metric_id("g_loss"),
                name: "g_loss",
                sci: false,
            },
            PlotMetric {
                id: metric_id("d_loss"),
                name: "d_loss",
                sci: false,
            },
            PlotMetric {
                id: metric_id("mel_loss"),
                name: "mel_loss",
                sci: false,
            },
            PlotMetric {
                id: metric_id("lr"),
                name: "lr",
                sci: true,
            },
        ];

        let (renderer, cpu, gpu) = if enabled {
            let mut r = TuiMetricsRendererWrapper::new(interrupter.clone(), None);
            for m in &plots {
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

            // CPU (sysinfo) and GPU (NVML) usage, registered as attribute-less
            // (text) metrics so the TUI shows them without a plot.
            let cpu = CpuUse::new();
            let cpu_id = metric_id("cpu");
            register_generic(&mut r, &cpu_id, "CPU Usage");
            let gpu = CudaMetric::new();
            let gpu_id = metric_id("gpu");
            register_generic(&mut r, &gpu_id, "GPU");

            (Some(r), Some((cpu_id, cpu)), Some((gpu_id, gpu)))
        } else {
            (None, None, None)
        };

        Self {
            renderer,
            interrupter,
            plots,
            cpu,
            gpu,
            total_steps,
            steps_per_epoch,
            total_epochs,
        }
    }

    /// True once the user asked to stop early (pressed `q` in the TUI).
    pub fn interrupted(&self) -> bool {
        self.interrupter.should_stop()
    }

    /// Push the current step's losses, learning rate, system usage, and progress.
    pub fn update(&mut self, step: usize, g: f32, d: f32, mel: f32, lr: f64) {
        let Some(r) = self.renderer.as_mut() else {
            return;
        };

        let values = [g as f64, d as f64, mel as f64, lr];
        for (m, &v) in self.plots.iter().zip(values.iter()) {
            let formatted = if m.sci {
                format!("{v:.3e}")
            } else {
                format!("{v:.3}")
            };
            let entry =
                MetricEntry::new(m.id.clone(), SerializedEntry::new(formatted, v.to_string()));
            r.update_train(MetricState::Numeric(entry, NumericEntry::Value(v)));
        }

        let epoch = (step / self.steps_per_epoch.max(1)).min(self.total_epochs.saturating_sub(1));

        // System probes: Burn's metrics ignore the item/metadata and read the
        // machine, so a placeholder metadata is fine.
        let meta = MetricMetadata {
            progress: Progress {
                items_processed: step + 1,
                items_total: self.total_steps,
            },
            global_progress: Progress {
                items_processed: epoch,
                items_total: self.total_epochs,
            },
            iteration: Some(step),
            lr: Some(lr),
        };
        if let Some((id, cpu)) = self.cpu.as_mut() {
            let serialized = cpu.update(&(), &meta);
            r.update_train(MetricState::Generic(MetricEntry::new(
                id.clone(),
                serialized,
            )));
        }
        if let Some((id, gpu)) = self.gpu.as_mut() {
            let serialized = gpu.update(&(), &meta);
            r.update_train(MetricState::Generic(MetricEntry::new(
                id.clone(),
                serialized,
            )));
        }

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

/// Register a text-only (attribute-less) metric with the renderer.
fn register_generic(r: &mut TuiMetricsRendererWrapper, id: &MetricId, name: &str) {
    r.register_metric(MetricDefinition {
        metric_id: id.clone(),
        name: name.to_string(),
        description: None,
        attributes: MetricAttributes::None,
    });
}

/// Build a stable metric id from its name (Burn keys entries by this).
fn metric_id(name: &str) -> MetricId {
    MetricId::new(Arc::new(name.to_string()))
}
