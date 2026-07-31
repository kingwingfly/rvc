//! The learning-rate decay and the weight-EMA window, which every loop here
//! shares.
//!
//! Both are expressed as fractions of the **whole scheduled run** rather than
//! per epoch or per step, so the same numbers mean the same thing whether a
//! fine-tune is two epochs or two hundred. That is the property worth keeping in
//! one place: a schedule that quietly depended on the epoch count would make
//! every `--epochs` change also a learning-rate change.

/// An exponential LR decay from `lr` to `lr * lr_final` across the run, plus the
/// per-step EMA decay derived from a smoothing window.
#[derive(Debug, Clone, Copy)]
pub struct Schedule {
    base_lr: f64,
    lr_final: f64,
    total_steps: usize,
    /// Per-step EMA decay (`keep`), or `0.0` when the EMA is disabled.
    pub ema_decay: f64,
    /// The smoothing window in steps `ema_decay` was derived from — worth
    /// logging, since it is the number a reader can sanity-check against the run
    /// length. `0` when the EMA is off.
    pub ema_window: f64,
}

impl Schedule {
    /// `lr_final` is the end-of-run rate as a fraction of `base_lr` (`1.0`
    /// disables decay); `ema_frac` is the EMA window as a fraction of the run
    /// (`0.0` disables the EMA).
    pub fn new(base_lr: f64, lr_final: f64, ema_frac: f64, total_steps: usize) -> Self {
        // A window under one step cannot average anything, so it reads as "off"
        // rather than as an EMA that simply copies the live weights. The cap
        // keeps a very long run's decay from rounding to 1.0 and freezing the
        // average at its initial weights.
        let ema_window = (ema_frac * total_steps as f64).max(0.0);
        let ema_decay = match ema_window >= 1.0 {
            true => (1.0 - 1.0 / ema_window).min(0.9999),
            false => 0.0,
        };
        Self {
            base_lr,
            lr_final,
            total_steps,
            ema_decay,
            ema_window,
        }
    }

    /// The learning rate at `step`: `base_lr` at step 0, `base_lr * lr_final` at
    /// the last scheduled step.
    pub fn lr(&self, step: usize) -> f64 {
        let progress = step as f64 / self.total_steps.max(1) as f64;
        self.base_lr * self.lr_final.powf(progress)
    }

    /// The end-of-run learning rate, for the line a run prints about itself.
    pub fn final_lr(&self) -> f64 {
        self.base_lr * self.lr_final
    }
}

#[cfg(test)]
mod tests {
    use super::Schedule;

    #[test]
    fn the_decay_spans_the_run_whatever_its_length() {
        // Same start, same end, whatever the step count — that is the whole
        // point of scheduling by fraction rather than per step.
        for total in [10usize, 1000] {
            let s = Schedule::new(1e-4, 0.1, 0.1, total);
            assert!((s.lr(0) - 1e-4).abs() < 1e-12);
            let last = s.lr(total - 1);
            assert!(last > s.final_lr() && last < 1e-4, "{last}");
        }
        // `lr_final == 1.0` is a flat rate.
        let flat = Schedule::new(1e-4, 1.0, 0.0, 100);
        assert!((flat.lr(0) - flat.lr(99)).abs() < 1e-12);
    }

    #[test]
    fn an_ema_window_under_one_step_is_off() {
        // Not "an EMA that copies the live weights": that would write a raw twin
        // and claim an average nothing was averaged into.
        assert_eq!(Schedule::new(1e-4, 0.1, 0.1, 5).ema_decay, 0.0);
        assert_eq!(Schedule::new(1e-4, 0.1, 0.0, 1000).ema_decay, 0.0);
        assert!(Schedule::new(1e-4, 0.1, 0.1, 1000).ema_decay > 0.98);
        // However long the run, the average must still move.
        assert!(Schedule::new(1e-4, 0.1, 1.0, 100_000_000).ema_decay <= 0.9999);
    }
}
