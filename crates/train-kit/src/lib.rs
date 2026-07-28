//! Training scaffolding that knows about no model.
//!
//! Everything a training loop needs and no loop should own: what a checkpoint is
//! on disk, how a weight EMA is kept, how gradients are folded across
//! micro-batches and devices, and the live dashboard. All of it is generic over
//! the module being trained, so a GAN over a vocoder and a cross-entropy loop
//! over a transformer share it.

mod checkpoint;
mod dashboard;
mod ema;
mod grad;
mod misc;

pub use checkpoint::{BestMeta, Checkpoint};
pub use dashboard::Dashboard;
pub use ema::{ema_update, materialize};
pub use grad::accumulate;
pub use misc::{human, scalar};
