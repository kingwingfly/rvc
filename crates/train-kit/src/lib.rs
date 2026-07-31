//! Training scaffolding that knows about no model.
//!
//! Everything a training loop needs and no loop should own: what a checkpoint is
//! on disk, how a weight EMA is kept, how gradients are folded across
//! micro-batches and devices, how the learning rate decays, which weights were
//! the best, and the live dashboard. All of it is generic over the module being
//! trained, so a GAN over a vocoder and a cross-entropy loop over a transformer
//! share it.
//!
//! What is *not* here is the shape of a step. A GAN needs two models, two
//! optimizers and a discriminator update wedged between two backward passes; a
//! cross-entropy loop needs one of each. Forcing one skeleton over both would
//! cost more than the duplication it removed, so each trainer still writes its
//! own loop — out of these parts.

mod best;
mod checkpoint;
mod dashboard;
mod devices;
mod ema;
mod grad;
mod misc;
mod rng;
mod schedule;

pub use best::Best;
pub use checkpoint::{BestMeta, Checkpoint, ensure_absent};
pub use dashboard::Dashboard;
pub use devices::distinct;
pub use ema::{ema_update, materialize};
pub use grad::accumulate;
pub use misc::{human, scalar};
pub use rng::Rng;
pub use schedule::Schedule;
