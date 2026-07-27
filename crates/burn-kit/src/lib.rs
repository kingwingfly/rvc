//! Burn plumbing shared by every model in the toolkit, and tied to none of them.
//!
//! Two jobs that recur once there is more than one network in a workspace:
//!
//! - [`device`] — turn one `--device` spelling into whichever backend's device
//!   type is being asked for, so voice conversion, recognition and synthesis
//!   cannot disagree about what `auto` means.
//! - [`store`] — read a PyTorch or safetensors checkpoint, remap its key names
//!   onto a Burn module tree, and upcast fp16 to fp32.
//!
//! Nothing here knows what a model is. That is the point: engines are siblings,
//! so anything two of them need lives somewhere neither owns.

pub mod device;
#[cfg(feature = "store")]
pub mod store;

#[cfg(feature = "cuda")]
pub use device::cuda_device;
#[cfg(feature = "tch")]
pub use device::libtorch_device;
#[cfg(feature = "wgpu")]
pub use device::wgpu_device;
// The probes stay unconditional: without LibTorch linked they answer `None`,
// which is a different and useful answer from "no GPU".
pub use device::{
    AutoBackend, DeviceSpec, auto_backend, guard_init, libtorch_gpu, libtorch_has_cuda,
    visible_cuda_devices,
};

/// Errors from resolving a compute device.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// The requested device is unavailable, or the chosen backend cannot drive
    /// it. Always a message rather than a panic: asking for hardware you don't
    /// have is a user mistake, not a bug.
    #[error("{0}")]
    Device(String),
}

/// Convenience alias.
pub type Result<T> = std::result::Result<T, Error>;
