//! Which compute backend and device the native Burn generator runs on.
//!
//! [`DeviceSpec`] is a plain parsed value with no Burn types, so it is available
//! even without a compute backend compiled in — the CLI needs it either way.
//!
//! One `--device` spelling drives every backend; each resolver turns it into that
//! backend's device type, rejecting what it can't do with a plain error. Shared
//! with `rvc-train` so `convert` and `train` agree on what `auto` means. `auto`
//! reaches CPU only as a last resort — the decoder is far slower than realtime
//! there.

use crate::error::{Result, VcError};

/// A `--device` request, independent of which backend will honour it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DeviceSpec {
    /// The fastest device the chosen backend can see.
    #[default]
    Auto,
    /// Host CPU. LibTorch or WebGPU — the CubeCL/CUDA backend has no CPU device.
    Cpu,
    /// GPU by index (`gpu:1`). Not `Cuda`: two of the three backends aren't
    /// CUDA-specific. `cuda:N` stays accepted because it's what users type.
    Gpu(usize),
    /// Apple Metal Performance Shaders. LibTorch only.
    Mps,
    /// Vulkan. LibTorch only, and absent from official LibTorch builds.
    Vulkan,
}

impl std::str::FromStr for DeviceSpec {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        let s = s.trim().to_ascii_lowercase();
        match s.as_str() {
            "auto" => Ok(Self::Auto),
            "cpu" => Ok(Self::Cpu),
            "cuda" | "gpu" => Ok(Self::Gpu(0)),
            "mps" | "metal" => Ok(Self::Mps),
            "vulkan" => Ok(Self::Vulkan),
            other => match other.split_once(':') {
                Some(("cuda" | "gpu", n)) => n.parse().map(Self::Gpu).map_err(|_| {
                    format!("`{other}`: expected an index after `gpu:`, e.g. `gpu:1`")
                }),
                _ => Err(format!(
                    "unknown device `{other}` \
                     (expected auto, cpu, gpu, gpu:N, cuda, cuda:N, mps or vulkan)"
                )),
            },
        }
    }
}

impl std::fmt::Display for DeviceSpec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Auto => f.write_str("auto"),
            Self::Cpu => f.write_str("cpu"),
            Self::Gpu(n) => write!(f, "gpu:{n}"),
            Self::Mps => f.write_str("mps"),
            Self::Vulkan => f.write_str("vulkan"),
        }
    }
}

/// How many CUDA devices LibTorch sees, or `None` when LibTorch isn't linked.
///
/// The only free probe available: it's a driver query that creates no context,
/// unlike `Cuda::is_available` or CubeCL's equivalent (which builds a client on
/// device 0 to answer). CubeCL has none, so it borrows this one.
pub fn visible_cuda_devices() -> Option<usize> {
    #[cfg(feature = "tch")]
    {
        Some(tch::Cuda::device_count().max(0) as usize)
    }
    #[cfg(not(feature = "tch"))]
    {
        None
    }
}

/// Whether the linked LibTorch was *built* with CUDA (`None` if not linked).
///
/// Distinct from [`visible_cuda_devices`], and the distinction is what `auto`
/// needs: a CPU-only LibTorch seeing no GPU says nothing about the machine,
/// whereas a CUDA-capable one seeing none means there is no GPU to find.
pub fn libtorch_has_cuda() -> Option<bool> {
    #[cfg(feature = "tch")]
    {
        Some(tch::utils::has_cuda())
    }
    #[cfg(not(feature = "tch"))]
    {
        None
    }
}

/// A compute backend, named without reference to Burn types so both the
/// inference and training paths can map it onto their own enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AutoBackend {
    Cuda,
    LibTorch,
    Wgpu,
}

/// What `auto` should run on, decided once so inference and training can't drift
/// apart. Only ever returns a backend this build actually contains, and never
/// errors — `auto` is a request for whatever works.
///
/// Preference order: LibTorch on a GPU (measured ~9x faster than CubeCL here),
/// CubeCL/CUDA, WebGPU (portable, and the only one that reaches AMD/Intel/Apple),
/// then LibTorch on CPU, which is the sole CPU device on offer.
pub fn auto_backend() -> AutoBackend {
    let (tch, cuda, wgpu) = (
        cfg!(feature = "tch"),
        cfg!(feature = "cuda"),
        cfg!(feature = "wgpu"),
    );
    // `None` when LibTorch isn't linked, in which case nothing here can probe for
    // a GPU and the CUDA/WebGPU paths have to just try.
    let seen = visible_cuda_devices();

    if tch && seen.is_some_and(|n| n > 0) {
        return AutoBackend::LibTorch;
    }
    // A CPU-only LibTorch reporting zero devices says nothing about the machine,
    // so CubeCL still deserves a try; `guard_init` turns its panic into an error.
    let cuda_plausible = seen.is_none() || libtorch_has_cuda() == Some(false);
    if cuda && cuda_plausible {
        return AutoBackend::Cuda;
    }
    // A CUDA-capable LibTorch that sees no device means no NVIDIA GPU — but
    // WebGPU may still find a Vulkan/Metal one.
    if wgpu {
        return AutoBackend::Wgpu;
    }
    if tch {
        return AutoBackend::LibTorch;
    }
    // Nothing compiled in that can run: say CUDA and let the caller report it.
    AutoBackend::Cuda
}

/// Resolve a spec for the CubeCL/CUDA backend, whose only device kind is CUDA.
#[cfg(feature = "cuda")]
pub fn cuda_device(spec: DeviceSpec) -> Result<burn::backend::cuda::CudaDevice> {
    use burn::backend::cuda::CudaDevice;

    let index = match spec {
        DeviceSpec::Auto => 0,
        DeviceSpec::Gpu(n) => n,
        other => {
            return Err(VcError::Device(format!(
                "the cuda backend runs only on CUDA devices, not `{other}` \
                 — try `--backend tch --device {other}`, or `--backend wgpu`"
            )));
        }
    };
    // Only meaningful when LibTorch is also linked; when it is, it spares the
    // user a cubecl panic several seconds into loading.
    if let Some(n) = visible_cuda_devices() {
        if n == 0 {
            return Err(VcError::Device(
                "no CUDA device is available (no NVIDIA driver or GPU visible) \
                 — try `--backend tch --device cpu`, or `--backend onnx`"
                    .into(),
            ));
        }
        if index >= n {
            return Err(VcError::Device(format!(
                "device gpu:{index} requested but only {n} CUDA device(s) are visible \
                 (valid: gpu:0..gpu:{})",
                n - 1
            )));
        }
    }
    Ok(CudaDevice::new(index))
}

/// Resolve a spec for LibTorch, probing what it can actually see.
///
/// The only place `LibTorchDevice::Cuda` is constructed, deliberately: on a
/// CPU-only LibTorch that variant is a hard panic baked in by `burn-tch`'s build
/// script, and the count check below is what keeps it unreachable.
#[cfg(feature = "tch")]
pub fn libtorch_device(spec: DeviceSpec) -> Result<burn::backend::libtorch::LibTorchDevice> {
    use burn::backend::libtorch::LibTorchDevice;

    let cuda_n = visible_cuda_devices().unwrap_or(0);
    match spec {
        // NB: `LibTorchDevice::default()` is Cpu, so `auto` has to probe. Falling
        // back to it silently would look like the GPU path but run ~100x slower.
        DeviceSpec::Auto => {
            if cuda_n > 0 {
                Ok(LibTorchDevice::Cuda(0))
            } else if tch::utils::has_mps() {
                Ok(LibTorchDevice::Mps)
            } else if tch::utils::has_vulkan() {
                Ok(LibTorchDevice::Vulkan)
            } else {
                tracing::warn!(
                    "no GPU visible to LibTorch; falling back to CPU, \
                     where the decoder is far slower than realtime"
                );
                Ok(LibTorchDevice::Cpu)
            }
        }
        DeviceSpec::Gpu(n) if cuda_n == 0 => Err(VcError::Device(format!(
            "device gpu:{n} requested but LibTorch sees no CUDA device — either this \
             LibTorch is a CPU-only build, or no NVIDIA driver/GPU is present. \
             Use `--device cpu`, `--backend wgpu`, or `--backend cuda`"
        ))),
        DeviceSpec::Gpu(n) if n >= cuda_n => Err(VcError::Device(format!(
            "device gpu:{n} requested but LibTorch sees {cuda_n} CUDA device(s) \
             (valid: gpu:0..gpu:{})",
            cuda_n - 1
        ))),
        DeviceSpec::Gpu(n) => Ok(LibTorchDevice::Cuda(n)),
        DeviceSpec::Mps if !tch::utils::has_mps() => Err(VcError::Device(
            "this LibTorch has no Metal (MPS) support".into(),
        )),
        DeviceSpec::Vulkan if !tch::utils::has_vulkan() => Err(VcError::Device(
            "this LibTorch has no Vulkan support (official builds ship without it)".into(),
        )),
        DeviceSpec::Mps => Ok(LibTorchDevice::Mps),
        DeviceSpec::Vulkan => Ok(LibTorchDevice::Vulkan),
        DeviceSpec::Cpu => Ok(LibTorchDevice::Cpu),
    }
}

/// Resolve a spec for WebGPU.
///
/// wgpu has no cheap adapter probe, so a bad index surfaces as a panic caught by
/// [`guard_init`] rather than a count check. `vulkan`/`mps` name graphics APIs —
/// which is what wgpu picks between anyway — so both mean "the default adapter".
#[cfg(feature = "wgpu")]
pub fn wgpu_device(spec: DeviceSpec) -> Result<burn::backend::wgpu::WgpuDevice> {
    use burn::backend::wgpu::WgpuDevice;

    Ok(match spec {
        // `DefaultDevice` lets wgpu rank adapters (discrete over integrated).
        DeviceSpec::Auto | DeviceSpec::Vulkan | DeviceSpec::Mps => WgpuDevice::DefaultDevice,
        DeviceSpec::Gpu(n) => WgpuDevice::DiscreteGpu(n),
        DeviceSpec::Cpu => WgpuDevice::Cpu,
    })
}

/// Run `f`, turning a backend panic into a device error.
///
/// CubeCL aborts instead of returning when there's no driver, and LibTorch's
/// `TORCH_CHECK`s arrive as panics — neither is catchable otherwise. Wording is
/// neutral because callers wrap spans from "load a model" to "a whole training
/// run".
pub fn guard_init<T>(backend: &str, f: impl FnOnce() -> T) -> Result<T> {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)).map_err(|e| {
        let msg = e
            .downcast_ref::<String>()
            .cloned()
            .or_else(|| e.downcast_ref::<&str>().map(|s| (*s).to_string()))
            .unwrap_or_else(|| "unknown panic".into());
        VcError::Device(format!("the {backend} backend aborted: {msg}"))
    })
}

#[cfg(test)]
mod tests {
    use super::DeviceSpec;

    #[test]
    fn every_accepted_spelling_parses() {
        for (text, want) in [
            ("auto", DeviceSpec::Auto),
            ("  AUTO ", DeviceSpec::Auto),
            ("cpu", DeviceSpec::Cpu),
            ("cuda", DeviceSpec::Gpu(0)),
            ("gpu", DeviceSpec::Gpu(0)),
            ("cuda:2", DeviceSpec::Gpu(2)),
            ("GPU:3", DeviceSpec::Gpu(3)),
            ("mps", DeviceSpec::Mps),
            ("metal", DeviceSpec::Mps),
            ("vulkan", DeviceSpec::Vulkan),
        ] {
            assert_eq!(text.parse(), Ok(want), "parsing {text:?}");
        }
    }

    #[test]
    fn a_bad_spelling_names_the_alternatives() {
        // The message is the whole point: `--device` is free-form text, so a typo
        // has to say what was expected rather than just refusing.
        for bad in ["", "cuda:", "cuda:x", "cuda:-1", "rocm", "cuda:1:2"] {
            let err = bad.parse::<DeviceSpec>().unwrap_err();
            assert!(
                err.contains("gpu:") || err.contains("expected"),
                "{bad:?} gave an unhelpful message: {err}"
            );
        }
    }

    #[test]
    fn display_round_trips_through_parse() {
        // `--device` values get echoed into logs and error text, so the printed
        // form has to be one the parser would accept back.
        for spec in [
            DeviceSpec::Auto,
            DeviceSpec::Cpu,
            DeviceSpec::Gpu(0),
            DeviceSpec::Gpu(7),
            DeviceSpec::Mps,
            DeviceSpec::Vulkan,
        ] {
            assert_eq!(spec.to_string().parse(), Ok(spec));
        }
    }
}
