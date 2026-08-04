//! Choosing a compute backend at run time.
//!
//! The enum, its aliases and the `auto` rule are [`cli_kit::Backend`], shared
//! with every other binary — adding an engine means reusing that enum, never
//! declaring another one beside it. What is local is one thing: **Seed-VC has no
//! ONNX path, and saying why is more use than a generic refusal.**
//!
//! [`load`] is the whole of the erasure. Each arm builds a
//! [`BurnModel<B>`](crate::model::BurnModel) for one concrete Burn backend, boxes
//! it as a [`Model`], and stops there — so `seedvc-cli` never names a Burn type
//! and the choice is purely run-time, exactly as `tts-cli`'s loader is.

use burn_kit::DeviceSpec;
use cli_kit::Backend;

use crate::error::{Error, Result};
use crate::model::{Model, ModelPaths};

/// Settle `auto` and refuse what this engine cannot run.
///
/// Split out of [`load`] so a caller can ask **before** resolving paths: the
/// refusal costs nothing, while the fetch behind it is a gigabyte, and a cold
/// cache would otherwise download the whole model in order to reject the backend
/// afterwards. Callers that hold the weights already lose nothing by skipping it
/// — [`load`] asks the same question again.
pub fn resolve(backend: Backend) -> Result<Backend> {
    // `false`, and not because the export is merely absent: there is no ONNX
    // export of Seed-VC anywhere, so nothing on disk could ever resolve `auto`
    // to ONNX Runtime. The same argument training passes for the same reason.
    let backend = backend.resolve(false);
    // Not "unsupported yet" and not a build-time absence — which is why this is
    // not one of the `#[cfg]`ed arms in `load`: no exporter for Seed-VC exists,
    // and Burn imports ONNX graphs without being able to emit one, so no
    // `--features` and no rebuild would make it work. `export/` is where that
    // would change.
    if backend == Backend::Onnx {
        return Err(Error::Device(
            "Seed-VC has no ONNX export — `export/` mirrors RVC and GPT-SoVITS only, and Burn \
             reads ONNX graphs without being able to write one. Run it on a Burn backend \
             (`--backend auto|cuda|tch|wgpu`)"
                .into(),
        ));
    }
    Ok(backend)
}

/// Load the four checkpoints onto the chosen backend.
///
/// Naming a backend this build has no code for is an error with a reason; only
/// `auto` substitutes, and it resolves by hardware alone because there is no
/// artefact on disk that could decide it.
pub fn load(
    paths: &ModelPaths<'_>,
    backend: Backend,
    device: DeviceSpec,
) -> Result<Box<dyn Model>> {
    let backend = resolve(backend)?;
    tracing::info!("loading Seed-VC ({backend}, device {device})");

    macro_rules! burn_model {
        ($inner:ty, $device:expr, $name:literal) => {{
            let device = $device;
            let model = burn_kit::guard_init($name, || {
                crate::model::BurnModel::<$inner>::load(paths, &device)
            })??;
            Box::new(model) as Box<dyn Model>
        }};
    }

    let model: Box<dyn Model> = match backend {
        #[cfg(feature = "tch")]
        Backend::Tch => burn_model!(
            burn::backend::LibTorch<f32>,
            burn_kit::libtorch_device(device)?,
            "tch"
        ),
        #[cfg(feature = "cuda")]
        Backend::Cuda => burn_model!(burn::backend::Cuda, burn_kit::cuda_device(device)?, "cuda"),
        #[cfg(feature = "wgpu")]
        Backend::Wgpu => burn_model!(burn::backend::Wgpu, burn_kit::wgpu_device(device)?, "wgpu"),
        Backend::Auto | Backend::Onnx => unreachable!("resolved and refused above"),
        // Only reachable on a `--no-default-features` build, where the arm that
        // would have handled it was `#[cfg]`ed away.
        #[allow(unreachable_patterns)]
        other => return Err(Error::Device(other.unavailable().to_string())),
    };
    Ok(model)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    /// `--backend onnx` has to fail before it touches a file, and the message has
    /// to say *why* rather than "unsupported" — a user who reads "not compiled
    /// in" will go looking for a feature flag that cannot exist.
    #[test]
    fn onnx_is_refused_with_a_reason() {
        let missing = Path::new("/nonexistent");
        let paths = ModelPaths {
            dit: missing,
            campplus: missing,
            bigvgan: missing,
            content: missing,
        };
        // Matched rather than `expect_err`ed: a boxed trait object is not
        // `Debug`, and making it one would be a bound on every implementation
        // for the sake of one test.
        let err = match load(&paths, Backend::Onnx, DeviceSpec::Auto) {
            Ok(_) => panic!("ONNX Runtime cannot run Seed-VC, and nothing was there to load"),
            Err(e) => e.to_string(),
        };
        assert!(err.contains("no ONNX export"), "{err}");
        assert!(err.contains("--backend auto|cuda|tch|wgpu"), "{err}");
    }
}
