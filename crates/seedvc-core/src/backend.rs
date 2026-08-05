//! Choosing a compute backend at run time.
//!
//! The enum, its aliases and the `auto` rule are [`cli_kit::Backend`], shared
//! with every other binary — adding an engine means reusing that enum, never
//! declaring another one beside it. What is local is whether the six ONNX graphs
//! are on disk, because that is what decides how `auto` resolves: Seed-VC now
//! has an export (`export/export_seedvc.py` writes it), so the question is no
//! longer *whether* ONNX Runtime can run it but *where the graphs are*.
//!
//! [`load`] is the whole of the erasure for the Burn path. Each arm builds a
//! [`BurnModel<B>`](crate::model::BurnModel) for one concrete Burn backend, boxes
//! it as a [`Model`], and stops there — so `seedvc-cli` never names a Burn type
//! and the choice is purely run-time, exactly as `tts-cli`'s loader is. ONNX
//! Runtime is a second implementation of the same trait, loaded by the CLI
//! directly when `--onnx` names an export; this module only decides whether that
//! request is legal.

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
///
/// `has_onnx` is whether the caller was pointed at an export directory
/// (`--onnx`). `auto` resolves to ONNX Runtime only when one was, and an
/// explicit `--backend onnx` is accepted only then too — otherwise the request
/// is refused with the reason below.
pub fn resolve(backend: Backend, has_onnx: bool) -> Result<Backend> {
    // The export is the one artefact on disk that could decide `auto`: the six
    // graphs `export/export_seedvc.py` writes. Without one, `auto` falls back to
    // hardware, exactly as it did before the export existed.
    let backend = backend.resolve(has_onnx);
    // Only reachable when `has_onnx` is false — a true one accepts it above — and
    // the reason has to say the graphs *exist*, because the user asked for a
    // runtime that can now run them. "Not supported" would send them looking for
    // a feature flag that cannot exist; `--onnx` is the thing that is missing.
    if backend == Backend::Onnx && !has_onnx {
        return Err(Error::Device(
            "Seed-VC has an ONNX export now (`export/export_seedvc.py` writes it), but no \
             directory was given — point `--onnx` at the six graphs, or run on a Burn backend \
             (`--backend auto|cuda|tch|wgpu`)"
                .into(),
        ));
    }
    Ok(backend)
}

/// Load the four checkpoints onto the chosen backend.
///
/// Naming a backend this build has no code for is an error with a reason; only
/// `auto` substitutes, and it resolves by hardware alone because this function
/// is the Burn path — ONNX is a second loader, not an arm of this one. Callers
/// that can see an export directory pass `has_onnx` to [`resolve`] themselves.
pub fn load(
    paths: &ModelPaths<'_>,
    backend: Backend,
    device: DeviceSpec,
) -> Result<Box<dyn Model>> {
    let backend = resolve(backend, false)?;
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

    /// `--backend onnx` without an export directory has to fail before it touches
    /// a file, and the message has to say *why* rather than "unsupported" — a
    /// user who reads "not compiled in" will go looking for a feature flag that
    /// cannot exist, when the missing thing is the `--onnx` directory.
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
            Ok(_) => panic!("--backend onnx was accepted with no --onnx directory"),
            Err(e) => e.to_string(),
        };
        assert!(err.contains("--onnx"), "{err}");
        assert!(err.contains("export/export_seedvc.py"), "{err}");
        assert!(err.contains("--backend auto|cuda|tch|wgpu"), "{err}");
    }

    /// The same request is accepted when the caller was pointed at an export
    /// directory — that is what `--onnx` does, and `auto` must pick ONNX Runtime
    /// when one is given.
    #[test]
    fn onnx_is_accepted_with_an_export_dir() {
        // `matches!` rather than `assert_eq!`: the error type is not `PartialEq`,
        // and it has no reason to be for one test.
        assert!(matches!(resolve(Backend::Onnx, true), Ok(Backend::Onnx)));
        assert!(matches!(resolve(Backend::Auto, true), Ok(Backend::Onnx)));
        // Without one, `auto` stays on hardware — the export is the whole
        // reason it could ever have been ONNX.
        assert_ne!(resolve(Backend::Auto, false).unwrap(), Backend::Onnx);
    }
}
