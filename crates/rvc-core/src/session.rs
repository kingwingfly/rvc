//! Thin wrapper over an `ort` [`Session`] with a GPU-first execution provider
//! stack.
//!
//! The ONNX Runtime shared library is located via the `ORT_DYLIB_PATH`
//! environment variable (the `load-dynamic` feature); we never bundle or
//! download it. The CUDA execution provider is registered first with a CPU
//! fallback, so an appropriately built onnxruntime uses the 6 GB GPU and
//! otherwise degrades gracefully to CPU.

use std::path::Path;

use ort::execution_providers::{CPUExecutionProvider, CUDAExecutionProvider};
use ort::session::Session;
use ort::value::Tensor;

use crate::error::{Result, VcError};

/// Build a session for `path` with CUDA-then-CPU execution providers.
pub fn build_session(path: impl AsRef<Path>) -> Result<Session> {
    let path = path.as_ref();
    if !path.exists() {
        return Err(VcError::NotFound(path.to_path_buf()));
    }
    // `with_execution_providers` returns `Error<SessionBuilder>` (carries a
    // recovery payload); collapse it to the plain `ort::Error` our `?` expects.
    let mut builder = Session::builder()?
        .with_execution_providers([
            CUDAExecutionProvider::default().build(),
            CPUExecutionProvider::default().build(),
        ])
        .map_err(ort::Error::from)?;
    let session = builder.commit_from_file(path)?;
    Ok(session)
}

/// Run a single-input, single-output `f32` model, returning the output's shape
/// (as `i64` dims) and a flat copy of its data.
///
/// Used for the ContentVec and RMVPE models, whose input/output names we read
/// off the session rather than hard-coding.
pub fn run_single_f32(
    session: &mut Session,
    input_shape: Vec<i64>,
    input_data: Vec<f32>,
) -> Result<(Vec<i64>, Vec<f32>)> {
    let in_name = session.inputs()[0].name().to_string();
    let out_name = session.outputs()[0].name().to_string();

    let tensor = Tensor::from_array((input_shape, input_data))?;
    let outputs = session.run(ort::inputs![in_name => tensor])?;

    let value = outputs
        .get(out_name.as_str())
        .ok_or_else(|| VcError::MissingOutput(out_name.clone()))?;
    let (shape, data) = value.try_extract_tensor::<f32>()?;
    Ok((shape.to_vec(), data.to_vec()))
}
