//! Shared weight-loading helpers: read an RVC `.pth`, remap the flat reference
//! key names onto our module tree, and upcast the fp16 checkpoint to fp32.

use std::error::Error;
use std::path::Path;
use std::rc::Rc;

use burn::tensor::backend::Backend;
use burn::tensor::{DType, TensorData};
use burn_store::pytorch::PytorchReader;
use burn_store::{ApplyResult, KeyRemapper, ModuleSnapshot, PyTorchToBurnAdapter, TensorSnapshot};

/// Load a PyTorch checkpoint subtree into `module`.
///
/// `top_level_key` selects the state_dict inside the checkpoint (RVC uses
/// `"model"`); `remaps` is an ordered list of `(regex, replacement)` applied to
/// every tensor name before matching.
pub fn load_pytorch_into<B, M>(
    module: &mut M,
    path: &Path,
    top_level_key: &str,
    remaps: &[(&str, &str)],
) -> Result<ApplyResult, Box<dyn Error>>
where
    B: Backend,
    M: ModuleSnapshot<B>,
{
    let reader = PytorchReader::with_top_level_key(path, top_level_key)?;
    // The tensor name lives in the map key; stamp it onto each snapshot's path.
    let snapshots: Vec<TensorSnapshot> = reader
        .into_tensors()
        .into_iter()
        .map(|(name, mut snap)| {
            snap.path_stack = Some(name.split('.').map(str::to_string).collect());
            snap
        })
        .collect();

    let mut remapper = KeyRemapper::new();
    for (from, to) in remaps {
        remapper = remapper.add_pattern(*from, *to).expect("static remap patterns are valid");
    }
    let (snapshots, _) = remapper.remap(snapshots);

    // Pretrained bases are fp16; upcast to the model's float.
    let snapshots: Vec<TensorSnapshot> = snapshots.into_iter().map(upcast_f16).collect();

    // `PyTorchToBurnAdapter` handles Linear transposition and norm param names.
    Ok(module.apply(snapshots, None, Some(Box::new(PyTorchToBurnAdapter)), false))
}

/// Upcast an fp16 tensor snapshot to fp32, preserving its path/id.
fn upcast_f16(s: TensorSnapshot) -> TensorSnapshot {
    if s.dtype != DType::F16 {
        return s;
    }
    let data_fn = s.clone_data_fn();
    let new_fn: Rc<dyn Fn() -> Result<TensorData, _>> =
        Rc::new(move || data_fn().map(|d| d.convert_dtype(DType::F32)));
    TensorSnapshot::from_closure(
        new_fn,
        DType::F32,
        s.shape.clone(),
        s.path_stack.clone().unwrap_or_default(),
        s.container_stack.clone().unwrap_or_default(),
        s.tensor_id.unwrap_or_default(),
    )
}
