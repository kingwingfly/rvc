//! Weight loading: read a checkpoint, remap the reference implementation's key
//! names onto our module tree, and upcast fp16 to fp32.
//!
//! PyTorch `.pth` is what the RVC-lineage projects publish; Hugging Face model
//! repos ship safetensors. `burn-store` reads the two through differently shaped
//! APIs — a snapshot reader and a builder — so the two loaders here do not share
//! an implementation, only a meaning.

use std::error::Error;
use std::path::Path;
use std::rc::Rc;

use burn::tensor::backend::Backend;
use burn::tensor::{DType, TensorData};
use burn_store::pytorch::PytorchReader;
use burn_store::{
    ApplyResult, KeyRemapper, ModuleAdapter, ModuleSnapshot, PyTorchToBurnAdapter,
    SafetensorsStore, TensorSnapshot,
};

/// Build the key remapper from `(regex, replacement)` pairs, applied in order.
fn remapper(remaps: &[(&str, &str)]) -> KeyRemapper {
    let mut remapper = KeyRemapper::new();
    for (from, to) in remaps {
        remapper = remapper
            .add_pattern(*from, *to)
            .expect("static remap patterns are valid");
    }
    remapper
}

/// One checkpoint entry: its name, dtype and shape.
pub type TensorInfo = (String, DType, Vec<usize>);

/// List a PyTorch checkpoint's tensors.
///
/// The first thing to run against a new checkpoint: the module tree has to mirror
/// these names, and guessing them from a reference implementation's source is how
/// a port ends up with a silent mismatch.
pub fn pytorch_keys(
    path: &Path,
    top_level_key: Option<&str>,
) -> Result<Vec<TensorInfo>, Box<dyn Error>> {
    let reader = open_pytorch(path, top_level_key)?;
    let mut out: Vec<_> = reader
        .into_tensors()
        .into_iter()
        .map(|(name, snap)| {
            let dims = snap
                .shape
                .clone()
                .into_ranges()
                .into_iter()
                .map(|r| r.end)
                .collect();
            (name, snap.dtype, dims)
        })
        .collect();
    out.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(out)
}

/// Open a checkpoint, with or without a wrapping key.
///
/// Hugging Face's `pytorch_model.bin` puts the state dict at the root; the
/// RVC-lineage `.pth` files wrap theirs under `"model"`.
fn open_pytorch(path: &Path, top_level_key: Option<&str>) -> Result<PytorchReader, Box<dyn Error>> {
    Ok(match top_level_key {
        Some(key) => PytorchReader::with_top_level_key(path, key)?,
        None => PytorchReader::new(path)?,
    })
}

/// Load a PyTorch checkpoint subtree into `module`.
///
/// `top_level_key` selects the state_dict inside the checkpoint — `Some("model")`
/// for the RVC-lineage `.pth` files, `None` for a Hugging Face
/// `pytorch_model.bin`, which puts it at the root. `remaps` is an ordered list of
/// `(regex, replacement)` applied to every tensor name before matching.
pub fn load_pytorch_into<B, M>(
    module: &mut M,
    path: &Path,
    top_level_key: Option<&str>,
    remaps: &[(&str, &str)],
) -> Result<ApplyResult, Box<dyn Error>>
where
    B: Backend,
    M: ModuleSnapshot<B>,
{
    let reader = open_pytorch(path, top_level_key)?;
    // The tensor name lives in the map key; stamp it onto each snapshot's path.
    let snapshots: Vec<TensorSnapshot> = reader
        .into_tensors()
        .into_iter()
        .map(|(name, mut snap)| {
            snap.path_stack = Some(name.split('.').map(str::to_string).collect());
            snap
        })
        .collect();

    let (snapshots, _) = remapper(remaps).remap(snapshots);

    // Pretrained bases are fp16; upcast to the model's float.
    let snapshots: Vec<TensorSnapshot> = snapshots.into_iter().map(upcast_f16).collect();

    // `PyTorchToBurnAdapter` handles Linear transposition and norm param names.
    Ok(module.apply(snapshots, None, Some(Box::new(PyTorchToBurnAdapter)), false))
}

/// Load a safetensors checkpoint into `module` — the form Hugging Face model
/// repos ship, and the reason this toolkit prefers first-party repos to
/// community re-exports of the same weights.
///
/// Partial application is allowed so the caller gets an [`ApplyResult`] to
/// inspect rather than an error: `missing` and `unused` counts are how a port is
/// checked, and a hard failure would report only the first mismatch.
pub fn load_safetensors_into<B, M>(
    module: &mut M,
    path: &Path,
    remaps: &[(&str, &str)],
) -> Result<ApplyResult, Box<dyn Error>>
where
    B: Backend,
    M: ModuleSnapshot<B>,
{
    // Both adapters are needed and the store takes one: Hugging Face checkpoints
    // are routinely fp16 (`torch_dtype: float16`), and their Linear weights still
    // arrive `[out, in]` where Burn wants `[in, out]`.
    let mut store = SafetensorsStore::from_file(path)
        .remap(remapper(remaps))
        .with_from_adapter(Upcast.chain(PyTorchToBurnAdapter))
        .allow_partial(true);
    Ok(module.load_from(&mut store)?)
}

/// Write a module to a safetensors file, replacing whatever was there.
///
/// Overwrite rather than fail: re-running training to an existing output must
/// replace it, not error at save time and lose the weights just trained.
pub fn save_safetensors<B, M>(module: &M, path: &Path) -> Result<(), Box<dyn Error>>
where
    B: Backend,
    M: ModuleSnapshot<B>,
{
    let mut store = SafetensorsStore::from_file(path).overwrite(true);
    module.save_into(&mut store)?;
    Ok(())
}

/// Widen fp16 tensors to fp32 on the way in, and never the other way.
///
/// `burn_store::HalfPrecisionAdapter` looks like exactly this and is a trap: it
/// is bidirectional, so it *narrows* an fp32 checkpoint to fp16 on load. That is
/// silent almost everywhere — the model simply loses half its mantissa — and
/// surfaces only where something checks, such as LibTorch refusing a conv whose
/// bias dtype no longer matches its input.
#[derive(Clone)]
struct Upcast;

impl ModuleAdapter for Upcast {
    fn adapt(&self, snapshot: &TensorSnapshot) -> TensorSnapshot {
        upcast_f16(snapshot.clone())
    }

    fn clone_box(&self) -> Box<dyn ModuleAdapter> {
        Box::new(self.clone())
    }
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

#[cfg(test)]
mod tests {
    use super::*;
    use burn_store::ModuleAdapter;

    fn snapshot(dtype: DType, data: TensorData) -> TensorSnapshot {
        TensorSnapshot::from_closure(
            Rc::new(move || Ok(data.clone())),
            dtype,
            burn::tensor::Shape::from([2]),
            vec!["w".to_string()],
            vec!["Linear".to_string()],
            Default::default(),
        )
    }

    #[test]
    fn loading_widens_fp16_and_leaves_fp32_alone() {
        // The second half is the regression: `burn_store::HalfPrecisionAdapter`
        // converts *both* directions, so using it here quietly halved the
        // precision of every fp32 checkpoint. It went unnoticed until LibTorch
        // rejected a conv whose bias no longer matched its input dtype.
        let half = snapshot(
            DType::F16,
            TensorData::from([1.0f32, 2.0]).convert_dtype(DType::F16),
        );
        assert_eq!(Upcast.adapt(&half).dtype, DType::F32, "fp16 must widen");

        let full = snapshot(DType::F32, TensorData::from([1.0f32, 2.0]));
        assert_eq!(
            Upcast.adapt(&full).dtype,
            DType::F32,
            "fp32 must be left alone, not narrowed"
        );
    }
}
