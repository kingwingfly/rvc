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

/// Reject a load that left parameters at their initialised values.
///
/// Every loader here allows a partial apply so that a coverage report can be
/// *inspected* rather than a single mismatch aborting the load. The cost of that
/// is that **an empty apply is a success unless somebody looks**: a checkpoint
/// whose names no longer match the module tree leaves every parameter freshly
/// initialised, and the model then runs and produces confident garbage with
/// nothing said anywhere.
///
/// The order of the three checks is the point, and **`errors` first** is the one
/// that is not visible from `burn_store`'s API. Its applier computes `missing`
/// as *visited and not applied and not skipped and **not errored***, so a path
/// that failed to apply is dropped from `applied` and `missing` alike — which
/// means a checkpoint carrying the right tensor *names* at the wrong *shape*
/// reads as 100% coverage to anyone checking only `missing`. That is precisely
/// the failure this workspace has already shipped once: a port that loads at
/// full coverage and computes the wrong thing.
///
/// The third check is the weakest of the three and should be read as such.
/// `missing` counts the *module's* parameters rather than the *file's* tensors,
/// so a checkpoint of an entirely different model comes back as every parameter
/// missing and is refused one branch earlier; `applied.is_empty()` only ever
/// fires on a module with no parameters at all, or if `burn_store` narrows
/// `missing` again the way it already narrows it around `errors`. It is kept as
/// a backstop and because it is the honest message for that case — not because
/// a caller omitting it has a hole. `errors` is the one nobody may omit.
///
/// `unused` is deliberately neither checked nor a parameter. A correct load
/// leaves tensors over all the time — RMVPE's 118 `num_batches_tracked`
/// counters, ContentVec's 57 LayerNorm aliases, or a multi-module `.pth` of
/// which one module is being read — so no threshold is right for every caller,
/// and a wrong one refuses working weights, which is worse than no check at all.
///
/// The error is a `String` rather than [`Error`](crate::Error), and that is what
/// keeps the sentence accurate rather than being a shortcut: every engine's
/// `From<burn_kit::Error>` flattens to its own `Device` variant, so a bare `?`
/// on an `Error` here would report a shape mismatch as "device error". A
/// `String` cannot be `?`-ed into any of them, so each caller is obliged to name
/// the variant that fits. Logging stays with the caller too, which is the half
/// that knows which file it just read.
pub fn check_coverage(label: &str, result: &ApplyResult) -> Result<(), String> {
    if let Some(first) = result.errors.first() {
        return Err(format!(
            "{label}: {} of the checkpoint's tensors could not be applied \
             (first: {first}) — the file does not match the model",
            result.errors.len(),
        ));
    }
    if let Some((path, _)) = result.missing.first() {
        return Err(format!(
            "{label}: {} of {} parameters had no weights in the checkpoint \
             (first: {path}) — the file's tensor names do not match the model",
            result.missing.len(),
            result.missing.len() + result.applied.len(),
        ));
    }
    if result.applied.is_empty() {
        // A backstop rather than the sole reporter of "handed a different model
        // entirely", and the distinction was measured rather than reasoned:
        // `Applier::into_result` derives `missing` from the paths it *visited*,
        // and it visits every parameter of the module rather than every tensor
        // of the file, so a checkpoint matching nothing arrives as every
        // parameter missing and the branch above already refuses it. Kept
        // because it is free, because it is the honest message for that case,
        // and because `missing` has already been narrowed once — around
        // `errors` — so it is not a field to trust unconditionally.
        // `a_foreign_checkpoint_is_every_parameter_missing_not_an_empty_report`
        // is what would fail if that changed.
        return Err(format!(
            "{label}: nothing was applied — no parameter of the module appears \
             in the checkpoint at all"
        ));
    }
    Ok(())
}

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

/// Load a safetensors checkpoint that [`save_safetensors`] wrote.
///
/// The difference from [`load_safetensors_into`] is the missing
/// `PyTorchToBurnAdapter`, and it is not cosmetic: that adapter transposes Linear
/// weights from PyTorch's `[out, in]` to Burn's `[in, out]`, but a file this
/// toolkit saved is *already* in Burn's layout. Loading one through the
/// PyTorch path transposes it a second time — which at least fails loudly on
/// `ShapeMismatch` for a rectangular weight, and would silently transpose a
/// square one.
pub fn load_burn_safetensors_into<B, M>(
    module: &mut M,
    path: &Path,
) -> Result<ApplyResult, Box<dyn Error>>
where
    B: Backend,
    M: ModuleSnapshot<B>,
{
    // `Upcast` stays: training can save fp16 and inference runs fp32.
    let mut store = SafetensorsStore::from_file(path)
        .with_from_adapter(Upcast)
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
    use burn::tensor::Shape;
    use burn_store::{ApplyError, ModuleAdapter};

    /// One report, built field by field because `ApplyResult` has no `Default`.
    fn report(
        applied: &[&str],
        missing: &[&str],
        unused: &[&str],
        errors: Vec<ApplyError>,
    ) -> ApplyResult {
        ApplyResult {
            applied: applied.iter().map(|s| s.to_string()).collect(),
            skipped: Vec::new(),
            missing: missing
                .iter()
                .map(|s| (s.to_string(), String::new()))
                .collect(),
            unused: unused.iter().map(|s| s.to_string()).collect(),
            errors,
        }
    }

    #[test]
    fn a_shape_mismatch_is_rejected_though_nothing_is_missing() {
        // The regression, and the entire reason `check_coverage` exists. The
        // applier drops an errored path from `missing` as well as from
        // `applied`, so this exact pairing — one failed tensor, zero missing —
        // is what a loader checking only `missing` reads as full coverage. A
        // checkpoint one config revision away from the model produces it, and
        // the parameter keeps its initialised value while the model runs.
        let result = report(
            &["enc.weight"],
            &[],
            &[],
            vec![ApplyError::ShapeMismatch {
                path: "enc.bias".to_string(),
                expected: Shape::from([192]),
                found: Shape::from([256]),
            }],
        );
        assert!(result.missing.is_empty(), "the premise of this test");

        let err = check_coverage("the generator", &result)
            .expect_err("a tensor that failed to apply must not read as full coverage");
        assert!(err.contains("the generator"), "{err}");
        assert!(err.contains("enc.bias"), "{err}");
    }

    #[test]
    fn a_complete_load_passes_with_tensors_left_over() {
        // `unused` is not an error and must never be treated as one: RMVPE
        // leaves 118 `num_batches_tracked` counters behind on a load that is
        // exactly right, and refusing valid weights is worse than not checking.
        let result = report(&["enc.weight"], &[], &["enc.num_batches_tracked"], vec![]);
        check_coverage("the generator", &result).expect("leftover tensors are not a failure");
    }

    #[test]
    fn a_missing_parameter_and_an_empty_apply_are_each_rejected() {
        let missing = report(&["enc.weight"], &["enc.bias"], &[], vec![]);
        let err = check_coverage("rmvpe weights", &missing).expect_err("missing must be refused");
        assert!(err.contains("enc.bias"), "{err}");

        // A report with nothing applied *and* nothing missing, which the
        // `missing` branch above could only describe as "0 of 0 parameters".
        // The applier does not currently produce it — see
        // `a_foreign_checkpoint_is_every_parameter_missing_not_an_empty_report`
        // — so this pins the backstop, not a state reachable today.
        let empty = report(&[], &[], &["something.else"], vec![]);
        let err =
            check_coverage("rmvpe weights", &empty).expect_err("an empty apply must be refused");
        assert!(err.contains("nothing was applied"), "{err}");
    }

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

    #[test]
    fn a_saved_module_loads_back_unchanged() {
        // The regression: `load_safetensors_into` applies `PyTorchToBurnAdapter`
        // because Hugging Face ships `[out, in]` Linear weights, but a file we
        // saved ourselves is already `[in, out]`. Reading one back through that
        // path transposes it a second time. A rectangular weight makes the bug
        // visible — square dims would have loaded "fine" and silently scrambled.
        type B = burn::backend::NdArray;
        let device = Default::default();
        let saved = burn::nn::LinearConfig::new(3, 5).init::<B>(&device);
        let expected = saved.weight.val().to_data();

        let dir = std::env::temp_dir().join("burn-kit-roundtrip");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("linear.safetensors");
        save_safetensors::<B, _>(&saved, &path).unwrap();

        let mut loaded = burn::nn::LinearConfig::new(3, 5).init::<B>(&device);
        let result = load_burn_safetensors_into::<B, _>(&mut loaded, &path).unwrap();
        assert!(result.errors.is_empty(), "{:?}", result.errors);
        assert_eq!(loaded.weight.val().dims(), [3, 5]);
        loaded.weight.val().to_data().assert_eq(&expected, true);

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn a_foreign_checkpoint_is_every_parameter_missing_not_an_empty_report() {
        // What `burn_store`'s *real* applier does, as opposed to what a
        // hand-built `ApplyResult` can be made to say. Every other test in this
        // module constructs the report itself, so none of them can establish
        // which states the applier actually produces — and two of
        // `check_coverage`'s three branches exist because of a claim about one
        // of those states.
        //
        // The claim under test: handed a checkpoint whose names match nothing,
        // does the applier report an *empty* `missing` — leaving
        // `applied.is_empty()` as the only field that could notice?
        //
        // It does not. `Applier::into_result` derives `missing` from the paths
        // it *visited*, and it visits every parameter of the module rather than
        // every tensor of the file, so a file that matches nothing comes back as
        // every parameter missing. `applied.is_empty()` is therefore a backstop
        // rather than the sole reporter of this case: a caller that checks
        // `errors` and `missing` alone still refuses a foreign checkpoint.
        //
        // Which is worth pinning precisely because it is the *cheap* half of the
        // check to get wrong in the other direction: if a future `burn-store`
        // narrows `missing` the way it already narrows it around `errors`, this
        // test fails and the third branch stops being redundant.
        type B = burn::backend::NdArray;
        let device = Default::default();
        let saved = burn::nn::LinearConfig::new(3, 5).init::<B>(&device);

        let dir = std::env::temp_dir().join("burn-kit-foreign");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("linear.safetensors");
        save_safetensors::<B, _>(&saved, &path).unwrap();

        // Rename every incoming tensor to a path the module has no slot for.
        let mut loaded = burn::nn::LinearConfig::new(3, 5).init::<B>(&device);
        let result =
            load_safetensors_into::<B, _>(&mut loaded, &path, &[(r"^", "not_this_model.")])
                .unwrap();
        std::fs::remove_file(&path).ok();

        assert!(result.applied.is_empty(), "nothing of this module is there");
        assert!(result.errors.is_empty(), "{:?}", result.errors);
        let missing: Vec<&str> = result.missing.iter().map(|(p, _)| p.as_str()).collect();
        assert_eq!(
            missing,
            ["bias", "weight"],
            "the applier visits the module's parameters, not the file's tensors"
        );
        assert_eq!(
            result.unused.len(),
            2,
            "and the file's two tensors are over"
        );

        let err = check_coverage("the generator", &result)
            .expect_err("a checkpoint of a different model must be refused");
        assert!(err.contains("had no weights"), "{err}");
    }
}
