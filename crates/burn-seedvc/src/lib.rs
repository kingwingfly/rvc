//! The Seed-VC network in [Burn](https://burn.dev).
//!
//! Seed-VC converts a voice **without training on it**: a 1–30 s reference clip
//! is the whole speaker specification, where RVC and GPT-SoVITS each want a
//! fine-tune. That is the reason this port exists beside them rather than
//! instead of them.
//!
//! The signal path, in the order the pieces depend on each other:
//!
//! 1. [`content`] runs the source audio through a frozen Whisper encoder, which
//!    is what carries *what was said* while dropping most of who said it,
//! 2. [`length_regulator`] resamples those features to the mel frame rate,
//! 3. [`campplus`] turns the reference clip into one timbre vector, from its own
//!    separate checkpoint and over the Kaldi filterbank in [`fbank`], which is
//!    the only front end it has ever been shown — [`style_encoder`] looks like
//!    the module that does this and is a fossil the released inference path
//!    never builds,
//! 4. [`dit`] is a diffusion transformer that predicts a mel, conditioned on the
//!    content stream and that timbre vector, with [`wavenet`] as its final block,
//! 5. [`flow`] is the sampler that drives the transformer over N steps,
//! 6. [`bigvgan`] vocodes the mel back to a waveform.
//!
//! Only the transformer is generative; everything either side of it is frozen.
//!
//! Nothing here names a compute backend, and there are no app dependencies.
//!
//! # Provenance
//!
//! Ported from Seed-VC (<https://github.com/Plachtaa/seed-vc>, GPL-3.0), read as
//! a reference and never run or vendored. The target preset is
//! `seed-uvit-whisper-small-wavenet` — see [`config`] for why that one, and for
//! every dimension the checkpoint expects.
//!
//! **This is why the whole workspace is GPL-3.0.** A port written from reading
//! GPL source is a derivative work, and the licence carries forward to anything
//! that links it.

pub mod config;

// All public: the weight-coverage example is this crate's only test, and it has
// to construct every module it checks.
pub mod bigvgan;
pub mod content;
pub mod dit;
pub mod flow;
pub mod length_regulator;
pub mod style_encoder;
pub mod vq;
pub mod wavenet;

// The timbre encoder and its filterbank live in `burn-campplus`, because a
// second engine is about to read them and the workspace rule is that anything
// two of them need moves to a neutral crate first — `burn-hubert`'s move out of
// `burn-gptsovits`, one crate over. Re-exported under the names they had, so
// `burn_seedvc::campplus::CamPPlus` and `burn_seedvc::fbank::Fbank` still
// resolve and not one call site changed. Burn derives parameter paths from the
// field names of the *containing* struct rather than from the crate a module was
// declared in, so the move cannot touch a checkpoint key: CAMPPlus still loads
// at 815/0/122.
pub use burn_campplus::{self as campplus, CamPPlus, CamPPlusConfig, fbank};

/// The second spelling PyTorch's weight norm has on disk, as remap patterns.
///
/// **Which one a checkpoint uses is a property of the torch version that saved
/// it, not of the model.** `torch.nn.utils.weight_norm` writes
/// `weight_g`/`weight_v`; `torch.nn.utils.parametrizations.weight_norm`, which
/// supersedes it, writes `parametrizations.weight.original0`/`original1` for the
/// same two tensors in the same order. Every Seed-VC-lineage checkpoint released
/// so far uses the legacy spelling — upstream imports the old `weight_norm` in
/// every module this crate mirrors — so today these patterns match nothing.
///
/// They are here because the failure when that changes is **silent**: NVIDIA
/// re-publishing `bigvgan_generator.pt` from a newer torch, or a fine-tune saved
/// through `parametrizations`, would leave `weight_g` and `weight_v` simply
/// missing, and a weight-normalised convolution keeps its *initialised*
/// direction and magnitude rather than erroring. That is a mis-scaled network
/// that runs. `burn-hubert` accepts both spellings for exactly this reason,
/// after RVC's ContentVec and `chinese-hubert-base` turned out to differ by it.
///
/// Anchored at `$` and matching any parent path, because unlike `burn-hubert` —
/// which has one weight-normalised convolution and can name it — these apply
/// across the whole of the transformer, the vocoder and the quantiser. Appended
/// **after** each loader's own remaps, so a prefix strip still runs first.
pub(crate) const WEIGHT_NORM_REMAPS: [(&str, &str); 2] = [
    (r"\.parametrizations\.weight\.original0$", ".weight_g"),
    (r"\.parametrizations\.weight\.original1$", ".weight_v"),
];

pub use bigvgan::{BigVgan, BigVganConfig};
pub use config::SeedVcConfig;
pub use dit::Dit;
pub use length_regulator::InterpolateRegulator;
pub use vq::{ResidualVq, VqConfig};

#[cfg(test)]
mod tests {
    use burn::module::ParamId;
    use burn::tensor::TensorData;
    use burn_store::{KeyRemapper, TensorSnapshot};

    /// Run `patterns` over `names` the way `load_pytorch_into` does, and return
    /// the paths that come out.
    ///
    /// The tensors are dummies: a remap is a pure function of the *name*, which
    /// is the whole point — it lets the two weight-norm spellings be checked
    /// with no checkpoint, and there is no checkpoint to check them against.
    /// Every Seed-VC-lineage file released so far uses the legacy spelling, so
    /// these patterns match nothing on disk today and this test is the only
    /// thing standing between them and being quietly wrong.
    fn remap(patterns: &[(&str, &str)], names: &[&str]) -> Vec<String> {
        let mut remapper = KeyRemapper::new();
        for (from, to) in patterns {
            remapper = remapper.add_pattern(*from, *to).expect("valid regex");
        }
        let snapshots = names
            .iter()
            .map(|name| {
                TensorSnapshot::from_data(
                    TensorData::new(vec![0.0f32; 4], [2, 2]),
                    name.split('.').map(str::to_string).collect(),
                    vec!["Test".to_string()],
                    ParamId::new(),
                )
            })
            .collect();
        let (out, _) = remapper.remap(snapshots);
        out.iter().map(TensorSnapshot::full_path).collect()
    }

    #[test]
    fn both_weight_norm_spellings_reach_the_same_fields() {
        // `parametrizations.weight.original0`/`original1` are what
        // `torch.nn.utils.parametrizations.weight_norm` writes for the same two
        // tensors, in the same order, that the superseded `weight_norm` spells
        // `weight_g`/`weight_v`. Getting the pairing backwards is the mistake
        // this pins: it would load at 100% and mis-scale every convolution.
        let out = remap(
            &super::WEIGHT_NORM_REMAPS,
            &[
                "blocks.0.conv1.parametrizations.weight.original0",
                "blocks.0.conv1.parametrizations.weight.original1",
            ],
        );
        assert_eq!(
            out,
            ["blocks.0.conv1.weight_g", "blocks.0.conv1.weight_v"],
            "original0 is the magnitude `g` and original1 the direction `v`"
        );
    }

    #[test]
    fn the_legacy_spelling_is_left_alone() {
        // Accepting the new spelling must not disturb the old one, which is
        // what every checkpoint this crate actually loads uses. `burn-hubert`
        // made the same change and proved it by re-running
        // `chinese-hubert-base` to an unchanged 210/0.
        let names = [
            "blocks.0.conv1.weight_g",
            "blocks.0.conv1.weight_v",
            "blocks.0.conv1.bias",
        ];
        assert_eq!(remap(&super::WEIGHT_NORM_REMAPS, &names), names);
    }

    #[test]
    fn a_suffix_rename_composes_with_a_prefix_strip() {
        // The order the loaders append in: the prefix strip and the `conv.conv`
        // flattening run first, and the weight-norm rename then matches a suffix
        // of whatever they produced. That is why these are appended last rather
        // than written in any convenient place.
        //
        // Note what `\.conv\.conv\. -> .` does here: it collapses **both**
        // wrapper levels, because upstream's `SConv1d` wraps a `NormConv1d`
        // which wraps the `Conv1d` and neither wrapper holds a tensor. So
        // `blocks.0.conv.conv.<param>` is `blocks.0.<param>`, not
        // `blocks.0.conv.<param>` — the natural misreading, and one this
        // assertion would have let through had it been written from the prose.
        let mut patterns = vec![
            (r"^net\.cfm\.module\.estimator\.", ""),
            (r"\.conv\.conv\.", "."),
        ];
        patterns.extend(super::WEIGHT_NORM_REMAPS);
        assert_eq!(
            remap(
                &patterns,
                &["net.cfm.module.estimator.blocks.0.conv.conv.parametrizations.weight.original1"],
            ),
            ["blocks.0.weight_v"],
        );
    }
}
