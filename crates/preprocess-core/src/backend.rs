//! Choosing a compute backend at run time, and the one separation model behind
//! it.
//!
//! The enum, its aliases and the `auto` rule are [`cli_kit::Backend`], shared
//! with every other binary — there is one such enum for the whole workspace and
//! a second must not appear beside it. That is the same trade `seedvc-core`
//! makes and the reason it, too, depends on `cli-kit`: [`load`] erases the Burn
//! backend behind a `Box<dyn Separator>`, so it has to name one.
//!
//! [`load`] is the whole of that erasure. Each arm builds an [`MdxSeparator`]
//! for one concrete Burn backend and boxes it, so nothing above this module
//! names a Burn type and the choice is purely a run-time one.

use std::path::Path;

use anyhow::{Context, Result};
use audio_kit::StereoSamples;
use burn::tensor::backend::Backend as BurnBackend;
use burn_kit::DeviceSpec;
use burn_mdx::{MdxConfig, STEMS_8K_INSTVOC, Stft, TfcTdfNet};
use cli_kit::Backend;

use crate::separate::Separator;

/// Settle `auto` and refuse what this stage cannot run.
///
/// Split out of [`load`] so a caller can ask **before** resolving paths: the
/// refusal costs nothing, while the fetch behind it is 448 MB, and a cold cache
/// would otherwise download the whole checkpoint in order to reject the backend
/// afterwards.
///
/// `auto` resolves by hardware alone. There is no artefact on disk that could
/// decide it, because there is only one weight format here — `false` is passed
/// to [`Backend::resolve`] for that reason, not because ONNX Runtime is
/// second-class.
pub fn resolve(backend: Backend) -> Result<Backend> {
    let backend = backend.resolve(false);
    // The reason has to name a *model*, not a missing flag. MDX23C is published
    // as a PyTorch checkpoint and has no ONNX graph at all; the MDX-Net v2
    // graphs that do exist are a different, older architecture that is
    // deliberately not ported — so there is nothing a user could point an
    // `--onnx` at, and saying "unsupported" would send them looking for a flag
    // that cannot exist.
    anyhow::ensure!(
        backend != Backend::Onnx,
        "--backend onnx: the separation model (MDX23C) is published as a PyTorch \
         checkpoint and has no ONNX graph — run it on a Burn backend \
         (`--backend auto|cuda|tch|wgpu`)"
    );
    Ok(backend)
}

/// Load the separation checkpoint onto the chosen backend.
///
/// Naming a backend this build has no code for is an error with a reason; only
/// `auto` substitutes.
pub fn load(weights: &Path, backend: Backend, device: DeviceSpec) -> Result<Box<dyn Separator>> {
    let backend = resolve(backend)?;
    tracing::info!(
        "loading the separation model ({backend}, device {device}): {}",
        weights.display()
    );

    macro_rules! burn_separator {
        ($inner:ty, $device:expr, $name:literal) => {{
            let device = $device;
            let model =
                burn_kit::guard_init($name, || MdxSeparator::<$inner>::load(weights, device))??;
            Box::new(model) as Box<dyn Separator>
        }};
    }

    let model: Box<dyn Separator> = match backend {
        #[cfg(feature = "tch")]
        Backend::Tch => burn_separator!(
            burn::backend::LibTorch<f32>,
            burn_kit::libtorch_device(device)?,
            "tch"
        ),
        #[cfg(feature = "cuda")]
        Backend::Cuda => {
            burn_separator!(burn::backend::Cuda, burn_kit::cuda_device(device)?, "cuda")
        }
        #[cfg(feature = "wgpu")]
        Backend::Wgpu => {
            burn_separator!(burn::backend::Wgpu, burn_kit::wgpu_device(device)?, "wgpu")
        }
        Backend::Auto | Backend::Onnx => unreachable!("resolved and refused above"),
        // Only reachable on a `--no-default-features` build, where the arm that
        // would have handled it was `#[cfg]`ed away.
        #[allow(unreachable_patterns)]
        other => return Err(other.unavailable()),
    };
    Ok(model)
}

/// MDX23C on one Burn backend: the STFT front end, the network, and the inverse
/// transform.
///
/// The three belong together because the front end is the model's own — its
/// `n_fft`, hop and kept-bin count come out of the same configuration the
/// network's channel widths do, and pairing a network with somebody else's
/// transform runs happily and computes something else.
pub struct MdxSeparator<B: BurnBackend> {
    net: TfcTdfNet<B>,
    stft: Stft,
    cfg: MdxConfig,
    device: B::Device,
}

impl<B: BurnBackend> MdxSeparator<B> {
    /// Load the checkpoint and report its coverage.
    ///
    /// The report is checked rather than swallowed: a missing tensor is a
    /// silently mis-initialised layer, which separates *something* and cannot be
    /// told from a bad recording by ear. The expected reading is 319 applied, 0
    /// missing, 0 unused — there are no training-only tensors in this file,
    /// because the norms are instance norms.
    ///
    /// The check is [`burn_kit::check_coverage`] rather than a condition written
    /// here, and that is not tidying. **`errors` is not covered by `missing`**:
    /// `burn_store`'s applier counts a path as missing only when it was visited
    /// and *not* errored, so a tensor of the wrong shape is excluded from both
    /// tallies and a mismatched checkpoint reads as **full coverage**. A
    /// hand-written `missing.is_empty()` — which is what stood here — passes it.
    pub fn load(weights: &Path, device: B::Device) -> Result<Self> {
        let cfg = MdxConfig::mdx23c_8k_instvoc_hq();
        let mut net = TfcTdfNet::<B>::new(&cfg, &device);
        let applied = net
            .load_pytorch(weights)
            .map_err(|e| anyhow::anyhow!("{e}"))
            .with_context(|| format!("loading {}", weights.display()))?;
        burn_kit::check_coverage(
            &format!("separation weights {}", weights.display()),
            &applied,
        )
        .map_err(|e| anyhow::anyhow!("{e}"))?;
        tracing::info!(
            "separation weights: {} applied / {} missing / {} unused",
            applied.applied.len(),
            applied.missing.len(),
            applied.unused.len()
        );
        Ok(Self {
            stft: cfg.stft(),
            cfg,
            net,
            device,
        })
    }
}

impl<B: BurnBackend> Separator for MdxSeparator<B> {
    fn sample_rate(&self) -> u32 {
        self.cfg.sample_rate
    }

    fn chunk_frames(&self) -> usize {
        self.cfg.chunk_size()
    }

    fn stems(&self) -> &'static [&'static str] {
        &STEMS_8K_INSTVOC
    }

    fn separate(&self, chunk: &StereoSamples) -> Result<Vec<StereoSamples>> {
        anyhow::ensure!(
            chunk.frames() == self.chunk_frames(),
            "the network's unit of work is {} frames, not {}",
            self.chunk_frames(),
            chunk.frames()
        );
        let spec = self
            .stft
            .forward::<B>(&[chunk.left.clone(), chunk.right.clone()], &self.device);
        let out = self.net.forward(spec);
        let [_, stems, channels, bins, frames] = out.dims();
        self.stft
            .inverse(out.reshape([stems, channels, bins, frames]))
            .into_iter()
            .map(|mut stem| {
                // Popped rather than indexed, so the two channels cannot be
                // read in the wrong order by a later edit: `right` is the last
                // plane and `left` the one before it, which is the order the
                // front end packed them in.
                anyhow::ensure!(
                    stem.len() == 2,
                    "the network returned {} channels per stem, not 2",
                    stem.len()
                );
                let right = stem.pop().expect("two channels");
                let left = stem.pop().expect("two channels");
                Ok(StereoSamples { left, right })
            })
            .collect()
    }
}
