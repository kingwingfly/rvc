//! One clip in, one speaker vector out — and how to get an implementation of
//! that on whichever compute backend the caller asked for.
//!
//! [`SpeakerEmbedder`] is the whole boundary between [`crate::diarize`] and a
//! runtime, the shape every engine here uses: a trait object rather than a type
//! parameter, so the stage is not generic and the backend is a constructor call.
//! The network behind it is [`burn_campplus`], which was lifted out of
//! `burn-seedvc` for exactly this second reader.
//!
//! # Everything about the input is fixed, and none of it is checkable
//!
//! CAM++ eats a **Kaldi filterbank at 16 kHz**, and the front end that computes
//! one — mean subtraction included — is [`burn_campplus::fbank`]. Substituting
//! the 22.05 kHz mel the vocoders speak runs happily, produces a plausible
//! vector, and quietly stops separating speakers, so
//! [`Fbank`](burn_campplus::fbank::Fbank) is the only thing that ever builds the
//! tensor here and [`ANALYSIS_SR`] is the only rate audio is decoded at for it.
//!
//! # Batching is safe, and it is what makes this usable
//!
//! [`SpeakerEmbedder::embed`] takes many clips at once because a minute of audio
//! is a hundred windows and a hundred separate forward passes is minutes rather
//! than seconds. That it is *allowed* is a property of the network rather than
//! an assumption: [`burn_campplus::Norm`] always normalises by the checkpoint's
//! stored running statistics, never by the batch's, and the pooling that follows
//! is per item — so an embedding cannot depend on what else was in the batch. A
//! batch-statistics norm would have made this silently order-dependent.
//!
//! The clips in one call must be the **same length**, because they are stacked
//! into one tensor. Padding a short one out would put its silence inside the
//! per-clip mean and then pool over it, which is wrong rather than merely
//! different — so the caller drops a short tail instead.

use std::path::Path;

use anyhow::{Context, Result, bail};
use burn::tensor::backend::Backend as BurnBackend;
use burn::tensor::{Tensor, TensorData};
use burn_campplus::fbank::{Fbank, FbankConfig};
use burn_campplus::{CamPPlus, CamPPlusConfig};
use burn_kit::DeviceSpec;
use cli_kit::Backend;

/// The rate CAM++ was trained at, and the only rate audio is analysed at.
///
/// Equal to Whisper's content rate and not the same constraint — the two
/// networks were trained at 16 kHz independently. What a stage *writes* is the
/// user's `--sr` and has nothing to do with this.
pub const ANALYSIS_SR: u32 = 16_000;

/// Clips embedded per forward pass.
///
/// Purely a memory/throughput knob, not a correctness one — see the module docs
/// for why the batch cannot change an embedding. Sized so a 3 s window at 16 kHz
/// stays well inside a 6 GB card alongside the model.
const BATCH: usize = 16;

/// A speaker embedding, backend erased.
pub trait SpeakerEmbedder: Send {
    /// Embed equal-length mono clips at [`ANALYSIS_SR`], one 192-dim vector each.
    ///
    /// Every clip must be at least [`Self::min_samples`] long; a shorter one has
    /// no analysis window in it at all and the front end refuses it.
    fn embed(&self, clips: &[&[f32]]) -> Result<Vec<Vec<f32>>>;

    /// The shortest clip that has a filterbank frame in it — 400 samples, one
    /// 25 ms window, since Kaldi's `snip_edges` framing pads neither end.
    fn min_samples(&self) -> usize;
}

/// Cosine similarity between two embeddings.
///
/// The divisor is doing real work and is not a formality to drop: CAM++ ends in
/// a non-affine batch norm, not a normalisation over the vector, so an embedding
/// is not unit norm.
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f64 = a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum();
    let norm = |v: &[f32]| v.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    (dot / (norm(a) * norm(b)).max(f64::MIN_POSITIVE)) as f32
}

/// The network and its front end on one device.
struct BurnEmbedder<B: BurnBackend> {
    model: CamPPlus<B>,
    fbank: Fbank<B>,
    device: B::Device,
    window: usize,
}

impl<B: BurnBackend> BurnEmbedder<B> {
    fn load(campplus: &Path, device: &B::Device) -> Result<Self> {
        let cfg = FbankConfig::default();
        let mut model = CamPPlus::new(&CamPPlusConfig::default(), device);
        let applied = model
            .load_pytorch(campplus)
            .map_err(|e| anyhow::anyhow!("{e}"))
            .with_context(|| format!("loading CAM++ from {}", campplus.display()))?;
        // Read rather than glanced at: a renamed tensor leaves its module at the
        // values `new` initialised, and a speaker encoder running on those still
        // returns finite, repeatable, entirely meaningless vectors.
        if !applied.missing.is_empty() {
            bail!(
                "CAM++ checkpoint {} is missing {} tensors this port needs, first few: {:?}",
                campplus.display(),
                applied.missing.len(),
                &applied.missing[..applied.missing.len().min(5)],
            );
        }
        tracing::debug!(
            applied = applied.applied.len(),
            unused = applied.unused.len(),
            "loaded CAM++"
        );
        Ok(Self {
            model,
            fbank: Fbank::new(&cfg, device),
            device: device.clone(),
            window: cfg.window(),
        })
    }
}

impl<B: BurnBackend> SpeakerEmbedder for BurnEmbedder<B> {
    fn embed(&self, clips: &[&[f32]]) -> Result<Vec<Vec<f32>>> {
        let mut out = Vec::with_capacity(clips.len());
        for batch in clips.chunks(BATCH) {
            let samples = batch[0].len();
            anyhow::ensure!(
                batch.iter().all(|c| c.len() == samples),
                "a batch of clips to embed must all be the same length"
            );
            anyhow::ensure!(
                samples >= self.window,
                "a clip of {samples} samples is shorter than CAM++'s {}-sample analysis window",
                self.window
            );

            let flat: Vec<f32> = batch.iter().flat_map(|c| c.iter().copied()).collect();
            let wav = Tensor::<B, 2>::from_data(
                TensorData::new(flat, [batch.len(), samples]),
                &self.device,
            );
            // `Fbank::forward` is both of upstream's two lines — the filterbank
            // and the per-clip mean subtraction CAM++ has no input
            // normalisation to stand in for. Building that tensor any other way
            // is the trap `burn-campplus`'s docs name.
            let embeddings: Vec<f32> = self
                .model
                .forward(self.fbank.forward(wav))
                .into_data()
                .to_vec()
                .map_err(|e| anyhow::anyhow!("reading embeddings back: {e:?}"))?;
            let width = embeddings.len() / batch.len();
            out.extend(embeddings.chunks(width).map(<[f32]>::to_vec));
        }
        Ok(out)
    }

    fn min_samples(&self) -> usize {
        self.window
    }
}

/// Settle `auto` and refuse what this stage cannot run.
///
/// Split out of [`load`] so a caller can ask **before** resolving paths: the
/// refusal costs nothing and the fetch behind it is 28 MB over a network that
/// may not be there.
pub fn resolve(backend: Backend) -> Result<Backend> {
    // `false`: there are no ONNX weights to find, so nothing on disk could make
    // `auto` pick ONNX Runtime.
    let backend = backend.resolve(false);
    if backend == Backend::Onnx {
        // Naming what is missing rather than "unsupported", which would send a
        // user hunting for a feature flag that cannot exist: nothing in this
        // workspace exports CAM++, so there is no graph for ORT to open.
        bail!(
            "speaker embedding has no ONNX export — CAM++ is published as a PyTorch checkpoint \
             and `export/` mirrors no graph for it, so `--backend onnx` has nothing to load; \
             run it on Burn instead (`--backend auto|cuda|tch|wgpu`)"
        );
    }
    Ok(backend)
}

/// Load CAM++ onto the chosen backend.
///
/// Naming a backend this build has no code for is an error with a reason; only
/// `auto` substitutes.
pub fn load(
    campplus: &Path,
    backend: Backend,
    device: DeviceSpec,
) -> Result<Box<dyn SpeakerEmbedder>> {
    let backend = resolve(backend)?;
    tracing::info!("loading CAM++ ({backend}, device {device})");

    macro_rules! burn_embedder {
        ($inner:ty, $device:expr, $name:literal) => {{
            let device = $device;
            let model =
                burn_kit::guard_init($name, || BurnEmbedder::<$inner>::load(campplus, &device))??;
            Box::new(model) as Box<dyn SpeakerEmbedder>
        }};
    }

    Ok(match backend {
        #[cfg(feature = "tch")]
        Backend::Tch => burn_embedder!(
            burn::backend::LibTorch<f32>,
            burn_kit::libtorch_device(device)?,
            "tch"
        ),
        #[cfg(feature = "cuda")]
        Backend::Cuda => {
            burn_embedder!(burn::backend::Cuda, burn_kit::cuda_device(device)?, "cuda")
        }
        #[cfg(feature = "wgpu")]
        Backend::Wgpu => {
            burn_embedder!(burn::backend::Wgpu, burn_kit::wgpu_device(device)?, "wgpu")
        }
        Backend::Auto | Backend::Onnx => unreachable!("resolved and refused above"),
        // Only reachable on a `--no-default-features` build, where the arm that
        // would have handled it was `#[cfg]`ed away.
        #[allow(unreachable_patterns)]
        other => return Err(other.unavailable()),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two vectors pointing the same way are 1 whatever their lengths, which is
    /// the property a threshold is read against — an embedding is not unit norm,
    /// so a bare dot product would rank a loud window above a quiet one.
    #[test]
    fn cosine_ignores_magnitude() {
        let a = [1.0, 2.0, -3.0];
        let scaled = [10.0, 20.0, -30.0];
        assert!((cosine(&a, &scaled) - 1.0).abs() < 1e-6);
        assert!((cosine(&a, &[-1.0, -2.0, 3.0]) + 1.0).abs() < 1e-6);
        assert_eq!(cosine(&[0.0, 0.0], &[1.0, 1.0]), 0.0);
    }

    /// `--backend onnx` has to fail before anything is fetched, and the message
    /// has to say there is no graph rather than "unsupported".
    #[test]
    fn onnx_is_refused_with_a_reason() {
        let err = resolve(Backend::Onnx)
            .expect_err("onnx accepted")
            .to_string();
        assert!(err.contains("no ONNX export"), "{err}");
        assert!(err.contains("--backend auto|cuda|tch|wgpu"), "{err}");
        // A Burn backend named out loud is returned untouched.
        for burn in [Backend::Cuda, Backend::Tch, Backend::Wgpu] {
            assert_eq!(resolve(burn).unwrap(), burn);
        }
    }
}
