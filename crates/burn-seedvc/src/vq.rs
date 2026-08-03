//! The residual vector quantiser over the 768-dim content stream.
//!
//! A descript-audio-codec `ResidualVectorQuantize`: project the content down to
//! **8 dimensions**, snap each frame to its nearest entry of a 1024-entry
//! codebook, project back up to 768, and let the next quantiser in the stack
//! encode whatever residual is left. The low-dimensional bottleneck is the whole
//! trick — a codebook only helps if its entries are dense in the space they
//! cover, and 1024 points are hopeless in 768 dimensions and reasonable in 8.
//! **The projections are the easy thing to get backwards**: `in_proj` is 768→8
//! and `out_proj` is 8→768, and the codebook lives in the *narrow* space between
//! them, never in the content's own.
//!
//! # This module is not on the inference path of this preset
//!
//! `net.vq.module.quantizers.0.*` is in the released checkpoint, but nothing in
//! upstream's current tree builds it: `commons.build_model` assembles exactly
//! `cfm` and `length_regulator`, and the only other `vq` in the repository
//! belongs to the unrelated ASTRAL quantiser of the v2 presets. It is a residue
//! of the training script that produced these weights. It is ported anyway,
//! because the tensors are there and **a subtree nobody claims is
//! indistinguishable from a subtree somebody forgot** — which is the entire point
//! of `examples/load` reporting every prefix.
//!
//! The consequence for whoever wires inference: do not reach for this. The
//! conditioning the transformer consumes comes from [`crate::length_regulator`],
//! which on this preset takes continuous content and quantises nothing.
//!
//! # Where the dimensions come from
//!
//! Not from `config.yml` — it has no `vq` section at all, which is the same
//! evidence as above. [`VqConfig::default`] is read off the checkpoint's shapes
//! and is documented as such rather than dressed up as a preset.
//! `content_dim` must agree with [`crate::config::SeedVcConfig::content_dim`];
//! the other three have no second source to check against.
//!
//! # Provenance
//!
//! Mirrors `dac/nn/quantize.py` of descript-audio-codec (MIT), which Seed-VC
//! (<https://github.com/Plachtaa/seed-vc>, GPL-3.0) imports rather than vendors.
//! Read as a reference and never run.

use std::error::Error;
use std::path::Path;

use burn::module::Module;
use burn::nn::{Embedding, EmbeddingConfig};
use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor};
use burn_store::ApplyResult;
use burn_vits::WeightNormConv1d;

/// Shapes of the residual quantiser, read off the checkpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VqConfig {
    /// Width of the stream being quantised — the content encoder's 768.
    pub content_dim: usize,
    /// Entries per codebook (`codebook.weight` is `[1024, 8]`).
    pub codebook_size: usize,
    /// The bottleneck the codebook lives in, 8 — not the content width.
    pub codebook_dim: usize,
    /// Quantisers in the residual stack. One in this checkpoint, so "residual"
    /// describes the architecture rather than anything that happens.
    pub n_codebooks: usize,
}

impl Default for VqConfig {
    fn default() -> Self {
        Self {
            content_dim: 768,
            codebook_size: 1024,
            codebook_dim: 8,
            n_codebooks: 1,
        }
    }
}

/// L2-normalise along the last dimension, as `torch.nn.functional.normalize`
/// does by default — clamping the *norm* at `1e-12` rather than the output, so a
/// zero frame stays zero instead of blowing up.
fn l2_normalise<B: Backend, const D: usize>(x: Tensor<B, D>) -> Tensor<B, D> {
    let norm = x
        .clone()
        .powi_scalar(2)
        .sum_dim(D - 1)
        .sqrt()
        .clamp_min(1e-12);
    x / norm
}

/// One codebook, with the projections either side of it.
#[derive(Module, Debug)]
pub struct VectorQuantize<B: Backend> {
    in_proj: WeightNormConv1d<B>,
    out_proj: WeightNormConv1d<B>,
    codebook: Embedding<B>,
}

impl<B: Backend> VectorQuantize<B> {
    fn new(cfg: &VqConfig, device: &B::Device) -> Self {
        // Width-1 convolutions, which is how DAC spells a per-frame linear map;
        // both carry `torch.nn.utils.weight_norm`, hence the `weight_g`/`weight_v`
        // pairs in the checkpoint where a plain conv would have one `weight`.
        Self {
            in_proj: WeightNormConv1d::new(cfg.content_dim, cfg.codebook_dim, 1, 1, 0, 1, device),
            out_proj: WeightNormConv1d::new(cfg.codebook_dim, cfg.content_dim, 1, 1, 0, 1, device),
            codebook: EmbeddingConfig::new(cfg.codebook_size, cfg.codebook_dim).init(device),
        }
    }

    /// Nearest codebook entry to each frame of `latents` (`[batch, codebook_dim,
    /// frames]`, i.e. already in the bottleneck), as ids `[batch, frames]`.
    ///
    /// DAC normalises **both** sides before comparing — ViT-VQGAN's trick, which
    /// decouples the lookup from the codebook's scale. That is also what lets the
    /// Euclidean distance collapse to a single matmul rather than the usual
    /// three-term expansion: with every entry on the unit sphere,
    /// `‖x − e‖² = ‖x‖² + 1 − 2·x·e`, and neither remaining term varies across
    /// candidates, so the nearest entry is exactly the largest dot product.
    fn encode(&self, latents: Tensor<B, 3>) -> Tensor<B, 2, Int> {
        let x = l2_normalise(latents.swap_dims(1, 2));
        let book = l2_normalise(self.codebook.weight.val());
        x.matmul(book.transpose().unsqueeze())
            .argmax(2)
            .squeeze_dim(2)
    }

    /// The entries `codes` names, back in the bottleneck: `[batch, codebook_dim,
    /// frames]`.
    fn decode(&self, codes: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        self.codebook.forward(codes).swap_dims(1, 2)
    }

    /// `z`: `[batch, content_dim, frames]` → the quantised reconstruction at the
    /// same shape, and the ids that produced it.
    ///
    /// DAC also returns the commitment and codebook losses and the pre-quantised
    /// latents. Those exist to train the codebook, which happens nowhere in this
    /// toolkit, so they are not computed.
    pub fn forward(&self, z: Tensor<B, 3>) -> (Tensor<B, 3>, Tensor<B, 2, Int>) {
        let latents = self.in_proj.forward(z);
        let codes = self.encode(latents);
        (self.out_proj.forward(self.decode(codes.clone())), codes)
    }
}

/// The residual stack.
#[derive(Module, Debug)]
pub struct ResidualVq<B: Backend> {
    quantizers: Vec<VectorQuantize<B>>,
}

impl<B: Backend> ResidualVq<B> {
    pub fn new(cfg: &VqConfig, device: &B::Device) -> Self {
        Self {
            quantizers: (0..cfg.n_codebooks)
                .map(|_| VectorQuantize::new(cfg, device))
                .collect(),
        }
    }

    /// `z`: `[batch, content_dim, frames]` → the summed reconstruction, and one
    /// id sequence per quantiser.
    ///
    /// Each stage sees only what the ones before it failed to represent, so the
    /// reconstruction is the *sum* of their outputs while the residual shrinks —
    /// which is why a later stage's ids mean nothing without the earlier ones.
    pub fn forward(&self, z: Tensor<B, 3>) -> (Tensor<B, 3>, Vec<Tensor<B, 2, Int>>) {
        let mut residual = z;
        let mut summed: Option<Tensor<B, 3>> = None;
        let mut codes = Vec::with_capacity(self.quantizers.len());

        for quantizer in &self.quantizers {
            let (z_q, ids) = quantizer.forward(residual.clone());
            residual = residual - z_q.clone();
            summed = Some(match summed {
                Some(acc) => acc + z_q,
                None => z_q,
            });
            codes.push(ids);
        }

        (summed.expect("at least one quantiser"), codes)
    }

    /// Load this module's slice of the Seed-VC checkpoint.
    pub fn load_pytorch(&mut self, path: impl AsRef<Path>) -> Result<ApplyResult, Box<dyn Error>> {
        let remaps = [(r"^net\.vq\.module\.", "")];
        burn_kit::store::load_pytorch_into::<B, _>(self, path.as_ref(), None, &remaps)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::module::Param;
    use burn::tensor::TensorData;

    type B = burn_ndarray::NdArray;

    /// A quantiser whose codebook is known.
    ///
    /// The lookup is what these tests are about, so they poke
    /// [`VectorQuantize::encode`] directly rather than through `forward`, whose
    /// randomly initialised `in_proj` would transform the probe before it ever
    /// reached a codebook and decide the outcome by luck.
    fn with_codebook(rows: Vec<Vec<f32>>) -> VectorQuantize<B> {
        let (entries, dim) = (rows.len(), rows[0].len());
        let device = Default::default();
        let cfg = VqConfig {
            codebook_size: entries,
            codebook_dim: dim,
            ..Default::default()
        };
        let mut vq = VectorQuantize::<B>::new(&cfg, &device);
        let flat: Vec<f32> = rows.into_iter().flatten().collect();
        vq.codebook.weight = Param::from_tensor(Tensor::from_data(
            TensorData::new(flat, [entries, dim]),
            &device,
        ));
        vq
    }

    #[test]
    fn the_lookup_compares_directions_not_magnitudes() {
        // The load-bearing consequence of DAC normalising both sides: an entry
        // pointing the same way as the frame wins however short it is. Drop the
        // normalisation and `[100, 0]` swallows every probe in this codebook —
        // a port that loads at 100% coverage and quantises everything to one id.
        let vq = with_codebook(vec![vec![100.0, 0.0], vec![0.0, 1.0], vec![-1.0, -1.0]]);
        let device = Default::default();
        let probe = |v: [f32; 2]| {
            // `[batch, dim, frames]` — one frame, so the two channels are `v`.
            let x = Tensor::<B, 3>::from_data(TensorData::new(v.to_vec(), [1, 2, 1]), &device);
            let ids: Vec<i64> = vq.encode(x).into_data().to_vec().unwrap();
            ids[0]
        };
        assert_eq!(probe([3.0, 0.0]), 0);
        assert_eq!(probe([0.0, 0.01]), 1);
        assert_eq!(probe([-9.0, -8.0]), 2);
    }

    #[test]
    fn decoding_ids_returns_their_entries() {
        let vq = with_codebook(vec![vec![1.0, 0.0], vec![0.0, 5.0], vec![-2.0, -2.0]]);
        let device = Default::default();
        let codes = Tensor::<B, 2, Int>::from_data(TensorData::new(vec![1i32, 2], [1, 2]), &device);
        let back: Vec<f32> = vq.decode(codes).into_data().to_vec().unwrap();
        // `[batch, dim, frames]`, so the two entries interleave by channel.
        assert_eq!(back, [0.0, -2.0, 5.0, -2.0]);
    }

    #[test]
    fn the_stack_preserves_the_content_width() {
        // Why the two projections exist at all: the codebook is 8 wide and
        // whatever consumes the result is 768 wide.
        let cfg = VqConfig::default();
        let device = Default::default();
        let vq = ResidualVq::<B>::new(&cfg, &device);
        let (out, codes) = vq.forward(Tensor::zeros([1, cfg.content_dim, 20], &device));
        assert_eq!(out.dims(), [1, cfg.content_dim, 20]);
        assert_eq!(codes.len(), cfg.n_codebooks);
        assert_eq!(codes[0].dims(), [1, 20]);
        assert!(!out.contains_nan().into_scalar());
    }
}
