//! HuBERT features to semantic tokens — GPT-SoVITS's `extract_latent`.
//!
//! Two steps and no more: a strided convolution that halves the frame rate from
//! cnhubert's 50 Hz to the 25 Hz the semantic sequence runs at, and a
//! nearest-neighbour lookup in a 1024-entry codebook.
//!
//! This is the boundary the whole model is organised around. The T2S stage
//! predicts these token ids from text, and the SoVITS stage turns them back into
//! waveform, so building a training set means running exactly this over the
//! corpus.

use burn::module::{Module, Param};
use burn::nn::conv::{Conv1d, Conv1dConfig};
use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor};

/// How many entries the codebook has, and how wide they are.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QuantizerConfig {
    /// SSL feature width — 768, cnhubert's hidden size.
    pub dim: usize,
    /// Codebook entries. 1024 in every released configuration, and the size of
    /// the vocabulary the T2S model predicts over.
    pub codebook_size: usize,
    /// Convolution stride from the SSL rate to the semantic rate. 2 for the
    /// 25 Hz models, 1 for the 50 Hz ones.
    pub stride: usize,
}

impl Default for QuantizerConfig {
    /// The v2 configuration: 25 Hz semantic tokens over a 1024-entry codebook.
    fn default() -> Self {
        Self {
            dim: 768,
            codebook_size: 1024,
            stride: 2,
        }
    }
}

/// One codebook.
///
/// The checkpoint carries `embed_avg`, `cluster_size` and `inited` beside
/// `embed`; those are the EMA statistics the codebook is *trained* with and have
/// no role in a lookup, so they are not modelled and are reported unused.
#[derive(Module, Debug)]
pub struct Codebook<B: Backend> {
    embed: Param<Tensor<B, 2>>,
}

impl<B: Backend> Codebook<B> {
    fn new(cfg: &QuantizerConfig, device: &B::Device) -> Self {
        Self {
            embed: Param::from_tensor(Tensor::zeros([cfg.codebook_size, cfg.dim], device)),
        }
    }

    /// Nearest entry to each frame of `x` (`[batch, frames, dim]`).
    ///
    /// Distance is expanded rather than computed directly:
    /// `|x - e|² = |x|² - 2·x·eᵀ + |e|²`. The `|x|²` term is the same across
    /// every candidate for a given frame, so it cannot change which one wins and
    /// is dropped — leaving one matmul.
    fn encode(&self, x: Tensor<B, 3>) -> Tensor<B, 2, Int> {
        let embed = self.embed.val();
        let sq = embed.clone().powi_scalar(2).sum_dim(1).transpose();
        let scores = x.matmul(embed.transpose().unsqueeze()) * 2.0 - sq.unsqueeze();
        scores.argmax(2).squeeze_dim(2)
    }

    /// The entries `codes` names (`[batch, frames]` → `[batch, frames, dim]`).
    fn decode(&self, codes: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let [batch, frames] = codes.dims();
        let dim = self.embed.val().dims()[1];
        let flat = codes.reshape([batch * frames]);
        self.embed
            .val()
            .select(0, flat)
            .reshape([batch, frames, dim])
    }
}

/// The residual vector quantiser.
///
/// One layer in every released configuration, so "residual" describes the
/// architecture it came from rather than anything that happens here. Kept as a
/// list because the checkpoint names it `vq.layers.N` and a deeper stack would
/// otherwise need the tree to change.
#[derive(Module, Debug)]
pub struct Vq<B: Backend> {
    layers: Vec<Codebook<B>>,
}

/// `extract_latent`: SSL features in, semantic tokens out.
#[derive(Module, Debug)]
pub struct Quantizer<B: Backend> {
    ssl_proj: Conv1d<B>,
    vq: Vq<B>,
}

impl<B: Backend> Quantizer<B> {
    pub fn new(cfg: &QuantizerConfig, layers: usize, device: &B::Device) -> Self {
        Self {
            ssl_proj: Conv1dConfig::new(cfg.dim, cfg.dim, cfg.stride)
                .with_stride(cfg.stride)
                .init(device),
            vq: Vq {
                layers: (0..layers).map(|_| Codebook::new(cfg, device)).collect(),
            },
        }
    }

    /// Semantic tokens for cnhubert features.
    ///
    /// `ssl`: `[batch, dim, frames]` at 50 Hz — cnhubert's `last_hidden_state`,
    /// transposed. Returns `[batch, frames / stride]` token ids.
    ///
    /// With more than one codebook each takes the residual the previous left, so
    /// the returned ids are the *first* layer's; the rest refine the
    /// reconstruction and are what [`Quantizer::decode`] sums back.
    pub fn encode(&self, ssl: Tensor<B, 3>) -> Tensor<B, 2, Int> {
        let x = self.ssl_proj.forward(ssl).swap_dims(1, 2);
        let first = self.vq.layers.first().expect("at least one codebook");
        first.encode(x)
    }

    /// Load the quantiser's slice of an `s2G*.pth`.
    ///
    /// The state dict lives under `"weight"`, and the whole rest of the
    /// synthesizer is reported unused until it exists — as are the codebook's
    /// `embed_avg`, `cluster_size` and `inited`, which are EMA statistics from
    /// training the codebook and play no part in a lookup.
    pub fn load_pytorch(
        &mut self,
        path: impl AsRef<std::path::Path>,
    ) -> Result<burn_store::ApplyResult, Box<dyn std::error::Error>> {
        // Two levels of naming that carry no structure: the component prefix,
        // and the `_codebook` submodule each layer wraps its tensors in.
        let remaps = [
            (r"^quantizer\.", ""),
            (r"\.layers\.(\d+)\._codebook\.", ".layers.$1."),
        ];
        burn_kit::store::load_pytorch_into::<B, _>(self, path.as_ref(), Some("weight"), &remaps)
    }

    /// The features `codes` stand for — the inverse of [`Quantizer::encode`],
    /// and what the SoVITS stage consumes. Returns `[batch, dim, frames]`.
    pub fn decode(&self, codes: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let first = self.vq.layers.first().expect("at least one codebook");
        first.decode(codes).swap_dims(1, 2)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::TensorData;

    type B = burn_ndarray::NdArray;

    /// A codebook with known entries.
    ///
    /// Built directly rather than through [`Quantizer`], whose `ssl_proj` is
    /// randomly initialised and would transform the probe before it ever reached
    /// the lookup — a test routed through it passes or fails by luck.
    fn codebook(rows: Vec<Vec<f32>>) -> (Codebook<B>, usize) {
        let (n, dim) = (rows.len(), rows[0].len());
        let device = Default::default();
        let flat: Vec<f32> = rows.into_iter().flatten().collect();
        let embed = Param::from_tensor(Tensor::from_data(TensorData::new(flat, [n, dim]), &device));
        (Codebook { embed }, dim)
    }

    #[test]
    fn a_frame_picks_its_nearest_codebook_entry() {
        // The lookup drops the |x|² term because it is constant across
        // candidates. That is only sound if entries have differing norms, so the
        // codebook here is one where the nearest entry is *not* the largest dot
        // product — `[-3, -2.5]` is closest to `[-3, -3]` but scores highest
        // against `[10, 0]` on dot product alone.
        let (book, dim) = codebook(vec![vec![10.0, 0.0], vec![0.0, 1.0], vec![-3.0, -3.0]]);
        let device = Default::default();
        let probe = |v: [f32; 2]| {
            let x = Tensor::<B, 3>::from_data(TensorData::new(v.to_vec(), [1, 1, dim]), &device);
            let codes: Vec<i64> = book.encode(x).into_data().to_vec().unwrap();
            codes[0]
        };
        assert_eq!(probe([9.0, 0.0]), 0);
        assert_eq!(probe([0.1, 1.1]), 1);
        assert_eq!(probe([-3.0, -2.5]), 2);
    }

    #[test]
    fn decoding_a_code_returns_its_entry() {
        let (book, dim) = codebook(vec![vec![1.0, 2.0], vec![30.0, 40.0]]);
        let device = Default::default();
        let codes = Tensor::<B, 2, Int>::from_data(TensorData::new(vec![1i32, 0], [1, 2]), &device);
        let out = book.decode(codes);
        assert_eq!(out.dims(), [1, 2, dim]);
        let v: Vec<f32> = out.into_data().to_vec().unwrap();
        assert_eq!(v, [30.0, 40.0, 1.0, 2.0]);
    }

    #[test]
    fn a_code_survives_a_round_trip() {
        // What the whole boundary rests on: the id the encoder emits has to name
        // the entry the decoder returns, or the SoVITS stage reconstructs from
        // different features than T2S was trained to predict.
        let (book, dim) = codebook(vec![vec![1.0, 0.0], vec![0.0, 5.0], vec![-2.0, -2.0]]);
        let device = Default::default();
        let x = Tensor::<B, 3>::from_data(
            TensorData::new(vec![0.0, 4.8, -1.9, -2.1], [1, 2, dim]),
            &device,
        );
        let codes = book.encode(x);
        assert_eq!(codes.clone().into_data().to_vec::<i64>().unwrap(), [1, 2]);
        let back: Vec<f32> = book.decode(codes).into_data().to_vec().unwrap();
        assert_eq!(back, [0.0, 5.0, -2.0, -2.0]);
    }

    #[test]
    fn the_stride_halves_the_frame_rate() {
        // cnhubert runs at 50 Hz and the semantic sequence at 25; everything
        // downstream — how many tokens a second of audio is worth, how long a
        // T2S sequence gets — follows from this one stride.
        let device = Default::default();
        let cfg = QuantizerConfig::default();
        let q = Quantizer::<B>::new(&cfg, 1, &device);
        let ssl = Tensor::<B, 3>::zeros([1, cfg.dim, 100], &device);
        assert_eq!(q.encode(ssl).dims(), [1, 50]);
    }
}
