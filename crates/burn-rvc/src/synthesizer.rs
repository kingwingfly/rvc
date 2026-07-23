//! The full generator `SynthesizerTrnMs768NSFsid`.
//!
//! Being assembled incrementally; fields are added as each submodule lands, and
//! [`Synthesizer::load_pytorch`] loads the matching subset of the reference
//! `state_dict` so weight coverage can be checked at every step.

use std::error::Error;
use std::path::Path;

use burn::module::Module;
use burn::nn::{Embedding, EmbeddingConfig};
use burn::tensor::backend::Backend;
use burn::tensor::{Distribution, Int, Tensor};
use burn_store::{ApplyResult, ModuleSnapshot, SafetensorsStore};

use crate::config::SynthesizerConfig;
use crate::flow::ResidualCouplingBlock;
use crate::generator::GeneratorNsf;
use crate::posterior::PosteriorEncoder;
use crate::store::load_pytorch_into;
use crate::text_encoder::TextEncoder;

/// Number of coupling layers in the flow.
const N_FLOWS: usize = 4;

/// Outputs of [`Synthesizer::forward_train`] needed by the losses.
pub struct TrainForward<B: Backend> {
    /// Decoded audio segment `[b, 1, segment_frames · hop]`.
    pub y_hat: Tensor<B, 3>,
    /// Flow output over the full sequence (prior sample under the flow).
    pub z_p: Tensor<B, 3>,
    /// Prior mean `[b, inter, T]`.
    pub m_p: Tensor<B, 3>,
    /// Prior log-std `[b, inter, T]`.
    pub logs_p: Tensor<B, 3>,
    /// Posterior log-std `[b, inter, T]`.
    pub logs_q: Tensor<B, 3>,
}

/// RVC v2 generator `SynthesizerTrnMs768NSFsid`.
#[derive(Module, Debug)]
pub struct Synthesizer<B: Backend> {
    /// Prior / content encoder (`enc_p`).
    pub enc_p: TextEncoder<B>,
    /// NSF-HiFiGAN decoder (`dec`).
    pub dec: GeneratorNsf<B>,
    /// Posterior encoder (`enc_q`, training-only).
    pub enc_q: PosteriorEncoder<B>,
    /// Normalizing flow (`flow`).
    pub flow: ResidualCouplingBlock<B>,
    /// Speaker embedding table (`emb_g`).
    pub emb_g: Embedding<B>,
}

impl<B: Backend> Synthesizer<B> {
    /// Initialise from a config.
    pub fn new(cfg: &SynthesizerConfig, device: &B::Device) -> Self {
        Self {
            enc_p: TextEncoder::new(cfg, device),
            dec: GeneratorNsf::new(cfg, device),
            enc_q: PosteriorEncoder::new(
                cfg.spec_channels,
                cfg.inter_channels,
                cfg.hidden_channels,
                cfg.gin_channels,
                device,
            ),
            flow: ResidualCouplingBlock::new(
                cfg.inter_channels,
                cfg.hidden_channels,
                cfg.gin_channels,
                N_FLOWS,
                device,
            ),
            emb_g: EmbeddingConfig::new(cfg.spk_embed_dim, cfg.gin_channels).init(device),
        }
    }

    /// Training forward (`SynthesizerTrnMs768NSFsid.forward`): runs the
    /// posterior + flow over the full sequence, then decodes a random segment.
    ///
    /// `spec` is the linear spectrogram `[b, spec_channels, T]`; `ids_slice[i]`
    /// is sample `i`'s start frame; `segment_frames` z-frames are decoded.
    /// Returns the decoded segment and the tensors the KL/adversarial losses need.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_train(
        &self,
        phone: Tensor<B, 3>,
        pitch: Tensor<B, 2, Int>,
        nsff0: Tensor<B, 2>,
        spec: Tensor<B, 3>,
        speaker_id: i64,
        ids_slice: &[usize],
        segment_frames: usize,
    ) -> TrainForward<B> {
        let [b, _, _] = phone.dims();
        let device = phone.device();

        let sid_t = Tensor::<B, 2, Int>::full([b, 1], speaker_id, &device);
        let g = self.emb_g.forward(sid_t).swap_dims(1, 2); // [b, gin, 1]

        let (m_p, logs_p) = self.enc_p.forward(phone, pitch);
        let (m_q, logs_q) = self.enc_q.forward(spec, g.clone());
        // z = m_q + eps * exp(logs_q)
        let eps = Tensor::random(m_q.dims(), Distribution::Normal(0.0, 1.0), &device);
        let z = m_q + eps * logs_q.clone().exp();
        let z_p = self.flow.forward(z.clone(), g.clone());

        // Slice a random segment per sample, then decode it.
        let mut z_slices = Vec::with_capacity(b);
        let mut f0_slices = Vec::with_capacity(b);
        for (i, &s) in ids_slice.iter().enumerate() {
            z_slices.push(z.clone().narrow(0, i, 1).narrow(2, s, segment_frames));
            f0_slices.push(nsff0.clone().narrow(0, i, 1).narrow(1, s, segment_frames));
        }
        let z_slice = Tensor::cat(z_slices, 0);
        let nsff0_slice = Tensor::cat(f0_slices, 0);
        let y_hat = self.dec.forward(z_slice, nsff0_slice, g);

        TrainForward {
            y_hat,
            z_p,
            m_p,
            logs_p,
            logs_q,
        }
    }

    /// Voice-conversion inference (`SynthesizerTrnMs768NSFsid.infer`).
    ///
    /// - `phone`: content features `[batch, time, 768]`
    /// - `pitch`: coarse pitch ids `[batch, time]`
    /// - `nsff0`: fine F0 in Hz `[batch, time]`
    /// - `sid`: speaker id
    ///
    /// Returns the waveform `[batch, 1, time·hop]`.
    pub fn infer(
        &self,
        phone: Tensor<B, 3>,
        pitch: Tensor<B, 2, Int>,
        nsff0: Tensor<B, 2>,
        sid: i64,
    ) -> Tensor<B, 3> {
        let [b, _, _] = phone.dims();
        let device = phone.device();

        // g = emb_g(sid).unsqueeze(-1) -> [b, gin, 1]
        let sid_t = Tensor::<B, 2, Int>::full([b, 1], sid, &device);
        let g = self.emb_g.forward(sid_t).swap_dims(1, 2);

        let (m_p, logs_p) = self.enc_p.forward(phone, pitch);
        // z_p = m_p + exp(logs_p) * N(0,1) * 0.66666
        let noise = Tensor::random(m_p.dims(), Distribution::Normal(0.0, 1.0), &device);
        let z_p = m_p + logs_p.exp() * noise * 0.666_66;

        let z = self.flow.reverse(z_p, g.clone());
        self.dec.forward(z, nsff0, g)
    }

    /// Load the reference RVC generator checkpoint (`f0G*.pth` or a trained
    /// `G_*.pth`), remapping the flat reference names onto this module tree and
    /// upcasting the fp16 pretrained weights to the model dtype.
    ///
    /// Returns the [`ApplyResult`]: `applied` names, `missing` model params (must
    /// be empty for a fully-ported module), and `unused` checkpoint tensors (the
    /// not-yet-ported submodules).
    pub fn load_pytorch(&mut self, path: impl AsRef<Path>) -> Result<ApplyResult, Box<dyn Error>> {
        // Flat reference lists -> our per-layer tree; the flow interleaves
        // coupling layers (even indices) with paramless flips, so pack the
        // coupling layers into a contiguous Vec.
        let remaps = [
            (
                r"^enc_p\.encoder\.attn_layers\.(\d+)\.",
                "enc_p.encoder.layers.$1.attn.",
            ),
            (
                r"^enc_p\.encoder\.norm_layers_1\.(\d+)\.",
                "enc_p.encoder.layers.$1.norm_1.",
            ),
            (
                r"^enc_p\.encoder\.ffn_layers\.(\d+)\.",
                "enc_p.encoder.layers.$1.ffn.",
            ),
            (
                r"^enc_p\.encoder\.norm_layers_2\.(\d+)\.",
                "enc_p.encoder.layers.$1.norm_2.",
            ),
            (r"^flow\.flows\.2\.", "flow.flows.1."),
            (r"^flow\.flows\.4\.", "flow.flows.2."),
            (r"^flow\.flows\.6\.", "flow.flows.3."),
        ];
        load_pytorch_into::<B, _>(self, path.as_ref(), "model", &remaps)
    }

    /// Save weights in Burn-native safetensors (our field naming; round-trips
    /// with [`Synthesizer::load_safetensors`]).
    pub fn save_safetensors(&self, path: impl AsRef<Path>) -> Result<(), Box<dyn Error>> {
        let mut store = SafetensorsStore::from_file(path.as_ref());
        self.save_into(&mut store)?;
        Ok(())
    }

    /// Load weights previously written by [`Synthesizer::save_safetensors`].
    pub fn load_safetensors(
        &mut self,
        path: impl AsRef<Path>,
    ) -> Result<ApplyResult, Box<dyn Error>> {
        let mut store = SafetensorsStore::from_file(path.as_ref());
        Ok(self.load_from(&mut store)?)
    }

    /// Load weights, dispatching on file extension: `.pth`/`.pt` (PyTorch/RVC,
    /// remapped + upcast) or `.safetensors` (Burn-native).
    pub fn load_weights(&mut self, path: impl AsRef<Path>) -> Result<ApplyResult, Box<dyn Error>> {
        let path = path.as_ref();
        match path.extension().and_then(|e| e.to_str()) {
            Some("pth") | Some("pt") => self.load_pytorch(path),
            Some("safetensors") => self.load_safetensors(path),
            other => {
                Err(format!("unsupported weights format: {other:?} ({})", path.display()).into())
            }
        }
    }
}
