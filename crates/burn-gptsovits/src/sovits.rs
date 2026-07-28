//! The SoVITS stage (`s2`) — semantic tokens to waveform.
//!
//! A modified VITS, which is why most of it comes from `burn-vits` rather than
//! being written again. Being assembled component by component; each one is
//! checked against `s2G*.pth` as it lands, and the harness's `unused` count is
//! the progress bar.
//!
//! | part | where it comes from |
//! |---|---|
//! | `enc_q` posterior encoder | `burn-vits`, unchanged |
//! | `flow` | `burn-vits`, unchanged |
//! | `dec` HiFiGAN | `burn-vits`'s `ResBlock1` under a new upsample stack |
//! | `enc_p` | new — three attention stacks and a cross-attention MRTE |
//! | `ref_enc` | new — a mel style encoder |

use burn::module::Module;
use burn::tensor::Tensor;
use burn::tensor::backend::Backend;
use burn_vits::{PosteriorEncoder, ResidualCouplingBlock};

use crate::decoder::{Decoder, DecoderConfig};
use crate::reference::{ReferenceConfig, ReferenceEncoder};
use crate::text_encoder::{TextEncoder, TextEncoderConfig};

/// The shape of one `s2` checkpoint.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SovitsConfig {
    /// Linear-spectrogram bins the posterior encoder consumes — `n_fft / 2 + 1`.
    pub spec_channels: usize,
    /// Latent width shared by the posterior encoder, the flow and the decoder.
    pub inter_channels: usize,
    /// Width of the WaveNet stacks.
    pub hidden_channels: usize,
    /// Speaker-conditioning width, from the reference encoder.
    pub gin_channels: usize,
    /// Coupling layers in the flow.
    pub n_flows: usize,
    /// WaveNet depth inside each coupling layer — 4 here, where RVC uses 3.
    pub flow_layers: usize,
    /// The waveform decoder.
    pub decoder: DecoderConfig,
    /// Semantic tokens and phonemes to the prior.
    pub text_encoder: TextEncoderConfig,
    /// The speaker vector.
    pub reference: ReferenceConfig,
}

impl Default for SovitsConfig {
    /// v2: 32 kHz, `n_fft` 2048, 192-wide latent.
    fn default() -> Self {
        Self {
            spec_channels: 1025,
            inter_channels: 192,
            hidden_channels: 192,
            gin_channels: 512,
            n_flows: 4,
            flow_layers: 4,
            decoder: DecoderConfig::default(),
            text_encoder: TextEncoderConfig::default(),
            reference: ReferenceConfig::default(),
        }
    }
}

/// The parts of `s2` that exist so far.
///
/// Deliberately not called `Synthesizer` yet: it cannot synthesise until `enc_p`
/// is here to turn tokens and text into a prior, and a name that promises
/// otherwise would be the kind of thing that reads as finished in a diff.
#[derive(Module, Debug)]
pub struct SovitsPartial<B: Backend> {
    /// Spectrogram to latent. Training only — inference goes through `enc_p`.
    pub enc_q: PosteriorEncoder<B>,
    /// Latent to prior, invertible and speaker-conditioned.
    pub flow: ResidualCouplingBlock<B>,
    /// Latent to waveform.
    pub dec: Decoder<B>,
    /// Semantic tokens and phonemes to the prior.
    pub enc_p: TextEncoder<B>,
    /// Reference audio to the speaker vector.
    pub ref_enc: ReferenceEncoder<B>,
}

impl<B: Backend> SovitsPartial<B> {
    pub fn new(cfg: &SovitsConfig, device: &B::Device) -> Self {
        Self {
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
                cfg.n_flows,
                cfg.flow_layers,
                device,
            ),
            dec: Decoder::new(&cfg.decoder, device),
            enc_p: TextEncoder::new(&cfg.text_encoder, device),
            ref_enc: ReferenceEncoder::new(&cfg.reference, device),
        }
    }

    /// Encode a spectrogram to the prior the flow produces.
    ///
    /// `spec`: `[batch, spec_channels, frames]`, `g`: `[batch, gin, 1]`. Returns
    /// `(z, m, logs)` — the sampled latent and the distribution it came from,
    /// which is what the KL term needs.
    pub fn encode(
        &self,
        spec: Tensor<B, 3>,
        g: Tensor<B, 3>,
    ) -> (Tensor<B, 3>, Tensor<B, 3>, Tensor<B, 3>) {
        let (m, logs) = self.enc_q.forward(spec, g.clone());
        // Reparameterised sample: `z = m + eps * exp(logs)`.
        let eps = Tensor::random(
            m.dims(),
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &m.device(),
        );
        let z = m.clone() + eps * logs.clone().exp();
        (z, m, logs)
    }

    /// Load the parts of an `s2G*.pth` that are modelled so far.
    ///
    /// The state dict sits under `"weight"`. Two families of remap, both the
    /// same ones `burn-rvc` needs and for the same reason — the two projects
    /// inherited the layout together: each attention stack is stored as four
    /// parallel lists rather than a list of layers, and the flow's couplings sit
    /// at even indices with parameter-free `Flip`s between them. A third, of the
    /// same kind, renumbers `ref_enc.spectral`.
    ///
    /// The four `quantizer.*` tensors are reported unused on purpose: they load
    /// through [`crate::Quantizer`], which is separately useful for turning a
    /// corpus into semantic tokens without any of the synthesizer.
    pub fn load_pytorch(
        &mut self,
        path: impl AsRef<std::path::Path>,
    ) -> Result<burn_store::ApplyResult, Box<dyn std::error::Error>> {
        let remaps = [
            // Upstream keeps each attention stack as four parallel lists;
            // `burn-vits` groups them per layer. Three stacks, so the stack name
            // is captured alongside the index.
            (
                r"^enc_p\.(encoder_ssl|encoder_text|encoder2)\.attn_layers\.(\d+)\.",
                "enc_p.$1.layers.$2.attn.",
            ),
            (
                r"^enc_p\.(encoder_ssl|encoder_text|encoder2)\.norm_layers_1\.(\d+)\.",
                "enc_p.$1.layers.$2.norm_1.",
            ),
            (
                r"^enc_p\.(encoder_ssl|encoder_text|encoder2)\.ffn_layers\.(\d+)\.",
                "enc_p.$1.layers.$2.ffn.",
            ),
            (
                r"^enc_p\.(encoder_ssl|encoder_text|encoder2)\.norm_layers_2\.(\d+)\.",
                "enc_p.$1.layers.$2.norm_2.",
            ),
            // `ref_enc.spectral` is an `nn.Sequential` of
            // linear/activation/dropout, so its two learnable layers land at
            // indices 0 and 3. Only the learnable ones exist here.
            (r"^ref_enc\.spectral\.3\.", "ref_enc.spectral.1."),
            // `Flip` layers sit between the couplings and carry no parameters.
            (r"^flow\.flows\.2\.", "flow.flows.1."),
            (r"^flow\.flows\.4\.", "flow.flows.2."),
            (r"^flow\.flows\.6\.", "flow.flows.3."),
        ];
        burn_kit::store::load_pytorch_into::<B, _>(self, path.as_ref(), Some("weight"), &remaps)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    type B = burn_ndarray::NdArray;

    #[test]
    fn the_posterior_keeps_the_frame_rate_and_narrows_the_width() {
        // The flow and the decoder both assume the latent is `inter_channels`
        // wide and one vector per spectrogram frame; nothing downstream resamples.
        let cfg = SovitsConfig::default();
        let device = Default::default();
        let model = SovitsPartial::<B>::new(&cfg, &device);

        let frames = 20;
        let spec = Tensor::zeros([1, cfg.spec_channels, frames], &device);
        let g = Tensor::zeros([1, cfg.gin_channels, 1], &device);
        let (z, m, logs) = model.encode(spec, g);
        for t in [&z, &m, &logs] {
            assert_eq!(t.dims(), [1, cfg.inter_channels, frames]);
        }
    }

    #[test]
    fn the_flow_preserves_its_input_shape() {
        // It is a bijection over the latent — a coupling layer that changed the
        // width would not be invertible, and the decoder is sized for it.
        let cfg = SovitsConfig::default();
        let device = Default::default();
        let model = SovitsPartial::<B>::new(&cfg, &device);

        let z = Tensor::zeros([1, cfg.inter_channels, 20], &device);
        let g = Tensor::zeros([1, cfg.gin_channels, 1], &device);
        assert_eq!(model.flow.forward(z, g).dims(), [1, cfg.inter_channels, 20]);
    }
}
