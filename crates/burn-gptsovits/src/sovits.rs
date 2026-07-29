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
use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor};
use burn_vits::{PosteriorEncoder, ResidualCouplingBlock};

use crate::decoder::{Decoder, DecoderConfig};
use crate::quantizer::{Quantizer, QuantizerConfig};
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
    /// The codebook.
    pub quantizer: QuantizerConfig,
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
            quantizer: QuantizerConfig::default(),
        }
    }
}

/// The periods GPT-SoVITS v2's discriminator bank uses.
///
/// Five, where RVC v2 uses eight — confirmed against `s2D2333k.pth`, which holds
/// six `discriminators.N` entries: the scale discriminator plus one per period.
pub const GPTSOVITS_V2_PERIODS: [usize; 5] = [2, 3, 5, 7, 11];

/// What [`SovitsPartial::forward_train`] hands the losses.
pub struct TrainForward<B: Backend> {
    /// The decoded audio segment, `[batch, 1, segment_frames * 640]`.
    pub y_hat: Tensor<B, 3>,
    /// The posterior sample pushed through the flow, `[batch, inter, frames]`.
    pub z_p: Tensor<B, 3>,
    /// Prior mean from `enc_p`, `[batch, inter, frames]`.
    pub m_p: Tensor<B, 3>,
    /// Prior log-std from `enc_p`.
    pub logs_p: Tensor<B, 3>,
    /// Posterior log-std from `enc_q`.
    pub logs_q: Tensor<B, 3>,
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
    /// The codebook. Loaded separately — see [`SovitsPartial::load_pytorch`].
    pub quantizer: Quantizer<B>,
    reference_width: usize,
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
            quantizer: Quantizer::new(&cfg.quantizer, 1, device),
            reference_width: cfg.reference.in_dim,
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

    /// The speaker vector for a reference spectrogram.
    ///
    /// `refer`: `[batch, spec_channels, frames]`. Only the first `in_dim` bins
    /// are used — upstream slices `refer[:, :704]`, which is where that width
    /// comes from rather than from any mel count.
    pub fn speaker(&self, refer: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch, _, frames] = refer.dims();
        let width = self.reference_width;
        self.ref_enc
            .forward(refer.slice([0..batch, 0..width, 0..frames]))
    }

    /// Semantic tokens to waveform.
    ///
    /// `codes`: `[batch, tokens]` at 25 Hz, `text`: `[batch, phones]`,
    /// `g`: the speaker vector from [`SovitsPartial::speaker`]. `noise_scale`
    /// is how much of the prior's variance to actually sample — upstream's
    /// default is 0.5, i.e. deliberately less than the distribution suggests.
    pub fn decode(
        &self,
        codes: Tensor<B, 2, Int>,
        text: Tensor<B, 2, Int>,
        g: Tensor<B, 3>,
        noise_scale: f64,
    ) -> Tensor<B, 3> {
        let quantized = self.quantizer_decode(codes);
        let (m, logs) = self.enc_p.forward(quantized, text, g.clone());

        let eps = Tensor::random(
            m.dims(),
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &m.device(),
        );
        let z_p = m + eps * logs.exp() * noise_scale;
        // Inference runs the flow backwards: the prior is what `enc_p` produced,
        // and the decoder wants the latent it came from.
        let z = self.flow.reverse(z_p, g.clone());
        self.dec.forward(z, g)
    }

    /// The training forward pass: both encoders over the whole utterance, then
    /// the decoder over one random segment of it.
    ///
    /// `codes`: `[batch, tokens]` at 25 Hz, `text`: `[batch, phones]`, `spec`:
    /// the clip's own linear spectrogram `[batch, spec_channels, frames]` at
    /// `frames == tokens * 2`. `ids_slice[i]` is sample `i`'s start frame and
    /// `segment_frames` frames are decoded from there.
    ///
    /// The speaker vector comes from `spec` — the clip conditions on itself, as
    /// upstream's `ge = ref_enc(y)` does. That is the same call inference makes,
    /// so fine-tuning cannot drift away from how the weights are later used.
    ///
    /// Only a segment is decoded because the decoder is the expensive half and
    /// the adversarial losses are local anyway; `enc_p`, `enc_q` and the flow all
    /// see the full sequence, which is what the KL term is computed over.
    pub fn forward_train(
        &self,
        codes: Tensor<B, 2, Int>,
        text: Tensor<B, 2, Int>,
        spec: Tensor<B, 3>,
        ids_slice: &[usize],
        segment_frames: usize,
    ) -> TrainForward<B> {
        let g = self.speaker(spec.clone());

        let (m_p, logs_p) = self
            .enc_p
            .forward(self.quantizer_decode(codes), text, g.clone());
        let (z, _m_q, logs_q) = self.encode(spec, g.clone());
        let z_p = self.flow.forward(z.clone(), g.clone());

        // Gather each sample's own segment. `narrow` twice rather than one
        // `slice`, because the start frame differs per sample.
        let mut slices = Vec::with_capacity(ids_slice.len());
        for (i, &s) in ids_slice.iter().enumerate() {
            slices.push(z.clone().narrow(0, i, 1).narrow(2, s, segment_frames));
        }
        let y_hat = self.dec.forward(Tensor::cat(slices, 0), g);

        TrainForward {
            y_hat,
            z_p,
            m_p,
            logs_p,
            logs_q,
        }
    }

    /// Codes to the 50 Hz features `enc_p` expects.
    ///
    /// The quantiser works at 25 Hz and `enc_p` at 50, so every token is
    /// repeated once — nearest-neighbour upsampling, matching upstream's
    /// `F.interpolate(mode="nearest")`. Interpolating *between* codes would be
    /// wrong: they name codebook entries, and a point between two of them is not
    /// a third entry.
    fn quantizer_decode(&self, codes: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let quantized = self.quantizer.decode(codes);
        let [batch, dim, tokens] = quantized.dims();
        quantized
            .unsqueeze_dim::<4>(3)
            .expand([batch, dim, tokens, 2])
            .reshape([batch, dim, tokens * 2])
    }

    /// Load weights, dispatching on extension: an `s2G*.pth` from upstream, or a
    /// `.safetensors` this toolkit fine-tuned. Both are `s2`; only the container
    /// differs, which is what lets `--s2 tuned.safetensors` stand in for the base
    /// model everywhere the base model is accepted.
    pub fn load_weights(
        &mut self,
        path: impl AsRef<std::path::Path>,
    ) -> Result<burn_store::ApplyResult, Box<dyn std::error::Error>> {
        let path = path.as_ref();
        match path.extension().and_then(|e| e.to_str()) {
            Some("safetensors") => burn_kit::store::load_burn_safetensors_into::<B, _>(self, path),
            _ => self.load_pytorch(path),
        }
    }

    /// Write the module in Burn's own safetensors layout.
    pub fn save_safetensors(
        &self,
        path: impl AsRef<std::path::Path>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        burn_kit::store::save_safetensors::<B, _>(self, path.as_ref())
    }

    /// Load a file [`SovitsPartial::save_safetensors`] wrote.
    ///
    /// Note which loader this uses: `load_burn_safetensors_into`, *not*
    /// `load_safetensors_into`. The latter applies `PyTorchToBurnAdapter`, which
    /// transposes Linear weights from `[out, in]` to `[in, out]` — right for a
    /// Hugging Face checkpoint and wrong for one already in Burn's layout, where
    /// it transposes a second time. A rectangular weight then fails loudly on
    /// shape, but a square one is silently wrong.
    pub fn load_safetensors(
        &mut self,
        path: impl AsRef<std::path::Path>,
    ) -> Result<burn_store::ApplyResult, Box<dyn std::error::Error>> {
        burn_kit::store::load_burn_safetensors_into::<B, _>(self, path.as_ref())
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
    /// The codebook's `embed_avg`, `cluster_size` and `inited` stay unused —
    /// EMA statistics from training it, with no part in a lookup.
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
            // The convolution that halves the frame rate sits at the top level
            // upstream, but belongs to the quantiser: it is the step that takes
            // cnhubert's 50 Hz to the 25 Hz the codes are at. Anchored, so
            // `enc_p.ssl_proj` — a different layer entirely — is left alone.
            (r"^ssl_proj\.", "quantizer.ssl_proj."),
            (
                r"^quantizer\.vq\.layers\.(\d+)\._codebook\.",
                "quantizer.vq.layers.$1.",
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
