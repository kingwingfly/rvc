//! Architecture hyperparameters for the RVC v2 synthesizer.
//!
//! Values mirror the RVC-Project reference (`configs/v2/48k.json` +
//! `SynthesizerTrnMs768NSFsid`); the defaults here are the 48 kHz v2 model.

/// Full configuration of `SynthesizerTrnMs768NSFsid`.
#[derive(Debug, Clone)]
pub struct SynthesizerConfig {
    /// Posterior-encoder input width (linear-spectrogram bins, `n_fft/2 + 1`).
    pub spec_channels: usize,
    /// Latent width shared by the flow / prior / posterior (`inter_channels`).
    pub inter_channels: usize,
    /// Transformer hidden width.
    pub hidden_channels: usize,
    /// FFN inner width in the text encoder.
    pub filter_channels: usize,
    /// Attention heads.
    pub n_heads: usize,
    /// Transformer layers.
    pub n_layers: usize,
    /// FFN convolution kernel size.
    pub kernel_size: usize,
    /// Relative-position attention window.
    pub window_size: usize,
    /// HiFi-GAN resblock kernel sizes.
    pub resblock_kernel_sizes: Vec<usize>,
    /// HiFi-GAN resblock dilation sizes (per kernel).
    pub resblock_dilation_sizes: Vec<Vec<usize>>,
    /// NSF upsampling factors.
    pub upsample_rates: Vec<usize>,
    /// Channels at the head of the decoder before upsampling.
    pub upsample_initial_channel: usize,
    /// Transposed-conv kernel sizes (per upsample stage).
    pub upsample_kernel_sizes: Vec<usize>,
    /// Speaker embedding count.
    pub spk_embed_dim: usize,
    /// Global conditioning (speaker) width.
    pub gin_channels: usize,
    /// Output sample rate.
    pub sample_rate: usize,
}

impl SynthesizerConfig {
    /// The 48 kHz v2 model (the toolkit default).
    pub fn v2_48k() -> Self {
        Self {
            spec_channels: 1025,
            inter_channels: 192,
            hidden_channels: 192,
            filter_channels: 768,
            n_heads: 2,
            n_layers: 6,
            kernel_size: 3,
            window_size: 10,
            resblock_kernel_sizes: vec![3, 7, 11],
            resblock_dilation_sizes: vec![vec![1, 3, 5], vec![1, 3, 5], vec![1, 3, 5]],
            upsample_rates: vec![12, 10, 2, 2],
            upsample_initial_channel: 512,
            upsample_kernel_sizes: vec![24, 20, 4, 4],
            spk_embed_dim: 109,
            gin_channels: 256,
            sample_rate: 48_000,
        }
    }

    /// The 40 kHz v2 model.
    pub fn v2_40k() -> Self {
        Self {
            spec_channels: 1025,
            upsample_rates: vec![10, 10, 2, 2],
            upsample_initial_channel: 512,
            upsample_kernel_sizes: vec![16, 16, 4, 4],
            sample_rate: 40_000,
            ..Self::v2_48k()
        }
    }

    /// Total upsampling factor (samples per latent frame).
    pub fn hop_length(&self) -> usize {
        self.upsample_rates.iter().product()
    }
}
