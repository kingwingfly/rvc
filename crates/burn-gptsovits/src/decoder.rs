//! The HiFiGAN decoder (`dec`) — latent to waveform.
//!
//! Plain HiFiGAN: `burn-rvc`'s generator without the NSF source module, since
//! GPT-SoVITS does not condition on an F0 contour. The residual blocks come from
//! `burn-vits`; what is written here is the upsample chain around them and the
//! speaker conditioning.
//!
//! Five transposed convolutions multiply out to 640 samples per frame, which is
//! the hop of the 32 kHz spectrogram the rest of `s2` works in — so one latent
//! frame is 20 ms of audio and the rates are not free to change.

use burn::module::Module;
use burn::nn::PaddingConfig1d;
use burn::nn::conv::{Conv1d, Conv1dConfig};
use burn::tensor::Tensor;
use burn::tensor::activation::tanh;
use burn::tensor::backend::Backend;
use burn_vits::{LRELU_SLOPE, ResBlock1, WeightNormConvTranspose1d, leaky_relu};

/// The shape of the decoder.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecoderConfig {
    /// Latent width in — `inter_channels`.
    pub in_channels: usize,
    /// Channels the first upsample starts from, halving at each step.
    pub upsample_initial_channel: usize,
    pub upsample_rates: Vec<usize>,
    pub upsample_kernel_sizes: Vec<usize>,
    pub resblock_kernel_sizes: Vec<usize>,
    pub resblock_dilation_sizes: Vec<Vec<usize>>,
    /// Speaker-conditioning width.
    pub gin_channels: usize,
}

impl Default for DecoderConfig {
    /// v2, 32 kHz.
    fn default() -> Self {
        Self {
            in_channels: 192,
            upsample_initial_channel: 512,
            upsample_rates: vec![10, 8, 2, 2, 2],
            upsample_kernel_sizes: vec![16, 16, 8, 2, 2],
            resblock_kernel_sizes: vec![3, 7, 11],
            resblock_dilation_sizes: vec![vec![1, 3, 5], vec![1, 3, 5], vec![1, 3, 5]],
            gin_channels: 512,
        }
    }
}

impl DecoderConfig {
    /// Audio samples per latent frame — the product of the upsample rates.
    ///
    /// 640 here, matching the spectrogram hop, so the decoder's output lines up
    /// with the posterior encoder's input sample for sample.
    pub fn samples_per_frame(&self) -> usize {
        self.upsample_rates.iter().product()
    }
}

/// HiFiGAN.
#[derive(Module, Debug)]
pub struct Decoder<B: Backend> {
    conv_pre: Conv1d<B>,
    ups: Vec<WeightNormConvTranspose1d<B>>,
    resblocks: Vec<ResBlock1<B>>,
    conv_post: Conv1d<B>,
    cond: Conv1d<B>,
    num_kernels: usize,
}

impl<B: Backend> Decoder<B> {
    pub fn new(cfg: &DecoderConfig, device: &B::Device) -> Self {
        let mut ups = Vec::with_capacity(cfg.upsample_rates.len());
        let mut resblocks = Vec::new();
        for (i, (&rate, &kernel)) in cfg
            .upsample_rates
            .iter()
            .zip(&cfg.upsample_kernel_sizes)
            .enumerate()
        {
            let in_ch = cfg.upsample_initial_channel >> i;
            let out_ch = cfg.upsample_initial_channel >> (i + 1);
            ups.push(WeightNormConvTranspose1d::new(
                in_ch,
                out_ch,
                kernel,
                rate,
                (kernel - rate) / 2,
                device,
            ));
            // Every kernel size gets a block at this width; their outputs are
            // averaged, which is what `num_kernels` divides by.
            for (&k, dilations) in cfg
                .resblock_kernel_sizes
                .iter()
                .zip(&cfg.resblock_dilation_sizes)
            {
                resblocks.push(ResBlock1::new(out_ch, k, dilations, device));
            }
        }

        let last = cfg.upsample_initial_channel >> cfg.upsample_rates.len();
        Self {
            conv_pre: Conv1dConfig::new(cfg.in_channels, cfg.upsample_initial_channel, 7)
                .with_padding(PaddingConfig1d::Explicit(3, 3))
                .init(device),
            ups,
            resblocks,
            conv_post: Conv1dConfig::new(last, 1, 7)
                .with_padding(PaddingConfig1d::Explicit(3, 3))
                .with_bias(false)
                .init(device),
            cond: Conv1dConfig::new(cfg.gin_channels, cfg.upsample_initial_channel, 1).init(device),
            num_kernels: cfg.resblock_kernel_sizes.len(),
        }
    }

    /// `x`: `[batch, in_channels, frames]`, `g`: `[batch, gin, 1]`.
    /// Returns `[batch, 1, frames * samples_per_frame]` in `[-1, 1]`.
    pub fn forward(&self, x: Tensor<B, 3>, g: Tensor<B, 3>) -> Tensor<B, 3> {
        // The speaker vector is added once, before any upsampling — it colours
        // the whole utterance rather than varying along it.
        let mut x = self.conv_pre.forward(x) + self.cond.forward(g);

        for (i, up) in self.ups.iter().enumerate() {
            x = up.forward(leaky_relu(x, LRELU_SLOPE));
            // Average the parallel residual blocks for this stage.
            let blocks = &self.resblocks[i * self.num_kernels..(i + 1) * self.num_kernels];
            let mut sum = blocks[0].forward(x.clone());
            for block in &blocks[1..] {
                sum = sum + block.forward(x.clone());
            }
            x = sum / self.num_kernels as f64;
        }

        tanh(self.conv_post.forward(leaky_relu(x, LRELU_SLOPE)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    type B = burn_ndarray::NdArray;

    #[test]
    fn one_latent_frame_becomes_exactly_one_hop_of_audio() {
        // 640 samples at 32 kHz is 20 ms, and it has to equal the spectrogram
        // hop the posterior encoder consumes — the two ends of the model are
        // trained against each other sample for sample.
        let cfg = DecoderConfig::default();
        assert_eq!(cfg.samples_per_frame(), 640);

        let device = Default::default();
        let dec = Decoder::<B>::new(&cfg, &device);
        let frames = 3;
        let x = Tensor::zeros([1, cfg.in_channels, frames], &device);
        let g = Tensor::zeros([1, cfg.gin_channels, 1], &device);
        assert_eq!(
            dec.forward(x, g).dims(),
            [1, 1, frames * cfg.samples_per_frame()]
        );
    }

    #[test]
    fn the_output_is_bounded_like_audio() {
        // `tanh` at the end is what makes it a waveform rather than an
        // unnormalised signal; losing it clips instead of failing.
        let cfg = DecoderConfig::default();
        let device = Default::default();
        let dec = Decoder::<B>::new(&cfg, &device);
        let x = Tensor::random(
            [1, cfg.in_channels, 4],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        let g = Tensor::zeros([1, cfg.gin_channels, 1], &device);
        let out: Vec<f32> = dec.forward(x, g).into_data().to_vec().unwrap();
        assert!(
            out.iter()
                .all(|v| v.is_finite() && (-1.0..=1.0).contains(v))
        );
    }
}
