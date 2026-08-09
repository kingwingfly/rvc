//! BigVGAN — 80-band mel in at 22.05 kHz, waveform out.
//!
//! The last stage of the signal path and the only one Seed-VC does not train:
//! `nvidia/bigvgan_v2_22khz_80band_256x` is used exactly as NVIDIA released it,
//! **from its own Hugging Face repo** ([`crate::config::BIGVGAN_REPO`]) rather
//! than from the Seed-VC checkpoint, which contains none of these tensors. Its
//! upsample rates multiply to 256, matching the mel hop, so one mel frame becomes
//! 256 samples and the frame grid the rest of the crate agrees on carries through
//! to the waveform.
//!
//! Structurally this is HiFi-GAN: a width-7 pre-conv, six transposed-conv
//! upsample stages, three residual blocks averaged after each, and a width-7
//! post-conv down to one channel. Two things make it BigVGAN, and both are easy
//! to get silently wrong.
//!
//! # The snake activation, and which variant this is
//!
//! Every leaky-ReLU of HiFi-GAN is replaced by a periodic activation with
//! **learned per-channel parameters**. The parameterisation is read off
//! `config.json` in the weight repo rather than inferred: `"activation":
//! "snakebeta"` with `"snake_logscale": true`, so the function is
//!
//! ```text
//! snakebeta(x) = x + (1 / (exp(β) + 1e-9)) · sin²(exp(α) · x)
//! ```
//!
//! with α and β separate `[channels]` tensors — **not** the plain `Snake`, whose
//! single α appears in both places. The checkpoint agrees independently: every
//! activation site carries an `alpha` *and* a `beta` of equal width, which the
//! α-only variant would not. Two ways to get this wrong load at 100% and produce
//! a plausible-looking waveform — using α where β belongs, which is right only
//! where the two coincide and after training they do not, and dropping the `exp`,
//! which on a log-scale checkpoint whose values sit near zero turns a frequency
//! of ~1 into ~0 and multiplies the periodic term by roughly 1e9.
//!
//! # Anti-aliasing, and where its filters come from
//!
//! sin² doubles the bandwidth of whatever it is fed, so applying it at the signal
//! rate folds everything above Nyquist back down as audible aliasing. BigVGAN
//! wraps each activation in 2× upsample → activate → 2× downsample, both
//! resampling steps being grouped convolutions with a Kaiser-windowed sinc
//! low-pass (cutoff 0.25, transition half-width 0.3, 12 taps). That wrapper is
//! the "anti-aliased multi-periodicity" the paper is named for.
//!
//! **Those kernels are computed rather than learned — and this checkpoint stores
//! them anyway.** Upstream registers each with `register_buffer`, which is
//! persistent by default, so all 218 are in the file: 18 residual blocks × 6
//! activations × 2 resamplers, plus the post-activation's pair. Being a
//! deterministic function of (cutoff, half-width, taps), all 218 are the same
//! twelve numbers. [`UpSample1d::new`] and [`DownSample1d::new`] therefore
//! *derive* them, so the module is correct with no checkpoint at all, and then
//! let the file overwrite them — which keeps the coverage report at zero unused
//! and turns the stored copies into a free check on the derivation. That check is
//! not left implicit: `examples/load` snapshots a derived kernel, loads, and
//! prints the largest disagreement.
//!
//! # What is reused and what is not
//!
//! The weight-normalised convolutions are `burn-vits`'s, unchanged, because
//! PyTorch's `weight_g`/`weight_v` parameterisation is identical here. The
//! residual block is **not**: `burn_vits::ResBlock1` has the same two conv stacks
//! but hard-codes leaky-ReLU between them and carries no activation parameters,
//! so it cannot hold this checkpoint's `activations.*` subtree. Bending it would
//! change a block RVC and GPT-SoVITS both depend on, for a model neither of them
//! runs; [`AmpBlock1`] lives here instead. The final convolution is local for a
//! smaller reason of the same shape: `use_bias_at_final: false` means the
//! checkpoint has no `conv_post.bias`, and `WeightNormConv1d` always has one.
//!
//! # Provenance
//!
//! Ported from `modules/bigvgan/` of Seed-VC
//! (<https://github.com/Plachtaa/seed-vc>, GPL-3.0), which vendors NVIDIA's
//! BigVGAN, read as a reference and never run or vendored. **BigVGAN itself is
//! MIT** and the alias-free resampling it borrows is Apache-2.0; both are
//! compatible with this workspace's GPL-3.0, which comes from Seed-VC.

use std::error::Error;
use std::f64::consts::PI;
use std::path::Path;

use burn::module::{Module, Param};
use burn::tensor::backend::Backend;
use burn::tensor::module::{conv_transpose1d, conv1d};
use burn::tensor::ops::{ConvOptions, ConvTransposeOptions};
use burn::tensor::{Distribution, Tensor, TensorData};
use burn_store::ApplyResult;
use burn_vits::{WeightNormConv1d, WeightNormConvTranspose1d, get_padding};

/// Every dimension of the released vocoder, read from its `config.json`.
///
/// Read rather than inferred from tensor shapes, for the reason
/// [`crate::config`] gives: a shape says what fits, not what was meant.
/// `use_tanh_at_final` and `use_bias_at_final` have no shape to infer them from
/// at all — the first picks the output nonlinearity, the second decides whether
/// a tensor exists.
#[derive(Debug, Clone)]
pub struct BigVganConfig {
    /// Mel bands in, which fixes the pre-conv's input width.
    pub num_mels: usize,
    /// Channels the pre-conv widens to; each upsample stage halves it.
    pub upsample_initial_channel: usize,
    /// Per-stage stride. Their product is the samples-per-mel-frame ratio.
    pub upsample_rates: Vec<usize>,
    /// Per-stage transposed-conv kernel, one per entry of `upsample_rates`.
    pub upsample_kernel_sizes: Vec<usize>,
    /// Kernel of each residual block after a stage — three blocks, averaged.
    pub resblock_kernel_sizes: Vec<usize>,
    /// Dilations within each of those blocks, one row per kernel size.
    pub resblock_dilation_sizes: Vec<Vec<usize>>,
    /// `tanh` at the output, or a hard clamp to ±1. **False on v2** — these
    /// weights were trained against the clamp, and `tanh` would compress every
    /// peak instead of passing it.
    pub use_tanh_at_final: bool,
    /// Whether the final convolution has a bias. False on v2, so the tensor is
    /// absent from the checkpoint rather than zero.
    pub use_bias_at_final: bool,
    /// Whether α and β are stored as their logarithms. True on v2.
    pub snake_logscale: bool,
}

impl BigVganConfig {
    /// `nvidia/bigvgan_v2_22khz_80band_256x`, verbatim from its `config.json`.
    ///
    /// The upsample rates multiply to 256 — the `256x` of the name, and the hop
    /// [`crate::config::SeedVcConfig`] pairs with it.
    pub fn v2_22khz_80band_256x() -> Self {
        Self {
            num_mels: 80,
            upsample_initial_channel: 1536,
            upsample_rates: vec![4, 4, 2, 2, 2, 2],
            upsample_kernel_sizes: vec![8, 8, 4, 4, 4, 4],
            resblock_kernel_sizes: vec![3, 7, 11],
            resblock_dilation_sizes: vec![vec![1, 3, 5], vec![1, 3, 5], vec![1, 3, 5]],
            use_tanh_at_final: false,
            use_bias_at_final: false,
            snake_logscale: true,
        }
    }

    /// Samples of waveform per mel frame — the product of the upsample rates.
    pub fn hop(&self) -> usize {
        self.upsample_rates.iter().product()
    }
}

/// Ratio the anti-aliasing wrapper resamples by, up and down.
const AA_RATIO: usize = 2;
/// Taps in the anti-aliasing low-pass — upstream's `Activation1d` default, which
/// nothing in BigVGAN overrides.
const AA_TAPS: usize = 12;

// --- the snake activation ---------------------------------------------------

/// `x + (1/β)·sin²(αx)` with α and β learned per channel.
///
/// See the module docs for why this is `SnakeBeta` rather than `Snake`, and why
/// the exponential is not optional.
#[derive(Module, Debug)]
pub struct SnakeBeta<B: Backend> {
    alpha: Param<Tensor<B, 1>>,
    beta: Param<Tensor<B, 1>>,
    logscale: bool,
}

impl<B: Backend> SnakeBeta<B> {
    fn new(channels: usize, logscale: bool, device: &B::Device) -> Self {
        // Upstream's own initialisation: zeros in log space, ones in linear —
        // both meaning "gain 1", so an unloaded module is `x + sin²(x)` rather
        // than something degenerate.
        let init = if logscale {
            Tensor::zeros([channels], device)
        } else {
            Tensor::ones([channels], device)
        };
        Self {
            alpha: Param::from_tensor(init.clone()),
            beta: Param::from_tensor(init),
            logscale,
        }
    }

    /// `x`: `[batch, channels, time]`, unchanged in shape.
    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let channels = x.dims()[1];
        let (mut alpha, mut beta) = (
            self.alpha.val().reshape([1, channels, 1]),
            self.beta.val().reshape([1, channels, 1]),
        );
        if self.logscale {
            alpha = alpha.exp();
            beta = beta.exp();
        }
        // The 1e-9 is upstream's and is load-bearing rather than cosmetic: β is
        // an exponential of a learned value and nothing stops it underflowing to
        // zero, which would divide the periodic term by zero.
        let periodic = (x.clone() * alpha).sin().powi_scalar(2);
        x + periodic / (beta + 1e-9)
    }
}

// --- alias-free resampling --------------------------------------------------

/// Edge-value padding (`F.pad(..., mode="replicate")`), which is what the
/// resamplers use rather than zeros: a zero-padded signal has a step at each end,
/// and a 12-tap sinc turns that step into ringing inside the output.
fn replicate_pad<B: Backend>(x: Tensor<B, 3>, left: usize, right: usize) -> Tensor<B, 3> {
    let [b, c, t] = x.dims();
    let first = x.clone().slice([0..b, 0..c, 0..1]).expand([b, c, left]);
    let last = x
        .clone()
        .slice([0..b, 0..c, (t - 1)..t])
        .expand([b, c, right]);
    Tensor::cat(vec![first, x, last], 2)
}

/// A Kaiser-windowed sinc low-pass, unit sum, `taps` long.
///
/// Julius's `LowPassFilters` design as upstream vendors it: β comes from the
/// transition width by Kaiser's own attenuation estimate. The normalisation is
/// not cosmetic — without it the filter passes a constant at a gain slightly off
/// 1, and that error compounds over the 218 resampling sites in one forward pass.
fn kaiser_sinc_filter1d(cutoff: f64, half_width: f64, taps: usize) -> Vec<f32> {
    let half = taps / 2;
    let atten = 2.285 * (half as f64 - 1.0) * PI * (4.0 * half_width) + 7.95;
    let beta = if atten > 50.0 {
        0.1102 * (atten - 8.7)
    } else if atten >= 21.0 {
        0.5842 * (atten - 21.0).powf(0.4) + 0.07886 * (atten - 21.0)
    } else {
        0.0
    };

    // `torch.kaiser_window(taps, periodic=False, beta)` — symmetric, so the
    // argument spans [-1, 1] over `taps - 1` steps rather than `taps`.
    let denom = bessel_i0(beta);
    let window = (0..taps).map(|i| {
        let r = 2.0 * i as f64 / (taps as f64 - 1.0) - 1.0;
        bessel_i0(beta * (1.0 - r * r).max(0.0).sqrt()) / denom
    });

    // An even tap count has no centre sample, so its grid sits half a step off;
    // an odd one is centred on zero.
    let offset = if taps % 2 == 0 { 0.5 } else { 0.0 };
    let mut filter: Vec<f64> = window
        .enumerate()
        .map(|(i, w)| {
            let t = 2.0 * cutoff * (i as f64 - half as f64 + offset);
            let sinc = if t == 0.0 {
                1.0
            } else {
                (PI * t).sin() / (PI * t)
            };
            2.0 * cutoff * w * sinc
        })
        .collect();

    let sum: f64 = filter.iter().sum();
    for tap in &mut filter {
        *tap /= sum;
    }
    filter.into_iter().map(|tap| tap as f32).collect()
}

/// Modified Bessel function of the first kind, order 0, by its power series.
///
/// `I0(x) = Σ (x²/4)ᵏ / (k!)²`. The argument here is at most the Kaiser β —
/// ≈4.66 for this filter — where the series converges in a couple of dozen terms.
/// The loop stops on the term rather than on a fixed count, so it stays honest if
/// the filter is ever redesigned.
fn bessel_i0(x: f64) -> f64 {
    let (mut sum, mut term) = (1.0, 1.0);
    for k in 1..64 {
        let step = x / (2.0 * k as f64);
        term *= step * step;
        sum += term;
        if term < 1e-18 * sum {
            break;
        }
    }
    sum
}

/// The resamplers' shared kernel, as a `[1, 1, taps]` parameter.
fn filter_param<B: Backend>(taps: usize, device: &B::Device) -> Param<Tensor<B, 3>> {
    let coefficients = kaiser_sinc_filter1d(0.5 / AA_RATIO as f64, 0.6 / AA_RATIO as f64, taps);
    let tensor = Tensor::from_data(TensorData::new(coefficients, [1, 1, taps]), device);
    // A buffer upstream, so it is not trained here either; it is a `Param` only
    // because that is how a Burn module holds a tensor the store can fill.
    Param::from_tensor(tensor).set_require_grad(false)
}

/// 2× interpolation through the low-pass, as one transposed convolution.
#[derive(Module, Debug)]
pub struct UpSample1d<B: Backend> {
    filter: Param<Tensor<B, 3>>,
    taps: usize,
    /// Replicate padding applied first, in *input* samples.
    pad: usize,
    /// Output samples trimmed either side afterwards, so the result is exactly
    /// `AA_RATIO ×` the input length.
    trim_left: usize,
    trim_right: usize,
}

impl<B: Backend> UpSample1d<B> {
    fn new(taps: usize, device: &B::Device) -> Self {
        let pad = taps / AA_RATIO - 1;
        Self {
            filter: filter_param(taps, device),
            taps,
            pad,
            trim_left: pad * AA_RATIO + (taps - AA_RATIO) / 2,
            trim_right: pad * AA_RATIO + (taps - AA_RATIO).div_ceil(2),
        }
    }

    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [b, c, _] = x.dims();
        let x = replicate_pad(x, self.pad, self.pad);
        // One kernel shared by every channel, so the convolution is grouped and
        // the filter is broadcast rather than stored per channel.
        let weight = self.filter.val().expand([c, 1, self.taps]);
        let y = conv_transpose1d(
            x,
            weight,
            None,
            ConvTransposeOptions::new([AA_RATIO], [0], [0], [1], c),
        );
        // Interpolation spreads each input sample over `AA_RATIO` outputs, so the
        // filter's unit sum has to be undone to keep the amplitude.
        let y = y * AA_RATIO as f32;
        let len = y.dims()[2];
        y.slice([0..b, 0..c, self.trim_left..(len - self.trim_right)])
    }
}

/// 2× decimation through the same low-pass, as one strided convolution.
#[derive(Module, Debug)]
pub struct DownSample1d<B: Backend> {
    filter: Param<Tensor<B, 3>>,
    taps: usize,
    pad_left: usize,
    pad_right: usize,
}

impl<B: Backend> DownSample1d<B> {
    fn new(taps: usize, device: &B::Device) -> Self {
        Self {
            filter: filter_param(taps, device),
            taps,
            // Asymmetric for an even tap count, because the filter's centre falls
            // between two samples.
            pad_left: taps / 2 - usize::from(taps % 2 == 0),
            pad_right: taps / 2,
        }
    }

    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let c = x.dims()[1];
        let x = replicate_pad(x, self.pad_left, self.pad_right);
        let weight = self.filter.val().expand([c, 1, self.taps]);
        conv1d(x, weight, None, ConvOptions::new([AA_RATIO], [0], [1], c))
    }
}

/// upsample → activate → downsample: the periodic activation applied where it
/// cannot alias.
#[derive(Module, Debug)]
pub struct AliasFreeActivation<B: Backend> {
    act: SnakeBeta<B>,
    upsample: UpSample1d<B>,
    downsample: DownSample1d<B>,
}

impl<B: Backend> AliasFreeActivation<B> {
    fn new(channels: usize, logscale: bool, device: &B::Device) -> Self {
        Self {
            act: SnakeBeta::new(channels, logscale, device),
            upsample: UpSample1d::new(AA_TAPS, device),
            downsample: DownSample1d::new(AA_TAPS, device),
        }
    }

    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        self.downsample
            .forward(self.act.forward(self.upsample.forward(x)))
    }
}

// --- the residual block and the generator -----------------------------------

/// BigVGAN's `AMPBlock1`: HiFi-GAN's ResBlock1 conv stacks with an alias-free
/// snake wherever the leaky-ReLU was.
///
/// The activations are one flat list of `2 × dilations` because the checkpoint
/// indexes them that way. The forward pass takes the even entries before the
/// dilated convolutions and the odd ones before the width-1 convolutions —
/// upstream's `activations[::2]` and `activations[1::2]`.
#[derive(Module, Debug)]
pub struct AmpBlock1<B: Backend> {
    convs1: Vec<WeightNormConv1d<B>>,
    convs2: Vec<WeightNormConv1d<B>>,
    activations: Vec<AliasFreeActivation<B>>,
}

impl<B: Backend> AmpBlock1<B> {
    fn new(
        channels: usize,
        kernel: usize,
        dilations: &[usize],
        logscale: bool,
        device: &B::Device,
    ) -> Self {
        Self {
            convs1: dilations
                .iter()
                .map(|&d| {
                    WeightNormConv1d::new(
                        channels,
                        channels,
                        kernel,
                        1,
                        get_padding(kernel, d),
                        d,
                        device,
                    )
                })
                .collect(),
            convs2: dilations
                .iter()
                .map(|_| {
                    WeightNormConv1d::new(
                        channels,
                        channels,
                        kernel,
                        1,
                        get_padding(kernel, 1),
                        1,
                        device,
                    )
                })
                .collect(),
            activations: (0..2 * dilations.len())
                .map(|_| AliasFreeActivation::new(channels, logscale, device))
                .collect(),
        }
    }

    /// `x`: `[batch, channels, time]`, unchanged in shape.
    fn forward(&self, mut x: Tensor<B, 3>) -> Tensor<B, 3> {
        for (i, (c1, c2)) in self.convs1.iter().zip(self.convs2.iter()).enumerate() {
            let xt = self.activations[2 * i].forward(x.clone());
            let xt = c1.forward(xt);
            let xt = self.activations[2 * i + 1].forward(xt);
            x = x + c2.forward(xt);
        }
        x
    }
}

/// `weight_norm(Conv1d(..., bias=False))` — the output convolution.
///
/// Local rather than `burn_vits::WeightNormConv1d` because that one always
/// carries a bias and this checkpoint has none: `use_bias_at_final: false`. A
/// biased conv would load with one tensor missing and be right only by virtue of
/// the zero initialiser, which is exactly the silence the coverage numbers exist
/// to break.
#[derive(Module, Debug)]
pub struct WeightNormConv1dNoBias<B: Backend> {
    weight_g: Param<Tensor<B, 3>>,
    weight_v: Param<Tensor<B, 3>>,
    padding: usize,
}

impl<B: Backend> WeightNormConv1dNoBias<B> {
    fn new(in_ch: usize, out_ch: usize, kernel: usize, padding: usize, device: &B::Device) -> Self {
        let v = Tensor::random(
            [out_ch, in_ch, kernel],
            Distribution::Normal(0.0, 0.02),
            device,
        );
        let g = v.clone().powf_scalar(2.0).sum_dim(2).sum_dim(1).sqrt();
        Self {
            weight_g: Param::from_tensor(g),
            weight_v: Param::from_tensor(v),
            padding,
        }
    }

    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let v = self.weight_v.val();
        let norm = v.clone().powf_scalar(2.0).sum_dim(2).sum_dim(1).sqrt();
        let weight = v * (self.weight_g.val() / norm);
        conv1d(
            x,
            weight,
            None,
            ConvOptions::new([1], [self.padding], [1], 1),
        )
    }
}

/// The vocoder.
#[derive(Module, Debug)]
pub struct BigVgan<B: Backend> {
    conv_pre: WeightNormConv1d<B>,
    ups: Vec<WeightNormConvTranspose1d<B>>,
    resblocks: Vec<AmpBlock1<B>>,
    activation_post: AliasFreeActivation<B>,
    conv_post: WeightNormConv1dNoBias<B>,
    num_kernels: usize,
    use_tanh_at_final: bool,
}

impl<B: Backend> BigVgan<B> {
    pub fn new(cfg: &BigVganConfig, device: &B::Device) -> Self {
        assert!(
            !cfg.use_bias_at_final,
            "only the bias-free output convolution is modelled; \
             `use_bias_at_final: true` would need `burn_vits::WeightNormConv1d` here"
        );

        let initial = cfg.upsample_initial_channel;
        let ups = cfg
            .upsample_rates
            .iter()
            .zip(&cfg.upsample_kernel_sizes)
            .enumerate()
            .map(|(i, (&rate, &kernel))| {
                WeightNormConvTranspose1d::new(
                    initial >> i,
                    initial >> (i + 1),
                    kernel,
                    rate,
                    (kernel - rate) / 2,
                    device,
                )
            })
            .collect();

        // Three blocks per stage, stored flat: the checkpoint's `resblocks.7` is
        // stage 2's middle block, not stage 7's.
        let mut resblocks = Vec::new();
        for i in 0..cfg.upsample_rates.len() {
            let channels = initial >> (i + 1);
            for (&kernel, dilations) in cfg
                .resblock_kernel_sizes
                .iter()
                .zip(&cfg.resblock_dilation_sizes)
            {
                resblocks.push(AmpBlock1::new(
                    channels,
                    kernel,
                    dilations,
                    cfg.snake_logscale,
                    device,
                ));
            }
        }

        let final_channels = initial >> cfg.upsample_rates.len();
        Self {
            conv_pre: WeightNormConv1d::new(cfg.num_mels, initial, 7, 1, 3, 1, device),
            ups,
            resblocks,
            activation_post: AliasFreeActivation::new(final_channels, cfg.snake_logscale, device),
            conv_post: WeightNormConv1dNoBias::new(final_channels, 1, 7, 3, device),
            num_kernels: cfg.resblock_kernel_sizes.len(),
            use_tanh_at_final: cfg.use_tanh_at_final,
        }
    }

    /// `mel`: `[batch, num_mels, frames]` → `[batch, 1, frames · hop]`.
    ///
    /// The mel is the **natural-log** one BigVGAN trains against: magnitude
    /// spectrogram through a Slaney filterbank, clamped at 1e-5 and logged, which
    /// is exactly what `burn_vits::Spectral::mel` produces at n_fft 1024 / hop
    /// 256. Feeding it a dB-scaled or power mel is the failure that sounds like a
    /// broken vocoder and is not.
    pub fn forward(&self, mel: Tensor<B, 3>) -> Tensor<B, 3> {
        let mut x = self.conv_pre.forward(mel);

        for (i, up) in self.ups.iter().enumerate() {
            x = up.forward(x);
            // The three blocks are averaged, not summed: they are alternative
            // receptive fields over the same signal, and summing would triple the
            // level going into the next stage.
            let blocks = &self.resblocks[i * self.num_kernels..(i + 1) * self.num_kernels];
            let mut acc = blocks[0].forward(x.clone());
            for block in &blocks[1..] {
                acc = acc + block.forward(x.clone());
            }
            x = acc / self.num_kernels as f32;
        }

        let x = self.conv_post.forward(self.activation_post.forward(x));
        if self.use_tanh_at_final {
            x.tanh()
        } else {
            x.clamp(-1.0, 1.0)
        }
    }

    /// Load `bigvgan_generator.pt` from the weight repo.
    ///
    /// Three differences from the rest of this crate, all because these weights
    /// are somebody else's release:
    ///
    /// - the file is **not** the Seed-VC checkpoint, so there is no `net.` prefix
    ///   and no other module's tensors for a coverage report to filter out,
    /// - its state dict sits under a `"generator"` key, beside training
    ///   bookkeeping this port has no use for,
    /// - two remaps flatten levels that exist upstream purely as `nn.Module`
    ///   nesting and carry no tensors of their own: each upsample stage is a
    ///   one-element `ModuleList` inside a `ModuleList`, and each downsampler
    ///   wraps its filter in a `LowPassFilter1d`.
    pub fn load_pytorch(&mut self, path: impl AsRef<Path>) -> Result<ApplyResult, Box<dyn Error>> {
        let mut remaps = vec![
            (r"^ups\.(\d+)\.0\.", "ups.${1}."),
            (r"\.downsample\.lowpass\.filter$", ".downsample.filter"),
        ];
        // The vocoder is the likeliest of the three to change spelling: it is
        // NVIDIA's release rather than Seed-VC's, so a re-publish from a newer
        // torch is somebody else's decision entirely.
        remaps.extend(crate::WEIGHT_NORM_REMAPS);
        burn_kit::store::load_pytorch_into::<B, _>(self, path.as_ref(), Some("generator"), &remaps)
    }

    /// The anti-aliasing kernel this module derived, so `examples/load` can hold
    /// it against the copies the checkpoint stores.
    pub fn derived_filter(&self) -> Vec<f32> {
        self.activation_post
            .upsample
            .filter
            .val()
            .into_data()
            .to_vec()
            .expect("the filter is f32")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    type B = burn_ndarray::NdArray;

    #[test]
    fn the_low_pass_is_symmetric_and_passes_a_constant() {
        // Both properties are what the design is *for*, and each breaks under one
        // of the two mistakes that are easy to make here: an off-by-one in the
        // sample grid destroys the symmetry, and forgetting the sum normalisation
        // changes the gain a constant sees.
        let taps = kaiser_sinc_filter1d(0.5 / AA_RATIO as f64, 0.6 / AA_RATIO as f64, AA_TAPS);
        assert_eq!(taps.len(), AA_TAPS);
        for (a, b) in taps.iter().zip(taps.iter().rev()) {
            assert!((a - b).abs() < 1e-7, "asymmetric: {a} vs {b}");
        }
        let sum: f32 = taps.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "gain {sum}");
    }

    #[test]
    fn resampling_round_trips_a_length() {
        // The trims either side of the transposed convolution are the fiddliest
        // arithmetic in the file, and getting them wrong slides every activation
        // against the signal it is applied to rather than erroring.
        let device = Default::default();
        let up = UpSample1d::<B>::new(AA_TAPS, &device);
        let down = DownSample1d::<B>::new(AA_TAPS, &device);

        let x = Tensor::<B, 3>::random([2, 3, 37], Distribution::Normal(0.0, 1.0), &device);
        let wide = up.forward(x);
        assert_eq!(wide.dims(), [2, 3, 74]);
        assert_eq!(down.forward(wide).dims(), [2, 3, 37]);
    }

    #[test]
    fn upsampling_a_constant_leaves_it_constant() {
        // Interpolation gain and replicate padding at once: a DC signal must come
        // back at the same level, edges included. A missing `× AA_RATIO` halves
        // it, and zero padding dips the ends.
        let device = Default::default();
        let up = UpSample1d::<B>::new(AA_TAPS, &device);
        let out: Vec<f32> = up
            .forward(Tensor::ones([1, 1, 24], &device))
            .into_data()
            .to_vec()
            .unwrap();
        assert_eq!(out.len(), 48);
        for v in out {
            assert!((v - 1.0).abs() < 1e-3, "{v}");
        }
    }

    #[test]
    fn snake_beta_is_the_identity_plus_sin_squared_at_the_zero_initialisation() {
        // exp(0) = 1 in both places, so the activation reduces to x + sin²(x).
        // This is the test that catches a missing `exp`: without it α and β are 0,
        // the periodic term becomes 0/1e-9, and every sample comes out as x.
        let device = Default::default();
        let act = SnakeBeta::<B>::new(4, true, &device);
        let inputs = [0.0f32, 0.5, 1.0, -1.0];
        let x = Tensor::<B, 3>::from_data(TensorData::new(inputs.to_vec(), [1, 4, 1]), &device);
        let y: Vec<f32> = act.forward(x).into_data().to_vec().unwrap();
        for (got, input) in y.iter().zip(inputs) {
            let want = input + input.sin().powi(2);
            assert!((got - want).abs() < 1e-6, "{got} vs {want}");
        }
    }

    #[test]
    fn a_mel_frame_becomes_exactly_one_hop_of_samples() {
        // The contract the rest of the crate depends on: 22050/256 mel frames a
        // second in, 22050 samples a second out. Built at a toy width, because a
        // real BigVGAN is 122 M parameters and this checks only the arithmetic of
        // the upsample chain. It still has to survive six halvings: 128 leaves the
        // output convolution two channels, where 32 would leave it zero and fail
        // inside `expand` rather than anywhere informative.
        let cfg = BigVganConfig {
            upsample_initial_channel: 128,
            ..BigVganConfig::v2_22khz_80band_256x()
        };
        let device = Default::default();
        let model = BigVgan::<B>::new(&cfg, &device);

        let frames = 5;
        let out = model.forward(Tensor::zeros([1, cfg.num_mels, frames], &device));
        assert_eq!(cfg.hop(), 256);
        assert_eq!(out.dims(), [1, 1, frames * cfg.hop()]);
        // Asserted separately, because Burn's approximate comparisons call NaN
        // equal to NaN and a NaN here would reach every output sample in silence.
        assert!(!out.contains_nan().into_scalar());
    }
}
