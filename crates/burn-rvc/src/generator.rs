//! The NSF-HiFiGAN decoder (`GeneratorNSF`) and its pieces.

use std::f32::consts::PI;

use burn::module::Module;
use burn::nn::conv::{Conv1d, Conv1dConfig};
use burn::nn::{Linear, LinearConfig, PaddingConfig1d};
use burn::tensor::activation::tanh;
use burn::tensor::backend::Backend;
use burn::tensor::{Distribution, Tensor, TensorData};

use crate::config::SynthesizerConfig;
use crate::nn::leaky_relu;
use crate::weightnorm::{WeightNormConv1d, WeightNormConvTranspose1d};

const LRELU_SLOPE: f64 = 0.1;
const SINE_AMP: f32 = 0.1;
const NOISE_STD: f32 = 0.003;
const VOICED_THRESHOLD: f32 = 0.0;

fn get_padding(kernel: usize, dilation: usize) -> usize {
    dilation * (kernel - 1) / 2
}

/// HiFi-GAN ResBlock type 1: two dilated conv stacks with residual adds.
#[derive(Module, Debug)]
pub struct ResBlock1<B: Backend> {
    convs1: Vec<WeightNormConv1d<B>>,
    convs2: Vec<WeightNormConv1d<B>>,
}

impl<B: Backend> ResBlock1<B> {
    fn new(channels: usize, kernel: usize, dilations: &[usize], device: &B::Device) -> Self {
        let convs1 = dilations
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
            .collect();
        let convs2 = dilations
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
            .collect();
        Self { convs1, convs2 }
    }

    /// `x`: `[batch, channels, time]`.
    pub fn forward(&self, mut x: Tensor<B, 3>) -> Tensor<B, 3> {
        for (c1, c2) in self.convs1.iter().zip(self.convs2.iter()) {
            let xt = leaky_relu(x.clone(), LRELU_SLOPE);
            let xt = c1.forward(xt);
            let xt = leaky_relu(xt, LRELU_SLOPE);
            let xt = c2.forward(xt);
            x = x + xt;
        }
        x
    }
}

/// The NSF harmonic source (`SourceModuleHnNSF`). Only `l_linear` is learnable;
/// the sine generation is deterministic.
#[derive(Module, Debug)]
pub struct SourceModule<B: Backend> {
    l_linear: Linear<B>,
    sampling_rate: usize,
}

impl<B: Backend> SourceModule<B> {
    fn new(sampling_rate: usize, device: &B::Device) -> Self {
        // harmonic_num = 0 -> dim = 1.
        Self {
            l_linear: LinearConfig::new(1, 1).init(device),
            sampling_rate,
        }
    }

    /// `f0`: `[batch, time]` (Hz) → harmonic excitation `[batch, 1, time·upp]`.
    ///
    /// The sine bank is computed as a constant (RVC runs `SineGen` under
    /// `no_grad`); only `l_linear` is differentiable.
    pub fn forward(&self, f0: Tensor<B, 2>, upp: usize) -> Tensor<B, 3> {
        let [b, t] = f0.dims();
        let l = t * upp;
        let device = f0.device();
        let f0_data = f0.to_data().to_vec::<f32>().expect("f0 must be f32");

        let mut sine_all = Vec::with_capacity(b * l);
        let mut uv_all = Vec::with_capacity(b * l);
        for row in 0..b {
            let (sine, uv) =
                sine_excitation(&f0_data[row * t..row * t + t], self.sampling_rate, upp);
            sine_all.extend(sine);
            uv_all.extend(uv);
        }

        let sine = Tensor::<B, 2>::from_data(TensorData::new(sine_all, [b, l]), &device);
        let uv = Tensor::<B, 2>::from_data(TensorData::new(uv_all, [b, l]), &device);
        // noise_amp = uv·noise_std + (1-uv)·amp/3
        let one_minus_uv = uv.clone().mul_scalar(-1.0).add_scalar(1.0);
        let noise_amp = uv.clone().mul_scalar(NOISE_STD as f64)
            + one_minus_uv.mul_scalar((SINE_AMP / 3.0) as f64);
        let noise =
            Tensor::<B, 2>::random([b, l], Distribution::Normal(0.0, 1.0), &device) * noise_amp;
        let source = (sine * uv + noise).reshape([b, l, 1]);

        let merged = tanh(self.l_linear.forward(source)); // [b, l, 1]
        merged.swap_dims(1, 2) // [b, 1, l]
    }
}

/// Deterministic sine source for one F0 contour (RVC's `SineGen`, harmonic_num=0).
/// Returns `(sine, uv)` each of length `time·upp`.
fn sine_excitation(f0: &[f32], sampling_rate: usize, upp: usize) -> (Vec<f32>, Vec<f32>) {
    let t = f0.len();
    let l = t * upp;
    let sr = sampling_rate as f32;

    // Per-frame instantaneous phase increment and its cumulative sum.
    let mut rad = vec![0f32; t];
    let mut cum = vec![0f32; t];
    let mut acc = 0f32;
    for i in 0..t {
        let r = (f0[i] / sr).rem_euclid(1.0);
        rad[i] = r;
        acc += r;
        cum[i] = acc * upp as f32;
    }

    // Upsample: cumulative phase linearly (align_corners), rad nearest.
    let cum_up = linear_interp_align_corners(&cum, l);
    let rad_up = nearest_interp(&rad, upp);

    // Detect phase wraps in the upsampled cumulative phase.
    let mut phase = 0f32;
    let mut prev = cum_up.first().copied().unwrap_or(0.0).rem_euclid(1.0);
    let mut sine = vec![0f32; l];
    for j in 0..l {
        let cur = cum_up[j].rem_euclid(1.0);
        let shift = if j > 0 && cur - prev < 0.0 { -1.0 } else { 0.0 };
        prev = cur;
        phase += rad_up[j] + shift;
        sine[j] = (phase * 2.0 * PI).sin() * SINE_AMP;
    }

    let uv_frame: Vec<f32> = f0
        .iter()
        .map(|&f| (f > VOICED_THRESHOLD) as i32 as f32)
        .collect();
    let uv = nearest_interp(&uv_frame, upp);
    (sine, uv)
}

/// Nearest-neighbour upsample by integer factor `upp`.
fn nearest_interp(x: &[f32], upp: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(x.len() * upp);
    for &v in x {
        for _ in 0..upp {
            out.push(v);
        }
    }
    out
}

/// Linear upsample to length `l` with `align_corners=True`.
fn linear_interp_align_corners(x: &[f32], l: usize) -> Vec<f32> {
    let t = x.len();
    if t == 1 {
        return vec![x[0]; l];
    }
    let scale = (t - 1) as f32 / (l - 1) as f32;
    (0..l)
        .map(|j| {
            let pos = j as f32 * scale;
            let i0 = pos.floor() as usize;
            let i1 = (i0 + 1).min(t - 1);
            let frac = pos - i0 as f32;
            x[i0] * (1.0 - frac) + x[i1] * frac
        })
        .collect()
}

/// The NSF-HiFiGAN decoder (`dec`).
#[derive(Module, Debug)]
pub struct GeneratorNsf<B: Backend> {
    conv_pre: Conv1d<B>,
    m_source: SourceModule<B>,
    ups: Vec<WeightNormConvTranspose1d<B>>,
    noise_convs: Vec<Conv1d<B>>,
    resblocks: Vec<ResBlock1<B>>,
    conv_post: Conv1d<B>,
    cond: Conv1d<B>,
    num_kernels: usize,
    num_upsamples: usize,
    upp: usize,
}

impl<B: Backend> GeneratorNsf<B> {
    /// Build from the synthesizer config.
    pub fn new(cfg: &SynthesizerConfig, device: &B::Device) -> Self {
        let uic = cfg.upsample_initial_channel;
        let conv_pre = Conv1dConfig::new(cfg.inter_channels, uic, 7)
            .with_padding(PaddingConfig1d::Explicit(3, 3))
            .init(device);

        let mut ups = Vec::new();
        let mut noise_convs = Vec::new();
        let n_up = cfg.upsample_rates.len();
        for i in 0..n_up {
            let u = cfg.upsample_rates[i];
            let k = cfg.upsample_kernel_sizes[i];
            let in_ch = uic >> i;
            let out_ch = uic >> (i + 1);
            ups.push(WeightNormConvTranspose1d::new(
                in_ch,
                out_ch,
                k,
                u,
                (k - u) / 2,
                device,
            ));

            if i + 1 < n_up {
                let stride_f0: usize = cfg.upsample_rates[i + 1..].iter().product();
                noise_convs.push(
                    Conv1dConfig::new(1, out_ch, stride_f0 * 2)
                        .with_stride(stride_f0)
                        .with_padding(PaddingConfig1d::Explicit(stride_f0 / 2, stride_f0 / 2))
                        .init(device),
                );
            } else {
                noise_convs.push(Conv1dConfig::new(1, out_ch, 1).init(device));
            }
        }

        let mut resblocks = Vec::new();
        let mut last_ch = uic >> 1;
        for i in 0..n_up {
            let ch = uic >> (i + 1);
            last_ch = ch;
            for (k, d) in cfg
                .resblock_kernel_sizes
                .iter()
                .zip(cfg.resblock_dilation_sizes.iter())
            {
                resblocks.push(ResBlock1::new(ch, *k, d, device));
            }
        }

        let conv_post = Conv1dConfig::new(last_ch, 1, 7)
            .with_padding(PaddingConfig1d::Explicit(3, 3))
            .with_bias(false)
            .init(device);
        let cond = Conv1dConfig::new(cfg.gin_channels, uic, 1).init(device);

        Self {
            conv_pre,
            m_source: SourceModule::new(cfg.sample_rate, device),
            ups,
            noise_convs,
            resblocks,
            conv_post,
            cond,
            num_kernels: cfg.resblock_kernel_sizes.len(),
            num_upsamples: n_up,
            upp: cfg.hop_length(),
        }
    }

    /// `x`: `[batch, inter, time]`, `f0`: `[batch, time]`, `g`: `[batch, gin, 1]`.
    /// Returns the waveform `[batch, 1, time·upp]`.
    pub fn forward(&self, x: Tensor<B, 3>, f0: Tensor<B, 2>, g: Tensor<B, 3>) -> Tensor<B, 3> {
        let har_source = self.m_source.forward(f0, self.upp); // [b, 1, t·upp]
        let mut x = self.conv_pre.forward(x) + self.cond.forward(g);

        for i in 0..self.num_upsamples {
            x = leaky_relu(x, LRELU_SLOPE);
            x = self.ups[i].forward(x);
            x = x + self.noise_convs[i].forward(har_source.clone());

            let mut xs: Option<Tensor<B, 3>> = None;
            for j in 0..self.num_kernels {
                let out = self.resblocks[i * self.num_kernels + j].forward(x.clone());
                xs = Some(match xs {
                    Some(acc) => acc + out,
                    None => out,
                });
            }
            x = xs.expect("num_kernels > 0") / self.num_kernels as f64;
        }

        x = leaky_relu(x, LRELU_SLOPE);
        x = self.conv_post.forward(x);
        tanh(x)
    }
}
