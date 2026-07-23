//! HiFi-GAN discriminators (`MultiPeriodDiscriminatorV2`) for adversarial
//! training: one scale discriminator + eight period discriminators.

use std::error::Error;
use std::path::Path;

use burn::module::Module;
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;
use burn_store::ApplyResult;

use crate::nn::leaky_relu;
use crate::store::load_pytorch_into;
use crate::weightnorm::{WeightNormConv1d, WeightNormConv2d};

const LRELU_SLOPE: f64 = 0.1;

/// RVC v2 period list.
const PERIODS: [usize; 8] = [2, 3, 5, 7, 11, 17, 23, 37];

/// Scale discriminator over the raw waveform (`DiscriminatorS`).
#[derive(Module, Debug)]
pub struct DiscriminatorS<B: Backend> {
    convs: Vec<WeightNormConv1d<B>>,
    conv_post: WeightNormConv1d<B>,
}

impl<B: Backend> DiscriminatorS<B> {
    fn new(device: &B::Device) -> Self {
        let convs = vec![
            WeightNormConv1d::new(1, 16, 15, 1, 7, 1, device),
            WeightNormConv1d::new_grouped(16, 64, 41, 4, 20, 1, 4, device),
            WeightNormConv1d::new_grouped(64, 256, 41, 4, 20, 1, 16, device),
            WeightNormConv1d::new_grouped(256, 1024, 41, 4, 20, 1, 64, device),
            WeightNormConv1d::new_grouped(1024, 1024, 41, 4, 20, 1, 256, device),
            WeightNormConv1d::new(1024, 1024, 5, 1, 2, 1, device),
        ];
        Self { convs, conv_post: WeightNormConv1d::new(1024, 1, 3, 1, 1, 1, device) }
    }

    /// `x`: `[batch, 1, time]` → (`score [batch, L]`, feature maps).
    pub fn forward(&self, mut x: Tensor<B, 3>) -> (Tensor<B, 2>, Vec<Tensor<B, 3>>) {
        let mut fmap = Vec::with_capacity(self.convs.len() + 1);
        for conv in &self.convs {
            x = leaky_relu(conv.forward(x), LRELU_SLOPE);
            fmap.push(x.clone());
        }
        x = self.conv_post.forward(x);
        fmap.push(x.clone());
        let [b, c, t] = x.dims();
        (x.reshape([b, c * t]), fmap)
    }
}

/// Period discriminator: reshapes the waveform to 2-D on its period
/// (`DiscriminatorP`).
#[derive(Module, Debug)]
pub struct DiscriminatorP<B: Backend> {
    convs: Vec<WeightNormConv2d<B>>,
    conv_post: WeightNormConv2d<B>,
    period: usize,
}

impl<B: Backend> DiscriminatorP<B> {
    fn new(period: usize, device: &B::Device) -> Self {
        // kernel (5,1), stride (3,1) except last (1,1); padding (2,0).
        let convs = vec![
            WeightNormConv2d::new(1, 32, [5, 1], [3, 1], [2, 0], device),
            WeightNormConv2d::new(32, 128, [5, 1], [3, 1], [2, 0], device),
            WeightNormConv2d::new(128, 512, [5, 1], [3, 1], [2, 0], device),
            WeightNormConv2d::new(512, 1024, [5, 1], [3, 1], [2, 0], device),
            WeightNormConv2d::new(1024, 1024, [5, 1], [1, 1], [2, 0], device),
        ];
        Self { convs, conv_post: WeightNormConv2d::new(1024, 1, [3, 1], [1, 1], [1, 0], device), period }
    }

    /// `x`: `[batch, 1, time]` → (`score [batch, L]`, feature maps).
    pub fn forward(&self, x: Tensor<B, 3>) -> (Tensor<B, 2>, Vec<Tensor<B, 4>>) {
        let [b, c, t] = x.dims();
        // Pad time to a multiple of the period (reflect), then fold to 2-D.
        let rem = t % self.period;
        let x = if rem != 0 { reflect_pad_last(x, self.period - rem) } else { x };
        let t2 = x.dims()[2];
        let mut x = x.reshape([b, c, t2 / self.period, self.period]);

        let mut fmap = Vec::with_capacity(self.convs.len() + 1);
        for conv in &self.convs {
            x = leaky_relu(conv.forward(x), LRELU_SLOPE);
            fmap.push(x.clone());
        }
        x = self.conv_post.forward(x);
        fmap.push(x.clone());
        let [b, c, h, w] = x.dims();
        (x.reshape([b, c * h * w]), fmap)
    }
}

/// Reflect-pad the last dim of a `[b, c, t]` tensor by `n` (numpy `reflect`:
/// mirror without repeating the edge sample).
fn reflect_pad_last<B: Backend>(x: Tensor<B, 3>, n: usize) -> Tensor<B, 3> {
    let t = x.dims()[2];
    // Append x[t-2], x[t-3], ..., x[t-1-n].
    let tail = x.clone().slice([0..x.dims()[0], 0..x.dims()[1], (t - 1 - n)..(t - 1)]).flip([2]);
    Tensor::cat(vec![x, tail], 2)
}

/// The full `MultiPeriodDiscriminatorV2`: scale + 8 period discriminators.
#[derive(Module, Debug)]
pub struct MultiPeriodDiscriminator<B: Backend> {
    /// Scale discriminator (`discriminators.0`).
    pub scale: DiscriminatorS<B>,
    /// Period discriminators (`discriminators.1..8`).
    pub periods: Vec<DiscriminatorP<B>>,
}

impl<B: Backend> MultiPeriodDiscriminator<B> {
    /// Build the standard v2 discriminator bank.
    pub fn new(device: &B::Device) -> Self {
        Self {
            scale: DiscriminatorS::new(device),
            periods: PERIODS.iter().map(|&p| DiscriminatorP::new(p, device)).collect(),
        }
    }

    /// Load the reference discriminator checkpoint (`f0D*.pth`), remapping the
    /// flat `discriminators.N` list onto `scale` + `periods`.
    pub fn load_pytorch(&mut self, path: impl AsRef<Path>) -> Result<ApplyResult, Box<dyn Error>> {
        let remaps = [
            (r"^discriminators\.0\.", "scale."),
            (r"^discriminators\.1\.", "periods.0."),
            (r"^discriminators\.2\.", "periods.1."),
            (r"^discriminators\.3\.", "periods.2."),
            (r"^discriminators\.4\.", "periods.3."),
            (r"^discriminators\.5\.", "periods.4."),
            (r"^discriminators\.6\.", "periods.5."),
            (r"^discriminators\.7\.", "periods.6."),
            (r"^discriminators\.8\.", "periods.7."),
        ];
        load_pytorch_into::<B, _>(self, path.as_ref(), "model", &remaps)
    }
}
