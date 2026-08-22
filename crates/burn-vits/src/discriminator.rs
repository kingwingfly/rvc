//! HiFi-GAN discriminators for adversarial training: one scale discriminator
//! plus a period discriminator for each period the model asks for.
//!
//! The period list is the only thing that differs across the family — RVC v2
//! uses eight, GPT-SoVITS v2 five, v2Pro seven — so it is a parameter, and the
//! checkpoint remap below is generated from its length rather than written out.

use std::error::Error;
use std::path::Path;

use burn::module::Module;
use burn::tensor::Tensor;
use burn::tensor::backend::Backend;
use burn_store::{ApplyResult, ModuleSnapshot, SafetensorsStore};

use crate::nn::leaky_relu;
use crate::weightnorm::{WeightNormConv1d, WeightNormConv2d};
use burn_kit::store::load_pytorch_into;

use crate::resblock::LRELU_SLOPE;

/// Scale discriminator over the raw waveform (`DiscriminatorS`).
#[derive(Module, Debug)]
pub struct DiscriminatorS<B: Backend> {
    convs: Vec<WeightNormConv1d<B>>,
    conv_post: WeightNormConv1d<B>,
}

impl<B: Backend> DiscriminatorS<B> {
    pub fn new(device: &B::Device) -> Self {
        let convs = vec![
            WeightNormConv1d::new(1, 16, 15, 1, 7, 1, device),
            WeightNormConv1d::new_grouped(16, 64, 41, 4, 20, 1, 4, device),
            WeightNormConv1d::new_grouped(64, 256, 41, 4, 20, 1, 16, device),
            WeightNormConv1d::new_grouped(256, 1024, 41, 4, 20, 1, 64, device),
            WeightNormConv1d::new_grouped(1024, 1024, 41, 4, 20, 1, 256, device),
            WeightNormConv1d::new(1024, 1024, 5, 1, 2, 1, device),
        ];
        Self {
            convs,
            conv_post: WeightNormConv1d::new(1024, 1, 3, 1, 1, 1, device),
        }
    }

    /// `x`: `[batch, 1, time]` → (`score [batch, L]`, feature maps).
    pub fn forward(&self, mut x: Tensor<B, 3>) -> (Tensor<B, 2>, Vec<Tensor<B, 3>>) {
        // Reflect-pad up to a length the whole conv chain divides evenly. Each
        // `k=41, s=4` layer consumes its input exactly only when `len % 4 == 1`,
        // so four of them need `len % 256 == 1`. A ragged length works forward,
        // but burn 0.21's autodiff then hands LibTorch a weight gradient one
        // kernel too long and it aborts — this is what lets `--backend tch`
        // train. Costs <1% of a segment and changes no weight shapes.
        let t = x.dims()[2];
        let pad = (1 + SCALE_ALIGN - t % SCALE_ALIGN) % SCALE_ALIGN;
        if pad > 0 && pad < t {
            x = reflect_pad_last(x, pad);
        }
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
    pub fn new(period: usize, device: &B::Device) -> Self {
        // kernel (5,1), stride (3,1) except last (1,1); padding (2,0).
        let convs = vec![
            WeightNormConv2d::new(1, 32, [5, 1], [3, 1], [2, 0], device),
            WeightNormConv2d::new(32, 128, [5, 1], [3, 1], [2, 0], device),
            WeightNormConv2d::new(128, 512, [5, 1], [3, 1], [2, 0], device),
            WeightNormConv2d::new(512, 1024, [5, 1], [3, 1], [2, 0], device),
            WeightNormConv2d::new(1024, 1024, [5, 1], [1, 1], [2, 0], device),
        ];
        Self {
            convs,
            conv_post: WeightNormConv2d::new(1024, 1, [3, 1], [1, 1], [1, 0], device),
            period,
        }
    }

    /// `x`: `[batch, 1, time]` → (`score [batch, L]`, feature maps).
    pub fn forward(&self, x: Tensor<B, 3>) -> (Tensor<B, 2>, Vec<Tensor<B, 4>>) {
        let [b, c, t] = x.dims();
        // Pad time to a multiple of the period (reflect), then fold to 2-D.
        let rem = t % self.period;
        let x = if rem != 0 {
            reflect_pad_last(x, self.period - rem)
        } else {
            x
        };
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

/// Product of the strides of `DiscriminatorS`'s four `k=41, s=4` convolutions.
pub const SCALE_ALIGN: usize = 4 * 4 * 4 * 4;

/// Reflect-pad the last dim of a `[b, c, t]` tensor by `n` (numpy `reflect`:
/// mirror without repeating the edge sample).
fn reflect_pad_last<B: Backend>(x: Tensor<B, 3>, n: usize) -> Tensor<B, 3> {
    let t = x.dims()[2];
    // Append x[t-2], x[t-3], ..., x[t-1-n].
    let tail = x
        .clone()
        .slice([0..x.dims()[0], 0..x.dims()[1], (t - 1 - n)..(t - 1)])
        .flip([2]);
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
    pub fn new(periods: &[usize], device: &B::Device) -> Self {
        Self {
            scale: DiscriminatorS::new(device),
            periods: periods
                .iter()
                .map(|&p| DiscriminatorP::new(p, device))
                .collect(),
        }
    }

    /// Load a reference discriminator checkpoint, remapping the flat
    /// `discriminators.N` list onto `scale` + `periods`.
    ///
    /// Entry 0 is always the scale discriminator and the rest are the periods in
    /// order — the layout every VITS-lineage project inherited, so the mapping is
    /// derived from the period count rather than spelled out per model.
    ///
    /// `top_level_key` is where the state dict sits inside the archive, and the
    /// family does *not* agree on it: RVC's `f0D*.pth` uses `model`, GPT-SoVITS's
    /// `s2D*.pth` uses `weight`. It is a parameter for that reason — hardcoding
    /// either one loads nothing at all from the other, and a checkpoint that
    /// applies zero tensors still returns `Ok`.
    pub fn load_pytorch(
        &mut self,
        path: impl AsRef<Path>,
        top_level_key: Option<&str>,
    ) -> Result<ApplyResult, Box<dyn Error>> {
        let mut remaps = vec![(r"^discriminators\.0\.".to_string(), "scale.".to_string())];
        for i in 0..self.periods.len() {
            remaps.push((
                format!(r"^discriminators\.{}\.", i + 1),
                format!("periods.{i}."),
            ));
        }
        let remaps: Vec<(&str, &str)> = remaps
            .iter()
            .map(|(a, b)| (a.as_str(), b.as_str()))
            .collect();
        load_pytorch_into::<B, _>(self, path.as_ref(), top_level_key, &remaps)
    }

    /// Save the discriminator in Burn-native safetensors (round-trips with
    /// [`MultiPeriodDiscriminator::load_safetensors`]). Written alongside the
    /// generator so training can be resumed with the adversary intact.
    pub fn save_safetensors(&self, path: impl AsRef<Path>) -> Result<(), Box<dyn Error>> {
        let mut store = SafetensorsStore::from_file(path.as_ref()).overwrite(true);
        self.save_into(&mut store)?;
        Ok(())
    }

    /// Load a discriminator previously written by
    /// [`MultiPeriodDiscriminator::save_safetensors`].
    pub fn load_safetensors(
        &mut self,
        path: impl AsRef<Path>,
    ) -> Result<ApplyResult, Box<dyn Error>> {
        let mut store = SafetensorsStore::from_file(path.as_ref());
        Ok(self.load_from(&mut store)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    type B = burn::backend::NdArray;
    type Dev = burn::backend::ndarray::NdArrayDevice;

    fn flat<const D: usize>(t: Tensor<B, D>) -> Vec<f32> {
        t.into_data().to_vec().unwrap()
    }

    fn ramp(t: usize, device: &Dev) -> Tensor<B, 3> {
        let v: Vec<f32> = (0..t).map(|i| i as f32).collect();
        Tensor::<B, 1>::from_floats(v.as_slice(), device).reshape([1, 1, t])
    }

    /// numpy's `reflect`: mirror about the last sample **without repeating it**,
    /// which is what `F.pad(mode="reflect")` does and what the two
    /// discriminators' padding has to be to leave the waveform continuous.
    /// A literal table, because the off-by-one here (`edge` rather than
    /// `reflect`) produces a duplicated sample that nothing downstream notices.
    #[test]
    fn reflect_pad_mirrors_without_repeating_the_edge() {
        let device = Default::default();
        let x = Tensor::<B, 1>::from_floats([1.0, 2.0, 3.0, 4.0, 5.0], &device).reshape([1, 1, 5]);
        let got = flat(reflect_pad_last(x.clone(), 1));
        assert_eq!(got, vec![1.0, 2.0, 3.0, 4.0, 5.0, 4.0]);
        let got = flat(reflect_pad_last(x.clone(), 3));
        assert_eq!(got, vec![1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0, 2.0]);
        // The longest pad the callers' guard permits: `n == t - 1`.
        let got = flat(reflect_pad_last(x, 4));
        assert_eq!(got, vec![1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0, 2.0, 1.0]);
    }

    /// The `SCALE_ALIGN` pad is load-bearing and this is the invariant it
    /// exists to produce, read off the real forward pass rather than
    /// re-derived: every `k=41, s=4` layer must consume its input with **no
    /// remainder**, `(len - 1) % 4 == 0`, because burn 0.21's autodiff builds a
    /// wrongly-shaped weight gradient for a grouped strided `conv1d` that does
    /// not, and LibTorch aborts on it.
    ///
    /// `convs[0]` is `k=15, s=1, p=7`, so it preserves length: the width of the
    /// first feature map *is* the padded length, which is the only way to see
    /// the pad from outside. Four strided layers make the condition
    /// `padded % 256 == 1`.
    #[test]
    fn the_scale_pad_leaves_no_remainder_under_any_stride() {
        let device = Default::default();
        let d = DiscriminatorS::<B>::new(&device);
        for t in [300usize, 512, 769, 1000, 1920] {
            let (score, fmap) = d.forward(ramp(t, &device));
            let len: Vec<usize> = fmap.iter().map(|f| f.dims()[2]).collect();
            assert_eq!(len.len(), 7);
            assert_eq!(
                len[0] % SCALE_ALIGN,
                1,
                "t = {t}: padded to {}, which no stride chain divides",
                len[0]
            );
            assert!(len[0] >= t && len[0] - t < SCALE_ALIGN, "t = {t}: {len:?}");
            for i in 0..4 {
                assert_eq!((len[i] - 1) % 4, 0, "t = {t}, layer {}: {len:?}", i + 1);
                assert_eq!(len[i + 1], (len[i] - 1) / 4 + 1, "t = {t}: {len:?}");
            }
            // The tail is stride 1 and exactly padded, so it changes nothing.
            assert_eq!(len[5], len[4], "t = {t}: {len:?}");
            assert_eq!(len[6], len[5], "t = {t}: {len:?}");
            assert_eq!(len[6], (len[0] - 1) / SCALE_ALIGN + 1, "t = {t}: {len:?}");
            assert_eq!(score.dims(), [1, len[6]]);
            assert!(flat(score).iter().all(|v| v.is_finite()), "t = {t}");
        }
    }

    /// The other half of that pad: its guard. `reflect_pad_last` reads
    /// `x[t - 1 - n]`, so a pad of `t` or more underflows the subtraction and
    /// panics — the guard is `pad < t`, and below the boundary the alignment is
    /// simply given up rather than forced.
    ///
    /// 128 and 129 are the two sides of it at `SCALE_ALIGN = 256`: the pad each
    /// wants is 129 and 128, so the first is refused and the second applied.
    #[test]
    fn the_scale_pad_is_skipped_when_it_would_outrun_the_input() {
        let device = Default::default();
        let d = DiscriminatorS::<B>::new(&device);
        let padded = |t: usize| d.forward(ramp(t, &device)).1[0].dims()[2];
        assert_eq!(
            padded(128),
            128,
            "a pad of 129 into 128 samples must be refused"
        );
        assert_eq!(
            padded(129),
            257,
            "a pad of 128 into 129 samples must be taken"
        );
        // And the refused case still runs: it is a lost alignment, not a panic.
        assert!(
            flat(d.forward(ramp(128, &device)).0)
                .iter()
                .all(|v| v.is_finite())
        );
    }

    /// The period fold is a partition: with the waveform padded to a multiple
    /// of the period, `reshape([b, c, t / p, p])` puts sample `i` at row
    /// `i / p`, column `i % p`, so each **column is one phase** of the period
    /// and the `(5, 1)` kernels walk down a single phase for the whole clip.
    ///
    /// This pins the framework convention the fold rests on rather than our
    /// code — the reshape inside `DiscriminatorP::forward` cannot be observed
    /// from outside — and it reads the columns back with `narrow` rather than
    /// as a flat dump, so a column-major reshape would not agree with it.
    #[test]
    fn the_period_fold_puts_each_phase_in_its_own_column() {
        let device = Default::default();
        let (t, p) = (6usize, 3usize);
        let folded = ramp(t, &device).reshape([1, 1, t / p, p]);
        for phase in 0..p {
            let col = flat(folded.clone().narrow(3, phase, 1));
            let want: Vec<f32> = (0..t / p).map(|r| (r * p + phase) as f32).collect();
            assert_eq!(col, want, "column {phase} is not one phase of {p}");
        }
    }

    /// Every period both models use, over a length that is a multiple of some
    /// of them and not of others. The first feature map is `[b, 32, h, p]`:
    /// its width must be the period exactly — the kernels are `(5, 1)` with no
    /// width padding, so the fold's column count survives to the output — and
    /// its height is the row count after `k=5, s=3, p=2`, `(rows - 1) / 3 + 1`,
    /// which pins that the pad reached the *smallest* multiple of the period
    /// and not some larger one.
    ///
    /// RVC v2 uses `[2, 3, 5, 7, 11, 17, 23, 37]` and GPT-SoVITS's `s2` the
    /// first five of them, so this covers both banks.
    #[test]
    fn the_period_discriminators_pad_to_the_smallest_multiple() {
        let device = Default::default();
        let t = 128usize;
        for p in [2usize, 3, 5, 7, 11, 17, 23, 37] {
            let d = DiscriminatorP::<B>::new(p, &device);
            let (score, fmap) = d.forward(ramp(t, &device));
            let rows = t.div_ceil(p);
            assert!(rows * p >= t && rows * p - t < p);
            assert_eq!(
                fmap[0].dims(),
                [1, 32, (rows - 1) / 3 + 1, p],
                "period {p}: rows should be {rows}"
            );
            // The phase axis is never mixed: every kernel is one column wide
            // with no width padding, so the period survives to the last map.
            for (i, f) in fmap.iter().enumerate() {
                assert_eq!(f.dims()[3], p, "period {p}: map {i} lost the phase axis");
            }
            assert_eq!(fmap[4].dims()[1], 1024);
            let post = fmap[5].dims();
            assert_eq!(score.dims(), [1, post[1] * post[2] * post[3]]);
            assert!(flat(score).iter().all(|v| v.is_finite()), "period {p}");
        }
    }

    /// The checkpoint contract, spelled out: these are upstream's own
    /// `discriminators.N.*` names with `load_pytorch`'s prefix rewrite applied,
    /// and they must be the module's real parameter paths or warm-start from
    /// `f0D48k.pth` / `s2D2333k.pth` silently loads nothing.
    ///
    /// Burn derives a path from the field names of the struct that *contains* a
    /// module, so this is what a field rename here would break — and a rename
    /// breaks it without breaking anything a shape test can see, because a
    /// checkpoint that applies zero tensors still returns `Ok`.
    #[test]
    fn the_parameter_paths_are_the_ones_the_checkpoint_remap_targets() {
        let device = Default::default();
        let d = MultiPeriodDiscriminator::<B>::new(&[2, 3], &device);

        let mut want = Vec::new();
        let mut conv = |prefix: String| {
            for leaf in ["weight_g", "weight_v", "bias"] {
                want.push(format!("{prefix}.{leaf}"));
            }
        };
        // `discriminators.0` → `scale`: six convs and a post.
        for i in 0..6 {
            conv(format!("scale.convs.{i}"));
        }
        conv("scale.conv_post".to_string());
        // `discriminators.{n+1}` → `periods.{n}`: five convs and a post.
        for n in 0..2 {
            for i in 0..5 {
                conv(format!("periods.{n}.convs.{i}"));
            }
            conv(format!("periods.{n}.conv_post"));
        }

        let mut got: Vec<String> = d
            .collect(None, None, false)
            .iter()
            .map(|s| s.full_path())
            .collect();
        got.sort();
        want.sort();
        assert_eq!(got, want);
    }
}
