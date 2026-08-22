//! The VITS training objectives, shared by every model in the family.

use burn::tensor::Tensor;
use burn::tensor::backend::Backend;

/// L1 mel-spectrogram loss between ground-truth and generated audio.
pub fn mel_l1<B: Backend>(mel_real: Tensor<B, 3>, mel_fake: Tensor<B, 3>) -> Tensor<B, 1> {
    (mel_real - mel_fake).abs().mean()
}

/// KL divergence between the flow-transformed posterior and the prior
/// (`kl_loss` from VITS/RVC), averaged over all elements.
///
/// This is VITS's Monte-Carlo estimate using the sampled `z_p` (not the closed
/// form), matching `infer/lib/train/losses.py::kl_loss`:
/// `logs_p - logs_q - 0.5 + 0.5·(z_p - m_p)²·exp(-2·logs_p)`.
pub fn kl<B: Backend>(
    z_p: Tensor<B, 3>,
    logs_q: Tensor<B, 3>,
    m_p: Tensor<B, 3>,
    logs_p: Tensor<B, 3>,
) -> Tensor<B, 1> {
    let diff = (z_p - m_p).powf_scalar(2.0);
    let term = diff * logs_p.clone().mul_scalar(-2.0).exp();
    let kl = logs_p - logs_q - 0.5 + term.mul_scalar(0.5);
    kl.mean()
}

/// LSGAN generator adversarial loss from one discriminator's fake score.
pub fn gen_adv<B: Backend>(score_fake: Tensor<B, 2>) -> Tensor<B, 1> {
    (-score_fake + 1.0).powf_scalar(2.0).mean()
}

/// LSGAN discriminator loss (real + fake) from one discriminator's scores.
pub fn disc_loss<B: Backend>(score_real: Tensor<B, 2>, score_fake: Tensor<B, 2>) -> Tensor<B, 1> {
    let real = (-score_real + 1.0).powf_scalar(2.0).mean();
    let fake = score_fake.powf_scalar(2.0).mean();
    real + fake
}

/// Feature-matching loss between real/fake feature-map lists (any rank).
///
/// **Unweighted.** Upstream's `feature_loss` folds its `* 2` into the return
/// value; here the ×2 is the caller's `FM_WEIGHT`, because the two trainers
/// already carry `C_MEL` and `C_KL` beside it and one loss hiding its own
/// coefficient while the others do not is how a third trainer ends up applying
/// it twice or not at all.
pub fn feature_matching<B: Backend, const D: usize>(
    real: &[Tensor<B, D>],
    fake: &[Tensor<B, D>],
) -> Tensor<B, 1> {
    let device = real[0].device();
    let mut sum = Tensor::<B, 1>::zeros([1], &device);
    for (r, f) in real.iter().zip(fake.iter()) {
        // Detach the real features (they are the target).
        sum = sum + (r.clone().detach() - f.clone()).abs().mean();
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    use burn::tensor::TensorData;

    type B = NdArray;

    fn t<const D: usize>(data: &[f32], shape: [usize; D]) -> Tensor<B, D> {
        Tensor::from_data(TensorData::new(data.to_vec(), shape), &Default::default())
    }

    fn scalar(x: Tensor<B, 1>) -> f64 {
        x.into_scalar() as f64
    }

    /// Scalar reference. `mean(|real - fake|)` over 0.5, 2, 0 and 6 is 2.125,
    /// which is arithmetic done here rather than by the tensor library.
    #[test]
    fn mel_l1_is_the_mean_absolute_difference() {
        let real = t(&[1.0, 2.0, 3.0, 4.0], [1, 2, 2]);
        let fake = t(&[1.5, 0.0, 3.0, 10.0], [1, 2, 2]);
        assert!((scalar(mel_l1(real, fake)) - 2.125).abs() < 1e-6);
    }

    /// Scalar reference, transcribed from `train/losses.py::kl_loss` of
    /// RVC-Project `2.3.260718` rather than from this module:
    /// `logs_p - logs_q - 0.5 + 0.5·(z_p - m_p)²·exp(-2·logs_p)`, meaned.
    ///
    /// Both `logs_p` values are non-zero in one of the two elements, so a
    /// dropped `exp(-2·logs_p)` or a sign flip on it changes the answer.
    #[test]
    fn kl_matches_the_monte_carlo_estimate() {
        let reference = |z_p: f64, logs_q: f64, m_p: f64, logs_p: f64| {
            logs_p - logs_q - 0.5 + 0.5 * (z_p - m_p).powi(2) * (-2.0 * logs_p).exp()
        };
        let expected = (reference(1.0, 0.25, 0.0, 0.5) + reference(-1.0, -0.5, 2.0, 0.0)) / 2.0;

        let got = kl(
            t(&[1.0, -1.0], [1, 1, 2]),
            t(&[0.25, -0.5], [1, 1, 2]),
            t(&[0.0, 2.0], [1, 1, 2]),
            t(&[0.5, 0.0], [1, 1, 2]),
        );
        let got = scalar(got);
        assert!((got - expected).abs() < 1e-6, "{got} vs {expected}");
    }

    /// Exactly-zero invariant. With `logs_p == logs_q == 0` a sample exactly one
    /// unit from the prior mean makes the estimate `0 - 0 - 0.5 + 0.5·1·1`,
    /// which is zero with no rounding anywhere — so this asserts equality
    /// rather than a tolerance, and any drift in the constant or the exponent
    /// shows up as a non-zero.
    #[test]
    fn one_sigma_from_the_prior_mean_costs_exactly_zero() {
        let zeros = [0.0; 4];
        let got = kl(
            t(&[1.0, -1.0, 1.0, -1.0], [1, 1, 4]),
            t(&zeros, [1, 1, 4]),
            t(&zeros, [1, 1, 4]),
            t(&zeros, [1, 1, 4]),
        );
        assert_eq!(scalar(got), 0.0);
    }

    /// Scalar reference plus the exactly-zero case: LSGAN's generator term is
    /// `mean((1 - score)²)`, which is 1, 0, 1 and 4 over these scores — 1.5 —
    /// and exactly zero for a discriminator scoring every fake as real.
    #[test]
    fn gen_adv_is_the_squared_distance_from_one() {
        let scores = t(&[0.0, 1.0, 2.0, -1.0], [2, 2]);
        assert!((scalar(gen_adv(scores)) - 1.5).abs() < 1e-6);
        assert_eq!(scalar(gen_adv(t(&[1.0; 4], [2, 2]))), 0.0);
    }

    /// Scalar reference plus the exactly-zero case. `mean((1 - real)²)` over
    /// 1 and 0 is 0.5, `mean(fake²)` over 0 and 0.5 is 0.125, and the loss is
    /// their sum. A discriminator that is exactly right scores zero.
    #[test]
    fn disc_loss_adds_the_real_and_fake_halves() {
        let real = t(&[1.0, 0.0], [1, 2]);
        let fake = t(&[0.0, 0.5], [1, 2]);
        assert!((scalar(disc_loss(real, fake)) - 0.625).abs() < 1e-6);
        assert_eq!(
            scalar(disc_loss(t(&[1.0; 4], [1, 4]), t(&[0.0; 4], [1, 4]))),
            0.0
        );
    }

    /// Scalar reference: one `mean(|real - fake|)` per feature map, summed over
    /// the list — 1.5 from the first pair and 3.5 from the second. The ×2
    /// upstream folds into `feature_loss` is deliberately *not* here, so 5.0
    /// rather than 10.0 is the number that says the coefficient still lives at
    /// the call site.
    #[test]
    fn feature_matching_sums_one_mean_per_layer() {
        let real = [t(&[1.0, 2.0], [1, 1, 2]), t(&[4.0, 4.0], [1, 1, 2])];
        let fake = [t(&[0.0, 0.0], [1, 1, 2]), t(&[1.0, 0.0], [1, 1, 2])];
        assert!((scalar(feature_matching(&real, &fake)) - 5.0).abs() < 1e-6);
    }

    /// Exactly-zero invariant: a generator whose features match the real ones
    /// pays nothing, with no rounding to absorb.
    #[test]
    fn identical_features_cost_exactly_zero() {
        let real = [t(&[1.0, -2.0, 3.0, 4.0], [1, 1, 4])];
        assert_eq!(scalar(feature_matching(&real, &real.clone())), 0.0);
    }
}
