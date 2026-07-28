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
