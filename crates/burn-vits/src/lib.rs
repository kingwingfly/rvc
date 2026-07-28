//! The VITS building blocks that RVC and GPT-SoVITS have in common, in
//! [Burn](https://burn.dev).
//!
//! Both projects descend from the same VITS source, which is why these blocks
//! are shareable at all — and, more usefully, why their `state_dict` names line
//! up. Anything either model keeps to itself stays in that model's crate: RVC's
//! NSF source module and 768-dim content projection, GPT-SoVITS's quantiser and
//! reference encoder.
//!
//! Moving a module here changes no parameter path. Burn derives those from the
//! field names of the struct that *contains* a module, not from the crate it was
//! declared in, so a checkpoint that loaded before still loads.
//!
//! The training objectives live here too — mel-L1, KL, LSGAN and feature
//! matching are VITS's losses, not one model's, and the differentiable STFT they
//! are computed against is the same transform on both sides.
//!
//! Nothing here names a compute backend, and there are no app dependencies.

mod attention;
mod discriminator;
mod flow;
mod losses;
mod nn;
mod posterior;
mod resblock;
mod spectral;
mod wavenet;
mod weightnorm;

pub use attention::{Encoder, EncoderConfig, Ffn, MultiHeadAttention};
pub use discriminator::{DiscriminatorP, DiscriminatorS, MultiPeriodDiscriminator, SCALE_ALIGN};
pub use flow::{ResidualCouplingBlock, ResidualCouplingLayer};
pub use losses::{disc_loss, feature_matching, gen_adv, kl, mel_l1};
pub use nn::{VitsLayerNorm, leaky_relu};
pub use posterior::PosteriorEncoder;
pub use resblock::{LRELU_SLOPE, ResBlock1, get_padding};
pub use spectral::{Spectral, SpectralConfig};
pub use wavenet::Wn;
pub use weightnorm::{WeightNormConv1d, WeightNormConv2d, WeightNormConvTranspose1d};
