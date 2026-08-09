//! MDX23C — the TFC-TDF-UNet v3 source-separation network — in
//! [Burn](https://burn.dev).
//!
//! Vocal/instrumental separation, so a corpus recorded over a music bed can be
//! cleaned before anything else touches it. `[channels][samples]` of 44.1 kHz
//! audio in, one waveform per stem out.
//!
//! # The checkpoint this is built against
//!
//! The architecture **is** the checkpoint's config — `dim_f`, `n_fft`, the
//! channel widths and the block counts all differ across the MDX family — so
//! the exact file is part of the port and not a deployment detail:
//!
//! | | |
//! |---|---|
//! | repo | Hugging Face **`Politrees/UVR_resources`** |
//! | revision | **`929e057b81aa49bc2e6490bef8671f47b2c120f6`** |
//! | weights | `models/MDX23C/MDX23C-8KFFT-InstVoc_HQ.ckpt` (448,101,203 bytes) |
//! | config | `models/MDX23C/model_2_stem_full_band_8k.yaml` (709 bytes) |
//! | stems | `["Vocals", "Instrumental"]`, in that output order |
//!
//! The `.ckpt` is a **bare `state_dict` at the root** — no `state_dict` or
//! `model` wrapper, unlike the RVC-lineage `.pth` files — of 319 tensors, and
//! every one of them is claimed by [`MdxConfig::mdx23c_8k_instvoc_hq`]. That
//! repo was chosen over the other mirrors because it also holds the MDX-Net v2
//! ONNX graphs under `models/MDXNet/`, so one revision pins both runtimes'
//! assets.
//!
//! ## Why this variant and not `UVR-MDX-NET-Voc_FT.onnx`
//!
//! The older MDX-Net v2 models are what UVR5 shipped first and are the obvious
//! target, and they **cannot be Burn-ported with verifiable coverage**. They
//! are published as ONNX only, and the graphs are BN-fused exports: roughly
//! half of every file's 220 initializers carry no name at all (`685`, `688`,
//! …), because `Conv → BatchNorm` was constant-folded and `Linear(bias=False)`
//! became a `MatMul` over a transposed anonymous constant. Worse, one
//! architecture has two naming schemes on disk —
//! `UVR-MDX-NET-Inst_HQ_3.onnx` names its survivors
//! `encoding_blocks.N.tdf.1.*` while `UVR-MDX-NET-Voc_FT.onnx` names the same
//! tensors `BatchNormalization_460.bnu.*` — so no remap is right for both. The
//! one `.pt` on the Hub (`mediainbox/uvr-mdx-models`) is an `onnx2torch`
//! TorchScript trace whose own metadata contradicts UVR's table. A coverage
//! triple measured against any of that would be fiction.
//!
//! **That is a reason to run them on ONNX Runtime, not a reason to drop them.**
//! [`Stft`] is parameterised by `(n_fft, hop, dim_f)` precisely so it can be
//! the front end for either: an MDX-Net v2 ONNX graph is the *whole* network,
//! and the only thing it needs from a host is this transform. Two facts a
//! caller wiring that path will need and will not find written down anywhere
//! else in this workspace:
//!
//! - UVR keys its hyper-parameter table by the **MD5 of the last 10,240,000
//!   bytes** of the `.onnx`, not of the whole file. Hashing the file gives a
//!   key that is in no table.
//! - In that table (`mdx_model_data.json`), `mdx_dim_t_set` is an **exponent**:
//!   `8` means 256 frames. `mdx_n_fft_scale_set` is `n_fft`, `mdx_dim_f_set` is
//!   `dim_f`, and `hop` is always 1024. Verified: `UVR-MDX-NET-Voc_FT.onnx`
//!   hashes to `77d07b2667ddf05b9e3175941b4454a0` → 7680/3072/8, and
//!   `UVR-MDX-NET-Inst_HQ_3.onnx` to `55657dd70583b0fedfba5f67df11d711` →
//!   6144/3072/8.
//!
//! # What is checked
//!
//! `examples/keys` lists a checkpoint's tensors, `examples/load` reports
//! coverage — **319 applied / 0 missing / 0 unused**, identical on every
//! backend. That says the module tree matches the file and nothing about the
//! arithmetic, which for this network is most of the risk: the U-net runs with
//! *time* as the image height and *frequency* as its width, and a port that
//! keeps the natural orientation loads perfectly and separates nothing. So
//! `examples/separate` mixes a known voice with a known bed and reports the
//! recovery gap, and [`stft`]'s round-trip tests pin the transform with no
//! weights at all.
//!
//! # Memory, and the unit of work
//!
//! The network's unit is **one chunk of `chunk_size` samples** — 261,120, or
//! 5.92 s at 44.1 kHz — and nothing here is aware of a longer recording. That
//! is deliberate: separation runs on songs, and a caller that decodes a
//! ten-minute stereo file into memory before starting pays several hundred
//! megabytes before the first forward pass. A streaming caller reads one chunk,
//! calls [`Stft::forward`] → [`TfcTdfNet::forward`] → [`Stft::inverse`], and
//! keeps nothing but its own crossfade tail.
//!
//! Even one chunk is not small: the first encoder level holds
//! `[1, 128, 256, 1024]` activations, 134 MB each, and the widest decoder
//! concatenation is 268 MB. A 6 GB GPU fits it; pure-CPU `ndarray` will run it
//! and should not be asked to.
//!
//! # Provenance
//!
//! Ported from ZFTurbo's `Music-Source-Separation-Training`,
//! `models/mdx23c_tfc_tdf_v3.py`, read as a reference and never run.

pub mod net;
pub mod stft;

use std::error::Error;
use std::path::Path;

use burn::module::Module;
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::tensor::Tensor;
use burn::tensor::backend::Backend;

use net::{DecoderBlock, EncoderBlock, FinalConv, TfcTdf};
pub use stft::Stft;

/// Which stem each output index of [`MdxConfig::mdx23c_8k_instvoc_hq`] is.
///
/// The order is the config's `training.instruments` list, and it is not
/// recoverable from the weights — a caller that guesses gets the accompaniment
/// where it wanted the voice, with nothing to indicate it.
pub const STEMS_8K_INSTVOC: [&str; 2] = ["vocals", "instrumental"];

/// The shape of the network, which is `model_2_stem_full_band_8k.yaml` field
/// for field.
///
/// Every value here is read off the checkpoint rather than assumed — see
/// [`MdxConfig::mdx23c_8k_instvoc_hq`] for the tensor each one is pinned by.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MdxConfig {
    /// `audio.n_fft`.
    pub n_fft: usize,
    /// `audio.hop_length`.
    pub hop: usize,
    /// `audio.dim_f` — bins kept out of `n_fft / 2 + 1`.
    pub dim_f: usize,
    /// `audio.dim_t` — frames per chunk, and hence the chunk length.
    pub dim_t: usize,
    /// `audio.num_channels`. 2; the front end is stereo and folding a mono
    /// source to one channel would halve the network's input width.
    pub audio_channels: usize,
    /// `audio.sample_rate`, 44100. Not used by any tensor op — recorded so a
    /// caller resamples rather than guessing.
    pub sample_rate: u32,
    /// `model.num_subbands` — how many pieces the frequency axis is folded into
    /// extra channels.
    pub num_subbands: usize,
    /// `model.num_scales` — encoder levels, and equally decoder levels.
    pub num_scales: usize,
    /// `model.scale`, `[time, frequency]` per resampling stage.
    pub scale: [usize; 2],
    /// `model.num_blocks_per_scale`.
    pub num_blocks_per_scale: usize,
    /// `model.num_channels` — the width the first convolution emits.
    pub num_channels: usize,
    /// `model.growth` — channels added per level.
    pub growth: usize,
    /// `model.bottleneck_factor` — how far the TDF narrows the frequency axis.
    pub bottleneck_factor: usize,
    /// How many stems the head emits, from `training.instruments`.
    pub stems: usize,
}

impl MdxConfig {
    /// `MDX23C-8KFFT-InstVoc_HQ`.
    ///
    /// Pinned as a constant rather than parsed from the YAML, for the reason
    /// `burn_rvc`'s `HubertConfig::chinese_base` is: a disagreeing checkpoint
    /// should fail as a shape mismatch on load, not be silently accommodated.
    /// Each value is confirmed by a tensor in the file:
    ///
    /// - `first_conv.weight` is `[128, 16, 1, 1]` → `num_channels = 128` and
    ///   `dim_c = num_subbands × audio_channels × 2 = 16`.
    /// - `encoder_blocks.0.…tdf.2.weight` is `[256, 1024]` → the frequency
    ///   axis enters at `dim_f / num_subbands = 1024` and narrows by 4.
    /// - `bottleneck_block.blocks.0.tdf.2.weight` is `[8, 32]` → five halvings
    ///   of 1024, so `num_scales = 5`.
    /// - `encoder_blocks.0.downscale.conv.2.weight` is `[256, 128, 2, 2]` →
    ///   `growth = 128`, `scale = [2, 2]`.
    /// - `blocks.0` and `blocks.1` exist per stack → `num_blocks_per_scale = 2`.
    /// - `final_conv.2.weight` is `[32, 128, 1, 1]` → `stems × dim_c = 32`,
    ///   so two stems.
    pub fn mdx23c_8k_instvoc_hq() -> Self {
        Self {
            n_fft: 8192,
            hop: 1024,
            dim_f: 4096,
            dim_t: 256,
            audio_channels: 2,
            sample_rate: 44100,
            num_subbands: 4,
            num_scales: 5,
            scale: [2, 2],
            num_blocks_per_scale: 2,
            num_channels: 128,
            growth: 128,
            bottleneck_factor: 4,
            stems: 2,
        }
    }

    /// Channels the network sees after the subband fold: re and im for every
    /// audio channel, times the subband count.
    pub fn dim_c(&self) -> usize {
        self.num_subbands * self.audio_channels * 2
    }

    /// Samples per forward pass, `hop × (dim_t - 1)` — 261,120, which is the
    /// config's `audio.chunk_size` and not an independent number.
    pub fn chunk_size(&self) -> usize {
        self.hop * (self.dim_t - 1)
    }

    /// The front end this configuration implies.
    pub fn stft(&self) -> Stft {
        Stft::new(self.n_fft, self.hop, self.dim_f)
    }
}

/// MDX23C, upstream's `TFC_TDF_net`.
#[derive(Module, Debug)]
pub struct TfcTdfNet<B: Backend> {
    first_conv: Conv2d<B>,
    encoder_blocks: Vec<EncoderBlock<B>>,
    bottleneck_block: TfcTdf<B>,
    decoder_blocks: Vec<DecoderBlock<B>>,
    final_conv: FinalConv<B>,
    num_subbands: usize,
    stems: usize,
    scale: [usize; 2],
}

impl<B: Backend> TfcTdfNet<B> {
    pub fn new(cfg: &MdxConfig, device: &B::Device) -> Self {
        let dim_c = cfg.dim_c();
        let mut channels = cfg.num_channels;
        // The frequency axis the *blocks* see: the fold has already moved a
        // factor of `num_subbands` into the channel axis.
        let mut bins = cfg.dim_f / cfg.num_subbands;

        let encoder_blocks = (0..cfg.num_scales)
            .map(|_| {
                let block = EncoderBlock::new(
                    channels,
                    cfg.growth,
                    cfg.num_blocks_per_scale,
                    bins,
                    cfg.bottleneck_factor,
                    cfg.scale,
                    device,
                );
                bins /= cfg.scale[1];
                channels += cfg.growth;
                block
            })
            .collect();

        let bottleneck_block = TfcTdf::new(
            channels,
            channels,
            cfg.num_blocks_per_scale,
            bins,
            cfg.bottleneck_factor,
            device,
        );

        let decoder_blocks = (0..cfg.num_scales)
            .map(|_| {
                bins *= cfg.scale[1];
                let block = DecoderBlock::new(
                    channels,
                    cfg.growth,
                    cfg.num_blocks_per_scale,
                    bins,
                    cfg.bottleneck_factor,
                    cfg.scale,
                    device,
                );
                channels -= cfg.growth;
                block
            })
            .collect();

        Self {
            first_conv: Conv2dConfig::new([dim_c, cfg.num_channels], [1, 1])
                .with_bias(false)
                .init(device),
            encoder_blocks,
            bottleneck_block,
            decoder_blocks,
            // The head eats the network's output *and* the mixture, hence
            // `channels + dim_c`.
            final_conv: FinalConv::new(
                channels + dim_c,
                channels,
                cfg.stems * dim_c,
                device,
            ),
            num_subbands: cfg.num_subbands,
            stems: cfg.stems,
            scale: cfg.scale,
        }
    }

    /// `[batch, 2 × audio_channels, dim_f, frames]` complex spectrum in →
    /// `[batch, stems, 2 × audio_channels, dim_f, frames]` out.
    ///
    /// The input is what [`Stft::forward`] produces; flatten the first two axes
    /// of the output and [`Stft::inverse`] turns it back into waveforms.
    ///
    /// `frames` must be divisible by `scale[0]^num_scales` (32 for this
    /// configuration) and `dim_f / num_subbands` likewise by `scale[1]^…`.
    /// Neither is padded here, unlike RMVPE: a chunk length is the caller's
    /// unit of work rather than a property of one buffer, and silently
    /// stretching it would move every seam in a chunked separation.
    pub fn forward(&self, spec: Tensor<B, 4>) -> Tensor<B, 5> {
        let [batch, channels, dim_f, frames] = spec.dims();
        let stride = self.scale[0].pow(self.encoder_blocks.len() as u32);
        assert!(
            frames.is_multiple_of(stride),
            "{frames} frames is not a multiple of {stride}; pad the chunk, not the tensor"
        );

        let mix = self.fold_subbands(spec);
        let first_conv_out = self.first_conv.forward(mix.clone());

        // Time becomes the height and frequency the width, which is what puts
        // the frequency axis last for every `Tdf`'s `Linear`.
        let mut x = first_conv_out.clone().swap_dims(2, 3);
        let mut skips = Vec::with_capacity(self.encoder_blocks.len());
        for block in &self.encoder_blocks {
            let (skip, down) = block.forward(x);
            skips.push(skip);
            x = down;
        }
        x = self.bottleneck_block.forward(x);
        for block in &self.decoder_blocks {
            let skip = skips.pop().expect("one skip per decoder level");
            x = block.forward(x, skip);
        }
        let x = x.swap_dims(2, 3);

        // Upstream's comment is "reduce artifacts": the head sees the network's
        // output gated by the first convolution's, rather than the output
        // alone.
        let x = x * first_conv_out;
        let x = self.final_conv.forward(Tensor::cat(vec![mix, x], 1));

        let x = self.unfold_subbands(x);
        x.reshape([batch, self.stems, channels, dim_f, frames])
    }

    /// `[b, c, f, t]` → `[b, c × k, f / k, t]`: upstream's `cac2cws`.
    ///
    /// The frequency axis is split into `k` contiguous bands and each becomes
    /// its own channel, so the convolutions see a shorter, wider image. Note
    /// the split is *outermost* — band 0 is the lowest `f / k` bins — which is
    /// what the two-step reshape encodes and what [`Self::unfold_subbands`]
    /// has to undo in the same order.
    fn fold_subbands(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let [b, c, f, t] = x.dims();
        let k = self.num_subbands;
        x.reshape([b, c * k, f / k, t])
    }

    /// `[b, c, f, t]` → `[b, c / k, f × k, t]`: upstream's `cws2cac`.
    fn unfold_subbands(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let [b, c, f, t] = x.dims();
        let k = self.num_subbands;
        x.reshape([b, c / k, f * k, t])
    }

    /// Load `MDX23C-8KFFT-InstVoc_HQ.ckpt`.
    ///
    /// A bare state dict at the root, so no `top_level_key`. Every remap here
    /// undoes an `nn.Sequential`'s positional child names, and none of them
    /// changes a layout:
    ///
    /// - `tfc1`/`tfc2` are `Sequential(norm, act, conv)` → `.0` and `.2`.
    /// - `tdf` is `Sequential(norm, act, linear, norm, act, linear)` → `.0`,
    ///   `.2`, `.3`, `.5`.
    /// - `downscale`/`upscale` wrap their `Sequential` in a field called
    ///   `conv`, so the file says `downscale.conv.0` and `downscale.conv.2`;
    ///   these are the two remaps that remove a level rather than renaming one.
    /// - the head is `final_conv.0` and `final_conv.2`.
    ///
    /// Expected: **319 applied, 0 missing, 0 unused.** Nothing in this file is
    /// a training-only tensor — the norms are instance norms, so there are no
    /// running statistics and no `num_batches_tracked` to explain away.
    pub fn load_pytorch(
        &mut self,
        path: impl AsRef<Path>,
    ) -> Result<burn_store::ApplyResult, Box<dyn Error>> {
        let remaps = [
            (r"\.tfc1\.0\.", ".tfc1.norm."),
            (r"\.tfc1\.2\.", ".tfc1.conv."),
            (r"\.tfc2\.0\.", ".tfc2.norm."),
            (r"\.tfc2\.2\.", ".tfc2.conv."),
            (r"\.tdf\.0\.", ".tdf.norm1."),
            (r"\.tdf\.2\.", ".tdf.down."),
            (r"\.tdf\.3\.", ".tdf.norm2."),
            (r"\.tdf\.5\.", ".tdf.up."),
            (r"\.downscale\.conv\.0\.", ".downscale.norm."),
            (r"\.downscale\.conv\.2\.", ".downscale.conv."),
            (r"\.upscale\.conv\.0\.", ".upscale.norm."),
            (r"\.upscale\.conv\.2\.", ".upscale.conv."),
            (r"^final_conv\.0\.", "final_conv.conv1."),
            (r"^final_conv\.2\.", "final_conv.conv2."),
        ];
        burn_kit::store::load_pytorch_into::<B, _>(self, path.as_ref(), None, &remaps)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::Distribution;

    type B = burn_ndarray::NdArray;

    /// The released shape is 112M parameters over 134 MB activations, which no
    /// unit test can run. This keeps every structural property that matters —
    /// the subband fold, five halvings of both axes, a bottleneck, and a
    /// decoder whose skips have to line up exactly — at a size a CPU can do in
    /// a second.
    fn tiny() -> MdxConfig {
        MdxConfig {
            n_fft: 128,
            hop: 16,
            dim_f: 64,
            dim_t: 33,
            audio_channels: 2,
            sample_rate: 44100,
            num_subbands: 2,
            num_scales: 2,
            scale: [2, 2],
            num_blocks_per_scale: 1,
            num_channels: 4,
            growth: 4,
            bottleneck_factor: 2,
            stems: 2,
        }
    }

    /// The shape arithmetic is what fails loudly, so pin it end to end — a
    /// mis-sized skip concatenation or a subband fold in the wrong direction
    /// cannot survive this.
    #[test]
    fn a_chunk_survives_the_whole_network() {
        let cfg = tiny();
        let device = Default::default();
        let model = TfcTdfNet::<B>::new(&cfg, &device);

        let frames = cfg.dim_t - 1; // a multiple of scale[0]^num_scales
        let spec = Tensor::<B, 4>::random(
            [1, 2 * cfg.audio_channels, cfg.dim_f, frames],
            Distribution::Normal(0.0, 1.0),
            &device,
        );
        let out = model.forward(spec);
        assert_eq!(
            out.dims(),
            [1, cfg.stems, 2 * cfg.audio_channels, cfg.dim_f, frames]
        );
        let v: Vec<f32> = out.into_data().to_vec().unwrap();
        // Asserted separately because Burn's `assert_approx_eq` compares NaN to
        // NaN without complaint, so a NaN would otherwise pass silently.
        assert!(v.iter().all(|x| x.is_finite()), "output must be finite");
    }

    /// The fold and the unfold have to be exact inverses, and both are a bare
    /// `reshape` — so a wrong one is not an error, it is a permutation of the
    /// frequency axis that no shape check can see.
    #[test]
    fn the_subband_fold_is_undone_exactly() {
        let cfg = tiny();
        let device = Default::default();
        let model = TfcTdfNet::<B>::new(&cfg, &device);
        let x = Tensor::<B, 4>::random([1, 4, 64, 8], Distribution::Normal(0.0, 1.0), &device);

        let round_trip = model.unfold_subbands(model.fold_subbands(x.clone()));
        assert_eq!(round_trip.dims(), x.dims());
        let (a, b): (Vec<f32>, Vec<f32>) = (
            x.into_data().to_vec().unwrap(),
            round_trip.into_data().to_vec().unwrap(),
        );
        assert_eq!(a, b);
    }

    /// The chunk length is derived, not declared; a caller that computes it
    /// differently feeds the network a frame count the U-net cannot halve.
    #[test]
    fn the_released_chunk_is_the_configs_chunk_size() {
        let cfg = MdxConfig::mdx23c_8k_instvoc_hq();
        assert_eq!(cfg.chunk_size(), 261_120);
        assert_eq!(cfg.stft().frames(cfg.chunk_size()), cfg.dim_t);
        assert_eq!(cfg.dim_c(), 16);
    }
}
