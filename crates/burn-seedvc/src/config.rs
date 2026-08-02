//! Every dimension the released checkpoint expects, in one place.
//!
//! These are read off upstream's `config_dit_mel_seed_uvit_whisper_small_wavenet.yml`
//! rather than inferred from tensor shapes, because a shape tells you what fits,
//! not what was meant — and the two only diverge once, silently, before anyone
//! notices.

/// The `seed-uvit-whisper-small-wavenet` preset.
///
/// Chosen as the first target of the port for one reason that outweighs the
/// others: **its content encoder is `openai/whisper-small`, which `burn-whisper`
/// already loads.** The alternative presets want either XLSR (a wav2vec2 variant
/// nothing here has) or, for v2, the ASTRAL-Quantization tokeniser, which is a
/// port of its own before any of the rest can start.
///
/// Checkpoint: `DiT_seed_v2_uvit_whisper_small_wavenet_bigvgan_pruned.pth` from
/// Hugging Face `Plachta/Seed-VC`.
#[derive(Debug, Clone)]
pub struct SeedVcConfig {
    /// Waveform rate the whole model works at. Not 16 kHz: the content encoder
    /// consumes 16 kHz, but everything from the mel onwards is 22.05 kHz.
    pub sample_rate: u32,
    /// Mel bands the transformer predicts and the vocoder consumes.
    pub n_mels: usize,
    pub n_fft: usize,
    pub hop_length: usize,

    /// Width of the frozen content encoder's output — `whisper-small`'s `d_model`.
    pub content_dim: usize,
    /// Channels the length regulator projects content into, which is also the
    /// transformer's working width.
    pub hidden_dim: usize,
    /// Entries in the length regulator's codebook.
    pub codebook_size: usize,

    /// Diffusion-transformer depth.
    pub depth: usize,
    /// Attention heads per block.
    pub heads: usize,

    /// WaveNet final block: depth, kernel and dilation.
    pub wavenet_layers: usize,
    pub wavenet_kernel: usize,
    pub wavenet_dilation: usize,
}

impl SeedVcConfig {
    /// The preset above.
    pub fn uvit_whisper_small_wavenet() -> Self {
        Self {
            sample_rate: 22_050,
            n_mels: 80,
            n_fft: 1024,
            hop_length: 256,

            content_dim: 768,
            hidden_dim: 512,
            codebook_size: 2048,

            depth: 13,
            heads: 8,

            wavenet_layers: 8,
            wavenet_kernel: 5,
            wavenet_dilation: 1,
        }
    }

    /// Frames of mel per second — `sample_rate / hop_length`, ≈86 Hz here.
    ///
    /// The rate the length regulator has to resample *to*, which is the one
    /// number that has to agree between three modules written independently.
    pub fn frame_rate(&self) -> f32 {
        self.sample_rate as f32 / self.hop_length as f32
    }
}

/// Rate the content encoder consumes, fixed by Whisper's front end.
pub const CONTENT_SR: u32 = 16_000;

/// The vocoder's own Hugging Face repo, which ships separately from Seed-VC's
/// checkpoint and is used unmodified.
pub const BIGVGAN_REPO: &str = "nvidia/bigvgan_v2_22khz_80band_256x";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_rate_matches_the_hop() {
        let cfg = SeedVcConfig::uvit_whisper_small_wavenet();
        // 22050 / 256 — the number every module has to agree on.
        assert!(
            (cfg.frame_rate() - 86.13).abs() < 0.01,
            "{}",
            cfg.frame_rate()
        );
    }
}
