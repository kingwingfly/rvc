//! Model dimensions, named as Hugging Face's `config.json` names them.

/// The shape of one Whisper checkpoint.
///
/// Field names mirror `config.json` so a new size is a transcription, not a
/// derivation. Everything else in the crate reads its shapes from here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WhisperConfig {
    /// Mel bins the encoder consumes. 80 through large-v2, 128 from large-v3.
    pub num_mel_bins: usize,
    /// Encoder frames after the stride-2 convolution: 30 s at 100 fps, halved.
    pub max_source_positions: usize,
    /// Longest token sequence the decoder's learned positions cover.
    pub max_target_positions: usize,
    /// Residual stream width, shared by encoder and decoder.
    pub d_model: usize,
    pub encoder_attention_heads: usize,
    pub encoder_layers: usize,
    pub encoder_ffn_dim: usize,
    pub decoder_attention_heads: usize,
    pub decoder_layers: usize,
    pub decoder_ffn_dim: usize,
    pub vocab_size: usize,
}

impl WhisperConfig {
    /// `openai/whisper-large-v3-turbo` — 809M parameters.
    ///
    /// large-v3 with the decoder cut from 32 layers to 4, which is where nearly
    /// all of the speed comes from: the encoder runs once per 30 s window, the
    /// decoder runs once per generated token.
    pub fn large_v3_turbo() -> Self {
        Self {
            num_mel_bins: 128,
            max_source_positions: 1500,
            max_target_positions: 448,
            d_model: 1280,
            encoder_attention_heads: 20,
            encoder_layers: 32,
            encoder_ffn_dim: 5120,
            decoder_attention_heads: 20,
            decoder_layers: 4,
            decoder_ffn_dim: 5120,
            vocab_size: 51866,
        }
    }

    /// `openai/whisper-large-v3` — the same encoder, with all 32 decoder layers.
    pub fn large_v3() -> Self {
        Self {
            decoder_layers: 32,
            ..Self::large_v3_turbo()
        }
    }

    /// `openai/whisper-small` — 244M parameters, 80 mel bins.
    ///
    /// Here because it is the content encoder of Seed-VC's
    /// `seed-uvit-whisper-small-wavenet` preset, which uses the encoder alone as
    /// a frozen feature extractor and deletes the decoder. Its 768-wide output is
    /// what that model's length regulator projects from — so the width is a
    /// compatibility constraint there, not a size trade-off.
    ///
    /// Transcribed from the repo's own `config.json`, like every preset above:
    /// nothing in this crate reads that file, so a new size is a new constructor.
    pub fn small() -> Self {
        Self {
            num_mel_bins: 80,
            max_source_positions: 1500,
            max_target_positions: 448,
            d_model: 768,
            encoder_attention_heads: 12,
            encoder_layers: 12,
            encoder_ffn_dim: 3072,
            decoder_attention_heads: 12,
            decoder_layers: 12,
            decoder_ffn_dim: 3072,
            vocab_size: 51865,
        }
    }

    /// A tiny model with the same *structure*, for tests that check shapes and
    /// cache behaviour rather than weights.
    #[doc(hidden)]
    pub fn tiny() -> Self {
        Self {
            num_mel_bins: 8,
            max_source_positions: 12,
            max_target_positions: 16,
            d_model: 16,
            encoder_attention_heads: 2,
            encoder_layers: 2,
            encoder_ffn_dim: 32,
            decoder_attention_heads: 2,
            decoder_layers: 2,
            decoder_ffn_dim: 32,
            vocab_size: 40,
        }
    }
}
