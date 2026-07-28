//! Text to waveform.
//!
//! ```text
//!   reference audio ─┬─ cnhubert ─ quantiser ─── prompt semantic tokens ─┐
//!                    └─ spectrogram ─ ref_enc ── speaker vector ───────┐ │
//!   text ─ text-kit ─── phonemes ──────────────────────────────────┐   │ │
//!                  └─── BERT ─────── prosody ────────────────────┐ │   │ │
//!                                                                ▼ ▼   ▼ ▼
//!                                                      s1 ── semantic tokens
//!                                                                    │
//!                                                        s2 ─────────┴── audio
//! ```
//!
//! The reference clip does two jobs and they are easy to conflate: its
//! *semantic tokens* prime `s1`, so the generated speech continues its rhythm
//! and delivery, and its *spectrogram* becomes the speaker vector, which is what
//! makes the voice sound like it. Both come from the same few seconds of audio.

use std::path::Path;

use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor, TensorData};
use burn_gptsovits::{Hubert, HubertConfig, SovitsConfig, SovitsPartial, T2s, T2sConfig};
use burn_vits::{Spectral, SpectralConfig};
use text_kit::{Language, Phonemes};

use crate::error::{Result, TtsError};
use crate::prosody::{ProsodyEncoder, ProsodyFeatures};
use crate::sample::{Rng, SampleOptions, sample};

/// Audio rate the reference is analysed at, and cnhubert's rate.
pub const ANALYSIS_SR: u32 = 16_000;
/// Rate the synthesizer produces.
pub const OUTPUT_SR: u32 = 32_000;

/// How to synthesise.
#[derive(Debug, Clone)]
pub struct SynthOptions {
    /// Language of the text. Prosody features are Chinese-only; every other
    /// language is given zeros, which is what upstream does too.
    pub language: Language,
    pub sample: SampleOptions,
    /// Cap on generated tokens. At 25 Hz, 1500 is a minute — enough for any
    /// sentence, and a bound on a model that has failed to stop.
    pub max_tokens: usize,
    /// Seed, so a synthesis can be repeated exactly.
    pub seed: u64,
    /// How much of `s2`'s prior variance to sample. Upstream uses 0.5.
    pub noise_scale: f64,
}

impl Default for SynthOptions {
    fn default() -> Self {
        Self {
            language: Language::Zh,
            sample: SampleOptions::default(),
            max_tokens: 1500,
            seed: 0,
            noise_scale: 0.5,
        }
    }
}

/// Read an integer tensor as ids, whichever width the backend stores.
fn int_ids<B: Backend>(t: Tensor<B, 2, Int>) -> Result<Vec<u32>> {
    let data = t.into_data();
    if let Ok(v) = data.to_vec::<i64>() {
        return Ok(v.into_iter().map(|x| x as u32).collect());
    }
    data.to_vec::<i32>()
        .map(|v| v.into_iter().map(|x| x as u32).collect())
        .map_err(|e| TtsError::Weights(format!("token ids: {e:?}")))
}

/// A reference voice, analysed once and reusable for every line.
///
/// Carries the reference's *transcript* as well as its audio, and that is not
/// optional: `s1` continues a sequence, so it is shown the reference's phonemes
/// beside the reference's tokens and then asked to keep going with the target's
/// phonemes. Given only the target text it sees phonemes for one utterance next
/// to audio of another, finds nothing to continue, and stops after a token or
/// two.
pub struct Reference<B: Backend> {
    /// Semantic tokens that prime `s1`.
    tokens: Tensor<B, 2, Int>,
    /// Speaker vector for `s2`.
    speaker: Tensor<B, 3>,
    /// The transcript's phonemes, prepended to every line's.
    phones: Phonemes,
    /// Its prosody features, likewise.
    prosody: ProsodyFeatures,
}

/// A loaded GPT-SoVITS.
pub struct Synthesizer<B: Backend> {
    hubert: Hubert<B>,
    t2s: T2s<B>,
    sovits: SovitsPartial<B>,
    spectral: Spectral<B>,
    prosody: Option<Box<dyn ProsodyEncoder>>,
    t2s_config: T2sConfig,
    device: B::Device,
}

impl<B: Backend> Synthesizer<B> {
    /// Load from a directory holding `chinese-hubert-base/`, an `s1*.ckpt` and
    /// an `s2G*.pth`.
    ///
    /// `prosody` is optional: without it the model is given zero features, which
    /// costs expressiveness on Chinese but still speaks.
    pub fn load(
        hubert: &Path,
        s1: &Path,
        s2: &Path,
        prosody: Option<Box<dyn ProsodyEncoder>>,
        device: &B::Device,
    ) -> Result<Self> {
        let weights =
            |what: &str, e: Box<dyn std::error::Error>| TtsError::Weights(format!("{what}: {e}"));

        let mut model = Hubert::<B>::new(&HubertConfig::chinese_base(), device);
        model
            .load_pytorch(hubert)
            .map_err(|e| weights("cnhubert", e))?;

        let t2s_config = T2sConfig::default();
        let mut t2s = T2s::<B>::new(&t2s_config, device);
        t2s.load_pytorch(s1).map_err(|e| weights("s1", e))?;

        let mut sovits = SovitsPartial::<B>::new(&SovitsConfig::default(), device);
        sovits.load_pytorch(s2).map_err(|e| weights("s2", e))?;

        Ok(Self {
            hubert: model,
            t2s,
            sovits,
            spectral: Spectral::new(&SpectralConfig::gptsovits_v2_32k(), device),
            prosody,
            t2s_config,
            device: device.clone(),
        })
    }

    /// Analyse a reference clip and its transcript.
    ///
    /// `audio` is mono `f32` at [`ANALYSIS_SR`]; `text` is what is said in it.
    /// A few seconds is enough and is what the model was trained for; much more
    /// mostly costs time, since the speaker vector is an average.
    pub fn reference(
        &mut self,
        audio: &[f32],
        text: &str,
        language: Language,
    ) -> Result<Reference<B>> {
        if audio.len() < ANALYSIS_SR as usize / 2 {
            return Err(TtsError::Weights(
                "reference audio is under half a second — too little to describe a voice".into(),
            ));
        }
        let wav: Tensor<B, 2> = Tensor::from_data(
            TensorData::new(audio.to_vec(), [1, audio.len()]),
            &self.device,
        );
        let ssl = self.hubert.forward(wav).swap_dims(1, 2);
        let tokens = self.sovits.quantizer.encode(ssl);

        // The speaker vector wants the synthesizer's own rate. Duplicating
        // samples is a crude 2x resample, and adequate: `ref_enc` averages over
        // time and the imaging artefacts land above the bins it reads.
        let upsampled: Vec<f32> = audio.iter().flat_map(|&s| [s, s]).collect();
        let wav32: Tensor<B, 2> = Tensor::from_data(
            TensorData::new(upsampled, [1, audio.len() * 2]),
            &self.device,
        );
        let speaker = self.sovits.speaker(self.spectral.linear(wav32));

        let phones = text_kit::phonemize(text, language)?;
        if phones.phones.is_empty() {
            return Err(TtsError::Weights(
                "the reference transcript produced no phonemes — `s1` needs it to \
                 have something to continue from"
                    .into(),
            ));
        }
        let prosody = self.prosody_for(&phones)?;

        Ok(Reference {
            tokens,
            speaker,
            phones,
            prosody,
        })
    }

    /// Synthesise one line, returning mono `f32` at [`OUTPUT_SR`].
    pub fn say(
        &mut self,
        text: &str,
        reference: &Reference<B>,
        opts: &SynthOptions,
    ) -> Result<Vec<f32>> {
        let phonemes = text_kit::phonemize(text, opts.language)?;
        if phonemes.phones.is_empty() {
            return Ok(Vec::new());
        }
        let prosody = self.prosody_for(&phonemes)?;
        let codes = self.generate(&phonemes, &prosody, reference, opts)?;
        if codes.is_empty() {
            return Ok(Vec::new());
        }

        let codes: Tensor<B, 2, Int> = Tensor::from_data(
            TensorData::new(
                codes.iter().map(|&c| c as i32).collect::<Vec<_>>(),
                [1, codes.len()],
            ),
            &self.device,
        );
        // The phonemes go to `s2` as well: `enc_p` conditions the prior on them,
        // not only on the tokens `s1` produced from them.
        let text_ids: Tensor<B, 2, Int> = Tensor::from_data(
            TensorData::new(
                phonemes.ids().iter().map(|&i| i as i32).collect::<Vec<_>>(),
                [1, phonemes.phones.len()],
            ),
            &self.device,
        );

        let audio =
            self.sovits
                .decode(codes, text_ids, reference.speaker.clone(), opts.noise_scale);
        audio
            .into_data()
            .to_vec()
            .map_err(|e| TtsError::Weights(format!("synthesised audio was not f32: {e:?}")))
    }

    /// Prosody features, or zeros where there is no encoder for the language.
    fn prosody_for(&mut self, phonemes: &Phonemes) -> Result<ProsodyFeatures> {
        let hidden = self.prosody.as_ref().map_or(1024, |p| p.hidden());
        let phones = phonemes.phones.len();
        match (&mut self.prosody, &phonemes.word2ph) {
            (Some(encoder), Some(word2ph)) => encoder.encode(&phonemes.normalized, word2ph),
            // No encoder, or a language whose front-end reports no per-character
            // alignment. Upstream feeds zeros in both cases.
            _ => Ok(ProsodyFeatures::zeros(hidden, phones)),
        }
    }

    /// Run `s1` until it stops.
    fn generate(
        &self,
        phonemes: &Phonemes,
        prosody: &ProsodyFeatures,
        reference: &Reference<B>,
        opts: &SynthOptions,
    ) -> Result<Vec<u32>> {
        // The reference's phonemes come first and the target's after, matching
        // the audio prompt: one sequence the model can continue.
        let mut ids: Vec<i32> = reference.phones.ids().iter().map(|&i| i as i32).collect();
        ids.extend(phonemes.ids().iter().map(|&i| i as i32));

        let mut features = reference.prosody.data.clone();
        features.extend_from_slice(&prosody.data);
        let hidden = prosody.hidden;
        let n_phones = ids.len();

        let phones: Tensor<B, 2, Int> =
            Tensor::from_data(TensorData::new(ids, [1, n_phones]), &self.device);
        let bert: Tensor<B, 3> = Tensor::from_data(
            TensorData::new(features, [1, n_phones, hidden]),
            &self.device,
        );

        let text = self.t2s.embed_text(phones, bert);
        let prompt = reference.tokens.clone();
        let audio = self.t2s.embed_audio(prompt.clone(), 0);

        // First pass over the whole prompt. Text attends within itself, audio
        // attends over all the text and causally over itself — not one causal
        // mask across the join, which would stop the text seeing its own end.
        let mut state = self.t2s.state();
        let mut logits = self.t2s.forward_prompt(text, audio, &mut state);

        // `Int` is i64 on LibTorch and i32 on some others, so the width is read
        // from the tensor rather than assumed.
        let mut previous: Vec<u32> = int_ids(prompt)?;
        let generated_from = previous.len();

        let eos = self.t2s_config.eos();
        let mut rng = Rng::new(opts.seed);
        for _ in 0..opts.max_tokens {
            let scores: Vec<f32> = logits
                .clone()
                .into_data()
                .to_vec()
                .map_err(|e| TtsError::Weights(format!("logits: {e:?}")))?;
            let next = sample(&scores, &previous, &opts.sample, &mut rng);
            if next == eos {
                break;
            }
            previous.push(next);

            let token: Tensor<B, 2, Int> =
                Tensor::from_data(TensorData::new(vec![next as i32], [1, 1]), &self.device);
            // Audio positions restart at zero and are independent of the text's:
            // upstream applies the two positional encodings separately and only
            // then concatenates. So the offset counts audio tokens alone.
            let x = self.t2s.embed_audio(token, previous.len() - 1);
            logits = self.t2s.forward(x, &mut state);
        }

        if previous.len() - generated_from == opts.max_tokens {
            tracing::warn!(
                "s1 hit the {}-token cap without stopping — the line may be cut off",
                opts.max_tokens
            );
        }
        // Only what was generated; the prompt belongs to the reference.
        Ok(previous.split_off(generated_from))
    }
}
