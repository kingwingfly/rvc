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
//!
//! Nothing here names a runtime. The networks sit behind
//! [`Engine`](crate::Engine), so sampling, the repetition penalty and the stop
//! rule are shared by the Burn and the ONNX Runtime paths rather than written
//! twice — and a [`Reference`] is plain data, so it outlives whichever engine
//! produced it.

use text_kit::{Language, Phonemes};

use crate::engine::{EOS, Engine, PRIOR_CHANNELS};
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
    /// Language of the text, for the runs whose script does not say. Text is
    /// split by script first and each run goes to its own front-end, so a
    /// Chinese line with an English word in it phonemizes both halves.
    ///
    /// Prosody features are Chinese-only; every other language, and any line
    /// that mixes two, is given zeros — which is what upstream does too.
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

/// A reference voice, analysed once and reusable for every line.
///
/// Carries the reference's *transcript* as well as its audio, and that is not
/// optional: `s1` continues a sequence, so it is shown the reference's phonemes
/// beside the reference's tokens and then asked to keep going with the target's
/// phonemes. Given only the target text it sees phonemes for one utterance next
/// to audio of another, finds nothing to continue, and stops after a token or
/// two.
pub struct Reference {
    /// Semantic tokens that prime `s1`.
    tokens: Vec<u32>,
    /// Speaker vector for `s2`.
    speaker: Vec<f32>,
    /// The transcript's phonemes, prepended to every line's.
    phones: Phonemes,
    /// Its prosody features, likewise.
    prosody: ProsodyFeatures,
}

/// A loaded GPT-SoVITS.
pub struct Synthesizer {
    engine: Box<dyn Engine>,
    prosody: Option<Box<dyn ProsodyEncoder>>,
}

impl Synthesizer {
    /// Wrap a loaded engine.
    ///
    /// Which weights it holds and which runtime executes them is the engine's
    /// business — see [`BurnEngine::load`](crate::BurnEngine::load) and
    /// [`OnnxEngine::load`](crate::onnx_engine).
    ///
    /// `prosody` is optional: without it the model is given zero features, which
    /// costs expressiveness on Chinese but still speaks.
    pub fn new(engine: Box<dyn Engine>, prosody: Option<Box<dyn ProsodyEncoder>>) -> Self {
        Self { engine, prosody }
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
    ) -> Result<Reference> {
        if audio.len() < ANALYSIS_SR as usize / 2 {
            return Err(TtsError::Weights(
                "reference audio is under half a second — too little to describe a voice".into(),
            ));
        }
        let (tokens, speaker) = self.engine.analyse(audio)?;

        let phones = text_kit::phonemize_mixed(text, language)?;
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
        reference: &Reference,
        opts: &SynthOptions,
    ) -> Result<Vec<f32>> {
        let phonemes = text_kit::phonemize_mixed(text, opts.language)?;
        if phonemes.phones.is_empty() {
            return Ok(Vec::new());
        }
        let prosody = self.prosody_for(&phonemes)?;
        let codes = self.generate(&phonemes, &prosody, reference, opts)?;
        if codes.is_empty() {
            return Ok(Vec::new());
        }

        // The prior's sample is drawn here rather than inside the engine, which
        // makes `s2` a pure function of its inputs and is the only reason a Burn
        // run and an ONNX run of the same tokens can be diffed. Its generator is
        // seeded afresh, so the noise does not depend on how many tokens `s1`
        // happened to draw before it.
        let mut rng = Rng::new(opts.seed);
        let noise: Vec<f32> = (0..PRIOR_CHANNELS * codes.len() * 2)
            .map(|_| rng.next_normal() * opts.noise_scale as f32)
            .collect();

        // The phonemes go to `s2` as well: `enc_p` conditions the prior on them,
        // not only on the tokens `s1` produced from them.
        let text_ids: Vec<u32> = phonemes.ids().iter().map(|&i| i as u32).collect();
        self.engine
            .s2(&codes, &text_ids, &reference.speaker, &noise)
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
        &mut self,
        phonemes: &Phonemes,
        prosody: &ProsodyFeatures,
        reference: &Reference,
        opts: &SynthOptions,
    ) -> Result<Vec<u32>> {
        // The reference's phonemes come first and the target's after, matching
        // the audio prompt: one sequence the model can continue.
        let ids: Vec<u32> = reference
            .phones
            .ids()
            .iter()
            .chain(phonemes.ids().iter())
            .map(|&i| i as u32)
            .collect();

        let mut features = reference.prosody.data.clone();
        features.extend_from_slice(&prosody.data);

        // The whole prompt in one pass. Text attends within itself, audio
        // attends over all the text and causally over itself — not one causal
        // mask across the join, which would stop the text seeing its own end.
        let mut logits =
            self.engine
                .s1_prompt(&ids, &features, prosody.hidden, &reference.tokens)?;

        let mut previous = reference.tokens.clone();
        let generated_from = previous.len();

        let mut rng = Rng::new(opts.seed);
        for _ in 0..opts.max_tokens {
            let next = sample(&logits, &previous, &opts.sample, &mut rng);
            if next == EOS {
                break;
            }
            previous.push(next);
            // Audio positions restart at zero and are independent of the text's:
            // upstream applies the two positional encodings separately and only
            // then concatenates. So the offset counts audio tokens alone.
            logits = self.engine.s1_step(next, previous.len() - 1)?;
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
