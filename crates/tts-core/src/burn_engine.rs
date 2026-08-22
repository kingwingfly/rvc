//! The native Burn implementation of [`Engine`].
//!
//! Generic over the compute backend and erased behind the trait at the
//! constructor, which is what lets `tts-cli` pick a backend at run time without
//! naming a Burn type anywhere.

use std::path::Path;

use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor, TensorData};
use burn_gptsovits::{Hubert, HubertConfig, SovitsConfig, SovitsPartial, T2s, T2sConfig, T2sState};
use burn_vits::{Spectral, SpectralConfig};

use crate::engine::{Engine, PRIOR_CHANNELS};
use crate::error::{Result, TtsError};

pub struct BurnEngine<B: Backend> {
    hubert: Hubert<B>,
    t2s: T2s<B>,
    sovits: SovitsPartial<B>,
    spectral: Spectral<B>,
    /// The generation in progress, or nothing before the first prompt.
    state: Option<T2sState<B>>,
    device: B::Device,
}

/// Reject a load that left parameters at their initialised values.
///
/// `missing` is what matters — a checkpoint carrying tensors the model has no
/// slot for is merely a superset (a fine-tuned `s2` holds `enc_q`, which only
/// training reads), but a *model parameter* with no tensor behind it is random
/// weight the model will happily run.
///
/// `errors` matters for the same reason and is not covered by `missing`: the
/// applier drops a path that failed to apply from *both* lists, so a shape
/// mismatch — an `s2G` from a later GPT-SoVITS against the v2 config, say —
/// reads as full coverage while the parameter keeps its initialised value. The
/// safetensors store raises those itself; the PyTorch path only reports them.
fn covered(what: &str, result: &burn_kit::ApplyResult) -> Result<()> {
    if let Some(first) = result.errors.first() {
        return Err(TtsError::Weights(format!(
            "{what}: {} of the checkpoint's tensors could not be applied \
             ({first}) — the file does not match the model",
            result.errors.len(),
        )));
    }
    if result.applied.is_empty() || !result.missing.is_empty() {
        return Err(TtsError::Weights(format!(
            "{what}: {} of {} parameters had no weights in the checkpoint \
             (applied {}) — the file's tensor names do not match the model",
            result.missing.len(),
            result.missing.len() + result.applied.len(),
            result.applied.len(),
        )));
    }
    Ok(())
}

impl<B: Backend> BurnEngine<B> {
    /// Load from a `chinese-hubert-base/pytorch_model.bin`, an `s1*.ckpt` and an
    /// `s2G*.pth`.
    ///
    /// `s1` and `s2` are each taken as either an upstream checkpoint or a
    /// `.safetensors` from a fine-tune — the extension decides, so a tuned stage
    /// substitutes for its base wherever the base is accepted.
    pub fn load(hubert: &Path, s1: &Path, s2: &Path, device: &B::Device) -> Result<Self> {
        let weights =
            |what: &str, e: Box<dyn std::error::Error>| TtsError::Weights(format!("{what}: {e}"));

        // Every loader here allows a partial apply, so that a coverage report can
        // be inspected rather than a single mismatch aborting the load. That
        // makes an empty apply a *success* unless somebody looks: a checkpoint
        // whose names no longer match the module tree would leave every parameter
        // at its freshly-initialised value and synthesise noise. Which is exactly
        // the shape of failure worth refusing, so check every report — cnhubert's
        // most of all, since `load_pytorch_into` cannot even fail on a file that
        // is a different model entirely.
        let mut model = Hubert::<B>::new(&HubertConfig::chinese_base(), device);
        let applied = model
            .load_pytorch(hubert)
            .map_err(|e| weights("cnhubert", e))?;
        covered("cnhubert", &applied)?;

        let mut t2s = T2s::<B>::new(&T2sConfig::default(), device);
        let applied = t2s.load_weights(s1).map_err(|e| weights("s1", e))?;
        covered("s1", &applied)?;

        let mut sovits = SovitsPartial::<B>::new(&SovitsConfig::default(), device);
        let applied = sovits.load_weights(s2).map_err(|e| weights("s2", e))?;
        covered("s2", &applied)?;

        Ok(Self {
            hubert: model,
            t2s,
            sovits,
            spectral: Spectral::new(&SpectralConfig::gptsovits_v2_32k(), device),
            state: None,
            device: device.clone(),
        })
    }

    fn ids(&self, values: &[u32]) -> Tensor<B, 2, Int> {
        let n = values.len();
        let data: Vec<i32> = values.iter().map(|&v| v as i32).collect();
        Tensor::from_data(TensorData::new(data, [1, n]), &self.device)
    }

    fn logits(&self, t: Tensor<B, 2>) -> Result<Vec<f32>> {
        t.into_data()
            .to_vec()
            .map_err(|e| TtsError::Weights(format!("s1 logits: {e:?}")))
    }
}

impl<B: Backend> Engine for BurnEngine<B> {
    fn analyse(&mut self, audio: &[f32]) -> Result<(Vec<u32>, Vec<f32>)> {
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

        // `Int` is i64 on LibTorch and i32 on some others, so the width is read
        // from the tensor rather than assumed.
        let data = tokens.into_data();
        let tokens: Vec<u32> = match data.to_vec::<i64>() {
            Ok(v) => v.into_iter().map(|x| x as u32).collect(),
            Err(_) => data
                .to_vec::<i32>()
                .map_err(|e| TtsError::Weights(format!("semantic tokens: {e:?}")))?
                .into_iter()
                .map(|x| x as u32)
                .collect(),
        };
        let speaker = speaker
            .into_data()
            .to_vec()
            .map_err(|e| TtsError::Weights(format!("speaker vector: {e:?}")))?;
        Ok((tokens, speaker))
    }

    fn s1_prompt(
        &mut self,
        phones: &[u32],
        bert: &[f32],
        hidden: usize,
        prompt: &[u32],
    ) -> Result<Vec<f32>> {
        let features: Tensor<B, 3> = Tensor::from_data(
            TensorData::new(bert.to_vec(), [1, phones.len(), hidden]),
            &self.device,
        );
        let text = self.t2s.embed_text(self.ids(phones), features);
        let audio = self.t2s.embed_audio(self.ids(prompt), 0);

        let mut state = self.t2s.state();
        let logits = self.t2s.forward_prompt(text, audio, &mut state);
        self.state = Some(state);
        self.logits(logits)
    }

    fn s1_step(&mut self, token: u32, position: usize) -> Result<Vec<f32>> {
        let state = self
            .state
            .as_mut()
            .ok_or_else(|| TtsError::Weights("s1 step before prompt".into()))?;
        let x = self.t2s.embed_audio(
            Tensor::from_data(TensorData::new(vec![token as i32], [1, 1]), &self.device),
            position,
        );
        let logits = self.t2s.forward(x, state);
        self.logits(logits)
    }

    fn s2(
        &mut self,
        codes: &[u32],
        text: &[u32],
        speaker: &[f32],
        noise: &[f32],
    ) -> Result<Vec<f32>> {
        let frames = codes.len() * 2;
        let noise: Tensor<B, 3> = Tensor::from_data(
            TensorData::new(noise.to_vec(), [1, PRIOR_CHANNELS, frames]),
            &self.device,
        );
        let speaker: Tensor<B, 3> = Tensor::from_data(
            TensorData::new(speaker.to_vec(), [1, speaker.len(), 1]),
            &self.device,
        );
        let audio = self
            .sovits
            .decode_with_noise(self.ids(codes), self.ids(text), speaker, noise);
        audio
            .into_data()
            .to_vec()
            .map_err(|e| TtsError::Weights(format!("synthesised audio was not f32: {e:?}")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_stop_token_matches_the_model_the_engine_loads() {
        // `EOS` is a plain constant so the ONNX engine can share it without
        // depending on `burn-gptsovits`. That only holds while the two agree,
        // and generation running past the end is not an error anywhere — it is
        // an utterance that never stops.
        assert_eq!(crate::engine::EOS, T2sConfig::default().eos());
    }

    /// `covered` is a copy of `burn_kit::check_coverage` rather than a call to
    /// it, and a copy nothing pins is a copy that drifts — so these ask the four
    /// questions that matter: three reports it must refuse, and one it must
    /// accept. The order is the whole of it, since two of the three states are
    /// indistinguishable from success to a check that asks in the wrong one.
    ///
    /// One `ApplyResult` stands in for all three loads. `covered` is generic
    /// over the label and reads nothing else, so cnhubert, `s1` and `s2` are the
    /// same code path with a different string.
    ///
    /// This one is built field by field because `ApplyResult` has no `Default`.
    fn report(
        applied: &[&str],
        missing: &[&str],
        unused: &[&str],
        errors: Vec<burn_kit::ApplyError>,
    ) -> burn_kit::ApplyResult {
        burn_kit::ApplyResult {
            applied: applied.iter().map(|s| s.to_string()).collect(),
            skipped: Vec::new(),
            missing: missing
                .iter()
                .map(|s| (s.to_string(), String::new()))
                .collect(),
            unused: unused.iter().map(|s| s.to_string()).collect(),
            errors,
        }
    }

    /// The refusal `covered` produced, or a panic naming what it let through.
    fn message(result: &burn_kit::ApplyResult) -> String {
        match covered("s2", result) {
            Err(TtsError::Weights(why)) => why,
            other => panic!("this report had to be refused, got {other:?}"),
        }
    }

    #[test]
    fn a_wrong_shaped_checkpoint_is_refused_though_nothing_is_missing() {
        // The failure `covered` exists for, and the one that reads as a clean
        // load if `errors` is asked second: the applier drops an errored path
        // from `applied` and from `missing` alike, so a checkpoint carrying the
        // right names at the wrong shapes arrives with `missing` empty and
        // `applied` full — indistinguishable from success to anyone counting
        // only what is absent. An `s2G` from a later GPT-SoVITS against the v2
        // config is exactly that file.
        let result = report(
            &["enc_p.text_embedding.weight"],
            &[],
            &[],
            vec![burn_kit::ApplyError::ShapeMismatch {
                path: "enc_p.text_embedding.weight".to_string(),
                expected: burn::tensor::Shape::from([732, 192]),
                found: burn::tensor::Shape::from([512, 192]),
            }],
        );
        assert!(result.missing.is_empty(), "the premise of this test");
        assert!(!result.applied.is_empty(), "and the other half of it");

        let err = message(&result);
        assert!(err.contains("text_embedding"), "{err}");
    }

    #[test]
    fn a_parameter_with_no_weights_behind_it_is_refused() {
        // What the two spellings of PyTorch's weight norm cost when only one is
        // accepted: two tensors go missing and the convolution keeps its
        // initialised values, which is not an error anywhere else.
        let result = report(
            &["encoder.layers.0.attention.k_proj.weight"],
            &["encoder.pos_conv_embed.conv.weight_g"],
            &[],
            vec![],
        );
        let err = message(&result);
        assert!(err.contains("had no weights"), "{err}");
    }

    #[test]
    fn a_load_that_applied_nothing_is_refused() {
        // A rename that matched nothing. The applier does not currently produce
        // this shape — it visits the module's parameters, so a foreign file
        // arrives as every parameter *missing* — but the check costs nothing and
        // `burn_store` has narrowed `missing` once already, around `errors`.
        let result = report(&[], &[], &["something.else"], vec![]);
        message(&result);
    }

    #[test]
    fn tensors_left_over_are_not_a_failure() {
        // `unused` is not a defect and must never be treated as one here: a
        // fine-tuned `s2` carries `enc_q`, which only training reads, and the
        // quantiser's three EMA statistics are over on every load. A threshold
        // would refuse working weights, which is worse than not checking.
        let result = report(
            &["dec.conv_pre.weight_g"],
            &[],
            &["enc_q.pre.weight", "quantizer.ema_count"],
            vec![],
        );
        covered("s2", &result).expect("leftover tensors are not a failure");
    }
}
