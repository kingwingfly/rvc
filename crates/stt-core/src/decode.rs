//! Greedy decoding of one 30 s window.
//!
//! Whisper is an autoregressive language model conditioned on audio, and that is
//! also its failure mode: given silence it will happily continue a plausible
//! sentence forever. Three things hold it in check here — a hard cap on
//! generated tokens, the suppression sets the model ships with, and (upstream of
//! this file) feeding it pre-segmented speech rather than long quiet stretches.

use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor, TensorData};
use burn_whisper::{TextDecoder, Whisper};

use crate::tokenizer::Tokens;

/// How the decoder is primed and how far it is allowed to run.
#[derive(Debug, Clone)]
pub struct DecodeOptions {
    /// ISO code to force, or `None` to let the model detect it. Forcing helps
    /// on short or breathy clips, where detection is least reliable.
    pub language: Option<String>,
    /// Translate to English instead of transcribing verbatim.
    pub translate: bool,
    /// Cap on generated tokens per window. Whisper's own default is 224 (half
    /// its 448-wide positional table) and a lower cap bounds what a
    /// hallucination loop costs — but a dense 30 s segment can legitimately need
    /// more, so hitting this is reported rather than passed off as the end.
    pub max_tokens: usize,
}

impl Default for DecodeOptions {
    fn default() -> Self {
        Self {
            language: None,
            translate: false,
            max_tokens: 224,
        }
    }
}

/// What one window produced.
pub struct Decoded {
    /// Generated token ids, control tokens included.
    pub tokens: Vec<u32>,
    /// The language token the prompt carried, as an id.
    pub language: u32,
    /// Stopped at `max_tokens` rather than at `<|endoftext|>`, so the text is
    /// cut off mid-utterance. Either the segment is too dense for the cap or the
    /// model is looping.
    pub truncated: bool,
}

/// Transcribe one already-encoded window.
///
/// `audio` is the encoder output, `[1, frames, d_model]`. Returns the tokens the
/// decoder generated, stopping at `<|endoftext|>` or the token cap.
pub fn greedy<B: Backend>(
    model: &Whisper<B>,
    audio: Tensor<B, 3>,
    tokens: &Tokens,
    opts: &DecodeOptions,
    device: &B::Device,
) -> crate::Result<Decoded> {
    let language = match &opts.language {
        Some(code) => tokens.language(code)?,
        None => detect_language(model, audio.clone(), tokens, device),
    };

    // The prompt Whisper was trained on: start, language, task, and a request
    // for plain text. Timestamps are off — segment boundaries come from the
    // slicer upstream, which knows where the silences are.
    let prompt = vec![
        tokens.sot,
        language,
        if opts.translate {
            tokens.translate
        } else {
            tokens.transcribe
        },
        tokens.no_timestamps,
    ];

    let mut state = model.decoder.state();
    let mut logits = forward(&model.decoder, &prompt, audio.clone(), &mut state, device);
    let mut out = Vec::new();
    let mut truncated = true;

    for step in 0..opts.max_tokens {
        let suppress = if step == 0 {
            [
                tokens.suppress.as_slice(),
                tokens.suppress_at_start.as_slice(),
            ]
            .concat()
        } else {
            tokens.suppress.clone()
        };
        let next = argmax_excluding(logits, &suppress);
        if next == tokens.eot {
            truncated = false;
            break;
        }
        out.push(next);
        logits = forward(&model.decoder, &[next], audio.clone(), &mut state, device);
    }

    Ok(Decoded {
        tokens: out,
        language,
        truncated,
    })
}

/// One decoder call, returning the logits for the last fed position.
fn forward<B: Backend>(
    decoder: &TextDecoder<B>,
    ids: &[u32],
    audio: Tensor<B, 3>,
    state: &mut burn_whisper::DecodeState<B>,
    device: &B::Device,
) -> Vec<f32> {
    let ids: Vec<i32> = ids.iter().map(|&i| i as i32).collect();
    let n = ids.len();
    let input: Tensor<B, 2, Int> = Tensor::from_data(TensorData::new(ids, [1, n]), device);

    let logits = decoder.forward(input, audio, state);
    let [_, seq, vocab] = logits.dims();
    logits
        .slice([0..1, seq - 1..seq, 0..vocab])
        .into_data()
        .to_vec()
        .expect("logits are f32")
}

/// Highest-scoring token, ignoring the suppressed ids.
fn argmax_excluding(logits: Vec<f32>, suppress: &[u32]) -> u32 {
    let mut best = (0u32, f32::NEG_INFINITY);
    for (i, &v) in logits.iter().enumerate() {
        let i = i as u32;
        if v > best.1 && !suppress.contains(&i) {
            best = (i, v);
        }
    }
    best.0
}

/// Which language token the model expects after `<|startoftranscript|>`.
///
/// One decoder step against the audio, reading only the language slots — the
/// same trick the reference uses, and much cheaper than decoding twice.
fn detect_language<B: Backend>(
    model: &Whisper<B>,
    audio: Tensor<B, 3>,
    tokens: &Tokens,
    device: &B::Device,
) -> u32 {
    let mut state = model.decoder.state();
    let logits = forward(&model.decoder, &[tokens.sot], audio, &mut state, device);
    tokens
        .languages
        .values()
        .copied()
        .max_by(|a, b| {
            let score = |i: &u32| {
                logits
                    .get(*i as usize)
                    .copied()
                    .unwrap_or(f32::NEG_INFINITY)
            };
            score(a).total_cmp(&score(b))
        })
        .unwrap_or(tokens.eot)
}
