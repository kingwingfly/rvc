//! Greedy decoding of one 30 s window.
//!
//! Whisper is an autoregressive language model conditioned on audio, and that is
//! also its failure mode: given silence it will happily continue a plausible
//! sentence forever. Three things hold it in check here — a hard cap on
//! generated tokens, the suppression sets the model ships with, and (upstream of
//! this file) feeding it pre-segmented speech rather than long quiet stretches.
//!
//! Nothing here knows which runtime is underneath: everything crosses the
//! [`Engine`] boundary as token ids and `f32` logits.

use crate::engine::Engine;
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

/// Transcribe the window the engine currently holds encoded.
pub fn greedy(
    engine: &mut dyn Engine,
    tokens: &Tokens,
    opts: &DecodeOptions,
) -> crate::Result<Decoded> {
    let language = match &opts.language {
        Some(code) => tokens.language(code)?,
        None => detect_language(engine, tokens)?,
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

    // Language detection above left a token in the cache; start clean.
    engine.restart();
    let mut logits = engine.step(&prompt)?;
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
        let next = argmax_excluding(&logits, &suppress);
        if next == tokens.eot {
            truncated = false;
            break;
        }
        out.push(next);
        logits = engine.step(&[next])?;
    }

    Ok(Decoded {
        tokens: out,
        language,
        truncated,
    })
}

/// Highest-scoring token, ignoring the suppressed ids.
fn argmax_excluding(logits: &[f32], suppress: &[u32]) -> u32 {
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
fn detect_language(engine: &mut dyn Engine, tokens: &Tokens) -> crate::Result<u32> {
    // `Tokens::load` lets `lang_to_id` be absent, defaulting it to an empty map,
    // so this is reachable from a real checkpoint. The old fallback put
    // `<|endoftext|>` in the prompt's language slot and decoded from there —
    // a malformed prompt that produces fluent text and reports nothing.
    if tokens.languages.is_empty() {
        return Err(crate::SttError::Config(
            "this model's generation_config.json has no `lang_to_id`, so there \
             are no language tokens to detect between — name one with \
             `--language`"
                .into(),
        ));
    }

    engine.restart();
    let logits = engine.step(&[tokens.sot])?;
    Ok(tokens
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
        .expect("non-empty, checked above"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::Engine;
    use std::collections::HashMap;

    /// Everything `greedy` asks of a runtime, in the order it asked.
    ///
    /// Recorded as a log rather than as counters because the interesting
    /// properties are about *ordering* — that detection's probe is thrown away
    /// before the real prompt is fed, and that nothing re-encodes the window.
    #[derive(Debug, Clone, PartialEq)]
    enum Ev {
        Encode,
        Restart,
        Step(Vec<u32>),
    }

    /// A synthetic vocabulary, small enough that a logit row fits on a line.
    const VOCAB: usize = 16;
    const SOT: u32 = 10;
    const EOT: u32 = 11;
    const TRANSCRIBE: u32 = 12;
    const TRANSLATE: u32 = 13;
    const NO_TIMESTAMPS: u32 = 14;
    const EN: u32 = 8;
    const ZH: u32 = 9;

    /// An [`Engine`] that is nothing but a hand-written logit table.
    ///
    /// This is the whole reason `Engine` is a trait rather than a type
    /// parameter: the decode loop is shared by both runtimes, so pinning it
    /// here pins it for both, and it needs no weights to do so.
    ///
    /// `rows` are handed out one per `step` call and the last one repeats, so a
    /// table only has to describe as many steps as a test cares about.
    struct Stub {
        rows: Vec<Vec<f32>>,
        calls: usize,
        log: Vec<Ev>,
    }

    impl Stub {
        fn new(rows: Vec<Vec<f32>>) -> Self {
            Self {
                rows,
                calls: 0,
                log: Vec::new(),
            }
        }

        /// The tokens fed to the model, in order: the prompt, then one per step.
        fn fed(&self) -> Vec<Vec<u32>> {
            self.log
                .iter()
                .filter_map(|e| match e {
                    Ev::Step(t) => Some(t.clone()),
                    _ => None,
                })
                .collect()
        }
    }

    impl Engine for Stub {
        fn mel_bins(&self) -> usize {
            80
        }

        fn encode(&mut self, _mel: &[f32], _frames: usize) -> crate::Result<()> {
            self.log.push(Ev::Encode);
            Ok(())
        }

        fn step(&mut self, tokens: &[u32]) -> crate::Result<Vec<f32>> {
            self.log.push(Ev::Step(tokens.to_vec()));
            let row = self.rows[self.calls.min(self.rows.len() - 1)].clone();
            self.calls += 1;
            Ok(row)
        }

        fn restart(&mut self) {
            self.log.push(Ev::Restart);
        }
    }

    /// A logit row that is `-1.0` everywhere except at the named ids.
    ///
    /// The floor is negative and uniform so that "no peak" is unambiguous: ties
    /// resolve to the lowest index, which `ties_resolve_to_the_lowest_id` pins.
    fn row(peaks: &[(u32, f32)]) -> Vec<f32> {
        let mut v = vec![-1.0f32; VOCAB];
        for &(i, x) in peaks {
            v[i as usize] = x;
        }
        v
    }

    /// Whisper's own token layout in miniature. `suppress_at_start` carries the
    /// end-of-text id, as `begin_suppress_tokens` does upstream — a Whisper
    /// decoder may not stop before it has said anything.
    fn tokens() -> Tokens {
        Tokens {
            sot: SOT,
            eot: EOT,
            transcribe: TRANSCRIBE,
            translate: TRANSLATE,
            no_timestamps: NO_TIMESTAMPS,
            languages: HashMap::from([("en".to_string(), EN), ("zh".to_string(), ZH)]),
            suppress: vec![1],
            suppress_at_start: vec![2, EOT],
        }
    }

    /// `expect_err` wants `Debug` on the success type, which `Decoded` is not.
    fn failure(result: crate::Result<Decoded>, why: &str) -> String {
        match result {
            Ok(_) => panic!("{why}"),
            Err(e) => e.to_string(),
        }
    }

    fn opts(language: Option<&str>, max_tokens: usize) -> DecodeOptions {
        DecodeOptions {
            language: language.map(str::to_string),
            translate: false,
            max_tokens,
        }
    }

    /// The prompt Whisper is primed with, and the fact that decoding does not
    /// touch the audio: `encode` is the caller's job (`lib.rs` does it once per
    /// window), so an `Ev::Encode` here would mean the window was re-encoded.
    #[test]
    fn the_prompt_is_start_language_task_and_notimestamps() {
        let mut engine = Stub::new(vec![row(&[(5, 3.0)]), row(&[(EOT, 5.0)])]);
        let decoded = greedy(&mut engine, &tokens(), &opts(Some("en"), 8)).expect("decode");

        assert_eq!(
            engine.log,
            vec![
                Ev::Restart,
                Ev::Step(vec![SOT, EN, TRANSCRIBE, NO_TIMESTAMPS]),
                Ev::Step(vec![5]),
            ]
        );
        assert_eq!(decoded.tokens, vec![5]);
        assert_eq!(decoded.language, EN);
        assert!(!decoded.truncated);
    }

    #[test]
    fn translate_swaps_the_task_token() {
        let mut engine = Stub::new(vec![row(&[(EOT, 5.0)]), row(&[(EOT, 5.0)])]);
        let mut opts = opts(Some("zh"), 8);
        opts.translate = true;
        greedy(&mut engine, &tokens(), &opts).expect("decode");

        assert_eq!(engine.fed()[0], vec![SOT, ZH, TRANSLATE, NO_TIMESTAMPS]);
    }

    /// Detection is one step against a bare `<|startoftranscript|>`, and the
    /// token it leaves in the cache is thrown away before the real prompt.
    ///
    /// The ordering is the assertion: a `Restart` between the probe and the
    /// prompt. Without it the prompt would be decoded with one stale position
    /// in front of it — fluent, confident and wrong, with no error anywhere.
    #[test]
    fn detection_probes_with_a_bare_start_token_and_then_restarts() {
        let mut engine = Stub::new(vec![
            row(&[(EN, 1.0), (ZH, 4.0)]),
            row(&[(5, 3.0)]),
            row(&[(EOT, 5.0)]),
        ]);
        let decoded = greedy(&mut engine, &tokens(), &opts(None, 8)).expect("decode");

        assert_eq!(
            engine.log,
            vec![
                Ev::Restart,
                Ev::Step(vec![SOT]),
                Ev::Restart,
                Ev::Step(vec![SOT, ZH, TRANSCRIBE, NO_TIMESTAMPS]),
                Ev::Step(vec![5]),
            ]
        );
        assert_eq!(decoded.language, ZH, "the higher-scoring language slot");
    }

    #[test]
    fn a_forced_language_is_never_probed_for() {
        let mut engine = Stub::new(vec![row(&[(EOT, 5.0)]), row(&[(EOT, 5.0)])]);
        greedy(&mut engine, &tokens(), &opts(Some("en"), 8)).expect("decode");

        assert_eq!(engine.log.iter().filter(|e| **e == Ev::Restart).count(), 1);
        assert!(!engine.log.contains(&Ev::Step(vec![SOT])), "no probe");
    }

    #[test]
    fn an_unknown_forced_language_is_an_error_before_any_step() {
        let mut engine = Stub::new(vec![row(&[(EOT, 5.0)])]);
        failure(
            greedy(&mut engine, &tokens(), &opts(Some("xx"), 8)),
            "no such code",
        );

        assert!(engine.log.is_empty(), "nothing was asked of the model");
    }

    /// The loop's defining property: step *n + 1* feeds the argmax of step *n*.
    ///
    /// `fed` is what carries it — the prompt is one call of exactly four
    /// tokens, and every call after it is exactly one. An engine's cache
    /// therefore advances by exactly one position per generated token, which is
    /// the bookkeeping a drift of one would break.
    #[test]
    fn the_token_fed_at_each_step_is_the_previous_steps_argmax() {
        let mut engine = Stub::new(vec![
            row(&[(5, 3.0)]),
            row(&[(3, 3.0)]),
            row(&[(6, 3.0)]),
            row(&[(EOT, 9.0)]),
        ]);
        let decoded = greedy(&mut engine, &tokens(), &opts(Some("en"), 32)).expect("decode");

        assert_eq!(decoded.tokens, vec![5, 3, 6]);
        assert_eq!(
            engine.fed(),
            vec![
                vec![SOT, EN, TRANSCRIBE, NO_TIMESTAMPS],
                vec![5],
                vec![3],
                vec![6]
            ]
        );
        let widths: Vec<usize> = engine.fed().iter().map(Vec::len).collect();
        assert_eq!(
            widths,
            vec![4, 1, 1, 1],
            "one position per step after the prompt"
        );
        assert!(
            !decoded.tokens.contains(&EOT),
            "the stop token is not emitted"
        );
        assert!(!decoded.truncated);
    }

    /// The cap is a cap, and hitting it is reported rather than passed off as
    /// the end of the utterance.
    ///
    /// The step count is one more than the tokens generated: the final token's
    /// logits are computed and then discarded when the loop's bound runs out.
    #[test]
    fn the_token_cap_bounds_the_run_and_is_reported() {
        let mut engine = Stub::new(vec![row(&[(5, 3.0)])]);
        let decoded = greedy(&mut engine, &tokens(), &opts(Some("en"), 3)).expect("decode");

        assert_eq!(decoded.tokens, vec![5, 5, 5]);
        assert!(decoded.truncated, "cut off, not finished");
        assert_eq!(engine.fed().len(), 4, "the prompt, then one step per token");
    }

    #[test]
    fn a_zero_cap_generates_nothing_and_still_reports_truncation() {
        let mut engine = Stub::new(vec![row(&[(5, 3.0)])]);
        let decoded = greedy(&mut engine, &tokens(), &opts(Some("en"), 0)).expect("decode");

        assert!(decoded.tokens.is_empty());
        assert!(decoded.truncated);
        assert_eq!(engine.fed(), vec![vec![SOT, EN, TRANSCRIBE, NO_TIMESTAMPS]]);
    }

    /// `begin_suppress_tokens` applies at the **first generated position and no
    /// other**, which is an off-by-one waiting to happen in either direction.
    ///
    /// The table is one row repeated, so the only thing that can distinguish
    /// the two steps is the suppression set: id 2 loses at step 0 and wins at
    /// step 1. Applying the start set one step late, or at every step, reverses
    /// or flattens this.
    #[test]
    fn begin_suppression_applies_to_the_first_token_and_no_other() {
        let mut engine = Stub::new(vec![row(&[(2, 9.0), (5, 3.0)])]);
        let decoded = greedy(&mut engine, &tokens(), &opts(Some("en"), 2)).expect("decode");

        assert_eq!(decoded.tokens, vec![5, 2]);
    }

    /// A leading `<|endoftext|>` must not empty the segment.
    ///
    /// Whisper's `begin_suppress_tokens` carries the stop token for exactly
    /// this reason; without it, a model that is briefly unsure at the first
    /// position returns nothing at all and the clip reads as silence.
    #[test]
    fn an_immediate_end_of_text_is_suppressed_at_the_first_position() {
        let mut engine = Stub::new(vec![row(&[(EOT, 9.0), (5, 3.0)]), row(&[(EOT, 9.0)])]);
        let decoded = greedy(&mut engine, &tokens(), &opts(Some("en"), 8)).expect("decode");

        assert_eq!(decoded.tokens, vec![5]);
        assert!(!decoded.truncated);
    }

    /// The other side of that: once past the first position, `<|endoftext|>`
    /// stops the loop immediately, which is the empty transcript `lib.rs` turns
    /// into `None` for a clip of breath or room tone.
    #[test]
    fn end_of_text_at_the_first_position_stops_when_it_is_not_begin_suppressed() {
        let mut suppressionless = tokens();
        suppressionless.suppress_at_start = vec![2];
        let mut engine = Stub::new(vec![row(&[(EOT, 9.0)])]);
        let decoded = greedy(&mut engine, &suppressionless, &opts(Some("en"), 8)).expect("decode");

        assert!(decoded.tokens.is_empty());
        assert!(!decoded.truncated, "it finished; it was not cut off");
    }

    #[test]
    fn general_suppression_holds_at_every_step() {
        let mut engine = Stub::new(vec![row(&[(1, 9.0), (5, 3.0)])]);
        let decoded = greedy(&mut engine, &tokens(), &opts(Some("en"), 4)).expect("decode");

        assert_eq!(decoded.tokens, vec![5, 5, 5, 5]);
        assert!(!decoded.tokens.contains(&1));
    }

    /// A model whose `generation_config.json` carries no `lang_to_id` cannot be
    /// asked to detect a language, and says so.
    ///
    /// `Tokens::load` defaults that map to empty rather than erroring (see
    /// `tokenizer.rs`'s `a_config_with_no_lang_to_id_still_loads_with_no_languages`),
    /// so before this check the fallback put `<|endoftext|>` into the prompt's
    /// language slot and decoded from there — a malformed prompt that produces
    /// fluent text and no error.
    #[test]
    fn a_model_with_no_language_tokens_is_refused_rather_than_prompted_with_eot() {
        let mut languageless = tokens();
        languageless.languages.clear();
        let mut engine = Stub::new(vec![row(&[(5, 3.0)]), row(&[(EOT, 5.0)])]);

        let message = failure(
            greedy(&mut engine, &languageless, &opts(None, 8)),
            "nothing to detect",
        );
        assert!(message.contains("lang_to_id"), "{message}");
        assert!(
            !engine.fed().iter().any(|t| t.contains(&EOT)),
            "the stop token never reached the prompt"
        );
    }

    #[test]
    fn ties_resolve_to_the_lowest_id() {
        assert_eq!(argmax_excluding(&[1.0, 1.0, 1.0], &[]), 0);
        assert_eq!(argmax_excluding(&[1.0, 1.0, 1.0], &[0]), 1);
        assert_eq!(argmax_excluding(&[1.0, 2.0, 2.0], &[]), 1);
    }

    #[test]
    fn suppressed_ids_lose_however_high_they_score() {
        assert_eq!(argmax_excluding(&[0.0, 9.0, 1.0], &[1]), 2);
        assert_eq!(argmax_excluding(&[0.0, 9.0, 1.0], &[1, 2]), 0);
    }

    /// There is no softmax anywhere in this file — the loop is an argmax over
    /// raw logits — so a non-finite value can only have arrived from an engine.
    /// `burn-whisper` shipped exactly that once: a reversed causal mask makes a
    /// decode step's only position `-inf`, and `-inf` through a softmax is
    /// `NaN` rather than an error.
    ///
    /// This pins what the loop does with such a row, which is **not** to fail:
    /// `v > best.1` is false for every comparison against `NaN`, so the seed
    /// wins and id 0 comes out — repeatedly, up to the cap. Two consequences
    /// worth knowing before reading a garbled transcript as a bad checkpoint:
    /// the id is 0 whatever the suppression set says, and the tell is a run of
    /// one repeated token together with `truncated`.
    #[test]
    fn a_row_of_nan_yields_id_zero_repeatedly_rather_than_an_error() {
        assert_eq!(argmax_excluding(&[f32::NAN; 4], &[]), 0);
        assert_eq!(
            argmax_excluding(&[f32::NAN; 4], &[0]),
            0,
            "the seed is returned unchecked, so suppression does not hold here"
        );
        assert_eq!(argmax_excluding(&[], &[]), 0, "an empty row, likewise");

        let mut engine = Stub::new(vec![vec![f32::NAN; VOCAB]]);
        let decoded =
            greedy(&mut engine, &tokens(), &opts(Some("en"), 3)).expect("no error is raised");
        assert_eq!(decoded.tokens, vec![0, 0, 0]);
        assert!(decoded.truncated);
    }
}
