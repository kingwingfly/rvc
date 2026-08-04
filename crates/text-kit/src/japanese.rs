//! Japanese grapheme-to-phoneme.
//!
//! Reproduces `GPT_SoVITS/text/japanese.py` — which is espnet's
//! `pyopenjtalk_g2p_prosody` wrapped in a splitter — on top of
//! [`jpreprocess`](https://crates.io/crates/jpreprocess), a pure-Rust rewrite of
//! OpenJTalk that emits the same HTS full-context labels `pyopenjtalk.make_label`
//! does. So the analyser changes language and the rules that read its output do
//! not: everything below parses label *strings*, and the tests hand-write them.
//!
//! ## The dictionary is the caller's, not this crate's
//!
//! NAIST-JDic is 28.7 MB and `jpreprocess`'s own `naist-jdic` feature downloads
//! it from a `build.rs`. That is the shape `rvc-core/build.rs` exists to prevent
//! for LibTorch, so the feature stays off and the dictionary arrives as a path:
//! [`JapaneseDict::open`] loads a directory somebody else fetched, and
//! [`crate::phonemize`] takes the handle as an argument. **A Japanese run with no
//! dictionary is an error and never a fallback to another front-end** — the
//! nonsense a mis-chosen front-end produces is fluent, which is exactly what
//! makes it hard to notice.
//!
//! ## What the prosody symbols are
//!
//! Phonemes here carry accent structure, which Mandarin's tone digits carry
//! instead: `[` where the pitch rises, `]` where it falls, `#` at an accent
//! phrase boundary, `_` for a pause. Three of the four are in the 732-symbol
//! table; `#` is not, and it is emitted anyway because upstream's `cleaner.py`
//! folds it to `UNK` and that fold is what the checkpoint was trained on.
//!
//! `word2ph` is `None`, as upstream returns it for everything that is not
//! Chinese — `japanese.py` says so in a `todo` and never implements it.

use std::path::{Path, PathBuf};

use jpreprocess::{DefaultTokenizer, JPreprocess, SystemDictionaryConfig};

use crate::{Phonemes, Result, TextError, normalize};

/// An opened NAIST-JDic dictionary, ready to analyse Japanese.
///
/// Loading one reads tens of megabytes off disk, so open it once and pass it to
/// every call. Cheap to share: the analyser is `Send + Sync`.
pub struct JapaneseDict {
    engine: JPreprocess<DefaultTokenizer>,
    /// Kept only so that a failure can name the directory it came from — the
    /// commonest cause is a dictionary built by a different `jpreprocess`.
    path: PathBuf,
}

impl JapaneseDict {
    /// Open an unpacked NAIST-JDic directory.
    pub fn open(dir: &Path) -> Result<Self> {
        let engine = SystemDictionaryConfig::File(dir.to_path_buf())
            .load()
            .map_err(|reason| TextError::JapaneseDictionary {
                path: dir.to_path_buf(),
                reason: reason.to_string(),
            })?;
        Ok(Self {
            engine: JPreprocess::with_dictionaries(engine, None),
            path: dir.to_path_buf(),
        })
    }

    /// HTS full-context labels for one run of Japanese characters.
    fn labels(&self, text: &str) -> Result<Vec<String>> {
        self.engine
            .extract_fullcontext(text)
            .map(|labels| labels.iter().map(ToString::to_string).collect())
            .map_err(|reason| TextError::JapaneseDictionary {
                path: self.path.clone(),
                reason: reason.to_string(),
            })
    }
}

/// Phonemize Japanese text.
///
/// `word2ph` is always `None` — see the module note.
pub fn phonemize(text: &str, dict: Option<&JapaneseDict>) -> Result<Phonemes> {
    let dict = dict.ok_or(TextError::NoJapaneseDictionary)?;
    let normalized = text_normalize(text);

    // Upstream splits on every character the analyser is not given, phonemizes
    // each run of the ones it is, and puts the marks back between them — so the
    // two lists interleave and there is always one more run than mark.
    let mut runs = vec![String::new()];
    let mut marks = Vec::new();
    for c in normalized.chars() {
        if is_japanese(c) {
            runs.last_mut().expect("a run to push onto").push(c);
        } else {
            marks.push(c);
            runs.push(String::new());
        }
    }

    let mut phones: Vec<String> = Vec::new();
    for (i, run) in runs.iter().enumerate() {
        if !run.is_empty() {
            // Upstream takes `[1:-1]` of the symbols, dropping the `^` and the
            // `$`/`?` that `g2p_prosody` brackets an utterance with. Neither
            // reaches the symbol table, so neither is in it.
            let mut symbols = g2p_prosody(&dict.labels(run)?);
            symbols.pop();
            if !symbols.is_empty() {
                symbols.remove(0);
            }
            phones.append(&mut symbols);
        }
        // A bare space is dropped rather than emitted: nothing in the table
        // spells one, so it would arrive at the model as `UNK` — an audible
        // stumble where the text only had a word gap.
        if let Some(&mark) = marks.get(i) {
            if mark != ' ' {
                phones.push(post_replace(mark));
            }
        }
    }

    Ok(Phonemes {
        phones,
        word2ph: None,
        normalized,
    })
}

/// The value espnet's `_numeric_feature_by_regex` returns for a field that does
/// not match — `xx`, OpenJTalk's "not applicable", is the usual way that happens.
///
/// **It is load-bearing rather than an error code.** The three prosody rules
/// compare against whatever comes back, so substituting `0` would make `a1 == 0`
/// fire on every missing accent field, and an `Option` short-circuited to "no
/// mark" would silence the rules at exactly the boundaries they exist for.
const MISSING: i32 = -50;

/// Phoneme + prosody symbols for one utterance's HTS full-context labels.
///
/// espnet's `pyopenjtalk_g2p_prosody`, which GPT-SoVITS copied verbatim and
/// which implements *Prosodic features control by symbols as input of
/// sequence-to-sequence acoustic modeling for neural TTS*.
///
/// Takes label strings rather than a [`JapaneseDict`], which is what keeps the
/// whole rule set testable with no dictionary on the machine at all.
fn g2p_prosody(labels: &[String]) -> Vec<String> {
    let mut phones: Vec<String> = Vec::with_capacity(labels.len());

    for (n, label) in labels.iter().enumerate() {
        let Some(p3) = current_phoneme(label) else {
            continue;
        };
        // Unvoiced vowels are spelled in capitals; upstream reads them as the
        // ordinary vowel, so the model never sees a separate symbol for one.
        let p3 = if matches!(p3, "A" | "E" | "I" | "O" | "U") {
            p3.to_ascii_lowercase()
        } else {
            p3.to_string()
        };

        if p3 == "sil" {
            // `sil` only ever appears at the two ends of an utterance.
            if n == 0 {
                phones.push("^".to_string());
            } else if n == labels.len() - 1 {
                // `e3` is 1 for a question and 0 for a statement. Anything else
                // appends nothing at all, which is upstream's behaviour and not
                // an oversight to fix here: the `[1:-1]` above would then eat a
                // real phoneme, and matching that is the point of the port.
                match numeric_feature(label, "!", '_', false) {
                    0 => phones.push("$".to_string()),
                    1 => phones.push("?".to_string()),
                    _ => {}
                }
            }
            continue;
        }
        if p3 == "pau" {
            phones.push("_".to_string());
            continue;
        }
        phones.push(p3.clone());

        // Accent type and position. `a1` is the only signed field.
        let a1 = numeric_feature(label, "/A:", '+', true);
        let a2 = numeric_feature(label, "+", '+', false);
        let a3 = numeric_feature(label, "+", '/', false);
        // Mora count of the accent phrase.
        let f1 = numeric_feature(label, "/F:", '_', false);
        // The *next* label's `a2` is what says whether this phoneme ends a mora
        // group. The last label has no successor, so it falls to the sentinel —
        // upstream never reaches that line, because the last label is the
        // closing `sil` and `continue`s above.
        let a2_next = labels
            .get(n + 1)
            .map_or(MISSING, |next| numeric_feature(next, "+", '+', false));

        // `contains` rather than a set membership test on purpose: upstream's
        // `p3 in "aeiouAEIOUNcl"` is Python substring containment, which is what
        // admits the two-character sokuon `cl` alongside the single vowels.
        if a3 == 1 && a2_next == 1 && "aeiouAEIOUNcl".contains(p3.as_str()) {
            phones.push("#".to_string());
        } else if a1 == 0 && a2_next == a2 + 1 && a2 != f1 {
            phones.push("]".to_string());
        } else if a2 == 1 && a2_next == 2 {
            phones.push("[".to_string());
        }
    }

    phones
}

/// `p3`, the phoneme a label is about: upstream's `\-(.*?)\+` over
/// `p1^p2-p3+p4=p5/A:…`.
fn current_phoneme(label: &str) -> Option<&str> {
    let (_, after) = label.split_once('-')?;
    let (p3, _) = after.split_once('+')?;
    Some(p3)
}

/// espnet's `_numeric_feature_by_regex`, sentinel and all.
///
/// `open` is the literal that precedes the number and `close` the character that
/// must follow it, so `("/F:", '_')` spells `/F:(\d+)_`; `signed` widens the
/// digits to `[0-9\-]+`, which only `/A:` needs. Scanning resumes past a
/// candidate that fails rather than giving up, because `re.search` does — in
/// `+h=o/A:-3+1+7`, `\+(\d+)\+` has to walk past `+h` to reach `+1+`.
fn numeric_feature(label: &str, open: &str, close: char, signed: bool) -> i32 {
    let mut from = 0;
    while let Some(offset) = label[from..].find(open) {
        let start = from + offset + open.len();
        let end = start
            + label[start..]
                .find(|c: char| !(c.is_ascii_digit() || (signed && c == '-')))
                .unwrap_or(label.len() - start);
        if label[end..].starts_with(close) {
            if let Ok(value) = label[start..end].parse() {
                return value;
            }
        }
        from = start;
    }
    MISSING
}

/// Upstream's `_japanese_characters`: what goes to the analyser, as opposed to
/// the marks that get split out and put back around it.
fn is_japanese(c: char) -> bool {
    c.is_ascii_alphanumeric()
        || matches!(
            c as u32,
            // 々, the iteration mark — the one character `segment::script` also
            // has to know about, or a text that is only 人々 routes to Chinese.
            0x3005
            // Hiragana and katakana.
            | 0x3040..=0x30FF
            // CJK Unified Ideographs.
            | 0x4E00..=0x9FFF
            // Fullwidth digits and Latin letters, and halfwidth katakana.
            // Upstream's explicit range starts at fullwidth *one*; fullwidth
            // zero is covered by its `\d`, which matches every Unicode decimal
            // digit, so it belongs here too.
            | 0xFF10..=0xFF19 | 0xFF21..=0xFF3A | 0xFF41..=0xFF5A | 0xFF66..=0xFF9D
        )
}

/// The marks upstream's `replace_consecutive_punctuation` collapses — the six in
/// GPT-SoVITS's `punctuation` list, which is not the same set as
/// [`normalize::is_mark`]'s: this one carries `…` and no others.
fn is_punctuation(c: char) -> bool {
    matches!(c, '!' | '?' | '…' | ',' | '.' | '-')
}

/// Upstream's `symbols_to_japanese`, `lower()` and `text_normalize` in the order
/// `cleaner.py` applies them.
fn text_normalize(text: &str) -> String {
    // `％` is a mark, so the split would hand it to `post_replace` and it would
    // become `UNK`; upstream spells it out before the split can see it.
    let spelled = text.replace('％', "パーセント");

    // Lowercased whole rather than per run: Latin letters count as Japanese
    // characters here, so they reach the analyser, whose dictionary is lowercase.
    let lowered = spelled.to_lowercase();

    // `replace_consecutive_punctuation`, for the reason `normalize::english`
    // gives: a run of marks becomes a run of pauses, and `s1` answers a long
    // pause by copying the reference clip instead of continuing the line. Note
    // that Japanese's own 。and、are *not* in upstream's set, so 。。 survives
    // this and is folded to `..` later by `post_replace`.
    let mut out = String::with_capacity(lowered.len());
    for c in lowered.chars() {
        if is_punctuation(c) && out.ends_with(is_punctuation) {
            continue;
        }
        out.push(c);
    }
    out
}

/// Upstream's `post_replace_ph` over one mark.
///
/// Its whole table is punctuation folding that [`normalize::punctuation`]
/// already spells, plus two entries that table has no reason to carry — `·` and
/// a newline, neither of which can reach the Mandarin or English front-ends.
/// Reusing it also folds `：` and `—`, which upstream leaves to become `UNK`;
/// that is a divergence, and it is the direction that produces a pause where
/// upstream produces a stumble.
fn post_replace(mark: char) -> String {
    match mark {
        '·' => ",".to_string(),
        '\n' => ".".to_string(),
        c => match normalize::punctuation(c) {
            Some(folded) => folded.to_string(),
            None => c.to_string(),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One HTS full-context label, with only the fields the prosody rules read
    /// filled in. Everything else is `xx`, which is what OpenJTalk writes when a
    /// field does not apply — and which the `-50` sentinel exists for.
    fn label(p3: &str, a1: &str, a2: &str, a3: &str, f1: &str) -> String {
        format!(
            "x^x-{p3}+x=x/A:{a1}+{a2}+{a3}/B:xx-xx_xx/C:xx_xx+xx/D:xx+xx_xx\
             /E:xx_xx!xx_xx-xx/F:{f1}_xx#xx_xx@xx_xx|xx_xx/G:xx_xx%xx_xx_xx\
             /H:xx_xx/I:xx-xx@xx+xx&xx-xx|xx+xx/J:xx_xx/K:xx+xx-xx"
        )
    }

    /// The opening or closing `sil`, whose `e3` decides statement from question.
    fn silence(e3: &str) -> String {
        format!(
            "x^x-sil+x=x/A:xx+xx+xx/B:xx-xx_xx/C:xx_xx+xx/D:xx+xx_xx\
             /E:xx_xx!{e3}_xx-xx/F:xx_xx#xx_xx@xx_xx|xx_xx/G:xx_xx%xx_xx_xx\
             /H:xx_xx/I:xx-xx@xx+xx&xx-xx|xx+xx/J:xx_xx/K:xx+xx-xx"
        )
    }

    #[test]
    fn the_sentinels_bracket_the_utterance_and_say_which_mood_it_is() {
        let statement = g2p_prosody(&[
            silence("0"),
            label("k", "xx", "xx", "xx", "xx"),
            silence("0"),
        ]);
        assert_eq!(statement, ["^", "k", "$"]);

        let question = g2p_prosody(&[
            silence("0"),
            label("k", "xx", "xx", "xx", "xx"),
            silence("1"),
        ]);
        assert_eq!(question, ["^", "k", "?"]);
    }

    #[test]
    fn a_pause_becomes_an_underscore() {
        // `_` is in the symbol table, so a mid-sentence breath survives into the
        // model as a pause rather than as `UNK`.
        let out = g2p_prosody(&[
            silence("0"),
            label("k", "xx", "xx", "xx", "xx"),
            "x^x-pau+x=x/A:xx+xx+xx/F:xx_xx".to_string(),
            label("t", "xx", "xx", "xx", "xx"),
            silence("0"),
        ]);
        assert_eq!(out, ["^", "k", "_", "t", "$"]);
    }

    #[test]
    fn an_unvoiced_vowel_reads_as_the_ordinary_vowel() {
        let out = g2p_prosody(&[label("I", "xx", "xx", "xx", "xx")]);
        assert_eq!(out, ["i"]);
    }

    #[test]
    fn an_accent_phrase_boundary_emits_a_hash() {
        // a3 == 1 and the next label's a2 == 1, with a vowel in hand.
        let out = g2p_prosody(&[
            label("a", "xx", "2", "1", "2"),
            label("k", "xx", "1", "xx", "xx"),
        ]);
        assert_eq!(out[0], "a");
        assert_eq!(out[1], "#");
    }

    #[test]
    fn the_hash_rule_admits_the_two_character_sokuon() {
        // `cl` is not a vowel; it qualifies because upstream's test is Python
        // substring containment against "aeiouAEIOUNcl", not set membership.
        let out = g2p_prosody(&[
            label("cl", "xx", "2", "1", "2"),
            label("k", "xx", "1", "xx", "xx"),
        ]);
        assert_eq!(out, ["cl", "#", "k"]);

        // A consonant with the same fields gets nothing, which is what makes the
        // containment test load-bearing rather than decorative.
        let out = g2p_prosody(&[
            label("k", "xx", "2", "1", "2"),
            label("t", "xx", "1", "xx", "xx"),
        ]);
        assert_eq!(out, ["k", "t"]);
    }

    #[test]
    fn pitch_falls_where_the_accent_nucleus_ends() {
        // a1 == 0, the next a2 is this one plus one, and a2 is not the last mora.
        let out = g2p_prosody(&[
            label("a", "0", "2", "xx", "4"),
            label("k", "xx", "3", "xx", "xx"),
        ]);
        assert_eq!(out, ["a", "]", "k"]);

        // a2 == f1 is the phrase's last mora: the fall belongs to the next
        // phrase's start, not here.
        let out = g2p_prosody(&[
            label("a", "0", "2", "xx", "2"),
            label("k", "xx", "3", "xx", "xx"),
        ]);
        assert_eq!(out, ["a", "k"]);
    }

    #[test]
    fn pitch_rises_from_the_first_mora_to_the_second() {
        let out = g2p_prosody(&[
            label("o", "xx", "1", "xx", "xx"),
            label("k", "xx", "2", "xx", "xx"),
        ]);
        assert_eq!(out, ["o", "[", "k"]);
    }

    #[test]
    fn a_missing_field_reads_as_the_sentinel_and_not_as_zero() {
        assert_eq!(
            numeric_feature(&label("a", "xx", "xx", "xx", "xx"), "/A:", '+', true),
            MISSING
        );
        // `/A:-3+…` is the only signed field, and dropping the sign would turn a
        // pitch-fall test into its opposite.
        assert_eq!(
            numeric_feature(&label("a", "-3", "1", "7", "4"), "/A:", '+', true),
            -3
        );
        // `\+(\d+)\+` has to walk past `+x=` before it reaches `/A:-3+1+7`.
        assert_eq!(
            numeric_feature(&label("a", "-3", "1", "7", "4"), "+", '+', false),
            1
        );
        assert_eq!(
            numeric_feature(&label("a", "-3", "1", "7", "4"), "+", '/', false),
            7
        );
    }

    #[test]
    fn the_last_label_has_no_successor_to_read_a2_from() {
        // Upstream would raise `IndexError` here and never does, because the
        // last label is always the closing `sil`. A sequence that ends on a
        // phoneme falls to the sentinel instead, and `-50` is chosen so that no
        // rule can fire on it: `a2_next == a2 + 1` would need an `a2` of -51,
        // and `a2` is a mora index. A `0` in its place would fire the fall rule
        // wherever `a2` were -1 and — far likelier — the rise rule is only
        // silent here because -50 is not 2.
        let last = label("a", "0", "1", "xx", "4");

        // With a successor, this exact label falls.
        assert_eq!(
            g2p_prosody(&[last.clone(), label("k", "xx", "2", "xx", "xx")]),
            ["a", "]", "k"]
        );
        // Alone, it is the last label, and nothing fires.
        assert_eq!(g2p_prosody(&[last]), ["a"]);
    }

    #[test]
    fn a_label_whose_accent_fields_are_all_absent_emits_no_mark() {
        // The first label of an utterance is the opening `sil`, whose `/A:` is
        // `xx+xx+xx`; every rule has to stay silent on that rather than reading
        // the sentinel as a number.
        let out = g2p_prosody(&[
            silence("0"),
            label("k", "xx", "xx", "xx", "xx"),
            label("a", "xx", "xx", "xx", "xx"),
            silence("0"),
        ]);
        assert_eq!(out, ["^", "k", "a", "$"]);
    }

    #[test]
    fn consecutive_marks_collapse_but_japanese_ones_do_not() {
        assert_eq!(text_normalize("a?!?b"), "a?b");
        // 。and、are outside upstream's punctuation list, so they survive
        // normalization and are folded to `.` and `,` only after the analyser.
        assert_eq!(text_normalize("あ。。い"), "あ。。い");
    }

    #[test]
    fn a_percent_sign_is_spelled_out_before_the_split_can_drop_it() {
        assert_eq!(text_normalize("50％"), "50パーセント");
    }

    #[test]
    fn marks_fold_onto_the_symbols_the_table_has() {
        assert_eq!(post_replace('。'), ".");
        assert_eq!(post_replace('、'), ",");
        assert_eq!(post_replace('？'), "?");
        assert_eq!(post_replace('·'), ",");
        assert_eq!(post_replace('\n'), ".");
        // Unknown marks pass through unchanged, exactly as upstream's table does
        // — `cleaner.py` is what turns them into `UNK`.
        assert_eq!(post_replace('«'), "«");
    }

    #[test]
    fn japanese_characters_are_what_the_analyser_gets() {
        for c in ['あ', 'ア', '漢', '々', 'a', '1', 'Ａ', 'ｱ'] {
            assert!(is_japanese(c), "{c} should go to the analyser");
        }
        for c in ['。', '、', '？', ' ', '«'] {
            assert!(!is_japanese(c), "{c} should be split out as a mark");
        }
    }

    #[test]
    fn japanese_without_a_dictionary_says_so_rather_than_guessing() {
        let err = phonemize("こんにちは", None).unwrap_err();
        assert!(
            err.to_string().contains("dictionary"),
            "the error has to name what is missing: {err}"
        );
    }
}
