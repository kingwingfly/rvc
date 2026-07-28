//! Mandarin grapheme-to-phoneme.
//!
//! Reproduces `GPT_SoVITS/text/chinese2.py::g2p`: segment into words, get each
//! character's initial and tone-marked final, apply tone sandhi, then map
//! `initial + final` through the opencpop table to the two phonemes the model
//! was trained on. The tone digit is appended to the final phoneme, which is why
//! the table is keyed on the *toneless* syllable.
//!
//! One deliberate difference, and it is the accuracy ceiling of this file:
//! upstream reads pronunciations with `pypinyin`, which carries a ~130k-entry
//! phrase dictionary (and optionally `g2pw`, a BERT polyphone disambiguator).
//! The `pinyin` crate is per-character, so a 多音字 whose reading depends on its
//! word — 银行 as *hang* not *xing*, 重要 as *zhong* not *chong* — takes the
//! commonest reading instead of the right one. [`POLYPHONES`] covers the
//! frequent cases; a real phrase dictionary is the fix.

use std::collections::HashMap;
use std::sync::OnceLock;

use jieba_rs::Jieba;
use pinyin::ToPinyin;

use crate::Phonemes;
use crate::normalize;

/// `pinyin -> "INITIAL FINAL"`, from upstream's `opencpop-strict.txt`.
///
/// Embedded rather than read at run time: it is 429 lines, it must match the
/// checkpoint, and a missing data file at synthesis time is a worse failure than
/// a slightly larger binary.
const OPENCPOP: &str = include_str!("../data/opencpop-strict.txt");

/// Word readings the per-character `pinyin` crate gets wrong, as tone-numbered
/// syllables. Far from complete — see the module note.
const POLYPHONES: [(&str, &[&str]); 12] = [
    ("银行", &["yin2", "hang2"]),
    ("重要", &["zhong4", "yao4"]),
    ("重复", &["chong2", "fu4"]),
    ("音乐", &["yin1", "yue4"]),
    ("快乐", &["kuai4", "le4"]),
    ("觉得", &["jue2", "de5"]),
    ("睡觉", &["shui4", "jiao4"]),
    ("行为", &["xing2", "wei2"]),
    ("的话", &["de5", "hua4"]),
    ("地方", &["di4", "fang5"]),
    ("为了", &["wei4", "le5"]),
    ("还是", &["hai2", "shi4"]),
];

fn opencpop() -> &'static HashMap<&'static str, (&'static str, &'static str)> {
    static MAP: OnceLock<HashMap<&'static str, (&'static str, &'static str)>> = OnceLock::new();
    MAP.get_or_init(|| {
        OPENCPOP
            .lines()
            .filter_map(|line| {
                let (syllable, phones) = line.split_once('\t')?;
                let phones = phones.trim();
                // Every entry is exactly "INITIAL FINAL"; the initial is empty
                // for syllables that have none, spelled as a leading space.
                let (initial, final_) = phones.split_once(' ')?;
                Some((syllable.trim(), (initial, final_)))
            })
            .collect()
    })
}

fn jieba() -> &'static Jieba {
    static JIEBA: OnceLock<Jieba> = OnceLock::new();
    JIEBA.get_or_init(Jieba::new)
}

fn polyphones() -> &'static HashMap<&'static str, &'static [&'static str]> {
    static MAP: OnceLock<HashMap<&'static str, &'static [&'static str]>> = OnceLock::new();
    MAP.get_or_init(|| POLYPHONES.iter().copied().collect())
}

/// Phonemize Mandarin text.
///
/// `word2ph` counts the phonemes each *character* produced — one for
/// punctuation, two for a syllable. The T2S model needs it to spread per-
/// character BERT features across phonemes, and upstream asserts it sums to the
/// phone count and matches the normalized text's length; both hold here too.
pub fn phonemize(text: &str) -> Phonemes {
    let normalized = normalize::chinese(text);
    let mut phones = Vec::new();
    let mut word2ph = Vec::new();

    for token in jieba().cut(&normalized, true) {
        let word = token.word;
        // Punctuation survives as itself and counts as one phone, so the
        // character-to-phone mapping stays aligned with the text.
        if word.chars().all(|c| !is_han(c)) {
            for c in word.chars() {
                let symbol = normalize::punctuation(c);
                if let Some(symbol) = symbol {
                    phones.push(symbol.to_string());
                    word2ph.push(1);
                }
            }
            continue;
        }

        let syllables = readings(word);
        let toned = sandhi(word, syllables);
        for syllable in &toned {
            let (initial, final_with_tone) = match split_syllable(syllable) {
                Some(parts) => parts,
                // Unreadable syllable: emit nothing rather than a wrong phoneme,
                // but still account for the character.
                None => {
                    phones.push(crate::symbols::UNKNOWN.to_string());
                    word2ph.push(1);
                    continue;
                }
            };
            phones.push(initial);
            phones.push(final_with_tone);
            word2ph.push(2);
        }
    }

    Phonemes {
        phones,
        word2ph: Some(word2ph),
        normalized,
    }
}

/// Tone-numbered pinyin for each character of `word`.
fn readings(word: &str) -> Vec<String> {
    if let Some(known) = polyphones().get(word) {
        return known.iter().map(|s| s.to_string()).collect();
    }
    word.chars()
        .map(|c| match c.to_pinyin() {
            // `with_tone_num_end` gives "hao3"; a neutral tone comes back
            // unmarked, and upstream's `neutral_tone_with_five` spells it 5.
            Some(p) => {
                let s = p.with_tone_num_end().to_string();
                if s.ends_with(['1', '2', '3', '4', '5']) {
                    s
                } else {
                    format!("{s}5")
                }
            }
            None => String::new(),
        })
        .collect()
}

/// Split a tone-numbered syllable into the two phonemes the model expects.
///
/// Transcribed from upstream, rewrites included: the toneless syllable is looked
/// up in the opencpop table, and the tone digit is appended to the final. The
/// rewrites exist because pinyin spelling and the table's keys disagree on a
/// handful of syllables.
fn split_syllable(syllable: &str) -> Option<(String, String)> {
    let (body, tone) = syllable.split_at(syllable.len().checked_sub(1)?);
    if !matches!(tone, "1" | "2" | "3" | "4" | "5") || body.is_empty() {
        return None;
    }

    // Whole-syllable rewrites for the ones written differently in isolation.
    let key = match body {
        "ing" => "ying",
        "i" => "yi",
        "in" => "yin",
        "u" => "wu",
        // `uei`/`iou`/`uen` are how pinyin spells them after an initial; the
        // table keys them by their standalone contraction.
        other => match other.len() {
            _ if other.ends_with("uei") => return rewrite(other, "uei", "ui", tone),
            _ if other.ends_with("iou") => return rewrite(other, "iou", "iu", tone),
            _ if other.ends_with("uen") => return rewrite(other, "uen", "un", tone),
            _ => other,
        },
    };

    if let Some(found) = lookup(key, tone) {
        return Some(found);
    }
    // A bare final starting with v/i/u is written yu/y/w when it stands alone.
    let alone = match key.chars().next()? {
        'v' => format!("yu{}", &key[1..]),
        'i' => format!("y{}", &key[1..]),
        'u' => format!("w{}", &key[1..]),
        _ => return None,
    };
    lookup(&alone, tone)
}

fn rewrite(body: &str, from: &str, to: &str, tone: &str) -> Option<(String, String)> {
    let key = format!("{}{to}", &body[..body.len() - from.len()]);
    lookup(&key, tone)
}

fn lookup(key: &str, tone: &str) -> Option<(String, String)> {
    let (initial, final_) = opencpop().get(key)?;
    Some((initial.to_string(), format!("{final_}{tone}")))
}

/// Mandarin tone sandhi, over one word.
///
/// The rules that change a tone rather than merely colour it: 不 before a fourth
/// tone, 一 between repeats, and a third tone before another third tone. Upstream
/// implements considerably more (轻声 by part of speech, 儿化); this covers the
/// cases a listener notices.
fn sandhi(word: &str, mut syllables: Vec<String>) -> Vec<String> {
    let chars: Vec<char> = word.chars().collect();

    for i in 0..syllables.len() {
        // 不 (bu4) becomes second tone before a fourth-tone syllable.
        if chars.get(i) == Some(&'不')
            && syllables.get(i + 1).is_some_and(|next| next.ends_with('4'))
        {
            set_tone(&mut syllables[i], '2');
        }
        // 一 (yi1) becomes neutral between two copies of the same character.
        if chars.get(i) == Some(&'一') && i > 0 && chars.get(i + 1) == chars.get(i - 1) {
            set_tone(&mut syllables[i], '5');
        }
    }

    // Third tone before third tone rises to second. Applied right to left so a
    // run of three (你很好) shifts only where it should.
    for i in (0..syllables.len().saturating_sub(1)).rev() {
        if syllables[i].ends_with('3') && syllables[i + 1].ends_with('3') {
            set_tone(&mut syllables[i], '2');
        }
    }
    syllables
}

fn set_tone(syllable: &mut String, tone: char) {
    if syllable.pop().is_some() {
        syllable.push(tone);
    }
}

fn is_han(c: char) -> bool {
    matches!(c as u32, 0x4E00..=0x9FFF | 0x3400..=0x4DBF | 0xF900..=0xFAFF)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn phones(text: &str) -> Vec<String> {
        phonemize(text).phones
    }

    #[test]
    fn a_syllable_becomes_an_initial_and_a_toned_final() {
        // 你好 -> ni3 hao3, with sandhi lifting the first to second tone.
        assert_eq!(phones("你好"), ["n", "i2", "h", "ao3"]);
    }

    #[test]
    fn word2ph_sums_to_the_phone_count_and_covers_every_character() {
        // Both are assertions upstream makes, and the T2S model relies on them to
        // spread per-character BERT features across phonemes — a mismatch
        // misaligns prosody against the text rather than failing.
        for text in ["你好世界", "我不要", "今天天气很好"] {
            let out = phonemize(text);
            let word2ph = out.word2ph.expect("chinese always reports word2ph");
            assert_eq!(
                word2ph.iter().sum::<usize>(),
                out.phones.len(),
                "{text}: word2ph must sum to the phone count"
            );
            assert_eq!(
                word2ph.len(),
                out.normalized.chars().count(),
                "{text}: one entry per character of the normalized text"
            );
        }
    }

    #[test]
    fn every_phoneme_is_in_the_models_vocabulary() {
        // An out-of-table phoneme silently becomes UNK downstream, which sounds
        // like a dropped syllable rather than an error.
        for text in ["你好世界", "这是一个测试", "我们明天去北京", "音乐很好听"]
        {
            for p in phones(text) {
                assert_ne!(
                    crate::symbols::id(&p),
                    crate::symbols::id(crate::symbols::UNKNOWN),
                    "{text}: `{p}` is not a known symbol"
                );
            }
        }
    }

    #[test]
    fn bu_lowers_before_a_fourth_tone() {
        // 不要 is bu2 yao4, not bu4 yao4 — the rule a learner meets first.
        let p = phones("不要");
        assert_eq!(p[1], "u2", "不 should have shifted to second tone: {p:?}");
    }

    #[test]
    fn a_known_polyphone_takes_its_word_reading() {
        // 银行 is *hang*, not *xing*. Per-character pinyin gets this wrong, which
        // is the whole reason the override table exists.
        let p = phones("银行");
        assert!(
            p.contains(&"h".to_string()),
            "银行 should use the hang reading: {p:?}"
        );
    }

    #[test]
    fn punctuation_survives_as_one_phone_each() {
        let out = phonemize("你好，世界！");
        assert!(out.phones.contains(&",".to_string()));
        assert!(out.phones.contains(&"!".to_string()));
    }
}
