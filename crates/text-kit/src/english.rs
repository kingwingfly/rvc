//! English grapheme-to-phoneme.
//!
//! Reproduces the deterministic half of `GPT_SoVITS/text/english.py`: look the
//! word up in CMUdict, and when that misses, walk upstream's cascade of
//! fallbacks — a quote the tokenizer glued on, a hyphenated compound, a short
//! stranger read as letters, a possessive `'s` voiced by what precedes it, a
//! compound split into dictionary words.
//!
//! The phonemes are ARPAbet with their stress digits intact (`HH AH0 L OW1`),
//! and they address [`crate::symbols::SYMBOLS`] **by identity**: all 71 of them
//! are already entries 6 to 94 of the table, spelled exactly as CMUdict spells
//! them. Lowercasing or stripping stress here would not fail — it would index
//! the wrong embeddings and produce confident nonsense, which is why nothing in
//! this file touches their case.
//!
//! ## The one thing not reproduced
//!
//! Upstream's cascade ends in `g2p_en`'s neural LSTM, a small seq2seq trained to
//! spell out pronunciations for words no dictionary has. There is no Rust
//! equivalent and porting one is a model, not a front-end, so **a word that
//! survives the whole cascade is spelled letter by letter** rather than guessed
//! at. "gpu" comes out as *gee pee you*, which is right; an invented proper noun
//! comes out spelled, which is wrong but wrong in an obvious, recoverable way —
//! the alternative is a plausible-sounding invention nobody can spot. The 126k
//! entries embedded here cover ordinary prose; what falls through is mostly
//! names, and names are worth a dictionary of their own before they are worth a
//! network.
//!
//! `word2ph` is `None` for English, as upstream's `cleaner.py` returns it: there
//! is no per-character alignment to report once normalization has rewritten
//! "$3.50" into five words. Consumers feed zeros to the prosody encoder instead,
//! so English is intelligible but flatter than Chinese.

use std::collections::HashMap;
use std::sync::OnceLock;

use crate::Phonemes;
use crate::normalize;

/// CMUdict 0.7b, compacted: `word<TAB>PH PH PH`, one line per word, first
/// pronunciation only.
///
/// Embedded rather than read at run time for `chinese.rs`'s reason — a missing
/// data file at synthesis time is a worse failure than a larger binary — and
/// compacted from upstream's 3.7 MB `cmudict.rep` plus the entries
/// `cmudict-fast.rep` adds: the `;;;` header, the `(2)` variant spellings that
/// nothing looks up, and the headwords that are not words all go. Upstream's
/// `engdict-hot.rep` overrides are folded in, and the six abbreviations it
/// deletes for reading wrong (`ae ai ar ios hud os`) are absent.
const CMUDICT: &str = include_str!("../data/cmudict.txt");

fn cmudict() -> &'static HashMap<&'static str, &'static str> {
    static MAP: OnceLock<HashMap<&'static str, &'static str>> = OnceLock::new();
    MAP.get_or_init(|| {
        CMUDICT
            .lines()
            .filter_map(|line| line.split_once('\t'))
            .collect()
    })
}

/// Phonemize English text.
///
/// `word2ph` is always `None` — see the module note.
pub fn phonemize(text: &str) -> Phonemes {
    let normalized = normalize::english(text);
    let mut phones: Vec<String> = Vec::new();

    for token in tokens(&normalized) {
        let spoken = match &token {
            Token::Mark(mark) => vec![*mark],
            Token::Word(word) => match word.chars().count() {
                // Upstream's one correction to the letter readings: a lone
                // capital "A" is the letter (EY1), a lone lowercase "a" is the
                // article (AH0). Only the original casing can tell them apart.
                1 if word == "A" => vec!["EY1"],
                1 => letter(word.chars().next().expect("one character")),
                _ => pronounce(&word.to_ascii_lowercase()),
            },
        };
        phones.extend(spoken.into_iter().map(str::to_string));
    }

    // Upstream's `cleaner.py` prepends a comma to a very short line: `s1`
    // continues a sequence, and two or three phonemes give it too little to
    // continue from, so it answers by copying the reference clip instead. Text
    // that produced nothing at all is left empty rather than turned into a lone
    // comma — callers check for empty to catch a transcript in the wrong
    // language, and a comma would hide it.
    if !phones.is_empty() && phones.len() < 4 {
        phones.insert(0, ",".to_string());
    }

    Phonemes {
        phones,
        word2ph: None,
        normalized,
    }
}

/// One word, or one mark that survives as a phoneme of its own.
enum Token {
    Word(String),
    Mark(&'static str),
}

/// Split normalized English into words and marks.
///
/// Upstream tokenizes with NLTK's `TweetTokenizer`; the input here has already
/// been reduced to letters, apostrophes, spaces and four marks, so the only
/// question left is which apostrophes and hyphens belong to a word.
fn tokens(text: &str) -> Vec<Token> {
    let chars: Vec<char> = text.chars().collect();
    let mut out = Vec::new();
    let mut word = String::new();

    for (i, &c) in chars.iter().enumerate() {
        if c.is_ascii_alphabetic() {
            word.push(c);
            continue;
        }
        // An apostrophe or hyphen joins the word when it sits between letters
        // ("don't", "well-known"); an apostrophe may also open one ("'bout").
        // Anywhere else — a closing quote, a dash — it is a mark of its own.
        let joins = (c == '\'' || c == '-')
            && chars.get(i + 1).is_some_and(char::is_ascii_alphabetic)
            && (!word.is_empty() || c == '\'');
        if joins {
            word.push(c);
            continue;
        }
        if !word.is_empty() {
            out.push(Token::Word(std::mem::take(&mut word)));
        }
        match c {
            // A quote left standing alone becomes a hyphen, which is upstream's
            // `replace_phs` rep_map: the model has a symbol for one and not the
            // other, and both read as a short break.
            '\'' => out.push(Token::Mark("-")),
            c => {
                if let Some(mark) = normalize::punctuation(c) {
                    out.push(Token::Mark(mark));
                }
            }
        }
    }
    if !word.is_empty() {
        out.push(Token::Word(word));
    }
    out
}

/// Pronounce one lowercase word of two letters or more.
///
/// Upstream's `qryword`, in its order, minus the name dictionary (a 760 kB
/// pickle of surnames) and the neural last resort.
fn pronounce(word: &str) -> Vec<&'static str> {
    // The length guard is upstream's and it matters: recursion can hand this a
    // single letter, and the dictionary's entry for "a" is the *article* (AH0).
    // A letter reached by spelling something out is the letter (EY1).
    if word.chars().count() > 1 {
        if let Some(entry) = cmudict().get(word) {
            return entry.split(' ').collect();
        }
    }
    // The tokenizer keeps a quote that opened a word; the dictionary does not.
    if let Some(bare) = word.strip_prefix('\'') {
        if !bare.is_empty() {
            return pronounce(bare);
        }
    }
    // A hyphenated compound is two words with a break between them. Spelling it
    // out would be wrong in a way spelling an acronym is not.
    if word.contains('-') {
        return word
            .split('-')
            .filter(|part| !part.is_empty())
            .flat_map(pronounce)
            .collect();
    }
    // Short strangers are read as letters before anything else is tried,
    // because a two- or three-letter unknown is nearly always an initialism.
    if word.chars().count() <= 3 {
        return spell(word);
    }
    if let Some(stem) = word.strip_suffix("'s") {
        if !stem.is_empty() {
            return possessive(pronounce(stem));
        }
    }
    if let Some(parts) = compound(word) {
        return parts.into_iter().flat_map(pronounce).collect();
    }
    spell(word)
}

/// `'s` after a stem, voiced by what the stem ends in.
///
/// The ordinary English rule, and upstream states it the same way: voiceless
/// consonants take /s/, sibilants take an extra vowel because /s/ after /s/ is
/// inaudible, and everything else — voiced consonants and all vowels — takes
/// /z/.
fn possessive(mut phones: Vec<&'static str>) -> Vec<&'static str> {
    match phones.last() {
        Some(&"P" | &"T" | &"K" | &"F" | &"TH" | &"HH") => phones.push("S"),
        Some(&"S" | &"Z" | &"SH" | &"ZH" | &"CH" | &"JH") => phones.extend(["AH0", "Z"]),
        Some(_) => phones.push("Z"),
        None => {}
    }
    phones
}

/// Read a word out letter by letter.
fn spell(word: &str) -> Vec<&'static str> {
    word.chars()
        .flat_map(|c| match c {
            // A letter A read as a letter is EY1; only the article is AH0, and
            // nothing spelled out is an article.
            'a' => vec!["EY1"],
            '\'' => vec!["-"],
            c if c.is_ascii_alphabetic() => letter(c),
            _ => Vec::new(),
        })
        .collect()
}

/// One letter's own name, from the dictionary — "w" is four phonemes.
fn letter(c: char) -> Vec<&'static str> {
    let mut buffer = [0u8; 4];
    let key = c.to_ascii_lowercase().encode_utf8(&mut buffer);
    cmudict()
        .get(key)
        .map_or_else(Vec::new, |entry| entry.split(' ').collect())
}

/// Split an unknown word into dictionary words, fewest parts first.
///
/// Upstream reaches for `wordsegment`, a Viterbi search over Google's unigram
/// and bigram counts; this searches the dictionary already embedded here, which
/// catches the compounds that matter ("crypto" + "currency") without another
/// 30 MB of data. Parts shorter than three letters are refused: with them almost
/// any string splits into something, and a wrong split is read out as
/// confidently as a right one.
fn compound(word: &str) -> Option<Vec<&str>> {
    const MIN_PART: usize = 3;
    if !word.chars().all(|c| c.is_ascii_alphabetic()) || word.len() < 2 * MIN_PART {
        return None;
    }

    // `best[i]` scores the best split of `word[..i]`: fewest parts wins, and a
    // tie goes to the split whose shortest part is longest, which prefers
    // "crypto currency" over an even division into three-letter noise.
    let n = word.len();
    let mut best: Vec<Option<(usize, std::cmp::Reverse<usize>)>> = vec![None; n + 1];
    let mut from = vec![0usize; n + 1];
    best[0] = Some((0, std::cmp::Reverse(usize::MAX)));

    for end in MIN_PART..=n {
        for start in 0..=end - MIN_PART {
            let Some((parts, std::cmp::Reverse(shortest))) = best[start] else {
                continue;
            };
            if !cmudict().contains_key(&word[start..end]) {
                continue;
            }
            let candidate = (parts + 1, std::cmp::Reverse(shortest.min(end - start)));
            if best[end].is_none_or(|current| candidate < current) {
                best[end] = Some(candidate);
                from[end] = start;
            }
        }
    }

    let (parts, _) = best[n]?;
    if parts < 2 {
        return None;
    }
    let mut split = Vec::with_capacity(parts);
    let mut at = n;
    while at > 0 {
        split.push(&word[from[at]..at]);
        at = from[at];
    }
    split.reverse();
    Some(split)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn phones(text: &str) -> Vec<String> {
        phonemize(text).phones
    }

    #[test]
    fn a_dictionary_word_keeps_its_stress_digits() {
        // The stress digit is part of the symbol: "AH0" and "AH1" are different
        // entries of the phoneme table, so dropping it addresses the wrong
        // embedding rather than merely losing emphasis.
        assert_eq!(phones("hello"), ["HH", "AH0", "L", "OW1"]);
        assert_eq!(phones("world"), ["W", "ER1", "L", "D"]);
    }

    #[test]
    fn every_phoneme_is_in_the_models_vocabulary() {
        // An out-of-table phoneme silently becomes UNK downstream, which sounds
        // like a dropped syllable rather than an error.
        for text in [
            "The quick brown fox jumps over the lazy dog.",
            "I used the AI tool to draw a picture, e.g. a cat.",
            "In this; paper, we propose 1 DSPGAN, a GAN-based universal vocoder.",
            "It costs $6.24 and weighs 3/4 of a kilogram, i.e. not much!",
            "She'd read the cafe's menu in 1984 — 22nd time this month?",
            "supercalifragilisticexpialidocious",
        ] {
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
    fn a_possessive_takes_its_voicing_from_the_stem() {
        // The dictionary has no entry for most possessives, so this is the
        // cascade doing the work. Getting it backwards ("cat-z") is the kind of
        // error a listener hears immediately and a coverage check never sees.
        assert_eq!(possessive(vec!["K", "AE1", "T"]).last(), Some(&"S"));
        assert_eq!(possessive(vec!["D", "AO1", "G"]).last(), Some(&"Z"));
        assert_eq!(
            possessive(vec!["R", "OW1", "Z"]),
            ["R", "OW1", "Z", "AH0", "Z"]
        );
        // End to end, through a stem whose possessive the dictionary lacks.
        let spoken = phones("openai's");
        assert_eq!(spoken.last().map(String::as_str), Some("Z"));
    }

    #[test]
    fn an_unknown_word_is_spelled_rather_than_guessed() {
        // No neural fallback exists in Rust, so the honest failure is to read
        // the letters. A guess would sound fluent and be unverifiable.
        assert_eq!(phones("gpu"), ["JH", "IY1", "P", "IY1", "Y", "UW1"]);
        // Every letter of a longer stranger still resolves to real phonemes.
        assert!(!phones("zzyzxq").is_empty());
    }

    #[test]
    fn a_compound_word_splits_into_dictionary_words() {
        // Reading "crypto currency" letter by letter would be sixteen letter
        // names where two words belong.
        assert_eq!(
            compound("cryptocurrency"),
            Some(vec!["crypto", "currency"]),
            "an unknown compound should split at the dictionary boundary"
        );
        assert_eq!(compound("hello"), None, "a known word is not a compound");
    }

    #[test]
    fn numbers_and_ordinals_are_read_before_phonemizing() {
        // The phoneme table has no digits at all; an unexpanded "3" reaches the
        // model as UNK.
        assert_eq!(phonemize("3").normalized, "three");
        assert_eq!(phonemize("22nd").normalized, "twenty-second");
        assert_eq!(phonemize("1984").normalized, "nineteen eighty-four");
    }

    #[test]
    fn an_acronym_is_spelled_out_letter_by_letter() {
        // "USA" is not a word. Normalization splits the capitals and the
        // single-letter path reads each one.
        assert_eq!(
            phones("USA"),
            ["Y", "UW1", "EH1", "S", "EY1"],
            "USA should read as its three letter names"
        );
    }

    #[test]
    fn english_reports_no_per_character_alignment() {
        // Upstream returns None here, and the consumers feed zeros to the
        // prosody encoder as a result. Reporting counts that do not line up
        // with the text would misalign prosody rather than fail.
        assert_eq!(phonemize("Hello, world!").word2ph, None);
    }

    #[test]
    fn a_short_line_gets_a_leading_comma_but_an_empty_one_stays_empty() {
        // Upstream's guard: too few phonemes and `s1` finds nothing to continue
        // from, so it copies the reference clip instead. Text that produced
        // nothing is left alone so callers can still detect it.
        assert_eq!(phones("hi"), [",", "HH", "AY1"]);
        assert!(phones("").is_empty());
        assert!(phones("你好").is_empty());
    }

    #[test]
    fn punctuation_survives_as_one_phone_each() {
        let spoken = phones("Wait, what?");
        assert!(spoken.contains(&",".to_string()));
        assert!(spoken.contains(&"?".to_string()));
    }
}
