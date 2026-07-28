//! Text normalization shared by the language front-ends.
//!
//! Only the parts that change which phonemes come out: full-width punctuation
//! folded to the five marks the models know, and Arabic digits read aloud.
//! Upstream's `zh_normalization` handles a great deal more (dates, currency,
//! fractions, phone numbers); this covers plain prose and says so.

/// The punctuation the models have symbols for. Anything else is dropped rather
/// than passed through, because an unknown symbol becomes `UNK` — an audible
/// stumble instead of a pause.
pub fn punctuation(c: char) -> Option<&'static str> {
    Some(match c {
        '，' | ',' => ",",
        '。' | '.' => ".",
        '！' | '!' => "!",
        '？' | '?' => "?",
        '…' => "…",
        '、' | ';' | '；' | ':' | '：' => ",",
        '—' | '-' | '－' => "-",
        _ => return None,
    })
}

/// Normalize Mandarin text: fold punctuation, spell digits out.
pub fn chinese(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for c in text.chars() {
        match c {
            '0'..='9' => out.push(digit(c)),
            c if punctuation(c).is_some() => out.push(fold(c)),
            // Whitespace carries no phoneme in Chinese and would otherwise
            // become a word boundary jieba then has to guess around.
            c if c.is_whitespace() => {}
            c => out.push(c),
        }
    }
    out
}

/// A digit as its Chinese character, read one at a time.
///
/// Digit-by-digit rather than as a number: "2024" becomes 二零二四, which is how
/// a year is read, and the alternative needs the full number grammar to beat it.
fn digit(c: char) -> char {
    match c {
        '0' => '零',
        '1' => '一',
        '2' => '二',
        '3' => '三',
        '4' => '四',
        '5' => '五',
        '6' => '六',
        '7' => '七',
        '8' => '八',
        _ => '九',
    }
}

/// Collapse a punctuation mark onto its ASCII representative, keeping one
/// character so per-character counts stay aligned with the text.
fn fold(c: char) -> char {
    match punctuation(c) {
        Some(s) => s.chars().next().unwrap_or(c),
        None => c,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_width_punctuation_folds_to_the_known_marks() {
        assert_eq!(chinese("你好，世界！"), "你好,世界!");
    }

    #[test]
    fn digits_are_read_one_at_a_time() {
        assert_eq!(chinese("2024年"), "二零二四年");
    }

    #[test]
    fn normalization_keeps_one_character_per_character() {
        // Chinese g2p reports a phone count per character, and the caller checks
        // it against the normalized text's length — so folding must not change
        // how many characters there are.
        for text in ["你好，世界！", "2024年", "他说：不要"] {
            let out = chinese(text);
            assert!(out.chars().all(|c| !c.is_whitespace()));
            assert!(!out.is_empty());
        }
    }
}
