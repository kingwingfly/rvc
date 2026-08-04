//! Splitting mixed text into runs of one language each.
//!
//! Required rather than optional for this toolkit: the corpus is code-switched
//! Chinese and English, and each language has its own grapheme-to-phoneme path.
//! Upstream ships a `LangSegmenter` built on a statistical language detector; the
//! split here is by **script**, which is what actually distinguishes zh from ja
//! from en in practice and needs no model.
//!
//! Latin runs go to English. Japanese is decided by the presence of kana,
//! because a Japanese sentence is nearly always mixed kana and kanji. Han with
//! **no** kana beside it is genuinely ambiguous — 東京 and 日本語 are valid in
//! either language — so the caller's `default` breaks the tie, and that means
//! Chinese for everyone who did not ask for Japanese.

/// A language this crate can phonemize.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Language {
    /// Mandarin Chinese.
    Zh,
    /// English.
    En,
    /// Japanese.
    Ja,
}

impl Language {
    /// The ISO code, matching what `stt` reports and what upstream calls it.
    pub fn code(self) -> &'static str {
        match self {
            Self::Zh => "zh",
            Self::En => "en",
            Self::Ja => "ja",
        }
    }
}

impl std::str::FromStr for Language {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "zh" | "cmn" | "chinese" | "mandarin" => Ok(Self::Zh),
            "en" | "eng" | "english" => Ok(Self::En),
            "ja" | "jpn" | "japanese" => Ok(Self::Ja),
            other => Err(format!(
                "unsupported language `{other}` (expected zh, en or ja)"
            )),
        }
    }
}

/// One stretch of text in a single language.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Run {
    pub language: Language,
    pub text: String,
}

/// Which script a character belongs to, as far as language choice cares.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Script {
    Han,
    Kana,
    Latin,
    /// Punctuation, digits, whitespace — joins whichever run it lands in.
    Neutral,
}

fn script(c: char) -> Script {
    match c as u32 {
        // CJK Unified Ideographs, plus Extension A and the compatibility block.
        0x4E00..=0x9FFF | 0x3400..=0x4DBF | 0xF900..=0xFAFF => Script::Han,
        // U+3005, the iteration mark (々). Upstream's `_japanese_characters`
        // includes it; without this it falls to `Neutral` and a word like 人々
        // splits around it.
        0x3005 => Script::Han,
        // Hiragana and katakana, and the halfwidth katakana forms.
        0x3040..=0x30FF | 0x31F0..=0x31FF | 0xFF66..=0xFF9D => Script::Kana,
        _ if c.is_ascii_alphabetic() => Script::Latin,
        // Latin-1 letters and the extended blocks (accented European text).
        0x00C0..=0x024F => Script::Latin,
        _ => Script::Neutral,
    }
}

/// Split `text` into runs, one language each.
///
/// `default` decides runs that carry no script information at all — a line of
/// bare punctuation, or digits on their own.
pub fn split(text: &str, default: Language) -> Vec<Run> {
    // Kana anywhere makes the Han in the same text Japanese: Japanese mixes the
    // two constantly, whereas Chinese has no kana at all.
    //
    // **With no kana to go on, `default` decides.** Han is genuinely ambiguous —
    // 東京, 日本語 and 人々 are all valid in either language — so guessing
    // Chinese regardless would phonemize a caller's explicitly Japanese line as
    // Mandarin, confidently and silently. That is the same failure the `Ja` arm
    // of `phonemize` refuses to make by falling back, so it must not be made
    // here either. Chinese remains the answer whenever the caller did not say
    // Japanese, which is every existing caller.
    let kana = text.chars().any(|c| script(c) == Script::Kana);
    let han_language = if kana || default == Language::Ja {
        Language::Ja
    } else {
        Language::Zh
    };

    let mut runs: Vec<Run> = Vec::new();
    for c in text.chars() {
        let language = match script(c) {
            Script::Han | Script::Kana => han_language,
            Script::Latin => Language::En,
            // Neutral characters extend the current run rather than starting
            // one, so "你好, world" splits in two places and not three.
            Script::Neutral => match runs.last() {
                Some(run) => run.language,
                None => default,
            },
        };
        match runs.last_mut() {
            Some(run) if run.language == language => run.text.push(c),
            _ => runs.push(Run {
                language,
                text: c.to_string(),
            }),
        }
    }

    // Trailing neutrals attached to a run are fine, but a run that is *only*
    // neutral carries no phonemes worth a language switch — fold it backwards.
    runs.retain(|r| !r.text.trim().is_empty());
    runs
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bare_han_follows_the_caller_when_they_asked_for_japanese() {
        // Han with no kana is ambiguous, so `default` breaks the tie. Getting
        // this wrong is silent: 東京 would come out as Mandarin `dong jing`
        // with a plausible phoneme sequence and no error anywhere.
        let ja = split("東京", Language::Ja);
        assert_eq!(ja.len(), 1);
        assert_eq!(ja[0].language, Language::Ja);

        // …and still Chinese for every caller who did not ask for Japanese,
        // which is the existing behaviour and the common case.
        let zh = split("東京", Language::Zh);
        assert_eq!(zh[0].language, Language::Zh);
        assert_eq!(split("東京", Language::En)[0].language, Language::Zh);
    }

    #[test]
    fn the_iteration_mark_stays_inside_its_word() {
        // U+3005 is Han, not neutral: upstream's `_japanese_characters`
        // includes it, and treating it as neutral splits 人々 in two.
        let runs = split("人々", Language::Ja);
        assert_eq!(runs.len(), 1, "人々 should be one run, got {runs:?}");
        assert_eq!(runs[0].text, "人々");
    }

    fn split_codes(text: &str) -> Vec<(&'static str, String)> {
        split(text, Language::Zh)
            .into_iter()
            .map(|r| (r.language.code(), r.text))
            .collect()
    }

    #[test]
    fn code_switched_text_splits_at_the_script_boundary() {
        assert_eq!(
            split_codes("你好world"),
            [("zh", "你好".into()), ("en", "world".into())]
        );
    }

    #[test]
    fn punctuation_stays_with_the_run_it_follows() {
        // A comma of its own is not a language change; splitting on it would give
        // the Chinese g2p a fragment with no characters in it.
        assert_eq!(
            split_codes("你好, world!"),
            [("zh", "你好, ".into()), ("en", "world!".into())]
        );
    }

    #[test]
    fn kana_makes_the_whole_text_japanese() {
        // 私 and 東京 are Han, and in isolation would be Chinese; the かな next to
        // them is what settles it.
        assert_eq!(
            split_codes("私は東京にいます"),
            [("ja", "私は東京にいます".into())]
        );
        assert_eq!(split_codes("東京"), [("zh", "東京".into())]);
    }

    #[test]
    fn text_with_no_script_falls_back_to_the_default() {
        assert_eq!(split("12345", Language::En)[0].language, Language::En);
        assert!(split("   ", Language::Zh).is_empty());
    }
}
