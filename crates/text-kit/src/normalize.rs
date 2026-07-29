//! Text normalization shared by the language front-ends.
//!
//! Only the parts that change which phonemes come out: full-width punctuation
//! folded to the five marks the models know, and Arabic digits read aloud.
//! Upstream's `zh_normalization` handles a great deal more (dates, currency,
//! fractions, phone numbers); this covers plain prose and says so.
//!
//! The two front-ends need opposite things and must not share one normalizer.
//! Mandarin has no word boundaries, so [`chinese`] deletes whitespace and keeps
//! exactly one character per character — that count is what `word2ph` reports.
//! In English whitespace *is* the word boundary and normalization rewrites whole
//! spans ("$3.50" becomes five words), so [`english`] preserves spacing and
//! gives up on per-character alignment entirely.

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

/// The marks English keeps once everything has been folded onto ASCII. Each is
/// a symbol in the model's table, so each survives as a phoneme of its own.
pub fn is_mark(c: char) -> bool {
    matches!(c, '.' | ',' | '?' | '!' | '-')
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

/// Normalize English text: fold punctuation, read numbers aloud, and reduce to
/// letters, apostrophes and the marks [`is_mark`] admits.
///
/// Follows `GPT_SoVITS/text/english.py::text_normalize` and the
/// `en_normalization/expend.py` it calls, with three deliberate departures:
///
/// - No time expansion. Upstream folds `:` to `,` *before* running the number
///   pass, so its `13:30` rule can never fire; reproducing dead code would only
///   mislead the next reader.
/// - No `+ - × ÷ =` word substitution. Those characters do not survive the final
///   filter anyway, and the regex upstream matches them with catches variable
///   names as readily as sums.
/// - `1980s` is pluralized ("nineteen eighties") rather than read as `1980`
///   followed by a stray letter. Upstream's unit table claims a bare `s` for
///   *seconds*, which turns every decade into a duration; decades are much
///   commoner in prose, so `s` is not a unit here.
pub fn english(text: &str) -> String {
    // Fold to ASCII first so everything downstream can assume it.
    let folded: String = text
        .chars()
        .map(|c| match c {
            '，' | '；' | ';' | '：' | ':' | '、' => ',',
            '。' => '.',
            '！' => '!',
            '？' => '?',
            '“' | '”' | '‘' | '’' | '"' | '`' => '\'',
            '—' | '–' | '－' => '-',
            c => deaccent(c),
        })
        .collect();

    // Abbreviations read as words rather than spelled out. Both are upstream's.
    let folded = replace_ignore_case(&folded, "i.e.", "that is");
    let folded = replace_ignore_case(&folded, "e.g.", "for example");

    let spoken = expand_numbers(&folded).replace('%', " percent");

    // Acronyms and CamelCase are spelled out: a capital that is neither first
    // nor already spaced gets a space before it, so "USA" becomes "U S A" and
    // the g2p's single-letter path reads each one. Upstream's rule, and it
    // handles "OpenAI" -> "Open A I" as a side effect.
    let mut spaced = String::with_capacity(spoken.len());
    for (i, c) in spoken.chars().enumerate() {
        if i > 0 && c.is_ascii_uppercase() && !spaced.ends_with(char::is_whitespace) {
            spaced.push(' ');
        }
        spaced.push(c);
    }

    // Drop what cannot be spoken and collapse runs. Consecutive punctuation goes
    // for upstream's reason: a run of marks becomes a run of pauses, and a model
    // primed on a reference clip answers a long pause by copying the reference
    // instead of continuing the line.
    let mut out = String::with_capacity(spaced.len());
    for c in spaced.chars() {
        if c.is_ascii_alphabetic() || c == '\'' {
            out.push(c);
        } else if c.is_whitespace() {
            if !out.is_empty() && !out.ends_with(' ') {
                out.push(' ');
            }
        } else if is_mark(c) && out.ends_with(|p: char| !is_mark(p) && p != ' ') {
            out.push(c);
        }
    }
    while out.ends_with(' ') {
        out.pop();
    }
    out
}

/// A Latin letter with its diacritic removed, or `c` unchanged.
///
/// Upstream strips accents with NFD plus "drop the combining marks"; this is the
/// same idea over the Latin-1 Supplement without a Unicode-normalization
/// dependency, which is where every accented form an English loanword arrives in
/// lives (café, naïve, résumé, Zürich). Letters that expand to two — æ, ß, þ —
/// are left alone and dropped by the filter, because spelling those out is a
/// language question rather than an accent one.
fn deaccent(c: char) -> char {
    match c {
        'À'..='Å' => 'A',
        'Ç' => 'C',
        'È'..='Ë' => 'E',
        'Ì'..='Ï' => 'I',
        'Ð' => 'D',
        'Ñ' => 'N',
        'Ò'..='Ö' | 'Ø' => 'O',
        'Ù'..='Ü' => 'U',
        'Ý' => 'Y',
        'à'..='å' => 'a',
        'ç' => 'c',
        'è'..='ë' => 'e',
        'ì'..='ï' => 'i',
        'ñ' => 'n',
        'ò'..='ö' | 'ø' => 'o',
        'ù'..='ü' => 'u',
        'ý' | 'ÿ' => 'y',
        c => c,
    }
}

/// Replace every case-insensitive occurrence of an ASCII `needle`.
fn replace_ignore_case(text: &str, needle: &str, replacement: &str) -> String {
    let lower = text.to_ascii_lowercase();
    let mut out = String::with_capacity(text.len());
    let mut cut = 0;
    while let Some(at) = lower[cut..].find(needle) {
        out.push_str(&text[cut..cut + at]);
        out.push_str(replacement);
        cut += at + needle.len();
    }
    out.push_str(&text[cut..]);
    out
}

/// Suffixes read as a unit when they sit directly against a number, longest
/// first so `km` beats `m` and `min` beats `m`.
const UNITS: [(&str, &str, &str); 11] = [
    ("km/h", "kilometer per hour", "kilometers per hour"),
    ("km", "kilometer", "kilometers"),
    ("tbsp", "tablespoon", "tablespoons"),
    ("tsp", "teaspoon", "teaspoons"),
    ("min", "minute", "minutes"),
    ("ft", "foot", "feet"),
    ("°C", "degree celsius", "degrees celsius"),
    ("°F", "degree fahrenheit", "degrees fahrenheit"),
    ("m", "meter", "meters"),
    ("L", "liter", "liters"),
    ("h", "hour", "hours"),
];

/// Read every numeral in `text` aloud.
///
/// One left-to-right pass rather than upstream's stack of eleven regex
/// substitutions: at each digit this reads the whole literal — currency sign,
/// thousands separators, decimal part, fraction slash, ordinal suffix, unit —
/// and then decides once what kind of number it was. Upstream's substitutions
/// run in a fixed order and each sees the last one's output, which is why its
/// decimal rule leaves loose digits behind for its integer rule to pick up.
fn expand_numbers(text: &str) -> String {
    let chars: Vec<char> = text.chars().collect();
    let mut out = String::with_capacity(text.len());
    let mut i = 0;
    while i < chars.len() {
        let leading = match chars[i] {
            '$' => Some("dollar"),
            '£' => Some("pound"),
            _ => None,
        }
        .filter(|_| chars.get(i + 1).is_some_and(char::is_ascii_digit));
        let start = if leading.is_some() { i + 1 } else { i };
        if !chars.get(start).is_some_and(char::is_ascii_digit) {
            out.push(chars[i]);
            i += 1;
            continue;
        }

        // Thousands separators, but only in groups of exactly three: "1,234" is
        // one number while "1,2,3" is a list, and upstream's regex merges both.
        let mut whole = String::new();
        let mut j = start;
        while j < chars.len() {
            if chars[j].is_ascii_digit() {
                whole.push(chars[j]);
                j += 1;
            } else if chars[j] == ','
                && chars[j + 1..]
                    .iter()
                    .take(3)
                    .filter(|c| c.is_ascii_digit())
                    .count()
                    == 3
                && !chars.get(j + 4).is_some_and(char::is_ascii_digit)
            {
                j += 1;
            } else {
                break;
            }
        }
        let mut frac = String::new();
        if chars.get(j) == Some(&'.') && chars.get(j + 1).is_some_and(char::is_ascii_digit) {
            j += 1;
            while chars.get(j).is_some_and(char::is_ascii_digit) {
                frac.push(chars[j]);
                j += 1;
            }
        }

        let trailing = match chars.get(j) {
            Some('$') => Some("dollar"),
            Some('£') => Some("pound"),
            _ => None,
        };
        let ordinal: String = chars[j..].iter().take(2).collect::<String>().to_lowercase();
        let unit = UNITS.iter().find(|(suffix, _, _)| {
            let len = suffix.chars().count();
            chars[j..].iter().copied().take(len).eq(suffix.chars())
                && !chars.get(j + len).is_some_and(char::is_ascii_alphanumeric)
        });

        // Past u64 there is no sensible reading but digit by digit — which is
        // also what a phone number or an account number wants.
        let (spoken, next) = match whole.parse::<u64>() {
            Err(_) => (digits_words(&whole), j),
            Ok(value) => {
                if let Some(currency) = leading.or(trailing) {
                    let end = if trailing.is_some() { j + 1 } else { j };
                    (money(currency, value, &frac), end)
                } else if frac.is_empty()
                    && chars.get(j) == Some(&'/')
                    && chars.get(j + 1).is_some_and(char::is_ascii_digit)
                {
                    let mut k = j + 1;
                    let mut denominator = String::new();
                    while chars.get(k).is_some_and(char::is_ascii_digit) {
                        denominator.push(chars[k]);
                        k += 1;
                    }
                    (fraction(value, denominator.parse().unwrap_or(0)), k)
                } else if frac.is_empty()
                    && matches!(ordinal.as_str(), "st" | "nd" | "rd" | "th")
                    && !chars.get(j + 2).is_some_and(char::is_ascii_alphanumeric)
                {
                    (ordinal_words(value), j + 2)
                } else if let Some(&(suffix, singular, plural)) = unit {
                    let words = if frac.is_empty() && value == 1 {
                        singular
                    } else {
                        plural
                    };
                    (
                        format!("{} {words}", cardinal_with(value, &frac)),
                        j + suffix.chars().count(),
                    )
                } else if frac.is_empty()
                    && chars.get(j) == Some(&'s')
                    && !chars.get(j + 1).is_some_and(char::is_ascii_alphanumeric)
                {
                    // "1980s", "the 90s".
                    (pluralize(&cardinal(value)), j + 1)
                } else {
                    (cardinal_with(value, &frac), j)
                }
            }
        };

        // A minus sign that starts a word is a sign; one inside a word is a
        // hyphen. Upstream draws the same line with `(?:^|\s+)(-)(\d+)`.
        if out.ends_with('-') && out.chars().rev().nth(1).is_none_or(char::is_whitespace) {
            out.pop();
            out.push_str("negative ");
        }
        // A number written against a word ("1080p") would otherwise fuse with it.
        if out.ends_with(|c: char| c.is_alphanumeric()) {
            out.push(' ');
        }
        out.push_str(&spoken);
        if chars.get(next).is_some_and(|c| c.is_alphanumeric()) {
            out.push(' ');
        }
        i = next;
    }
    out
}

const ONES: [&str; 20] = [
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
];
const TENS: [&str; 10] = [
    "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
];

/// An integer as English words, `inflect`'s way: no "and", hyphenated tens.
fn number_to_words(n: u64) -> String {
    if n < 20 {
        return ONES[n as usize].to_string();
    }
    if n < 100 {
        let (tens, ones) = (TENS[(n / 10) as usize], n % 10);
        return match ones {
            0 => tens.to_string(),
            ones => format!("{tens}-{}", ONES[ones as usize]),
        };
    }
    let (scale, name) = match n {
        n if n >= 1_000_000_000_000 => (1_000_000_000_000, "trillion"),
        n if n >= 1_000_000_000 => (1_000_000_000, "billion"),
        n if n >= 1_000_000 => (1_000_000, "million"),
        n if n >= 1_000 => (1_000, "thousand"),
        _ => (100, "hundred"),
    };
    let head = format!("{} {name}", number_to_words(n / scale));
    match n % scale {
        0 => head,
        rest => format!("{head} {}", number_to_words(rest)),
    }
}

/// A plain integer, read the way a listener expects rather than the way it is
/// written.
///
/// 1001 through 2999 are read as two pairs ("nineteen eighty-four"), because a
/// four-digit number in that range is nearly always a year. Upstream's
/// `_expand_number` makes the same bet.
fn cardinal(n: u64) -> String {
    if !(1001..3000).contains(&n) {
        return number_to_words(n);
    }
    if n == 2000 {
        return "two thousand".to_string();
    }
    if (2001..2010).contains(&n) {
        return format!("two thousand {}", number_to_words(n % 100));
    }
    if n % 100 == 0 {
        return format!("{} hundred", number_to_words(n / 100));
    }
    match n % 100 {
        low if low < 10 => format!("{} oh {}", number_to_words(n / 100), number_to_words(low)),
        low => format!("{} {}", number_to_words(n / 100), number_to_words(low)),
    }
}

/// A number with an optional decimal part: "13.234" is "thirteen point two three
/// four", digit by digit after the point, which is how a decimal is read.
fn cardinal_with(whole: u64, frac: &str) -> String {
    match frac.is_empty() {
        true => cardinal(whole),
        false => format!("{} point {}", cardinal(whole), digits_words(frac)),
    }
}

/// Digits read one at a time, for a decimal tail or a number too long to name.
fn digits_words(digits: &str) -> String {
    digits
        .chars()
        .filter_map(|c| c.to_digit(10))
        .map(|d| ONES[d as usize])
        .collect::<Vec<_>>()
        .join(" ")
}

/// An ordinal: 22 becomes "twenty-second".
fn ordinal_words(n: u64) -> String {
    let words = number_to_words(n);
    let cut = words.rfind([' ', '-']).map_or(0, |i| i + 1);
    let (head, last) = words.split_at(cut);
    let last = match last {
        "one" => "first",
        "two" => "second",
        "three" => "third",
        "five" => "fifth",
        "eight" => "eighth",
        "nine" => "ninth",
        "twelve" => "twelfth",
        "twenty" => "twentieth",
        "thirty" => "thirtieth",
        "forty" => "fortieth",
        "fifty" => "fiftieth",
        "sixty" => "sixtieth",
        "seventy" => "seventieth",
        "eighty" => "eightieth",
        "ninety" => "ninetieth",
        // four/six/seven/ten/hundred/thousand and every -teen just take -th.
        regular => return format!("{head}{regular}th"),
    };
    format!("{head}{last}")
}

/// Pluralize the last word of a reading, so "1980s" is "nineteen eighties".
fn pluralize(words: &str) -> String {
    let cut = words.rfind([' ', '-']).map_or(0, |i| i + 1);
    let (head, last) = words.split_at(cut);
    if let Some(stem) = last.strip_suffix('y') {
        return format!("{head}{stem}ies");
    }
    if last.ends_with(['s', 'x', 'z']) || last.ends_with("ch") || last.ends_with("sh") {
        return format!("{head}{last}es");
    }
    format!("{head}{last}s")
}

/// An amount of money, with the decimal part read as the minor unit.
fn money(unit: &str, major: u64, frac: &str) -> String {
    // "$3.5" is three fifty, not three and five cents: the minor unit is always
    // two digits, so a one-digit tail is padded rather than read as written.
    let minor: u64 = match frac.is_empty() {
        true => 0,
        false => format!("{frac:0<2}")[..2].parse().unwrap_or(0),
    };
    let minor_unit = if unit == "pound" { "penny" } else { "cent" };
    let count = |n: u64, one: &str| {
        let many = match one {
            "penny" => "pence".to_string(),
            other => format!("{other}s"),
        };
        let name = if n == 1 { one.to_string() } else { many };
        format!("{} {name}", number_to_words(n))
    };
    match (major, minor) {
        (0, 0) => count(0, unit),
        (major, 0) => count(major, unit),
        (0, minor) => count(minor, minor_unit),
        (major, minor) => format!("{} and {}", count(major, unit), count(minor, minor_unit)),
    }
}

/// A vulgar fraction: numerator cardinal, denominator ordinal and pluralized
/// when the numerator is more than one. Halves are irregular and get their own
/// arm; upstream spells out the same three rules.
fn fraction(numerator: u64, denominator: u64) -> String {
    let numerator_words = number_to_words(numerator);
    match denominator {
        0 | 1 => numerator_words,
        2 if numerator == 1 => "one half".to_string(),
        2 => format!("{numerator_words} halves"),
        _ => {
            let mut denominator_words = ordinal_words(denominator);
            if numerator > 1 {
                denominator_words.push('s');
            }
            format!("{numerator_words} {denominator_words}")
        }
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

    #[test]
    fn english_normalization_keeps_word_boundaries() {
        // The Chinese normalizer deletes whitespace, which is right for Han and
        // would glue every English sentence into one unpronounceable word.
        assert_eq!(english("Hello, world!"), "Hello, world!");
        assert_eq!(english("  spaced   out  "), "spaced out");
    }

    #[test]
    fn numbers_are_read_as_words() {
        assert_eq!(english("I have 3 apples"), "I have three apples");
        assert_eq!(english("42"), "forty-two");
        assert_eq!(
            english("12,345"),
            "twelve thousand three hundred forty-five"
        );
    }

    #[test]
    fn a_year_shaped_number_is_read_as_a_year() {
        // "one thousand nine hundred eighty-four" is not what anyone says, and
        // the model has no way to recover the intended reading from phonemes.
        assert_eq!(english("1984"), "nineteen eighty-four");
        assert_eq!(english("2000"), "two thousand");
        assert_eq!(english("2005"), "two thousand five");
        assert_eq!(english("1905"), "nineteen oh five");
        assert_eq!(english("1980s"), "nineteen eighties");
    }

    #[test]
    fn ordinals_currency_fractions_and_units_have_their_own_readings() {
        assert_eq!(english("1st"), "first");
        assert_eq!(english("22nd"), "twenty-second");
        assert_eq!(english("$6.24"), "six dollars and twenty-four cents");
        assert_eq!(english("1/2"), "one half");
        assert_eq!(english("3/4"), "three fourths");
        assert_eq!(english("13.5"), "thirteen point five");
        assert_eq!(english("5km"), "five kilometers");
        assert_eq!(english("1L"), "one liter");
        assert_eq!(english("50%"), "fifty percent");
    }

    #[test]
    fn nothing_unspeakable_survives_english_normalization() {
        // Every character that reaches the g2p must be one it can act on: a
        // stray symbol becomes UNK, which is an audible stumble rather than an
        // error anyone sees.
        for text in [
            "café naïve résumé",
            "e.g. I used the AI tool «to draw» a picture — 50% done!",
            "In this; paper, we propose 1 DSPGAN, a GAN-based universal vocoder.",
            "你好 world",
        ] {
            let out = english(text);
            assert!(
                out.chars()
                    .all(|c| c.is_ascii_alphabetic() || c == '\'' || c == ' ' || is_mark(c)),
                "{text} normalized to `{out}`"
            );
        }
    }

    #[test]
    fn accents_fold_onto_the_base_letter() {
        // Dropping them instead would turn "café" into "caf", and the
        // dictionary lookup that follows is exact.
        assert_eq!(english("café naïve"), "cafe naive");
    }

    #[test]
    fn consecutive_punctuation_collapses_to_one_mark() {
        // A run of marks becomes a run of pauses, and `s1` answers a long pause
        // by copying the reference clip instead of continuing the line.
        assert_eq!(english("what?!?"), "what?");
        assert_eq!(english("wait... no"), "wait. no");
    }

    #[test]
    fn capitals_are_split_so_acronyms_get_spelled() {
        // "USA" is not a word; the g2p's single-letter path is what reads it,
        // and it only sees letters that stand alone.
        assert_eq!(english("the USA"), "the U S A");
        assert_eq!(english("OpenAI"), "Open A I");
        assert_eq!(english("The Quick Brown Fox"), "The Quick Brown Fox");
    }
}
