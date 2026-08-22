//! `stt` — speech recognition as a Unix filter.
//!
//! Raw f32le mono PCM at 16 kHz on stdin, text on stdout, logs on stderr:
//!
//! ```sh
//! ffmpeg -i take.mp3 -f f32le -ar 16000 -ac 1 - | stt
//! ```
//!
//! `--format text` (the default) writes one line per segment so it pipes
//! straight into a translator or `voice tts`. `--format jsonl` adds timings and
//! the detected language, which is what a subtitle file or a TTS training
//! manifest needs.
//!
//! Like the voice-conversion filter, this **streams**: stdin is read chunk by
//! chunk and each line is written and flushed as its segment closes, so a long
//! recording produces text as it goes instead of after it ends. Nothing is
//! traded away for that — a segment is only cut once no later sample could move
//! its boundaries, so the transcript is the one the whole recording would have
//! given (`audio_kit::Slicer`). Latency to a line is therefore the segment's own
//! length, plus `--min-silence` to prove it has ended, plus its decode.

use anyhow::{Context, Result};
use clap::{Args, Subcommand, ValueEnum};
use futures::StreamExt;
use std::path::PathBuf;
use stt_core::{DecodeOptions, TranscribeOptions};
use tokio::io::{AsyncWriteExt, BufWriter};

use crate::backend::load_transcriber;
pub use cli_kit::Backend;

/// The whole of the `stt` command tree, defined once and worn two ways: the
/// `stt` binary flattens it at its top level, `voice` nests it under an `stt`
/// subcommand. Every engine here has the same shape — the bare invocation is
/// the stdin→stdout filter, and everything else is a subcommand.
#[derive(Debug, Args)]
pub struct SttCli {
    #[command(flatten)]
    pub transcribe: SttArgs,
    #[command(subcommand)]
    pub command: Option<SttCommand>,
}

/// What this engine can do, and nothing about the executable that hosts it.
///
/// **`completions` is not a member, on purpose.** A completion script describes
/// one binary, so a nested `voice stt completions` could only ever emit
/// `voice`'s — which is exactly what it used to do. Each `main.rs` flattens this
/// enum into its own and adds `Completions` beside it, so the standalone binary
/// keeps the subcommand and `voice` grows only one, at its top level.
#[derive(Debug, Subcommand)]
pub enum SttCommand {
    /// Transcribe audio files to `<stem>.txt` (or `.jsonl`) in a directory.
    Convert(crate::convert::ConvertArgs),
    /// Prefetch the weights recognition needs, so the first run is offline.
    Download(DownloadArgs),
}

/// What a default `stt` run would fetch on demand, fetched up front instead.
///
/// Nothing else: the repo is the only thing recognition downloads, and there is
/// no training-only weight here to leave out.
#[derive(Debug, Args)]
pub struct DownloadArgs {
    /// Override the model repo as `owner/name`
    /// [default: openai/whisper-large-v3-turbo].
    #[arg(long, value_name = "OWNER/NAME")]
    pub repo: Option<String>,
    /// Directory the downloaded models are cached in. Shared by every engine
    /// unless `$STT_CACHE_DIR` (or `$VOICE_CACHE_DIR`) says otherwise.
    #[arg(long, default_value_os_t = hub_kit::cache_dir_for("STT_CACHE_DIR"))]
    pub cache_dir: PathBuf,
    #[command(flatten)]
    pub download: cli_kit::DownloadOpts,
}

pub async fn download(args: DownloadArgs) -> Result<()> {
    // Before the first fetch, so a stalled transfer is bounded rather than
    // discovered.
    args.download.install()?;
    let assets = hub_kit::fetch_whisper(args.repo.as_deref(), &args.cache_dir)
        .await
        .context("failed to fetch the Whisper model")?;
    // Where it landed is the point of the command, so it goes to stdout — it is
    // also what `--model` takes, which is how an offline machine is set up.
    println!("whisper: {}", assets.dir.display());
    Ok(())
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, ValueEnum)]
pub enum Format {
    /// One line of text per segment — pipes into anything.
    #[default]
    Text,
    /// One JSON object per line: start, end, text, language.
    Jsonl,
}

#[derive(Debug, Args)]
pub struct SttArgs {
    /// Directory holding a Hugging Face Whisper repo
    /// [default: auto-downloaded from Hugging Face].
    #[arg(short, long)]
    pub model: Option<PathBuf>,
    /// Override the model repo as `owner/name`
    /// [default: openai/whisper-large-v3-turbo].
    #[arg(long, value_name = "OWNER/NAME")]
    pub repo: Option<String>,
    /// Directory the downloaded models are cached in. Shared by every engine
    /// unless `$STT_CACHE_DIR` (or `$VOICE_CACHE_DIR`) says otherwise.
    #[arg(long, default_value_os_t = hub_kit::cache_dir_for("STT_CACHE_DIR"))]
    pub cache_dir: PathBuf,
    /// Output format.
    #[arg(long, value_enum, default_value_t = Format::Text)]
    pub format: Format,
    /// Force a language by ISO code (`en`, `zh`, `ja`) instead of detecting it.
    /// Worth setting on short or breathy clips, where detection is least sure.
    #[arg(short, long)]
    pub language: Option<String>,
    /// Translate to English rather than transcribing verbatim.
    #[arg(long)]
    pub translate: bool,
    /// Recognition backend; `auto` takes an ONNX export from `--model` if there
    /// is one, else the fastest Burn backend.
    #[arg(long, value_enum, default_value_t = Backend::Auto)]
    pub backend: Backend,
    /// Compute device: `auto`, `cpu`, `gpu`, `gpu:N`, `mps` or `vulkan`.
    #[arg(long, default_value = "auto", value_name = "DEVICE", value_parser = cli_kit::parse_device)]
    pub device: burn_kit::DeviceSpec,
    /// Samples per input read chunk. Read from stdin, so the `convert`
    /// subcommand — which decodes files instead — ignores it.
    #[arg(long, default_value_t = 16000)]
    pub chunk: usize,
    /// Energy floor in dBFS: quieter than this counts as a gap between
    /// utterances. Lower it to keep very soft passages in one segment.
    // `allow_negative_numbers` because this flag's value is always negative and
    // clap would otherwise read `--silence-db -50`'s `-50` as short flags. The
    // same annotation is on every negative-valued flag in the workspace.
    #[arg(long, allow_negative_numbers = true, default_value_t = -40.0)]
    pub silence_db: f32,
    /// Minimum silent-gap length (seconds) that counts as a segment boundary.
    #[arg(long, default_value_t = 0.5)]
    pub min_silence: f32,
    /// Drop any segment shorter than this (seconds).
    #[arg(long, default_value_t = 0.2)]
    pub min_clip: f32,
    /// Hard cap on segment length (seconds). Cannot exceed Whisper's 30 s
    /// encoder window — one segment must fit one window.
    #[arg(long, default_value_t = stt_core::WINDOW_SECONDS)]
    pub max_clip: f32,
    /// Edge-pad each segment by up to this many seconds of bordering quiet, so
    /// onsets and soft breathy tails are not cut off.
    ///
    /// **It also sets streaming latency.** A voiced run's end can only be
    /// finalised once `max(--min-silence, 2 x --pad)` of silence has followed
    /// it, so raising this past half of `--min-silence` makes the bare
    /// invocation wait longer before emitting anything — while `convert`, which
    /// has the whole file, is unaffected. That asymmetry is the reason this is
    /// worth a flag rather than a constant.
    // Read from the slicer's own `Default` rather than spelled again: this one
    // does not diverge from it the way `--min-silence` and `--min-clip`
    // deliberately do, so a literal here could only ever drift out of step.
    #[arg(long, default_value_t = audio_kit::SliceOptions::default().pad)]
    pub pad: f32,
    /// Cap on tokens generated per segment. Raise it if dense speech is being
    /// cut off (a warning says so); lower it to bound a hallucination loop.
    #[arg(long, default_value_t = 224)]
    pub max_tokens: usize,
}

impl SttArgs {
    /// Reject values clap's types accept but the decoder cannot use.
    pub fn verify(&self) -> Result<()> {
        // Finiteness first, because every comparison below it is written as an
        // inequality and `NaN` fails all of them — so a `NaN` would slip past
        // each guard in turn and be reported by none of them. `inf` is the
        // worse half: it *passes* the guards that are phrased as lower bounds,
        // and `--pad inf` then reaches the slicer, where the padded end is
        // `raw_end + pad` in samples and panics on "attempt to add with
        // overflow". `f32::from_str` accepts all three spellings, so the
        // parser cannot be the thing that refuses them.
        for (name, v) in [
            ("--silence-db", self.silence_db),
            ("--min-silence", self.min_silence),
            ("--min-clip", self.min_clip),
            ("--max-clip", self.max_clip),
            ("--pad", self.pad),
        ] {
            anyhow::ensure!(
                v.is_finite(),
                "{name} must be a finite number of {}, not {v}",
                if name == "--silence-db" {
                    "dBFS"
                } else {
                    "seconds"
                }
            );
        }
        // 30 s is not a tuning choice: it is the width of Whisper's encoder
        // window, and a longer segment simply would not fit one.
        anyhow::ensure!(
            self.max_clip > 0.0 && self.max_clip <= stt_core::WINDOW_SECONDS,
            "--max-clip must be in (0, {w}]: one segment has to fit Whisper's {w} s encoder window",
            w = stt_core::WINDOW_SECONDS
        );
        anyhow::ensure!(self.pad >= 0.0, "--pad must not be negative");
        anyhow::ensure!(self.chunk > 0, "--chunk must be at least 1 sample");
        anyhow::ensure!(self.max_tokens > 0, "--max-tokens must be at least 1");
        anyhow::ensure!(self.min_clip >= 0.0, "--min-clip must not be negative");
        anyhow::ensure!(
            self.min_silence > 0.0,
            "--min-silence must be positive: a zero-length gap is every sample boundary"
        );
        anyhow::ensure!(
            self.silence_db <= 0.0,
            "--silence-db is dBFS, so it must be at most 0 (full scale); {} would \
             treat every sample as silence",
            self.silence_db
        );
        anyhow::ensure!(
            self.min_clip <= self.max_clip,
            "--min-clip ({}) exceeds --max-clip ({}), so every segment would be cut \
             to a length that is then discarded",
            self.min_clip,
            self.max_clip
        );
        // `--max-clip` alone does not bound a segment, which is the surprise
        // here. The slicer splits an over-long run only where both halves
        // clear `min_clip`, so it needs a margin of
        // `max(min_clip, max_clip / 4)` at each end and gives up on anything
        // no longer than twice that. With `2 * min_clip > max_clip` the
        // margins meet in the middle and a run in `(max_clip, 2 * min_clip]`
        // is emitted whole, past the cap the user asked for. Where that also
        // clears the encoder window, `mel::compute` truncates it to the first
        // 30 s and transcribes only that, and nothing downstream says so:
        // `--format jsonl` reports the segment's full length, so the timings
        // look right and the words simply stop. The message names that second
        // consequence only when the arithmetic actually reaches it.
        //
        // The bound is sharp. At `--min-clip 15 --max-clip 30` no input
        // length produces a segment over 30 s; at 15.1 the worst case is a
        // 30.20 s segment, and at 20 it is 40.00 s.
        anyhow::ensure!(
            2.0 * self.min_clip <= self.max_clip,
            "--min-clip ({}) is more than half of --max-clip ({}), so a run of \
             unbroken speech between {} and {} s cannot be split and is emitted \
             whole, past the cap{}",
            self.min_clip,
            self.max_clip,
            self.max_clip,
            2.0 * self.min_clip,
            if 2.0 * self.min_clip > stt_core::WINDOW_SECONDS {
                format!(
                    " — and past Whisper's {} s window, where it is truncated and \
                     the rest of the speech silently dropped",
                    stt_core::WINDOW_SECONDS
                )
            } else {
                String::new()
            }
        );
        Ok(())
    }

    /// The Whisper directory this run reads, fetched on demand when `--model`
    /// names none.
    pub async fn model_dir(&self) -> Result<PathBuf> {
        match &self.model {
            Some(dir) => Ok(dir.clone()),
            None => {
                tracing::info!("resolving Whisper weights from Hugging Face...");
                Ok(
                    hub_kit::fetch_whisper(self.repo.as_deref(), &self.cache_dir)
                        .await
                        .context("failed to fetch the Whisper model")?
                        .dir,
                )
            }
        }
    }

    /// Where to cut and what to ask the model for, as `stt-core` wants it.
    pub fn options(&self) -> TranscribeOptions {
        TranscribeOptions {
            slice: audio_kit::SliceOptions {
                silence_db: self.silence_db,
                min_silence: self.min_silence,
                min_clip: self.min_clip,
                max_clip: self.max_clip,
                pad: self.pad,
            },
            decode: DecodeOptions {
                language: self.language.clone(),
                translate: self.translate,
                max_tokens: self.max_tokens,
            },
        }
    }
}

/// One segment as a line of output, in the requested format.
///
/// Shared with `convert`, so a file on disk and the same audio down the pipe
/// cannot come out differently formatted.
pub(crate) fn segment_line(format: Format, s: &stt_core::Segment) -> String {
    match format {
        Format::Text => format!("{}\n", s.text),
        Format::Jsonl => format!(
            "{{\"start\":{:.3},\"end\":{:.3},\"language\":\"{}\",\"text\":{}}}\n",
            s.start,
            s.end,
            s.language,
            json_string(&s.text)
        ),
    }
}

pub async fn transcribe(args: SttArgs) -> Result<()> {
    args.verify()?;

    let dir = args.model_dir().await?;

    let mut stt =
        tokio::task::block_in_place(|| load_transcriber(&dir, args.backend, args.device))?;

    let opts = args.options();

    let mut input = Box::pin(audio_kit::read_f32le(tokio::io::stdin(), args.chunk));
    let mut slicer = audio_kit::Slicer::new(stt_core::SAMPLE_RATE, &opts.slice);
    let mut out = BufWriter::new(tokio::io::stdout());
    let mut segments = 0usize;

    let mut eof = false;
    while !eof {
        // The slicer hands back a clip only once no later sample could move its
        // boundaries, so transcribing here costs nothing in accuracy.
        let clips = match input.next().await {
            Some(chunk) => slicer.push(&chunk.context("reading stdin")?),
            None => {
                eof = true;
                slicer.finish()
            }
        };
        for clip in clips {
            let start = clip.start as f32 / stt_core::SAMPLE_RATE as f32;
            let decoded =
                tokio::task::block_in_place(|| stt.segment(&clip.samples, start, &opts.decode))
                    .context("transcription failed")?;
            let Some(s) = decoded else { continue };
            // Same formatter the `convert` subcommand uses, so the two paths
            // cannot render a segment differently.
            out.write_all(segment_line(args.format, &s).as_bytes())
                .await
                .context("writing stdout")?;
            // Flush per line: a downstream reader is the point of a filter, and
            // it must see a segment as soon as that segment has been decoded.
            out.flush().await.context("writing stdout")?;
            segments += 1;
        }
    }
    tracing::info!("{segments} segments");
    Ok(())
}

/// Escape a transcript as a JSON string.
///
/// Hand-rolled rather than pulling `serde` into the CLI for one field: the input
/// is model-generated text, so the escapes that matter are quotes, backslashes
/// and the control characters below 0x20.
fn json_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    /// The `stt` binary's tree, minus `completions` — which belongs to the
    /// binary rather than to the engine, so it is `main.rs`'s and is tested
    /// there. This is what `voice stt` hosts, verbatim.
    #[derive(Debug, Parser)]
    #[command(name = "stt")]
    struct Bin {
        #[command(flatten)]
        transcribe: SttArgs,
        #[command(subcommand)]
        command: Option<SttCommand>,
    }

    /// `voice`'s tree, so a flag is exercised in the *nested* position too.
    ///
    /// This is the whole reason `allow_negative_numbers` goes on the argument
    /// and not on the command: a `*-cli` crate exports an `Args` type that
    /// somebody else's `Command` hosts, and a `Command`-level annotation is
    /// left behind when that happens.
    #[derive(Debug, Parser)]
    #[command(name = "voice")]
    struct Nested {
        #[command(subcommand)]
        command: NestedCommand,
    }

    #[derive(Debug, Subcommand)]
    enum NestedCommand {
        Stt(SttCli),
    }

    fn parse(argv: &[&str]) -> Bin {
        Bin::parse_from(argv)
    }

    /// Just the filter's own arguments, which is what `verify` and `options`
    /// are defined on. `SttArgs` is an `Args` rather than a `Parser` — that is
    /// the whole point, it is hosted — so it is reached through a tree.
    fn filter(argv: &[&str]) -> SttArgs {
        Bin::parse_from(argv).transcribe
    }

    // Every negative-value test below passes the value SPLIT from its flag
    // (`--flag -50` rather than `--flag=-50`). That is the only form that
    // reproduces the defect: clap reads a leading `-` as the start of a
    // short-flag cluster unless the argument opted out, and the `=` form is
    // never ambiguous. A test written the natural way — the way anybody writes
    // one for a flag they have just added — passes against a broken flag.

    #[test]
    fn the_silence_floor_the_readme_recommends_parses_when_split_from_its_flag() {
        // `stt`'s README says "lower the floor (`--silence-db=-50`)", and the
        // rest of the toolkit's help writes the same advice without the `=`.
        // Both forms have to reach the slicer.
        let a = parse(&["stt", "--silence-db", "-50"]);
        assert_eq!(a.transcribe.silence_db, -50.0);
        assert_eq!(
            parse(&["stt", "--silence-db=-50"]).transcribe.silence_db,
            -50.0
        );
    }

    #[test]
    fn the_silence_floor_parses_nested_under_voice_too() {
        let NestedCommand::Stt(cli) =
            Nested::parse_from(["voice", "stt", "--silence-db", "-50"]).command;
        assert_eq!(cli.transcribe.silence_db, -50.0);
    }

    #[test]
    fn the_silence_floor_reaches_convert_through_the_flatten_as_well() {
        // `ConvertArgs` flattens the very same `SttArgs`, so the annotation has
        // to survive one more level of nesting than the bare invocation needs.
        let Some(SttCommand::Convert(a)) =
            parse(&["stt", "convert", "--silence-db", "-50", "in.wav"]).command
        else {
            panic!("expected convert");
        };
        assert_eq!(a.stt.silence_db, -50.0);
    }

    #[test]
    fn every_other_numeric_flag_still_refuses_a_negative_value() {
        // The negative control. `allow_negative_numbers` widens what a value
        // may look like, and `verify` accepts none of these below zero — so
        // none of them may carry the annotation. A chunk of -1 samples, a
        // negative pad or a negative token cap should still be a usage error.
        for argv in [
            ["stt", "--pad", "-1"],
            ["stt", "--chunk", "-1"],
            ["stt", "--max-tokens", "-1"],
            ["stt", "--min-clip", "-1"],
            ["stt", "--max-clip", "-1"],
            ["stt", "--min-silence", "-1"],
        ] {
            assert!(
                Bin::try_parse_from(argv).is_err(),
                "{} must not parse: verify() accepts no value below zero for it",
                argv[1]
            );
        }
    }

    #[test]
    fn the_only_flag_that_may_be_negative_is_the_one_measured_in_dbfs() {
        // The mechanical question this whole class turns on is not what the
        // help text shows but: does `verify` accept a value below zero? It does
        // for exactly one flag here, and that one is annotated. Stated as a
        // test so a flag added later with a negative-legal range is a failure
        // rather than a silently untypeable option.
        let ok = filter(&["stt", "--silence-db", "-40"]);
        assert!(ok.verify().is_ok());
        let refused = filter(&["stt", "--silence-db=3"]);
        assert!(
            refused.verify().is_err(),
            "--silence-db is dBFS, so a positive value must be refused"
        );
    }

    #[test]
    fn a_segment_cap_wider_than_whispers_window_is_refused() {
        // 30 s is not a tuning choice: it is the width of the encoder window,
        // and `mel::compute` truncates anything longer to it without a word —
        // so a cap above it would produce transcripts that silently stop early
        // while `--format jsonl` reported the segment's full length.
        let a = filter(&["stt", "--max-clip", "31"]);
        let msg = format!(
            "{:#}",
            a.verify().expect_err("31 s cannot fit a 30 s window")
        );
        assert!(
            msg.contains("--max-clip"),
            "the message must name the flag: {msg}"
        );
        // And the boundary itself is legal, so the default is reachable.
        assert!(filter(&["stt"]).verify().is_ok());
        let edge = filter(&["stt", "--max-clip", "30"]);
        assert!(
            edge.verify().is_ok(),
            "the window's own width must be accepted"
        );
    }

    #[test]
    fn a_minimum_longer_than_the_maximum_is_refused_rather_than_dropping_everything() {
        // Otherwise every segment is cut to `--max-clip` and then discarded for
        // being under `--min-clip`: a run that transcribes nothing and reports
        // no reason.
        let a = filter(&["stt", "--min-clip", "20", "--max-clip", "10"]);
        let msg = format!(
            "{:#}",
            a.verify().expect_err("min above max must be refused")
        );
        assert!(
            msg.contains("--min-clip") && msg.contains("--max-clip"),
            "{msg}"
        );
    }

    #[test]
    fn a_zero_length_gap_is_refused_because_every_sample_boundary_is_one() {
        assert!(filter(&["stt", "--min-silence", "0"]).verify().is_err());
        assert!(filter(&["stt", "--chunk", "0"]).verify().is_err());
        assert!(filter(&["stt", "--max-tokens", "0"]).verify().is_err());
        assert!(filter(&["stt", "--max-clip", "0"]).verify().is_err());
    }

    #[test]
    fn a_minimum_over_half_the_maximum_is_refused_because_it_defeats_splitting() {
        // `--max-clip` alone does not bound a segment, which is the whole
        // surprise. The slicer splits an over-long run only where both halves
        // clear `min_clip`, so it needs `max(min_clip, max_clip / 4)` of
        // margin at each end and gives up on anything no longer than twice
        // that. Set `--min-clip` above half of `--max-clip` and the margins
        // meet: a run in `(max_clip, 2 * min_clip]` is emitted whole, and
        // `mel::compute` then truncates it to the first 30 s and transcribes
        // only that — while `--format jsonl` reports the segment's full
        // length, so the timings look right and the words stop.
        //
        // Both numbers below are measured, by slicing 30.05..42.00 s of
        // unbroken tone in 0.05 s steps: at `--min-clip 15` no input length
        // produces a segment over 30 s at all, and at 15.1 the worst case is
        // exactly 30.20 s.
        let a = filter(&["stt", "--min-clip", "15.1", "--max-clip", "30"]);
        let msg = format!(
            "{:#}",
            a.verify().expect_err("15.1 defeats splitting at 30")
        );
        assert!(
            msg.contains("--min-clip") && msg.contains("--max-clip"),
            "the message must name both flags that interact: {msg}"
        );
        // Exactly half is the last safe value, and it must stay reachable.
        assert!(
            filter(&["stt", "--min-clip", "15", "--max-clip", "30"])
                .verify()
                .is_ok(),
            "2 * min_clip == max_clip is the boundary and is safe"
        );
        // The default pair is well inside it, so nobody meets this by accident.
        assert!(filter(&["stt"]).verify().is_ok());
    }

    #[test]
    fn a_non_finite_value_is_refused_before_it_reaches_the_slicer() {
        // The other half of the negative-value class, and the one that bites
        // harder. `f32::from_str` takes "inf" and "NaN", so the *parser* can
        // never refuse them — only `verify` can. `--pad inf` panicked the
        // slicer outright with "attempt to add with overflow" (the padded end
        // is `raw_end + pad` in samples), and `--min-silence inf` was accepted
        // in full, making every gap too short to be a boundary so the whole
        // recording became one segment.
        //
        // `NaN` is the reason this check comes FIRST rather than beside the
        // others: every guard below it is an inequality, and `NaN` fails all
        // of them — so it would slip past each in turn and be reported by
        // none.
        for (flag, value) in [
            ("--pad", "inf"),
            ("--pad", "NaN"),
            ("--min-silence", "inf"),
            ("--min-silence", "NaN"),
            ("--min-clip", "inf"),
            ("--max-clip", "inf"),
            ("--silence-db", "NaN"),
        ] {
            let a = filter(&["stt", flag, value]);
            let err = a
                .verify()
                .expect_err(&format!("`{flag} {value}` must be refused"));
            let msg = format!("{err:#}");
            assert!(
                msg.contains(flag),
                "the message must name the offending flag, got: {msg}"
            );
        }
    }

    #[test]
    fn every_slicer_and_decoder_knob_reaches_the_options_the_engine_reads() {
        // The other half of the same worry. Above: a flag that cannot be typed.
        // Here: a flag that is typed, accepted, and then dropped on the way
        // into the engine — which `-h` cannot show and the engine cannot
        // report, because it never learns the value existed. Every knob is set
        // to something that is NOT its default, so a conversion that hard-codes
        // a field or forgets one fails rather than coincidentally agreeing.
        let a = filter(&[
            "stt",
            "--silence-db",
            "-50",
            "--min-silence",
            "0.7",
            "--min-clip",
            "2.5",
            "--max-clip",
            "9.0",
            "--pad",
            "0.4",
            "--language",
            "ja",
            "--translate",
            "--max-tokens",
            "128",
        ]);
        a.verify().expect("every value here is legal");
        let opts = a.options();
        assert_eq!(opts.slice.silence_db, -50.0);
        assert_eq!(opts.slice.min_silence, 0.7);
        assert_eq!(opts.slice.min_clip, 2.5);
        assert_eq!(opts.slice.max_clip, 9.0);
        assert_eq!(opts.slice.pad, 0.4);
        assert_eq!(opts.decode.language.as_deref(), Some("ja"));
        assert!(opts.decode.translate);
        assert_eq!(opts.decode.max_tokens, 128);
    }

    #[test]
    fn the_default_cap_is_the_encoder_window_and_the_default_pad_is_the_slicers_own() {
        // Two defaults that are derived rather than spelled, and so are the two
        // that a refactor can quietly turn into a literal that then drifts.
        let a = filter(&["stt"]);
        assert_eq!(a.max_clip, stt_core::WINDOW_SECONDS);
        assert_eq!(a.pad, audio_kit::SliceOptions::default().pad);
        // `--pad` is documented as under half of `--min-silence` out of the
        // box, which is what makes the streaming latency it sets cost nothing.
        assert!(
            2.0 * a.pad <= a.min_silence,
            "the default pad ({}) must not set a latency above --min-silence ({})",
            a.pad,
            a.min_silence
        );
    }

    #[test]
    fn the_bare_invocation_is_the_filter_and_carries_no_subcommand() {
        // The house shape: running the engine's binary with no subcommand is
        // the stdin->stdout filter, and everything that is not streaming is a
        // subcommand beside it.
        assert!(parse(&["stt"]).command.is_none());
        assert!(matches!(
            parse(&["stt", "convert", "in.wav"]).command,
            Some(SttCommand::Convert(_))
        ));
        assert!(matches!(
            parse(&["stt", "download"]).command,
            Some(SttCommand::Download(_))
        ));
    }

    #[test]
    fn the_batch_counterpart_is_spelled_convert_like_every_other_engines() {
        // Not `transcribe`. All four engines spell the batch counterpart of the
        // bare invocation identically, which is what lets the toolkit be
        // learned once.
        assert!(Bin::try_parse_from(["stt", "transcribe", "in.wav"]).is_err());
        assert!(Bin::try_parse_from(["stt", "convert", "in.wav"]).is_ok());
    }

    #[test]
    fn convert_demands_an_input_rather_than_silently_transcribing_nothing() {
        assert!(Bin::try_parse_from(["stt", "convert"]).is_err());
        let Some(SttCommand::Convert(a)) = parse(&["stt", "convert", "a.wav", "b.mp3"]).command
        else {
            panic!("expected convert");
        };
        assert_eq!(a.input.len(), 2);
        assert_eq!(a.output_dir, std::path::PathBuf::from("."));
    }

    #[test]
    fn the_engines_own_subcommands_do_not_include_completions() {
        // `completions` describes an *executable*, so it is not part of this
        // engine's clap type. While it sat here, every engine grew a nested
        // copy under `voice` and `voice stt completions bash` emitted a script
        // beginning `_voice()` while advertising `stt`'s. Four spellings of one
        // script, three of them lies. This is the correction — do not undo it.
        assert!(
            Bin::try_parse_from(["stt", "completions", "bash"]).is_err(),
            "`completions` must reach the binary through main.rs's own enum, \
             not through SttCommand"
        );
    }

    #[test]
    fn a_transcript_is_escaped_as_json_rather_than_pasted_into_one() {
        // Model-generated text is what goes through here, so the escapes that
        // matter are the ones a transcript can actually contain.
        let s = stt_core::Segment {
            start: 1.5,
            end: 2.25,
            text: "he said \"hi\"\n\tand\\left".to_string(),
            language: "en".to_string(),
        };
        let line = segment_line(Format::Jsonl, &s);
        assert_eq!(
            line,
            "{\"start\":1.500,\"end\":2.250,\"language\":\"en\",\
             \"text\":\"he said \\\"hi\\\"\\n\\tand\\\\left\"}\n"
        );
        // And a control character below 0x20 that has no short escape.
        let bell = stt_core::Segment {
            text: "a\u{7}b".to_string(),
            ..s.clone()
        };
        assert!(segment_line(Format::Jsonl, &bell).contains("\\u0007"));
    }

    #[test]
    fn the_text_format_is_one_bare_line_per_segment_so_it_pipes() {
        let s = stt_core::Segment {
            start: 0.0,
            end: 1.0,
            text: "hello".to_string(),
            language: "en".to_string(),
        };
        assert_eq!(segment_line(Format::Text, &s), "hello\n");
        assert_eq!(Format::default(), Format::Text);
    }

    #[test]
    fn download_takes_the_same_repo_and_cache_the_filter_would_have_used() {
        // `download` fetches exactly what a default bare invocation would fetch
        // on demand, so the two have to be pointed at the same place by the
        // same flags — otherwise prefetching leaves the first real run
        // downloading anyway.
        let Some(SttCommand::Download(d)) = parse(&[
            "stt",
            "download",
            "--repo",
            "me/whisper",
            "--cache-dir",
            "/tmp/c",
        ])
        .command
        else {
            panic!("expected download");
        };
        assert_eq!(d.repo.as_deref(), Some("me/whisper"));
        assert_eq!(d.cache_dir, std::path::PathBuf::from("/tmp/c"));
        let f = filter(&["stt", "--repo", "me/whisper", "--cache-dir", "/tmp/c"]);
        assert_eq!(f.repo, d.repo);
        assert_eq!(f.cache_dir, d.cache_dir);
        // And the default cache is the shared one, not a per-command guess.
        let (df, dd) = (
            filter(&["stt"]).cache_dir,
            match parse(&["stt", "download"]).command {
                Some(SttCommand::Download(d)) => d.cache_dir,
                _ => panic!("expected download"),
            },
        );
        assert_eq!(df, dd);
        assert_eq!(df, hub_kit::cache_dir_for("STT_CACHE_DIR"));
    }
}
