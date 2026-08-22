//! `voice` argument definitions.
//!
//! No engine's flags are defined here. Each `*-cli` crate exports the whole of
//! its own command tree — [`rvc_cli::args::RvcCli`], [`stt_cli::SttCli`],
//! [`tts_cli::TtsCli`], [`seedvc_cli::SeedVcCli`],
//! [`preprocess_cli::PreprocessCli`] — and this file only nests them, so
//! `voice rvc convert` and `rvc convert` are one definition worn two ways.
//!
//! There is deliberately no top-level asset command. There used to be a `voice
//! models`, which announced itself as fetching "shared model assets" and in
//! fact fetched only voice conversion's two. Every engine has a `download` of
//! its own now, and nesting them is what says which engine's weights a
//! gigabyte is about to be spent on.

use clap::{Parser, Subcommand};
use rvc_cli::args::{CompletionsArgs, RvcCli};

use preprocess_cli::PreprocessCli;
use seedvc_cli::SeedVcCli;
use stt_cli::SttCli;
use tts_cli::TtsCli;

/// voice — speech toolkit: recognition, synthesis and voice conversion, each a
/// filter that composes in a pipe.
#[derive(Debug, Parser)]
#[command(name = "voice", version, about)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Debug, Subcommand)]
pub enum Command {
    /// Voice conversion: f32le mono PCM @16 kHz on stdin, converted PCM on
    /// stdout; `convert`, `train` and `download` beside it.
    // Every engine's tree is boxed because each carries its trainer's arguments,
    // and an enum is as large as its biggest variant.
    Rvc(Box<RvcCli>),
    /// Speech recognition: f32le mono PCM @16 kHz on stdin, text on stdout;
    /// `download` beside it.
    Stt(Box<SttCli>),
    /// Speech synthesis: text on stdin, f32le mono PCM on stdout; `train` and
    /// `download` beside it.
    Tts(Box<TtsCli>),
    /// Zero-shot voice conversion: f32le mono PCM @16 kHz on stdin, converted
    /// PCM @22.05 kHz on stdout; `convert` and `download` beside it. The voice
    /// comes from a reference clip, so there is no `train`.
    // Named explicitly because clap's kebab-case default would spell this
    // variant `seed-vc`, and the subcommand has to be the binary's own name.
    #[command(name = "seedvc")]
    SeedVc(Box<SeedVcCli>),
    /// Corpus preparation: `analyze` a corpus, `separate` a music bed off it,
    /// `diarize` down to one speaker, `denoise`, `normalize`, `trim`,
    /// `resample`, and `clip` into per-utterance training clips. Not an engine
    /// and has no bare invocation — the subcommand *is* which stage runs.
    //
    // This list is `--help` text, so it goes stale silently as stages are
    // added: it named two of the eight for as long as there were more than two.
    // Adding a stage means editing it.
    // No `#[command(name)]`: clap kebab-cases `Preprocess` to `preprocess`,
    // which is already the binary's name. The `SeedVc` variant above needs one
    // only because kebab-case would have spelled it `seed-vc`.
    Preprocess(Box<PreprocessCli>),
    /// Print a shell completion script (bash, zsh, fish, powershell, elvish).
    Completions(CompletionsArgs),
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory;

    /// A completion script, as the shell would receive it.
    ///
    /// The tree's *name* is not the thing to assert on: `clap_complete` reads
    /// the binary name it is handed, and the defect this crate exists to
    /// prevent was a nested subcommand advertising one binary and emitting
    /// another's script. So the check is on the artifact.
    fn completion_script(cmd: clap::Command) -> String {
        let mut cmd = cmd;
        let name = cmd.get_name().to_string();
        let mut out = Vec::new();
        clap_complete::generate(clap_complete::Shell::Bash, &mut cmd, name, &mut out);
        String::from_utf8(out).expect("bash completions are utf-8")
    }

    #[test]
    fn voices_completion_script_is_voices_own() {
        let script = completion_script(Cli::command());
        assert!(
            script.starts_with("_voice()"),
            "expected a script for `voice`, got: {}",
            script.lines().next().unwrap_or_default()
        );
    }

    #[test]
    fn no_engine_offers_a_nested_completions_of_its_own() {
        // **This is the correction, and it is the one a refactor would undo.**
        // While `Completions` sat in each engine's shared enum, every engine
        // grew a nested copy here, and `voice rvc completions bash` emitted a
        // script beginning `_voice()` — `run` was generic over the hosting
        // binary precisely so that arm could build `voice`'s tree, so the
        // nested subcommand advertised the engine's completions and produced
        // `voice`'s. Four spellings of one script, three of them lies.
        //
        // A completion script describes an *executable*, and under `voice` the
        // executable is `voice`. So there is exactly one, at the top level.
        for engine in ["rvc", "stt", "tts", "seedvc", "preprocess"] {
            assert!(
                Cli::try_parse_from(["voice", engine, "completions", "bash"]).is_err(),
                "`voice {engine} completions` must not parse: a completion script \
                 belongs to the binary, and here the binary is `voice`"
            );
        }
        // And the one that does exist, does.
        assert!(Cli::try_parse_from(["voice", "completions", "bash"]).is_ok());
    }

    #[test]
    fn there_is_no_top_level_download() {
        // There used to be a `voice models`, which announced itself as fetching
        // "shared model assets" and in fact fetched only voice conversion's
        // two. The honest form of a cross-engine fetch names the engine whose
        // gigabytes are being spent, so every `download` is one level in.
        assert!(
            Cli::try_parse_from(["voice", "download"]).is_err(),
            "a top-level download cannot say whose weights it is about to fetch"
        );
        assert!(Cli::try_parse_from(["voice", "models"]).is_err());
        for engine in ["rvc", "stt", "tts", "seedvc"] {
            assert!(
                Cli::try_parse_from(["voice", engine, "download"]).is_ok(),
                "`voice {engine} download` is the honest form and must parse"
            );
        }
        // `preprocess` is the one without one, and that absence is meaningful:
        // it has no default bare invocation, so there is nothing a `download`
        // could mean without naming a stage — which is what running the stage
        // does.
        assert!(Cli::try_parse_from(["voice", "preprocess", "download"]).is_err());
    }

    #[test]
    fn every_engine_appears_under_its_own_name_and_no_other() {
        // `voice` nests, it never renames. Clap kebab-cases a variant, so
        // `SeedVc` would render as `seed-vc` while the binary is `seedvc` — the
        // `#[command(name = "seedvc")]` on that variant is a rename *back* to
        // the engine's own name, which is the rule rather than an exception.
        for engine in ["rvc", "stt", "tts", "seedvc", "preprocess"] {
            assert!(
                Cli::command().find_subcommand(engine).is_some(),
                "`voice {engine}` must be spelled exactly as the binary is"
            );
        }
        assert!(
            Cli::command().find_subcommand("seed-vc").is_none(),
            "clap's kebab-case default must not leak: the binary is `seedvc`"
        );
        assert!(Cli::try_parse_from(["voice", "seed-vc", "download"]).is_err());
        assert!(Cli::try_parse_from(["voice", "seedvc", "download"]).is_ok());
    }

    #[test]
    fn no_engines_subcommand_was_flattened_into_a_hyphenated_name() {
        // A hyphenated name at the top level is the tell that somebody
        // flattened a level by hand — `voice tts-train` rather than
        // `voice tts train`. There is one hyphen-free rule and this states it.
        for sub in Cli::command().get_subcommands() {
            assert!(
                !sub.get_name().contains('-'),
                "`voice {}` is hyphenated, which is what flattening a level \
                 looks like; nest instead",
                sub.get_name()
            );
        }
        assert!(Cli::try_parse_from(["voice", "tts-train"]).is_err());
        assert!(Cli::try_parse_from(["voice", "rvc-convert"]).is_err());
    }

    #[test]
    fn each_engines_whole_verb_list_is_hosted_unchanged() {
        // The property that makes `voice tts train` literally the same code
        // path as `tts train`: `voice` hosts each engine's clap type, so a
        // subcommand added to an engine appears here with no edit to this
        // crate at all. Naming the expected verbs is what turns a *missing*
        // one into a failure — a name-set comparison alone would pass an
        // engine that lost `download`.
        for (engine, want) in [
            ("rvc", &["convert", "train", "download"][..]),
            ("stt", &["convert", "download"][..]),
            ("tts", &["convert", "train", "download"][..]),
            // No `train`: the reference clip is the whole speaker
            // specification, so this engine's missing verb is its defining
            // property rather than a gap.
            ("seedvc", &["convert", "download"][..]),
            (
                "preprocess",
                &[
                    "clip",
                    "denoise",
                    "separate",
                    "diarize",
                    "analyze",
                    "normalize",
                    "trim",
                    "resample",
                ][..],
            ),
        ] {
            let cmd = Cli::command();
            let hosted = cmd
                .find_subcommand(engine)
                .unwrap_or_else(|| panic!("`voice {engine}` must exist"));
            let got: Vec<&str> = hosted.get_subcommands().map(|s| s.get_name()).collect();
            for verb in want {
                assert!(
                    got.contains(verb),
                    "`voice {engine} {verb}` is missing; hosted: {got:?}"
                );
            }
            // `completions` is the one that must NOT have come along.
            assert!(
                !got.contains(&"completions"),
                "`voice {engine}` hosts a `completions`, which can only emit \
                 `voice`'s script while advertising {engine}'s"
            );
        }
    }

    #[test]
    fn the_four_engines_all_have_a_bare_invocation_and_preprocess_does_not() {
        // The house shape: an engine names one transformation, so "the engine
        // on a pipe" is a complete description and the bare invocation means
        // exactly one thing. `preprocess` names a *phase*, and which stage to
        // run is precisely what the subcommand chooses — so a bare one would
        // have to pick silently, and whichever it picked would be wrong for
        // everybody who wanted a different one.
        for engine in ["rvc", "stt", "tts", "seedvc"] {
            assert!(
                Cli::try_parse_from(["voice", engine]).is_ok(),
                "`voice {engine}` alone is that engine's filter and must parse"
            );
        }
        assert!(
            Cli::try_parse_from(["voice", "preprocess"]).is_err(),
            "a bare `preprocess` would have to choose a stage silently"
        );
    }

    #[test]
    fn a_flag_is_defined_once_and_reaches_the_engine_through_the_nesting() {
        // The whole point of depending on the `*-cli` crates as libraries. If
        // `voice` ever grew its own copy of an engine's arguments, this is
        // where a value would stop arriving — the flag would parse against the
        // copy and the engine would run on its default.
        let cli = Cli::try_parse_from([
            "voice",
            "stt",
            "--silence-db",
            "-50",
            "--min-silence",
            "0.7",
            "--max-clip",
            "9.0",
        ])
        .expect("the nested filter takes the engine's own flags");
        let Command::Stt(stt) = cli.command else {
            panic!("expected stt");
        };
        // Read through `options()`, so this fails if the value is parsed and
        // then dropped on the way into the engine as well as if it never
        // parsed at all.
        let opts = stt.transcribe.options();
        assert_eq!(opts.slice.silence_db, -50.0);
        assert_eq!(opts.slice.min_silence, 0.7);
        assert_eq!(opts.slice.max_clip, 9.0);
    }

    #[test]
    fn a_negative_value_survives_the_nesting_on_every_engine_that_has_one() {
        // `allow_negative_numbers` goes on the *argument* and not on the
        // command, and this is the position that proves why: a `*-cli` crate
        // exports an `Args` type that this crate's `Command` hosts, and a
        // command-level annotation is left behind by exactly that move. Each
        // value is SPLIT from its flag, which is the only form that reproduces
        // the defect.
        assert!(Cli::try_parse_from(["voice", "rvc", "convert", "-t", "-5", "in.wav"]).is_ok());
        assert!(Cli::try_parse_from(["voice", "stt", "--silence-db", "-50"]).is_ok());
        assert!(
            Cli::try_parse_from([
                "voice",
                "preprocess",
                "clip",
                "--silence-db",
                "-50",
                "in.wav",
            ])
            .is_ok()
        );
        assert!(
            Cli::try_parse_from([
                "voice",
                "preprocess",
                "normalize",
                "--lufs",
                "-23",
                "in.wav",
            ])
            .is_ok()
        );
    }
}
