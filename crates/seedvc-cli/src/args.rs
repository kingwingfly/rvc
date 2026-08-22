//! Command-line argument definitions (clap derive).
//!
//! The shape is fixed by the house rule every engine follows: the bare
//! invocation is the stdin→stdout filter, and everything that is not streaming
//! is a subcommand beside it.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use clap::{Args, Subcommand};

pub use cli_kit::{Backend, CompletionsArgs};

/// The whole of the `seedvc` **engine**, defined once and worn two ways: the
/// `seedvc` binary flattens it at its top level, `voice` nests it under a
/// `seedvc` subcommand.
///
/// `completions` is deliberately not in here — it describes the *binary* being
/// completed, not the engine, so each `main` adds it beside this. See
/// [`SeedVcCommand`].
#[derive(Debug, Args)]
pub struct SeedVcCli {
    #[command(flatten)]
    pub filter: FilterArgs,
    #[command(subcommand)]
    pub command: Option<SeedVcCommand>,
}

/// What this engine can do, and nothing about the executable that hosts it.
///
/// **`completions` is not a member, on purpose.** A completion script describes
/// one binary, so a nested `voice seedvc completions` could only ever emit
/// `voice`'s — which is exactly what it used to do. Each `main.rs` flattens this
/// enum into its own and adds `Completions` beside it, so the standalone binary
/// keeps the subcommand and `voice` grows only one, at its top level.
#[derive(Debug, Subcommand)]
pub enum SeedVcCommand {
    /// Convert audio files into the reference's voice (WAV output).
    // Boxed because it carries every model path and every sampler knob on top of
    // its own two, and clippy is right that `completions` shouldn't pay for that.
    Convert(Box<ConvertArgs>),
    /// Prefetch the weights a conversion needs, so the first run is offline.
    Download(DownloadArgs),
}

/// Where the model comes from — the four checkpoints, or the six ONNX graphs —
/// and which voice to convert into.
///
/// The reference lives here rather than beside the input files because both the
/// filter and `convert` need it, and neither can be given it by clap: these
/// options are flattened beside the subcommands for the bare invocation, so
/// marking it `required` would demand one of `download` and `completions` too.
#[derive(Debug, Args, Clone)]
pub struct ModelOpts {
    /// A 1–30 s recording of the target voice. This is the whole speaker
    /// specification — there is nothing to train.
    #[arg(short, long)]
    pub reference: Option<PathBuf>,
    /// Seconds of `--reference` that are read. The reference's mel and each
    /// source chunk share one 30 s window, so this is a two-sided dial and not a
    /// quality knob: the timbre vector is a pooled average and saturates within
    /// seconds, while every second of reference is a second the source loses
    /// from every chunk. At the 25 s default a chunk carries under 5 s of
    /// source; at 5 s it carries nearly 25, so a long clip costs five times the
    /// chunks and five times the seams for a voice that is no better specified.
    /// Only ever shortens: a clip under the cap is read whole.
    #[arg(long, value_name = "SECONDS", default_value_t = seedvc_core::reference::REFERENCE_SECONDS)]
    pub reference_secs: f32,
    /// Seed-VC checkpoint: the transformer and the length regulator
    /// [default: auto-downloaded from Hugging Face].
    #[arg(long)]
    pub checkpoint: Option<PathBuf>,
    /// CAMPPlus timbre encoder [default: auto-downloaded from Hugging Face].
    #[arg(long)]
    pub campplus: Option<PathBuf>,
    /// BigVGAN vocoder weights [default: auto-downloaded from Hugging Face].
    #[arg(long)]
    pub bigvgan: Option<PathBuf>,
    /// Whisper content encoder directory [default: auto-downloaded from
    /// Hugging Face].
    #[arg(long)]
    pub content: Option<PathBuf>,
    /// Directory holding the six ONNX graphs `export/export_seedvc.py` writes —
    /// either directly, or in an `onnx/` subdirectory inside it. Naming one runs
    /// the whole conversion on ONNX Runtime and ignores the four checkpoint
    /// flags above: a graph carries its weights, so there is nothing to fetch
    /// and nothing to override. It cannot be combined with a `--backend` that
    /// names Burn, since only ONNX Runtime can read a graph — that pair is an
    /// error rather than one of the two winning.
    #[arg(long)]
    pub onnx: Option<PathBuf>,
    /// Directory the downloaded models are cached in. Shared by every engine
    /// unless `$SEEDVC_CACHE_DIR` (or `$VOICE_CACHE_DIR`) says otherwise.
    #[arg(long, default_value_os_t = hub_kit::cache_dir_for("SEEDVC_CACHE_DIR"))]
    pub cache_dir: PathBuf,
}

impl ModelOpts {
    /// The reference recording every conversion path needs.
    ///
    /// Optional to clap for the reason the struct's own docs give; "required" is
    /// decided here, and callers check it before anything is fetched or loaded.
    pub fn reference(&self) -> Result<&Path> {
        self.reference.as_deref().context(
            "-r/--reference is required: a 1-30 s recording of the voice to \
             convert into (there is no model to train — the clip is the whole \
             speaker specification)",
        )
    }

    /// Reject a reference cap the analysis cannot use.
    ///
    /// Called from **both** hosting commands' `verify`, which run before a byte
    /// is fetched: adding it to one is the silent half of this check, since the
    /// filter and `convert` analyse the reference through the same function.
    pub fn verify(&self) -> Result<()> {
        anyhow::ensure!(
            self.reference_secs.is_finite() && self.reference_secs > 0.0,
            "--reference-secs is {}: it is the longest stretch of the reference that is read, \
             so it has to be a positive, finite number of seconds",
            self.reference_secs,
        );
        // The content encoder refuses a clip past its own window rather than
        // truncating one, so a cap above it reads no more audio — it only moves
        // the refusal to after four checkpoints have been fetched and loaded.
        // Named here, where it costs a message instead.
        anyhow::ensure!(
            self.reference_secs <= seedvc_core::reference::MAX_REFERENCE_SECONDS,
            "--reference-secs is {} and the content encoder's window is {} s: a longer cap \
             cannot read more of the clip, so trim the reference itself if it runs past that",
            self.reference_secs,
            seedvc_core::reference::MAX_REFERENCE_SECONDS,
        );
        Ok(())
    }
}

/// The sampler's knobs, shared by the filter and `convert`.
#[derive(Debug, Args, Clone, Copy)]
pub struct SamplerOpts {
    /// Euler steps the flow-matching sampler takes from noise to mel. Cost and
    /// smoothness are both linear in it.
    #[arg(long, default_value_t = 30)]
    pub steps: usize,
    /// Classifier-free guidance scale: how far each step is pushed away from the
    /// unconditioned prediction. Raise to follow the reference harder, at the
    /// cost of artefacts; zero or below skips the unconditional pass entirely
    /// and halves the work per step.
    // `allow_negative_numbers` because "zero or below" is the sentence above,
    // and a value below zero is what selects that branch: `verify` accepts one,
    // and `burn_seedvc::flow`'s `if self.guidance > 0.0` is what reads it.
    // Without the annotation clap takes the leading `-` for a short-flag
    // cluster, so `--guidance -1` fails with `unexpected argument '-1'` while
    // `--guidance=-1` works — which reads as a shell quoting problem and is not
    // one. The other flags on this struct deliberately go without it: a count of
    // Euler steps, a stretch factor and a seed have no negative value to reach.
    #[arg(long, allow_negative_numbers = true, default_value_t = 0.7)]
    pub guidance: f64,
    /// Scales the output's duration against the source's. Above 1 is slower,
    /// below is faster; pitch is unchanged.
    #[arg(long, default_value_t = 1.0)]
    pub length_adjust: f64,
    /// Seed for the sampler's noise, so a conversion can be repeated exactly.
    #[arg(long, default_value_t = 0)]
    pub seed: u64,
}

impl SamplerOpts {
    /// Reject values clap's types accept but the sampler cannot use.
    pub fn verify(&self) -> Result<()> {
        anyhow::ensure!(
            self.steps > 0,
            "--steps must be at least 1: the sampler integrates the flow over \
             that many Euler steps, and zero of them leaves pure noise"
        );
        anyhow::ensure!(
            self.length_adjust > 0.0 && self.length_adjust.is_finite(),
            "--length-adjust is {}: it must be a positive, finite factor (1.0 keeps the \
             source's duration)",
            self.length_adjust,
        );
        // `nan` and `inf` are values `f64::from_str` accepts, so clap does too.
        // Guidance is the one knob here with no bound in either direction, so
        // finiteness is the whole of what can be checked — and it is worth
        // checking, because an infinite scale makes the extrapolation
        // `(1 + g)*conditioned - g*unconditional` non-finite for every element
        // of the mel, and a vocoder renders that to silence rather than to an
        // error. Named here, before the four checkpoints are fetched.
        anyhow::ensure!(
            self.guidance.is_finite(),
            "--guidance is {}: it scales how far each step is pushed away from the \
             unconditioned prediction, so it has to be a finite number (0 or below skips \
             that pass entirely)",
            self.guidance,
        );
        Ok(())
    }

    /// The engine's own conversion settings.
    pub fn options(&self) -> seedvc_core::ConvertOptions {
        seedvc_core::ConvertOptions {
            sampler: seedvc_core::Sampler {
                steps: self.steps,
                guidance: self.guidance,
            },
            length_adjust: self.length_adjust,
            seed: self.seed,
        }
    }
}

/// The bare invocation: raw f32le mono PCM in at 16 kHz, raw f32le mono PCM out
/// at the vocoder's 22.05 kHz.
#[derive(Debug, Args)]
pub struct FilterArgs {
    #[command(flatten)]
    pub models: ModelOpts,
    #[command(flatten)]
    pub sampler: SamplerOpts,
    /// Samples per input read chunk from stdin (16 kHz mono f32le).
    #[arg(long, default_value_t = 1600)]
    pub chunk: usize,
    /// Mel frames of new audio each converted chunk carries — the latency dial,
    /// since a chunk is a window rather than a filter and nothing can be emitted
    /// until a whole one exists. At the model's 86.13 Hz the default is 2.0 s,
    /// with 16 frames of crossfade on top. Clamped to whatever the reference
    /// leaves of the shared 30 s window, so `--reference-secs` bounds it.
    ///
    /// In **frames** rather than seconds, unlike `rvc`'s `--block-secs`: every
    /// window this engine has is measured in frames, and the source samples
    /// behind one frame move with `--length-adjust` — so a block in seconds
    /// would change size with a flag that has nothing to do with latency.
    ///
    /// On the bare invocation only. `convert` has whole files and drives the
    /// batch path, which picks one chunk from the room the reference leaves and
    /// has no streaming geometry to put this in.
    #[arg(long, value_name = "FRAMES", default_value_t = seedvc_core::StreamParams::realtime().block)]
    pub block_frames: usize,
    /// Inference backend: `onnx`, `cuda` (aliases `burn`, `burn-cuda`), `tch`
    /// (`libtorch`, `burn-tch`) or `wgpu` (`webgpu`, `burn-wgpu`); `auto` picks
    /// ONNX Runtime when `--onnx` names an export, else the fastest compiled in.
    #[arg(long, value_enum, default_value_t = Backend::Auto)]
    pub backend: Backend,
    /// Compute device: `auto` (fastest visible), `cpu`, `gpu`, `gpu:N`, `mps` or
    /// `vulkan` (`cuda`/`cuda:N` also accepted). The `cuda` backend has GPUs only.
    #[arg(long, default_value = "auto", value_name = "DEVICE", value_parser = cli_kit::parse_device)]
    pub device: burn_kit::DeviceSpec,
}

impl FilterArgs {
    /// Reject values clap's types accept but the pipeline cannot use.
    pub fn verify(&self) -> Result<()> {
        // A zero-sample read would spin on stdin forever without ever handing
        // the converter a block to work on.
        anyhow::ensure!(self.chunk > 0, "--chunk must be at least 1 sample");
        // A chunk that carries no *new* audio advances nothing, which is the
        // same hang one flag along. `Converter::new` refuses it too, but in
        // terms of the crossfade — naming the flag that caused it is worth
        // doing here, before the four checkpoints are fetched.
        anyhow::ensure!(
            self.block_frames > 0,
            "--block-frames must be at least 1: a chunk carrying no new audio never advances, \
             so the filter would read its input forever without emitting any of it"
        );
        self.models.verify()?;
        self.sampler.verify()
    }

    /// The streaming geometry this invocation asks for.
    ///
    /// Everything but the block stays at the preset, and the block *defaults* to
    /// the preset's own field — so omitting the flag is the same number through
    /// the same code path, not a value converted back and forth.
    pub fn params(&self) -> seedvc_core::StreamParams {
        seedvc_core::StreamParams {
            block: self.block_frames,
            ..seedvc_core::StreamParams::realtime()
        }
    }
}

/// Batch conversion: files in, one WAV each out.
#[derive(Debug, Args)]
pub struct ConvertArgs {
    #[command(flatten)]
    pub models: ModelOpts,
    #[command(flatten)]
    pub sampler: SamplerOpts,
    /// One or more input audio files (mp3, wav, ...).
    #[arg(required = true)]
    pub input: Vec<PathBuf>,
    /// Directory to write converted `<stem>.wav` files into.
    #[arg(short = 'o', long, default_value = ".")]
    pub output_dir: PathBuf,
    /// Inference backend: `onnx`, `cuda` (aliases `burn`, `burn-cuda`), `tch`
    /// (`libtorch`, `burn-tch`) or `wgpu` (`webgpu`, `burn-wgpu`); `auto` picks
    /// ONNX Runtime when `--onnx` names an export, else the fastest compiled in.
    #[arg(long, value_enum, default_value_t = Backend::Auto)]
    pub backend: Backend,
    /// Compute device: `auto` (fastest visible), `cpu`, `gpu`, `gpu:N`, `mps` or
    /// `vulkan` (`cuda`/`cuda:N` also accepted). The `cuda` backend has GPUs only.
    #[arg(long, default_value = "auto", value_name = "DEVICE", value_parser = cli_kit::parse_device)]
    pub device: burn_kit::DeviceSpec,
}

impl ConvertArgs {
    /// Reject values clap's types accept but the pipeline cannot use.
    pub fn verify(&self) -> Result<()> {
        self.models.verify()?;
        self.sampler.verify()
    }
}

/// What a default run would fetch on demand, fetched up front instead.
///
/// All four networks, because a conversion opens all four. There is nothing
/// optional to leave out as `tts` has, and no warm-start base as `rvc` has —
/// Seed-VC is zero-shot, so nothing it downloads is ever a training input.
#[derive(Debug, Args)]
pub struct DownloadArgs {
    /// Directory the downloaded models are cached in. Shared by every engine
    /// unless `$SEEDVC_CACHE_DIR` (or `$VOICE_CACHE_DIR`) says otherwise.
    #[arg(long, default_value_os_t = hub_kit::cache_dir_for("SEEDVC_CACHE_DIR"))]
    pub cache_dir: PathBuf,
    #[command(flatten)]
    pub download: cli_kit::DownloadOpts,
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    /// The `seedvc` binary's tree, minus `completions` — which belongs to the
    /// binary rather than to the engine, so it is not part of what is tested
    /// here. The bare invocation's flags sit beside the subcommands exactly as
    /// `main.rs` arranges them, because that adjacency is what stops clap
    /// marking `-r` required and is therefore load-bearing for these tests.
    #[derive(Debug, Parser)]
    #[command(name = "seedvc")]
    struct Bin {
        #[command(flatten)]
        filter: FilterArgs,
        #[command(subcommand)]
        command: Option<SeedVcCommand>,
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
        #[command(name = "seedvc")]
        SeedVc(SeedVcCli),
    }

    /// `try_parse_from(..).unwrap()` rather than `parse_from`: the latter
    /// prints and calls `exit`, which takes the whole test binary down and
    /// hides every other failure in the file.
    fn parse(argv: &[&str]) -> Bin {
        Bin::try_parse_from(argv).unwrap()
    }

    fn convert(argv: &[&str]) -> Box<ConvertArgs> {
        match parse(argv).command {
            Some(SeedVcCommand::Convert(a)) => a,
            other => panic!("expected convert, got {other:?}"),
        }
    }

    // ---------------------------------------------------------------------
    // The negative-value flag class.
    //
    // Every test below passes the value SPLIT from its flag (`--flag -1`
    // rather than `--flag=-1`). That is the only form that reproduces the
    // defect: clap reads a leading `-` as the start of a short-flag cluster
    // unless the argument opted out, and the `=` form is never ambiguous. A
    // test written the natural way — the way anybody writes one for a flag
    // they have just added — passes against a broken flag.
    //
    // `--guidance` is the one flag here whose value may begin with `-`, and
    // the mechanical test that says so is not what its examples show but
    // whether `verify` accepts a value below zero. It does, and its own help
    // says what such a value means: "zero or below skips the unconditional
    // pass entirely and halves the work per step".
    // ---------------------------------------------------------------------

    #[test]
    fn a_negative_guidance_parses_when_split_from_its_flag() {
        let a = convert(&["seedvc", "convert", "--guidance", "-1", "in.wav"]);
        assert_eq!(a.sampler.guidance, -1.0);
        // And it is a value the sampler is documented to accept, so `verify`
        // has to let it through as well: a parser and a validator that
        // disagree make a documented value unreachable either way.
        a.verify().unwrap();
    }

    #[test]
    fn a_negative_guidance_parses_on_the_bare_invocation_too() {
        // `SamplerOpts` is flattened into `FilterArgs` as well, so the
        // annotation has to survive that flatten — the bare invocation is the
        // primary mode of this binary, not an afterthought.
        let bin = parse(&["seedvc", "--guidance", "-0.5", "-r", "me.wav"]);
        assert_eq!(bin.filter.sampler.guidance, -0.5);
        bin.filter.verify().unwrap();
    }

    #[test]
    fn a_negative_guidance_parses_nested_under_voice_too() {
        let NestedCommand::SeedVc(cli) =
            Nested::try_parse_from(["voice", "seedvc", "convert", "--guidance", "-1", "in.wav"])
                .unwrap()
                .command;
        let Some(SeedVcCommand::Convert(a)) = cli.command else {
            panic!("expected convert");
        };
        assert_eq!(a.sampler.guidance, -1.0);
    }

    // The negative controls. `allow_negative_numbers` widens what a value may
    // look like, so it belongs only where a negative one is meaningful — a
    // count of Euler steps, a number of samples, a stretch factor and a
    // duration are none of them, and each must still be refused by the parser
    // rather than reaching a `verify` that would have to say so twice.

    #[test]
    fn the_flags_that_cannot_be_negative_are_still_refused_by_the_parser() {
        for (flag, value) in [
            ("--steps", "-1"),
            ("--length-adjust", "-1"),
            ("--reference-secs", "-1"),
            ("--seed", "-1"),
        ] {
            assert!(
                Bin::try_parse_from(["seedvc", "convert", flag, value, "in.wav"]).is_err(),
                "{flag} {value} parsed, but a negative value there is not meaningful"
            );
        }
        for (flag, value) in [("--chunk", "-1"), ("--block-frames", "-1")] {
            assert!(
                Bin::try_parse_from(["seedvc", flag, value, "-r", "me.wav"]).is_err(),
                "{flag} {value} parsed, but a negative value there is not meaningful"
            );
        }
    }

    // ---------------------------------------------------------------------
    // What `verify` refuses. Each of these reaches arithmetic if it is not
    // caught here, and `verify` runs before a byte of the four checkpoints is
    // fetched — which is the whole reason the check is worth making twice.
    // ---------------------------------------------------------------------

    #[test]
    fn zero_euler_steps_is_refused_by_name() {
        let a = convert(&["seedvc", "convert", "--steps", "0", "in.wav"]);
        let err = a.verify().unwrap_err().to_string();
        assert!(err.contains("--steps"), "{err}");
    }

    #[test]
    fn a_zero_length_adjust_is_refused_by_name() {
        let a = convert(&["seedvc", "convert", "--length-adjust", "0", "in.wav"]);
        let err = a.verify().unwrap_err().to_string();
        assert!(err.contains("--length-adjust"), "{err}");
    }

    #[test]
    fn a_non_finite_sampler_knob_is_refused_before_it_reaches_the_flow() {
        // `nan` and `inf` are values `f64::from_str` accepts, so clap does
        // too. Guidance at infinity makes the extrapolation
        // `(1 + g)·conditioned − g·unconditional` non-finite for every
        // element of the mel, and a `length_adjust` of infinity saturates the
        // frame count it scales. `StreamParams::chunk` catches the second one
        // in the core, but only on the streaming path and only after four
        // checkpoints have been loaded.
        for (flag, value) in [
            ("--guidance", "inf"),
            ("--guidance", "nan"),
            ("--length-adjust", "inf"),
            ("--length-adjust", "nan"),
        ] {
            let a = convert(&["seedvc", "convert", flag, value, "in.wav"]);
            let Err(err) = a.verify() else {
                panic!("{flag} {value} was accepted");
            };
            let err = err.to_string();
            assert!(err.contains(flag), "{flag} {value}: {err}");
        }
    }

    #[test]
    fn a_zero_read_chunk_is_refused_by_name() {
        let bin = parse(&["seedvc", "--chunk", "0", "-r", "me.wav"]);
        let err = bin.filter.verify().unwrap_err().to_string();
        assert!(err.contains("--chunk"), "{err}");
    }

    #[test]
    fn a_block_that_carries_no_new_audio_is_refused_by_name() {
        let bin = parse(&["seedvc", "--block-frames", "0", "-r", "me.wav"]);
        let err = bin.filter.verify().unwrap_err().to_string();
        assert!(err.contains("--block-frames"), "{err}");
    }

    #[test]
    fn a_reference_cap_the_analysis_cannot_use_is_refused_by_name() {
        for value in ["0", "31", "inf", "nan"] {
            let a = convert(&["seedvc", "convert", "--reference-secs", value, "in.wav"]);
            let err = a.verify().unwrap_err().to_string();
            assert!(err.contains("--reference-secs"), "{value}: {err}");
        }
    }

    #[test]
    fn the_reference_cap_is_checked_by_both_hosting_commands() {
        // The check lives on `ModelOpts`, which both the filter and `convert`
        // flatten — adding it to one is the silent half, since both analyse
        // the reference through the same function. Pinned here so a `verify`
        // that stops delegating shows up.
        let bin = parse(&["seedvc", "--reference-secs", "31", "-r", "me.wav"]);
        let err = bin.filter.verify().unwrap_err().to_string();
        assert!(err.contains("--reference-secs"), "{err}");
        let a = convert(&["seedvc", "convert", "--reference-secs", "31", "in.wav"]);
        assert!(
            a.verify()
                .unwrap_err()
                .to_string()
                .contains("--reference-secs")
        );
        // The cap the engine itself defaults to has to survive its own check.
        assert_eq!(
            seedvc_core::reference::MAX_REFERENCE_SECONDS,
            30.0,
            "the message above quotes this, so a change to it changes the advice"
        );
        convert(&["seedvc", "convert", "in.wav"]).verify().unwrap();
    }

    // ---------------------------------------------------------------------
    // A flag that is typed, accepted, and then dropped on the way into the
    // engine — which `-h` cannot show and the engine cannot report, because it
    // never learns the value existed. Each knob is set to something that is
    // NOT its default, so a conversion that hard-codes a field or forgets one
    // fails rather than coincidentally agreeing.
    // ---------------------------------------------------------------------

    #[test]
    fn every_sampler_knob_reaches_the_engine() {
        let a = convert(&[
            "seedvc",
            "convert",
            "--steps",
            "7",
            "--guidance",
            "-1",
            "--length-adjust",
            "1.25",
            "--seed",
            "99",
            "in.wav",
        ]);
        let opts = a.sampler.options();
        assert_eq!(opts.sampler.steps, 7);
        assert_eq!(opts.sampler.guidance, -1.0);
        assert_eq!(opts.length_adjust, 1.25);
        assert_eq!(opts.seed, 99);
    }

    #[test]
    fn the_sampler_defaults_are_the_ones_the_engine_would_have_picked() {
        // `SamplerOpts` restates the engine's own defaults rather than
        // deferring to them, so the two can drift in silence while `-h` goes
        // on printing whichever clap holds.
        let got = convert(&["seedvc", "convert", "in.wav"]).sampler.options();
        let want = seedvc_core::ConvertOptions::default();
        assert_eq!(got.sampler.steps, want.sampler.steps);
        assert_eq!(got.sampler.guidance, want.sampler.guidance);
        assert_eq!(got.length_adjust, want.length_adjust);
        assert_eq!(got.seed, want.seed);
    }

    #[test]
    fn the_block_flag_is_the_only_thing_it_moves_in_the_stream_geometry() {
        // `params()` overrides one field of the preset and inherits the rest,
        // so a value that is not the default has to arrive while every other
        // field stays where `realtime()` put it.
        let bin = parse(&["seedvc", "--block-frames", "64", "-r", "me.wav"]);
        let (got, preset) = (bin.filter.params(), seedvc_core::StreamParams::realtime());
        assert_eq!(got.block, 64);
        assert_ne!(got.block, preset.block, "64 has to differ from the default");
        assert_eq!(got.crossfade, preset.crossfade);
        // Omitting the flag is the preset's own number through the same code
        // path, not a value converted back and forth.
        let bin = parse(&["seedvc", "-r", "me.wav"]);
        assert_eq!(bin.filter.params().block, preset.block);
    }

    // ---------------------------------------------------------------------
    // The reference is the whole speaker specification, so its absence is the
    // one thing that must be reported before anything is fetched.
    // ---------------------------------------------------------------------

    #[test]
    fn a_missing_reference_is_refused_with_the_flag_that_supplies_it() {
        // Clap cannot mark it required — these options sit beside the
        // subcommands, so `required` would demand one of `download` and
        // `completions` too — which is why the check is here at all.
        let bin = parse(&["seedvc"]);
        let err = bin.filter.models.reference().unwrap_err().to_string();
        assert!(err.contains("--reference"), "{err}");
        let a = convert(&["seedvc", "convert", "in.wav"]);
        assert!(
            a.models
                .reference()
                .unwrap_err()
                .to_string()
                .contains("--reference")
        );
    }

    #[test]
    fn download_and_completions_do_not_demand_a_reference() {
        // The other half of the same trade: `-r` being optional to clap is
        // what lets these two parse at all, and that is the property the
        // check above pays for.
        assert!(Bin::try_parse_from(["seedvc", "download"]).is_ok());
    }

    /// There is no `train`, and its absence is the engine's defining property
    /// rather than a gap — a 1-30 s reference clip is the whole speaker
    /// specification. Pinned so somebody adding one has to delete a test that
    /// says why.
    #[test]
    fn there_is_no_train_subcommand() {
        assert!(Bin::try_parse_from(["seedvc", "train", "clips/"]).is_err());
    }
}
