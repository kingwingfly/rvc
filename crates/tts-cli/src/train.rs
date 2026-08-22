//! `tts train` — fine-tune GPT-SoVITS on a corpus.
//!
//! ```sh
//! tts train corpus/ -o models/mine --stage both
//! ```
//!
//! Two stages, adapting different things. `s1` is next-token prediction over
//! semantic tokens and carries **delivery** — pacing, emphasis, where a speaker
//! breathes. `s2` is the VITS adversarial loop and carries **timbre**, which is
//! what makes a clone sound like the speaker rather than like whichever few
//! seconds of reference audio it was prompted with. They write independent
//! files, so either can be deployed without the other:
//!
//! ```sh
//! tts -r clip.wav -t "…" --s1 models/mine.s1.safetensors \
//!                        --s2 models/mine.s2.safetensors < script.txt
//! ```
//!
//! `corpus/` holds `<name>.wav` beside `<name>.txt`. Producing those transcripts
//! is what `stt` is for:
//!
//! ```sh
//! for f in corpus/*.wav
//!   ffmpeg -v quiet -i $f -f f32le -ar 16000 -ac 1 - | stt > (basename $f .wav).txt
//! end
//! ```

use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::{Args, ValueEnum};
use tts_train::{S1Settings, S2Settings};

use crate::args::Lang;
use cli_kit::Backend;

/// Which half of the model to adapt.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, ValueEnum)]
pub enum Stage {
    /// Delivery only: pacing, emphasis, breathing.
    S1,
    /// Timbre only: what the voice sounds like.
    S2,
    /// Both, preparing the corpus once and training twice.
    #[default]
    Both,
}

impl Stage {
    pub fn wants_s1(self) -> bool {
        matches!(self, Self::S1 | Self::Both)
    }

    pub fn wants_s2(self) -> bool {
        matches!(self, Self::S2 | Self::Both)
    }
}

#[derive(Debug, Args)]
pub struct TrainArgs {
    /// Directory of `<name>.wav` + `<name>.txt` pairs.
    pub corpus: PathBuf,
    /// Stem for the fine-tuned weights. Each stage appends its own name, so
    /// `-o models/mine` writes `models/mine.s1.safetensors` and
    /// `models/mine.s2.safetensors` — they are separate models, and deploying
    /// one must not imply the other.
    #[arg(short = 'o', long, default_value = "models/voice")]
    pub out: PathBuf,
    /// Which stage to train.
    #[arg(long, value_enum, default_value_t = Stage::Both)]
    pub stage: Stage,
    /// Directory holding the base models [default: auto-downloaded].
    #[arg(short = 'm', long = "models")]
    pub model_dir: Option<PathBuf>,
    /// Directory holding the ONNX prosody encoder [default: auto-downloaded].
    #[arg(long)]
    pub prosody: Option<PathBuf>,
    /// `s2` discriminator base to warm-start from [default:
    /// `<cache-dir>/pretrained/s2D2333k.pth`, downloaded on first use and
    /// reused by every run].
    /// Only `--stage s2` and `--stage both` read one.
    #[arg(long, conflicts_with = "no_pretrained")]
    pub pretrained_d: Option<PathBuf>,
    /// Train `s2`'s discriminator from scratch, downloading no base. A fresh
    /// adversary spends its early steps learning what real audio is instead of
    /// critiquing this voice, so this is rarely what you want. The `s1` and
    /// `s2` generators are warm-started regardless — fine-tuning is what they
    /// are for.
    #[arg(long)]
    pub no_pretrained: bool,

    /// Directory the downloaded models are cached in. Shared by every engine
    /// unless `$TTS_CACHE_DIR` (or `$VOICE_CACHE_DIR`) says otherwise.
    #[arg(long, default_value_os_t = hub_kit::cache_dir_for("TTS_CACHE_DIR"))]
    pub cache_dir: PathBuf,
    /// Language of the transcripts.
    #[arg(short, long, value_enum, default_value_t = Lang::Zh)]
    pub language: Lang,
    #[arg(short, long, default_value_t = 10)]
    pub epochs: u32,
    /// Clips per optimizer step, per device. They are processed one at a time
    /// and the gradients accumulated, since clips differ in length.
    #[arg(short, long, default_value_t = 1)]
    pub batch_size: usize,
    /// Learning rate for `s1`.
    #[arg(long, default_value_t = 1e-5)]
    pub lr: f64,
    /// Learning rate for `s2`. Separate from `--lr` because the stages are
    /// different objectives: `s1` is cross-entropy on a large transformer and
    /// wants a small rate, `s2` is a GAN warm-started from a converged base.
    #[arg(long, default_value_t = 1e-4)]
    pub s2_lr: f64,
    /// End-of-run learning rate as a fraction of the starting one.
    #[arg(long, default_value_t = 0.1)]
    pub lr_final: f64,
    /// EMA window as a fraction of the run; the saved model is the EMA.
    /// `0` saves the raw weights.
    #[arg(long, default_value_t = 0.1)]
    pub ema_frac: f64,
    /// Skip clips longer than this many semantic tokens (25 per second).
    /// Skipped rather than truncated: a cut-off clip teaches an early stop.
    #[arg(long, default_value_t = 1500)]
    pub max_tokens: usize,
    /// Latent frames `s2`'s decoder renders per step (50 per second). The
    /// adversarial losses are local, so this trades VRAM against little else.
    #[arg(long, default_value_t = 32)]
    pub segment_frames: usize,
    /// Skip clips longer than this many latent frames (50 per second), which is
    /// `s2`'s half of `--max-tokens`. Defaults to twice it, since a semantic
    /// token is two latent frames.
    ///
    /// **It is the cap that keeps a small card alive, and that is why it is
    /// separate.** `enc_q` and the flow run over the *whole* utterance, so `s2`'s
    /// memory grows with the longest clip rather than with the batch — where
    /// `--max-tokens` only bounds `s1`'s sequence. While this was derived,
    /// raising `--max-tokens` to let `s1` see longer lines silently doubled
    /// `s2`'s peak VRAM as well.
    #[arg(long)]
    pub max_frames: Option<usize>,
    /// Discriminator learning rate as a multiple of the generator's. Below 1
    /// holds off a discriminator that is winning.
    #[arg(long, default_value_t = 1.0)]
    pub d_lr_ratio: f64,
    /// Update the discriminator every N steps — the coarser version of the same
    /// lever as `--d-lr-ratio`.
    #[arg(long, default_value_t = 1)]
    pub d_interval: usize,
    /// Do not keep a best-so-far `s2` checkpoint beside the final weights.
    #[arg(long)]
    pub no_save_best: bool,
    /// Overwrite weights already at `-o` instead of refusing to start.
    #[arg(short = 'y', long)]
    pub yes: bool,
    /// Compute backend. All three Burn backends train, and the saved weights are
    /// the same whichever you pick; `onnx` cannot train at all.
    #[arg(long, value_enum, default_value_t = Backend::Auto)]
    pub backend: Backend,
    /// Compute device(s): `auto`, `cpu`, `gpu`, `gpu:N`, `mps` or `vulkan`.
    /// Comma-separate for data-parallel training — the first is the master.
    #[arg(
        long,
        alias = "devices",
        default_value = "auto",
        value_name = "DEVICE",
        value_delimiter = ',',
        value_parser = cli_kit::parse_device
    )]
    pub device: Vec<burn_kit::DeviceSpec>,
    /// Disable the TUI dashboard and log to stderr.
    #[arg(long)]
    pub no_tui: bool,
}

impl TrainArgs {
    /// `--max-frames`, or the derivation it defaults to.
    ///
    /// A method rather than a computed clap default, because the derivation
    /// reads another flag: clap resolves defaults per argument and cannot see
    /// `--max-tokens` while building `--max-frames`. The cost is that `-h`
    /// prints no default for it, which the help text states in words instead.
    pub fn max_frames(&self) -> usize {
        self.max_frames.unwrap_or(self.max_tokens * 2)
    }

    /// Reject a fine-tuning configuration that cannot converge, or cannot start.
    ///
    /// Checked before the corpus is prepared and before a base is downloaded, for
    /// the same reason as the overwrite guard: a mistyped learning rate should
    /// cost a message, not an hour of GPU and a model full of `NaN`.
    pub fn verify(&self) -> Result<()> {
        anyhow::ensure!(self.epochs > 0, "--epochs must be at least 1");
        anyhow::ensure!(self.batch_size > 0, "--batch-size must be at least 1");
        anyhow::ensure!(self.d_interval > 0, "--d-interval must be at least 1");
        anyhow::ensure!(self.max_tokens > 0, "--max-tokens must be at least 1");
        anyhow::ensure!(
            self.max_frames.is_none_or(|f| f > 0),
            "--max-frames must be at least 1"
        );
        anyhow::ensure!(
            self.max_frames() >= self.segment_frames,
            "--max-frames ({}) is below --segment-frames ({}), so every clip would be \
             skipped for being too long to render one step from",
            self.max_frames(),
            self.segment_frames
        );
        // Two, not one, and the mechanism is the mel front end: `center=False`
        // costs a reflect pad wider than a single frame, and `reflect_pad`
        // slices `(l - 1 - p)..(l - 1)` unsigned — so `--segment-frames 1`
        // underflows that subtraction and panics inside the mel loss instead of
        // being refused here. `rvc train` carries the same floor for the same
        // reason, and both ask the spectral config for it rather than spelling
        // a number the config decides.
        let floor = tts_train::mel_min_frames();
        anyhow::ensure!(
            self.segment_frames >= floor,
            "--segment-frames must be at least {floor}: it is the window the decoder is \
             trained on, and below that the mel front end has too little audio to \
             reflect-pad"
        );
        anyhow::ensure!(
            self.lr > 0.0 && self.lr.is_finite(),
            "--lr must be a positive, finite number"
        );
        anyhow::ensure!(
            self.s2_lr > 0.0 && self.s2_lr.is_finite(),
            "--s2-lr must be a positive, finite number"
        );
        anyhow::ensure!(
            self.d_lr_ratio > 0.0 && self.d_lr_ratio.is_finite(),
            "--d-lr-ratio must be a positive, finite number (1.0 = same LR as the generator)"
        );
        anyhow::ensure!(
            self.lr_final > 0.0 && self.lr_final <= 1.0,
            "--lr-final is a fraction of --lr and must be in (0, 1], not {}: at or \
             below 0 the run would end with no learning rate at all, and above 1 it \
             would end faster than it started",
            self.lr_final
        );
        anyhow::ensure!(
            (0.0..1.0).contains(&self.ema_frac),
            "--ema-frac is a fraction of the run and must be in [0, 1); 0 saves the \
             raw weights"
        );
        anyhow::ensure!(!self.device.is_empty(), "--device names no device");
        // Only `s2` has a discriminator, so a `--pretrained-d` under `--stage s1`
        // would be fetched, or read from disk, and then never opened. Saying so
        // is better than obeying a flag that cannot do anything.
        anyhow::ensure!(
            self.pretrained_d.is_none() || self.stage.wants_s2(),
            "--pretrained-d is an `s2` discriminator, which `--stage s1` never trains \
             (use `--stage s2` or `--stage both`)"
        );
        Ok(())
    }
}

pub async fn run(args: TrainArgs) -> Result<()> {
    args.verify()?;

    // Before the base models are fetched and the corpus is encoded — preparation
    // is the expensive half of a fine-tune, and discovering the collision after
    // it has already spent what the check exists to save. Only the stages that
    // will actually run are checked, and only the members they write: `s1` has
    // no discriminator, and `--no-save-best` drops the `checkpoint/` family.
    let ema = args.ema_frac > 0.0;
    let mut planned = Vec::new();
    if args.stage.wants_s1() {
        planned.extend(crate::backend::checkpoint(&args.out, "s1").members(ema, false));
    }
    if args.stage.wants_s2() {
        let s2 = crate::backend::checkpoint(&args.out, "s2");
        planned.extend(s2.members(ema, true));
        if !args.no_save_best {
            planned.extend(s2.best().members(ema, true));
        }
    }
    train_kit::ensure_absent(planned, args.yes)?;

    let dir = match &args.model_dir {
        Some(dir) => dir.clone(),
        None => hub_kit::fetch_gptsovits(&args.cache_dir)
            .await
            .context("failed to fetch the GPT-SoVITS models")?,
    };
    let paths = hub_kit::gptsovits_paths(&dir)?;

    // An explicit path wins, a copy already sitting in the model directory is
    // used as-is rather than downloaded again, and `--no-pretrained` (or a run
    // that never reaches `s2`) fetches nothing.
    let s2d = match (&args.pretrained_d, args.stage.wants_s2() && !args.no_pretrained) {
        (Some(p), _) => Some(p.clone()),
        (None, false) => None,
        (None, true) => match paths.s2d.clone() {
            Some(p) => Some(p),
            None => Some(
                hub_kit::fetch_pretrained(
                    &hub_kit::default_gptsovits_s2d(),
                    &hub_kit::pretrained_dir(&args.cache_dir),
                )
                .await
                .context("failed to fetch the s2 discriminator base (override with --pretrained-d, or pass --no-pretrained to train it from scratch)")?,
            ),
        },
    };
    let prosody_dir = match &args.prosody {
        Some(dir) => Some(dir.clone()),
        None => hub_kit::fetch_prosody_bert(None, &args.cache_dir)
            .await
            .ok(),
    };

    let pairs = tts_train::pairs(&args.corpus)?;
    tracing::info!("{} clips in {}", pairs.len(), args.corpus.display());

    let stop = cli_kit::stop_on_ctrl_c();
    let use_tui = cli_kit::use_tui(args.no_tui);
    let s1 = S1Settings {
        epochs: args.epochs,
        batch_size: args.batch_size,
        lr: args.lr,
        lr_final: args.lr_final,
        ema_frac: args.ema_frac,
        use_tui,
        max_tokens: args.max_tokens,
    };
    let s2 = S2Settings {
        epochs: args.epochs,
        batch_size: args.batch_size,
        lr: args.s2_lr,
        lr_final: args.lr_final,
        ema_frac: args.ema_frac,
        use_tui,
        d_lr_ratio: args.d_lr_ratio,
        d_interval: args.d_interval,
        segment_frames: args.segment_frames,
        // A token is two latent frames, so the derivation is the honest default
        // — but it stays overridable, because the two caps bound different
        // things (see `--max-frames`).
        max_frames: args.max_frames(),
        save_best: !args.no_save_best,
    };

    tokio::task::block_in_place(|| {
        crate::backend::train(
            crate::backend::TrainInputs {
                hubert: &paths.hubert,
                s1: &paths.s1,
                s2: &paths.s2,
                s2d: s2d.as_deref(),
                prosody: prosody_dir.as_deref(),
                pairs: &pairs,
                language: args.language.into(),
                s1_settings: &s1,
                s2_settings: &s2,
                out: &args.out,
                stage: args.stage,
                stop: &stop,
            },
            args.backend,
            &args.device,
        )
    })?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::args::{TtsCli, TtsCommand};
    use clap::{Parser, Subcommand};

    /// The `tts` binary's own tree, minus `completions` — which belongs to the
    /// executable rather than to the engine.
    #[derive(Parser)]
    #[command(name = "tts")]
    struct Cli {
        #[command(flatten)]
        tts: TtsCli,
    }

    /// `voice`'s tree, so every flag below is exercised in the *nested*
    /// position too. That is not symmetry for its own sake: a `*-cli` crate
    /// exports an `Args` type that somebody else's `Command` hosts, so any
    /// annotation placed on the command rather than on the argument is left
    /// behind exactly here, and only here.
    #[derive(Parser)]
    #[command(name = "voice")]
    struct Nested {
        #[command(subcommand)]
        command: NestedCommand,
    }

    #[derive(Subcommand)]
    enum NestedCommand {
        Tts(TtsCli),
    }

    fn try_train(extra: &[&str]) -> Result<TrainArgs, clap::Error> {
        let mut argv = vec!["tts", "train", "corpus/"];
        argv.extend_from_slice(extra);
        Ok(match Cli::try_parse_from(&argv)?.tts.command {
            Some(TtsCommand::Train(a)) => *a,
            other => panic!("expected `train`, got {other:?}"),
        })
    }

    fn train(extra: &[&str]) -> TrainArgs {
        try_train(extra).unwrap_or_else(|e| panic!("{extra:?} rejected: {e}"))
    }

    fn nested_train(extra: &[&str]) -> TrainArgs {
        let mut argv = vec!["voice", "tts", "train", "corpus/"];
        argv.extend_from_slice(extra);
        let NestedCommand::Tts(cli) = Nested::try_parse_from(&argv)
            .unwrap_or_else(|e| panic!("{argv:?} rejected under voice: {e}"))
            .command;
        match cli.command {
            Some(TtsCommand::Train(a)) => *a,
            other => panic!("expected `train`, got {other:?}"),
        }
    }

    /// The mirror of `rvc train`'s floor, and the same defect: one 640-sample
    /// frame is under the 704 the 32 kHz mel front end reflect-pads by, so
    /// `--segment-frames 1` panicked inside the mel loss — after the corpus had
    /// been prepared, which is the expensive half of this loop.
    #[test]
    fn a_segment_shorter_than_the_mel_pad_is_refused() {
        let floor = tts_train::mel_min_frames();
        let err = train(&["--segment-frames", "1"])
            .verify()
            .expect_err("one frame is under the reflect pad")
            .to_string();
        assert!(err.contains(&floor.to_string()), "{err}");
        train(&["--segment-frames", &floor.to_string()])
            .verify()
            .expect("the floor itself must be accepted");
    }

    #[test]
    fn the_floor_is_asked_for_rather_than_written_down() {
        // The point of routing it through `tts_train::mel_min_frames` is that an
        // `n_fft` or `hop` change moves the number with no CLI edit. A literal
        // here would go stale in silence, so the check compares the refusal
        // against the config's own answer rather than against a `2`.
        let floor = tts_train::mel_min_frames();
        assert!(floor >= 1, "a floor of zero would guard nothing");
        // Zero frames renders nothing at all, and is refused by the same rule
        // rather than by a separate one.
        train(&["--segment-frames", "0"])
            .verify()
            .expect_err("zero frames is not a segment");
        // One above the floor is ordinary and must stay accepted, so the guard
        // cannot quietly become a wider one.
        train(&["--segment-frames", &(floor + 1).to_string()])
            .verify()
            .expect("above the floor is ordinary");
    }

    #[test]
    fn the_floor_holds_nested_under_voice_too() {
        let floor = tts_train::mel_min_frames();
        nested_train(&["--segment-frames", "1"])
            .verify()
            .expect_err("`voice tts train` must refuse exactly what `tts train` refuses");
        nested_train(&["--segment-frames", &floor.to_string()])
            .verify()
            .expect("and accept exactly what it accepts");
    }

    // ---- the negative-value flag class -------------------------------------
    //
    // Clap reads a leading `-` as the start of a short-flag cluster unless the
    // argument opted out with `allow_negative_numbers`, so `--lr -1` fails with
    // `unexpected argument '-1'`. The mechanical question for whether a flag
    // needs that opt-out is not what its help text shows but **does `verify`
    // accept a value below zero** — and for every flag in this file the answer
    // is no, which is why `tts-cli` carries no such annotation. The tests below
    // are that answer, run rather than asserted: each flag is given a negative
    // in the `=` form, which parses whatever the annotation says, and `verify`
    // has to be the thing that refuses it.

    #[test]
    fn no_train_flag_accepts_a_value_below_zero() {
        for flag in [
            "--lr=-1",
            "--s2-lr=-1",
            "--lr-final=-0.5",
            "--ema-frac=-0.1",
            "--d-lr-ratio=-1",
        ] {
            assert!(
                train(&[flag]).verify().is_err(),
                "{flag} was accepted, so a negative value reached the trainer"
            );
        }
    }

    /// The counts are unsigned, so clap refuses a negative before `verify` is
    /// ever reached — and that refusal is worth pinning as well, because
    /// widening one of them to accept `-1` would be a silent change in what a
    /// rate or an epoch count may be.
    #[test]
    fn the_unsigned_train_flags_refuse_a_negative_at_the_parser() {
        for flag in [
            "--epochs=-1",
            "--batch-size=-1",
            "--max-tokens=-1",
            "--max-frames=-1",
            "--segment-frames=-1",
            "--d-interval=-1",
        ] {
            assert!(
                try_train(&[flag]).is_err(),
                "{flag} parsed, so a negative count reached the trainer"
            );
        }
    }

    #[test]
    fn a_negative_split_from_its_flag_is_refused_rather_than_misread() {
        // The form the `preprocess` tests exist for, checked here as the
        // *negative* control: no flag in this file documents a negative value,
        // so `--lr -1` must fail at the parser rather than quietly becoming
        // `--lr` with a missing value or a cluster of short flags.
        assert!(try_train(&["--lr", "-1"]).is_err());
        assert!(try_train(&["--ema-frac", "-0.1"]).is_err());
    }

    // ---- the two caps, and which one bounds what ---------------------------

    #[test]
    fn the_frame_cap_is_two_per_token_unless_it_is_given() {
        // A semantic token is two latent frames, and the derivation is a method
        // rather than a clap default because clap resolves each argument alone
        // and cannot read `--max-tokens` while building `--max-frames`.
        assert_eq!(train(&["--max-tokens", "100"]).max_frames(), 200);
        assert_eq!(
            train(&["--max-tokens", "100", "--max-frames", "50"]).max_frames(),
            50
        );
        // The default pairing, so the two flags cannot drift apart in silence.
        let d = train(&[]);
        assert_eq!(d.max_frames(), d.max_tokens * 2);
    }

    #[test]
    fn a_frame_cap_below_the_segment_skips_every_clip_and_is_refused() {
        // Both bounds are compared against `frames()`, so a cap under the
        // segment length leaves `usable` empty — which `s2` would report only
        // after the corpus had been prepared.
        let err = train(&["--max-frames", "8", "--segment-frames", "32"])
            .verify()
            .expect_err("no clip could satisfy both")
            .to_string();
        assert!(err.contains("--max-frames"), "{err}");
        assert!(err.contains("--segment-frames"), "{err}");
    }

    // ---- --stage, and what belongs to which half ---------------------------

    #[test]
    fn the_stages_are_what_they_say_and_both_is_the_default() {
        assert_eq!(train(&[]).stage, Stage::Both);
        assert_eq!(nested_train(&[]).stage, Stage::Both);
        assert!(Stage::S1.wants_s1() && !Stage::S1.wants_s2());
        assert!(!Stage::S2.wants_s1() && Stage::S2.wants_s2());
        assert!(Stage::Both.wants_s1() && Stage::Both.wants_s2());
    }

    #[test]
    fn a_discriminator_base_is_refused_under_the_stage_that_has_no_adversary() {
        // `s1` is plain cross-entropy, so a `--pretrained-d` there would be
        // fetched — 94 MB — and then never opened. Saying so beats obeying a
        // flag that cannot do anything.
        let err = train(&["--stage", "s1", "--pretrained-d", "s2D.pth"])
            .verify()
            .expect_err("`s1` has no discriminator to warm-start")
            .to_string();
        assert!(err.contains("--pretrained-d"), "{err}");
        train(&["--stage", "s2", "--pretrained-d", "s2D.pth"])
            .verify()
            .expect("`s2` is exactly where one belongs");
        train(&["--stage", "both", "--pretrained-d", "s2D.pth"])
            .verify()
            .expect("and `both` reaches `s2`");
    }

    #[test]
    fn the_two_pretrained_flags_cannot_be_given_together() {
        // "Warm-start from this file" and "warm-start from nothing" are the two
        // answers to one question, and clap is where that is settled.
        assert!(try_train(&["--pretrained-d", "s2D.pth", "--no-pretrained"]).is_err());
    }

    // ---- each stage's output family ----------------------------------------

    #[test]
    fn each_stage_writes_a_family_of_its_own() {
        // `--stage both` runs two loops over one prepared corpus, and they must
        // not land on each other: `-o models/mine` is a *stem*, and the stage
        // name is appended to it rather than substituted into it.
        for out in ["models/mine", "models/voice.v2", "mine"] {
            let out = std::path::Path::new(out);
            let s1 = crate::backend::checkpoint(out, "s1");
            let s2 = crate::backend::checkpoint(out, "s2");
            let members: Vec<_> = s1
                .members(true, false)
                .into_iter()
                .chain(s2.members(true, true))
                .chain(s2.best().members(true, true))
                .collect();
            let unique: std::collections::HashSet<_> = members.iter().collect();
            assert_eq!(
                unique.len(),
                members.len(),
                "two members of {out:?} share a path: {members:?}"
            );
            // The trap the appending exists to avoid: `with_extension` sees only
            // the last dot, so `voice.v2` would become `voice.s1` and merge two
            // runs' outputs.
            assert!(
                s1.generator().to_string_lossy().contains(".s1."),
                "{:?}",
                s1.generator()
            );
        }
        let dotted = crate::backend::checkpoint(std::path::Path::new("models/voice.v2"), "s1");
        assert_eq!(
            dotted.generator(),
            std::path::Path::new("models/voice.v2.s1.safetensors")
        );
    }

    #[test]
    fn an_s1_family_has_no_discriminator_and_an_s2_family_does() {
        // What `run` hands `ensure_absent` has to be what the loop actually
        // writes: over-counting refuses a run that would have collided with
        // nothing, and under-counting is the overwrite the guard exists for.
        let out = std::path::Path::new("models/mine");
        let s1 = crate::backend::checkpoint(out, "s1").members(true, false);
        assert!(
            !s1.iter().any(|p| p.to_string_lossy().contains(".disc.")),
            "`s1` is one model and one loss; it saves no adversary: {s1:?}"
        );
        let s2 = crate::backend::checkpoint(out, "s2").members(true, true);
        assert!(s2.iter().any(|p| p.to_string_lossy().contains(".disc.")));
        // No EMA means no raw twin, because the live weights are themselves the
        // deployable ones.
        let flat = crate::backend::checkpoint(out, "s1").members(false, false);
        assert_eq!(flat.len(), 1, "{flat:?}");
    }

    #[test]
    fn a_run_refuses_to_overwrite_weights_already_at_its_output() {
        // Hours of GPU and a corpus that may be gone, so a second `-o` at the
        // same stem is far more often a mistake than an intent. `-y` is the
        // only way past it.
        let dir = std::env::temp_dir().join("tts-cli-overwrite-guard");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("temp dir");
        let out = dir.join("mine");
        let planned = crate::backend::checkpoint(&out, "s2").members(true, true);
        train_kit::ensure_absent(planned.clone(), false)
            .expect("an empty directory is not a collision");
        std::fs::write(&planned[0], b"an earlier run").expect("write");
        let err = train_kit::ensure_absent(planned.clone(), false)
            .expect_err("the weights are already there")
            .to_string();
        assert!(err.contains("mine.s2"), "{err}");
        train_kit::ensure_absent(planned, true).expect("-y is the way past it");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
