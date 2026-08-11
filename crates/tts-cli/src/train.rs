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
        anyhow::ensure!(
            self.segment_frames > 0,
            "--segment-frames must be at least 1: it is the window the decoder is trained on"
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
