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
use crate::backend::TtsBackend;

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
    #[arg(short = 'o', long, default_value = "models/tts/voice")]
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
    /// Cache directory for downloaded assets [default: the Hugging Face cache].
    #[arg(long)]
    pub cache_dir: Option<PathBuf>,
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
    /// Compute backend: `auto`, `cuda`, `tch` (`libtorch`) or `wgpu`.
    #[arg(long, value_enum, default_value_t = TtsBackend::Auto)]
    pub backend: TtsBackend,
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

pub async fn run(args: TrainArgs) -> Result<()> {
    let dir = match &args.model_dir {
        Some(dir) => dir.clone(),
        None => hub_kit::fetch_gptsovits(args.cache_dir.as_deref())
            .await
            .context("failed to fetch the GPT-SoVITS models")?,
    };
    let paths = hub_kit::gptsovits_paths(&dir)?;
    let prosody_dir = match &args.prosody {
        Some(dir) => Some(dir.clone()),
        None => hub_kit::fetch_prosody_bert(None, args.cache_dir.as_deref())
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
        // A token is two latent frames — one cap, expressed in the units each
        // stage thinks in.
        max_frames: args.max_tokens * 2,
        save_best: !args.no_save_best,
    };

    tokio::task::block_in_place(|| {
        crate::backend::train(
            crate::backend::TrainInputs {
                hubert: &paths.hubert,
                s1: &paths.s1,
                s2: &paths.s2,
                s2d: paths.s2d.as_deref(),
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
