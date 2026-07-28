//! `tts train` — fine-tune GPT-SoVITS on a corpus.
//!
//! ```sh
//! tts train corpus/ -o models/mine --reference-language zh
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
use clap::Args;
use train_kit::Checkpoint;
use tts_train::S1Settings;

use crate::args::Lang;
use crate::backend::TtsBackend;

#[derive(Debug, Args)]
pub struct TrainArgs {
    /// Directory of `<name>.wav` + `<name>.txt` pairs.
    pub corpus: PathBuf,
    /// Where to write the fine-tuned weights (a `<path>.safetensors` family).
    #[arg(short = 'o', long, default_value = "models/tts/voice")]
    pub out: PathBuf,
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
    /// Clips per optimizer step. They are processed one at a time and the
    /// gradients accumulated, since clips differ in length.
    #[arg(short, long, default_value_t = 1)]
    pub batch_size: usize,
    #[arg(long, default_value_t = 1e-5)]
    pub lr: f64,
    /// End-of-run learning rate as a fraction of `--lr`.
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
    /// Compute backend: `auto`, `cuda`, `tch` (`libtorch`) or `wgpu`.
    #[arg(long, value_enum, default_value_t = TtsBackend::Auto)]
    pub backend: TtsBackend,
    /// Compute device: `auto`, `cpu`, `gpu`, `gpu:N`, `mps` or `vulkan`.
    #[arg(long, default_value = "auto", value_name = "DEVICE", value_parser = cli_kit::parse_device)]
    pub device: burn_kit::DeviceSpec,
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

    let settings = S1Settings {
        epochs: args.epochs,
        batch_size: args.batch_size,
        lr: args.lr,
        lr_final: args.lr_final,
        ema_frac: args.ema_frac,
        use_tui: !args.no_tui && std::io::IsTerminal::is_terminal(&std::io::stdout()),
        max_tokens: args.max_tokens,
    };
    let out = Checkpoint::new(&args.out);

    tokio::task::block_in_place(|| {
        crate::backend::train_s1(
            crate::backend::TrainInputs {
                hubert: &paths.hubert,
                s1: &paths.s1,
                s2: &paths.s2,
                prosody: prosody_dir.as_deref(),
                pairs: &pairs,
                language: args.language.into(),
                settings: &settings,
                out: &out,
            },
            args.backend,
            args.device,
        )
    })?;
    Ok(())
}
