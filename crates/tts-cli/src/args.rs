//! `tts` — speech synthesis as a Unix filter.
//!
//! Text on stdin, raw f32le mono PCM on stdout, logs on stderr:
//!
//! ```sh
//! echo "你好世界" | tts --reference clip.wav | ffplay -f f32le -ar 32000 -ac 1 -
//! ```
//!
//! One line in, one utterance out, so a script is a file of lines. `--sr`
//! resamples the output, which is what feeds the voice-conversion filter:
//!
//! ```sh
//! tts --reference clip.wav --sr 16000 < script.txt \
//!   | rvc -m voice.safetensors --model-sr 48000 > out.f32le
//! ```

use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::{Args, Subcommand, ValueEnum};
use cli_kit::CompletionsArgs;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader, BufWriter};
use tts_core::{OUTPUT_SR, SampleOptions, SynthOptions};

use crate::backend::{ModelPaths, load};
use crate::download::DownloadArgs;
use crate::train::TrainArgs;
pub use cli_kit::Backend;
pub use preprocess_kit::PreprocessArgs;

/// The whole of the `tts` command tree, defined once and worn two ways: the
/// `tts` binary flattens it at its top level, `voice` nests it under a `tts`
/// subcommand. Every engine here has the same shape — the bare invocation is
/// the stdin→stdout filter, and everything else is a subcommand.
#[derive(Debug, Args)]
pub struct TtsCli {
    #[command(flatten)]
    pub synth: TtsArgs,
    #[command(subcommand)]
    pub command: Option<TtsCommand>,
}

#[derive(Debug, Subcommand)]
pub enum TtsCommand {
    /// Fine-tune GPT-SoVITS on a corpus of audio with transcripts.
    // Boxed because it carries every knob of two training loops, and an enum is
    // as large as its biggest variant.
    Train(Box<TrainArgs>),
    /// Slice a corpus into clean per-sentence clips (dead-air removed).
    Preprocess(PreprocessArgs),
    /// Prefetch the weights synthesis needs, so the first run is offline.
    Download(DownloadArgs),
    /// Print a shell completion script (bash, zsh, fish, powershell, elvish).
    Completions(CompletionsArgs),
}

/// Which language front-end to phonemize with.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, ValueEnum)]
pub enum Lang {
    /// Mandarin Chinese.
    #[default]
    Zh,
    /// English. Intelligible but flatter than Chinese: the prosody encoder
    /// needs a per-character phoneme count and the English front-end has none
    /// to give, so it is fed zeros.
    En,
    /// Japanese. No grapheme-to-phoneme front-end yet — this errors.
    Ja,
}

impl From<Lang> for text_kit::Language {
    fn from(l: Lang) -> Self {
        match l {
            Lang::Zh => Self::Zh,
            Lang::En => Self::En,
            Lang::Ja => Self::Ja,
        }
    }
}

#[derive(Debug, Args)]
pub struct TtsArgs {
    /// Reference recording of the voice to clone (any format ffmpeg reads).
    /// A few seconds of clean speech is what the model expects.
    // Checked in `synthesize` rather than by clap: these are flattened into a
    // command that also has a `train` subcommand, and clap would demand them
    // there too.
    #[arg(short, long)]
    pub reference: Option<PathBuf>,
    /// What is said in the reference recording. Required, and not a nicety:
    /// `s1` works by continuation, so it is primed with the reference's
    /// phonemes beside the reference's audio. Without it the model is shown
    /// text and audio that disagree and stops after a token or two.
    #[arg(short = 't', long)]
    pub reference_text: Option<String>,
    /// Directory holding `chinese-hubert-base/`, an `s1*.ckpt` and an `s2G*.pth`
    /// [default: auto-downloaded from Hugging Face].
    #[arg(short = 'm', long = "models")]
    pub model_dir: Option<PathBuf>,
    /// Fine-tuned `s1` weights, overriding the base model's. `s1` carries
    /// delivery — pacing, emphasis, where a speaker breathes.
    #[arg(long, value_name = "SAFETENSORS")]
    pub s1: Option<PathBuf>,
    /// Fine-tuned `s2` weights, overriding the base model's. `s2` carries
    /// timbre, so this is the one that makes a clone sound like the speaker
    /// rather than like the reference clip. Both are written by the `train`
    /// subcommand and are independent — either, both or neither.
    #[arg(long, value_name = "SAFETENSORS")]
    pub s2: Option<PathBuf>,
    /// Directory holding the ONNX prosody encoder. Without it the model gets
    /// zero prosody features — intelligible, but flatter on Chinese.
    #[arg(long)]
    pub prosody: Option<PathBuf>,
    /// Directory the downloaded models are cached in. Shared by every engine
    /// unless `$TTS_CACHE_DIR` (or `$VOICE_CACHE_DIR`) says otherwise.
    #[arg(long, default_value_os_t = hub_kit::cache_dir_for("TTS_CACHE_DIR"))]
    pub cache_dir: PathBuf,
    /// Language of the input text.
    #[arg(short, long, value_enum, default_value_t = Lang::Zh)]
    pub language: Lang,
    /// Output sample rate. The model produces 32 kHz; anything else is
    /// resampled, which is how this feeds the voice-conversion filter at
    /// 16 kHz.
    #[arg(long, default_value_t = OUTPUT_SR)]
    pub sr: u32,
    /// Runtime: `auto`, `onnx`, `cuda`, `tch` (`libtorch`) or `wgpu`. `auto`
    /// takes an ONNX export from `--models` if there is one, else the fastest
    /// Burn backend.
    #[arg(long, value_enum, default_value_t = Backend::Auto)]
    pub backend: Backend,
    /// Compute device: `auto`, `cpu`, `gpu`, `gpu:N`, `mps` or `vulkan`.
    /// Ignored by `--backend onnx`, which uses CUDA where it is available.
    #[arg(long, default_value = "auto", value_name = "DEVICE", value_parser = cli_kit::parse_device)]
    pub device: burn_kit::DeviceSpec,
    /// Sample from the `k` highest-scoring tokens. Lower is steadier, higher is
    /// more varied.
    #[arg(long, default_value_t = 15)]
    pub top_k: usize,
    /// Below 1 sharpens the distribution, above 1 flattens it.
    #[arg(long, default_value_t = 1.0)]
    pub temperature: f32,
    /// Penalty on tokens already generated. Holds off the repetition loop that
    /// otherwise stops an utterance ever ending.
    #[arg(long, default_value_t = 1.35)]
    pub repetition_penalty: f32,
    /// Seed, so a synthesis can be repeated exactly.
    #[arg(long, default_value_t = 0)]
    pub seed: u64,
    /// Cap on generated tokens per line. At 25 Hz, 1500 is a minute.
    #[arg(long, default_value_t = 1500)]
    pub max_tokens: usize,
}

impl TtsArgs {
    /// Reject values clap's types accept but sampling or synthesis cannot use.
    pub fn verify(&self) -> Result<()> {
        anyhow::ensure!(self.sr > 0, "--sr must be positive");
        // Sampling draws from the `k` best tokens, so `k = 0` draws from nothing.
        anyhow::ensure!(self.top_k > 0, "--top-k must be at least 1");
        anyhow::ensure!(self.max_tokens > 0, "--max-tokens must be at least 1");
        // The logits are divided by it, so zero is a division and a negative
        // value inverts the distribution into picking the *least* likely token.
        anyhow::ensure!(
            self.temperature > 0.0 && self.temperature.is_finite(),
            "--temperature must be a positive, finite number (below 1 sharpens, above 1 flattens)"
        );
        anyhow::ensure!(
            self.repetition_penalty > 0.0 && self.repetition_penalty.is_finite(),
            "--repetition-penalty must be a positive, finite number (1.0 = no penalty)"
        );
        Ok(())
    }
}

pub async fn synthesize(args: TtsArgs) -> Result<()> {
    args.verify()?;

    // Before anything is fetched or loaded. Clap cannot enforce these — they are
    // flattened into a command that also has a `train` subcommand, which does not
    // want them — so this is where "required" is decided, and a missing flag
    // should cost a message rather than a model load.
    let reference = args
        .reference
        .as_ref()
        .context("a --reference recording is required")?;
    let reference_text = args
        .reference_text
        .clone()
        .context("--reference-text is required: it is what `s1` continues from")?;

    let dir = match &args.model_dir {
        Some(dir) => dir.clone(),
        None => {
            tracing::info!("resolving GPT-SoVITS weights from Hugging Face...");
            hub_kit::fetch_gptsovits(&args.cache_dir)
                .await
                .context("failed to fetch the GPT-SoVITS models")?
        }
    };
    let mut paths = hub_kit::gptsovits_paths(&dir)?;
    if let Some(s1) = &args.s1 {
        tracing::info!("fine-tuned s1: {}", s1.display());
        paths.s1 = s1.clone();
    }
    if let Some(s2) = &args.s2 {
        tracing::info!("fine-tuned s2: {}", s2.display());
        paths.s2 = s2.clone();
    }

    let prosody: Option<Box<dyn tts_core::ProsodyEncoder>> = {
        let dir = match &args.prosody {
            Some(dir) => Some(dir.clone()),
            None => match hub_kit::fetch_prosody_bert(None, &args.cache_dir).await {
                Ok(dir) => Some(dir),
                // Losing prosody costs expressiveness, not intelligibility, so
                // this is a warning rather than a failure.
                Err(e) => {
                    tracing::warn!("no prosody encoder ({e}); synthesising without it");
                    None
                }
            },
        };
        match dir {
            #[cfg(feature = "onnx")]
            Some(dir) => match tts_core::OnnxProsody::load(&dir, 1024) {
                Ok(p) => Some(Box::new(p) as Box<dyn tts_core::ProsodyEncoder>),
                Err(e) => {
                    tracing::warn!("prosody encoder failed to load ({e}); continuing without it");
                    None
                }
            },
            #[cfg(not(feature = "onnx"))]
            Some(_) => None,
            None => None,
        }
    };

    let audio = read_reference(reference).await?;
    tracing::info!(
        "reference: {} ({:.1} s)",
        reference.display(),
        audio.len() as f32 / tts_core::ANALYSIS_SR as f32
    );

    let mut model = tokio::task::block_in_place(|| {
        load(
            ModelPaths {
                dir: &dir,
                hubert: &paths.hubert,
                s1: &paths.s1,
                s2: &paths.s2,
                tuned: args.s1.is_some() || args.s2.is_some(),
            },
            prosody,
            args.backend,
            args.device,
        )
    })?;

    let opts = SynthOptions {
        language: args.language.into(),
        sample: SampleOptions {
            top_k: args.top_k,
            top_p: 1.0,
            temperature: args.temperature,
            repetition_penalty: args.repetition_penalty,
        },
        max_tokens: args.max_tokens,
        seed: args.seed,
        ..Default::default()
    };

    let mut lines = BufReader::new(tokio::io::stdin()).lines();
    let mut out = BufWriter::new(tokio::io::stdout());
    let mut spoken = 0usize;

    let reference = tokio::task::block_in_place(|| {
        model.reference(&audio, &reference_text, args.language.into())
    })
    .context("failed to analyse the reference recording")?;

    while let Some(line) = lines.next_line().await? {
        if line.trim().is_empty() {
            continue;
        }
        let pcm = tokio::task::block_in_place(|| model.say(&line, &reference, &opts))
            .with_context(|| format!("synthesising {line:?}"))?;
        tracing::info!("{:.2} s  {line}", pcm.len() as f32 / OUTPUT_SR as f32);
        let pcm = audio_kit::resample_linear(&pcm, OUTPUT_SR, args.sr);
        audio_kit::write_f32le_chunk(&mut out, &pcm)
            .await
            .context("writing stdout")?;
        // Flush eagerly so downstream players get low latency.
        out.flush().await.ok();
        spoken += 1;
    }

    out.flush().await.context("final flush")?;
    tracing::info!("{spoken} lines");
    Ok(())
}

/// Decode a reference recording to mono `f32` at the analysis rate.
async fn read_reference(path: &std::path::Path) -> Result<Vec<f32>> {
    use futures::StreamExt;

    let opts = audio_kit::DecodeOptions::new(tts_core::ANALYSIS_SR);
    let mut stream = Box::pin(audio_kit::decode_path(path.to_path_buf(), opts));
    let mut audio = Vec::new();
    while let Some(chunk) = stream.next().await {
        audio.extend_from_slice(&chunk.with_context(|| format!("decoding {}", path.display()))?);
    }
    Ok(audio)
}
