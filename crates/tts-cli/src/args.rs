//! `tts` — speech synthesis as a Unix filter.
//!
//! Text on stdin, raw f32le mono PCM on stdout, logs on stderr:
//!
//! ```sh
//! echo "你好世界" | tts --reference clip.wav | ffplay -f f32le -ar 32000 -ac 1 -
//! ```
//!
//! One line in, one utterance out, so a script is a file of lines. `--sr`
//! resamples the output, which is what feeds `rvc serve`:
//!
//! ```sh
//! tts --reference clip.wav --sr 16000 < script.txt \
//!   | rvc serve -m voice.safetensors --model-sr 48000 > out.f32le
//! ```

use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::{Args, ValueEnum};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader, BufWriter};
use tts_core::{OUTPUT_SR, SampleOptions, SynthOptions};

use crate::backend::{ModelPaths, TtsBackend, load};

/// Which language front-end to phonemize with.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, ValueEnum)]
pub enum Lang {
    /// Mandarin Chinese.
    #[default]
    Zh,
    /// English. No grapheme-to-phoneme front-end yet — this errors.
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
    // Checked in `run` rather than by clap: these are flattened into a command
    // that also has a `train` subcommand, and clap would demand them there too.
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
    /// Fine-tuned `s1` weights, overriding the base model's. This is what
    /// `tts train` writes.
    #[arg(long, value_name = "SAFETENSORS")]
    pub s1: Option<PathBuf>,
    /// Directory holding the ONNX prosody encoder. Without it the model gets
    /// zero prosody features — intelligible, but flatter on Chinese.
    #[arg(long)]
    pub prosody: Option<PathBuf>,
    /// Cache directory for downloaded assets [default: the Hugging Face cache].
    #[arg(long)]
    pub cache_dir: Option<PathBuf>,
    /// Language of the input text.
    #[arg(short, long, value_enum, default_value_t = Lang::Zh)]
    pub language: Lang,
    /// Output sample rate. The model produces 32 kHz; anything else is
    /// resampled, which is how this feeds `rvc serve` at 16 kHz.
    #[arg(long, default_value_t = OUTPUT_SR)]
    pub sr: u32,
    /// Runtime: `auto`, `onnx`, `cuda`, `tch` (`libtorch`) or `wgpu`. `auto`
    /// takes an ONNX export from `--models` if there is one, else the fastest
    /// Burn backend.
    #[arg(long, value_enum, default_value_t = TtsBackend::Auto)]
    pub backend: TtsBackend,
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

pub async fn run(args: TtsArgs) -> Result<()> {
    // Before anything is fetched or loaded. Clap cannot enforce these — they are
    // flattened into a command that also has a `train` subcommand, which does not
    // want them — so `run` is where "required" is decided, and a missing flag
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
            hub_kit::fetch_gptsovits(args.cache_dir.as_deref())
                .await
                .context("failed to fetch the GPT-SoVITS models")?
        }
    };
    let mut paths = hub_kit::gptsovits_paths(&dir)?;
    if let Some(s1) = &args.s1 {
        tracing::info!("fine-tuned s1: {}", s1.display());
        paths.s1 = s1.clone();
    }

    let prosody: Option<Box<dyn tts_core::ProsodyEncoder>> = {
        let dir = match &args.prosody {
            Some(dir) => Some(dir.clone()),
            None => match hub_kit::fetch_prosody_bert(None, args.cache_dir.as_deref()).await {
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
        write_pcm(&mut out, &pcm, args.sr).await?;
        spoken += 1;
    }

    out.flush().await.context("final flush")?;
    tracing::info!("{spoken} lines");

    // Do not unwind the models. Dropping them aborts the process with glibc's
    // "corrupted double-linked list" *after* every sample has been written —
    // harmless to the audio, fatal to the exit code, which for a filter in a
    // pipeline is the part that gets checked. It needs both an ONNX Runtime
    // session on the CUDA execution provider and a CUDA Burn backend to
    // reproduce: `--prosody` omitted exits 0, `CUDA_VISIBLE_DEVICES=` exits 0,
    // and `rvc convert` drives the same two runtimes without tripping it.
    // The audio is already flushed, so the only thing skipped here is handing
    // memory back moments before the kernel reclaims it anyway.
    std::mem::forget(model);
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

/// Write PCM to stdout, resampling if the caller asked for another rate.
async fn write_pcm<W: AsyncWriteExt + Unpin>(out: &mut W, pcm: &[f32], sr: u32) -> Result<()> {
    let pcm = if sr == OUTPUT_SR {
        pcm.to_vec()
    } else {
        // Linear resample. The synthesizer's output is already band-limited well
        // below either rate's Nyquist, so this costs nothing audible and saves a
        // dependency on a filter design.
        let ratio = OUTPUT_SR as f64 / sr as f64;
        let n = (pcm.len() as f64 / ratio) as usize;
        (0..n)
            .map(|i| {
                let x = i as f64 * ratio;
                let (a, f) = (x as usize, (x - x.floor()) as f32);
                let b = (a + 1).min(pcm.len() - 1);
                pcm[a] * (1.0 - f) + pcm[b] * f
            })
            .collect()
    };
    let bytes: Vec<u8> = pcm.iter().flat_map(|s| s.to_le_bytes()).collect();
    out.write_all(&bytes).await.context("writing stdout")?;
    Ok(())
}
