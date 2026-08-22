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
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader, BufWriter};
use tts_core::{OUTPUT_SR, SampleOptions, SynthOptions};

use crate::backend::{ModelPaths, load};
use crate::convert::ConvertArgs;
use crate::download::DownloadArgs;
use crate::train::TrainArgs;
pub use cli_kit::Backend;

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

/// What this engine can do, and nothing about the executable that hosts it.
///
/// **`completions` is not a member, on purpose.** A completion script describes
/// one binary, so a nested `voice tts completions` could only ever emit
/// `voice`'s — which is exactly what it used to do. Each `main.rs` flattens this
/// enum into its own and adds `Completions` beside it, so the standalone binary
/// keeps the subcommand and `voice` grows only one, at its top level.
#[derive(Debug, Subcommand)]
pub enum TtsCommand {
    /// Speak text files, one WAV per file.
    // Boxed because it carries every synthesis knob on top of its own two, and
    // an enum is as large as its biggest variant.
    Convert(Box<ConvertArgs>),
    /// Fine-tune GPT-SoVITS on a corpus of audio with transcripts.
    // Boxed because it carries every knob of two training loops, and an enum is
    // as large as its biggest variant.
    Train(Box<TrainArgs>),
    /// Prefetch the weights synthesis needs, so the first run is offline.
    Download(DownloadArgs),
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
    /// Japanese. Fetches a 28.7 MB dictionary on first use, into the cache.
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
    // Every default below is read from `SampleOptions`/`SynthOptions`'s own
    // `Default`, never spelled again here. Two of these knobs existed as fields
    // the CLI pinned to a literal, and a third had a literal that merely
    // *happened* to match — the class of drift that only shows up as a model
    // behaving unlike its documentation.
    /// Sample from the `k` highest-scoring tokens. Lower is steadier, higher is
    /// more varied.
    #[arg(long, default_value_t = SampleOptions::default().top_k)]
    pub top_k: usize,
    /// Keep the smallest set of tokens whose probability sums past this;
    /// `1.0` disables it. Applied after `--top-k`, so the two compose: `--top-k`
    /// bounds the count and this bounds the mass, and whichever bites first
    /// wins.
    #[arg(long, default_value_t = SampleOptions::default().top_p)]
    pub top_p: f32,
    /// Below 1 sharpens the distribution, above 1 flattens it.
    #[arg(long, default_value_t = SampleOptions::default().temperature)]
    pub temperature: f32,
    /// Penalty on tokens already generated. Holds off the repetition loop that
    /// otherwise stops an utterance ever ending.
    #[arg(long, default_value_t = SampleOptions::default().repetition_penalty)]
    pub repetition_penalty: f32,
    /// Seed, so a synthesis can be repeated exactly.
    #[arg(long, default_value_t = SynthOptions::default().seed)]
    pub seed: u64,
    /// Cap on generated tokens per line. At 25 Hz, 1500 is a minute.
    #[arg(long, default_value_t = SynthOptions::default().max_tokens)]
    pub max_tokens: usize,
    /// How much of `s2`'s prior variance to sample. Upstream uses 0.5; lower is
    /// flatter and more repeatable, higher is more varied and more prone to
    /// artefacts. This is the decoder's randomness, not the token sampler's —
    /// `--seed` fixes both, so two runs at one seed match whatever this is.
    #[arg(long, default_value_t = SynthOptions::default().noise_scale)]
    pub noise_scale: f64,
}

impl TtsArgs {
    /// Reject values clap's types accept but sampling or synthesis cannot use.
    pub fn verify(&self) -> Result<()> {
        anyhow::ensure!(self.sr > 0, "--sr must be positive");
        // Sampling draws from the `k` best tokens, so `k = 0` draws from nothing.
        anyhow::ensure!(self.top_k > 0, "--top-k must be at least 1");
        anyhow::ensure!(
            self.top_p > 0.0 && self.top_p <= 1.0,
            "--top-p is a probability mass, so it must be in (0, 1]: {} would keep no tokens \
             at all (1.0 disables the cut)",
            self.top_p
        );
        anyhow::ensure!(
            self.noise_scale >= 0.0 && self.noise_scale.is_finite(),
            "--noise-scale must be a non-negative, finite number (0 is a deterministic decode)"
        );
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

    /// The sampling settings these knobs describe.
    pub fn options(&self) -> SynthOptions {
        SynthOptions {
            language: self.language.into(),
            sample: SampleOptions {
                top_k: self.top_k,
                top_p: self.top_p,
                temperature: self.temperature,
                repetition_penalty: self.repetition_penalty,
            },
            max_tokens: self.max_tokens,
            seed: self.seed,
            noise_scale: self.noise_scale,
        }
        // Every field is named, and the `..Default::default()` that used to
        // close this is deliberately gone. It read as "the rest are defaults"
        // and meant "the rest are unreachable": `noise_scale` was pinned at 0.5
        // by it, and a field added to `SynthOptions` later would have been
        // pinned the same way with nothing to notice it. Spelling all of them
        // makes the next addition a compile error here, which is where the
        // decision about exposing it belongs.
    }
}

/// Resolve the weights — downloading whatever `--models`, `--s1`, `--s2` and
/// `--prosody` did not name — and load them onto the chosen backend.
///
/// Shared with `convert`, which loads the same model once and then writes files
/// instead of stdout: which checkpoint a flag overrides, and what happens when
/// the prosody encoder is missing, must not be able to differ between the two.
pub async fn load_models(args: &TtsArgs) -> Result<tts_core::Synthesizer> {
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

    // Fetched only when the run is actually Japanese: the NAIST-JDic archive is
    // 28.7 MB, and a Chinese-or-English user must never pay for it. This is the
    // first place that knows both the language and the cache directory, which is
    // why the dictionary is opened here rather than inside `text-kit`.
    let japanese = match args.language {
        Lang::Ja => {
            tracing::info!("resolving the Japanese dictionary...");
            let dir = hub_kit::fetch_naist_jdic(&args.cache_dir)
                .await
                .context("failed to fetch the Japanese (NAIST-JDic) dictionary")?;
            Some(text_kit::JapaneseDict::open(&dir).with_context(|| {
                format!(
                    "failed to open the Japanese dictionary at {}",
                    dir.display()
                )
            })?)
        }
        _ => None,
    };

    tokio::task::block_in_place(|| {
        load(
            ModelPaths {
                dir: &dir,
                hubert: &paths.hubert,
                s1: &paths.s1,
                s2: &paths.s2,
                tuned: args.s1.is_some() || args.s2.is_some(),
            },
            prosody,
            japanese,
            args.backend,
            args.device,
        )
    })
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

    let audio = read_reference(reference).await?;
    tracing::info!(
        "reference: {} ({:.1} s)",
        reference.display(),
        audio.len() as f32 / tts_core::ANALYSIS_SR as f32
    );

    let mut model = load_models(&args).await?;
    let opts = args.options();

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
pub(crate) async fn read_reference(path: &std::path::Path) -> Result<Vec<f32>> {
    use futures::StreamExt;

    let opts = audio_kit::DecodeOptions::new(tts_core::ANALYSIS_SR);
    let mut stream = Box::pin(audio_kit::decode_path(path.to_path_buf(), opts));
    let mut audio = Vec::new();
    while let Some(chunk) = stream.next().await {
        audio.extend_from_slice(&chunk.with_context(|| format!("decoding {}", path.display()))?);
    }
    Ok(audio)
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::{Parser, Subcommand};

    /// The `tts` binary's own tree, minus `completions` — which belongs to the
    /// executable rather than to the engine, and is therefore not part of what
    /// [`TtsCli`] can be asked for.
    #[derive(Parser)]
    #[command(name = "tts")]
    struct Cli {
        #[command(flatten)]
        tts: TtsCli,
    }

    /// `voice`'s tree, hosting the very same type.
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

    fn filter(extra: &[&str]) -> TtsArgs {
        let mut argv = vec!["tts"];
        argv.extend_from_slice(extra);
        Cli::try_parse_from(&argv)
            .unwrap_or_else(|e| panic!("{argv:?} rejected: {e}"))
            .tts
            .synth
    }

    #[test]
    fn every_sampling_knob_reaches_the_options() {
        // Each value is deliberately *not* its default, so a field that is
        // hard-coded or dropped on the way into `SynthOptions` fails here
        // rather than coincidentally agreeing.
        let opts = filter(&[
            "--top-k",
            "7",
            "--top-p",
            "0.8",
            "--temperature",
            "0.6",
            "--repetition-penalty",
            "1.1",
            "--seed",
            "42",
            "--max-tokens",
            "300",
            "--noise-scale",
            "0.25",
            "--language",
            "en",
        ])
        .options();
        assert_eq!(opts.sample.top_k, 7);
        assert_eq!(opts.sample.top_p, 0.8);
        assert_eq!(opts.sample.temperature, 0.6);
        assert_eq!(opts.sample.repetition_penalty, 1.1);
        assert_eq!(opts.seed, 42);
        assert_eq!(opts.max_tokens, 300);
        assert_eq!(opts.noise_scale, 0.25);
        assert_eq!(opts.language, text_kit::Language::En);
    }

    #[test]
    fn the_defaults_are_the_ones_the_engine_would_have_picked() {
        // Every default in this file is read from `SampleOptions`/`SynthOptions`
        // rather than spelled again, and this is what keeps that true: two of
        // these knobs once existed as fields the CLI pinned to a literal, and a
        // third had a literal that merely *happened* to match. Compared field by
        // field because `SynthOptions` carries no `PartialEq`, which also makes
        // a field added there and forgotten here a compile error.
        let (got, want) = (filter(&[]).options(), tts_core::SynthOptions::default());
        assert_eq!(got.language, want.language);
        assert_eq!(got.sample.top_k, want.sample.top_k);
        assert_eq!(got.sample.top_p, want.sample.top_p);
        assert_eq!(got.sample.temperature, want.sample.temperature);
        assert_eq!(
            got.sample.repetition_penalty,
            want.sample.repetition_penalty
        );
        assert_eq!(got.max_tokens, want.max_tokens);
        assert_eq!(got.seed, want.seed);
        assert_eq!(got.noise_scale, want.noise_scale);
    }

    #[test]
    fn no_sampling_knob_accepts_a_value_below_zero() {
        // The mechanical question for whether a flag needs
        // `allow_negative_numbers` is not what its help prints but **does
        // `verify` accept a value below zero** — so it is asked by running
        // `verify`, in the `=` form that parses whatever the annotation says.
        // Every answer here is no, which is why this crate carries no such
        // annotation.
        for flag in [
            "--temperature=-1",
            "--top-p=-0.5",
            "--repetition-penalty=-1",
            "--noise-scale=-0.1",
        ] {
            assert!(
                filter(&[flag]).verify().is_err(),
                "{flag} was accepted, so a negative value reached the sampler"
            );
        }
        // And the unsigned ones are refused a step earlier, by the parser.
        for flag in ["--top-k=-1", "--max-tokens=-1", "--sr=-1", "--seed=-1"] {
            assert!(
                Cli::try_parse_from(["tts", flag]).is_err(),
                "{flag} parsed, so a negative count reached the sampler"
            );
        }
    }

    #[test]
    fn the_boundaries_verify_names_are_the_ones_it_enforces() {
        // Zero is a division by zero in the temperature, an empty candidate set
        // in `--top-k`, and no tokens at all in `--top-p` — each rejected with
        // its own message rather than by one blanket rule.
        for flag in [
            "--top-k=0",
            "--top-p=0",
            "--top-p=1.5",
            "--temperature=0",
            "--repetition-penalty=0",
            "--max-tokens=0",
            "--sr=0",
        ] {
            assert!(filter(&[flag]).verify().is_err(), "{flag} was accepted");
        }
        // The two values that look illegal and are not: `--top-p 1.0` disables
        // the cut, and `--noise-scale 0` is a deterministic decode.
        filter(&["--top-p=1.0"]).verify().expect("1.0 disables it");
        filter(&["--noise-scale=0"])
            .verify()
            .expect("0 is a deterministic decode, not an error");
    }

    #[test]
    fn the_reference_and_its_transcript_are_optional_to_clap_and_required_to_convert() {
        // They are `Option` because `TtsArgs` is flattened into a command that
        // also carries `train`, which wants neither — so the bare filter checks
        // them itself. Nothing is flattened beside `convert`, so there they are
        // what they really are: a clap usage error, reported before a model has
        // loaded.
        Cli::try_parse_from(["tts", "train", "corpus/"])
            .expect("`train` must not be made to supply a reference clip");
        assert!(filter(&[]).reference.is_none());
        assert!(filter(&[]).reference_text.is_none());

        assert!(
            Cli::try_parse_from(["tts", "convert", "-r", "clip.wav", "in.txt"]).is_err(),
            "`convert` without --reference-text must be a usage error: `s1` \
             generates by continuation, and a missing transcript stops it after \
             a token or two"
        );
        assert!(
            Cli::try_parse_from(["tts", "convert", "-t", "hello", "in.txt"]).is_err(),
            "`convert` without --reference must be a usage error"
        );
        Cli::try_parse_from(["tts", "convert", "-r", "clip.wav", "-t", "hello", "in.txt"])
            .expect("both together is the whole requirement");
    }

    #[test]
    fn the_engine_nests_under_voice_and_is_not_renamed() {
        // `voice tts train`, not `voice tts-train`: `voice` hosts this crate's
        // clap type unchanged, so a subcommand added here appears there with no
        // edit to `voice-cli` at all.
        for argv in [
            &["voice", "tts", "train", "corpus/"][..],
            &["voice", "tts", "download", "--no-prosody"][..],
        ] {
            Nested::try_parse_from(argv).unwrap_or_else(|e| panic!("{argv:?}: {e}"));
        }
        // The bare filter nests too, flags and all.
        let NestedCommand::Tts(cli) =
            Nested::try_parse_from(["voice", "tts", "--top-p", "0.8", "-r", "clip.wav"])
                .expect("the filter nests")
                .command;
        assert!(cli.command.is_none(), "no subcommand is the filter");
        assert_eq!(cli.synth.top_p, 0.8);
    }

    #[test]
    fn completions_belongs_to_the_binary_and_never_appears_under_voice() {
        // While `Completions` sat in the shared enum, `voice tts completions
        // bash` emitted a script beginning `_voice()` — four spellings of one
        // script, three of them lies. It is now the hosting `main`'s, so this
        // subcommand must not exist here at all.
        assert!(
            Nested::try_parse_from(["voice", "tts", "completions", "bash"]).is_err(),
            "`voice tts completions` must not parse"
        );
        assert!(
            Cli::try_parse_from(["tts", "completions", "bash"]).is_err(),
            "even in the standalone tree it belongs to `main`, not to `TtsCli`"
        );
    }

    #[test]
    fn the_output_rate_defaults_to_what_the_model_produces() {
        // Anything else is resampled, so the default has to be the model's own
        // rate rather than a number that happens to match it today.
        assert_eq!(filter(&[]).sr, tts_core::OUTPUT_SR);
        assert_eq!(filter(&["--sr", "16000"]).sr, 16000);
    }

    #[test]
    fn each_language_maps_to_the_front_end_it_names() {
        assert_eq!(text_kit::Language::from(Lang::Zh), text_kit::Language::Zh);
        assert_eq!(text_kit::Language::from(Lang::En), text_kit::Language::En);
        assert_eq!(text_kit::Language::from(Lang::Ja), text_kit::Language::Ja);
        assert_eq!(Lang::default(), Lang::Zh);
    }
}
