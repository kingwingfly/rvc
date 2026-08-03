//! `tts convert` — speak text files, one WAV per file.
//!
//! The batch form of the bare filter: the same model, the same reference and the
//! same sampling knobs, reading lines from files instead of stdin and writing
//! `{output_dir}/{stem}.wav` instead of raw PCM.

use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::{ArgGroup, Args};
use futures::stream;
use tts_core::OUTPUT_SR;

use crate::args::{TtsArgs, load_models, read_reference};

/// Text files in, one WAV each out.
///
/// The two reference flags are `Option` because [`TtsArgs`] is also flattened
/// beside `train`, which does not want them — but nothing is flattened beside
/// *this*, so the groups below make them what they really are here: required,
/// and reported by clap as a usage error rather than after a model has loaded.
#[derive(Debug, Args)]
#[command(group(ArgGroup::new("ref_clip").arg("reference").required(true)))]
#[command(group(ArgGroup::new("ref_text").arg("reference_text").required(true)))]
pub struct ConvertArgs {
    /// One or more text files. Every non-blank line of a file is one utterance,
    /// and the whole file becomes one WAV.
    #[arg(required = true)]
    pub input: Vec<PathBuf>,
    /// Directory to write `<stem>.wav` files into.
    #[arg(short = 'o', long, default_value = ".")]
    pub output_dir: PathBuf,
    #[command(flatten)]
    pub synth: TtsArgs,
}

pub async fn run(args: ConvertArgs) -> Result<()> {
    args.synth.verify()?;
    // Guaranteed by the groups on `ConvertArgs`; the messages are here so a
    // future change to those groups cannot turn into a panic.
    let reference = args
        .synth
        .reference
        .as_ref()
        .context("a --reference recording is required")?;
    let reference_text = args
        .synth
        .reference_text
        .as_deref()
        .context("--reference-text is required: it is what `s1` continues from")?;

    let audio = read_reference(reference).await?;
    tracing::info!(
        "reference: {} ({:.1} s)",
        reference.display(),
        audio.len() as f32 / tts_core::ANALYSIS_SR as f32
    );

    // Loading the model and analysing the reference are both expensive and
    // neither depends on the input, so they happen once for the whole batch.
    let mut model = load_models(&args.synth).await?;
    let opts = args.synth.options();
    let prompt = tokio::task::block_in_place(|| {
        model.reference(&audio, reference_text, args.synth.language.into())
    })
    .context("failed to analyse the reference recording")?;

    tokio::fs::create_dir_all(&args.output_dir)
        .await
        .with_context(|| format!("creating output dir {}", args.output_dir.display()))?;

    for input in &args.input {
        let text = tokio::fs::read_to_string(input)
            .await
            .with_context(|| format!("reading {}", input.display()))?;

        let mut pcm = Vec::new();
        for line in text.lines() {
            if line.trim().is_empty() {
                continue;
            }
            let spoken = tokio::task::block_in_place(|| model.say(line, &prompt, &opts))
                .with_context(|| format!("synthesising {line:?}"))?;
            tracing::info!("{:.2} s  {line}", spoken.len() as f32 / OUTPUT_SR as f32);
            pcm.extend_from_slice(&spoken);
        }
        // An empty WAV is a worse answer than a message: a file of nothing but
        // blank lines is a mistake, and the next command in the pipe would not
        // notice it.
        anyhow::ensure!(!pcm.is_empty(), "{} has no text to speak", input.display());

        let pcm = audio_kit::resample_linear(&pcm, OUTPUT_SR, args.synth.sr);
        let stem = input.file_stem().and_then(|s| s.to_str()).unwrap_or("out");
        let out_path = args.output_dir.join(format!("{stem}.wav"));
        let out_stream = stream::iter([Ok::<_, audio_kit::AudioError>(pcm)]);
        audio_kit::write_wav_file(&out_path, args.synth.sr, out_stream)
            .await
            .with_context(|| format!("writing {}", out_path.display()))?;
        tracing::info!("wrote {}", out_path.display());
    }
    Ok(())
}
