//! Score a recording's windows against a reference voice and print the
//! distribution — **the harness `--threshold`'s default was measured with.**
//!
//! Weight coverage says `burn-campplus`'s module tree matches
//! `campplus_cn_common.bin`. It says nothing about whether the pair of
//! transforms in front of it separates *speakers*, and nothing at all about
//! where a threshold should sit. This is the check that does both, and it needs
//! no reference implementation: run audio through exactly the path
//! [`preprocess_core::diarize`] runs it through and read the numbers back.
//!
//! **It measures the same quantity the stage compares.** One whole-reference
//! embedding against one fixed-length *window* embedding — not clip against
//! clip. Statistics pooling makes the embedding's shape length-independent, but
//! not the cosine distribution it lands in, so a threshold calibrated on
//! sentence-length clips is a threshold for a different measurement.
//!
//! ```sh
//! cargo run -p preprocess-core --features tch --example timbre -- \
//!     --campplus ~/.cache/voice/models--funasr--campplus/…/campplus_cn_common.bin \
//!     --reference dataset/take_000.wav dataset/take_050.wav other-speaker.wav
//! ```
//!
//! # Reading it
//!
//! Every file gets percentiles and a timeline, one character per window, dark
//! for a low cosine and bright for a high one — which is what makes a *bimodal*
//! file visible as banding rather than as a wide spread. The number that
//! answers "is this working" is the **gap**: the same-speaker 5th percentile
//! against the different-speaker 95th. If they overlap, no threshold separates
//! those two voices and the honest report says so rather than picking one.
//!
//! Build it with an isolated `CARGO_TARGET_DIR` if anything else is building in
//! a sibling worktree — example binaries are not hashed per checkout, and the
//! one that runs is whichever landed last.

use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;
use cli_kit::Backend;
use preprocess_core::diarize::reference;
use preprocess_core::embed::{ANALYSIS_SR, cosine, load};

#[derive(Parser)]
#[command(about = "pairwise cosine of a recording's windows against a reference voice")]
struct Cli {
    /// CAM++ weights (`campplus_cn_common.bin`).
    #[arg(long)]
    campplus: PathBuf,
    /// The voice to score against.
    #[arg(short, long)]
    reference: PathBuf,
    /// Recordings to score. Name the reference speaker's other material and a
    /// different speaker's in one run — the gap between them is the reading.
    #[arg(required = true)]
    input: Vec<PathBuf>,
    /// Analysis window, seconds.
    #[arg(long, default_value_t = 3.0)]
    window: f32,
    /// Step between windows, seconds.
    #[arg(long, default_value_t = 1.0)]
    hop: f32,
    /// Window RMS floor in dBFS. Quieter windows are reported separately rather
    /// than mixed into the percentiles: a window with no voice in it embeds to
    /// something arbitrary, and letting those into the distribution is what
    /// makes a threshold look unreachable.
    #[arg(long, default_value_t = -40.0)]
    silence_db: f32,
    #[arg(long, value_enum, default_value_t = Backend::Auto)]
    backend: Backend,
    #[arg(long, default_value = "auto", value_parser = cli_kit::parse_device)]
    device: burn_kit::DeviceSpec,
}

/// The value at `q` of an already-sorted slice, nearest-rank.
fn percentile(sorted: &[f32], q: f64) -> f32 {
    let i = ((q * (sorted.len() - 1) as f64).round() as usize).min(sorted.len() - 1);
    sorted[i]
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    let embedder = load(&cli.campplus, cli.backend, cli.device)?;
    let target = reference(embedder.as_ref(), &cli.reference).await?;
    println!(
        "reference : {}\nwindow    : {:.2} s, hop {:.2} s\n",
        cli.reference.display(),
        cli.window,
        cli.hop,
    );

    let win = (cli.window * ANALYSIS_SR as f32) as usize;
    let hop = ((cli.hop * ANALYSIS_SR as f32) as usize).max(1);

    for path in &cli.input {
        let audio = preprocess_core::decode_mono(path, ANALYSIS_SR).await?;
        let windows: Vec<&[f32]> = (0..)
            .map(|i| i * hop)
            .take_while(|start| start + win <= audio.len())
            .map(|start| &audio[start..start + win])
            .collect();
        if windows.is_empty() {
            println!(
                "{}: {:.2} s, shorter than one window — skipped",
                path.display(),
                audio.len() as f64 / ANALYSIS_SR as f64,
            );
            continue;
        }

        let scores: Vec<f32> = embedder
            .embed(&windows)?
            .iter()
            .map(|e| cosine(e, &target))
            .collect();
        // The level each window was scored at. A window with no voice in it
        // embeds to something arbitrary, so its cosine says nothing about a
        // speaker and reading it as if it did is what makes a distribution look
        // hopeless — hence the two sets of percentiles below.
        let levels: Vec<f32> = windows
            .iter()
            .map(|w| {
                let rms = (w.iter().map(|s| s * s).sum::<f32>() / w.len() as f32).sqrt();
                20.0 * rms.max(1e-12).log10()
            })
            .collect();
        let voiced: Vec<f32> = scores
            .iter()
            .zip(&levels)
            .filter(|(_, db)| **db >= cli.silence_db)
            .map(|(s, _)| *s)
            .collect();

        let stats = |label: &str, of: &[f32]| {
            if of.is_empty() {
                println!("  {label:14} none");
                return;
            }
            let mut sorted = of.to_vec();
            sorted.sort_by(f32::total_cmp);
            println!(
                "  {label:14} n {:3}  min {:.3}  p5 {:.3}  p25 {:.3}  med {:.3}  p75 {:.3}  \
                 p95 {:.3}  max {:.3}  mean {:.3}",
                sorted.len(),
                sorted[0],
                percentile(&sorted, 0.05),
                percentile(&sorted, 0.25),
                percentile(&sorted, 0.50),
                percentile(&sorted, 0.75),
                percentile(&sorted, 0.95),
                sorted[sorted.len() - 1],
                of.iter().sum::<f32>() / of.len() as f32,
            );
        };
        println!(
            "{} ({:.1} s, {} windows)",
            path.display(),
            audio.len() as f64 / ANALYSIS_SR as f64,
            scores.len(),
        );
        stats("all", &scores);
        stats(&format!("above {:.0} dB", cli.silence_db), &voiced);

        // A fixed 0.05-wide grid from 0 to 1: comparable between files, which a
        // histogram over each file's own range would not be. Only the windows
        // that had a voice in them, since those are the ones a threshold has to
        // separate.
        let mut bins = [0usize; 20];
        for s in &voiced {
            bins[((s.max(0.0) * 20.0) as usize).min(19)] += 1;
        }
        for (i, n) in bins.iter().enumerate() {
            if *n > 0 {
                println!(
                    "  {:.2}-{:.2} {:4} {}",
                    i as f32 * 0.05,
                    (i + 1) as f32 * 0.05,
                    n,
                    "#".repeat((*n * 40 / voiced.len().max(1)).max(1))
                );
            }
        }

        // One character per window in time order, cosine over level. Banding in
        // the first is a recording that alternates between two voices; reading
        // it against the second is what says whether a dark stretch was another
        // speaker or simply nobody speaking.
        let ramp: Vec<char> = " .:-=+*#%@".chars().collect();
        let bar = |v: f32, lo: f32, hi: f32| {
            let t = ((v - lo) / (hi - lo)).clamp(0.0, 0.999);
            ramp[(t * ramp.len() as f32) as usize]
        };
        println!(
            "  cos   |{}|",
            scores.iter().map(|s| bar(*s, 0.0, 1.0)).collect::<String>()
        );
        println!(
            "  level |{}|  ({:.0}..{:.0} dBFS)\n",
            levels
                .iter()
                .map(|d| bar(*d, cli.silence_db - 20.0, 0.0))
                .collect::<String>(),
            levels.iter().fold(f32::MAX, |m, d| m.min(*d)),
            levels.iter().fold(f32::MIN, |m, d| m.max(*d)),
        );
    }
    Ok(())
}
