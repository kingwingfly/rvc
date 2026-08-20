//! Match recordings to one level.
//!
//! One file in, one `<base>.wav` out, at a gain that is **constant over the
//! whole file**. That is the property the stage is built around rather than an
//! implementation detail: a corpus of soft, breathy close-mic material is
//! mostly quiet on purpose, and anything that moved the gain about while it
//! played would flatten exactly the content this toolkit exists to preserve.
//!
//! Two ways to say what level to land on, and they answer different questions:
//!
//! - [`Target::Peak`] — the loudest sample lands at a fraction of full scale.
//!   Exact, instant, and blind to everything but that one sample, so a single
//!   stray thump decides the gain for a whole recording.
//! - [`Target::Lufs`] — EBU R128 integrated loudness, which is what "as loud as
//!   each other" means to a listener: K-weighted, gated so the silence between
//!   sentences does not drag the reading down, and therefore unbothered by the
//!   thump.
//!
//! Use peak when you want a known headroom, LUFS when you want two takes to sit
//! at the same level.

use std::path::Path;

use anyhow::{Context, Result};
use audio_kit::write_wav_file;
use futures::stream;

use crate::InputFile;

/// Where [`Target::Peak`] puts the loudest sample when nothing says otherwise.
///
/// The number that `clip --normalize` used to hard-code with no way to reach
/// it: 5% of headroom, enough that a later resample's interpolation overshoot
/// does not clip. It lives here, once, so the shorthand flag and this stage
/// cannot drift apart.
pub const DEFAULT_PEAK: f32 = 0.95;

/// A gated loudness measurement needs at least one 400 ms block, and a
/// recording shorter than that has no integrated loudness at all — not a quiet
/// one, none. Stated here because the failure has to be reported rather than
/// papered over with a peak fallback the user did not ask for.
pub const MIN_LUFS_SECONDS: f64 = 0.4;

/// R128's absolute gate, in LUFS. A recording with nothing above it has no
/// integrated loudness.
///
/// This has to be tested for explicitly, because **ffmpeg reports the gate
/// itself rather than `-inf` when nothing clears it** — digital silence measures
/// exactly `-70.0`, which is finite, parses, and would be normalized *from* as
/// if it were a very quiet recording. A 2 s file of zeros would then be handed
/// +47 dB of gain against a `-23` target and come out as amplified nothing.
const ABSOLUTE_GATE_LUFS: f32 = -70.0;

/// What level to land on.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Target {
    /// The loudest sample, as a fraction of full scale.
    Peak(f32),
    /// EBU R128 integrated loudness, in LUFS (so a negative number).
    Lufs(f32),
}

/// What to normalize, and to what.
#[derive(Debug, Clone, Copy)]
pub struct NormalizeOptions {
    /// Sample rate to decode at and write out.
    pub sr: u32,
    /// The level to land on.
    pub target: Target,
}

/// What one file produced.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NormalizeReport {
    /// Duration decoded from the input.
    pub in_secs: f64,
    /// The constant gain applied, in dB. Zero means the file was already there.
    pub gain_db: f32,
    /// Loudest sample before and after, as a fraction of full scale.
    pub peak_before: f32,
    /// Loudest sample after the gain. **Above 1.0 is possible on a LUFS run**
    /// and is reported rather than limited: a limiter is dynamics processing,
    /// which is the one thing this stage promises not to do.
    pub peak_after: f32,
    /// Integrated loudness before and after, measured both times. `None` on a
    /// peak run, where nothing was measured.
    pub lufs: Option<(f32, f32)>,
}

/// Normalize one file into the output directory.
pub async fn file(
    input: &InputFile,
    opts: &NormalizeOptions,
    output_dir: &Path,
) -> Result<NormalizeReport> {
    let mut samples = crate::decode_mono(&input.path, opts.sr).await?;
    let in_secs = samples.len() as f64 / opts.sr as f64;
    let peak_before = peak(&samples);

    let (gain_db, lufs) = match opts.target {
        Target::Peak(t) => {
            let gain = peak_normalize(&mut samples, t);
            (db(gain), None)
        }
        Target::Lufs(t) => {
            let before = integrated_lufs(&samples, opts.sr)
                .with_context(|| format!("measuring {}", input.path.display()))?
                .with_context(|| {
                    format!(
                        "{} has no integrated loudness to normalize: {in_secs:.2}s of audio, \
                         all of it below the R128 gate (a gated reading needs at least \
                         {MIN_LUFS_SECONDS}s of audible content)",
                        input.path.display()
                    )
                })?;
            let gain_db = t - before;
            let gain = 10f32.powf(gain_db / 20.0);
            for s in samples.iter_mut() {
                *s *= gain;
            }
            // Measured again rather than assumed: R128's absolute -70 LUFS gate
            // is the one part of the measurement a gain does not slide, so the
            // achieved level is a reading and not arithmetic. It is also what
            // catches a wrong gain being applied at all.
            let after = integrated_lufs(&samples, opts.sr)
                .with_context(|| format!("re-measuring {}", input.path.display()))?
                .unwrap_or(f32::NEG_INFINITY);
            (gain_db, Some((before, after)))
        }
    };

    let peak_after = peak(&samples);
    let out_path = output_dir.join(format!("{}.wav", input.base));
    let out_stream = stream::iter([Ok::<_, audio_kit::AudioError>(samples)]);
    write_wav_file(&out_path, opts.sr, out_stream)
        .await
        .with_context(|| format!("writing {}", out_path.display()))?;

    Ok(NormalizeReport {
        in_secs,
        gain_db,
        peak_before,
        peak_after,
        lufs,
    })
}

/// The loudest sample, as a fraction of full scale.
pub fn peak(samples: &[f32]) -> f32 {
    samples.iter().fold(0.0f32, |m, x| m.max(x.abs()))
}

/// Scale `samples` in place so the loudest sits at `target` full-scale,
/// returning the linear gain applied.
///
/// No-op (gain 1.0) on silence: the ratio is unbounded there, and dividing by
/// a peak of ~0 writes infinities into a WAV.
///
/// Shared with [`crate::clip`], whose `--normalize` is the same operation under
/// a shorter name — one definition, so the two cannot end up scaling to
/// different levels.
pub fn peak_normalize(samples: &mut [f32], target: f32) -> f32 {
    let p = peak(samples);
    if p <= 1e-9 {
        return 1.0;
    }
    let gain = target / p;
    for s in samples.iter_mut() {
        *s *= gain;
    }
    gain
}

/// A linear gain as dB, with silence's `1.0` reading as `0.0`.
fn db(gain: f32) -> f32 {
    20.0 * gain.max(1e-12).log10()
}

/// EBU R128 integrated loudness of `samples`, in LUFS.
///
/// `Ok(None)` when the recording has no gated measurement to give — shorter
/// than one 400 ms block, or entirely below the absolute -70 LUFS gate. That is
/// a distinct answer from a quiet one, and the caller reports it rather than
/// falling back to a peak the user did not ask for.
///
/// # Why the measurement is ffmpeg's and the gain is ours
///
/// The obvious chain is `loudnorm`, and it is the wrong one twice over. In
/// single pass it is a *dynamic* normalizer — it compresses and true-peak
/// limits, which is precisely the processing a breathy corpus must not get —
/// and its linear mode needs measured values it will only ever print to a log,
/// where an in-process filter graph cannot read them. It also negotiates its
/// output to 192 kHz, which [`audio_kit::AudioFilter`] would hand back as if it
/// were `sr`.
///
/// `ebur128` has none of those problems: it passes the audio through untouched
/// and injects the running measurement as frame metadata, so ffmpeg does the
/// standard-compliant part and the gain stays one multiply.
pub fn integrated_lufs(samples: &[f32], sr: u32) -> Result<Option<f32>> {
    if (samples.len() as f64 / sr as f64) < MIN_LUFS_SECONDS {
        return Ok(None);
    }
    // `peak=none` because nothing here reads a true-peak, and computing one
    // oversamples every frame. `framelog=quiet` silences the per-frame lines —
    // and *only* those. The summary block this filter prints when its graph is
    // dropped goes out at info level regardless, so what actually keeps it off
    // stderr is `audio-kit` setting ffmpeg's log level, which is the only
    // control there is over it.
    let mut meter = audio_kit::AudioFilter::new(sr, "ebur128=metadata=1:peak=none:framelog=quiet")
        .context("building the ebur128 measurement graph")?;
    meter.process(samples).context("measuring loudness")?;
    // After the flush, not before: the integrated value is a running one, and
    // the reading that covers the whole recording is on the last frame out.
    meter.flush().context("draining the loudness meter")?;

    let Some(raw) = meter.metadata("lavfi.r128.I") else {
        return Ok(None);
    };
    let value: f32 = raw
        .parse()
        .with_context(|| format!("ebur128 reported an unreadable loudness: {raw:?}"))?;
    // Both spellings of "nothing cleared the gate": `-inf`, and the gate itself.
    // See [`ABSOLUTE_GATE_LUFS`] — the second is the one that looks like a
    // reading.
    Ok((value.is_finite() && value > ABSOLUTE_GATE_LUFS).then_some(value))
}

/// Every measurement below `expect()`s its meter rather than stepping around a
/// missing one.
///
/// Four of these tests used to open with `let Ok(..) = integrated_lufs(..)
/// else { return }`, which swallowed *any* error — and two of them, spelled
/// `Ok(Some(..))`, swallowed a `None` reading as well. A build where
/// `lavfi.r128.I` stopped appearing would have taken the whole set green while
/// they asserted nothing, and these are the tests holding this crate's only
/// external oracle: `-21.1 LUFS` is what `ffmpeg -af ebur128` independently
/// reports for the same waveform. A skip indistinguishable from a pass is worse
/// than an absent test, so if the meter is gone these now say so.
#[cfg(test)]
mod tests {
    use super::*;

    const SR: u32 = 48_000;

    /// A 1 kHz tone at `amplitude`, the signal ffmpeg's own `sine` source
    /// produces — so a reading here can be checked against `ffmpeg -af ebur128`
    /// on the same waveform.
    fn tone(secs: f32, amplitude: f32) -> Vec<f32> {
        let n = (secs * SR as f32) as usize;
        (0..n)
            .map(|i| {
                amplitude * (std::f64::consts::TAU * 1000.0 * i as f64 / SR as f64).sin() as f32
            })
            .collect()
    }

    #[test]
    fn normalizing_lifts_the_peak_to_the_target() {
        let mut s = vec![0.5, -0.25, 0.0, 0.5];
        let gain = peak_normalize(&mut s, 0.95);
        assert!((gain - 1.9).abs() < 1e-6, "{gain}");
        assert!((s[0] - 0.95).abs() < 1e-6, "{s:?}");
        assert!((s[1] + 0.475).abs() < 1e-6, "{s:?}");
        // Gain, not a rewrite: the shape is untouched.
        assert_eq!(s[2], 0.0);
    }

    #[test]
    fn a_silent_clip_is_left_alone() {
        // Without the guard this divides by ~0 and writes infinities into a
        // WAV — silence in, silence out is the only sane answer.
        let mut s = vec![0.0, 0.0, 0.0];
        assert_eq!(peak_normalize(&mut s, 0.95), 1.0);
        assert_eq!(s, vec![0.0, 0.0, 0.0]);

        let mut tiny = vec![1e-12, -1e-12];
        assert_eq!(peak_normalize(&mut tiny, 0.95), 1.0);
        assert_eq!(tiny, vec![1e-12, -1e-12]);
    }

    /// The known answer: ffmpeg's `sine` source at its own default amplitude of
    /// 0.125 measures -21.1 LUFS, which is what `ffmpeg -af ebur128` reports for
    /// it. A meter reading its own output back is not a check; this is the same
    /// waveform an independent tool has already put a number on.
    #[test]
    fn a_known_tone_measures_what_ffmpeg_says_it_does() {
        let measured = integrated_lufs(&tone(3.0, 0.125), SR)
            .expect("measure")
            .expect("3 s of tone has an integrated loudness");
        assert!(
            (measured - -21.1).abs() < 0.2,
            "expected ~-21.1 LUFS, got {measured}"
        );
    }

    /// The property the whole stage rests on: the reading moves by exactly the
    /// gain applied. If it did not, "normalize to -23" would be a fixed point
    /// nothing converges to.
    #[test]
    fn the_reading_tracks_the_gain_exactly() {
        let loud = integrated_lufs(&tone(3.0, 0.5), SR)
            .expect("measure")
            .expect("3 s of tone has an integrated loudness");
        let quiet = integrated_lufs(&tone(3.0, 0.25), SR)
            .expect("measure")
            .expect("3 s of tone has an integrated loudness");
        assert!(
            ((loud - quiet) - 6.02).abs() < 0.05,
            "halving the amplitude should read 6.02 LU quieter: {loud} vs {quiet}"
        );
    }

    /// What the stage promises: ask for -23 and the written audio measures -23.
    #[test]
    fn normalizing_to_a_lufs_target_lands_on_it() {
        let mut s = tone(3.0, 0.125);
        let before = integrated_lufs(&s, SR)
            .expect("measure")
            .expect("3 s of tone has an integrated loudness");
        let gain_db = -23.0 - before;
        let gain = 10f32.powf(gain_db / 20.0);
        for x in s.iter_mut() {
            *x *= gain;
        }
        let after = integrated_lufs(&s, SR).expect("measure").expect("finite");
        assert!((after - -23.0).abs() < 0.1, "landed on {after}, not -23");
    }

    /// A directory of this test's own, named after the process so parallel
    /// worktrees cannot collide — the same hazard the shared cargo target
    /// directory has.
    fn scratch(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("normalize-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("create scratch dir");
        dir
    }

    /// Write `samples` as the stage's input file and return what to hand
    /// [`file`].
    async fn input_file(dir: &Path, base: &str, samples: Vec<f32>) -> InputFile {
        let path = dir.join(format!("{base}.wav"));
        let stream = stream::iter([Ok::<_, audio_kit::AudioError>(samples)]);
        write_wav_file(&path, SR, stream).await.expect("write input");
        InputFile {
            path,
            base: base.to_string(),
        }
    }

    /// Read a 32-bit-float mono WAV back at the spec's own offsets.
    ///
    /// **Not** through [`crate::decode_mono`], and the difference is the whole
    /// point of these tests. A decoder is a resampler with a format converter
    /// in front of it, so it is exactly the thing that could clamp an
    /// overshoot away or round a sample — and an overshoot surviving to disk is
    /// one of the properties under test. Reading the bytes where the header
    /// says they are cannot do either.
    ///
    /// The declared `data` size is checked against the bytes actually present,
    /// because a correct header over a truncated body is what "the stage
    /// dropped the tail" looks like, and only comparing the two sees it.
    /// `audio-kit`'s own `a_mono_wav_decodes_back_to_the_samples_it_was_written_from`
    /// pins this reader's other half — that the decoder agrees with the writer
    /// sample for sample — so the pair covers both directions.
    fn read_wav_f32(path: &Path, expect_sr: u32) -> Vec<f32> {
        let bytes = std::fs::read(path).expect("read written wav");
        assert!(bytes.len() >= 44, "shorter than a WAV header");
        assert_eq!(&bytes[0..4], b"RIFF");
        assert_eq!(&bytes[8..12], b"WAVE");
        let u16_at = |o: usize| u16::from_le_bytes([bytes[o], bytes[o + 1]]);
        let u32_at =
            |o: usize| u32::from_le_bytes([bytes[o], bytes[o + 1], bytes[o + 2], bytes[o + 3]]);
        assert_eq!(u16_at(20), 3, "format tag must stay IEEE float");
        assert_eq!(u16_at(22), 1, "the stage writes mono");
        assert_eq!(
            u32_at(24),
            expect_sr,
            "written at a different rate than it was decoded at"
        );
        assert_eq!(u16_at(34), 32, "bits per sample");
        assert_eq!(&bytes[36..40], b"data");
        let declared = u32_at(40) as usize;
        assert_eq!(
            declared,
            bytes.len() - 44,
            "the header declares a data size the file does not carry"
        );
        bytes[44..]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    /// A peak run, measured on the file it wrote rather than on the vector the
    /// test still had in hand.
    ///
    /// Every other test in this module exercises [`peak_normalize`] or
    /// [`integrated_lufs`] directly, which leaves the stage itself — decode,
    /// gain, encode, and the report describing all three — checked by nothing.
    /// That is the shape this repository has already been bitten by twice: a
    /// number that was real and a harness that did not compute it.
    #[tokio::test]
    async fn a_peak_run_writes_a_file_whose_loudest_sample_is_the_target() {
        let dir = scratch("peak");
        let out = dir.join("out");
        std::fs::create_dir_all(&out).expect("create out");
        let samples = tone(1.0, 0.3);
        let expect_peak_before = peak(&samples);
        let input = input_file(&dir, "take", samples).await;

        let report = file(
            &input,
            &NormalizeOptions {
                sr: SR,
                target: Target::Peak(0.95),
            },
            &out,
        )
        .await
        .expect("normalize");

        let written = read_wav_f32(&out.join("take.wav"), SR);
        assert_eq!(
            written.len(),
            SR as usize,
            "a gain must not resample or truncate"
        );
        let peak_written = peak(&written);
        assert!(
            (peak_written - 0.95).abs() < 1e-6,
            "the loudest written sample is {peak_written}, not the 0.95 asked for"
        );

        // The report is the user's only window onto the stage, so it is
        // checked against the file rather than against itself.
        assert!((report.in_secs - 1.0).abs() < 1e-9, "{}", report.in_secs);
        assert!(
            (report.peak_before - expect_peak_before).abs() < 1e-6,
            "{} vs {expect_peak_before}",
            report.peak_before
        );
        assert!(
            (report.peak_after - peak_written).abs() < 1e-6,
            "the report says the file peaks at {} and it peaks at {peak_written}",
            report.peak_after
        );
        let expect_gain = 20.0 * (0.95 / expect_peak_before).log10();
        assert!(
            (report.gain_db - expect_gain).abs() < 1e-4,
            "{} vs {expect_gain}",
            report.gain_db
        );
        assert_eq!(report.lufs, None, "a peak run measures no loudness");
    }

    /// What the stage promises, asked of the stage: request -23 LUFS and the
    /// audio **on disk** measures -23.
    ///
    /// `normalizing_to_a_lufs_target_lands_on_it` above asserts the same number
    /// about a vector the test scaled itself, so it would pass unchanged if
    /// [`file`] applied the gain twice, wrote at the wrong rate or dropped the
    /// tail. This one re-measures what landed.
    #[tokio::test]
    async fn a_lufs_run_writes_a_file_that_measures_the_target() {
        let dir = scratch("lufs");
        let out = dir.join("out");
        std::fs::create_dir_all(&out).expect("create out");
        let samples = tone(3.0, 0.125);
        let before = integrated_lufs(&samples, SR)
            .expect("measure")
            .expect("3 s of tone has an integrated loudness");
        let input = input_file(&dir, "take", samples).await;

        let report = file(
            &input,
            &NormalizeOptions {
                sr: SR,
                target: Target::Lufs(-23.0),
            },
            &out,
        )
        .await
        .expect("normalize");

        let written = read_wav_f32(&out.join("take.wav"), SR);
        assert_eq!(
            written.len(),
            SR as usize * 3,
            "a gain must not resample or truncate"
        );
        let measured = integrated_lufs(&written, SR)
            .expect("re-measure")
            .expect("the written file has an integrated loudness");
        assert!(
            (measured - -23.0).abs() < 0.1,
            "the written file measures {measured}, not the -23 asked for"
        );

        let (r_before, r_after) = report.lufs.expect("a lufs run reports both readings");
        assert!((r_before - before).abs() < 0.05, "{r_before} vs {before}");
        assert!(
            (r_after - measured).abs() < 0.05,
            "the report says the file measures {r_after} and it measures {measured}"
        );
        assert!(
            (report.gain_db - (-23.0 - before)).abs() < 1e-4,
            "{}",
            report.gain_db
        );
        assert!(
            (report.peak_after - peak(&written)).abs() < 1e-6,
            "{} vs {}",
            report.peak_after,
            peak(&written)
        );
    }

    /// The one promise a limiter would break, checked on the samples that
    /// reached the file.
    ///
    /// [`NormalizeReport::peak_after`]'s doc says an overshoot is *reported
    /// rather than limited*, because limiting is dynamics processing and this
    /// stage exists not to do that. A quiet tone with one loud thump in it is
    /// the case that produces one: R128's gate ignores the thump, so the gain
    /// is decided by the tone and the thump is carried past full scale.
    #[tokio::test]
    async fn a_lufs_gain_that_overshoots_full_scale_is_written_not_limited() {
        let dir = scratch("overshoot");
        let out = dir.join("out");
        std::fs::create_dir_all(&out).expect("create out");
        let mut samples = tone(3.0, 0.05);
        // A three-sample thump, far too short to move a gated reading (its
        // contribution to a 400 ms block is ~0.1 dB) and loud enough to decide
        // the peak.
        for s in samples.iter_mut().skip(SR as usize).take(3) {
            *s = 0.5;
        }
        let input = input_file(&dir, "take", samples).await;

        let report = file(
            &input,
            &NormalizeOptions {
                sr: SR,
                target: Target::Lufs(-12.0),
            },
            &out,
        )
        .await
        .expect("normalize");

        let written = read_wav_f32(&out.join("take.wav"), SR);
        let peak_written = peak(&written);
        assert!(
            peak_written > 1.0,
            "the overshoot was limited away: peak {peak_written}"
        );
        assert!(
            (report.peak_after - peak_written).abs() < 1e-6,
            "the report says {} and the file says {peak_written}",
            report.peak_after
        );
        // Still on target despite the overshoot, which is the other half of
        // "reported rather than limited": nothing was traded for it.
        let measured = integrated_lufs(&written, SR)
            .expect("re-measure")
            .expect("the written file has an integrated loudness");
        assert!(
            (measured - -12.0).abs() < 0.3,
            "the written file measures {measured}, not the -12 asked for"
        );
    }

    /// Silence is refused rather than handed +47 dB, and nothing is written.
    ///
    /// See [`ABSOLUTE_GATE_LUFS`]: the failure this guards is not a crash but a
    /// plausible number, so the check is that [`file`] returns an error naming
    /// the reason and leaves no output behind for a later stage to pick up.
    #[tokio::test]
    async fn a_file_with_no_gated_loudness_is_refused_rather_than_amplified() {
        let dir = scratch("silence");
        let out = dir.join("out");
        std::fs::create_dir_all(&out).expect("create out");
        let input = input_file(&dir, "take", vec![0.0; SR as usize * 2]).await;

        let err = file(
            &input,
            &NormalizeOptions {
                sr: SR,
                target: Target::Lufs(-23.0),
            },
            &out,
        )
        .await
        .expect_err("silence has no loudness to normalize to");
        let msg = format!("{err:#}");
        assert!(
            msg.contains("no integrated loudness"),
            "the error must say why: {msg}"
        );
        assert!(
            !out.join("take.wav").exists(),
            "a refused file must leave no output for the next stage to read"
        );

        // A peak run on the same silence is not an error — there is simply no
        // gain to apply, and silence out is the only sane answer.
        let report = file(
            &input,
            &NormalizeOptions {
                sr: SR,
                target: Target::Peak(DEFAULT_PEAK),
            },
            &out,
        )
        .await
        .expect("a peak run on silence");
        assert_eq!(report.gain_db, 0.0);
        assert_eq!(report.peak_after, 0.0);
        let written = read_wav_f32(&out.join("take.wav"), SR);
        assert_eq!(written.len(), SR as usize * 2);
        assert!(written.iter().all(|s| *s == 0.0), "silence in, silence out");
    }

    /// [`ABSOLUTE_GATE_LUFS`] is a claim about ffmpeg, so it is checked against
    /// ffmpeg rather than trusted.
    ///
    /// The load-bearing part is that `is_finite()` alone is **not** enough:
    /// ebur128 reports the gate itself for a signal that clears nothing, so the
    /// reading for digital silence parses as an ordinary number. Were that ever
    /// to become `-inf`, this test would fail and the constant could go — which
    /// is the point of pinning it.
    #[test]
    fn ffmpeg_reports_the_gate_itself_for_silence_rather_than_negative_infinity() {
        let mut meter =
            audio_kit::AudioFilter::new(SR, "ebur128=metadata=1:peak=none:framelog=quiet")
                .expect("this ffmpeg build must provide `ebur128`");
        meter.process(&vec![0.0; SR as usize * 2]).expect("measure");
        meter.flush().expect("flush");
        let raw = meter
            .metadata("lavfi.r128.I")
            .expect("an integrated reading after the flush");
        let value: f32 = raw.parse().expect("a number, not `-inf`");
        assert!(
            value.is_finite(),
            "silence read as {raw}, so the finiteness test alone would have caught it"
        );
        assert!(
            (value - ABSOLUTE_GATE_LUFS).abs() < 0.01,
            "silence measures {value}, but ABSOLUTE_GATE_LUFS is {ABSOLUTE_GATE_LUFS}"
        );
    }

    /// Too short to gate, and digitally silent: two different reasons for the
    /// same `None`, and both have to be that rather than a number.
    #[test]
    fn a_recording_with_no_gated_reading_says_so() {
        assert_eq!(
            integrated_lufs(&tone(0.1, 0.5), SR).expect("measure"),
            None,
            "100 ms is under one 400 ms gating block"
        );
        assert_eq!(
            integrated_lufs(&vec![0.0; SR as usize * 2], SR).expect("measure"),
            None,
            "silence has no loudness, not a very low one"
        );
    }
}
