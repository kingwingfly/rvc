//! Split a recording into a voice stem and a music stem, so a corpus recorded
//! over a backing track can be cleaned before anything else touches it.
//!
//! One file in, one `<base>.vocals.wav` and/or `<base>.instrumental.wav` out.
//!
//! # Why this runs before slicing, and what it is actually worth
//!
//! The decibels undersell it. The network's own measurements put the music
//! removed from the vocals stem at **5–6.5 dB** across four excerpts of a real
//! stream where the bed is continuous, against the 15–20 dB it reaches on a
//! song, because it was trained on *sung* vocals inside real productions and a
//! speaking voice is not one. Driven through this stage, one of those excerpts
//! reads **4.1 dB**. So the bed is attenuated rather than gone, and this stage
//! does not promise otherwise.
//!
//! **Every one of those figures reads the mono downmix, and the stems are
//! stereo.** Folding both the mixture and the stem to mid puts the bed 4.1 dB
//! down in that excerpt's speech gaps; reading the same two files as written
//! gives 1.0 dB, because what the stem keeps of the bed is largely out of phase
//! between the channels and cancels in the fold, where the mixture's own level
//! barely moves. Neither reading is wrong and they are not interchangeable —
//! the mono one is what was recorded, and it is also the one that matters here,
//! since everything downstream decodes mono.
//!
//! What that buys is out of proportion to the number: transcribing the same
//! 60 s, the mixture yields **5** segments (one of them 16.9 s of merged
//! speech) and the vocals stem **14**, one per utterance, with the same words.
//! [`audio_kit::slice`] cuts on silence and a continuous bed leaves none — so a
//! corpus recorded behind music cannot be sliced into sentences *at all* until
//! the bed comes off. Sliced with [`crate::clip`]'s defaults, that same 60 s
//! gives **5** clips holding 58 of its 60 seconds before separation and **12**
//! holding 39 after: the mixture has no sentence boundaries to find.
//!
//! Two consequences of emptying those gaps, both worth knowing downstream: a
//! recogniser has more room to invent in a newly-silent gap (one of the 14 was
//! a clear hallucination), and one segment came out at 0.37 s. A consumer of
//! these stems wants a duration floor.
//!
//! # What it does not do
//!
//! It separates **voice from music**, not one voice from another. Every speaker
//! in the recording lands in the vocals stem, including a singer on the backing
//! track — so a streamer talking over somebody else's *singing* still gets
//! both. Filtering by who is speaking is a different model and a different
//! stage.
//!
//! # Streaming, and the one buffer that is left
//!
//! The network's unit of work is one chunk of a few seconds and it knows
//! nothing about a longer recording, so this stage owns the seams: it decodes a
//! stream, holds one chunk plus one hop of it, and emits finished audio as it
//! goes. A twenty-minute song never exists in memory as a decoded whole.
//!
//! The exception is on the way out, and it is [`audio_kit::write_wav_stereo`]'s
//! rather than this stage's: a RIFF header carries the length of what follows,
//! so that writer collects the signal before it can write a byte. Feeding it
//! through a channel rather than a `Vec` is deliberate — the day that writer
//! patches its header at close instead, this stage becomes fully streaming with
//! no change here.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use audio_kit::{DecodeOptions, StereoSamples, decode_path_stereo, write_wav_stereo_file};
use futures::{SinkExt, StreamExt};

use crate::InputFile;

/// One separation model, behind the boundary that keeps this stage non-generic.
///
/// The shape `stt-core`'s `Engine` uses, and for the same reason: the backend is
/// a constructor call rather than a type parameter, so nothing here names a Burn
/// type and the choice stays a run-time one. Everything the network is aware of
/// is on this side of the trait — one chunk of stereo audio at one rate — and
/// everything longer than a chunk is [`Overlap`]'s, which is why a second model
/// with a different chunk length or rate needs no change below it.
pub trait Separator: Send + Sync {
    /// The rate the model requires, which is also the rate the stems are
    /// written at. Not a choice: resampling the stems here would be a lossy
    /// step the next stage's decode performs anyway.
    fn sample_rate(&self) -> u32;

    /// Frames per forward pass. [`Separator::separate`] is given exactly this
    /// many and nothing else.
    fn chunk_frames(&self) -> usize;

    /// The stems, in the order [`Separator::separate`] returns them. Not
    /// recoverable from the weights — a caller that guesses gets the
    /// accompaniment where it wanted the voice, with nothing to indicate it.
    fn stems(&self) -> &'static [&'static str];

    /// One chunk in, one stereo waveform per stem out, each as long as the
    /// input.
    fn separate(&self, chunk: &StereoSamples) -> Result<Vec<StereoSamples>>;
}

/// Which stems reach disk.
///
/// Both come out of the same forward pass — the network emits the pair and
/// there is no cheaper way to ask for one — so this decides what is *written*,
/// never what is computed. `Both` therefore costs only the second file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stems {
    /// The voice. What corpus preparation wants, and the default.
    Vocals,
    /// The music. Useful for checking by ear what was taken out.
    Instrumental,
    /// Both, which is free but for the extra file.
    Both,
}

impl Stems {
    /// Whether a stem of this name is written.
    fn wants(self, name: &str) -> bool {
        match self {
            Self::Both => true,
            Self::Vocals => name == "vocals",
            Self::Instrumental => name == "instrumental",
        }
    }
}

/// What to separate, and which halves to keep.
#[derive(Debug, Clone, Copy)]
pub struct SeparateOptions {
    /// Which stems are written. See [`Stems`].
    pub stems: Stems,
}

/// What one written stem came to.
#[derive(Debug, Clone, PartialEq)]
pub struct StemReport {
    /// The stem's name, as the model orders them.
    pub name: &'static str,
    /// Where it was written.
    pub path: PathBuf,
    /// Its duration in seconds.
    pub secs: f64,
    /// Its RMS, at the input's own level — the gain this stage applies for the
    /// model's benefit is undone before anything is written, so this is
    /// comparable with the input's.
    pub rms: f32,
}

/// What one file produced.
#[derive(Debug, Clone, PartialEq)]
pub struct SeparateReport {
    /// Duration decoded from the input.
    pub in_secs: f64,
    /// Forward passes the recording cost. Two per chunk of audio at 50%
    /// overlap, plus the flush.
    pub passes: usize,
    /// The gain the mixture was scaled by on the way in. 1.0 means the
    /// recording already peaked where the model wants it; see [`TARGET_PEAK`].
    pub gain: f32,
    /// One entry per stem written, in the model's own order.
    pub stems: Vec<StemReport>,
}

/// Where the mixture's peak is put before the network sees it.
///
/// **The network is not scale-invariant, and that is the whole reason this
/// stage normalises at all.** Its instance norms are, but the head multiplies
/// the U-net's output by the first convolution's and concatenates the raw
/// mixture beside it, so the two paths scale differently and the model only
/// behaves at the levels it was trained on. Feeding it +8 dBFS degrades the
/// separation with no error anywhere, which is the failure this constant
/// prevents — and the same argument runs downward, so a recording that peaks at
/// −30 dBFS is scaled *up* to here rather than left alone.
///
/// The gain is undone before anything is written. That is load-bearing rather
/// than tidiness: a stem left at the model's working level is uniformly
/// attenuated or boosted against its own mixture, and every level a user reads
/// off it afterwards — "how much quieter is the stem in the speech gaps" — is
/// wrong by exactly that gain, which reads as a separation result.
pub const TARGET_PEAK: f32 = 0.7;

/// Separate one file into the output directory.
pub async fn file(
    input: &InputFile,
    opts: &SeparateOptions,
    separator: &dyn Separator,
    output_dir: &Path,
) -> Result<SeparateReport> {
    let sr = separator.sample_rate();
    let names = separator.stems();
    let wanted: Vec<usize> = (0..names.len())
        .filter(|i| opts.stems.wants(names[*i]))
        .collect();
    anyhow::ensure!(
        !wanted.is_empty(),
        "none of the model's stems ({}) match the ones asked for",
        names.join(", ")
    );

    // Pass one: the peak, and only the peak. Streamed like the second pass, so
    // knowing the level costs a decode rather than a copy of the recording —
    // and it has to be known before the first chunk reaches the model, because
    // a gain that changed part-way through would be a seam the crossfade cannot
    // hide.
    //
    // Over each channel independently and never over the mono fold: out-of-
    // phase content makes `0.5 (L + R)` understate a channel's true peak, and
    // the model is fed the channels.
    let mut decode = std::pin::pin!(decode_path_stereo(&input.path, DecodeOptions::new(sr)));
    let (mut frames, mut peak) = (0usize, 0.0f32);
    while let Some(chunk) = decode.next().await {
        let chunk = chunk.with_context(|| format!("decoding {}", input.path.display()))?;
        frames += chunk.frames();
        peak = chunk
            .left
            .iter()
            .chain(&chunk.right)
            .fold(peak, |a, s| a.max(s.abs()));
    }
    anyhow::ensure!(frames > 0, "{} decoded no audio", input.path.display());
    // A recording that is silent throughout has no peak to normalise against,
    // and `TARGET_PEAK / 0` would hand the model an infinity. Left alone, it
    // separates silence into silence, which is the right answer.
    let gain = if peak > 1e-6 { TARGET_PEAK / peak } else { 1.0 };

    let mut writers: Vec<Writer> = wanted
        .iter()
        .map(|stem| Writer::create(names[*stem], &input.base, sr, output_dir))
        .collect();
    let paths: Vec<PathBuf> = writers.iter().map(|w| w.path.clone()).collect();

    let outcome = run(input, sr, gain, separator, &wanted, &mut writers).await;

    // Close the channels whatever happened, so a writer task is never left
    // holding one end of a file it will never finish. A writer's own error is
    // reported ahead of `outcome`, because a failed write is what makes the
    // send that noticed it fail.
    let mut stems = Vec::with_capacity(writers.len());
    let mut first_error = None;
    for writer in writers {
        match writer.finish(sr).await {
            Ok(report) => stems.push(report),
            Err(e) => first_error = first_error.or(Some(e)),
        }
    }
    let failure = first_error.or_else(|| outcome.as_ref().err().map(|e| anyhow::anyhow!("{e:#}")));
    if let Some(e) = failure {
        // A stem that stopped part-way through is still a valid WAV of the
        // wrong length, and the batch driver only prints that it skipped the
        // file. Leaving it would put a truncated recording into a corpus that
        // nothing downstream could tell from a short one.
        for path in &paths {
            let _ = std::fs::remove_file(path);
        }
        return Err(e);
    }
    let passes = outcome?;

    Ok(SeparateReport {
        in_secs: frames as f64 / sr as f64,
        passes,
        gain,
        stems,
    })
}

/// The second pass: decode, separate, write. Split out so [`file`] can close the
/// writers on the way out of either branch.
async fn run(
    input: &InputFile,
    sr: u32,
    gain: f32,
    separator: &dyn Separator,
    wanted: &[usize],
    writers: &mut [Writer],
) -> Result<usize> {
    let mut overlap = Overlap::new(separator.chunk_frames(), separator.stems().len());
    let mut decode = std::pin::pin!(decode_path_stereo(&input.path, DecodeOptions::new(sr)));
    while let Some(chunk) = decode.next().await {
        let mut chunk = chunk.with_context(|| format!("decoding {}", input.path.display()))?;
        for s in chunk.left.iter_mut().chain(&mut chunk.right) {
            *s *= gain;
        }
        let ready = overlap.push(chunk, separator)?;
        emit(ready, gain, wanted, writers).await?;
    }
    let ready = overlap.flush(separator)?;
    emit(ready, gain, wanted, writers).await?;
    Ok(overlap.passes)
}

/// Hand one round of finished audio to the writers, undoing the input gain.
async fn emit(
    mut ready: Vec<StereoSamples>,
    gain: f32,
    wanted: &[usize],
    writers: &mut [Writer],
) -> Result<()> {
    let restore = 1.0 / gain;
    for (writer, stem) in writers.iter_mut().zip(wanted) {
        let mut out = std::mem::take(&mut ready[*stem]);
        if out.frames() == 0 {
            continue;
        }
        for s in out.left.iter_mut().chain(&mut out.right) {
            *s *= restore;
        }
        writer.send(out).await?;
    }
    Ok(())
}

/// One stem's output file, fed through a channel.
struct Writer {
    name: &'static str,
    path: PathBuf,
    tx: futures::channel::mpsc::Sender<audio_kit::Result<StereoSamples>>,
    task: tokio::task::JoinHandle<audio_kit::Result<()>>,
    frames: usize,
    /// Running sum of squares over both channels, for the report's RMS. Kept as
    /// a total rather than a buffer of the stem, which is the point of writing
    /// through a channel at all.
    energy: f64,
}

impl Writer {
    /// Infallible on purpose: the file is created by the task, so the one thing
    /// that can go wrong here — an unwritable output directory — is reported by
    /// [`Writer::finish`] along with everything else that happens to the file.
    fn create(name: &'static str, base: &str, sr: u32, output_dir: &Path) -> Self {
        // `<base>.vocals.wav`, so a batch's two stems sort together under the
        // recording they came from and the next stage can select one with a
        // glob.
        let path = output_dir.join(format!("{base}.{name}.wav"));
        let (tx, rx) = futures::channel::mpsc::channel(4);
        let task = tokio::spawn(write_wav_stereo_file(path.clone(), sr, rx));
        Self {
            name,
            path,
            tx,
            task,
            frames: 0,
            energy: 0.0,
        }
    }

    async fn send(&mut self, out: StereoSamples) -> Result<()> {
        self.frames += out.frames();
        self.energy += out
            .left
            .iter()
            .chain(&out.right)
            .map(|s| (*s as f64) * (*s as f64))
            .sum::<f64>();
        // A send only fails once the writer task is gone, and the reason it is
        // gone is the error `finish` will report. So this stops the pass
        // without claiming to know why.
        self.tx
            .send(Ok(out))
            .await
            .map_err(|_| anyhow::anyhow!("writing {} stopped", self.path.display()))
    }

    async fn finish(self, sr: u32) -> Result<StemReport> {
        let Self {
            name,
            path,
            tx,
            task,
            frames,
            energy,
        } = self;
        drop(tx);
        task.await
            .with_context(|| format!("writing {}", path.display()))?
            .with_context(|| format!("writing {}", path.display()))?;
        let samples = (frames * 2).max(1) as f64;
        Ok(StemReport {
            name,
            path,
            secs: frames as f64 / sr as f64,
            rms: (energy / samples).sqrt() as f32,
        })
    }
}

/// The seams: a windowed overlap-add across chunk boundaries, holding one chunk
/// of input and one hop of each stem's output and nothing else.
///
/// A periodic Hann at a 50% hop sums to one, and the accumulated weight is
/// divided out explicitly rather than relied on — so a changed hop degrades the
/// blend instead of changing the recording's loudness.
///
/// **The very first half-chunk is the one place that is not enough**, and it is
/// worth spelling out because the failure is silent and one sample wide. No
/// earlier pass overlaps it, so its accumulated weight is the rising half of the
/// window alone: at frame 0 that weight is *exactly zero*, and dividing a zeroed
/// sample by a guarded zero writes a zero where the recording's first sample
/// should be. So the leading half of the first pass is windowed by **one**
/// instead — a single pass covers it, and the honest reconstruction of a region
/// covered once is the model's own output. The end needs no such treatment: the
/// last pass's frames are always overlapped by the one before it, whose
/// descending tail is what makes their weights sum to one.
struct Overlap {
    chunk: usize,
    hop: usize,
    window: Vec<f32>,
    /// Input decoded but not yet consumed by a forward pass.
    pending: StereoSamples,
    /// Per stem, the second half of the last pass's windowed output — the part
    /// the next pass overlaps with.
    tails: Vec<StereoSamples>,
    /// The window weight already accumulated over those same frames.
    norm_tail: Vec<f32>,
    /// Forward passes so far, which is what the caller reports.
    passes: usize,
}

impl Overlap {
    fn new(chunk: usize, stems: usize) -> Self {
        let hop = chunk / 2;
        Self {
            chunk,
            hop,
            // Periodic rather than symmetric (`i / chunk`, not `i / (chunk-1)`),
            // which is the form that is COLA at a half-chunk hop.
            window: (0..chunk)
                .map(|i| 0.5 - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / chunk as f32).cos())
                .collect(),
            pending: StereoSamples::default(),
            tails: (0..stems)
                .map(|_| StereoSamples {
                    left: vec![0.0; hop],
                    right: vec![0.0; hop],
                })
                .collect(),
            norm_tail: vec![0.0; hop],
            passes: 0,
        }
    }

    /// Take decoded input, returning whatever finished audio it completed —
    /// one entry per stem, in the model's order, each possibly empty.
    fn push(
        &mut self,
        input: StereoSamples,
        separator: &dyn Separator,
    ) -> Result<Vec<StereoSamples>> {
        self.pending.left.extend_from_slice(&input.left);
        self.pending.right.extend_from_slice(&input.right);
        let mut out = self.empty();
        while self.pending.frames() >= self.chunk {
            let chunk = StereoSamples {
                left: self.pending.left[..self.chunk].to_vec(),
                right: self.pending.right[..self.chunk].to_vec(),
            };
            self.advance(&chunk, separator, self.hop, &mut out)?;
            self.pending.left.drain(..self.hop);
            self.pending.right.drain(..self.hop);
        }
        Ok(out)
    }

    /// Run out the input that is left, zero-padding the last chunk.
    ///
    /// Only the frames that came from real input are emitted; the padding's own
    /// output is dropped. A recording shorter than one chunk takes this path
    /// straight away and comes back the length it went in.
    fn flush(&mut self, separator: &dyn Separator) -> Result<Vec<StereoSamples>> {
        let mut out = self.empty();
        while self.pending.frames() > 0 {
            let take = self.pending.frames().min(self.hop);
            let mut chunk = StereoSamples {
                left: self.pending.left.clone(),
                right: self.pending.right.clone(),
            };
            chunk.left.resize(self.chunk, 0.0);
            chunk.right.resize(self.chunk, 0.0);
            self.advance(&chunk, separator, take, &mut out)?;
            self.pending.left.drain(..take);
            self.pending.right.drain(..take);
        }
        Ok(out)
    }

    /// One forward pass: window it, add the previous pass's tail, emit `emit`
    /// frames and keep the rest as the next tail.
    fn advance(
        &mut self,
        chunk: &StereoSamples,
        separator: &dyn Separator,
        emit: usize,
        out: &mut [StereoSamples],
    ) -> Result<()> {
        let stems = separator.separate(chunk)?;
        anyhow::ensure!(
            stems.len() == self.tails.len(),
            "the model returned {} stems where it declares {}",
            stems.len(),
            self.tails.len()
        );

        let Self {
            chunk: chunk_len,
            hop,
            window,
            tails,
            norm_tail,
            passes,
            pending: _,
        } = self;
        let (hop, chunk_len) = (*hop, *chunk_len);
        // The rising half of the window, except on the pass that has nothing
        // before it — see the type's docs for why that exception is not a
        // rounding detail.
        let first = *passes == 0;
        *passes += 1;
        let rise = |i: usize| if first { 1.0 } else { window[i] };

        // The denominator is shared by every stem — it is a property of the
        // window and the hop, not of the audio — so it is computed once and the
        // stems divide by it. With a periodic Hann at a half-chunk hop it is 1
        // at every frame, which is the invariant this makes checkable rather
        // than assumed.
        let norm: Vec<f32> = (0..emit)
            .map(|i| (norm_tail[i] + rise(i)).max(1e-6))
            .collect();

        for (stem, (tail, out)) in stems.iter().zip(tails.iter_mut().zip(out.iter_mut())) {
            anyhow::ensure!(
                stem.frames() == chunk_len,
                "the model returned {} frames for a {}-frame chunk",
                stem.frames(),
                chunk_len
            );
            for (src, tail, dst) in [
                (&stem.left, &mut tail.left, &mut out.left),
                (&stem.right, &mut tail.right, &mut out.right),
            ] {
                dst.extend((0..emit).map(|i| (tail[i] + rise(i) * src[i]) / norm[i]));
                *tail = (hop..chunk_len).map(|i| window[i] * src[i]).collect();
            }
        }
        *norm_tail = window[hop..].to_vec();
        Ok(())
    }

    fn empty(&self) -> Vec<StereoSamples> {
        vec![StereoSamples::default(); self.tails.len()]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A stand-in that puts everything in the first stem and nothing in the
    /// second, so whatever comes out of the overlap-add is comparable with what
    /// went in — sample for sample, not merely in shape.
    ///
    /// That is the property this stage can get wrong without any checkpoint
    /// being involved: an off-by-one in the tail, a window applied twice, or a
    /// flush that emits the padding would all survive a real model and be
    /// inaudible until a corpus was already sliced from the result.
    struct Passthrough {
        chunk: usize,
    }

    impl Separator for Passthrough {
        fn sample_rate(&self) -> u32 {
            44_100
        }
        fn chunk_frames(&self) -> usize {
            self.chunk
        }
        fn stems(&self) -> &'static [&'static str] {
            &["vocals", "instrumental"]
        }
        fn separate(&self, chunk: &StereoSamples) -> Result<Vec<StereoSamples>> {
            assert_eq!(
                chunk.frames(),
                self.chunk,
                "the chunk length is the contract"
            );
            Ok(vec![
                chunk.clone(),
                StereoSamples {
                    left: vec![0.0; self.chunk],
                    right: vec![0.0; self.chunk],
                },
            ])
        }
    }

    /// A ramp and its negation, so a left/right swap or a reversed tail is
    /// visible in the comparison rather than hidden by symmetry.
    ///
    /// **Neither channel starts at zero**, deliberately: the rising window is
    /// exactly 0 at frame 0, so a signal that began there would agree with a
    /// zeroed first sample and hide the one edge case that needed handling.
    fn signal(frames: usize) -> StereoSamples {
        StereoSamples {
            left: (0..frames).map(|i| 0.3 + i as f32 * 0.001).collect(),
            right: (0..frames).map(|i| -0.4 - i as f32 * 0.002).collect(),
        }
    }

    fn drive(frames: usize, chunk: usize, feed: usize) -> Vec<StereoSamples> {
        let model = Passthrough { chunk };
        let mut overlap = Overlap::new(chunk, 2);
        let input = signal(frames);
        let mut got = vec![StereoSamples::default(); 2];
        let mut at = 0;
        while at < frames {
            let end = (at + feed).min(frames);
            let piece = StereoSamples {
                left: input.left[at..end].to_vec(),
                right: input.right[at..end].to_vec(),
            };
            for (acc, part) in got
                .iter_mut()
                .zip(overlap.push(piece, &model).expect("push"))
            {
                acc.left.extend(part.left);
                acc.right.extend(part.right);
            }
            at = end;
        }
        for (acc, part) in got.iter_mut().zip(overlap.flush(&model).expect("flush")) {
            acc.left.extend(part.left);
            acc.right.extend(part.right);
        }
        got
    }

    /// Every length that exercises a different branch of the seam arithmetic,
    /// including the ones a corpus stage will actually meet: a clip shorter than
    /// one forward pass, and a recording that ends mid-chunk.
    #[test]
    fn the_overlap_add_reconstructs_its_input() {
        let chunk = 64;
        for frames in [1, 7, 31, 32, 33, 64, 65, 96, 97, 160, 200, 257] {
            for feed in [1, 5, 64, 1000] {
                let got = drive(frames, chunk, feed);
                let want = signal(frames);
                assert_eq!(
                    got[0].frames(),
                    frames,
                    "{frames} frames in, {} out (fed {feed} at a time)",
                    got[0].frames()
                );
                for (channel, (a, b)) in [
                    ("left", (&got[0].left, &want.left)),
                    ("right", (&got[0].right, &want.right)),
                ] {
                    for (i, (x, y)) in a.iter().zip(b).enumerate() {
                        assert!(
                            (x - y).abs() < 1e-5,
                            "{frames} frames, fed {feed}: {channel}[{i}] is {x}, not {y}"
                        );
                    }
                }
                // The stem the stand-in emptied has to come back empty, not
                // merely quiet: a scrambled stem axis would put the audio here.
                assert!(
                    got[1].left.iter().all(|s| s.abs() < 1e-6),
                    "{frames} frames: the empty stem is not empty"
                );
            }
        }
    }

    /// The window is the periodic Hann, which is the one that sums to unity at
    /// a half-chunk hop. The symmetric form (`i / (chunk - 1)`) is the natural
    /// thing to write and leaves a slow ripple across every seam that no length
    /// check would see.
    #[test]
    fn the_window_sums_to_one_across_a_hop() {
        let overlap = Overlap::new(64, 1);
        for i in 0..overlap.hop {
            let sum = overlap.window[i] + overlap.window[i + overlap.hop];
            assert!((sum - 1.0).abs() < 1e-6, "window[{i}] pair sums to {sum}");
        }
    }

    /// A model that fails part-way through a recording, to drive [`file`]'s
    /// error path.
    struct FailsMidway {
        chunk: usize,
        calls: std::sync::atomic::AtomicUsize,
    }

    impl Separator for FailsMidway {
        fn sample_rate(&self) -> u32 {
            44_100
        }
        fn chunk_frames(&self) -> usize {
            self.chunk
        }
        fn stems(&self) -> &'static [&'static str] {
            &["vocals", "instrumental"]
        }
        fn separate(&self, chunk: &StereoSamples) -> Result<Vec<StereoSamples>> {
            let n = self
                .calls
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            anyhow::ensure!(n < 1, "the model gave up on pass {n}");
            Ok(vec![chunk.clone(), chunk.clone()])
        }
    }

    /// A recording that stopped part-way through is still a valid WAV of the
    /// wrong length, and the batch driver above this only reports that it
    /// skipped the file — so a half-written stem left on disk would enter a
    /// corpus indistinguishable from a short recording. It has to be removed,
    /// and this is the only test that runs [`file`] end to end: peak scan,
    /// writers, failure, cleanup.
    #[tokio::test(flavor = "multi_thread")]
    async fn a_stem_that_failed_part_way_is_not_left_behind() {
        let dir = std::env::temp_dir().join(format!("preprocess-separate-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("scratch dir");

        // Long enough that the second forward pass — the one that fails — comes
        // after the first has already handed finished audio to the writer.
        let input = dir.join("take.wav");
        let audio = signal(3000);
        audio_kit::write_wav_stereo_file(
            &input,
            44_100,
            Box::pin(futures::stream::once(async move { Ok(audio) })),
        )
        .await
        .expect("write the input");

        let model = FailsMidway {
            chunk: 1000,
            calls: std::sync::atomic::AtomicUsize::new(0),
        };
        let err = file(
            &InputFile {
                path: input,
                base: "take".into(),
            },
            &SeparateOptions { stems: Stems::Both },
            &model,
            &dir,
        )
        .await
        .expect_err("the model failed, so the stage must");
        assert!(
            format!("{err:#}").contains("gave up"),
            "the model's own reason should survive: {err:#}"
        );
        for stem in ["vocals", "instrumental"] {
            let path = dir.join(format!("take.{stem}.wav"));
            assert!(!path.exists(), "{} was left behind", path.display());
        }
    }

    /// `Both` is the only selection that writes the accompaniment, and the
    /// names are the model's own — a mismatch here writes nothing at all, which
    /// is why [`file`] refuses an empty selection rather than producing no
    /// files.
    #[test]
    fn stem_selection_matches_the_models_names() {
        assert!(Stems::Vocals.wants("vocals"));
        assert!(!Stems::Vocals.wants("instrumental"));
        assert!(Stems::Instrumental.wants("instrumental"));
        assert!(!Stems::Instrumental.wants("vocals"));
        assert!(Stems::Both.wants("vocals") && Stems::Both.wants("instrumental"));
    }
}
