//! In-process libavfilter graph for streaming mono-`f32` audio filtering.
//!
//! Wraps ffmpeg's `abuffer → <chain> → abuffersink` filter graph so any
//! `avfilter` chain (e.g. `afftdn` de-noise) can be applied to the toolkit's
//! mono-`f32` stream without shelling out to the `ffmpeg` binary. Samples are
//! pushed frame-by-frame and pulled as they become available; the filter's
//! internal look-ahead means [`AudioFilter::process`] may return fewer samples
//! than it was given until [`AudioFilter::flush`] drains the tail.
//!
//! **De-hiss no longer comes through here**, and that is a deliberate
//! narrowing rather than an oversight: `anlmdn` corrupts the heap from ffmpeg
//! 9.0 onward, so [`crate::Denoiser`] computes non-local means itself. What is
//! left leaning on libavfilter is *measurement* — `ebur128` for
//! `preprocess normalize` — where there is nothing to port, no upstream defect,
//! and a rewrite would only be a second implementation of a broadcast standard.

use std::collections::HashMap;

use ffmpeg::util::format::Sample as SampleFormat;
use ffmpeg::util::format::sample::Type as SampleType;
use ffmpeg::util::frame::audio::Audio as AudioFrame;
use ffmpeg::{ChannelLayout, filter};
use ffmpeg_next as ffmpeg;

use crate::Samples;
use crate::error::{AudioError, Result};

/// A streaming libavfilter graph over mono `f32` PCM at a fixed sample rate.
///
/// Input and output are both mono packed `f32` at `sample_rate`; the graph in
/// between is whatever `avfilter` chain description was passed to [`Self::new`].
/// All three halves of that promise are enforced rather than assumed — format
/// and layout by the `aformat` [`Self::new`] appends, the rate by a check in
/// [`Self::drain`] that refuses a chain which resampled behind the caller's
/// back.
pub struct AudioFilter {
    graph: filter::Graph,
    sample_rate: u32,
    /// Presentation timestamp of the next input sample (in `1/sample_rate`
    /// units, i.e. the running sample index) so the graph sees monotonic pts.
    pts: i64,
    /// Whether EOF has been signalled to the source (idempotent flush).
    flushed: bool,
    /// The latest value of every metadata key any output frame has carried.
    /// See [`AudioFilter::metadata`].
    metadata: HashMap<String, String>,
}

impl AudioFilter {
    /// Build a graph applying `filter_desc` (an `avfilter` chain string such as
    /// `"afftdn=nr=12:nf=-30"`) to mono `f32` at `sample_rate`.
    pub fn new(sample_rate: u32, filter_desc: &str) -> Result<Self> {
        crate::decode::ensure_ffmpeg();

        let mut graph = filter::Graph::new();
        let abuffer = filter::find("abuffer").ok_or(AudioError::FilterUnavailable("abuffer"))?;
        let abuffersink =
            filter::find("abuffersink").ok_or(AudioError::FilterUnavailable("abuffersink"))?;

        // Declare the source format: mono packed float at our rate.
        let args = format!(
            "sample_rate={sr}:sample_fmt=flt:channel_layout=mono:time_base=1/{sr}",
            sr = sample_rate
        );
        graph.add(&abuffer, "in", &args)?;
        graph.add(&abuffersink, "out", "")?;

        // Append `aformat` so the sink always emits mono packed float, letting us
        // read plane 0 directly regardless of what the chain negotiates to (the
        // sink's own `sample_fmts` can't be set after init). Link source ("in") →
        // chain → sink ("out"); in parser terms the chain's dangling *input*
        // connects to our sink and its *output* to our source (the confusing
        // in/out convention of `avfilter_graph_parse_ptr`).
        let spec = format!("{filter_desc},aformat=sample_fmts=flt:channel_layouts=mono");
        graph.output("in", 0)?.input("out", 0)?.parse(&spec)?;
        graph.validate()?;

        Ok(Self {
            graph,
            sample_rate,
            pts: 0,
            flushed: false,
            metadata: HashMap::new(),
        })
    }

    /// The latest value a filter in the chain attached to an output frame under
    /// `key`, or `None` if no frame has carried it yet.
    ///
    /// This is how a *measuring* filter reports: `ebur128=metadata=1` passes the
    /// audio through untouched and injects its running reading as frame
    /// metadata under `lavfi.r128.*`, which is the only way that number reaches
    /// a caller — the summary it also prints goes to ffmpeg's log, where an
    /// in-process graph cannot read it.
    ///
    /// Keys are never cleared, only overwritten, so the value survives frames
    /// that carry nothing. A running measurement's final value therefore has to
    /// be read **after [`Self::flush`]**, since the reading that covers the
    /// whole signal is on the last frame out.
    pub fn metadata(&self, key: &str) -> Option<&str> {
        self.metadata.get(key).map(String::as_str)
    }

    /// Push `input` into the graph and return whatever filtered samples are
    /// ready (possibly empty while the filter fills its look-ahead).
    pub fn process(&mut self, input: &[f32]) -> Result<Samples> {
        if !input.is_empty() {
            let mut frame = AudioFrame::new(
                SampleFormat::F32(SampleType::Packed),
                input.len(),
                ChannelLayout::MONO,
            );
            frame.set_rate(self.sample_rate);
            frame.set_pts(Some(self.pts));
            self.pts += input.len() as i64;
            frame.plane_mut::<f32>(0).copy_from_slice(input);

            let mut src = self.graph.get("in").expect("abuffer source present");
            src.source().add(&frame)?;
        }
        self.drain()
    }

    /// Signal end-of-input and drain the filter's remaining (tail) samples.
    /// Idempotent: subsequent calls return empty.
    pub fn flush(&mut self) -> Result<Samples> {
        if !self.flushed {
            let mut src = self.graph.get("in").expect("abuffer source present");
            src.source().flush()?;
            self.flushed = true;
        }
        self.drain()
    }

    /// Pull every currently-available output frame into one `Vec`. Any sink
    /// error (`EAGAIN` when empty, `EOF` when finished) simply ends the drain.
    ///
    /// Every frame's rate is checked against the declared one, which is the
    /// half [`Self::new`]'s `aformat` cannot cover: `sample_fmts` and
    /// `channel_layouts` are pinned there, but a rate is *negotiated* between
    /// neighbouring filters, so a chain holding a resampler simply produces
    /// frames at its own rate. Nothing about them looks wrong — the samples are
    /// finite, the format is `flt`, the layout is mono — and the caller counts
    /// them at `self.sample_rate`, so a duration comes out wrong by the ratio
    /// and nothing says so. Refusing costs one comparison per frame.
    fn drain(&mut self) -> Result<Samples> {
        let mut out = Vec::new();
        let mut frame = AudioFrame::empty();
        let mut sink = self.graph.get("out").expect("abuffersink present");
        while sink.sink().frame(&mut frame).is_ok() {
            let n = frame.samples();
            if n > 0 && frame.rate() != self.sample_rate {
                return Err(AudioError::RateRenegotiated {
                    declared: self.sample_rate,
                    got: frame.rate(),
                });
            }
            for (k, v) in frame.metadata().iter() {
                self.metadata.insert(k.to_string(), v.to_string());
            }
            out.extend_from_slice(&frame.plane::<f32>(0)[..n]);
        }
        Ok(out)
    }
}

/// Building a graph is where a missing filter is discovered, and the tests
/// below `expect()` that rather than stepping around it.
///
/// They used to match [`AudioError::FilterUnavailable`] and `return`, which was
/// wrong in two directions at once. A skip that cannot be told apart from a
/// pass is the worse half — these are the only tests in the crate with an
/// *external* oracle, `-21.1 LUFS` being what `ffmpeg -af ebur128` reports for
/// the same waveform, so silently not running them costs more than any of the
/// rest. But the arm never fired either: `FilterUnavailable` is only ever
/// constructed for the graph's two **endpoints**, `abuffer` and `abuffersink`,
/// while a missing *chain* filter fails inside [`filter::Graph::parse`] and
/// arrives as [`AudioError::Ffmpeg`]. So the skip was unreachable and a real
/// absence already panicked, just with a message that named no filter.
///
/// Measured on ffmpeg 9.0.1: both filters are present, and both assertions run.
#[cfg(test)]
mod tests {
    use super::*;

    fn rms(x: &[f32]) -> f32 {
        if x.is_empty() {
            return 0.0;
        }
        (x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32).sqrt()
    }

    /// A `volume` filter is trivially exact, so this checks the round-trip
    /// (frame in → frame out) end-to-end without depending on afftdn's model.
    #[test]
    fn volume_halves_amplitude() {
        let sr = 48_000u32;
        let mut f = AudioFilter::new(sr, "volume=0.5").expect("build graph");
        let input: Vec<f32> = (0..sr as usize)
            .map(|i| 0.4 * (std::f32::consts::TAU * 220.0 * i as f32 / sr as f32).sin())
            .collect();
        let mut out = f.process(&input).expect("process");
        out.extend(f.flush().expect("flush"));
        assert_eq!(
            out.len(),
            input.len(),
            "a gain is sample-for-sample; anything else is a frame lost or a tail dropped"
        );
        let r = rms(&out) / rms(&input);
        assert!((r - 0.5).abs() < 0.02, "expected ~0.5x amplitude, got {r}");
    }

    /// afftdn (fixed floor) should strongly attenuate steady broadband hiss,
    /// proving the streaming abuffer→afftdn→abuffersink graph actually filters.
    #[test]
    fn afftdn_reduces_hiss() {
        let sr = 48_000u32;
        let mut f =
            AudioFilter::new(sr, "afftdn=nf=-20").expect("this ffmpeg build must provide `afftdn`");
        // Deterministic white-ish noise (no rand dep).
        let mut s: u64 = 0x1234_5678;
        let input: Vec<f32> = (0..sr as usize * 2)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                0.02 * ((s >> 40) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0)
            })
            .collect();
        let mut out = f.process(&input).expect("process");
        out.extend(f.flush().expect("flush"));
        // Compare a settled region well past the filter's warm-up.
        let a = rms(&input[sr as usize..sr as usize + sr as usize / 2]);
        let b = rms(&out[sr as usize..sr as usize + sr as usize / 2]);
        assert!(b < 0.6 * a, "hiss not reduced: in={a} out={b}");
    }

    /// `aformat` pins the sink's format and channel layout, and there is no
    /// third field for the rate — it is negotiated between neighbouring
    /// filters, so a chain holding a resampler simply produces frames at its
    /// own. The failure is total and silent: measured before this check,
    /// `aresample=24000` on a graph declared at 48 kHz returned **24 000
    /// samples for 48 000 pushed**, all finite, all mono `f32`, and a caller
    /// counting them at 48 kHz reads a one-second clip as half a second.
    ///
    /// [`crate::AudioFilter`]'s only in-tree chains are `ebur128` and `afftdn`,
    /// neither of which resamples, so this refuses nothing anyone runs today.
    /// It is here because `normalize`'s `integrated_lufs` names precisely this
    /// hazard as its reason for preferring `ebur128` over `loudnorm` — which
    /// negotiates 192 kHz of its own accord — and that reasoning was written
    /// down and checked by nothing.
    #[test]
    fn a_chain_that_resamples_is_refused_rather_than_miscounted() {
        let sr = 48_000u32;
        let mut f = AudioFilter::new(sr, "aresample=24000").expect("build graph");
        let err = f
            .process(&vec![0.1; sr as usize])
            .expect_err("a chain that changes the rate must not be counted at the declared one");
        assert!(
            matches!(
                err,
                AudioError::RateRenegotiated {
                    declared: 48_000,
                    got: 24_000
                }
            ),
            "{err}"
        );
    }

    /// The other half of the metadata contract: a reading taken early is a
    /// reading of *what has been pushed so far*, and it is a plausible number
    /// rather than an error.
    ///
    /// `a_measuring_filter_reports_through_metadata` pins the two ends — `None`
    /// before anything, a value after the flush — which a meter would satisfy
    /// even if the running value never moved. This pins the middle, and it uses
    /// a signal whose two halves measure ~26 LU apart so that reading at the
    /// wrong moment cannot come out looking right: a constant tone gives nearly
    /// the same number early and late, which is exactly why the mistake is easy
    /// to make and hard to see.
    ///
    /// `integrated_lufs` depends on this, and `preprocess normalize` depends on
    /// `integrated_lufs`: were the value read before the flush, a corpus would
    /// be normalized against a prefix of each recording.
    #[test]
    fn a_reading_taken_before_the_flush_covers_only_what_has_been_pushed() {
        let sr = 48_000u32;
        let mut f = AudioFilter::new(sr, "ebur128=metadata=1:peak=none:framelog=quiet")
            .expect("this ffmpeg build must provide `ebur128`");
        let tone = |secs: usize, amplitude: f32| -> Vec<f32> {
            (0..secs * sr as usize)
                .map(|i| amplitude * (std::f32::consts::TAU * 1000.0 * i as f32 / sr as f32).sin())
                .collect()
        };

        f.process(&tone(2, 0.02)).expect("process the quiet half");
        let mid: f32 = f
            .metadata("lavfi.r128.I")
            .expect("a running reading once a gating block has passed")
            .parse()
            .expect("a number");

        f.process(&tone(2, 0.4)).expect("process the loud half");
        f.flush().expect("flush");
        let final_reading: f32 = f
            .metadata("lavfi.r128.I")
            .expect("an integrated reading after the flush")
            .parse()
            .expect("a number");

        assert!(
            final_reading > mid + 3.0,
            "the reading did not move once the loud half arrived: {mid} then {final_reading}"
        );
    }

    /// A measuring filter reports through frame metadata, and the reading that
    /// covers the whole signal is only there once the graph has been flushed.
    /// Both halves are pinned here, because reading the value too early gives a
    /// plausible number for a prefix of the audio rather than an error.
    #[test]
    fn a_measuring_filter_reports_through_metadata() {
        let sr = 48_000u32;
        let mut f = AudioFilter::new(sr, "ebur128=metadata=1:peak=none:framelog=quiet")
            .expect("this ffmpeg build must provide `ebur128`");
        assert_eq!(f.metadata("lavfi.r128.I"), None, "nothing measured yet");

        // ffmpeg's own `sine` source at its default amplitude of 0.125, which
        // `ffmpeg -af ebur128` puts at -21.1 LUFS.
        let input: Vec<f32> = (0..sr as usize * 3)
            .map(|i| 0.125 * (std::f32::consts::TAU * 1000.0 * i as f32 / sr as f32).sin())
            .collect();
        let mut out = f.process(&input).expect("process");
        out.extend(f.flush().expect("flush"));
        // ebur128 is a pass-through: the audio is the measurement's by-product.
        assert_eq!(
            out.len(),
            input.len(),
            "ebur128 is a pass-through, so it must return every sample it was given"
        );

        let measured: f32 = f
            .metadata("lavfi.r128.I")
            .expect("an integrated reading after the flush")
            .parse()
            .expect("a number");
        assert!(
            (measured - -21.1).abs() < 0.2,
            "expected ~-21.1 LUFS, got {measured}"
        );
    }
}
