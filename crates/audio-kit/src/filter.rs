//! In-process libavfilter graph for streaming mono-`f32` audio filtering.
//!
//! Wraps ffmpeg's `abuffer → <chain> → abuffersink` filter graph so any
//! `avfilter` chain (e.g. `afftdn` de-noise) can be applied to the toolkit's
//! mono-`f32` stream without shelling out to the `ffmpeg` binary. Samples are
//! pushed frame-by-frame and pulled as they become available; the filter's
//! internal look-ahead means [`AudioFilter::process`] may return fewer samples
//! than it was given until [`AudioFilter::flush`] drains the tail.

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
        Ok(self.drain())
    }

    /// Signal end-of-input and drain the filter's remaining (tail) samples.
    /// Idempotent: subsequent calls return empty.
    pub fn flush(&mut self) -> Result<Samples> {
        if !self.flushed {
            let mut src = self.graph.get("in").expect("abuffer source present");
            src.source().flush()?;
            self.flushed = true;
        }
        Ok(self.drain())
    }

    /// Pull every currently-available output frame into one `Vec`. Any sink
    /// error (`EAGAIN` when empty, `EOF` when finished) simply ends the drain.
    fn drain(&mut self) -> Samples {
        let mut out = Vec::new();
        let mut frame = AudioFrame::empty();
        let mut sink = self.graph.get("out").expect("abuffersink present");
        while sink.sink().frame(&mut frame).is_ok() {
            for (k, v) in frame.metadata().iter() {
                self.metadata.insert(k.to_string(), v.to_string());
            }
            let n = frame.samples();
            out.extend_from_slice(&frame.plane::<f32>(0)[..n]);
        }
        out
    }
}

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
        assert!(
            (out.len() as isize - input.len() as isize).unsigned_abs() < sr as usize / 10,
            "length drifted: in={} out={}",
            input.len(),
            out.len()
        );
        let r = rms(&out) / rms(&input);
        assert!((r - 0.5).abs() < 0.02, "expected ~0.5x amplitude, got {r}");
    }

    /// afftdn (fixed floor) should strongly attenuate steady broadband hiss,
    /// proving the streaming abuffer→afftdn→abuffersink graph actually filters.
    #[test]
    fn afftdn_reduces_hiss() {
        let sr = 48_000u32;
        let mut f = match AudioFilter::new(sr, "afftdn=nf=-20") {
            Ok(f) => f,
            // afftdn missing from this ffmpeg build → nothing to test.
            Err(AudioError::FilterUnavailable(_)) => return,
            Err(e) => panic!("build graph: {e}"),
        };
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

    /// A measuring filter reports through frame metadata, and the reading that
    /// covers the whole signal is only there once the graph has been flushed.
    /// Both halves are pinned here, because reading the value too early gives a
    /// plausible number for a prefix of the audio rather than an error.
    #[test]
    fn a_measuring_filter_reports_through_metadata() {
        let sr = 48_000u32;
        let mut f = match AudioFilter::new(sr, "ebur128=metadata=1:peak=none:framelog=quiet") {
            Ok(f) => f,
            // ebur128 missing from this ffmpeg build → nothing to test.
            Err(AudioError::FilterUnavailable(_)) => return,
            Err(e) => panic!("build graph: {e}"),
        };
        assert_eq!(f.metadata("lavfi.r128.I"), None, "nothing measured yet");

        // ffmpeg's own `sine` source at its default amplitude of 0.125, which
        // `ffmpeg -af ebur128` puts at -21.1 LUFS.
        let input: Vec<f32> = (0..sr as usize * 3)
            .map(|i| 0.125 * (std::f32::consts::TAU * 1000.0 * i as f32 / sr as f32).sin())
            .collect();
        let mut out = f.process(&input).expect("process");
        out.extend(f.flush().expect("flush"));
        // ebur128 is a pass-through: the audio is the measurement's by-product.
        assert!(
            (out.len() as isize - input.len() as isize).unsigned_abs() < sr as usize / 10,
            "ebur128 changed the length: in={} out={}",
            input.len(),
            out.len()
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

    /// `anlmdn` — the de-hiss engine `rvc-core` ships — must **preserve** wanted
    /// content, which is the whole reason it's chosen over spectral subtraction
    /// for soft, breathy material. Its hiss *removal* is signal-dependent (it
    /// keys off the broadband self-similarity of real breath/whisper texture,
    /// so it can't be exercised by a synthetic tone) and is covered by manual
    /// evaluation; here we lock in that a structured signal passes through at
    /// ~unity energy.
    ///
    /// **`#[ignore]`d on ffmpeg 9.0.1, because `anlmdn` corrupts the heap there
    /// and an abort takes the whole test binary with it** — the other 33 tests
    /// in this crate never get to report. It is not our graph: the stock
    /// binary does it too, on defaults, with no chain of ours in sight —
    ///
    /// ```text
    /// ffmpeg -cpuflags 0 -filter_threads 1 -f lavfi -i sine=d=1:r=48000 -af anlmdn -f null -
    /// ```
    ///
    /// — so it is neither the SIMD kernels nor slice threading. `af_anlmdn.c`
    /// is byte-identical between n8.0 and n9.0 apart from an `#if HAVE_X86ASM`
    /// guard, which is why this reads as a small out-of-bounds write that older
    /// allocators happened to absorb: it aborts at 16/44.1/48 kHz and passes at
    /// 8/22.05 kHz, i.e. it turns on the slack in the frame buffer rather than
    /// on anything the filter is asked to do. Un-`ignore` when the system
    /// ffmpeg stops reproducing that one-liner; nothing here needs changing.
    #[test]
    #[ignore = "anlmdn corrupts the heap on ffmpeg 9.0.1; see the doc comment"]
    fn anlmdn_preserves_content() {
        let sr = 48_000u32;
        let mut f = match AudioFilter::new(sr, "anlmdn=s=0.008:p=0.002:r=0.006") {
            Ok(f) => f,
            // anlmdn missing from this ffmpeg build → nothing to test.
            Err(AudioError::FilterUnavailable(_)) => return,
            Err(e) => panic!("build graph: {e}"),
        };
        // A clean multi-harmonic tone (no added noise): energy must survive.
        let input: Vec<f32> = (0..sr as usize * 2)
            .map(|i| {
                let t = i as f32 / sr as f32;
                use std::f32::consts::TAU;
                0.2 * ((TAU * 220.0 * t).sin()
                    + 0.5 * (TAU * 440.0 * t).sin()
                    + 0.3 * (TAU * 880.0 * t).sin())
            })
            .collect();
        let mut out = f.process(&input).expect("process");
        out.extend(f.flush().expect("flush"));
        // Energy in a settled interior region should be within a few % of input
        // (anlmdn adds latency, so compare region energy, not sample-aligned).
        let a = rms(&input[sr as usize..sr as usize + sr as usize / 2]);
        let b = rms(&out[sr as usize..sr as usize + sr as usize / 2]);
        assert!(
            (b / a - 1.0).abs() < 0.1,
            "anlmdn altered clean-signal energy: in={a} out={b}"
        );
    }
}
