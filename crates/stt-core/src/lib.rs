//! Speech recognition for the `voice` toolkit — Whisper, end to end.
//!
//! Audio in as mono `f32` at 16 kHz, [`Segment`]s out. The pipeline is
//! segmentation → log-mel → encoder → greedy decode, and the first step matters
//! more than its size suggests: Whisper is a language model conditioned on
//! audio, and handed a long quiet stretch it will invent fluent sentences to
//! fill it. Feeding it speech-shaped pieces is the standard defence, and the
//! toolkit already has a slicer tuned not to cut soft or breathy passages.
//!
//! Two runtimes sit behind one [`Transcriber`]: the native Burn port, and ONNX
//! Runtime. They are picked by constructor and are otherwise indistinguishable —
//! the decode loop deals in token ids and `f32` logits, which is the most either
//! has to agree on.
//!
//! Nothing here needs the whole recording. Each span is encoded and decoded from
//! scratch — both engines reset their key/value cache and encoder output on
//! [`Transcriber::segment`] — so a caller holding a [`Slicer`] can transcribe
//! and print each clip the moment its audio has arrived. [`Transcriber::transcribe`]
//! is that same loop over a slicer fed in one go, so the batch and streaming
//! paths cannot drift apart.
//!
//! ```no_run
//! # use stt_core::{Transcriber, TranscribeOptions};
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! # let audio: Vec<f32> = vec![];
//! # #[cfg(feature = "tch")] {
//! let mut stt = Transcriber::libtorch("models/whisper".as_ref(), Default::default())?;
//! for segment in stt.transcribe(&audio, &TranscribeOptions::default())? {
//!     println!("[{:.2} -> {:.2}] {}", segment.start, segment.end, segment.text);
//! }
//! # }
//! # Ok(())
//! # }
//! ```

mod burn_engine;
mod decode;
mod engine;
mod error;
mod mel;
#[cfg(feature = "onnx")]
mod onnx_engine;
mod tokenizer;

pub use decode::DecodeOptions;
pub use error::{Result, SttError};
pub use mel::{HOP, SAMPLE_RATE, WINDOW_FRAMES, WINDOW_SAMPLES, WINDOW_SECONDS};
pub use tokenizer::{Tokens, Vocabulary};

use std::path::Path;

use audio_kit::slice::{SliceOptions, Slicer};
use burn_whisper::WhisperConfig;
use engine::Engine;

/// One transcribed stretch of speech.
#[derive(Debug, Clone)]
pub struct Segment {
    /// Seconds from the start of the input.
    pub start: f32,
    pub end: f32,
    pub text: String,
    /// ISO code the model used, detected or forced.
    pub language: String,
}

/// How to cut the input up and what to ask the model for.
#[derive(Debug, Clone)]
pub struct TranscribeOptions {
    /// Where to cut. Defaults match the corpus slicer's: energy is used only to
    /// find long silent gaps, never to gate quiet-but-present sound.
    pub slice: SliceOptions,
    pub decode: DecodeOptions,
}

impl Default for TranscribeOptions {
    fn default() -> Self {
        Self {
            slice: SliceOptions {
                // A segment must fit one encoder window, so unlike the training
                // slicer this one has a hard ceiling rather than 0 ("never split").
                max_clip: WINDOW_SECONDS,
                ..SliceOptions::default()
            },
            decode: DecodeOptions::default(),
        }
    }
}

/// A loaded Whisper model, ready to transcribe.
///
/// Not generic over a compute backend — the backend lives behind the engine
/// trait instead, so picking one is a constructor call and callers never name a
/// Burn type. `&mut self` throughout, because a decode carries key/value caches.
pub struct Transcriber {
    engine: Box<dyn Engine>,
    mel: mel::LogMel,
    tokens: Tokens,
    vocab: Vocabulary,
}

impl Transcriber {
    /// Pair a loaded engine with the JSON that shipped beside the weights.
    fn assemble(engine: Box<dyn Engine>, dir: &Path) -> Result<Self> {
        Ok(Self {
            mel: mel::LogMel::new(engine.mel_bins()),
            engine,
            tokens: Tokens::load(&dir.join("generation_config.json"))?,
            vocab: Vocabulary::load(&dir.join("tokenizer.json"))?,
        })
    }

    /// Native Burn on the CubeCL/CUDA backend.
    #[cfg(feature = "cuda")]
    pub fn cuda(dir: &Path, device: burn_kit::DeviceSpec) -> Result<Self> {
        let device = burn_kit::cuda_device(device)?;
        let cfg = read_config(&dir.join("config.json"))?;
        let engine = burn_kit::guard_init("cuda", || {
            burn_engine::BurnEngine::<burn::backend::Cuda>::load(
                &cfg,
                &dir.join("model.safetensors"),
                &device,
            )
        })??;
        Self::assemble(Box::new(engine), dir)
    }

    /// Native Burn on LibTorch — CUDA, MPS, Vulkan or CPU.
    #[cfg(feature = "tch")]
    pub fn libtorch(dir: &Path, device: burn_kit::DeviceSpec) -> Result<Self> {
        let device = burn_kit::libtorch_device(device)?;
        let cfg = read_config(&dir.join("config.json"))?;
        let engine = burn_kit::guard_init("tch", || {
            burn_engine::BurnEngine::<burn::backend::LibTorch<f32>>::load(
                &cfg,
                &dir.join("model.safetensors"),
                &device,
            )
        })??;
        Self::assemble(Box::new(engine), dir)
    }

    /// Native Burn on WebGPU.
    #[cfg(feature = "wgpu")]
    pub fn wgpu(dir: &Path, device: burn_kit::DeviceSpec) -> Result<Self> {
        let device = burn_kit::wgpu_device(device)?;
        let cfg = read_config(&dir.join("config.json"))?;
        let engine = burn_kit::guard_init("wgpu", || {
            burn_engine::BurnEngine::<burn::backend::Wgpu>::load(
                &cfg,
                &dir.join("model.safetensors"),
                &device,
            )
        })??;
        Self::assemble(Box::new(engine), dir)
    }

    /// ONNX Runtime, from an `optimum`-style two-graph export.
    ///
    /// Takes no device: `ort` picks its own execution provider — CUDA if the
    /// runtime was built with it, else CPU.
    #[cfg(feature = "onnx")]
    pub fn onnx(dir: &Path) -> Result<Self> {
        let cfg = read_config(&dir.join("config.json"))?;
        let engine = onnx_engine::OnnxEngine::load(
            dir,
            cfg.num_mel_bins,
            cfg.decoder_layers,
            cfg.decoder_attention_heads,
            cfg.d_model,
        )?;
        Self::assemble(Box::new(engine), dir)
    }

    /// Transcribe mono 16 kHz audio.
    ///
    /// Held here for callers that already have the whole recording; a filter
    /// should drive [`Slicer`] and [`Transcriber::segment`] itself, which is
    /// what this is.
    pub fn transcribe(&mut self, audio: &[f32], opts: &TranscribeOptions) -> Result<Vec<Segment>> {
        let mut slicer = Slicer::new(SAMPLE_RATE, &opts.slice);
        let mut out = Vec::new();
        let clips = slicer.push(audio).into_iter().chain(slicer.finish());
        for clip in clips {
            let start = clip.start as f32 / SAMPLE_RATE as f32;
            out.extend(self.segment(&clip.samples, start, &opts.decode)?);
        }
        Ok(out)
    }

    /// Transcribe one clip already cut out of the input.
    ///
    /// `start` is the clip's offset in seconds from the beginning of the
    /// recording, so the timings come back absolute however the audio was cut
    /// up. `None` when the model produced nothing but whitespace, which is what
    /// a clip of breath or room tone gives.
    pub fn segment(
        &mut self,
        clip: &[f32],
        start: f32,
        opts: &DecodeOptions,
    ) -> Result<Option<Segment>> {
        let segment = self.window(clip, opts)?;
        if segment.text.trim().is_empty() {
            return Ok(None);
        }
        Ok(Some(Segment {
            start,
            end: start + clip.len() as f32 / SAMPLE_RATE as f32,
            ..segment
        }))
    }

    /// Transcribe one span, which must fit a single 30 s encoder window.
    fn window(&mut self, audio: &[f32], opts: &DecodeOptions) -> Result<Segment> {
        let (frames, data) = self.mel.compute(audio);
        self.engine.encode(&data, frames)?;

        let decoded = decode::greedy(self.engine.as_mut(), &self.tokens, opts)?;
        if decoded.truncated {
            tracing::warn!(
                "a segment hit the {}-token cap and was cut off — raise the token \
                 cap, or split it with a shorter maximum clip length",
                opts.max_tokens
            );
        }

        let language = self
            .tokens
            .languages
            .iter()
            .find(|(_, id)| **id == decoded.language)
            .map(|(code, _)| code.clone())
            .unwrap_or_default();

        Ok(Segment {
            start: 0.0,
            end: 0.0,
            text: self.vocab.decode(&decoded.tokens)?.trim().to_string(),
            language,
        })
    }
}

/// Read the model dimensions from a Hugging Face `config.json`.
///
/// Whisper's sizes are all in that file, so a new checkpoint is a download
/// rather than a code change — including the 80-vs-128 mel split, which the
/// front-end has to agree with.
fn read_config(path: &Path) -> Result<WhisperConfig> {
    let raw = std::fs::read_to_string(path)?;
    let json: serde_json::Value = serde_json::from_str(&raw)?;
    let get = |key: &str| -> Result<usize> {
        json.get(key)
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .ok_or_else(|| SttError::Config(format!("config.json has no `{key}`")))
    };
    Ok(WhisperConfig {
        num_mel_bins: get("num_mel_bins")?,
        max_source_positions: get("max_source_positions")?,
        max_target_positions: get("max_target_positions")?,
        d_model: get("d_model")?,
        encoder_attention_heads: get("encoder_attention_heads")?,
        encoder_layers: get("encoder_layers")?,
        encoder_ffn_dim: get("encoder_ffn_dim")?,
        decoder_attention_heads: get("decoder_attention_heads")?,
        decoder_layers: get("decoder_layers")?,
        decoder_ffn_dim: get("decoder_ffn_dim")?,
        vocab_size: get("vocab_size")?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::Engine;
    use std::sync::{Arc, Mutex};

    /// What reached the model, shared with the test that built the stub.
    ///
    /// `Transcriber` owns its engine as a `Box<dyn Engine>`, so there is no
    /// getting it back; a handle taken before construction is how a test reads
    /// off what the segmentation glue actually handed downstream.
    #[derive(Default)]
    struct Recorder {
        /// `(frames, mel length)` per [`Engine::encode`] call, in order.
        encoded: Vec<(usize, usize)>,
    }

    /// A canned logit table standing in for a Whisper checkpoint.
    ///
    /// [`Engine`] is the whole boundary between the decode loop and a runtime —
    /// token ids in, `f32` logits out — so the segmentation glue above it can be
    /// exercised with no weights at all. That is not a convenience: it is the
    /// reason [`Transcriber`] is not generic over a backend, and this module is
    /// what cashes that in.
    ///
    /// One script per clip, read in the order [`Engine::encode`] is called, so a
    /// segment's text names the clip it came from and a test can tell two
    /// segments apart without a model.
    struct StubEngine {
        /// Token ids for clip 0, clip 1, ... A clip past the end of this list —
        /// or past the end of its own script — emits `eot`, which is the empty
        /// transcript a clip of room tone gives.
        scripts: Vec<Vec<u32>>,
        vocab_size: usize,
        log: Arc<Mutex<Recorder>>,
        /// Which script the current decode reads, and how far into it.
        clip: usize,
        emitted: usize,
    }

    impl StubEngine {
        fn new(scripts: Vec<Vec<u32>>, log: Arc<Mutex<Recorder>>) -> Self {
            Self {
                scripts,
                vocab_size: VOCAB_SIZE,
                log,
                clip: 0,
                emitted: 0,
            }
        }
    }

    impl Engine for StubEngine {
        fn mel_bins(&self) -> usize {
            MEL_BINS
        }

        fn encode(&mut self, mel: &[f32], frames: usize) -> Result<()> {
            let mut log = self.log.lock().unwrap();
            log.encoded.push((frames, mel.len()));
            self.clip = log.encoded.len() - 1;
            drop(log);
            self.emitted = 0;
            Ok(())
        }

        fn step(&mut self, _tokens: &[u32]) -> Result<Vec<f32>> {
            let next = self
                .scripts
                .get(self.clip)
                .and_then(|s| s.get(self.emitted))
                .copied()
                .unwrap_or(EOT);
            self.emitted += 1;
            // One-hot: greedy decoding picks the maximum, so this is a script.
            let mut logits = vec![0.0f32; self.vocab_size];
            logits[next as usize] = 1.0;
            Ok(logits)
        }

        fn restart(&mut self) {
            self.emitted = 0;
        }
    }

    /// Control-token ids for the stub, chosen above the toy vocabulary so a
    /// generated word id is never one of them.
    const SOT: u32 = 100;
    const EOT: u32 = 101;
    const TRANSCRIBE: u32 = 102;
    const TRANSLATE: u32 = 103;
    const NO_TIMESTAMPS: u32 = 104;
    const EN: u32 = 105;
    const VOCAB_SIZE: usize = 128;
    const MEL_BINS: usize = 80;

    /// Words 1..=8, so a clip's index can be read straight off its transcript.
    const WORDS: [&str; 8] = [
        "one", "two", "three", "four", "five", "six", "seven", "eight",
    ];

    /// A minimal `tokenizer.json`, written and loaded.
    ///
    /// A `WordLevel` model is enough: the segmentation glue only ever asks the
    /// vocabulary to turn ids back into text, and BPE merges are
    /// `tokenizer.rs`'s business rather than this module's.
    fn vocabulary() -> Vocabulary {
        let dir = std::env::temp_dir().join("stt-core-segmentation-tests");
        std::fs::create_dir_all(&dir).expect("temp dir");
        let path = dir.join("tokenizer.json");
        let entries: Vec<String> = WORDS
            .iter()
            .enumerate()
            .map(|(i, w)| format!("{:?}:{}", w, i + 1))
            .collect();
        let json = format!(
            r#"{{"version":"1.0","truncation":null,"padding":null,"added_tokens":[],
                 "normalizer":null,"pre_tokenizer":null,"post_processor":null,"decoder":null,
                 "model":{{"type":"WordLevel","vocab":{{"[UNK]":0,{}}},"unk_token":"[UNK]"}}}}"#,
            entries.join(",")
        );
        std::fs::write(&path, json).expect("write tokenizer.json");
        Vocabulary::load(&path).expect("load tokenizer.json")
    }

    fn control_tokens() -> Tokens {
        Tokens {
            sot: SOT,
            eot: EOT,
            transcribe: TRANSCRIBE,
            translate: TRANSLATE,
            no_timestamps: NO_TIMESTAMPS,
            languages: [("en".to_string(), EN)].into_iter().collect(),
            suppress: Vec::new(),
            suppress_at_start: Vec::new(),
        }
    }

    /// A [`Transcriber`] over a stub scripting `WORDS[i]` for clip `i`, plus the
    /// handle onto what it encodes.
    fn transcriber(clips: usize) -> (Transcriber, Arc<Mutex<Recorder>>) {
        scripted((0..clips).map(|i| vec![i as u32 + 1]).collect())
    }

    /// The same, with the token script for each clip given explicitly — an empty
    /// script is a clip that decodes to nothing.
    fn scripted(scripts: Vec<Vec<u32>>) -> (Transcriber, Arc<Mutex<Recorder>>) {
        let log = Arc::new(Mutex::new(Recorder::default()));
        let stt = Transcriber {
            mel: mel::LogMel::new(MEL_BINS),
            engine: Box::new(StubEngine::new(scripts, Arc::clone(&log))),
            tokens: control_tokens(),
            vocab: vocabulary(),
        };
        (stt, log)
    }

    /// Decode options that force a language rather than detecting one: which
    /// language a model reports is `decode.rs`'s business, and forcing it keeps
    /// the stub to a single scripted answer per clip.
    fn decode_opts() -> DecodeOptions {
        DecodeOptions {
            language: Some("en".to_string()),
            ..DecodeOptions::default()
        }
    }

    fn options(slice: SliceOptions) -> TranscribeOptions {
        TranscribeOptions {
            slice,
            decode: decode_opts(),
        }
    }

    /// `secs` of a half-scale square wave: -6 dBFS, far above any `silence_db`
    /// a test here sets.
    fn voiced(secs: f32) -> Vec<f32> {
        let n = (secs * SAMPLE_RATE as f32) as usize;
        (0..n)
            .map(|i| if i.is_multiple_of(2) { 0.5 } else { -0.5 })
            .collect()
    }

    fn silence(secs: f32) -> Vec<f32> {
        vec![0.0; (secs * SAMPLE_RATE as f32) as usize]
    }

    /// Drive [`Slicer`] and [`Transcriber::segment`] the way the filter does,
    /// recording **which chunk** each segment came back on.
    ///
    /// This is `stt-cli`'s bare invocation minus the I/O. The claim in this
    /// module's own docs is that a caller holding a [`Slicer`] can transcribe
    /// each clip *the moment its audio has arrived*, and a chunk index is the
    /// only way to state "the moment" as an assertion. Returns the segments
    /// paired with that index, and the index end-of-stream itself carries.
    fn stream(
        stt: &mut Transcriber,
        audio: &[f32],
        opts: &TranscribeOptions,
        chunk: usize,
    ) -> (Vec<(usize, Segment)>, usize) {
        let mut slicer = Slicer::new(SAMPLE_RATE, &opts.slice);
        let mut out = Vec::new();
        let mut at = 0usize;
        for part in audio.chunks(chunk) {
            for clip in slicer.push(part) {
                let start = clip.start as f32 / SAMPLE_RATE as f32;
                if let Some(s) = stt.segment(&clip.samples, start, &opts.decode).unwrap() {
                    out.push((at, s));
                }
            }
            at += 1;
        }
        // `at` now indexes one past the last chunk, which is end of stream.
        for clip in slicer.finish() {
            let start = clip.start as f32 / SAMPLE_RATE as f32;
            if let Some(s) = stt.segment(&clip.samples, start, &opts.decode).unwrap() {
                out.push((at, s));
            }
        }
        (out, at)
    }

    /// The default `TranscribeOptions` with `max_clip` left where it is, which
    /// is what every streaming test below wants.
    fn sentence_opts() -> TranscribeOptions {
        TranscribeOptions {
            decode: decode_opts(),
            ..TranscribeOptions::default()
        }
    }

    /// "Nothing here needs the whole recording" — the claim this module's
    /// opening paragraph makes, and the reason `stt` stopped reading the whole
    /// of stdin before transcribing anything.
    ///
    /// A voiced-silence-voiced stream: the first segment has to come back while
    /// the second sentence is still arriving, not at end of stream.
    #[test]
    fn a_segment_is_emitted_before_the_stream_ends() {
        let mut audio = voiced(2.0);
        audio.extend(silence(1.0));
        audio.extend(voiced(2.0));

        // 0.25 s a chunk, so a chunk index reads as a quarter second and the
        // whole 5 s is 20 chunks.
        const CHUNK: usize = SAMPLE_RATE as usize / 4;
        let opts = sentence_opts();
        let (mut stt, _) = transcriber(2);
        let (emitted, eos) = stream(&mut stt, &audio, &opts, CHUNK);

        assert_eq!(eos, 20, "5 s in 0.25 s chunks");
        assert_eq!(
            emitted.len(),
            2,
            "one segment per sentence: {:?}",
            emitted.iter().map(|(_, s)| &s.text).collect::<Vec<_>>()
        );
        assert_eq!(emitted[0].1.text, "one");

        // A run is final once `max(min_silence, 2 * pad)` of silence has
        // followed it — 0.3 s here — so the first sentence closes inside the 1 s
        // gap, chunks before the second sentence has even finished arriving.
        // Chunk 12 is 3 s in: the gap is over and the second sentence is 0 s
        // old, so anything at or past it is end-of-stream behaviour wearing a
        // smaller number.
        let (at, _) = emitted[0];
        assert!(
            at < 12,
            "the first segment must land inside the silent gap, not at chunk {at} of {eos}"
        );
        assert_eq!(
            emitted[1].0, eos,
            "the last sentence has no silence after it, so only `finish` can close it"
        );
    }

    /// What the assertion above is worth, written as code rather than as prose:
    /// buffering the input first puts **every** segment at end of stream.
    ///
    /// This is the shape `stt` had before it streamed. If [`Slicer`]'s lookahead
    /// ever stopped being bounded — or a future edit re-buffered the input — the
    /// test above would report the index this one does, and fail.
    #[test]
    fn buffering_the_whole_input_first_emits_nothing_until_the_end() {
        let mut audio = voiced(2.0);
        audio.extend(silence(1.0));
        audio.extend(voiced(2.0));

        const CHUNK: usize = SAMPLE_RATE as usize / 4;
        let opts = sentence_opts();
        let (mut stt, _) = transcriber(2);

        // The old shape: accumulate every chunk, then transcribe.
        let mut buffered = Vec::new();
        let mut at = 0usize;
        for part in audio.chunks(CHUNK) {
            buffered.extend_from_slice(part);
            at += 1;
        }
        let segments = stt.transcribe(&buffered, &opts).unwrap();

        assert_eq!(at, 20);
        assert_eq!(segments.len(), 2);
        // Both arrive at chunk 20, so `at < 12` above is exactly the assertion
        // this shape fails.
        assert!(
            at >= 12,
            "a buffered read cannot beat end of stream, and this is the number the \
             streaming test refuses"
        );
    }

    /// "[`Transcriber::transcribe`] is that same loop over a slicer fed in one
    /// go, so the batch and streaming paths cannot drift apart."
    ///
    /// Same audio and same options through both, at chunk sizes from one sample
    /// to larger than the recording: identical text and identical timings, bit
    /// for bit.
    #[test]
    fn the_batch_path_is_the_streaming_loop_fed_in_one_go() {
        let mut audio = voiced(1.6);
        audio.extend(silence(0.8));
        audio.extend(voiced(2.1));
        audio.extend(silence(0.9));
        audio.extend(voiced(1.4));

        let opts = sentence_opts();
        let (mut batch, _) = transcriber(3);
        let want = batch.transcribe(&audio, &opts).unwrap();
        assert_eq!(want.len(), 3, "three sentences, got {want:?}");
        assert_eq!(want[0].text, "one");
        assert_eq!(want[2].text, "three");

        for chunk in [1usize, 137, 4000, 16_000, 1_000_000] {
            let (mut stt, _) = transcriber(3);
            let (got, _) = stream(&mut stt, &audio, &opts, chunk);
            assert_eq!(got.len(), want.len(), "chunk {chunk}");
            for (a, (_, b)) in want.iter().zip(&got) {
                assert_eq!(a.text, b.text, "chunk {chunk}");
                assert_eq!(a.start.to_bits(), b.start.to_bits(), "chunk {chunk}");
                assert_eq!(a.end.to_bits(), b.end.to_bits(), "chunk {chunk}");
                assert_eq!(a.language, b.language, "chunk {chunk}");
            }
        }
    }

    /// A run that never falls silent is split at `max_clip`, and every piece
    /// fits the cap.
    ///
    /// The cap is not a tuning choice at this level: one segment has to fit one
    /// encoder window, which is what [`TranscribeOptions::default`] sets it to.
    #[test]
    fn max_clip_splits_a_run_that_never_falls_silent() {
        let audio = voiced(8.0);
        let opts = options(SliceOptions {
            max_clip: 3.0,
            min_clip: 0.5,
            ..SliceOptions::default()
        });

        let (mut stt, _) = transcriber(8);
        let segments = stt.transcribe(&audio, &opts).unwrap();
        assert!(
            segments.len() >= 3,
            "8 s at a 3 s cap needs at least three pieces, got {}",
            segments.len()
        );
        for s in &segments {
            assert!(
                s.end - s.start <= 3.0 + 1e-4,
                "segment {:.4}..{:.4} exceeds the 3 s cap",
                s.start,
                s.end
            );
        }
        // Contiguous and complete: a split must not delete audio.
        assert!(segments[0].start.abs() < 1e-6);
        for w in segments.windows(2) {
            assert_eq!(
                w[0].end.to_bits(),
                w[1].start.to_bits(),
                "a gap between {:?} and {:?}",
                w[0],
                w[1]
            );
        }
        assert!(
            (segments.last().unwrap().end - 8.0).abs() < 1e-3,
            "the pieces must cover the whole run, ended at {:.4}",
            segments.last().unwrap().end
        );
    }

    /// The one documented divergence, **and it is expected** — do not file it.
    ///
    /// [`Transcriber`] drives [`Slicer`] on *both* of its own paths, so `stt
    /// convert` and the bare filter cut identically (the test above). What they
    /// both diverge from is [`audio_kit::slice::slice`], the whole-recording
    /// slicer, and only on speech that never pauses: `Slicer` has to decide
    /// without knowing how long the run will turn out to be, so it cuts at the
    /// quietest frame of the window it has, where `slice` knows the full length
    /// and balances the split across it. Not buffering the recording is the
    /// point of the type, and this is its price.
    ///
    /// **Which of the two cuts *later* is not the rule, and assuming it is was
    /// wrong here.** `Slicer`'s own docs say it cuts "as late as `max_clip`
    /// allows", which reads as a claim about the knife and is a claim about the
    /// clock: the decision is deferred until `max_clip` has already elapsed, but
    /// the cut then lands wherever the search window is quietest. On the uniform
    /// signal below that is 29 920 against `slice`'s 32 000 — *earlier*. The
    /// numbers are derived in the body rather than remembered.
    ///
    /// Neither loses a sample, which is the property that actually matters.
    #[test]
    fn a_gapless_run_cuts_differently_from_the_whole_recording_slicer() {
        let audio = voiced(8.0);
        let slice_opts = SliceOptions {
            max_clip: 3.0,
            min_clip: 0.5,
            ..SliceOptions::default()
        };

        let (mut stt, _) = transcriber(8);
        let streamed = stt.transcribe(&audio, &options(slice_opts)).unwrap();
        let batch = audio_kit::slice::slice(&audio, SAMPLE_RATE, &slice_opts);

        let streamed_ranges: Vec<(usize, usize)> = streamed
            .iter()
            .map(|s| {
                (
                    (s.start * SAMPLE_RATE as f32).round() as usize,
                    (s.end * SAMPLE_RATE as f32).round() as usize,
                )
            })
            .collect();
        assert_ne!(
            streamed_ranges, batch,
            "this test exists to record that the two disagree on gapless speech. If \
             they have been made to agree, that is news — rewrite this test rather \
             than deleting it"
        );

        // Where it shows is the first cut, and neither direction is the rule:
        // `slice` halves 128 000 samples recursively (128 000 -> 64 000 ->
        // 32 000, the first division under the 48 000-sample cap), so it lands
        // on 32 000. `Slicer` force-cuts inside
        // `[start + margin, start + max_clip)` = `[12 000, 48 000)`, and a
        // uniform signal has no quietest frame — the tie breaks toward that
        // window's own centre, 30 000, whose nearest hop boundary at or below
        // it is 29 920. So the streaming cut here is *earlier*.
        //
        // "Cutting late" in `Slicer`'s own docs is about *when* the decision is
        // taken — only once `max_clip` has already elapsed — not about where the
        // knife lands.
        assert_eq!(streamed_ranges[0], (0, 29_920));
        assert_eq!(batch[0], (0, 32_000));

        // Both still partition the same 8 s with no gap and no overlap, which is
        // the property a divergence must not cost.
        for (name, ranges) in [("streaming", &streamed_ranges), ("batch", &batch)] {
            assert_eq!(ranges[0].0, 0, "{name} must start at the first sample");
            assert_eq!(
                ranges.last().unwrap().1,
                audio.len(),
                "{name} must reach the last sample"
            );
            for w in ranges.windows(2) {
                assert_eq!(w[0].1, w[1].0, "{name} must not delete audio");
            }
        }
    }

    /// Every clip reaches the model as one whole encoder window, however short
    /// it was.
    ///
    /// The front end pads, so the encoder always sees the 3000 frames its
    /// positional table is built for — something `Transcriber` rests on and
    /// never states, since a short window would index that table short.
    #[test]
    fn a_short_clip_is_padded_to_a_whole_encoder_window() {
        let mut audio = voiced(1.2);
        audio.extend(silence(0.8));
        audio.extend(voiced(1.1));

        let opts = options(SliceOptions {
            max_clip: WINDOW_SECONDS,
            min_clip: 0.5,
            ..SliceOptions::default()
        });
        let (mut stt, log) = transcriber(2);
        let segments = stt.transcribe(&audio, &opts).unwrap();
        assert_eq!(segments.len(), 2);
        for s in &segments {
            assert!(s.end - s.start < 2.0, "these clips are short: {s:?}");
        }

        assert_eq!(
            log.lock().unwrap().encoded,
            vec![(WINDOW_FRAMES, MEL_BINS * WINDOW_FRAMES); 2],
            "3000 frames of 80 mel-major bands, whatever came in"
        );
    }

    /// A clip of breath or room tone decodes to nothing, and nothing is what
    /// comes back — the `None` [`Transcriber::segment`] documents.
    #[test]
    fn a_clip_that_decodes_to_nothing_yields_no_segment() {
        let mut audio = voiced(1.2);
        audio.extend(silence(0.8));
        audio.extend(voiced(1.1));

        let opts = options(SliceOptions {
            max_clip: WINDOW_SECONDS,
            min_clip: 0.5,
            ..SliceOptions::default()
        });
        // Clip 0 scripts a word; clip 1 scripts nothing, so `greedy` stops at
        // `eot` with an empty token list and the vocabulary decodes it to "".
        let (mut stt, log) = scripted(vec![vec![1], vec![]]);
        let segments = stt.transcribe(&audio, &opts).unwrap();
        assert_eq!(segments.len(), 1, "the empty clip must be dropped");
        assert_eq!(segments[0].text, "one");
        // ...and it was still encoded, so this is the decode dropping it rather
        // than the slicer never producing it.
        assert_eq!(log.lock().unwrap().encoded.len(), 2);
    }

    /// Timings come back absolute however the audio was cut up, and the language
    /// as an ISO code rather than a token id.
    #[test]
    fn timings_are_absolute_and_the_language_is_an_iso_code() {
        let clip = voiced(1.5);
        let (mut stt, _) = transcriber(1);
        let s = stt
            .segment(&clip, 12.5, &decode_opts())
            .unwrap()
            .expect("clip 0 scripts a word");
        assert_eq!(s.text, "one");
        assert_eq!(s.language, "en");
        assert!((s.start - 12.5).abs() < 1e-6);
        assert!(
            (s.end - 14.0).abs() < 1e-6,
            "1.5 s starting at 12.5 ends at 14.0, got {}",
            s.end
        );
    }

    /// The cap is the one knob this engine changes from the corpus slicer's
    /// defaults, and it changes it away from "never split".
    ///
    /// `0` there means a run is never cut, which a recogniser cannot have: a
    /// segment that does not fit one encoder window is silently truncated by the
    /// front end (the test below).
    #[test]
    fn the_default_cap_is_the_encoder_window_not_never_split() {
        let d = SliceOptions::default();
        assert_eq!(d.max_clip, 0.0, "the corpus slicer never splits by default");

        let opts = TranscribeOptions::default();
        assert_eq!(opts.slice.max_clip, WINDOW_SECONDS);
        assert_eq!(opts.slice.max_clip, 30.0);
        // Every other knob is deliberately the corpus slicer's own.
        assert_eq!(opts.slice.silence_db, d.silence_db);
        assert_eq!(opts.slice.min_silence, d.min_silence);
        assert_eq!(opts.slice.min_clip, d.min_clip);
        assert_eq!(opts.slice.pad, d.pad);
    }

    /// **A clip longer than one encoder window is transcribed short, and its
    /// `end` still reports the length that went in.**
    ///
    /// `mel::LogMel::compute` takes `&audio[..WINDOW_SAMPLES]`, while
    /// [`Transcriber::segment`] derives `end` from the clip it was handed — so
    /// audio past 30 s reaches no model and the timing claims it did. `stt-cli`
    /// cannot reach this (`SttArgs::verify` refuses a `--max-clip` above
    /// [`WINDOW_SECONDS`]), but `stt-core` is a library: `segment` takes the clip
    /// directly, and `SliceOptions`' own default `max_clip` of `0.0` means
    /// "never split", so `transcribe` reaches it too.
    ///
    /// Pinned as behaviour rather than fixed. The assertions below are what a
    /// fix would have to change, and which way — reject the clip, or report the
    /// truncated `end` — is a decision about the public API rather than a
    /// repair.
    #[test]
    fn a_clip_past_the_encoder_window_is_truncated_but_reported_whole() {
        let clip = voiced(45.0);
        let (mut stt, log) = transcriber(1);
        let s = stt
            .segment(&clip, 0.0, &decode_opts())
            .unwrap()
            .expect("clip 0 scripts a word");
        assert!(
            (s.end - 45.0).abs() < 1e-3,
            "`end` follows the clip's length, not what was encoded: {}",
            s.end
        );
        // ...while only the first 30 s of it reached the model.
        assert_eq!(
            log.lock().unwrap().encoded,
            vec![(WINDOW_FRAMES, MEL_BINS * WINDOW_FRAMES)]
        );

        // And the corpus slicer's own default — `max_clip: 0.0`, never split —
        // is what lets a library caller reach it through `transcribe` too.
        let (mut stt, _) = transcriber(1);
        let segments = stt
            .transcribe(&clip, &options(SliceOptions::default()))
            .unwrap();
        assert_eq!(segments.len(), 1);
        assert!(
            segments[0].end > WINDOW_SECONDS + 10.0,
            "one uncut segment of {:.2} s",
            segments[0].end
        );
    }

    /// A long quiet stretch never reaches the model at all.
    ///
    /// That is the whole reason segmentation is in front of the decoder: handed
    /// silence, Whisper invents fluent sentences to fill it.
    #[test]
    fn silence_never_reaches_the_model() {
        let opts = sentence_opts();
        let (mut stt, log) = transcriber(1);
        assert!(stt.transcribe(&silence(4.0), &opts).unwrap().is_empty());
        assert!(stt.transcribe(&[], &opts).unwrap().is_empty());
        assert!(
            log.lock().unwrap().encoded.is_empty(),
            "the slicer must not hand the decoder a quiet stretch to fill"
        );
    }
}
