//! Corpus → training clips.
//!
//! A clip is an audio file beside a transcript, and preparing it means running
//! the two frozen encoders over it: cnhubert and the quantiser turn the audio
//! into semantic tokens, and the prosody BERT turns the transcript into a
//! feature per phoneme. Both are expensive and neither changes during training,
//! so the whole corpus is prepared once, up front.
//!
//! That is also why fine-tuning wants `stt`: a corpus is audio, and this needs
//! audio *with transcripts*.

use std::path::{Path, PathBuf};

use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, TensorData};
use burn_gptsovits::{Hubert, HubertConfig, Quantizer, QuantizerConfig};
use text_kit::Language;
use tts_core::{ANALYSIS_SR, ProsodyEncoder};

use crate::error::{Result, TrainError};

/// Waveform samples per latent frame at the synthesis rate — the decoder's
/// upsample product (`[10, 8, 2, 2, 2]`), which is also the spectrogram hop, so
/// one spectrogram frame is exactly one latent frame.
pub const SAMPLES_PER_FRAME: usize = 640;

/// One prepared example.
#[derive(Debug, Clone)]
pub struct Clip {
    /// Where it came from, for error messages.
    pub source: PathBuf,
    /// Phoneme ids from the transcript.
    pub phones: Vec<u32>,
    /// Prosody, one row of `bert_dim` per phoneme, phone-major.
    pub bert: Vec<f32>,
    pub bert_dim: usize,
    /// Semantic tokens the audio quantised to, at 25 Hz.
    pub tokens: Vec<u32>,
    /// Ground-truth waveform at [`tts_core::OUTPUT_SR`], trimmed to exactly
    /// `frames() * SAMPLES_PER_FRAME` samples.
    ///
    /// Empty unless preparation was asked for it: only `s2` compares generated
    /// audio against the original, and decoding the corpus a second time is
    /// wasted work for an `s1`-only run.
    pub audio: Vec<f32>,
}

impl Clip {
    /// Roughly how long the clip is, from its token count.
    pub fn seconds(&self) -> f32 {
        self.tokens.len() as f32 / 25.0
    }

    /// Latent frames the clip is worth: one per 50 Hz frame, so two per token.
    ///
    /// `enc_p` upsamples the 25 Hz codes to 50 Hz and `enc_q` reads a 50 Hz
    /// spectrogram, so this is the length both sides of the KL term must agree
    /// on. Bounded by the waveform as well when there is one, since ffmpeg's
    /// output length and the token count are rounded independently and the
    /// shorter is the only one both can honour.
    ///
    /// Always even. `enc_p` reaches its length by repeating each 25 Hz code
    /// twice, so it can only ever produce an even count; an odd `frames` would
    /// leave the prior one frame shorter than the posterior and the KL term
    /// would silently compare two different time bases.
    pub fn frames(&self) -> usize {
        let from_tokens = self.tokens.len() * 2;
        let frames = match self.audio.is_empty() {
            true => from_tokens,
            false => from_tokens.min(self.audio.len() / SAMPLES_PER_FRAME),
        };
        frames & !1
    }
}

/// Find `<stem>.wav` / `<stem>.txt` pairs under `dir`.
///
/// A transcript with no audio, or audio with no transcript, is skipped with a
/// warning rather than failing the run — a corpus assembled by hand usually has
/// a few of both, and stopping on the first would be tedious.
///
/// **Both halves of that sentence are warnings, and the second one used to be
/// silent.** The scan walks audio files and looks for the transcript beside
/// each, so a transcript whose audio it never *saw* fell out of the loop
/// entirely — and the two ways that happens are the two a real corpus produces:
/// an extension this list does not carry (`.aac`, `.wma`), and a spelling it
/// does not match, since the comparison is case-sensitive while the recorder
/// that wrote `TAKE01.WAV` is not. Either way the run trains on fewer clips
/// than the corpus holds, which is the failure a count cannot report because
/// nothing ever counted the missing ones.
pub fn pairs(dir: &Path) -> Result<Vec<(PathBuf, PathBuf)>> {
    let mut out = Vec::new();
    let mut transcripts = Vec::new();
    let entries = std::fs::read_dir(dir)
        .map_err(|e| TrainError::Corpus(format!("cannot read {}: {e}", dir.display())))?;
    for entry in entries.flatten() {
        let audio = entry.path();
        let ext = audio.extension().and_then(|e| e.to_str());
        if ext == Some("txt") {
            transcripts.push(audio);
            continue;
        }
        let is_audio =
            ext.is_some_and(|e| matches!(e, "wav" | "mp3" | "flac" | "m4a" | "ogg" | "opus"));
        if !is_audio {
            continue;
        }
        let text = audio.with_extension("txt");
        if text.exists() {
            out.push((audio, text));
        } else {
            tracing::warn!("no transcript beside {}; skipped", audio.display());
        }
    }
    // Sorted rather than left in `read_dir` order: which transcripts went
    // unused is a list a reader compares against their corpus, and a list whose
    // order changes between runs cannot be compared against anything.
    let paired: std::collections::HashSet<PathBuf> =
        out.iter().map(|(_, text)| text.clone()).collect();
    transcripts.retain(|t| !paired.contains(t));
    transcripts.sort();
    for text in &transcripts {
        tracing::warn!("no audio beside {}; skipped", text.display());
    }
    out.sort();
    if out.is_empty() {
        return Err(TrainError::Corpus(format!(
            "no <name>.wav + <name>.txt pairs under {} — \
             `stt` can produce the transcripts",
            dir.display()
        )));
    }
    Ok(out)
}

/// Run the frozen encoders over a corpus.
///
/// `prosody` is optional and its absence is not fatal: zeros cost expressiveness
/// rather than correctness, and a fine-tune without them still adapts.
///
/// `waveforms` additionally decodes each clip at the synthesis rate, which `s2`
/// needs as ground truth and `s1` never looks at. Preparation is the expensive
/// part of a fine-tune — cnhubert, the quantiser and the prosody BERT over every
/// clip — so a `both` run does it once with this on rather than twice.
pub fn prepare<B: Backend>(
    pairs: &[(PathBuf, PathBuf)],
    hubert: &Hubert<B>,
    quantizer: &Quantizer<B>,
    prosody: &mut Option<Box<dyn ProsodyEncoder>>,
    language: Language,
    waveforms: bool,
    device: &B::Device,
) -> Result<Vec<Clip>> {
    let bert_dim = prosody.as_ref().map_or(1024, |p| p.hidden());
    let mut clips = Vec::with_capacity(pairs.len());

    for (audio_path, text_path) in pairs {
        let text = std::fs::read_to_string(text_path)
            .map_err(|e| TrainError::Corpus(format!("{}: {e}", text_path.display())))?;
        let text = text.trim();
        if text.is_empty() {
            tracing::warn!("{} is empty; skipped", text_path.display());
            continue;
        }

        // `None`: the trainer has no Japanese dictionary threaded through yet, so
        // a Japanese corpus errors here rather than being phonemized as something
        // else. Synthesis fetches and opens one in `tts-cli`'s `load_models`;
        // training is the gap, and `--language ja` therefore fails on the first
        // transcript rather than part-way through a run.
        let phonemes = text_kit::phonemize_mixed(text, language, None)?;
        if phonemes.phones.is_empty() {
            tracing::warn!("{} produced no phonemes; skipped", text_path.display());
            continue;
        }

        let audio = crate::audio::read(audio_path, ANALYSIS_SR)?;
        if audio.len() < ANALYSIS_SR as usize / 4 {
            tracing::warn!(
                "{} is under a quarter second; skipped",
                audio_path.display()
            );
            continue;
        }

        let wav: Tensor<B, 2> =
            Tensor::from_data(TensorData::new(audio.clone(), [1, audio.len()]), device);
        let ssl = hubert.forward(wav).swap_dims(1, 2);
        let tokens = crate::ids(quantizer.encode(ssl))?;

        let bert = match (prosody.as_mut(), &phonemes.word2ph) {
            (Some(encoder), Some(word2ph)) => encoder.encode(&phonemes.normalized, word2ph)?.data,
            _ => vec![0.0; bert_dim * phonemes.phones.len()],
        };

        // Trimmed to whole latent frames so the spectrogram, the codes and the
        // waveform all end together — `Spectral::linear` yields exactly
        // `len / hop` frames, so a ragged tail would put `enc_q` and `enc_p` one
        // frame apart and the KL term would compare misaligned sequences.
        let waveform = match waveforms {
            false => Vec::new(),
            true => {
                let mut wav = crate::audio::read(audio_path, tts_core::OUTPUT_SR)?;
                let frames = (tokens.len() * 2).min(wav.len() / SAMPLES_PER_FRAME);
                wav.truncate(frames * SAMPLES_PER_FRAME);
                wav
            }
        };

        clips.push(Clip {
            source: audio_path.clone(),
            phones: phonemes.ids().iter().map(|&i| i as u32).collect(),
            bert,
            bert_dim,
            tokens,
            audio: waveform,
        });
    }

    if clips.is_empty() {
        return Err(TrainError::Corpus(
            "every clip was skipped — nothing to train on".into(),
        ));
    }
    Ok(clips)
}

/// Reject a load that left parameters at their initialised values.
///
/// `load_pytorch_into` allows a partial apply so that a coverage report can be
/// *inspected* rather than a single mismatch aborting the load, and it has no
/// strict mode — so **an empty apply is a success unless somebody looks**. Here
/// that silence is expensive twice over: a randomly-initialised cnhubert
/// extracts garbage into every clip of the corpus, preparation is the expensive
/// half of a fine-tune, and the model that comes out the far end is merely bad
/// rather than broken.
///
/// `missing` is the number that matters — a *model parameter* with no tensor
/// behind it is random weight the encoder will happily run. `unused` is not
/// checkable here and is deliberately not checked: the quantiser is three
/// tensors out of an `s2G*.pth` holding the whole synthesizer, so ~770 are left
/// over by construction.
///
/// `errors` is separate from `missing` because the applier drops a path that
/// failed to apply from *both* lists (`visited && !applied && !skipped &&
/// !errored`), so a shape mismatch — a checkpoint from a later GPT-SoVITS, say
/// — otherwise reads as full coverage while the parameter keeps its initialised
/// value. It is therefore tested *first*: a file whose every tensor failed to
/// apply would otherwise be reported as "0 missing".
///
/// There is deliberately no override. A 209-of-210 load is a corpus prepared by
/// an encoder with a hole in it, and the flag someone would reach for here is
/// the flag that makes the whole check pointless — so the fix is naming the
/// right checkpoint, not passing something.
///
/// What this cannot catch is a right-shaped *wrong* model: RVC's ContentVec is
/// the same architecture with other weights and would pass 210/0. Coverage
/// checks the module tree against the file, never the file against the intent.
fn covered(what: &str, result: &burn_kit::ApplyResult) -> Result<()> {
    if let Some(first) = result.errors.first() {
        return Err(TrainError::Weights(format!(
            "{what}: {} of the checkpoint's tensors could not be applied \
             ({first}) — the file does not match the model",
            result.errors.len(),
        )));
    }
    if result.applied.is_empty() || !result.missing.is_empty() {
        return Err(TrainError::Weights(format!(
            "{what}: {} of {} parameters had no weights in the checkpoint \
             (applied {}) — the file's tensor names do not match the model",
            result.missing.len(),
            result.missing.len() + result.applied.len(),
            result.applied.len(),
        )));
    }
    Ok(())
}

/// Build a `Hubert` and a `Quantizer` for preparation.
///
/// They are frozen, so they exist only long enough to prepare the corpus and are
/// dropped before training starts — which matters on a small card, where they
/// would otherwise sit beside the model being trained.
///
/// Both loads are coverage-checked, because both used to throw their
/// `ApplyResult` away: cnhubert applies 210 parameters and the quantiser 3, and
/// anything short of that is a corpus prepared by a random encoder.
pub fn encoders<B: Backend>(
    hubert_path: &Path,
    s2_path: &Path,
    device: &B::Device,
) -> Result<(Hubert<B>, Quantizer<B>)> {
    // The path is the useful half of either failure, so the loader's own error
    // and the coverage refusal are given the same label rather than one naming
    // the file and the other only the model.
    let what = format!("cnhubert ({})", hubert_path.display());
    let mut hubert = Hubert::<B>::new(&HubertConfig::chinese_base(), device);
    let applied = hubert
        .load_pytorch(hubert_path)
        .map_err(|e| TrainError::Weights(format!("{what}: {e}")))?;
    covered(&what, &applied)?;

    let what = format!("quantiser ({})", s2_path.display());
    let mut quantizer = Quantizer::<B>::new(&QuantizerConfig::default(), 1, device);
    let applied = quantizer
        .load_pytorch(s2_path)
        .map_err(|e| TrainError::Weights(format!("{what}: {e}")))?;
    covered(&what, &applied)?;

    Ok((hubert, quantizer))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    /// A directory that cleans up after itself, named after the test that owns
    /// it so two running side by side cannot collide. No `tempfile`: the
    /// workspace does not carry one, and a corpus scan needs nothing more than
    /// a few empty files.
    struct Corpus(PathBuf);

    impl Corpus {
        fn new(name: &str) -> Self {
            let dir = std::env::temp_dir().join(format!("tts-train-corpus-{name}"));
            let _ = std::fs::remove_dir_all(&dir);
            std::fs::create_dir_all(&dir).expect("temp dir");
            Self(dir)
        }

        /// Touch a file. Contents never matter here — `pairs` reads names.
        fn touch(self, name: &str) -> Self {
            std::fs::write(self.0.join(name), b"").expect("write");
            self
        }

        fn stems(&self, found: &[(PathBuf, PathBuf)]) -> Vec<String> {
            found
                .iter()
                .map(|(a, _)| a.file_name().unwrap().to_string_lossy().into_owned())
                .collect()
        }
    }

    impl Drop for Corpus {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    /// What the scan said while it ran.
    ///
    /// A warning is the entire difference between a corpus that is short and a
    /// corpus that is short *and says so*, and `pairs`'s return value cannot
    /// tell those apart — the file it left out is absent either way. So the
    /// only test that can distinguish them is one that reads the log.
    #[derive(Clone, Default)]
    struct Log(Arc<Mutex<Vec<u8>>>);

    impl std::io::Write for Log {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(buf);
            Ok(buf.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for Log {
        type Writer = Self;
        fn make_writer(&'a self) -> Self {
            self.clone()
        }
    }

    /// Run `f` with a subscriber of our own and hand back what it printed.
    /// Thread-scoped, so tests running in parallel do not capture each other.
    fn logged<T>(f: impl FnOnce() -> T) -> (T, String) {
        let log = Log::default();
        let subscriber = tracing_subscriber::fmt()
            .with_max_level(tracing::Level::WARN)
            .with_ansi(false)
            .with_writer(log.clone())
            .finish();
        let out = tracing::subscriber::with_default(subscriber, f);
        let text = String::from_utf8_lossy(&log.0.lock().unwrap()).into_owned();
        (out, text)
    }

    #[test]
    fn a_transcript_whose_audio_the_scan_never_saw_is_reported() {
        // The two shapes that produce one, side by side with a pair that works:
        // an extension outside the list, and the same extension in the case a
        // field recorder writes. Both leave a `.txt` the loop over audio files
        // never reaches, so before this was fixed the run trained on one clip
        // of three and said nothing at all.
        let dir = Corpus::new("orphan-transcripts")
            .touch("a.wav")
            .touch("a.txt")
            .touch("b.WAV")
            .touch("b.txt")
            .touch("c.aac")
            .touch("c.txt");

        let (found, log) = logged(|| pairs(&dir.0).expect("one pair is enough to train on"));
        assert_eq!(dir.stems(&found), ["a.wav"], "only the lower-case wav pairs");

        // Named individually: a count of "2 skipped" is not something a reader
        // can act on, and acting on it means opening the file that was missed.
        assert!(log.contains("b.txt"), "b.txt went unreported: {log}");
        assert!(log.contains("c.txt"), "c.txt went unreported: {log}");
        assert!(!log.contains("a.txt"), "a.txt was paired: {log}");
    }

    #[test]
    fn audio_with_no_transcript_is_the_half_that_always_warned() {
        // The contrast that makes the test above mean something: this arm has
        // reported itself since the function was written, and a fix that made
        // the other half loud must not have made this one quiet.
        let dir = Corpus::new("orphan-audio").touch("a.wav").touch("b.wav").touch("b.txt");

        let (found, log) = logged(|| pairs(&dir.0).expect("b is a pair"));
        assert_eq!(dir.stems(&found), ["b.wav"]);
        assert!(log.contains("a.wav"), "a.wav went unreported: {log}");
    }

    #[test]
    fn a_corpus_with_no_pairs_at_all_is_an_error_naming_the_tool_that_fixes_it() {
        // Loud, not empty: every clip missing is a mistake about the corpus,
        // where a few missing is a mistake about a few files.
        let dir = Corpus::new("no-pairs").touch("a.wav").touch("notes.md");
        let err = pairs(&dir.0).expect_err("no transcripts, nothing to train on");
        let msg = err.to_string();
        assert!(msg.contains("stt"), "the message must name `stt`: {msg}");
    }

    #[test]
    fn a_stem_carrying_a_dot_keeps_it() {
        // `with_extension` replaces the last suffix only, which is the property
        // this depends on: `take.v2.wav` must look for `take.v2.txt` and not
        // for `take.txt`, which would silently pair two different recordings
        // with one transcript.
        let dir = Corpus::new("dotted-stem")
            .touch("take.v2.wav")
            .touch("take.v2.txt")
            .touch("take.txt");
        let (found, _) = logged(|| pairs(&dir.0).expect("the dotted stem pairs"));
        assert_eq!(found.len(), 1);
        assert!(found[0].1.ends_with("take.v2.txt"), "{:?}", found[0].1);
    }

    #[test]
    fn every_extension_the_scan_claims_to_read_is_actually_read() {
        // One file per accepted extension: the list is a `matches!` arm, so a
        // format dropped from it fails nothing else.
        let mut dir = Corpus::new("extensions");
        for ext in ["wav", "mp3", "flac", "m4a", "ogg", "opus"] {
            dir = dir.touch(&format!("a.{ext}")).touch(&format!("a.{ext}.txt"));
        }
        // Each file above is `a.<ext>`, whose transcript is `a.txt` — write it
        // once rather than six near-identical names.
        std::fs::write(dir.0.join("a.txt"), b"").unwrap();
        let (found, _) = logged(|| pairs(&dir.0).expect("all six read"));
        assert_eq!(found.len(), 6, "{:?}", dir.stems(&found));
    }

    #[test]
    fn the_frame_count_is_even_and_inside_both_of_its_sources() {
        // `s2`'s `micro_step` slices `clip.audio[..frames * SAMPLES_PER_FRAME]`
        // and `clip.tokens[..frames / 2]` with nothing else bounding either, so
        // these three properties are what stand between a ragged corpus and a
        // panic — or worse, a KL term comparing a prior and a posterior one
        // frame apart.
        let clip = |tokens: usize, frames_of_audio: usize| Clip {
            source: PathBuf::from("t.wav"),
            phones: vec![1, 2, 3],
            bert: Vec::new(),
            bert_dim: 0,
            tokens: vec![7; tokens],
            audio: vec![0.0; frames_of_audio * SAMPLES_PER_FRAME],
        };

        // Audio shorter than the tokens claim: the waveform wins, and the odd
        // count it gives is rounded *down* rather than accepted.
        let c = clip(5, 7);
        assert_eq!(c.frames(), 6);
        // No waveform at all (an `s1`-only preparation): the tokens are the
        // only source, and two frames per token is already even.
        let c_no_audio = Clip {
            audio: Vec::new(),
            ..clip(5, 0)
        };
        assert_eq!(c_no_audio.frames(), 10);

        for c in [clip(5, 7), clip(4, 100), clip(1, 1), clip(3, 3), c_no_audio] {
            let frames = c.frames();
            assert_eq!(frames % 2, 0, "odd frame count from {} tokens", c.tokens.len());
            assert!(frames / 2 <= c.tokens.len(), "would slice past the tokens");
            if !c.audio.is_empty() {
                assert!(
                    frames * SAMPLES_PER_FRAME <= c.audio.len(),
                    "would slice past the waveform"
                );
            }
        }
    }

    #[test]
    fn the_mel_floor_is_asked_of_the_spectral_config_that_owns_it() {
        // `tts-cli` refuses `--segment-frames` below this, and the whole point
        // of routing it through a function is that an `n_fft` or `hop` change
        // moves the number without anybody editing a CLI. Pinned against the
        // config rather than against a literal, so this test cannot be the
        // thing that goes stale.
        assert_eq!(
            crate::mel_min_frames(),
            burn_vits::SpectralConfig::gptsovits_v2_32k().min_frames()
        );
    }

    #[test]
    fn a_checkpoint_that_failed_to_apply_is_refused_before_missing_is_consulted() {
        // The defect CLAUDE.md records three workers rediscovering: the applier
        // drops a path that *errored* from `applied` and from `missing` alike,
        // so a file with the right names at the wrong shapes reports zero
        // missing and reads as a clean load. `covered` therefore asks about
        // `errors` first, and this is what pins that order — a report with a
        // shape mismatch and an otherwise perfect count must still be refused.
        use burn::store::ApplyError;

        let report = |applied: Vec<String>, missing: Vec<(String, String)>, errors| ApplyResult {
            applied,
            skipped: Vec::new(),
            missing,
            unused: Vec::new(),
            errors,
        };

        let err = covered(
            "cnhubert",
            &report(
                vec!["a".into()],
                Vec::new(),
                vec![ApplyError::AdapterError {
                    path: "encoder.layers.0.weight".into(),
                    message: "shape mismatch".into(),
                }],
            ),
        )
        .expect_err("an errored apply must not read as full coverage");
        let msg = err.to_string();
        assert!(msg.contains("could not be applied"), "{msg}");
        assert!(msg.contains("encoder.layers.0.weight"), "{msg}");

        // An empty apply is the other silent success: nothing matched, nothing
        // is missing by the applier's own definition, and the encoder runs on
        // its initialised weights over the whole corpus.
        covered("quantiser", &report(Vec::new(), Vec::new(), Vec::new()))
            .expect_err("an empty apply must not read as full coverage");

        // And the ordinary good load still passes.
        covered("cnhubert", &report(vec!["a".into()], Vec::new(), Vec::new()))
            .expect("a complete apply is what this is meant to accept");
    }
}
