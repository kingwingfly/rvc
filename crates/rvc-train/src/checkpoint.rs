//! What a saved model is on disk, and how `--resume` finds it again.
//!
//! A checkpoint is a family of three files sharing one stem:
//!
//! ```text
//! voice.safetensors        the deployable generator (the EMA when enabled)
//! voice.raw.safetensors    its raw (non-EMA) live twin      [only with EMA on]
//! voice.disc.safetensors   the discriminator that co-evolved with that twin
//! ```
//!
//! The best-so-far family carries a fourth member, `voice.best.json`, recording
//! what it scored — the one piece of state that has to outlive the process.
//!
//! `--resume` prefers the raw twin: it is the generator that actually faced the
//! saved discriminator, so `raw-G <-> live-D` continues the adversarial game
//! faithfully, whereas the EMA is a smoothed average that never itself faced D.
//! Every checkpoint has the same shape — the *best* one is just this family under
//! a `checkpoint/` subdirectory with a `.best` stem — so anything the trainer
//! writes can be deployed or resumed the same way.

use std::error::Error;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result, anyhow};
use burn_rvc::{MultiPeriodDiscriminator, Synthesizer};

use burn::tensor::backend::Backend;

const SUFFIX: &str = "safetensors";

/// The file family of one saved model.
///
/// Members are addressed by stem rather than [`Path::with_extension`], which sees
/// only the last dot: on `voice.best.raw.safetensors` it would strip `.best`
/// along with `.raw` and silently point at a different run's files.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Checkpoint {
    dir: PathBuf,
    stem: String,
}

impl Checkpoint {
    /// The family a generator path belongs to. Accepts the output name with or
    /// without its suffix (`models/voice`, `models/voice.safetensors`), the raw
    /// twin and the discriminator sidecar — all name the same family, so pointing
    /// `--resume` at any member of a run resolves to that run.
    pub fn new(generator: &Path) -> Self {
        let dir = generator.parent().unwrap_or(Path::new("")).to_path_buf();
        let name = generator
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("voice");
        let stem = name.strip_suffix(&format!(".{SUFFIX}")).unwrap_or(name);
        let stem = stem.strip_suffix(".raw").unwrap_or(stem);
        Self {
            dir,
            stem: stem.strip_suffix(".disc").unwrap_or(stem).to_string(),
        }
    }

    /// The best-so-far sibling: `models/voice…` -> `models/checkpoint/voice.best…`.
    /// Its own subdirectory keeps it from ever colliding with the final output.
    pub fn best(&self) -> Self {
        Self {
            dir: self.dir.join("checkpoint"),
            stem: format!("{}.best", self.stem),
        }
    }

    /// The deployable generator — what `rvc convert -m` takes.
    pub fn generator(&self) -> PathBuf {
        self.member("")
    }

    /// The raw (non-EMA) live twin.
    pub fn raw(&self) -> PathBuf {
        self.member(".raw")
    }

    /// The discriminator sidecar.
    pub fn disc(&self) -> PathBuf {
        self.member(".disc")
    }

    /// The bookkeeping sidecar: what this checkpoint scored and when. Kept beside
    /// the weights rather than in the trainer so the *next* process knows what it
    /// has to beat — without it every run starts from an empty best and its first
    /// window overwrites a previous run's better model however much worse it is.
    pub fn meta(&self) -> PathBuf {
        // Not `member`: that one names weights, and this is not a safetensors file.
        self.dir.join(format!("{}.json", self.stem))
    }

    /// The recorded score of the weights on disk, if any. A missing sidecar is the
    /// ordinary first-run case; a damaged one is worth a word but must not stop a
    /// run, so it degrades to "nothing to beat".
    pub fn load_meta(&self) -> Option<BestMeta> {
        let p = self.meta();
        let text = std::fs::read_to_string(&p).ok()?;
        let meta = parse_meta(&text);
        if meta.is_none() {
            tracing::warn!("ignoring unreadable {}", p.display());
        }
        meta
    }

    /// Record what the saved weights scored. Callers write this only *after* the
    /// family lands, so a failed weight write can never advance the best that a
    /// later run inherits.
    pub fn save_meta(&self, meta: BestMeta) -> Result<()> {
        let p = self.meta();
        let BestMeta { mel, step } = meta;
        std::fs::write(&p, format!("{{\"mel\": {mel}, \"step\": {step}}}\n"))
            .with_context(|| format!("writing {}", p.display()))
    }

    /// Which generator to actually load when resuming from this family: the raw
    /// twin when it was written, else the deployable weights.
    pub fn resume_from(&self) -> PathBuf {
        let raw = self.raw();
        if raw.exists() { raw } else { self.generator() }
    }

    /// Write the whole family. `ema` is the deployable generator when EMA is on,
    /// in which case `live` is saved beside it as the raw twin; without EMA the
    /// live weights are themselves the deployable ones and no twin is written.
    ///
    /// All-or-nothing: the members are only useful together, so a partial write
    /// is an error rather than a checkpoint that resumes against a stale twin.
    ///
    /// Reports what landed and how long it took (a full family is hundreds of MB,
    /// so it's worth seeing in the log); callers report *why* they saved.
    pub fn save<B: Backend>(
        &self,
        ema: Option<&Synthesizer<B>>,
        live: &Synthesizer<B>,
        disc: &MultiPeriodDiscriminator<B>,
    ) -> Result<()> {
        let started = Instant::now();
        if !self.dir.as_os_str().is_empty() {
            std::fs::create_dir_all(&self.dir)
                .with_context(|| format!("creating {}", self.dir.display()))?;
        }
        let g = self.generator();
        ema.unwrap_or(live)
            .save_safetensors(&g)
            .map_err(|e| failed(&g, e))?;
        if ema.is_some() {
            let raw = self.raw();
            live.save_safetensors(&raw).map_err(|e| failed(&raw, e))?;
        }
        let d = self.disc();
        disc.save_safetensors(&d).map_err(|e| failed(&d, e))?;
        tracing::info!(
            "saved {} in {:.1}s",
            g.display(),
            started.elapsed().as_secs_f32()
        );
        Ok(())
    }

    fn member(&self, tag: &str) -> PathBuf {
        self.dir.join(format!("{}{tag}.{SUFFIX}", self.stem))
    }
}

/// Why a checkpoint was kept: the mel it scored and the step it came from (an
/// exit-window best and a mid-run one are otherwise indistinguishable in a log).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BestMeta {
    pub mel: f32,
    pub step: usize,
}

fn failed(path: &Path, e: Box<dyn Error>) -> anyhow::Error {
    anyhow!("saving {}: {e}", path.display())
}

/// Two numbers do not justify a serde dependency, and the only writer of this
/// file is [`Checkpoint::save_meta`]; anything else is treated as corrupt.
fn parse_meta(text: &str) -> Option<BestMeta> {
    let mel: f32 = number(text, "mel")?.parse().ok()?;
    let step: usize = number(text, "step")?.parse().ok()?;
    // A non-finite score would compare false against everything and freeze the best.
    mel.is_finite().then_some(BestMeta { mel, step })
}

/// The token following `"key":` in a flat one-object JSON.
fn number<'a>(text: &'a str, key: &str) -> Option<&'a str> {
    let at = text.find(&format!("\"{key}\""))?;
    let (_, rest) = text[at..].split_once(':')?;
    rest.split([',', '}']).next().map(str::trim)
}

#[cfg(test)]
mod tests {
    use super::{BestMeta, Checkpoint, parse_meta};
    use std::path::{Path, PathBuf};

    /// A scratch directory unique to this test binary *and* case.
    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("rvc-ckpt-{}-{name}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn every_spelling_names_the_same_family() {
        // The output name, the saved generator and its raw twin must all resolve
        // to one family — otherwise `--resume voice.raw.safetensors` would hunt
        // for a `voice.raw.disc.safetensors` that no run ever writes, and the
        // discriminator would silently start fresh.
        let want = Checkpoint::new(Path::new("out/voice"));
        assert_eq!(Checkpoint::new(Path::new("out/voice.safetensors")), want);
        assert_eq!(
            Checkpoint::new(Path::new("out/voice.raw.safetensors")),
            want
        );
        assert_eq!(want.generator(), PathBuf::from("out/voice.safetensors"));
        assert_eq!(want.disc(), PathBuf::from("out/voice.disc.safetensors"));
    }

    #[test]
    fn best_is_a_checkpoint_like_any_other() {
        let best = Checkpoint::new(Path::new("models/voice.safetensors")).best();
        assert_eq!(
            best.generator(),
            PathBuf::from("models/checkpoint/voice.best.safetensors")
        );
        assert_eq!(
            best.raw(),
            PathBuf::from("models/checkpoint/voice.best.raw.safetensors")
        );
        // Its own discriminator, not the final run's: `.best` must survive.
        assert_eq!(
            best.disc(),
            PathBuf::from("models/checkpoint/voice.best.disc.safetensors")
        );
        // Resuming from the best checkpoint reaches the same family.
        assert_eq!(Checkpoint::new(&best.raw()), best);

        // A bare output name still gets its checkpoint dir.
        let bare = Checkpoint::new(Path::new("voice.safetensors")).best();
        assert_eq!(
            bare.generator(),
            PathBuf::from("checkpoint/voice.best.safetensors")
        );
    }

    #[test]
    fn the_disc_sidecar_names_the_same_family() {
        // `--resume out/voice.disc.safetensors` must reach the run that wrote it,
        // not a `voice.disc.disc.safetensors` nobody ever writes.
        let want = Checkpoint::new(Path::new("out/voice"));
        assert_eq!(
            Checkpoint::new(Path::new("out/voice.disc.safetensors")),
            want
        );
        let best = want.best();
        assert_eq!(Checkpoint::new(&best.disc()), best);
    }

    #[test]
    fn resume_prefers_the_raw_twin_when_it_exists() {
        let dir = scratch("resume");
        let ck = Checkpoint::new(&dir.join("voice"));

        // Without EMA only the deployable weights exist, so they're what resumes.
        assert_eq!(ck.resume_from(), ck.generator());
        std::fs::write(ck.raw(), b"x").unwrap();
        assert_eq!(ck.resume_from(), ck.raw());

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn the_meta_sidecar_round_trips() {
        let best = Checkpoint::new(Path::new("models/voice.safetensors")).best();
        assert_eq!(
            best.meta(),
            PathBuf::from("models/checkpoint/voice.best.json")
        );

        let dir = scratch("meta");
        let ck = Checkpoint::new(&dir.join("voice")).best();
        std::fs::create_dir_all(dir.join("checkpoint")).unwrap();
        // Nothing recorded yet is the ordinary first-run case.
        assert_eq!(ck.load_meta(), None);

        let want = BestMeta {
            mel: 34.376,
            step: 1234,
        };
        ck.save_meta(want).unwrap();
        assert_eq!(ck.load_meta(), Some(want));

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_corrupt_sidecar_degrades_to_no_best() {
        // A truncated write or a hand-edit must not stop the run, and must not
        // leave a score that every later window compares false against.
        for bad in [
            "",
            "{",
            "{\"mel\": }",
            "{\"mel\": 1.0}",
            "{\"mel\": nan, \"step\": 0}",
        ] {
            assert_eq!(parse_meta(bad), None, "{bad:?} should not parse");
        }
        assert_eq!(
            parse_meta("{\"mel\": 12.5, \"step\": 7}\n"),
            Some(BestMeta { mel: 12.5, step: 7 })
        );

        let dir = scratch("corrupt");
        let ck = Checkpoint::new(&dir.join("voice"));
        std::fs::write(ck.meta(), b"not json").unwrap();
        assert_eq!(ck.load_meta(), None);

        std::fs::remove_dir_all(&dir).ok();
    }
}
