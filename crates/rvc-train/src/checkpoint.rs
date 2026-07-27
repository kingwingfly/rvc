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

use crate::trainer::IB;

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
    /// without its suffix (`models/voice`, `models/voice.safetensors`) and the
    /// raw twin (`models/voice.raw.safetensors`) — all name the same family.
    pub fn new(generator: &Path) -> Self {
        let dir = generator.parent().unwrap_or(Path::new("")).to_path_buf();
        let name = generator
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("voice");
        let stem = name.strip_suffix(&format!(".{SUFFIX}")).unwrap_or(name);
        Self {
            dir,
            stem: stem.strip_suffix(".raw").unwrap_or(stem).to_string(),
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
    pub fn save(
        &self,
        ema: Option<&Synthesizer<IB>>,
        live: &Synthesizer<IB>,
        disc: &MultiPeriodDiscriminator<IB>,
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

fn failed(path: &Path, e: Box<dyn Error>) -> anyhow::Error {
    anyhow!("saving {}: {e}", path.display())
}

#[cfg(test)]
mod tests {
    use super::Checkpoint;
    use std::path::{Path, PathBuf};

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
    fn resume_prefers_the_raw_twin_when_it_exists() {
        let dir = std::env::temp_dir().join(format!("rvc-ckpt-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let ck = Checkpoint::new(&dir.join("voice"));

        // Without EMA only the deployable weights exist, so they're what resumes.
        assert_eq!(ck.resume_from(), ck.generator());
        std::fs::write(ck.raw(), b"x").unwrap();
        assert_eq!(ck.resume_from(), ck.raw());

        std::fs::remove_dir_all(&dir).ok();
    }
}
