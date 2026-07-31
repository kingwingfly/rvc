//! Auto-download and cache the shared ONNX assets (ContentVec + RMVPE) from the
//! Hugging Face Hub.
//!
//! The trained generator (`voice.onnx`) is produced locally by the training
//! pipeline and is **not** fetched here. Repo IDs and filenames are overridable
//! because the exact best-maintained ONNX mirrors move over time; the defaults
//! below are a starting point, not a guarantee.

use std::path::{Path, PathBuf};

use hf_hub::HFClient;

/// Errors from model resolution/download.
#[derive(Debug, thiserror::Error)]
pub enum HubError {
    /// Error from the Hugging Face Hub client.
    #[error("hugging face hub error: {0}")]
    Hf(#[from] hf_hub::HFError),
    /// I/O error resolving the cache directory.
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
}

/// Convenience alias.
pub type Result<T> = std::result::Result<T, HubError>;

/// A single file within a Hub model repo.
#[derive(Debug, Clone)]
pub struct ModelRef {
    pub owner: String,
    pub name: String,
    pub file: String,
}

impl ModelRef {
    /// Construct from static parts.
    pub fn new(owner: impl Into<String>, name: impl Into<String>, file: impl Into<String>) -> Self {
        Self {
            owner: owner.into(),
            name: name.into(),
            file: file.into(),
        }
    }
}

/// Default ContentVec (768-dim, layer 12) ONNX encoder.
///
/// NOTE: verify/override for your environment — ONNX mirrors of the RVC content
/// encoder change; this is a widely used one.
pub fn default_contentvec() -> ModelRef {
    ModelRef::new(
        "NaruseMioShirakana",
        "MoeSS-SUBModel",
        "vec-768-layer-12.onnx",
    )
}

/// Default RMVPE F0 estimator ONNX.
///
/// NOTE: verify/override — see comment on [`default_contentvec`].
pub fn default_rmvpe() -> ModelRef {
    ModelRef::new("lj1995", "VoiceConversionWebUI", "rmvpe.onnx")
}

/// The toolkit-wide cache directory: `$VOICE_CACHE_DIR`, else
/// `$XDG_CACHE_HOME/voice`, else `~/.cache/voice`.
pub fn default_cache_dir() -> PathBuf {
    resolve_cache_dir(None, |k| std::env::var_os(k))
}

/// The same, consulting one engine's own variable first — `RVC_CACHE_DIR`,
/// `STT_CACHE_DIR`, `TTS_CACHE_DIR` — so a single engine's assets can be kept
/// somewhere else without moving everybody's. The name is passed in rather than
/// listed here: this crate knows about downloads, not about engines.
pub fn cache_dir_for(engine_var: &str) -> PathBuf {
    resolve_cache_dir(Some(engine_var), |k| std::env::var_os(k))
}

/// The resolution itself, over an environment reader so the precedence can be
/// tested without mutating the process's real environment.
fn resolve_cache_dir(
    engine_var: Option<&str>,
    env: impl Fn(&str) -> Option<std::ffi::OsString>,
) -> PathBuf {
    let read = |key: &str| env(key).filter(|v| !v.is_empty());
    for var in engine_var.into_iter().chain(Some("VOICE_CACHE_DIR")) {
        if let Some(dir) = read(var) {
            return PathBuf::from(dir);
        }
    }
    if let Some(xdg) = read("XDG_CACHE_HOME") {
        return PathBuf::from(xdg).join("voice");
    }
    if let Some(home) = read("HOME") {
        return PathBuf::from(home).join(".cache").join("voice");
    }
    // Never CWD-relative. A cache that moves with the shell's working directory
    // re-downloads gigabytes the first time the user runs from somewhere else,
    // and leaves a hidden folder wherever they happened to stand — including in
    // an output directory.
    std::env::temp_dir().join("voice-cache")
}

/// Tell the user once where the cache went, rather than silently re-downloading
/// everything the old `~/.cache/rvc` already holds. Nothing is copied or moved:
/// the old directory is the user's to keep or delete.
///
/// Only for the untouched default. Someone who named a directory — by flag or by
/// variable — chose it, and does not need to be told about a location they were
/// not using.
fn note_cache_move(cache: &Path) {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        let Some(home) = std::env::var_os("HOME") else {
            return;
        };
        let home = PathBuf::from(home).join(".cache");
        let (old, new) = (home.join("rvc"), home.join("voice"));
        if cache == new && old.exists() && !new.exists() {
            tracing::info!(
                "the asset cache is now {} (it used to be {}, which is left untouched)",
                new.display(),
                old.display()
            );
        }
    });
}

/// Download (or reuse the cached copy of) a single Hub file, returning its local
/// path. `cache_dir` is resolved by the caller — see [`cache_dir_for`].
pub async fn fetch(model: &ModelRef, cache_dir: &Path) -> Result<PathBuf> {
    note_cache_move(cache_dir);
    let client = HFClient::builder()
        .cache_dir(cache_dir.to_path_buf())
        .build()?;
    let repo = client.model(model.owner.clone(), model.name.clone());
    let path = repo
        .download_file()
        .filename(model.file.clone())
        .send()
        .await?;
    Ok(path)
}

/// The two shared assets the RVC inference pipeline needs.
#[derive(Debug, Clone)]
pub struct SharedAssets {
    pub contentvec: PathBuf,
    pub rmvpe: PathBuf,
}

/// Fetch both shared assets, using defaults unless overridden.
pub async fn fetch_shared(
    contentvec: Option<ModelRef>,
    rmvpe: Option<ModelRef>,
    cache_dir: &Path,
) -> Result<SharedAssets> {
    let contentvec = contentvec.unwrap_or_else(default_contentvec);
    let rmvpe = rmvpe.unwrap_or_else(default_rmvpe);
    Ok(SharedAssets {
        contentvec: fetch(&contentvec, cache_dir).await?,
        rmvpe: fetch(&rmvpe, cache_dir).await?,
    })
}

/// The four files a Hugging Face Whisper repo needs to be usable: the weights,
/// the dimensions, the control-token ids and the BPE vocabulary.
#[derive(Debug, Clone)]
pub struct WhisperAssets {
    /// The directory holding all four, which is what `stt-core` loads from.
    pub dir: PathBuf,
}

/// Default ASR model: `openai/whisper-large-v3-turbo` (809M, MIT).
///
/// Unlike the ContentVec and RMVPE entries above — community ONNX exports that
/// move over time — this is a first-party repo, which is the main reason the
/// toolkit ports models rather than consuming somebody's conversion of one.
pub const DEFAULT_WHISPER: (&str, &str) = ("openai", "whisper-large-v3-turbo");

/// Files that must be present for `stt-core` to load a checkpoint.
const WHISPER_FILES: [&str; 4] = [
    "model.safetensors",
    "config.json",
    "generation_config.json",
    "tokenizer.json",
];

/// Fetch a Whisper repo, returning the directory the files landed in.
///
/// `repo` overrides the default as `owner/name` — that is how a different size
/// (`openai/whisper-large-v3`) or a fine-tune is selected, since `stt-core`
/// reads every dimension from the repo's own `config.json`.
pub async fn fetch_whisper(repo: Option<&str>, cache_dir: &Path) -> Result<WhisperAssets> {
    let (owner, name) = match repo {
        Some(r) => r.split_once('/').unwrap_or((r, "")),
        None => DEFAULT_WHISPER,
    };

    let mut dir = None;
    for file in WHISPER_FILES {
        let path = fetch(&ModelRef::new(owner, name, file), cache_dir).await?;
        // Every file lands in the same snapshot directory; the weights are the
        // slow one, so report progress against that rather than the JSON.
        dir = path.parent().map(Path::to_path_buf);
    }
    Ok(WhisperAssets {
        dir: dir.unwrap_or_else(|| cache_dir.to_path_buf()),
    })
}

/// Default prosody encoder: an ONNX `chinese-roberta-wwm-ext-large` that emits
/// the third-from-last hidden layer.
///
/// That last detail is the whole requirement. A stock `optimum` export gives
/// `last_hidden_state`, which is a *different* representation — GPT-SoVITS (and
/// Style-Bert-VITS2, whose conversion this is) condition on layer −3, and
/// substituting the last one sounds wrong rather than failing.
pub const DEFAULT_PROSODY_BERT: (&str, &str) =
    ("tsukumijima", "chinese-roberta-wwm-ext-large-onnx");

/// Files the prosody encoder needs.
const PROSODY_FILES: [&str; 3] = ["model.onnx", "tokenizer.json", "config.json"];

/// Fetch the prosody encoder, returning the directory the files landed in.
pub async fn fetch_prosody_bert(repo: Option<&str>, cache_dir: &Path) -> Result<PathBuf> {
    let (owner, name) = match repo {
        Some(r) => r.split_once('/').unwrap_or((r, "")),
        None => DEFAULT_PROSODY_BERT,
    };
    let mut dir = None;
    for file in PROSODY_FILES {
        let path = fetch(&ModelRef::new(owner, name, file), cache_dir).await?;
        dir = path.parent().map(Path::to_path_buf);
    }
    Ok(dir.unwrap_or_else(|| cache_dir.to_path_buf()))
}

/// The official GPT-SoVITS v2 bundle.
pub const DEFAULT_GPTSOVITS: (&str, &str) = ("lj1995", "GPT-SoVITS");

/// Files the synthesis path needs from it, plus the discriminator `s2`
/// fine-tuning warm-starts from. The last is dead weight for synthesis — ~90 MB
/// that inference never opens — but fetching the bundle twice for want of one
/// file is the worse trade, and a corpus big enough to fine-tune on dwarfs it.
const GPTSOVITS_FILES: [&str; 5] = [
    "chinese-hubert-base/config.json",
    "chinese-hubert-base/pytorch_model.bin",
    "gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt",
    "gsv-v2final-pretrained/s2G2333k.pth",
    "gsv-v2final-pretrained/s2D2333k.pth",
];

/// Where each model landed inside a fetched (or hand-assembled) bundle.
#[derive(Debug, Clone)]
pub struct GptSovitsPaths {
    pub hubert: PathBuf,
    pub s1: PathBuf,
    pub s2: PathBuf,
    /// The `s2` discriminator, when the bundle carries one.
    ///
    /// Optional where the other three are not, because only fine-tuning wants
    /// it: synthesis never opens a discriminator, and a hand-assembled model
    /// directory that predates `s2` training must keep working for `tts`.
    pub s2d: Option<PathBuf>,
}

/// Fetch the v2 bundle, returning the directory it landed in.
pub async fn fetch_gptsovits(cache_dir: &Path) -> Result<PathBuf> {
    let (owner, name) = DEFAULT_GPTSOVITS;
    let mut root = None;
    for file in GPTSOVITS_FILES {
        let path = fetch(&ModelRef::new(owner, name, file), cache_dir).await?;
        // Files sit at varying depths inside the snapshot; the root is what the
        // shallowest one's parent gives.
        let depth = file.matches('/').count();
        let mut dir = path.clone();
        for _ in 0..=depth {
            dir = dir.parent().map(Path::to_path_buf).unwrap_or(dir);
        }
        root = Some(dir);
    }
    Ok(root.unwrap_or_else(|| cache_dir.to_path_buf()))
}

/// Locate the three checkpoints inside `dir`.
///
/// Tolerant about layout because the same directory can come from the Hub's
/// snapshot or from a hand-assembled folder: `s1`/`s2` are found by extension
/// and prefix rather than by an exact filename, since those carry epoch and step
/// numbers that change with every release.
pub fn gptsovits_paths(dir: &Path) -> Result<GptSovitsPaths> {
    let hubert = ["chinese-hubert-base/pytorch_model.bin", "pytorch_model.bin"]
        .iter()
        .map(|p| dir.join(p))
        .find(|p| p.exists())
        .ok_or_else(|| missing("chinese-hubert-base/pytorch_model.bin", dir))?;

    let find = |prefix: &str, ext: &str| -> Option<PathBuf> {
        for candidate in [dir.join("gsv-v2final-pretrained"), dir.to_path_buf()] {
            let Ok(entries) = std::fs::read_dir(&candidate) else {
                continue;
            };
            for entry in entries.flatten() {
                let path = entry.path();
                let name = entry.file_name();
                let name = name.to_string_lossy();
                if name.starts_with(prefix) && name.ends_with(ext) {
                    return Some(path);
                }
            }
        }
        None
    };

    Ok(GptSovitsPaths {
        hubert,
        s1: find("s1", ".ckpt").ok_or_else(|| missing("an s1*.ckpt", dir))?,
        s2: find("s2G", ".pth").ok_or_else(|| missing("an s2G*.pth", dir))?,
        s2d: find("s2D", ".pth"),
    })
}

fn missing(what: &str, dir: &Path) -> HubError {
    HubError::Io(std::io::Error::new(
        std::io::ErrorKind::NotFound,
        format!("{what} not found under {}", dir.display()),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A fake environment, so precedence is tested without `set_var` racing the
    /// other tests in this binary.
    fn env(pairs: &[(&str, &str)]) -> impl Fn(&str) -> Option<std::ffi::OsString> + use<> {
        let pairs: Vec<(String, String)> = pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        move |key| pairs.iter().find(|(k, _)| k == key).map(|(_, v)| v.into())
    }

    #[test]
    fn engine_variable_beats_the_toolkit_one() {
        let e = env(&[
            ("STT_CACHE_DIR", "/b"),
            ("VOICE_CACHE_DIR", "/a"),
            ("XDG_CACHE_HOME", "/xdg"),
            ("HOME", "/home/u"),
        ]);
        assert_eq!(
            resolve_cache_dir(Some("STT_CACHE_DIR"), &e),
            Path::new("/b")
        );
        // An engine that names no variable of its own, and one whose variable is
        // unset, both fall through to the shared answer.
        assert_eq!(resolve_cache_dir(None, &e), Path::new("/a"));
        assert_eq!(
            resolve_cache_dir(Some("TTS_CACHE_DIR"), &e),
            Path::new("/a")
        );
    }

    #[test]
    fn falls_through_voice_then_xdg_then_home() {
        let full = env(&[("XDG_CACHE_HOME", "/xdg"), ("HOME", "/home/u")]);
        assert_eq!(resolve_cache_dir(None, &full), Path::new("/xdg/voice"));

        let home_only = env(&[("HOME", "/home/u")]);
        assert_eq!(
            resolve_cache_dir(None, &home_only),
            Path::new("/home/u/.cache/voice")
        );
    }

    /// The point of the whole rewrite: whatever the environment says — including
    /// saying nothing, or setting the variables to empty — the cache never lands
    /// below the working directory, so a download can never write into an output
    /// folder the user happened to `cd` into.
    #[test]
    fn never_relative_to_the_working_directory() {
        for e in [
            env(&[]),
            env(&[
                ("HOME", ""),
                ("XDG_CACHE_HOME", ""),
                ("VOICE_CACHE_DIR", ""),
            ]),
        ] {
            for engine in [None, Some("RVC_CACHE_DIR")] {
                let dir = resolve_cache_dir(engine, &e);
                assert!(dir.is_absolute(), "{} is not absolute", dir.display());
                assert!(!dir.starts_with(std::env::current_dir().unwrap()));
            }
        }
    }

    /// `RVC_CACHE_DIR` predates the toolkit-wide variable, so scripts that set it
    /// must keep working.
    #[test]
    fn rvc_cache_dir_still_works() {
        let e = env(&[("RVC_CACHE_DIR", "/legacy"), ("HOME", "/home/u")]);
        assert_eq!(
            resolve_cache_dir(Some("RVC_CACHE_DIR"), &e),
            Path::new("/legacy")
        );
    }
}
