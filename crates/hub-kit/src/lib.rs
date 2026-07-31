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

/// Resolve the cache directory: `$RVC_CACHE_DIR`, else `$XDG_CACHE_HOME/rvc`,
/// else `~/.cache/rvc`.
pub fn default_cache_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("RVC_CACHE_DIR") {
        return PathBuf::from(dir);
    }
    if let Ok(xdg) = std::env::var("XDG_CACHE_HOME") {
        return PathBuf::from(xdg).join("rvc");
    }
    if let Ok(home) = std::env::var("HOME") {
        return PathBuf::from(home).join(".cache").join("rvc");
    }
    PathBuf::from(".rvc-cache")
}

/// Download (or reuse the cached copy of) a single Hub file, returning its local
/// path. Uses `cache_dir` or [`default_cache_dir`] when `None`.
pub async fn fetch(model: &ModelRef, cache_dir: Option<&Path>) -> Result<PathBuf> {
    let cache = cache_dir
        .map(Path::to_path_buf)
        .unwrap_or_else(default_cache_dir);
    let client = HFClient::builder().cache_dir(cache).build()?;
    let repo = client.model(model.owner.clone(), model.name.clone());
    let path = repo
        .download_file()
        .filename(model.file.clone())
        .send()
        .await?;
    Ok(path)
}

/// Where a training run keeps the warm-start bases it starts from: `pretrained/`
/// beside the run's output, derived from the `-o` stem's directory.
///
/// Not the cache, because a base is not a frozen asset every command reads — it
/// is training input, wanted only by whoever is training, and it belongs with
/// the run's other files rather than in a directory the user never looks at.
pub fn pretrained_dir(out: &Path) -> PathBuf {
    out.parent()
        .unwrap_or_else(|| Path::new(""))
        .join("pretrained")
}

/// RVC's pretrained generator base (`f0G48k.pth`, 76 MB).
pub fn default_pretrained_g() -> ModelRef {
    ModelRef::new("lj1995", "VoiceConversionWebUI", "pretrained_v2/f0G48k.pth")
}

/// RVC's pretrained discriminator base (`f0D48k.pth`, 143 MB).
pub fn default_pretrained_d() -> ModelRef {
    ModelRef::new("lj1995", "VoiceConversionWebUI", "pretrained_v2/f0D48k.pth")
}

/// GPT-SoVITS's `s2` discriminator base (`s2D2333k.pth`, 94 MB).
///
/// Its `s1`/`s2G` siblings are inference weights and stay in the cache
/// ([`fetch_gptsovits`]); this one is opened by nothing but a fine-tune.
pub fn default_gptsovits_s2d() -> ModelRef {
    ModelRef::new(
        "lj1995",
        "GPT-SoVITS",
        "gsv-v2final-pretrained/s2D2333k.pth",
    )
}

/// Download a warm-start base into `dir` under its upstream file name, reusing
/// the copy already there.
///
/// Flat, unlike [`fetch`]'s cache tree: a run's `pretrained/` is meant to be
/// read by eye and hand-populated by anyone who already has the weights. The
/// upstream names are distinct across engines, so one directory serves them all.
pub async fn fetch_pretrained(model: &ModelRef, dir: &Path) -> Result<PathBuf> {
    let name = model.file.rsplit('/').next().unwrap_or(&model.file);
    let dest = dir.join(name);
    if dest.exists() {
        return Ok(dest);
    }

    tracing::info!(
        "downloading {}/{}/{} to {}",
        model.owner,
        model.name,
        model.file,
        dir.display()
    );
    let client = HFClient::new()?;
    let repo = client.model(model.owner.clone(), model.name.clone());
    // `local_dir` reproduces the repo's own directory structure, so the file
    // lands a level down; move it up and drop the wrapper if it is now empty.
    // Only the completed download is ever named `dest`, which is what makes the
    // reuse check above safe after an interrupted one.
    let path = repo
        .download_file()
        .filename(model.file.clone())
        .local_dir(dir.to_path_buf())
        .send()
        .await?;
    if path != dest {
        std::fs::rename(&path, &dest)?;
        if let Some(parent) = path.parent() {
            let _ = std::fs::remove_dir(parent);
        }
    }
    Ok(dest)
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
    cache_dir: Option<&Path>,
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
pub async fn fetch_whisper(repo: Option<&str>, cache_dir: Option<&Path>) -> Result<WhisperAssets> {
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
        dir: dir.unwrap_or_else(default_cache_dir),
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
pub async fn fetch_prosody_bert(repo: Option<&str>, cache_dir: Option<&Path>) -> Result<PathBuf> {
    let (owner, name) = match repo {
        Some(r) => r.split_once('/').unwrap_or((r, "")),
        None => DEFAULT_PROSODY_BERT,
    };
    let mut dir = None;
    for file in PROSODY_FILES {
        let path = fetch(&ModelRef::new(owner, name, file), cache_dir).await?;
        dir = path.parent().map(Path::to_path_buf);
    }
    Ok(dir.unwrap_or_else(default_cache_dir))
}

/// The official GPT-SoVITS v2 bundle.
pub const DEFAULT_GPTSOVITS: (&str, &str) = ("lj1995", "GPT-SoVITS");

/// Files the synthesis path needs from it. The `s2` discriminator that
/// fine-tuning warm-starts from is deliberately *not* here: inference never
/// opens one, so it is a warm-start base like RVC's and goes to the run's own
/// `pretrained/` via [`fetch_pretrained`] — a synthesis-only user should not
/// carry 94 MB of adversary in their cache.
const GPTSOVITS_FILES: [&str; 4] = [
    "chinese-hubert-base/config.json",
    "chinese-hubert-base/pytorch_model.bin",
    "gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt",
    "gsv-v2final-pretrained/s2G2333k.pth",
];

/// Where each model landed inside a fetched (or hand-assembled) bundle.
#[derive(Debug, Clone)]
pub struct GptSovitsPaths {
    pub hubert: PathBuf,
    pub s1: PathBuf,
    pub s2: PathBuf,
    /// The `s2` discriminator, when the directory happens to carry one.
    ///
    /// Optional where the other three are not, because only fine-tuning wants
    /// it and the fetched bundle no longer includes it. A hand-assembled model
    /// directory that has one is still honoured, so nothing already on disk is
    /// downloaded a second time.
    pub s2d: Option<PathBuf>,
}

/// Fetch the v2 bundle, returning the directory it landed in.
pub async fn fetch_gptsovits(cache_dir: Option<&Path>) -> Result<PathBuf> {
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
    Ok(root.unwrap_or_else(default_cache_dir))
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
