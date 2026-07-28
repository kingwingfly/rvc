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
