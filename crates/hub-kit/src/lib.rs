//! Auto-download and cache the shared ContentVec + RMVPE assets from the
//! Hugging Face Hub.
//!
//! The trained generator (`voice.onnx`/`voice.safetensors`) is produced
//! locally by the training pipeline and is **not** fetched here. Repo IDs and
//! filenames are overridable because the exact best-maintained mirrors move
//! over time; the defaults below are a starting point, not a guarantee.
//!
//! Both assets exist in **two weight formats**, ONNX and PyTorch, because the
//! backend a caller picks decides which one it can load — ONNX Runtime reads
//! only the former, the Burn/LibTorch and Burn/CubeCL generators only the
//! latter. [`WeightFormat`] is how a caller says which; it is decided
//! entirely by the caller, since this crate must not depend on `cli-kit` and
//! so knows nothing about `--backend` or its aliases. [`fetch_contentvec`]
//! and [`fetch_rmvpe`] fetch the file (or, for a PyTorch ContentVec, the
//! directory) the chosen format needs.

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

/// Which weight format a caller wants ContentVec/RMVPE fetched in.
///
/// This crate has no notion of "backend" — that enum lives in `cli-kit`, the
/// one leaf every binary shares, and `cli_kit::backend` is deliberate that
/// **cli-kit knows nothing about weight formats**: only the engine loading a
/// file knows whether an artefact for a given backend is a single file or a
/// directory full of them. So the mapping runs the other way: the caller (an
/// engine crate, which does know both `Backend` and how it loads a model)
/// turns its `Backend` into a `WeightFormat` and passes that in here, rather
/// than hub-kit depending on cli-kit to accept a `Backend` directly.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightFormat {
    /// ONNX Runtime.
    Onnx,
    /// Burn on LibTorch or CubeCL/CUDA — PyTorch's own checkpoint format.
    Torch,
}

/// Default ContentVec (768-dim, layer 12) ONNX encoder.
///
/// A **community** mirror (`NaruseMioShirakana/MoeSS-SUBModel`) rather than a
/// first-party one — contrast [`default_rmvpe`] and [`fetch_contentvec`]'s
/// `Torch` arm, both first-party, and see [`DEFAULT_WHISPER`]'s note on why
/// that asymmetry is the main reason this toolkit ports models rather than
/// consuming somebody's conversion of one.
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

/// Default RMVPE F0 estimator, in the given weight format.
///
/// Both formats live in the same first-party repo — `rmvpe.onnx` beside
/// `rmvpe.pt` in `lj1995/VoiceConversionWebUI` — so selecting one is a
/// one-word change to the filename, unlike ContentVec below.
///
/// NOTE: verify/override — see comment on [`default_contentvec`].
pub fn default_rmvpe(format: WeightFormat) -> ModelRef {
    let file = match format {
        WeightFormat::Onnx => "rmvpe.onnx",
        WeightFormat::Torch => "rmvpe.pt",
    };
    ModelRef::new("lj1995", "VoiceConversionWebUI", file)
}

/// The repo and files a PyTorch ContentVec (a HuBERT checkpoint) is published
/// as: `lj1995/VoiceConversionWebUI`'s `hubert_base/` directory. First-party,
/// unlike the ONNX mirror [`default_contentvec`] points at — see the note
/// there.
const CONTENTVEC_TORCH_REPO: (&str, &str) = ("lj1995", "VoiceConversionWebUI");
const CONTENTVEC_TORCH_FILES: [&str; 3] = [
    "hubert_base/config.json",
    "hubert_base/pytorch_model.bin",
    "hubert_base/preprocessor_config.json",
];

/// Fetch ContentVec in the given format.
///
/// `Onnx` is [`fetch`]'s ordinary single-file case. `Torch` is not: PyTorch
/// ContentVec is a HuBERT checkpoint, which needs its config and preprocessor
/// alongside the weights, so there is no single [`ModelRef`] to hand back —
/// this follows [`fetch_whisper`]/[`fetch_gptsovits`]'s multi-file shape
/// instead and returns the directory the triple landed in.
pub async fn fetch_contentvec(format: WeightFormat, cache_dir: &Path) -> Result<PathBuf> {
    match format {
        WeightFormat::Onnx => fetch(&default_contentvec(), cache_dir).await,
        WeightFormat::Torch => {
            let (owner, name) = CONTENTVEC_TORCH_REPO;
            let mut dir = None;
            for file in CONTENTVEC_TORCH_FILES {
                let path = fetch(&ModelRef::new(owner, name, file), cache_dir).await?;
                dir = path.parent().map(Path::to_path_buf);
            }
            Ok(dir.unwrap_or_else(|| cache_dir.to_path_buf()))
        }
    }
}

/// Fetch RMVPE in the given format. Unlike [`fetch_contentvec`], both formats
/// are a single file, so this is [`fetch`] over [`default_rmvpe`].
pub async fn fetch_rmvpe(format: WeightFormat, cache_dir: &Path) -> Result<PathBuf> {
    fetch(&default_rmvpe(format), cache_dir).await
}

/// The toolkit-wide cache directory: `$VOICE_CACHE_DIR`, else `voice` under the
/// XDG cache root — `$XDG_CACHE_HOME` when it names an absolute path, otherwise
/// `~/.cache`, so the answer is `~/.cache/voice` on a machine that sets neither.
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
    // The XDG cache *root*, which is then always joined with `voice`. Written
    // this way round rather than as two independent branches because the shape
    // is what stops `~/voice`: a home directory is never itself the root, only
    // `~/.cache` is, so the toolkit's directory cannot surface beside the user's
    // own folders. `XDG_CACHE_HOME` is honoured only when absolute — the spec
    // says a relative value must be ignored, and here that is load-bearing
    // rather than pedantic, since a relative root is a cache that moves with the
    // shell's working directory: it re-downloads gigabytes the first time the
    // user runs from somewhere else and leaves a folder wherever they happened
    // to stand, including inside an output directory.
    let root = read("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .filter(|p| p.is_absolute())
        .or_else(|| read("HOME").map(|h| PathBuf::from(h).join(".cache")))
        // No `HOME` at all is a broken environment rather than a configuration,
        // and there is no `~` left to expand. A temporary directory keeps `-h`
        // and every download working; it does not survive a reboot, which is the
        // best that can be promised without somewhere to put a home cache.
        .unwrap_or_else(|| std::env::temp_dir().join("voice-cache"));
    root.join("voice")
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

/// Where the warm-start bases a training run starts from live: `pretrained/`
/// inside the asset cache.
///
/// In the cache and not beside the run, because a base is **read-only input
/// shared by every run on the machine**, exactly like an inference asset. It is
/// the published upstream file, byte for byte, whatever it is warm-starting;
/// putting a copy in each output directory would fetch the same 200 MB again
/// for every voice trained. A subdirectory rather than the cache root only
/// because these are stored flat under their upstream names — see
/// [`fetch_pretrained`].
pub fn pretrained_dir(cache: &Path) -> PathBuf {
    cache.join("pretrained")
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
/// Flat, unlike [`fetch`]'s cache tree: `pretrained/` is meant to be read by eye
/// and hand-populated by anyone who already has the weights. The upstream names
/// are distinct across engines, so one directory serves them all.
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

/// Fetch both shared assets, using each format's default unless overridden.
///
/// An explicit override is always a single file — that is what
/// `owner/name:file` (`parse_model_ref` in `rvc-cli`) can spell — so it is
/// fetched with plain [`fetch`] regardless of `format`; only the *default*
/// ContentVec path can be a directory, via [`fetch_contentvec`]'s `Torch` arm.
pub async fn fetch_shared(
    contentvec: Option<ModelRef>,
    contentvec_format: WeightFormat,
    rmvpe: Option<ModelRef>,
    rmvpe_format: WeightFormat,
    cache_dir: &Path,
) -> Result<SharedAssets> {
    let contentvec = match contentvec {
        Some(r) => fetch(&r, cache_dir).await?,
        None => fetch_contentvec(contentvec_format, cache_dir).await?,
    };
    let rmvpe = match rmvpe {
        Some(r) => fetch(&r, cache_dir).await?,
        None => fetch_rmvpe(rmvpe_format, cache_dir).await?,
    };
    Ok(SharedAssets { contentvec, rmvpe })
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
/// Unlike ContentVec's ONNX default above — a community export that moves
/// over time — this is a first-party repo, which is the main reason the
/// toolkit ports models rather than consuming somebody's conversion of one.
/// RMVPE and ContentVec's *Torch* format share that property with this repo:
/// both come from `lj1995/VoiceConversionWebUI` directly.
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

/// Files the synthesis path needs from it. The `s2` discriminator that
/// fine-tuning warm-starts from is deliberately *not* here: inference never
/// opens one, so it is fetched separately by [`fetch_pretrained`], only when a
/// fine-tune asks for it. Both end up in the same cache; what this split buys is
/// that a synthesis-only user never downloads 94 MB of adversary at all.
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

/// Seed-VC's own release, holding the checkpoint and nothing else this needs.
pub const DEFAULT_SEEDVC: (&str, &str) = ("Plachta", "Seed-VC");

/// The `seed-uvit-whisper-small-wavenet` preset, 110M parameters. The repo ships
/// several presets side by side and the name encodes which one this is — content
/// encoder, backbone, vocoder — so it is spelled in full rather than abbreviated.
const SEEDVC_FILE: &str = "DiT_seed_v2_uvit_whisper_small_wavenet_bigvgan_pruned.pth";

/// The timbre encoder's release (28 MB).
pub const DEFAULT_CAMPPLUS: (&str, &str) = ("funasr", "campplus");

/// Its single file, spelled as upstream spells it in every entry point.
const CAMPPLUS_FILE: &str = "campplus_cn_common.bin";

/// The vocoder's release, used unmodified.
pub const DEFAULT_BIGVGAN: (&str, &str) = ("nvidia", "bigvgan_v2_22khz_80band_256x");

/// Its generator, which is the only half a conversion opens. The repo also ships
/// the discriminator and its optimiser state — half a gigabyte of adversary —
/// and this vocoder is used frozen, so neither is ever fetched.
const BIGVGAN_FILE: &str = "bigvgan_generator.pt";

/// The content encoder, and **not** a choice.
///
/// The checkpoint above was conditioned on this exact encoder's 50 Hz features,
/// so a different size is not a quality trade but a different representation:
/// the transformer would be fed embeddings it has never seen. It is named here
/// rather than made overridable for that reason, unlike [`fetch_whisper`]'s own
/// default, where swapping the model is the point.
pub const SEEDVC_WHISPER: &str = "openai/whisper-small";

/// Where each of Seed-VC's four networks landed.
#[derive(Debug, Clone)]
pub struct SeedVcPaths {
    /// The DiT checkpoint: the transformer, the length regulator and the
    /// training-time style encoder that inference does not use.
    pub checkpoint: PathBuf,
    /// CAMPPlus, the timbre encoder the transformer is actually conditioned on.
    pub campplus: PathBuf,
    pub bigvgan: PathBuf,
    pub bigvgan_config: PathBuf,
    /// The Whisper repo's directory, in the shape [`WhisperAssets`] gives.
    pub whisper: PathBuf,
}

/// Fetch everything a Seed-VC conversion opens, from the **four separate repos**
/// they are published in.
///
/// That is what makes this unlike [`fetch_gptsovits`], which returns one
/// snapshot directory: no directory holds all of these, and manufacturing one
/// would mean copying most of a gigabyte back out of the cache. Only the
/// checkpoint is Seed-VC's own — the timbre encoder, the vocoder and the content
/// encoder are three other projects' releases, used unmodified — which is worth
/// knowing before hunting for a missing network's tensors in the wrong file.
///
/// Everything here is an inference asset and lands in the cache proper. There is
/// deliberately no [`fetch_pretrained`] counterpart: Seed-VC is zero-shot, a
/// reference clip is the entire speaker specification, so nothing here is ever a
/// warm-start base for a fine-tune.
/// The four are also fetchable one at a time — [`fetch_seedvc_checkpoint`],
/// [`fetch_campplus`], [`fetch_bigvgan`] and [`fetch_whisper`] — which is what a
/// caller holding some of the weights already should use. This composes them.
pub async fn fetch_seedvc(cache_dir: &Path) -> Result<SeedVcPaths> {
    let (bigvgan, bigvgan_config) = fetch_bigvgan(cache_dir).await?;
    Ok(SeedVcPaths {
        checkpoint: fetch_seedvc_checkpoint(cache_dir).await?,
        campplus: fetch_campplus(cache_dir).await?,
        bigvgan,
        bigvgan_config,
        whisper: fetch_whisper(Some(SEEDVC_WHISPER), cache_dir).await?.dir,
    })
}

/// Seed-VC's own checkpoint: the transformer and the length regulator.
///
/// Separately fetchable for the reason all four are: someone who was handed one
/// of these files should not be made to download the other three to use it, and
/// the upstream filename lives here rather than in every caller.
pub async fn fetch_seedvc_checkpoint(cache_dir: &Path) -> Result<PathBuf> {
    let (owner, name) = DEFAULT_SEEDVC;
    fetch(&ModelRef::new(owner, name, SEEDVC_FILE), cache_dir).await
}

/// The timbre encoder, from a different project's release (28 MB).
pub async fn fetch_campplus(cache_dir: &Path) -> Result<PathBuf> {
    let (owner, name) = DEFAULT_CAMPPLUS;
    fetch(&ModelRef::new(owner, name, CAMPPLUS_FILE), cache_dir).await
}

/// The vocoder's generator and its `config.json`, in that order.
///
/// Both, because the config is what identifies the vocoder in a hand-assembled
/// directory. **Nothing parses it today** — `seedvc-core` builds the vocoder from
/// `BigVganConfig::v2_22khz_80band_256x()`, a preset named after this very repo,
/// so the two cannot disagree while the repo is pinned. A second preset would
/// make reading it the honest answer.
pub async fn fetch_bigvgan(cache_dir: &Path) -> Result<(PathBuf, PathBuf)> {
    let (owner, name) = DEFAULT_BIGVGAN;
    let weights = fetch(&ModelRef::new(owner, name, BIGVGAN_FILE), cache_dir).await?;
    let config = fetch(&ModelRef::new(owner, name, "config.json"), cache_dir).await?;
    Ok((weights, config))
}

/// Locate the same four inside one hand-assembled directory.
///
/// Tolerant about naming for the reason [`gptsovits_paths`] is — someone who
/// already holds these weights should not download them again — so each file is
/// matched by what identifies it rather than by its full upstream name: the
/// checkpoint's encodes a preset that changes between releases, and the vocoder
/// ships under two names in its own repo.
///
/// Whisper is looked for **as a subdirectory**, never as loose files, because
/// its `config.json` and BigVGAN's share a name and a flat layout would hand
/// each loader the other's. Both parse, so the failure would be silent.
pub fn seedvc_paths(dir: &Path) -> Result<SeedVcPaths> {
    let entries: Vec<(String, PathBuf)> = std::fs::read_dir(dir)
        .map_err(|e| {
            HubError::Io(std::io::Error::new(
                e.kind(),
                format!("{}: {e}", dir.display()),
            ))
        })?
        .flatten()
        .map(|e| (e.file_name().to_string_lossy().to_lowercase(), e.path()))
        .collect();
    let find = |what: &str, matches: &dyn Fn(&str) -> bool| {
        entries
            .iter()
            .find(|(name, _)| matches(name))
            .map(|(_, path)| path.clone())
            .ok_or_else(|| missing(what, dir))
    };

    // The discriminator is excluded by name rather than trusted to fail on load:
    // a clone of the vocoder's repo carries `bigvgan_discriminator_optimizer.pt`
    // beside the generator, and `read_dir` order decides which a bare `.pt`
    // match would find.
    let bigvgan = find("bigvgan_generator.pt", &|n| n == BIGVGAN_FILE).or_else(|_| {
        find("a bigvgan*.pt generator", &|n| {
            n.contains("bigvgan") && n.ends_with(".pt") && !n.contains("discriminator")
        })
    })?;
    // Found by what it contains rather than by what it is called: the Hub
    // snapshot, a clone and a hand-made copy name this directory three different
    // things, and the weights file is the one name Whisper itself fixes.
    let whisper = entries
        .iter()
        .map(|(_, path)| path)
        .find(|path| path.join("model.safetensors").exists())
        .cloned()
        .ok_or_else(|| missing("a directory holding whisper-small", dir))?;
    Ok(SeedVcPaths {
        checkpoint: find("a DiT*.pth checkpoint", &|n| {
            n.starts_with("dit") && n.ends_with(".pth")
        })?,
        campplus: find(CAMPPLUS_FILE, &|n| {
            n.starts_with("campplus") && n.ends_with(".bin")
        })?,
        bigvgan,
        // Preferred over a bare `config.json` so that a directory carrying both
        // this and a stray one still pairs the vocoder with its own.
        bigvgan_config: find("the vocoder's config.json", &|n| {
            n.contains("bigvgan") && n.ends_with(".json")
        })
        .or_else(|_| find("the vocoder's config.json", &|n| n == "config.json"))?,
        whisper,
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

    /// The toolkit's directory hangs off a *cache* root, never off the home
    /// directory itself: `~/.cache/voice` is the fallback, `~/voice` is not a
    /// path this can produce. A relative `XDG_CACHE_HOME` is ignored rather than
    /// resolved against the working directory, which the XDG spec requires and
    /// which is the same rule as the test below.
    #[test]
    fn never_directly_under_home() {
        for e in [
            env(&[("HOME", "/home/u")]),
            env(&[("XDG_CACHE_HOME", "cache"), ("HOME", "/home/u")]),
            env(&[("XDG_CACHE_HOME", "../cache"), ("HOME", "/home/u")]),
        ] {
            assert_eq!(
                resolve_cache_dir(None, &e),
                Path::new("/home/u/.cache/voice")
            );
        }
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

    /// A hand-assembled Seed-VC directory resolves, under names that are not the
    /// upstream ones — which is the whole point of the resolver being tolerant.
    /// The two `config.json`s are the trap: the vocoder's must not be satisfied
    /// by Whisper's, since both parse and the mistake would be silent.
    #[test]
    fn seedvc_paths_tolerates_a_hand_assembled_directory() {
        let dir = std::env::temp_dir().join(format!("hub-kit-seedvc-{}", std::process::id()));
        let whisper = dir.join("whisper");
        std::fs::create_dir_all(&whisper).unwrap();
        for name in ["dit.pth", "campplus_cn_common.bin", "bigvgan.pt"] {
            std::fs::write(dir.join(name), []).unwrap();
        }
        for name in ["model.safetensors", "config.json"] {
            std::fs::write(whisper.join(name), []).unwrap();
        }

        // No vocoder config yet, so the whole thing must fail rather than reach
        // into the Whisper directory for one.
        assert!(seedvc_paths(&dir).is_err());

        std::fs::write(dir.join("bigvgan_config.json"), []).unwrap();
        let paths = seedvc_paths(&dir).unwrap();
        assert_eq!(paths.checkpoint, dir.join("dit.pth"));
        assert_eq!(paths.campplus, dir.join(CAMPPLUS_FILE));
        assert_eq!(paths.bigvgan, dir.join("bigvgan.pt"));
        assert_eq!(paths.bigvgan_config, dir.join("bigvgan_config.json"));
        assert_eq!(paths.whisper, whisper);

        // The upstream spelling is a bare `config.json`, and it still resolves.
        std::fs::rename(dir.join("bigvgan_config.json"), dir.join("config.json")).unwrap();
        assert_eq!(
            seedvc_paths(&dir).unwrap().bigvgan_config,
            dir.join("config.json")
        );

        std::fs::remove_dir_all(&dir).unwrap();
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
