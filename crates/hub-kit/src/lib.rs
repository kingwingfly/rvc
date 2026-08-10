//! Auto-download and cache every engine's model assets, almost all of them from
//! the Hugging Face Hub.
//!
//! The exception is [`fetch_naist_jdic`], which reads a GitHub release asset —
//! see its own doc comment for why that cannot be a [`ModelRef`].
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
use std::sync::Arc;
use std::time::Duration;

use hf_hub::HFClient;
use hf_hub::progress::{Progress, ProgressEvent, ProgressHandler};

/// Errors from model resolution/download.
#[derive(Debug, thiserror::Error)]
pub enum HubError {
    /// Error from the Hugging Face Hub client.
    #[error("hugging face hub error: {0}")]
    Hf(#[from] hf_hub::HFError),
    /// I/O error resolving the cache directory.
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    /// Error from the plain HTTP client, which only [`fetch_naist_jdic`] uses.
    #[error("http error: {0}")]
    Http(#[from] reqwest::Error),
    /// A download that arrived, but not intact.
    #[error("{0}")]
    Download(String),
    /// A transfer that stopped moving and was abandoned — see [`Retry::stall`].
    ///
    /// Its own variant rather than an [`HubError::Io`] timeout because it is the
    /// one failure here that the server never reported: the bytes simply stopped
    /// arriving. That is the case this crate used to sit in forever with no
    /// diagnostic at all, so it says what expired and what to turn.
    // Naming *where* the two flags live, not just what they are called: a
    // stall during a conversion or a training run reaches this message too, and
    // those subcommands do not take them. Sending that user to `download`,
    // which does, is both true everywhere and the thing that actually helps —
    // prefetching with a longer window is how an unreliable link is worked
    // around before the run that matters.
    #[error(
        "{what}: no data for {}s, abandoned after {tries} attempt(s) — the `download` \
         subcommand takes --download-timeout for a link that is slow to get going, \
         and --retries for one that is flaky",
        stall.as_secs()
    )]
    Stalled {
        /// What was being fetched, spelled the way the user asked for it.
        what: String,
        /// The window of silence that expired.
        stall: Duration,
        /// How many attempts were made in total.
        tries: u32,
    },
}

/// Convenience alias.
pub type Result<T> = std::result::Result<T, HubError>;

/// How long a download may go nowhere, and how often it is started again.
///
/// One value for the whole process rather than a parameter on each of the
/// thirteen `fetch_*` entry points below. Those thirteen have around twenty
/// call sites spread across all four engines, and every one of them would have
/// to grow an argument carrying the same answer — because there *is* only one
/// answer per invocation: this is a property of the network the process is on,
/// not of the asset being fetched. It is the shape `cli_kit::init_logging`
/// already has, installed once from argv at the top of a command and read
/// wherever it is needed.
///
/// [`Retry::current`] is that reading, and it falls back to [`Retry::default`]
/// when nothing installed one — so a library caller, a test, and every
/// subcommand that does not take the flags all keep working, with the timeout
/// applied rather than skipped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Retry {
    /// How long a transfer may make **no progress** before it is abandoned.
    ///
    /// A stall window, **not** a deadline on the download, and the distinction
    /// is the whole reason this is not a `tokio::time::timeout` around the
    /// fetch: Whisper large-v3-turbo is 1.6 GB, and any deadline generous
    /// enough for that on a slow link is far too long to notice a stall. What
    /// is measured instead is **bytes arriving**: a transfer moving at any rate
    /// at all keeps resetting this, and only one that has stopped runs it out.
    /// Bytes rather than progress reports, for the reason [`Heartbeat`] gives.
    pub stall: Duration,
    /// How many times a fetch that stalled is started again from the top.
    ///
    /// Zero means "try once and report". It is also handed to hf-hub as its own
    /// per-request retry budget, so the two layers agree about how patient the
    /// user asked to be — see [`hub_client`] for why they are two layers.
    pub retries: u32,
}

/// A minute of silence is a stall on any link that was ever going to work: the
/// Hub's own connect and first-byte latencies are seconds, and a transfer that
/// is merely slow still reports progress every chunk.
const DEFAULT_STALL: Duration = Duration::from_secs(60);

/// Three, because the failures this retries are transient by construction and
/// a fourth attempt says more about the network being down than about luck.
const DEFAULT_RETRIES: u32 = 3;

/// Backoff between our own attempts: `BACKOFF_BASE`, doubling.
///
/// Deliberately **not** a flag. The two numbers a user can act on are how long
/// to wait and how many times to try; a third asking them to tune the pause
/// between attempts buys nothing they could measure, and it is dwarfed by the
/// stall window that precedes it anyway.
const BACKOFF_BASE: Duration = Duration::from_secs(1);

impl Default for Retry {
    fn default() -> Self {
        Self {
            stall: DEFAULT_STALL,
            retries: DEFAULT_RETRIES,
        }
    }
}

static INSTALLED: std::sync::OnceLock<Retry> = std::sync::OnceLock::new();

impl Retry {
    /// The policy this process fetches under.
    pub fn current() -> Self {
        INSTALLED.get().copied().unwrap_or_default()
    }

    /// Install this as the process-wide policy, if nothing has yet.
    ///
    /// Idempotent rather than fallible: a command parses its flags once, and a
    /// second install would be a caller bug that is not worth an error path in
    /// front of every download. The mechanism itself takes a `Retry`
    /// explicitly, so nothing in this crate is reachable *only* through the
    /// global — which is what keeps the tests below able to exercise a policy
    /// this never sees.
    pub fn install(self) {
        let _ = INSTALLED.set(self);
    }
}

/// The Hub client every fetch goes through.
///
/// Two layers of retry meet here and they divide cleanly:
///
/// - **hf-hub retries individual HTTP requests**, and already classifies what
///   is worth retrying exactly as this needs — a connection reset, a read
///   timeout, 408/429/500/502/503/504 are transient; a 404 on a filename that
///   does not exist is not, and comes straight back. That classifier is
///   `pub(crate)` to hf-hub, so reimplementing it here would mean a second
///   opinion about which failures are permanent, drifting from the first.
/// - **[`guarded`] retries a whole fetch that stalled**, which is the one
///   failure hf-hub cannot see: no request has failed, no status has arrived,
///   the response body has simply stopped.
///
/// So `--retries` is handed to both, and the outer loop only ever runs for a
/// stall — the two cannot multiply into `retries * retries` attempts.
///
/// The `read_timeout`/`connect_timeout` pair is what makes the inner layer able
/// to see a stall at all. They are per-operation, not per-transfer: `timeout()`
/// would cap the whole download and kill a healthy 1.6 GB fetch on a slow link.
fn hub_client(cache_dir: &Path, retry: Retry) -> Result<HFClient> {
    let http = reqwest::Client::builder()
        .connect_timeout(retry.stall)
        .read_timeout(retry.stall)
        // hf-hub sets its default headers only on a client it built itself, so
        // supplying one means supplying the `User-Agent` too. An anonymous
        // client is what the Hub rate-limits first, and that failure would be
        // intermittent, remote, and look like anything but a missing header.
        .user_agent(concat!("voice/", env!("CARGO_PKG_VERSION")))
        .build()?;
    Ok(HFClient::builder()
        .cache_dir(cache_dir.to_path_buf())
        .client(http)
        .retry_max_attempts(retry.retries as usize)
        .retry_base_delay(BACKOFF_BASE)
        .build()?)
}

/// Records when the transfer last moved, for [`guarded`]'s watchdog.
///
/// **Bytes, not events**, and that distinction is the whole of this type. A
/// plain HTTP body reports once per chunk, so an event *is* progress there —
/// but hf-hub's xet path spawns a poller that emits an `AggregateProgress`
/// every 100 ms whether or not a byte has arrived, and a watchdog fed on
/// events would take that tick for a healthy transfer and never fire. So the
/// clock is reset only when the count reported rises above the highest count
/// this attempt has seen.
///
/// `tokio::time::Instant` rather than `std::time::Instant` so a test can drive
/// the clock instead of waiting out a real stall window.
struct Heartbeat {
    at: std::sync::Mutex<tokio::time::Instant>,
    /// The highest byte count any event has reported during this attempt.
    ///
    /// Two counts arrive interleaved on the xet path — the batch aggregate and
    /// the per-file delta — and one watermark over both is still flat exactly
    /// when nothing is moving, which is all the watchdog asks of it.
    bytes: std::sync::atomic::AtomicU64,
}

impl Heartbeat {
    fn touch(&self) {
        *self.at.lock().expect("heartbeat mutex") = tokio::time::Instant::now();
    }
}

impl ProgressHandler for Heartbeat {
    fn on_progress(&self, event: &ProgressEvent) {
        use hf_hub::progress::DownloadEvent;

        let reported = match event {
            ProgressEvent::Download(DownloadEvent::AggregateProgress {
                bytes_completed, ..
            }) => *bytes_completed,
            ProgressEvent::Download(DownloadEvent::Progress { files }) => {
                files.iter().map(|f| f.bytes_completed).sum()
            }
            // `Start` and `Complete` carry no running count: they say the
            // transfer reached a new phase, which is movement by definition.
            _ => return self.touch(),
        };
        if reported
            > self
                .bytes
                .fetch_max(reported, std::sync::atomic::Ordering::Relaxed)
        {
            self.touch();
        }
    }
}

/// Run one Hub download under the stall watchdog, restarting it if it stalls.
///
/// `attempt` is a closure rather than a future because a stalled attempt is
/// dropped and a *fresh* one started — a future that has already stalled cannot
/// be polled back into life.
///
/// The watchdog exists because the timeouts on [`hub_client`]'s client do not
/// cover the whole path: hf-hub builds a second, no-redirect client of its own
/// for the metadata `HEAD` that opens every cached download, and offers no way
/// to configure it. A server that accepts the connection and then says nothing
/// stalls there, before any client of ours is reached. Watching for progress
/// catches that case for free, because a `HEAD` that never returns emits no
/// progress event either.
async fn guarded<F, Fut, T>(retry: Retry, what: &str, mut attempt: F) -> Result<T>
where
    F: FnMut(Progress) -> Fut,
    Fut: std::future::Future<Output = Result<T>>,
{
    let beat = Arc::new(Heartbeat {
        at: std::sync::Mutex::new(tokio::time::Instant::now()),
        bytes: std::sync::atomic::AtomicU64::new(0),
    });
    for try_index in 0..=retry.retries {
        // A restarted attempt counts from wherever it resumes, so the previous
        // one's watermark would sit above everything the new one reports and
        // the clock would never be reset again.
        beat.bytes.store(0, std::sync::atomic::Ordering::Relaxed);
        beat.touch();
        let watchdog = async {
            loop {
                let last = *beat.at.lock().expect("heartbeat mutex");
                if last.elapsed() >= retry.stall {
                    return;
                }
                tokio::time::sleep_until(last + retry.stall).await;
            }
        };
        tokio::select! {
            done = attempt(Progress::from(beat.clone())) => return done,
            () = watchdog => {}
        }
        if try_index < retry.retries {
            // Doubling from `BACKOFF_BASE`. The stall window already dominates
            // this, so it is politeness to the server rather than pacing.
            let pause = BACKOFF_BASE * 2u32.pow(try_index.min(6));
            tracing::warn!(
                "{what}: no data for {}s, retrying in {}s ({} attempt(s) left)",
                retry.stall.as_secs(),
                pause.as_secs(),
                retry.retries - try_index
            );
            tokio::time::sleep(pause).await;
        }
    }
    Err(HubError::Stalled {
        what: what.to_string(),
        stall: retry.stall,
        tries: retry.retries + 1,
    })
}

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
    fetch_at(model, None, cache_dir).await
}

/// The same, from one named revision of the repo rather than from its tip.
///
/// Almost every asset here wants the tip: a repo that publishes one canonical
/// file has nothing to pin against, and pinning one would freeze a fix nobody
/// benefits from refusing. The exception is a repo that publishes **many
/// models** and whose files a port's architecture is derived from — see
/// [`fetch_mdx23c`], where the checkpoint *is* the configuration — so a repo
/// that reorganised its directories would otherwise hand a caller a different
/// network under the same filename.
pub async fn fetch_at(
    model: &ModelRef,
    revision: Option<&str>,
    cache_dir: &Path,
) -> Result<PathBuf> {
    note_cache_move(cache_dir);
    let retry = Retry::current();
    let client = hub_client(cache_dir, retry)?;
    let what = format!("{}/{}/{}", model.owner, model.name, model.file);
    guarded(retry, &what, |progress| {
        let repo = client.model(model.owner.clone(), model.name.clone());
        let revision = revision.map(str::to_string);
        async move {
            Ok(repo
                .download_file()
                .filename(model.file.clone())
                .maybe_revision(revision)
                .progress(progress)
                .send()
                .await?)
        }
    })
    .await
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
    // `local_dir` below writes the file itself, so hf-hub's cache tree is never
    // reached on this path and the directory named here is inert. It is `dir`
    // rather than hf-hub's own default because an inert setting that names this
    // toolkit's own cache cannot become a live one pointing at
    // `~/.cache/huggingface` if that ever changes.
    let retry = Retry::current();
    let client = hub_client(dir, retry)?;
    let what = format!("{}/{}/{}", model.owner, model.name, model.file);
    // `local_dir` reproduces the repo's own directory structure, so the file
    // lands a level down; move it up and drop the wrapper if it is now empty.
    // Only the completed download is ever named `dest`, which is what makes the
    // reuse check above safe after an interrupted one.
    let path = guarded(retry, &what, |progress| {
        let repo = client.model(model.owner.clone(), model.name.clone());
        async move {
            Ok(repo
                .download_file()
                .filename(model.file.clone())
                .local_dir(dir.to_path_buf())
                .progress(progress)
                .send()
                .await?)
        }
    })
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

/// Where the Japanese dictionary comes from, and **the one asset in this crate
/// that is not on the Hugging Face Hub**: NAIST-JDic, compiled into
/// `jpreprocess`'s own binary layout and attached to its v0.15.0 release.
///
/// It gets a URL and a function of its own rather than a [`ModelRef`] because
/// that type is `owner/name/file` *inside a Hub repo*, and a release asset has
/// no repo path, no revision and no siblings to list. Filling those three fields
/// in for a GitHub tarball would mean a value whose every part misdescribes
/// where the bytes come from, and [`fetch`] would then have to branch on which
/// kind of `ModelRef` it held.
///
/// The version is pinned into the URL because the layout is the *reader's*, not
/// a standard: a dictionary is only guaranteed readable by the `jpreprocess`
/// release that compiled it, so moving that dependency means moving this line.
///
/// **Deliberately not `jpreprocess`'s `naist-jdic` cargo feature.** That feature
/// downloads the same tarball from its `build.rs`, which would charge every
/// `cargo build`, every `cargo test` and every CI job 28 MB before compiling a
/// line — a build-time download is exactly the shape `rvc-core/build.rs` exists
/// to refuse. Assets are fetched when a run needs them, never when a build does.
pub const NAIST_JDIC_URL: &str = "https://github.com/jpreprocess/jpreprocess/releases/download/v0.15.0/naist-jdic-jpreprocess.tar.gz";

/// The archive's exact size, checked once it has arrived.
///
/// Worth the constant because a truncated body is otherwise reported by the gzip
/// decoder as a corrupt archive, which reads as "the release is broken" and
/// sends the user to the wrong place — the transfer stopped early, so
/// [`fetch_naist_jdic`] retries it rather than saying so and stopping.
const NAIST_JDIC_BYTES: usize = 28_668_638;

/// The single directory the archive holds, and the name it keeps in the cache.
const NAIST_JDIC_DIR: &str = "naist-jdic";

/// Fetch and unpack the Japanese dictionary, returning the directory it landed in.
///
/// An **inference asset**, so it lands in the cache root beside the weights
/// rather than under [`pretrained_dir`]: nothing trains on it, and one copy
/// serves every run on the machine. Costly enough (28 MB packed, ~100 MB
/// unpacked) that it is fetched only when a Japanese run actually needs it.
///
/// Reuse is [`fetch_pretrained`]'s rule applied to a directory: the archive is
/// unpacked into a staging directory and only *renamed* into place once it is
/// whole, so a directory under the final name is always a complete dictionary
/// and an interrupted fetch can never be mistaken for one.
///
/// The retrying here is its own loop rather than [`guarded`]'s, because this is
/// the one asset nothing of hf-hub's touches: there is no client to hand a
/// retry budget to, so both layers are this function's. What it retries is
/// spelled out in [`retryable`] — a 404 on the pinned URL means the release
/// moved and comes back at once, since three more attempts would only make the
/// same answer slower to arrive.
pub async fn fetch_naist_jdic(cache_dir: &Path) -> Result<PathBuf> {
    let dest = cache_dir.join(NAIST_JDIC_DIR);
    if dest.exists() {
        return Ok(dest);
    }

    tracing::info!(
        "downloading the Japanese dictionary ({} MB) to {}",
        NAIST_JDIC_BYTES / 1_000_000,
        cache_dir.display()
    );
    let retry = Retry::current();
    // `read_timeout` rather than `timeout`, for the reason `hub_client` gives:
    // a deadline over the whole body is a size limit in disguise. This one is
    // 28 MB and would survive either, but the two paths must not disagree about
    // what `--download-timeout` means.
    let client = reqwest::Client::builder()
        .connect_timeout(retry.stall)
        .read_timeout(retry.stall)
        .user_agent(concat!("voice/", env!("CARGO_PKG_VERSION")))
        .build()?;

    let mut spent = 0;
    let body = loop {
        let attempt = async {
            let bytes = client
                .get(NAIST_JDIC_URL)
                .send()
                .await?
                .error_for_status()?
                .bytes()
                .await?;
            // A short body is the transfer stopping early, which the gzip
            // decoder would report as a corrupt archive two steps later.
            if bytes.len() != NAIST_JDIC_BYTES {
                return Err(HubError::Download(format!(
                    "{NAIST_JDIC_URL} gave {} bytes where the release is {NAIST_JDIC_BYTES} — \
                     the transfer stopped early",
                    bytes.len()
                )));
            }
            Ok(bytes)
        };
        match attempt.await {
            Ok(bytes) => break bytes,
            Err(e) if retryable(&e) && spent < retry.retries => {
                let pause = BACKOFF_BASE * 2u32.pow(spent.min(6));
                tracing::warn!(
                    "the Japanese dictionary failed to download ({e}), retrying in {}s \
                     ({} attempt(s) left)",
                    pause.as_secs(),
                    retry.retries - spent
                );
                tokio::time::sleep(pause).await;
                spent += 1;
            }
            Err(e) => return Err(e),
        }
    };

    // Inflating 28 MB and writing ~100 MB of it is blocking work, and every
    // caller is inside an async runtime. The staging directory carries the
    // process id so two runs fetching at once cannot unpack over each other.
    let staging = cache_dir.join(format!(
        "{NAIST_JDIC_DIR}.incomplete-{}",
        std::process::id()
    ));
    tokio::task::spawn_blocking(move || -> Result<PathBuf> {
        // Whatever an earlier attempt of *this* process left behind; a live
        // sibling's staging directory has a different pid and is untouched.
        let _ = std::fs::remove_dir_all(&staging);
        std::fs::create_dir_all(&staging)?;
        tar::Archive::new(flate2::read::GzDecoder::new(&body[..])).unpack(&staging)?;
        // The archive holds exactly one top-level directory, so the payload is a
        // level below the staging root.
        //
        // A sibling that finished while we were unpacking is **not** a failure:
        // the rename is what publishes the name, so a directory called `dest` is
        // a whole dictionary by construction and losing the race means the asset
        // is already there. Reporting an error here would also be stricter than
        // the `dest.exists()` fast path at the top, which returns it with no
        // check at all. What must not be skipped either way is clearing the
        // staging tree — it is ~100 MB unpacked, and leaving one behind per
        // collision is how a cache grows without anything ever reading what it
        // grew.
        if let Err(e) = std::fs::rename(staging.join(NAIST_JDIC_DIR), &dest) {
            let _ = std::fs::remove_dir_all(&staging);
            if dest.exists() {
                return Ok(dest);
            }
            return Err(HubError::Io(std::io::Error::new(
                e.kind(),
                format!(
                    "unpacked the Japanese dictionary but could not move it into {}: {e}",
                    dest.display(),
                ),
            )));
        }
        let _ = std::fs::remove_dir(&staging);
        Ok(dest)
    })
    .await
    .map_err(|e| HubError::Io(std::io::Error::other(e)))?
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

/// UVR's model collection, which is where the separation checkpoint lives.
///
/// A **third-party mirror** of several projects' releases rather than any of
/// their own repos, and the reason to prefer it is that it holds the MDX23C
/// checkpoint *and* the older MDX-Net v2 ONNX graphs at one revision, so a
/// single pin covers both runtimes' assets if the second is ever wired up.
pub const DEFAULT_MDX23C: (&str, &str) = ("Politrees", "UVR_resources");

/// The revision every MDX asset is fetched from, and it is **not** decoration.
///
/// A separation network's architecture *is* its checkpoint — `dim_f`, `n_fft`,
/// the channel widths and the block counts all differ across the MDX family,
/// and the Burn port pins them as constants so a disagreeing file fails as a
/// shape mismatch. This repo publishes dozens of models and has reorganised its
/// directories before; without the pin, a re-upload under the same filename
/// would hand a caller a different network, which is a load failure at best.
const MDX23C_REVISION: &str = "929e057b81aa49bc2e6490bef8671f47b2c120f6";

/// `MDX23C-8KFFT-InstVoc_HQ`, 448 MB: the vocals/instrumental separator.
///
/// Its `model_2_stem_full_band_8k.yaml` sits beside it in the repo and is
/// deliberately **not** fetched: nothing reads it. The port carries the same
/// values as a constant, for the reason above — a config parsed at run time
/// could disagree with the weights and be believed, where a constant makes the
/// disagreement a shape mismatch on load.
pub async fn fetch_mdx23c(cache_dir: &Path) -> Result<PathBuf> {
    let (owner, name) = DEFAULT_MDX23C;
    let model = ModelRef::new(owner, name, "models/MDX23C/MDX23C-8KFFT-InstVoc_HQ.ckpt");
    fetch_at(&model, Some(MDX23C_REVISION), cache_dir).await
}

/// Whether starting [`fetch_naist_jdic`]'s download again could plausibly
/// succeed where this attempt did not.
///
/// The distinction is the point: a 404 on the pinned release URL means the
/// asset moved, and retrying it three times with backoff turns a clear answer
/// into a slow one while telling the user nothing new. What *is* worth another
/// go is anything that says the network faltered rather than that the file is
/// wrong — a timeout, a refused or reset connection, a 429 from the CDN, and
/// any 5xx, which is the server saying "not now" rather than "not here".
/// A short body is here too, and by construction: the length is known, so a
/// body that fell short of it is a transfer that stopped, never a release that
/// changed size — that would fail the check on every attempt and 404 first.
///
/// The same list hf-hub applies to its own requests, which is deliberate; its
/// classifier is `pub(crate)`, so agreeing with it is a thing this has to do on
/// purpose rather than by calling it.
fn retryable(e: &HubError) -> bool {
    match e {
        HubError::Download(_) => true,
        HubError::Http(e) => {
            e.is_timeout()
                || e.is_connect()
                || e.status().is_some_and(|s| {
                    s.is_server_error()
                        || s == reqwest::StatusCode::TOO_MANY_REQUESTS
                        || s == reqwest::StatusCode::REQUEST_TIMEOUT
                })
        }
        _ => false,
    }
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

    /// A policy short enough to read, driven by a paused clock rather than a
    /// real one — `start_paused` advances time only when every task is waiting
    /// on a timer, so these run instantly and cannot be flaky under load.
    fn quick() -> Retry {
        Retry {
            stall: Duration::from_secs(30),
            retries: 2,
        }
    }

    /// The point of the unit: a fetch that never produces a byte comes back as
    /// an error rather than hanging, and it costs exactly the attempts asked
    /// for. `pending()` is the stalled server — a connection that was accepted
    /// and then said nothing looks like this from here.
    #[tokio::test(start_paused = true)]
    async fn a_transfer_that_never_moves_is_abandoned() {
        let started = std::sync::Arc::new(std::sync::atomic::AtomicU32::new(0));
        let counter = started.clone();
        let started_at = tokio::time::Instant::now();

        let e = guarded(quick(), "a stalled fetch", |_progress| {
            counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            std::future::pending::<Result<PathBuf>>()
        })
        .await
        .expect_err("a fetch that never moves must not succeed");

        assert!(matches!(e, HubError::Stalled { tries: 3, .. }), "{e}");
        assert_eq!(started.load(std::sync::atomic::Ordering::Relaxed), 3);
        // Three stall windows, plus the 1s + 2s backoff between them.
        assert_eq!(started_at.elapsed(), Duration::from_secs(30 * 3 + 3));
        // The message has to name both knobs, since which one to turn depends
        // on whether the link is slow or flaky and only the user knows which.
        let text = e.to_string();
        assert!(text.contains("--download-timeout"), "{text}");
        assert!(text.contains("--retries"), "{text}");
    }

    /// One xet-shaped progress event: a cumulative byte count for the batch.
    fn moved(bytes: u64) -> ProgressEvent {
        ProgressEvent::Download(hf_hub::progress::DownloadEvent::AggregateProgress {
            bytes_completed: bytes,
            total_bytes: 1 << 30,
            bytes_per_sec: None,
        })
    }

    /// The other half, and the reason this is a stall watchdog rather than a
    /// `tokio::time::timeout`: a transfer that keeps reporting bytes runs for
    /// **longer** than the stall window and is not touched. Whisper
    /// large-v3-turbo is 1.6 GB, so any deadline is a size limit in disguise.
    #[tokio::test(start_paused = true)]
    async fn a_slow_transfer_that_keeps_moving_is_left_alone() {
        let retry = quick();
        let got = guarded(retry, "a slow fetch", |progress| async move {
            // Ten chunks at 20s each: 200s in total against a 30s window, so a
            // deadline of any size that admits this admits a stall too.
            for chunk in 1..=10 {
                tokio::time::sleep(Duration::from_secs(20)).await;
                progress.on_progress(&moved(chunk * 1_000_000));
            }
            Ok(PathBuf::from("/done"))
        })
        .await
        .expect("a transfer reporting progress must not be abandoned");
        assert_eq!(got, PathBuf::from("/done"));
    }

    /// The trap the byte watermark exists for. hf-hub's xet path spawns a
    /// poller that emits an `AggregateProgress` every 100 ms whether or not a
    /// byte has arrived, so a watchdog that counted *events* would read that
    /// tick as a healthy transfer and never fire — on the one path that moves
    /// the largest files here.
    #[tokio::test(start_paused = true)]
    async fn a_progress_tick_that_reports_no_new_bytes_is_not_progress() {
        let e = guarded(quick(), "a stalled xet batch", |progress| async move {
            for _ in 0..1_000 {
                tokio::time::sleep(Duration::from_millis(100)).await;
                progress.on_progress(&moved(4096));
            }
            Ok(PathBuf::from("/never"))
        })
        .await
        .expect_err("a byte count that stopped rising is a stall");
        assert!(matches!(e, HubError::Stalled { .. }), "{e}");
    }

    /// A failure that is not a stall is the inner layer's to retry, so it comes
    /// straight back out — this is what stops `--retries 3` becoming 3 × 3
    /// attempts once hf-hub's own budget is counted.
    #[tokio::test(start_paused = true)]
    async fn a_real_failure_is_not_retried_here() {
        let tries = std::sync::Arc::new(std::sync::atomic::AtomicU32::new(0));
        let counter = tries.clone();
        let e = guarded(quick(), "a missing file", |_progress| {
            counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            std::future::ready(Err::<PathBuf, _>(HubError::Download("no such file".into())))
        })
        .await
        .expect_err("the error must not be swallowed");
        assert!(matches!(e, HubError::Download(_)), "{e}");
        assert_eq!(tries.load(std::sync::atomic::Ordering::Relaxed), 1);
    }

    /// [`retryable`] is the classifier for the one asset hf-hub never touches,
    /// so it is the only place a 404 could start costing a retry budget.
    #[test]
    fn only_a_faltering_network_is_worth_another_attempt() {
        // A short body is a transfer that stopped: the length is known up
        // front, so this cannot mean the release changed size.
        assert!(retryable(&HubError::Download("28 bytes".into())));
        // A missing directory, a corrupt archive: nothing a second GET fixes.
        assert!(!retryable(&missing("a file", Path::new("/tmp"))));
        assert!(!retryable(&HubError::Stalled {
            what: "x".into(),
            stall: Duration::from_secs(1),
            tries: 1,
        }));
    }

    /// The default must be a policy, not the absence of one: every caller that
    /// installs nothing — a library user, a test, `convert`, `train`, and the
    /// bare filter — still fetches with the timeout applied.
    #[test]
    fn nothing_installed_still_has_a_timeout() {
        let policy = Retry::current();
        assert_eq!(policy, Retry::default());
        assert!(policy.stall > Duration::ZERO);
    }

    /// `default_rmvpe`'s whole job is a one-word filename change on the same
    /// first-party repo — this pins both words.
    #[test]
    fn default_rmvpe_selects_the_right_file_per_format() {
        let onnx = default_rmvpe(WeightFormat::Onnx);
        assert_eq!(onnx.owner, "lj1995");
        assert_eq!(onnx.name, "VoiceConversionWebUI");
        assert_eq!(onnx.file, "rmvpe.onnx");

        let torch = default_rmvpe(WeightFormat::Torch);
        assert_eq!(torch.owner, "lj1995");
        assert_eq!(torch.name, "VoiceConversionWebUI");
        assert_eq!(torch.file, "rmvpe.pt");
    }

    /// The pre-existing ONNX answer must be untouched by adding `WeightFormat`
    /// beside it — every existing call site depends on this exact repo/file.
    #[test]
    fn default_contentvec_is_unchanged_by_the_weight_format_addition() {
        let cv = default_contentvec();
        assert_eq!(cv.owner, "NaruseMioShirakana");
        assert_eq!(cv.name, "MoeSS-SUBModel");
        assert_eq!(cv.file, "vec-768-layer-12.onnx");
    }

    /// The PyTorch ContentVec directory: first-party repo, three files, the
    /// same triple `fetch_contentvec`'s `Torch` arm loops over.
    #[test]
    fn contentvec_torch_repo_and_files_are_the_hubert_base_triple() {
        assert_eq!(CONTENTVEC_TORCH_REPO, ("lj1995", "VoiceConversionWebUI"));
        assert_eq!(
            CONTENTVEC_TORCH_FILES,
            [
                "hubert_base/config.json",
                "hubert_base/pytorch_model.bin",
                "hubert_base/preprocessor_config.json",
            ]
        );
    }

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
