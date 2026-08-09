//! Command-line argument definitions (clap derive).

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use clap::{Args, Subcommand, ValueEnum};
pub use cli_kit::{Backend, CompletionsArgs};
use rvc_core::{ANALYSIS_SR, StreamParams};

/// The whole of the `rvc` command tree, defined once and worn two ways: the
/// `rvc` binary flattens it at its top level, `voice` nests it under an `rvc`
/// subcommand. Every engine here has the same shape — the bare invocation is
/// the stdin→stdout filter, and everything else is a subcommand.
#[derive(Debug, Args)]
pub struct RvcCli {
    #[command(flatten)]
    pub filter: FilterArgs,
    #[command(subcommand)]
    pub command: Option<RvcCommand>,
}

/// What this engine can do, and nothing about the executable that hosts it.
///
/// **`completions` is not a member, on purpose.** A completion script describes
/// one binary, so a nested `voice rvc completions` could only ever emit
/// `voice`'s — which is exactly what it used to do. Each `main.rs` flattens this
/// enum into its own and adds `Completions` beside it, so the standalone binary
/// keeps the subcommand and `voice` grows only one, at its top level.
#[derive(Debug, Subcommand)]
pub enum RvcCommand {
    /// Batch-convert audio files into the target timbre (WAV output).
    Convert(ConvertArgs),
    /// Train an RVC generator on a corpus (native Rust / burn).
    // Boxed because `TrainArgs` alone is ~320 bytes against 72 for the next
    // largest variant, and clippy is right that every parse shouldn't pay it.
    Train(Box<TrainArgs>),
    /// Prefetch the weights a conversion needs, so the first run is offline.
    Download(DownloadArgs),
}

/// Shared options for locating the three ONNX models.
#[derive(Debug, Args, Clone)]
pub struct ModelOpts {
    /// Trained RVC generator weights (`.safetensors`, or `.onnx` for an export).
    #[arg(short, long)]
    pub model: Option<PathBuf>,
    /// Generator output sample rate (40000 or 48000).
    #[arg(long, default_value_t = 48000)]
    pub model_sr: u32,
    /// ContentVec encoder: an `.onnx` file, or the directory a PyTorch one was
    /// unpacked to [default: auto-downloaded from Hugging Face in the format
    /// `--content-vec-backend` needs].
    #[arg(long)]
    pub content: Option<PathBuf>,
    /// RMVPE F0 estimator, `.onnx` or `.pt` [default: auto-downloaded from
    /// Hugging Face in the format `--rmvpe-backend` needs].
    #[arg(long)]
    pub rmvpe: Option<PathBuf>,
    /// Directory the downloaded models are cached in. Shared by every engine
    /// unless `$RVC_CACHE_DIR` (or `$VOICE_CACHE_DIR`) says otherwise.
    #[arg(long, default_value_os_t = hub_kit::cache_dir_for("RVC_CACHE_DIR"))]
    pub cache_dir: PathBuf,
    /// Speaker id fed to the generator (single-speaker models use 0).
    #[arg(long, default_value_t = 0)]
    pub speaker_id: i64,
    /// RMVPE voicing floor: frames whose salience falls below it are emitted as
    /// 0 Hz. Lower to keep more of a quiet take voiced; raise to hand more of it
    /// to the generator's noise branch, which is what renders breath.
    #[arg(long, default_value_t = rvc_core::FeatureExtractor::F0_THRESHOLD, value_name = "SALIENCE")]
    pub f0_threshold: f32,
}

impl ModelOpts {
    /// The generator weights every conversion path needs.
    ///
    /// Optional to clap only because these options are also flattened beside
    /// the subcommands, for the bare filter invocation — requiring `-m` there
    /// would require it of `train` and `download` too. So "required" is
    /// decided here, and callers check it before anything is fetched or loaded.
    pub fn model(&self) -> Result<&Path> {
        self.model.as_deref().context(
            "-m/--model is required: the trained generator weights \
             (train one with the `train` subcommand)",
        )
    }

    /// Reject values clap's types accept but the pipeline cannot use.
    pub fn verify(&self) -> Result<()> {
        anyhow::ensure!(
            matches!(self.model_sr, 40_000 | 48_000),
            "--model-sr must be 40000 or 48000, not {}: it is a property of the \
             trained generator, not a resampling request",
            self.model_sr
        );
        anyhow::ensure!(
            self.speaker_id >= 0,
            "--speaker-id must not be negative (single-speaker models use 0)"
        );
        // RMVPE's salience is a softmax over cents bins, so it lives in [0, 1].
        // At 1.0 nothing is ever voiced and the generator renders the whole take
        // on its noise branch — a legal setting, and not one to arrive at by
        // typo, which is why the bound is stated rather than clamped.
        anyhow::ensure!(
            (0.0..=1.0).contains(&self.f0_threshold),
            "--f0-threshold is an RMVPE salience and must be in [0, 1], not {}: \
             at 0 every frame is voiced and at 1 none is",
            self.f0_threshold
        );
        Ok(())
    }
}

/// Which runtime runs the two **feature** models, ContentVec and RMVPE.
///
/// They are independent of the generator and of each other — an ONNX generator
/// with a LibTorch RMVPE is a legitimate configuration — so each gets its own
/// flag. Both default to `--backend`, and they are `Option` for exactly that
/// reason: clap derive cannot default one argument to another, so the fallback
/// is applied after the parse, in [`Self::resolve`].
///
/// Flattened into `convert`, the bare filter, `train` and `download` alike.
/// One definition, so the four cannot drift the way the `--backend`/`--device`
/// help text beside them already had.
#[derive(Debug, Args, Clone, Copy)]
pub struct FeatureBackendOpts {
    /// Runtime for the ContentVec content encoder — the same spellings
    /// `--backend` takes [default: whatever `--backend` resolves to; under
    /// `train`, always `onnx`].
    #[arg(long, value_enum, value_name = "BACKEND")]
    pub content_vec_backend: Option<Backend>,
    /// Runtime for the RMVPE F0 estimator — the same spellings `--backend`
    /// takes [default: whatever `--backend` resolves to; under `train`, always
    /// `onnx`].
    #[arg(long, value_enum, value_name = "BACKEND")]
    pub rmvpe_backend: Option<Backend>,
}

/// The runtime each feature model will actually run on, once `--backend` has
/// been applied as the default and `auto` is gone.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FeatureBackends {
    pub content: Backend,
    pub rmvpe: Backend,
}

impl FeatureBackendOpts {
    /// Apply `--backend` as the default.
    ///
    /// `generator` is the **already-resolved** generator backend, never the raw
    /// `--backend` value: resolving twice would let a `.onnx` generator settle
    /// on ORT while its feature models independently re-derived `auto` from the
    /// hardware and went to Burn, which is a mixed pipeline nobody asked for.
    /// Inheriting an unresolved `auto` is not a runtime either.
    ///
    /// A backend named on either flag is passed through untouched — the
    /// contract [`Backend::resolve`] keeps and its tests assert. Substituting
    /// one the user did not ask for is how somebody ends up loading weights in
    /// the wrong format and blaming the model.
    ///
    /// The one value that is not passed through is `auto`, and it is not an
    /// exception: **`--rmvpe-backend auto` means the same as leaving the flag
    /// off**, which is what the help text promises. It cannot mean "re-derive
    /// from the hardware", because that is the double resolution above; and it
    /// must not be carried through as `Auto`, because `Auto` is not a runtime —
    /// it would leave a value that matches no arm and maps to no weight format.
    pub fn resolve(self, generator: Backend) -> FeatureBackends {
        let inherit = |chosen: Option<Backend>| match chosen {
            None | Some(Backend::Auto) => generator,
            Some(explicit) => explicit,
        };
        FeatureBackends {
            content: inherit(self.content_vec_backend),
            rmvpe: inherit(self.rmvpe_backend),
        }
    }
}

/// The weight format a backend can load — which is what decides *which file*
/// gets downloaded for it.
///
/// The mapping lives in the engine on purpose, and it is the only place in the
/// workspace that knows both halves. `cli-kit` owns [`Backend`] and states that
/// it knows nothing about weight formats; `hub-kit` owns
/// [`hub_kit::WeightFormat`] and must not depend on `cli-kit`. Only the engine
/// loading the file knows that a PyTorch ContentVec is a *directory* where the
/// ONNX one is a single file.
///
/// `auto` cannot reach here — [`FeatureBackendOpts::resolve`] turns it into the
/// generator's backend — but it is mapped rather than panicked on, since the
/// answer for a Burn backend is the same whichever one it turns out to be.
pub fn weight_format(backend: Backend) -> hub_kit::WeightFormat {
    match backend {
        Backend::Onnx => hub_kit::WeightFormat::Onnx,
        // Every Burn compute backend reads the same PyTorch checkpoint: the
        // format is a property of the weights, not of the kernels.
        Backend::Cuda | Backend::Tch | Backend::Wgpu | Backend::Auto => {
            hub_kit::WeightFormat::Torch
        }
    }
}

/// How the overlap between consecutive output blocks is weighted — the CLI
/// spelling of [`rvc_core::CrossfadeShape`].
///
/// A second enum rather than a `ValueEnum` on the core type, for the reason
/// `rvc-core` has no clap dependency and should not grow one: it is a library
/// that knows about models, not about command lines. Mapping at the boundary is
/// what `train_backend` does with [`Backend`] for the same reason. Two variants,
/// no aliases, so there is nothing here that can drift the way four independent
/// `--backend` enums did.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum CrossfadeShape {
    /// Gains sum to one in amplitude. The default, and the curve every voice
    /// converted by this toolkit so far was joined with.
    Linear,
    /// Gains sum to one in power (`cos`/`sin`), which is the right sum when the
    /// two sides of a seam agree on content but not on phase.
    EqualPower,
}

impl From<CrossfadeShape> for rvc_core::CrossfadeShape {
    fn from(shape: CrossfadeShape) -> Self {
        match shape {
            CrossfadeShape::Linear => Self::Linear,
            CrossfadeShape::EqualPower => Self::EqualPower,
        }
    }
}

/// The streaming geometry — block, look-back context and crossfade — in seconds.
///
/// This is the whole latency-versus-quality dial, and until now none of it was
/// reachable: a user tuning latency reached for `--chunk`, which is the stdin
/// *read* size and changes nothing about how audio is processed.
///
/// The three lengths are `Option` for the reason [`FeatureBackendOpts`]'s are —
/// clap derive cannot default one argument to another — and here the default is
/// not even a constant. It is whichever preset the hosting command was going to
/// use, and the two disagree: the filter's block is 0.5 s where `convert`'s is
/// 15 s. So the fallback is applied after the parse, in [`Self::resolve`],
/// against the preset the caller passes. Two structs with two sets of
/// `default_value_t` would state each help string twice and let the two drift.
///
/// **Omitting a flag reuses the preset's own sample count**, never a seconds
/// value converted back — so leaving these off is byte-identical to the
/// behaviour before they existed, by construction rather than by rounding luck.
#[derive(Debug, Args, Clone, Copy)]
pub struct GeometryOpts {
    /// Seconds of input converted per block. The latency dial: the filter
    /// cannot emit anything until a whole block has arrived, and a longer block
    /// gives the content and F0 models more to work with
    /// [default: 0.5 streaming, 15 under `convert`].
    #[arg(long, value_name = "SECONDS")]
    pub block_secs: Option<f32>,
    /// Seconds of left look-back prepended to each block so its analysis has
    /// history, then dropped from the output. Costs inference time, not latency.
    /// Must not exceed `--block-secs`
    /// [default: 0.25 streaming, 0.5 under `convert`].
    #[arg(long, value_name = "SECONDS")]
    pub context_secs: Option<f32>,
    /// Seconds of output blended between consecutive blocks to hide the seam.
    /// Must not exceed `--context-secs` [default: 0.05].
    #[arg(long, value_name = "SECONDS")]
    pub crossfade_secs: Option<f32>,
    /// Curve the crossfade is blended with: `linear`, or `equal-power` for
    /// seams that sound like a momentary dip rather than a click.
    ///
    /// **`linear` is the default and stays the default**: it is what every voice
    /// this toolkit has converted was joined with, and changing the curve
    /// changes the audio of every existing user without them asking.
    #[arg(long, value_enum, default_value_t = CrossfadeShape::Linear)]
    pub crossfade_shape: CrossfadeShape,
}

impl GeometryOpts {
    /// Apply `preset` as the default for whichever lengths were not given.
    pub fn resolve(self, preset: StreamParams) -> StreamParams {
        let samples = |secs: Option<f32>, default: usize| match secs {
            // `round`, not a truncating cast: 0.05 f32 is a hair over 0.05, so
            // `0.05 * 16000` is 800.0000119 and truncation would make the flag's
            // own documented default land one sample off the preset it names.
            Some(s) => (s * ANALYSIS_SR as f32).round() as usize,
            None => default,
        };
        StreamParams {
            block: samples(self.block_secs, preset.block),
            context: samples(self.context_secs, preset.context),
            crossfade: samples(self.crossfade_secs, preset.crossfade),
            shape: self.crossfade_shape.into(),
        }
    }

    /// Reject a geometry the streaming converter cannot honour.
    ///
    /// Takes the same `preset` [`Self::resolve`] does, because the dangerous
    /// combinations are the *mixed* ones — one length named on the command line
    /// against another that came from the preset — so checking the flags in
    /// isolation would miss exactly the case worth catching. Errors therefore
    /// say which of the two numbers the user actually typed.
    ///
    /// Called from the hosting command's `verify`, which runs before any model
    /// is fetched or loaded: a geometry that cannot work should cost a message,
    /// not two downloads and a GPU init.
    pub fn verify(&self, preset: StreamParams) -> Result<()> {
        for (flag, secs) in [
            ("--block-secs", self.block_secs),
            ("--context-secs", self.context_secs),
            ("--crossfade-secs", self.crossfade_secs),
        ] {
            if let Some(secs) = secs {
                anyhow::ensure!(
                    secs.is_finite() && secs >= 0.0,
                    "{flag} must be a non-negative, finite number of seconds, not {secs}"
                );
            }
        }

        let params = self.resolve(preset);
        // Reported in seconds, since that is the unit the flags are in, and
        // tagged so a message about a value nobody typed says so.
        let named = |secs: Option<f32>, resolved: usize| match secs {
            Some(_) => format!("{:.4} s", resolved as f32 / ANALYSIS_SR as f32),
            None => format!(
                "{:.4} s, this command's default",
                resolved as f32 / ANALYSIS_SR as f32
            ),
        };
        let block = named(self.block_secs, params.block);
        let context = named(self.context_secs, params.context);
        let crossfade = named(self.crossfade_secs, params.crossfade);

        // A block that rounds to nothing is a hang, not bad audio: `push` loops
        // `while buf.len() >= block`, draining zero samples each time.
        anyhow::ensure!(
            params.block > 0,
            "--block-secs ({block}) rounds to zero 16 kHz samples: a block that is never \
             full is never converted, so the filter would read its input forever without \
             emitting anything. The smallest usable value is 1/16000 s."
        );
        // `process_block` carries the tail of the block it just processed as the
        // next block's context, so a context longer than a block cannot be
        // filled — `ctx_n = context.min(block.len())` clamps it away. Silently
        // honouring less than was asked for is the failure worth refusing.
        anyhow::ensure!(
            params.context <= params.block,
            "--context-secs ({context}) must not exceed --block-secs ({block}): the \
             look-back is the tail of the previous block, so anything past one block's \
             worth does not exist to be prepended and would be silently dropped"
        );
        // The kept region of each block is its own output *plus* `crossfade`
        // samples of look-back, taken from the output the context prefix
        // produced. With no context to take it from, `start` saturates to zero
        // and every block emits its context prefix as audio — duplicated speech,
        // not a seam artefact.
        anyhow::ensure!(
            params.crossfade <= params.context,
            "--crossfade-secs ({crossfade}) must not exceed --context-secs ({context}): \
             the crossfade is an overlap, and the samples it overlaps into come out of \
             the context each block analyses and then drops. A longer crossfade leaves \
             nothing to take them from, so every block would emit its context prefix as \
             audio and repeat that much speech at each join."
        );
        Ok(())
    }
}

#[derive(Debug, Args)]
pub struct ConvertArgs {
    #[command(flatten)]
    pub models: ModelOpts,
    /// One or more input audio files (mp3, wav, ...).
    #[arg(required = true)]
    pub input: Vec<PathBuf>,
    /// Directory to write converted `<stem>_<model>.wav` files into.
    #[arg(short = 'o', long, default_value = ".")]
    pub output_dir: PathBuf,
    /// Pitch shift in semitones.
    #[arg(short = 't', long, default_value_t = 0)]
    pub transpose: i32,
    /// Inference backend: `auto`, `onnx`, `cuda` (aliases `burn`, `burn-cuda`),
    /// `tch` (`libtorch`, `burn-tch`) or `wgpu` (`webgpu`, `burn-wgpu`).
    #[arg(long, value_enum, default_value_t = Backend::Auto)]
    pub backend: Backend,
    /// Compute device: `auto` (fastest visible), `cpu`, `gpu`, `gpu:N`, `mps` or
    /// `vulkan` (`cuda`/`cuda:N` also accepted). The `cuda` backend has GPUs only.
    #[arg(long, default_value = "auto", value_name = "DEVICE", value_parser = cli_kit::parse_device)]
    pub device: burn_kit::DeviceSpec,
    #[command(flatten)]
    pub features: FeatureBackendOpts,
    #[command(flatten)]
    pub geometry: GeometryOpts,
    #[command(flatten)]
    pub denoise: DenoiseOpts,
}

impl ConvertArgs {
    /// Reject values clap's types accept but the pipeline cannot use.
    pub fn verify(&self) -> Result<()> {
        self.models.verify()?;
        self.geometry.verify(StreamParams::batch())?;
        self.denoise.verify()
    }
}

/// The bare invocation: raw f32le mono PCM in, raw f32le mono PCM out.
#[derive(Debug, Args)]
pub struct FilterArgs {
    #[command(flatten)]
    pub models: ModelOpts,
    /// Pitch shift in semitones.
    #[arg(short = 't', long, default_value_t = 0)]
    pub transpose: i32,
    /// Samples per input read from stdin (16 kHz mono f32le). This is the read
    /// size and nothing more — the audio is still converted a `--block-secs`
    /// block at a time, which is the knob latency is tuned with.
    #[arg(long, default_value_t = 1600)]
    pub chunk: usize,
    /// Inference backend: `auto`, `onnx`, `cuda` (aliases `burn`, `burn-cuda`),
    /// `tch` (`libtorch`, `burn-tch`) or `wgpu` (`webgpu`, `burn-wgpu`). Prefer
    /// `tch` or `onnx` here — `cuda` does not keep up with realtime.
    #[arg(long, value_enum, default_value_t = Backend::Auto)]
    pub backend: Backend,
    /// Compute device: `auto` (fastest visible), `cpu`, `gpu`, `gpu:N`, `mps` or
    /// `vulkan` (`cuda`/`cuda:N` also accepted). The `cuda` backend has GPUs only.
    #[arg(long, default_value = "auto", value_name = "DEVICE", value_parser = cli_kit::parse_device)]
    pub device: burn_kit::DeviceSpec,
    #[command(flatten)]
    pub features: FeatureBackendOpts,
    #[command(flatten)]
    pub geometry: GeometryOpts,
    #[command(flatten)]
    pub denoise: DenoiseOpts,
}

impl FilterArgs {
    /// Reject values clap's types accept but the pipeline cannot use.
    pub fn verify(&self) -> Result<()> {
        self.models.verify()?;
        // A zero-sample read would spin on stdin forever without ever handing
        // the converter a block to work on.
        anyhow::ensure!(self.chunk > 0, "--chunk must be at least 1 sample");
        self.geometry.verify(StreamParams::realtime())?;
        self.denoise.verify()
    }
}

/// De-hiss options shared by the streaming filter and `convert`. Off unless
/// `--denoise` is given; the tuning flags only take effect when it is.
///
/// The switch is this engine's and the tuning is not: de-hiss is an *optional*
/// stage here, on a pipeline that otherwise does not run it, so `--denoise` has
/// a job to do — but the three `anlmdn` knobs and the `research > patch`
/// invariant are the same wherever the stage runs, and they live in
/// [`cli_kit::DenoiseOpts`] so this engine and corpus preparation cannot end up
/// with two spellings or two sets of defaults.
#[derive(Debug, Args, Clone)]
pub struct DenoiseOpts {
    /// Remove steady background hiss from the output (ffmpeg `anlmdn`
    /// non-local-means de-noise, run in-process, tuned to preserve the soft
    /// broadband texture of quiet, breathy content).
    #[arg(long)]
    pub denoise: bool,
    #[command(flatten)]
    pub tuning: cli_kit::DenoiseOpts,
}

impl DenoiseOpts {
    /// Reject a de-hiss configuration `anlmdn` would refuse or misbehave on.
    ///
    /// Only when `--denoise` is on: the tuning flags have defaults, so checking
    /// them unconditionally would reject a run that never denoises anything.
    pub fn verify(&self) -> Result<()> {
        if !self.denoise {
            return Ok(());
        }
        self.tuning.verify()
    }

    /// The [`rvc_core::DenoiseParams`] these flags describe, or `None` when
    /// `--denoise` was not passed (stage disabled).
    pub fn params(&self) -> Option<rvc_core::DenoiseParams> {
        self.denoise.then(|| self.tuning.params())
    }
}

/// Prefetch what a conversion would fetch on its first run.
///
/// It carries `--backend` for one reason: **which weights are the right ones is
/// a property of the runtime that will read them**, and ONNX Runtime and Burn
/// read different files. Without it this command could only ever guess, and it
/// used to guess ONNX — which is wrong for anybody whose next command is
/// `--backend tch`, and is the opposite of "fetches exactly what a default bare
/// invocation would fetch on demand".
///
/// There is no `-m` here, so `auto` has no weights to inspect and resolves on
/// hardware alone. Pass `--backend onnx` when the generator you will run is an
/// `.onnx` export.
#[derive(Debug, Args)]
pub struct DownloadArgs {
    /// Directory the downloaded models are cached in. Shared by every engine
    /// unless `$RVC_CACHE_DIR` (or `$VOICE_CACHE_DIR`) says otherwise.
    #[arg(long, default_value_os_t = hub_kit::cache_dir_for("RVC_CACHE_DIR"))]
    pub cache_dir: PathBuf,
    /// Runtime the fetched weights must load into: `auto`, `onnx`, `cuda`
    /// (aliases `burn`, `burn-cuda`), `tch` (`libtorch`, `burn-tch`) or `wgpu`
    /// (`webgpu`, `burn-wgpu`). ONNX Runtime and Burn read different files.
    #[arg(long, value_enum, default_value_t = Backend::Auto)]
    pub backend: Backend,
    #[command(flatten)]
    pub features: FeatureBackendOpts,
    /// Override the ContentVec repo as `owner/name:file`
    /// [default: the toolkit's ContentVec for the chosen backend].
    #[arg(long)]
    pub content: Option<String>,
    /// Override the RMVPE repo as `owner/name:file`
    /// [default: the toolkit's RMVPE for the chosen backend].
    #[arg(long)]
    pub rmvpe: Option<String>,
    #[command(flatten)]
    pub download: cli_kit::DownloadOpts,
}

#[derive(Debug, Args)]
pub struct TrainArgs {
    /// Corpus: one or more audio files of the target voice.
    #[arg(required = true)]
    pub data: Vec<PathBuf>,
    /// Output path for the trained generator (writes `<path>.safetensors`).
    #[arg(short = 'o', long, default_value = "models/voice")]
    pub out: PathBuf,
    /// Generator output sample rate (48000).
    #[arg(long, default_value_t = 48000)]
    pub model_sr: u32,
    /// Number of training epochs; stop early with `q` or Ctrl-C (saves).
    #[arg(short = 'e', long, default_value_t = 5)]
    pub epochs: u32,
    /// Mini-batch size (lower it if a 6 GB GPU runs out of memory).
    #[arg(short = 'b', long, default_value_t = 4)]
    pub batch_size: usize,
    /// Speaker id embedded in the generator (single-speaker: 0).
    #[arg(long, default_value_t = 0)]
    pub speaker_id: i64,
    /// Base AdamW learning rate.
    #[arg(long, default_value_t = 1e-4)]
    pub lr: f64,
    /// End-of-run LR as a fraction of `--lr`, decayed over the whole run
    /// (`1.0` = no decay).
    #[arg(long, default_value_t = 0.1)]
    pub lr_final: f64,
    /// EMA smoothing window as a fraction of the run; the saved model is the EMA
    /// (`0` = save the raw weights).
    #[arg(long, default_value_t = 0.1)]
    pub ema_frac: f64,
    /// Micro-batches per optimizer step (effective batch = batch×N). `1` = off.
    #[arg(long, default_value_t = 1)]
    pub grad_accum: usize,
    /// Discriminator LR multiplier; `<1.0` tames buzzy static.
    #[arg(long, default_value_t = 1.0)]
    pub d_lr_ratio: f64,
    /// Update the discriminator every N steps.
    #[arg(long, default_value_t = 1)]
    pub d_interval: usize,
    /// Bias clip sampling toward cleaner clips by `snr^alpha` (noise-floor SNR,
    /// not loudness). `0` = uniform.
    #[arg(long, default_value_t = 0.0)]
    pub snr_weight: f32,
    /// Don't keep the best-so-far (lowest mel) weights in
    /// `<out-dir>/checkpoint/<name>.best[.disc].safetensors`; save only the
    /// final ones.
    #[arg(long)]
    pub no_save_best: bool,
    /// Compute backend: `auto`, `cuda` (alias `burn-cuda`), `tch` (`libtorch`,
    /// `burn-tch`) or `wgpu` (`webgpu`, `burn-wgpu`). All three train, and the
    /// saved weights are the same whichever you pick.
    #[arg(long, value_enum, default_value_t = Backend::Auto)]
    pub backend: Backend,
    /// Compute device(s): `auto`, `cpu`, `gpu`, `gpu:N`, `mps`, `vulkan`
    /// (`cuda`/`cuda:N` are accepted spellings of `gpu`). Comma-separate for
    /// data-parallel training across devices — the first is the master.
    #[arg(
        long,
        alias = "devices",
        default_value = "auto",
        value_name = "DEVICE",
        value_delimiter = ',',
        value_parser = cli_kit::parse_device
    )]
    pub device: Vec<burn_kit::DeviceSpec>,
    /// Overwrite weights already at `-o` instead of refusing to start.
    /// `--resume` implies it: continuing a run means writing over its files.
    #[arg(short = 'y', long)]
    pub yes: bool,
    /// Resume from a generator `.safetensors` (+ its `.disc.safetensors`
    /// sidecar). Pass either the EMA output or its `.raw.safetensors` twin —
    /// the raw (non-EMA) live weights are preferred automatically when present,
    /// as they pair faithfully with the saved discriminator. Conflicts with
    /// `--pretrained-g`.
    #[arg(
        long,
        alias = "continue",
        value_name = "SAFETENSORS",
        conflicts_with = "pretrained_g"
    )]
    pub resume: Option<PathBuf>,
    /// Pretrained generator base to warm-start from [default:
    /// `<cache-dir>/pretrained/f0G48k.pth`, downloaded from HF
    /// `lj1995/VoiceConversionWebUI` on first use and reused by every run].
    #[arg(long, conflicts_with = "no_pretrained")]
    pub pretrained_g: Option<PathBuf>,
    /// Pretrained discriminator base to warm-start from [default:
    /// `<cache-dir>/pretrained/f0D48k.pth`, downloaded on first use].
    #[arg(long, conflicts_with = "no_pretrained")]
    pub pretrained_d: Option<PathBuf>,
    /// Train from scratch: no warm-start, and no download of the bases. Poor on
    /// a small corpus, which is why they are fetched by default. `--resume`
    /// fetches nothing either: it continues from weights that already exist.
    #[arg(long)]
    pub no_pretrained: bool,
    /// ContentVec encoder ONNX [default: auto-downloaded]. The trainer's
    /// extractors are ONNX whichever backend trains the generator.
    #[arg(long)]
    pub content: Option<PathBuf>,
    /// RMVPE F0 ONNX [default: auto-downloaded]. As above: ONNX, whatever
    /// `--backend` says.
    #[arg(long)]
    pub rmvpe: Option<PathBuf>,
    #[command(flatten)]
    pub features: FeatureBackendOpts,
    /// Directory the downloaded models are cached in. Shared by every engine
    /// unless `$RVC_CACHE_DIR` (or `$VOICE_CACHE_DIR`) says otherwise.
    #[arg(long, default_value_os_t = hub_kit::cache_dir_for("RVC_CACHE_DIR"))]
    pub cache_dir: PathBuf,
    /// Disable the TUI dashboard and log to stderr (auto-off when not a TTY).
    #[arg(long)]
    pub no_tui: bool,
}

impl TrainArgs {
    /// Reject a training configuration that cannot converge, or cannot start.
    ///
    /// Checked before the corpus is decoded and before a base is downloaded: a
    /// typo in a learning rate should cost a message, not an hour of GPU and a
    /// model full of `NaN`.
    pub fn verify(&self) -> Result<()> {
        anyhow::ensure!(
            matches!(self.model_sr, 40_000 | 48_000),
            "--model-sr must be 40000 or 48000, not {}",
            self.model_sr
        );
        anyhow::ensure!(self.epochs > 0, "--epochs must be at least 1");
        anyhow::ensure!(self.batch_size > 0, "--batch-size must be at least 1");
        anyhow::ensure!(
            self.grad_accum > 0,
            "--grad-accum must be at least 1 (1 = off)"
        );
        anyhow::ensure!(self.d_interval > 0, "--d-interval must be at least 1");
        anyhow::ensure!(self.speaker_id >= 0, "--speaker-id must not be negative");
        anyhow::ensure!(
            self.lr > 0.0 && self.lr.is_finite(),
            "--lr must be a positive, finite number"
        );
        anyhow::ensure!(
            self.d_lr_ratio > 0.0 && self.d_lr_ratio.is_finite(),
            "--d-lr-ratio must be a positive, finite number (1.0 = same LR as the generator)"
        );
        // `1.0` is "no decay" and is the upper bound; at or below zero the LR
        // would reach zero or go negative partway through the run.
        anyhow::ensure!(
            self.lr_final > 0.0 && self.lr_final <= 1.0,
            "--lr-final is a fraction of --lr and must be in (0, 1], not {}: at or \
             below 0 the run would end with no learning rate at all, and above 1 it \
             would end faster than it started",
            self.lr_final
        );
        // `0` disables the EMA and saves the raw weights; `1` would mean a window
        // as long as the run, which never updates.
        anyhow::ensure!(
            (0.0..1.0).contains(&self.ema_frac),
            "--ema-frac is a fraction of the run and must be in [0, 1); 0 saves the \
             raw weights"
        );
        anyhow::ensure!(
            self.snr_weight >= 0.0 && self.snr_weight.is_finite(),
            "--snr-weight must be zero (uniform) or a positive, finite exponent"
        );
        anyhow::ensure!(!self.device.is_empty(), "--device names no device");
        self.feature_backends()?;
        Ok(())
    }

    /// Which runtime extracts the corpus's content and F0 features.
    ///
    /// **`train` is the one command where the two flags do not inherit
    /// `--backend`: their default here is `onnx`.** Not because ONNX Runtime
    /// cannot train — feature extraction is not training, it runs once over the
    /// corpus before the loop starts — but because `rvc-train` builds its own
    /// `FeatureExtractor` from two paths, and that constructor is ONNX-only. If
    /// the extractors followed `--backend`, the recommended training command
    /// would fetch `rmvpe.pt`, hand it to an ORT session builder, and fail.
    ///
    /// Naming a Burn backend explicitly is therefore an error and not a quiet
    /// downgrade: somebody who asked for it should not come away believing the
    /// corpus was analysed on Burn. When the trainer takes a prebuilt
    /// extractor, the default here becomes the resolved `--backend` like
    /// everywhere else and this method goes away.
    pub fn feature_backends(&self) -> Result<FeatureBackends> {
        let features = self.features.resolve(Backend::Onnx);
        for (flag, backend) in [
            ("--content-vec-backend", features.content),
            ("--rmvpe-backend", features.rmvpe),
        ] {
            anyhow::ensure!(
                backend == Backend::Onnx,
                "{flag} {backend} cannot be used with `train`: the trainer's \
                 feature extraction runs on ONNX Runtime only, whichever backend \
                 trains the generator. Drop the flag (or pass `{flag} onnx`) — \
                 `--backend {backend}` still trains on {backend}."
            );
        }
        Ok(features)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    /// The `rvc` binary's own tree, minus `completions` — which belongs to the
    /// executable rather than to the engine, and is not what these test.
    #[derive(Parser)]
    #[command(name = "rvc")]
    struct Cli {
        #[command(flatten)]
        rvc: RvcCli,
    }

    fn parse(argv: &[&str]) -> RvcCommand {
        Cli::try_parse_from(argv)
            .unwrap_or_else(|e| panic!("{argv:?} rejected: {e}"))
            .rvc
            .command
            .expect("expected a subcommand")
    }

    fn convert(extra: &[&str]) -> ConvertArgs {
        let mut argv = vec!["rvc", "convert", "-m", "voice.safetensors", "in.wav"];
        argv.extend_from_slice(extra);
        match parse(&argv) {
            RvcCommand::Convert(a) => a,
            other => panic!("expected `convert`, got {other:?}"),
        }
    }

    fn train(extra: &[&str]) -> TrainArgs {
        let mut argv = vec!["rvc", "train", "clip.wav"];
        argv.extend_from_slice(extra);
        match parse(&argv) {
            RvcCommand::Train(a) => *a,
            other => panic!("expected `train`, got {other:?}"),
        }
    }

    fn download(extra: &[&str]) -> DownloadArgs {
        let mut argv = vec!["rvc", "download"];
        argv.extend_from_slice(extra);
        match parse(&argv) {
            RvcCommand::Download(a) => a,
            other => panic!("expected `download`, got {other:?}"),
        }
    }

    /// The bare invocation — no subcommand, so [`parse`] cannot reach it.
    fn filter(extra: &[&str]) -> FilterArgs {
        let mut argv = vec!["rvc", "-m", "voice.safetensors"];
        argv.extend_from_slice(extra);
        let parsed =
            Cli::try_parse_from(&argv).unwrap_or_else(|e| panic!("{argv:?} rejected: {e}"));
        assert!(parsed.rvc.command.is_none(), "{argv:?} took a subcommand");
        parsed.rvc.filter
    }

    /// Neither flag given: both feature models follow `--backend`, whatever it
    /// resolved to. This is the property the whole design rests on — one flag
    /// still configures the whole pipeline.
    #[test]
    fn both_default_to_the_main_backend() {
        let opts = convert(&[]).features;
        assert_eq!(opts.content_vec_backend, None);
        assert_eq!(opts.rmvpe_backend, None);
        for generator in [Backend::Onnx, Backend::Cuda, Backend::Tch, Backend::Wgpu] {
            let features = opts.resolve(generator);
            assert_eq!(
                features.content, generator,
                "content, --backend {generator}"
            );
            assert_eq!(features.rmvpe, generator, "rmvpe, --backend {generator}");
        }
    }

    /// Each flag overrides only its own model, and neither is substituted for
    /// something else — the contract `Backend::resolve` keeps for `--backend`.
    #[test]
    fn an_override_is_never_substituted_and_never_leaks() {
        let features = convert(&["--content-vec-backend", "tch"])
            .features
            .resolve(Backend::Onnx);
        assert_eq!(features.content, Backend::Tch);
        assert_eq!(features.rmvpe, Backend::Onnx, "rmvpe kept --backend");

        let features = convert(&["--rmvpe-backend", "onnx"])
            .features
            .resolve(Backend::Tch);
        assert_eq!(features.content, Backend::Tch, "content kept --backend");
        assert_eq!(features.rmvpe, Backend::Onnx);
    }

    /// `auto` is a value clap offers on both flags, so it has to mean
    /// something: the same as leaving the flag off. It must not survive as
    /// `Backend::Auto`, which is not a runtime — it matches no loader arm and
    /// names no weight format, so it would silently take the Torch branch of
    /// `weight_format` and then fail to equal `Onnx` anywhere.
    #[test]
    fn auto_named_out_loud_still_means_inherit() {
        let features = convert(&["--content-vec-backend", "auto", "--rmvpe-backend", "auto"])
            .features
            .resolve(Backend::Onnx);
        assert_eq!(features.content, Backend::Onnx);
        assert_eq!(features.rmvpe, Backend::Onnx);
        assert_eq!(weight_format(features.content), hub_kit::WeightFormat::Onnx);
    }

    /// Both flags take `cli_kit::Backend`, so every alias any binary accepts
    /// works here too. Spelled out because the failure mode of a second enum is
    /// exactly this drifting apart.
    #[test]
    fn aliases_parse_on_both_flags() {
        let features = convert(&[
            "--content-vec-backend",
            "libtorch",
            "--rmvpe-backend",
            "burn-cuda",
        ])
        .features
        .resolve(Backend::Onnx);
        assert_eq!(features.content, Backend::Tch);
        assert_eq!(features.rmvpe, Backend::Cuda);
    }

    /// ONNX Runtime reads the `.onnx` artefacts; every Burn compute backend
    /// reads the same PyTorch one. This is what makes the auto-download fetch a
    /// file the chosen runtime can actually open.
    #[test]
    fn weight_format_follows_the_backend() {
        assert_eq!(weight_format(Backend::Onnx), hub_kit::WeightFormat::Onnx);
        for burn in [Backend::Cuda, Backend::Tch, Backend::Wgpu] {
            assert_eq!(weight_format(burn), hub_kit::WeightFormat::Torch, "{burn}");
        }
    }

    /// `download` has no `-m`, so `auto` resolves on hardware alone — but a
    /// named backend still decides the format, which is the whole point of the
    /// command growing a `--backend`.
    #[test]
    fn download_fetches_for_the_backend_it_is_told() {
        let a = download(&["--backend", "onnx"]);
        let features = a.features.resolve(a.backend.resolve(false));
        assert_eq!(weight_format(features.content), hub_kit::WeightFormat::Onnx);
        assert_eq!(weight_format(features.rmvpe), hub_kit::WeightFormat::Onnx);

        let a = download(&["--backend", "tch", "--rmvpe-backend", "onnx"]);
        let features = a.features.resolve(a.backend.resolve(false));
        assert_eq!(
            weight_format(features.content),
            hub_kit::WeightFormat::Torch
        );
        assert_eq!(weight_format(features.rmvpe), hub_kit::WeightFormat::Onnx);
    }

    /// `train` is the documented exception: its extractors are ONNX whatever
    /// trains the generator, so `--backend tch` must not drag them along — that
    /// would break the recommended training command.
    #[test]
    fn train_keeps_onnx_extractors_under_a_burn_backend() {
        let features = train(&["--backend", "tch"])
            .feature_backends()
            .expect("--backend tch must not change the trainer's extractors");
        assert_eq!(features.content, Backend::Onnx);
        assert_eq!(features.rmvpe, Backend::Onnx);
    }

    /// …and asking for one out loud is refused rather than quietly downgraded.
    #[test]
    fn train_refuses_a_burn_feature_backend() {
        let err = train(&["--rmvpe-backend", "tch"])
            .feature_backends()
            .expect_err("a Burn extractor on `train` must be an error")
            .to_string();
        assert!(err.contains("--rmvpe-backend"), "{err}");
        assert!(err.contains("ONNX Runtime only"), "{err}");
    }

    /// Passing none of the geometry flags must reproduce the preset **exactly**,
    /// field for field — this is the property that makes adding them a no-op for
    /// every existing user, and it is cheaper and stricter to assert here than
    /// to diff two converted WAVs.
    #[test]
    fn omitting_the_geometry_flags_reproduces_the_preset() {
        for (opts, preset, which) in [
            (filter(&[]).geometry, StreamParams::realtime(), "filter"),
            (convert(&[]).geometry, StreamParams::batch(), "convert"),
        ] {
            let resolved = opts.resolve(preset);
            assert_eq!(resolved.block, preset.block, "{which} block");
            assert_eq!(resolved.context, preset.context, "{which} context");
            assert_eq!(resolved.crossfade, preset.crossfade, "{which} crossfade");
            assert_eq!(resolved.shape, preset.shape, "{which} shape");
        }
    }

    /// …and so must passing each flag at the value its help text advertises.
    /// This is the one that a truncating cast breaks: `0.05f32 * 16000.0` is
    /// 800.0000119, so `as usize` would land on 799 and quietly move the
    /// default of anyone who spelled it out.
    #[test]
    fn the_documented_defaults_resolve_to_the_preset() {
        let realtime = filter(&[
            "--block-secs",
            "0.5",
            "--context-secs",
            "0.25",
            "--crossfade-secs",
            "0.05",
            "--crossfade-shape",
            "linear",
        ])
        .geometry
        .resolve(StreamParams::realtime());
        let preset = StreamParams::realtime();
        assert_eq!(realtime.block, preset.block);
        assert_eq!(realtime.context, preset.context);
        assert_eq!(realtime.crossfade, preset.crossfade);
        assert_eq!(realtime.shape, preset.shape);

        let batch = convert(&[
            "--block-secs",
            "15",
            "--context-secs",
            "0.5",
            "--crossfade-secs",
            "0.05",
        ])
        .geometry
        .resolve(StreamParams::batch());
        let preset = StreamParams::batch();
        assert_eq!(batch.block, preset.block);
        assert_eq!(batch.context, preset.context);
        assert_eq!(batch.crossfade, preset.crossfade);
    }

    /// One flag named overrides only itself; the rest still come from the
    /// preset. The mixed case is the one worth pinning, because it is where a
    /// "resolve everything or nothing" shortcut would silently reset two values
    /// the user never mentioned.
    #[test]
    fn a_named_length_overrides_only_itself() {
        let resolved = convert(&["--block-secs", "3"])
            .geometry
            .resolve(StreamParams::batch());
        assert_eq!(resolved.block, 3 * ANALYSIS_SR as usize);
        assert_eq!(resolved.context, StreamParams::batch().context);
        assert_eq!(resolved.crossfade, StreamParams::batch().crossfade);
    }

    /// The crossfade is an overlap taken out of the context each block drops,
    /// so a crossfade longer than the context makes every block emit its
    /// context prefix as audio. Refused, naming both numbers — and refused in
    /// the *mixed* case, where only one of them was typed.
    #[test]
    fn a_crossfade_longer_than_the_context_is_refused() {
        let err = filter(&["--crossfade-secs", "0.3"])
            .verify()
            .expect_err("0.3 s of crossfade against 0.25 s of context must be refused")
            .to_string();
        assert!(err.contains("--crossfade-secs"), "{err}");
        assert!(err.contains("--context-secs"), "{err}");
        // The context came from the preset, and the message has to say so or a
        // reader goes looking for a flag they never passed.
        assert!(err.contains("default"), "{err}");
    }

    /// A context longer than a block cannot be filled — `process_block` clamps
    /// it to the block it just processed — so honouring it silently would hand
    /// back less look-back than was asked for.
    #[test]
    fn a_context_longer_than_the_block_is_refused() {
        let err = filter(&["--context-secs", "2"])
            .verify()
            .expect_err("2 s of context against a 0.5 s block must be refused")
            .to_string();
        assert!(err.contains("--context-secs"), "{err}");
        assert!(err.contains("--block-secs"), "{err}");
    }

    /// A block that rounds to zero samples is a hang, not bad audio: `push`
    /// loops while the buffer is at least `block` long and drains nothing.
    #[test]
    fn a_block_that_rounds_to_nothing_is_refused() {
        for secs in ["0", "0.00001"] {
            let err = filter(&["--block-secs", secs])
                .verify()
                .expect_err("a block that rounds to zero samples must be refused")
                .to_string();
            assert!(err.contains("--block-secs"), "{secs}: {err}");
        }
    }

    /// Non-finite and negative seconds are caught before the conversion to
    /// samples, where a saturating cast would turn them into a plausible zero.
    #[test]
    fn a_nonsense_length_is_refused_rather_than_cast() {
        for arg in [
            "--crossfade-secs=nan",
            "--crossfade-secs=-1",
            "--crossfade-secs=inf",
        ] {
            let err = convert(&[arg])
                .verify()
                .expect_err("a non-finite or negative crossfade must be refused")
                .to_string();
            assert!(err.contains("--crossfade-secs"), "{arg}: {err}");
        }
    }

    /// `--f0-threshold` defaults to the constant `rvc-core` hands every
    /// non-ONNX estimator, so the flag's default cannot drift away from the
    /// value the toolkit uses when nobody passes it.
    #[test]
    fn the_voicing_floor_defaults_to_the_toolkits_own() {
        assert_eq!(
            convert(&[]).models.f0_threshold,
            rvc_core::FeatureExtractor::F0_THRESHOLD
        );
        assert_eq!(
            filter(&[]).models.f0_threshold,
            rvc_core::FeatureExtractor::F0_THRESHOLD
        );
    }

    /// It is an RMVPE salience, so it lives in [0, 1]. Outside that the flag
    /// describes nothing, and `NaN` compares false against every bound — which
    /// is the reason the check is a range test rather than two comparisons.
    #[test]
    fn a_voicing_floor_outside_zero_to_one_is_refused() {
        for arg in [
            "--f0-threshold=1.5",
            "--f0-threshold=-0.1",
            "--f0-threshold=nan",
        ] {
            let err = convert(&[arg])
                .verify()
                .expect_err("a salience outside [0, 1] must be refused")
                .to_string();
            assert!(err.contains("--f0-threshold"), "{arg}: {err}");
        }
        convert(&["--f0-threshold", "0.0"])
            .verify()
            .expect("0 is a legal, if extreme, floor");
        convert(&["--f0-threshold", "1.0"])
            .verify()
            .expect("1 is a legal, if extreme, floor");
    }
}
