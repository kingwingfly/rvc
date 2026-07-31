//! `--backend`, defined once so every binary accepts the same spellings.
//!
//! There used to be one of these enums per engine, drifting: one accepted a bare
//! `burn` for CubeCL/CUDA and the others did not, and each spelled the
//! "not compiled in" bail its own way. A user who learns a flag on one binary
//! should not have to re-learn it on the next, so the enum, its aliases, its
//! `auto` rule and that error all live here.
//!
//! What stays with the caller is *what the weights are*: only the engine knows
//! whether an ONNX artefact is a file extension or a directory full of graphs,
//! and this crate deliberately knows nothing about either.

use clap::ValueEnum;

/// Which runtime executes the model.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, ValueEnum)]
pub enum Backend {
    /// Pick by the weights first (an ONNX export runs on ONNX Runtime), then by
    /// what is available: LibTorch on a GPU, else CubeCL/CUDA, else WebGPU,
    /// else LibTorch on CPU.
    #[default]
    Auto,
    /// ONNX Runtime. Inference only — nothing trains on it.
    Onnx,
    /// Native Burn, CubeCL/CUDA compute. NVIDIA only.
    #[value(name = "cuda", alias = "burn", alias = "burn-cuda")]
    Cuda,
    /// Native Burn, LibTorch compute — CUDA, MPS, Vulkan or CPU.
    #[value(name = "tch", alias = "libtorch", alias = "burn-tch")]
    Tch,
    /// Native Burn, WebGPU compute — any Vulkan/Metal/DX12 GPU.
    #[value(name = "wgpu", alias = "webgpu", alias = "burn-wgpu")]
    Wgpu,
}

impl Backend {
    /// Resolve `auto` to something concrete: weights first, hardware second.
    ///
    /// `onnx_weights` is what the caller found on disk. The order matters
    /// because the files decide before preference does — an ONNX export cannot
    /// run on Burn and a checkpoint cannot run on ONNX Runtime. A backend named
    /// explicitly is returned untouched: if it cannot run, loading it says why,
    /// which is more use than a silent substitution.
    ///
    /// Training passes `false`; there is no ONNX training path anywhere.
    pub fn resolve(self, onnx_weights: bool) -> Self {
        match self {
            Self::Auto if onnx_weights => Self::Onnx,
            Self::Auto => match burn_kit::auto_backend() {
                burn_kit::AutoBackend::LibTorch => Self::Tch,
                burn_kit::AutoBackend::Cuda => Self::Cuda,
                burn_kit::AutoBackend::Wgpu => Self::Wgpu,
            },
            explicit => explicit,
        }
    }

    /// The canonical spelling — which is also the cargo feature that compiles
    /// this backend in, so [`Self::unavailable`] can name both at once.
    pub fn name(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Onnx => "onnx",
            Self::Cuda => "cuda",
            Self::Tch => "tch",
            Self::Wgpu => "wgpu",
        }
    }

    /// The error for a backend this build has no code for.
    ///
    /// Reachable only from a `--no-default-features` build, where the match arm
    /// that would have handled it was `#[cfg]`ed away. Every engine reports it
    /// identically because the fix is identical.
    pub fn unavailable(self) -> anyhow::Error {
        anyhow::anyhow!(
            "this binary was built without the {0} backend (rebuild with `--features {0}`)",
            self.name()
        )
    }
}

impl std::fmt::Display for Backend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[derive(Parser)]
    struct Cli {
        #[arg(long, value_enum, default_value_t = Backend::Auto)]
        backend: Backend,
    }

    /// Every spelling any binary ever accepted still parses, on all of them.
    #[test]
    fn aliases() {
        for (spelling, want) in [
            ("auto", Backend::Auto),
            ("onnx", Backend::Onnx),
            ("cuda", Backend::Cuda),
            ("burn", Backend::Cuda),
            ("burn-cuda", Backend::Cuda),
            ("tch", Backend::Tch),
            ("libtorch", Backend::Tch),
            ("burn-tch", Backend::Tch),
            ("wgpu", Backend::Wgpu),
            ("webgpu", Backend::Wgpu),
            ("burn-wgpu", Backend::Wgpu),
        ] {
            let cli = Cli::try_parse_from(["x", "--backend", spelling])
                .unwrap_or_else(|e| panic!("--backend {spelling} rejected: {e}"));
            assert_eq!(cli.backend, want, "--backend {spelling}");
        }
    }

    #[test]
    fn explicit_is_never_substituted() {
        for b in [Backend::Onnx, Backend::Cuda, Backend::Tch, Backend::Wgpu] {
            assert_eq!(b.resolve(true), b);
            assert_eq!(b.resolve(false), b);
        }
        assert_eq!(Backend::Auto.resolve(true), Backend::Onnx);
        assert_ne!(Backend::Auto.resolve(false), Backend::Onnx);
    }
}
