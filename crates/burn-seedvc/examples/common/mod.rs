//! Backend selection shared by the examples.
//!
//! The examples are this crate's stand-in for unit tests — there are none for
//! the network, so correctness is checked by loading real pretrained weights and
//! reporting coverage. Re-running that on each compute backend is what makes it
//! a smoke test for the *backend* rather than only for the port.
//!
//! `src/` stays backend-free: `cuda` and `tch` are off-by-default features that
//! exist only for these examples.

use burn::tensor::backend::Backend as BurnBackend;

/// Which Burn backend an example should instantiate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Backend {
    /// Pure-CPU reference. Always compiled; needs no GPU and no LibTorch.
    #[default]
    NdArray,
    /// CubeCL/CUDA. Requires `--features cuda` and an NVIDIA GPU.
    Cuda,
    /// LibTorch on **CPU**. Requires `--features tch` and `LIBTORCH`.
    ///
    /// Deliberately not CUDA: weight loading is host-side, so the coverage
    /// numbers come out identical either way, while `LibTorchDevice::Cuda` is a
    /// hard panic on a CPU-only LibTorch (`burn-tch` bakes that check in at build
    /// time). Proving the model *constructs and loads* under LibTorch is the
    /// point here; `rvc convert --backend tch` exercises its GPU path.
    Tch,
}

impl std::fmt::Display for Backend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::NdArray => "ndarray",
            Self::Cuda => "cuda",
            Self::Tch => "tch (cpu)",
        })
    }
}

/// Work that can run on any backend.
///
/// A trait with a generic method rather than a closure: each [`run_on`] arm
/// names a *different* concrete backend type, which no single closure type can
/// satisfy.
pub trait Job {
    fn run<B: BurnBackend>(self, device: &B::Device);
}

/// Run `job` on `backend`, exiting with a message if it wasn't compiled in.
pub fn run_on<J: Job>(backend: Backend, job: J) {
    match backend {
        Backend::NdArray => job.run::<burn_ndarray::NdArray>(&Default::default()),
        Backend::Cuda => {
            #[cfg(feature = "cuda")]
            {
                job.run::<burn::backend::cuda::Cuda>(&Default::default())
            }
            #[cfg(not(feature = "cuda"))]
            {
                let _ = job;
                not_compiled_in("cuda")
            }
        }
        Backend::Tch => {
            #[cfg(feature = "tch")]
            {
                // See `Backend::Tch`: CPU on purpose.
                job.run::<burn::backend::libtorch::LibTorch<f32>>(
                    &burn::backend::libtorch::LibTorchDevice::Cpu,
                )
            }
            #[cfg(not(feature = "tch"))]
            {
                let _ = job;
                not_compiled_in("tch")
            }
        }
    }
}

/// Pull `--backend <x>` (or `--backend=<x>`) out of the arguments, returning it
/// with the remaining positional ones.
///
/// Hand-rolled because this crate has no app dependencies and should not gain
/// `clap` for two examples.
pub fn parse_args() -> (Backend, Vec<String>) {
    let mut backend = Backend::default();
    let mut rest = Vec::new();
    let mut args = std::env::args().skip(1);

    while let Some(a) = args.next() {
        if let Some(v) = a.strip_prefix("--backend=") {
            backend = parse_backend(v);
        } else if a == "--backend" {
            match args.next() {
                Some(v) => backend = parse_backend(&v),
                None => fail("--backend needs a value"),
            }
        } else {
            rest.push(a);
        }
    }
    (backend, rest)
}

fn parse_backend(s: &str) -> Backend {
    match s.trim().to_ascii_lowercase().as_str() {
        "ndarray" | "cpu" => Backend::NdArray,
        "cuda" => Backend::Cuda,
        "tch" | "libtorch" => Backend::Tch,
        other => fail(&format!(
            "unknown backend `{other}` (expected ndarray, cuda or tch)"
        )),
    }
}

fn fail(msg: &str) -> ! {
    eprintln!("error: {msg}");
    std::process::exit(2);
}

/// A backend asked for but not compiled in is a build-time choice, not a bug —
/// so it gets a sentence naming the flag rather than a panic.
#[allow(dead_code)] // unused when every backend feature is on
fn not_compiled_in(backend: &str) -> ! {
    eprintln!(
        "error: this example was built without the `{backend}` backend.\n\
         Rebuild with:  cargo run -p burn-seedvc --example <name> --features {backend} -- ..."
    );
    if backend == "tch" {
        eprintln!("note: the `tch` feature also needs LIBTORCH set — see the README.");
    }
    std::process::exit(2);
}
