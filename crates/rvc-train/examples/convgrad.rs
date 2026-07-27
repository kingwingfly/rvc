//! Probe: which `conv1d` autodiff backward breaks on the LibTorch backend?
//!
//! Written to pin down why `rvc train --backend tch` dies in
//! `burn_autodiff::…::conv1d::Conv1DWithBias::backward` with
//! "size of tensor a (41) must match the size of tensor b (42) at dimension 2".
//!
//! 41 is `DiscriminatorS`'s kernel size, and dim 2 of a conv *weight* is the
//! kernel — so the suspect is the **weight** gradient, not the input gradient,
//! on a strided (and grouped) convolution whose input length leaves a remainder.
//!
//! Both gradients are checked, because a backend that returns a wrongly-shaped
//! weight gradient without erroring is worse than one that errors.
//!
//! ```sh
//! cargo run -p rvc-train --example convgrad --features tch,cuda
//! ```
//! (needs `LD_LIBRARY_PATH=$LIBTORCH/lib`: the RUNPATH is only baked into the
//! `rvc` binary, not into other crates' examples)

use burn::nn::PaddingConfig1d;
use burn::nn::conv::{Conv1d, Conv1dConfig};
use burn::tensor::Tensor;
use burn::tensor::backend::AutodiffBackend;

/// `(in_ch, out_ch, length, kernel, stride, padding, groups)`.
/// The last four are `DiscriminatorS`'s strided layers at the real training
/// segment length (36 frames × 480 hop = 17280 samples).
const CASES: &[(usize, usize, usize, usize, usize, usize, usize)] = &[
    (1, 1, 100, 4, 2, 1, 1),
    (1, 1, 101, 4, 2, 1, 1),
    (1, 16, 17280, 15, 1, 7, 1),
    // Isolate the trigger: groups alone, stride alone, then both.
    (16, 64, 100, 5, 1, 2, 4), // grouped, stride 1
    (16, 64, 100, 4, 2, 1, 4), // grouped, stride 2, divides exactly
    (16, 64, 101, 4, 2, 1, 4), // grouped, stride 2, does not divide
    (16, 64, 100, 4, 2, 1, 1), // ungrouped control of the same shape
    (16, 64, 17280, 41, 4, 20, 4),
    (64, 256, 4320, 41, 4, 20, 16),
    (256, 1024, 1080, 41, 4, 20, 64),
    (1024, 1024, 270, 41, 4, 20, 256),
];

fn probe<AB: AutodiffBackend>(device: &AB::Device) {
    for &(cin, cout, len, k, stride, pad, groups) in CASES {
        let exact = (len + 2 * pad).saturating_sub(k) % stride == 0;
        let conv: Conv1d<AB> = Conv1dConfig::new(cin, cout, k)
            .with_stride(stride)
            .with_padding(PaddingConfig1d::Explicit(pad, pad))
            .with_groups(groups)
            .with_bias(true)
            .init(device);

        let x = Tensor::<AB, 3>::ones([1, cin, len], device).require_grad();

        let got = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let y = conv.forward(x.clone());
            let out = y.dims()[2];
            let grads = y.sum().backward();
            let xg = x.grad(&grads).map(|g| g.dims()[2]);
            let wg = conv.weight.val().to_data().shape[2];
            let wgrad = conv.weight.grad(&grads).map(|g| g.dims()[2]);
            (out, xg, wg, wgrad)
        }));

        match got {
            Err(_) => println!(
                "cin={cin:<5} L={len:<6} k={k:<3} s={stride} g={groups:<4} \
                 {:<18} -> PANIC in backward",
                if exact {
                    "(divides)"
                } else {
                    "(does NOT divide)"
                }
            ),
            Ok((out, xg, wshape, wgrad)) => {
                let xv = match xg {
                    Some(n) if n == len => "x ok".to_string(),
                    Some(n) => format!("x WRONG {n}!={len}"),
                    None => "x none".to_string(),
                };
                let wv = match wgrad {
                    Some(n) if n == wshape => "w ok".to_string(),
                    Some(n) => format!("w WRONG {n}!={wshape}"),
                    None => "w none".to_string(),
                };
                println!(
                    "cin={cin:<5} L={len:<6} k={k:<3} s={stride} g={groups:<4} \
                     {:<18} out={out:<5} -> {xv}, {wv}",
                    if exact {
                        "(divides)"
                    } else {
                        "(does NOT divide)"
                    }
                );
            }
        }
    }
}

fn main() {
    // The probe expects panics; keep the 60-line torch backtrace out of the way.
    std::panic::set_hook(Box::new(|_| {}));

    #[cfg(feature = "tch")]
    {
        use burn::backend::{Autodiff, libtorch::LibTorch};
        println!("== libtorch (cpu) ==");
        probe::<Autodiff<LibTorch<f32>>>(&burn::backend::libtorch::LibTorchDevice::Cpu);
    }
    #[cfg(feature = "cuda")]
    {
        use burn::backend::{Autodiff, cuda::Cuda};
        println!("\n== cubecl/cuda ==");
        probe::<Autodiff<Cuda>>(&Default::default());
    }
    #[cfg(feature = "wgpu")]
    {
        use burn::backend::{Autodiff, wgpu::Wgpu};
        println!("\n== wgpu ==");
        probe::<Autodiff<Wgpu>>(&Default::default());
    }
    #[cfg(not(any(feature = "tch", feature = "cuda", feature = "wgpu")))]
    eprintln!("build with --features tch, cuda and/or wgpu");
}
