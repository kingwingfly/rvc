//! Compare the ONNX and Burn pitch estimators on real audio.
//!
//! Weight coverage says the module tree matches the checkpoint and nothing about
//! whether the forward pass computes the right thing — this repository has
//! already shipped a port that loaded at 100% and produced garbage. This is
//! RMVPE's arithmetic check, and it is a strong one because the two runtimes
//! share everything outside the network: the same `mel::RmvpeMel` front end, the
//! same `dsp::rmvpe_decode`, the same voicing threshold. A disagreement is
//! therefore the network's.
//!
//! Both estimators are driven **through `FeatureExtractor`**, not called
//! directly, so what is measured is the pair a real conversion would use.
//!
//! Measured on an RTX 2060 (LibTorch, `--device auto`) over four clips of
//! 1.5–2.4 s: mean absolute difference **0.11–1.11 Hz** over jointly voiced
//! frames, correlation **0.9953–0.99997**, and 0–6 frames of 154–240
//! disagreeing about voicing. The residual is expected rather than nil: the two pad the
//! frame count to a multiple of 32 differently — `Rmvpe::forward` with the zeros
//! upstream used, `f0::OnnxPitchEstimator` by reflection — so the last ≤ 31
//! frames see different context, and the GRU carries that back over the whole
//! clip.
//!
//! ```sh
//! cargo run -p rvc-core --features tch --example f0_runtimes -- \
//!     rmvpe.onnx rmvpe.pt vec-768-layer-12.onnx clip.wav
//! ```
//!
//! Expect the process to abort on exit on this maintainer's RTX 2060 — that is
//! the recorded ORT-session-drop defect, and it happens after every number has
//! been printed.

use std::path::Path;

use futures::StreamExt;
use rvc_core::FeatureExtractor;

#[tokio::main]
async fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [rmvpe_onnx, rmvpe_pt, content_onnx, clip] = &args[..] else {
        eprintln!("usage: f0_runtimes <rmvpe.onnx> <rmvpe.pt> <contentvec.onnx> <clip>");
        std::process::exit(2);
    };

    let mut stream = Box::pin(audio_kit::decode_path(
        Path::new(clip),
        audio_kit::DecodeOptions::new(FeatureExtractor::ANALYSIS_SR),
    ));
    let mut wav = Vec::new();
    while let Some(chunk) = stream.next().await {
        wav.extend_from_slice(&chunk.expect("decoding the clip"));
    }
    println!(
        "clip    : {clip}  {} samples ({:.2} s)",
        wav.len(),
        wav.len() as f32 / FeatureExtractor::ANALYSIS_SR as f32
    );

    // The content encoder is ONNX on both sides: this example is about RMVPE,
    // and holding the other half fixed is what makes the difference attributable.
    let mut onnx = FeatureExtractor::from_parts(
        rvc_core::onnx_content_encoder(Path::new(content_onnx)).expect("contentvec.onnx"),
        rvc_core::onnx_pitch_estimator(Path::new(rmvpe_onnx)).expect("rmvpe.onnx"),
    );
    let mut burn = FeatureExtractor::from_parts(
        rvc_core::onnx_content_encoder(Path::new(content_onnx)).expect("contentvec.onnx"),
        rvc_core::libtorch_pitch_estimator(
            Path::new(rmvpe_pt),
            FeatureExtractor::F0_THRESHOLD,
            burn_kit::DeviceSpec::Auto,
        )
        .expect("rmvpe.pt"),
    );

    let a = onnx.extract(&wav).expect("onnx extraction").f0;
    let b = burn.extract(&wav).expect("burn extraction").f0;
    println!("frames  : onnx {}  burn {}", a.len(), b.len());

    let n = a.len().min(b.len());
    let mut voiced = 0usize;
    let mut disagree = 0usize;
    let mut sum_abs = 0.0f64;
    let mut max_abs = 0.0f32;
    let (mut sa, mut sb, mut saa, mut sbb, mut sab) = (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for i in 0..n {
        let (x, y) = (a[i], b[i]);
        // 0 Hz is unvoiced, never a pitch, so it is counted rather than averaged:
        // including it would compare the two on frames where neither claims to
        // have measured anything, and would hide a shifted voicing decision
        // inside a small mean.
        match (x > 0.0, y > 0.0) {
            (true, true) => {
                voiced += 1;
                let d = (x - y).abs();
                sum_abs += d as f64;
                max_abs = max_abs.max(d);
                sa += x as f64;
                sb += y as f64;
                saa += (x * x) as f64;
                sbb += (y * y) as f64;
                sab += (x * y) as f64;
            }
            (p, q) if p != q => disagree += 1,
            _ => {}
        }
    }
    let m = voiced as f64;
    let corr = (m * sab - sa * sb) / ((m * saa - sa * sa) * (m * sbb - sb * sb)).sqrt();
    println!("voiced  : both {voiced} of {n}  (voicing disagreements: {disagree})");
    println!("mean |Δ|: {:.4} Hz", sum_abs / m);
    println!("max  |Δ|: {max_abs:.4} Hz");
    println!("corr    : {corr:.6}");
}
