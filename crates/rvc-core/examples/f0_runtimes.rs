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
//! **Read the median, not the mean.** Measured on an RTX 2060 (LibTorch,
//! `--device auto`) over eight clips of 1.0–6.6 s from this repository's own
//! breathy close-mic corpus: median absolute difference **0.005–1.40 Hz** over
//! jointly voiced frames — sub-hertz on seven of the eight — where the *mean*
//! over the same frames ranges **0.49–24.4 Hz** and correlation **0.31–0.9999**.
//!
//! Those two summaries describe the same contours because the disagreement is
//! not spread over the clip, it is a handful of frames: `chuan2_001` has a
//! median of **0.0047 Hz** across 502 voiced frames and a mean of 4.53 Hz, which
//! is one frame in five hundred. The `max |Δ|` line prints both readings and
//! their ratio to identify them — 0.42–0.45 or ≈2.0 on the clips above, an
//! **octave**. A frame whose salience map has two comparable peaks an octave
//! apart is decided by whichever is fractionally higher, so a perturbation far
//! too small to move a confident frame flips an ambiguous one outright.
//!
//! The perturbation is expected rather than nil: the two pad the frame count to
//! a multiple of 32 differently — `Rmvpe::forward` with the zeros upstream used,
//! `f0::OnnxPitchEstimator` by reflection — so the last ≤ 31 frames see
//! different context, and the GRU's reverse half carries that back over the
//! whole clip. Voicing disagreements (7–87 here) are the same effect at the
//! `F0_THRESHOLD` boundary rather than a second one; both runtimes use the same
//! 0.03, so a frame flips only where it was already borderline.
//!
//! **So an earlier figure of "0.11–1.11 Hz, 0–6 voicing disagreements" was a
//! favourable sample, not a tighter port.** It is what clean clips give, and
//! this toolkit exists for the material that does not. What a real defect looks
//! like is the median leaving sub-hertz territory on *every* clip: a swapped GRU
//! gate or a fused bias is not content-dependent and does not spare the
//! confident frames.
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
    let mut deltas: Vec<f32> = Vec::new();
    let mut disagree = 0usize;
    let mut worst = (0.0f32, 0.0f32, 0.0f32);
    let (mut sa, mut sb, mut saa, mut sbb, mut sab) = (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for i in 0..n {
        let (x, y) = (a[i], b[i]);
        // 0 Hz is unvoiced, never a pitch, so it is counted rather than averaged:
        // including it would compare the two on frames where neither claims to
        // have measured anything, and would hide a shifted voicing decision
        // inside a small mean.
        match (x > 0.0, y > 0.0) {
            (true, true) => {
                let d = (x - y).abs();
                deltas.push(d);
                if d > worst.0 {
                    worst = (d, x, y);
                }
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
    let voiced = deltas.len();
    let m = voiced as f64;
    let corr = (m * sab - sa * sb) / ((m * saa - sa * sa) * (m * sbb - sb * sb)).sqrt();

    // **The median is the check; the mean is context.** On a clip whose pitch is
    // unambiguous the two agree to a fraction of a hertz on every frame and the
    // two statistics coincide. On breathy material a handful of frames land near
    // a decision boundary in the salience map, where the padding difference is
    // enough to tip the argmax into a different peak — and one such frame in
    // sixty moves the *mean* by tens of hertz while the rest of the contour is
    // untouched. Reporting only the mean therefore says "the port disagrees by
    // 24 Hz" about a contour that is sub-hertz almost everywhere, which is the
    // metric mistake this repository has already made once, on `s2`'s export.
    // `(worst)` prints both readings so a tipped frame can be told from a
    // genuinely drifting contour: an octave apart is the former.
    let mut sorted = deltas;
    sorted.sort_by(f32::total_cmp);
    let median = sorted.get(voiced / 2).copied().unwrap_or(0.0);
    println!("voiced  : both {voiced} of {n}  (voicing disagreements: {disagree})");
    println!("med  |Δ|: {median:.4} Hz");
    println!(
        "mean |Δ|: {:.4} Hz",
        sorted.iter().map(|&d| d as f64).sum::<f64>() / m
    );
    println!(
        "max  |Δ|: {:.4} Hz  (onnx {:.2} Hz vs burn {:.2} Hz, ratio {:.3})",
        worst.0,
        worst.1,
        worst.2,
        worst.1 / worst.2
    );
    println!("corr    : {corr:.6}");
}
