//! Gated probe test: run the ContentVec + RMVPE ONNX models on real audio to
//! validate the assumed tensor I/O (shapes, axis layout) against actual model
//! files, without needing a trained generator.
//!
//! Run with (all paths required via env):
//! ```sh
//! ORT_DYLIB_PATH=/usr/lib/libonnxruntime.so \
//! RVC_TEST_CONTENT=.../vec-768-layer-12.onnx \
//! RVC_TEST_RMVPE=.../rmvpe.onnx \
//! RVC_TEST_WAV=/tmp/piner16k.f32 \
//!   cargo test -p rvc-core probe_shared_models -- --ignored --nocapture
//! ```

use std::io::Read;

use crate::encoder::ContentEncoder;
use crate::f0::F0Estimator;
use crate::session::build_session;

fn read_f32le(path: &str) -> Vec<f32> {
    let mut bytes = Vec::new();
    std::fs::File::open(path)
        .unwrap()
        .read_to_end(&mut bytes)
        .unwrap();
    bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

#[test]
#[ignore = "requires ORT_DYLIB_PATH + model/audio paths via env"]
fn probe_shared_models() {
    let content = std::env::var("RVC_TEST_CONTENT").expect("RVC_TEST_CONTENT");
    let rmvpe = std::env::var("RVC_TEST_RMVPE").expect("RVC_TEST_RMVPE");
    let wav = std::env::var("RVC_TEST_WAV").expect("RVC_TEST_WAV");

    let audio = read_f32le(&wav);
    eprintln!("audio samples: {}", audio.len());

    let mut enc = ContentEncoder::new(build_session(&content).expect("load contentvec"));
    let feats = enc.extract(&audio).expect("contentvec extract");
    eprintln!(
        "contentvec frames: {}  dim: {}",
        feats.len(),
        feats[0].len()
    );

    // Diagnostics: what does the RMVPE model actually expect?
    let sess = build_session(&rmvpe).expect("load rmvpe");
    for i in sess.inputs() {
        eprintln!("rmvpe input `{}`: {:?}", i.name(), i.dtype());
    }
    for o in sess.outputs() {
        eprintln!("rmvpe output `{}`: {:?}", o.name(), o.dtype());
    }

    let mut f0 = F0Estimator::new(sess, 0.03);
    let pitch = f0.extract(&audio).expect("rmvpe extract");
    let voiced = pitch.iter().filter(|&&x| x > 0.0).count();
    eprintln!("rmvpe frames: {}  voiced: {}", pitch.len(), voiced);

    assert!(!feats.is_empty());
    assert_eq!(feats[0].len(), 768);
    assert!(!pitch.is_empty());
}
