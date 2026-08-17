//! Check `audio_kit::Denoiser` against ffmpeg's `anlmdn` on the same audio.
//!
//! Unit tests pin this port's *properties* — length, chunk-invariance, energy
//! preservation. None of them would notice a swapped gate order or a
//! mis-scaled weight, because a wrong non-local means still preserves length
//! and still passes a clean tone through. This is the check that would: run the
//! same samples through the filter this was ported from and subtract.
//!
//! It takes the reference as a file rather than shelling out, because the
//! ffmpeg that can produce one is not the ffmpeg you have — `anlmdn` corrupts
//! the heap from 9.0 onward (see [`audio_kit::Denoiser`]). On Arch the last
//! working build is still in the package cache:
//!
//! ```sh
//! mkdir ff8 && tar --zstd -xf /var/cache/pacman/pkg/ffmpeg-2:8.1.2-11-x86_64.pkg.tar.zst \
//!     -C ff8 usr/bin/ffmpeg usr/lib
//! ```
//!
//! Then, at a length that is an exact multiple of the filter's 193-sample hop
//! so the reference never reaches the partial-frame bug that motivated this
//! port in the first place:
//!
//! ```sh
//! ffmpeg -f lavfi -i "sine=f=220:d=2:r=48000" -af atrim=end_sample=96500 \
//!     -f f32le -ac 1 -ar 48000 in.f32
//! LD_LIBRARY_PATH=ff8/usr/lib ff8/usr/bin/ffmpeg -f f32le -ar 48000 -ac 1 -i in.f32 \
//!     -af anlmdn=s=0.008:p=0.002:r=0.006 -f f32le ref.f32
//! cargo run -p audio-kit --example nlm_parity -- in.f32 ref.f32
//! ```
//!
//! Agreement to ~1e-7 is what a faithful port looks like; the two differ only
//! by float summation order. Anything at 1e-3 or worse is a real divergence.

use audio_kit::{DenoiseParams, Denoiser};

fn read_f32le(path: &str) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("reading {path}: {e}"));
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [input, reference] = args.as_slice() else {
        eprintln!("usage: nlm_parity <input.f32> <reference.f32>  (raw f32le, mono, 48 kHz)");
        std::process::exit(2);
    };

    let input = read_f32le(input);
    let reference = read_f32le(reference);

    let mut d = Denoiser::new(48_000, DenoiseParams::default());
    let mut ours = d.process(&input);
    ours.extend(d.flush());

    println!(
        "input {} | reference {} | ours {}",
        input.len(),
        reference.len(),
        ours.len()
    );

    // Compare only the region both produced. On a working ffmpeg the two run
    // to the same length, partial final frame included — the overflow that
    // motivated this port writes past the frame rather than shortening it.
    let n = ours.len().min(reference.len());
    assert!(n > 0, "nothing to compare");

    let (mut max_abs, mut sum_sq, mut ref_sq, mut at) = (0.0f32, 0.0f64, 0.0f64, 0usize);
    for i in 0..n {
        let d = (ours[i] - reference[i]).abs();
        if d > max_abs {
            max_abs = d;
            at = i;
        }
        sum_sq += (d as f64) * (d as f64);
        ref_sq += (reference[i] as f64) * (reference[i] as f64);
    }
    let rms_rel = (sum_sq / ref_sq.max(f64::MIN_POSITIVE)).sqrt();

    println!("compared {n} samples");
    println!("  max |diff|   {max_abs:.3e}  (at sample {at})");
    println!("  rms diff/rms {rms_rel:.3e}");
    println!(
        "  verdict      {}",
        if rms_rel < 1e-5 {
            "match (float summation order only)"
        } else {
            "DIVERGENT — the port computes something else"
        }
    );
}
