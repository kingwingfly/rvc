//! The assembled RVC model: ContentVec + RMVPE + trained generator.
//!
//! [`RvcModel::convert_segment`] takes a mono 16 kHz `f32` buffer and returns
//! converted audio at the generator's sample rate. Longer inputs should be fed
//! through the streaming [`crate::convert::Converter`], which slices them into
//! overlapping segments.

use ort::session::Session;
use ort::value::Tensor;

use crate::analysis::{ContentEncoder, PitchEstimator};
use crate::config::ConvertParams;
use crate::config::{CONTENT_DIM, RvcConfig};
use crate::dsp::{f0_to_coarse, shift_pitch, upsample_rows};
use crate::encoder::OnnxContentEncoder;
use crate::error::{Result, VcError};
use crate::f0::OnnxPitchEstimator;
use crate::session::build_session;

/// Loaded RVC pipeline ready to convert audio.
pub struct RvcModel {
    encoder: OnnxContentEncoder,
    f0: OnnxPitchEstimator,
    generator: Session,
    cfg: RvcConfig,
    rng: Xorshift,
}

impl RvcModel {
    /// Load all three ONNX models described by `cfg`.
    pub fn load(cfg: RvcConfig) -> Result<Self> {
        let encoder = OnnxContentEncoder::new(build_session(&cfg.models.content)?);
        let f0 = OnnxPitchEstimator::new(build_session(&cfg.models.rmvpe)?, cfg.f0_threshold);
        let generator = build_session(&cfg.models.generator)?;
        Ok(Self {
            encoder,
            f0,
            generator,
            cfg,
            rng: Xorshift::new(0x9E3779B97F4A7C15),
        })
    }

    /// The generator's output sample rate.
    pub fn output_sr(&self) -> u32 {
        self.cfg.model_sr
    }

    /// Convert a single mono 16 kHz segment, returning audio at `output_sr()`.
    pub fn convert_segment(&mut self, wav16k: &[f32], params: ConvertParams) -> Result<Vec<f32>> {
        // 1. Content features at 50 Hz, upsampled x2 -> 100 Hz to match F0.
        let feats = self.encoder.extract(wav16k)?;
        let feats = upsample_rows(&feats, 2);

        // 2. F0 at 100 Hz, pitch-shifted.
        let mut f0 = self.f0.extract(wav16k)?;
        shift_pitch(&mut f0, params.transpose);

        // 3. Align lengths.
        let n = feats.len().min(f0.len());
        if n == 0 {
            return Ok(Vec::new());
        }
        let coarse = f0_to_coarse(&f0[..n]);
        let pitchf: Vec<f32> = f0[..n].to_vec();

        // 4. Flatten features [n, 768] row-major.
        let mut phone = Vec::with_capacity(n * CONTENT_DIM);
        for row in &feats[..n] {
            phone.extend_from_slice(row);
        }

        self.run_generator(&phone, n, &coarse, &pitchf)
    }

    /// Build the generator inputs and run inference.
    fn run_generator(
        &mut self,
        phone: &[f32],
        n: usize,
        coarse: &[i64],
        pitchf: &[f32],
    ) -> Result<Vec<f32>> {
        let io = &self.cfg.io;

        let phone_t =
            Tensor::from_array((vec![1i64, n as i64, CONTENT_DIM as i64], phone.to_vec()))?;
        let plen_t = Tensor::from_array((vec![1i64], vec![n as i64]))?;
        let pitch_t = Tensor::from_array((vec![1i64, n as i64], coarse.to_vec()))?;
        let pitchf_t = Tensor::from_array((vec![1i64, n as i64], pitchf.to_vec()))?;
        let ds_t = Tensor::from_array((vec![1i64], vec![self.cfg.speaker_id]))?;

        let mut inputs = ort::inputs![
            io.phone.clone() => phone_t,
            io.phone_lengths.clone() => plen_t,
            io.pitch.clone() => pitch_t,
            io.pitchf.clone() => pitchf_t,
            io.ds.clone() => ds_t,
        ];

        if io.needs_rnd {
            let rnd: Vec<f32> = (0..io.rnd_dim * n)
                .map(|_| self.rng.next_gaussian())
                .collect();
            let rnd_t = Tensor::from_array((vec![1i64, io.rnd_dim as i64, n as i64], rnd))?;
            inputs.push((io.rnd.clone().into(), rnd_t.into()));
        }

        let out_name = io.audio_out.clone();
        let outputs = self.generator.run(inputs)?;
        let value = outputs
            .get(out_name.as_str())
            .ok_or_else(|| VcError::MissingOutput(out_name.clone()))?;
        let (_shape, data) = value.try_extract_tensor::<f32>()?;
        Ok(data.to_vec())
    }
}

/// Tiny xorshift RNG with a Box–Muller Gaussian, so the noise input needs no
/// external `rand` dependency. Noise quality here is not critical.
struct Xorshift {
    state: u64,
    spare: Option<f32>,
}

impl Xorshift {
    fn new(seed: u64) -> Self {
        Self {
            state: seed | 1,
            spare: None,
        }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    /// Uniform in (0, 1].
    fn next_f32(&mut self) -> f32 {
        // Top 24 bits -> [0,1); shift to (0,1] to keep ln() finite.
        let v = (self.next_u64() >> 40) as f32 / (1u32 << 24) as f32;
        v + f32::EPSILON
    }

    fn next_gaussian(&mut self) -> f32 {
        if let Some(s) = self.spare.take() {
            return s;
        }
        let u1 = self.next_f32();
        let u2 = self.next_f32();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = std::f32::consts::TAU * u2;
        self.spare = Some(r * theta.sin());
        r * theta.cos()
    }
}
