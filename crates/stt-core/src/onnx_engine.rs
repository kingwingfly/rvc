//! The ONNX Runtime implementation of [`Engine`].
//!
//! Consumes the two-graph export that `optimum` produces and `onnx-community`
//! publishes: `encoder_model.onnx`, and `decoder_model_merged.onnx` which folds
//! the first pass and the cached passes into one graph selected by a
//! `use_cache_branch` flag.
//!
//! The merged decoder's contract, which is most of this file:
//!
//! - Inputs `input_ids`, `encoder_hidden_states`, `use_cache_branch`, and for
//!   every layer four `past_key_values.{i}.{decoder,encoder}.{key,value}`.
//! - Outputs `logits` and the matching `present.{i}…`.
//! - The **decoder** entries grow by one position per step and are fed back.
//!   The **encoder** ones are cross-attention over audio that never changes, so
//!   they are computed on the first pass and then reused verbatim — that is the
//!   entire point of the merged graph, and it is why the first call is the
//!   expensive one.
//! - Every past input must be present even on the first pass, where they are
//!   ignored; zero-length tensors stand in.

use std::path::Path;

use ort::memory::Allocator;
use ort::session::Session;
use ort::value::Tensor;

use crate::engine::Engine;
use crate::error::{Result, SttError};

/// One layer's cached keys and values, flat with their shape.
#[derive(Clone)]
struct Kv {
    shape: Vec<i64>,
    data: Vec<f32>,
}

impl Kv {
    fn value(&self) -> Result<Tensor<f32>> {
        Tensor::from_array((self.shape.clone(), self.data.clone())).map_err(onnx)
    }
}

/// The zero-length placeholder the first pass feeds for every past input.
///
/// `use_cache_branch = false` means the graph ignores these, but ONNX still
/// requires the inputs to exist and to have the right rank. It has to be
/// [`Tensor::new`] rather than `from_array`: the latter validates raw data and
/// rejects any zero dimension, which is exactly the shape wanted here.
fn empty_kv(allocator: &Allocator, heads: usize, head_dim: usize) -> Result<Tensor<f32>> {
    Tensor::new(allocator, [1, heads, 0, head_dim]).map_err(onnx)
}

pub struct OnnxEngine {
    encoder: Session,
    decoder: Session,
    layers: usize,
    heads: usize,
    head_dim: usize,
    mel_bins: usize,
    /// Encoder output for the current window, kept host-side because the
    /// decoder takes it as an input on every step.
    audio: Option<(Vec<i64>, Vec<f32>)>,
    /// Cross-attention keys/values, computed once per window.
    cross: Vec<(Kv, Kv)>,
    /// Self-attention keys/values, one position longer each step.
    self_kv: Vec<(Kv, Kv)>,
}

impl OnnxEngine {
    /// Load from a directory holding the `optimum` export.
    ///
    /// Accepts the graphs either directly in `dir` or under `dir/onnx`, which is
    /// where the `onnx-community` repos put them.
    pub fn load(
        dir: &Path,
        mel_bins: usize,
        layers: usize,
        heads: usize,
        d_model: usize,
    ) -> Result<Self> {
        let find = |name: &str| -> Result<std::path::PathBuf> {
            for candidate in [dir.join("onnx").join(name), dir.join(name)] {
                if candidate.exists() {
                    return Ok(candidate);
                }
            }
            Err(SttError::Weights(format!(
                "{name} not found in {} or {}/onnx — an ONNX Whisper export has \
                 `encoder_model.onnx` and `decoder_model_merged.onnx`",
                dir.display(),
                dir.display()
            )))
        };

        Ok(Self {
            encoder: build_session(&find("encoder_model.onnx")?)?,
            decoder: build_session(&find("decoder_model_merged.onnx")?)?,
            layers,
            heads,
            head_dim: d_model / heads,
            mel_bins,
            audio: None,
            cross: Vec::new(),
            self_kv: Vec::new(),
        })
    }
}

impl Engine for OnnxEngine {
    fn mel_bins(&self) -> usize {
        self.mel_bins
    }

    fn encode(&mut self, mel: &[f32], frames: usize) -> Result<()> {
        let input = Tensor::from_array((
            vec![1i64, self.mel_bins as i64, frames as i64],
            mel.to_vec(),
        ))
        .map_err(onnx)?;

        let audio = {
            let outputs = self
                .encoder
                .run(ort::inputs!["input_features" => input])
                .map_err(onnx)?;
            let (shape, data) = outputs["last_hidden_state"]
                .try_extract_tensor::<f32>()
                .map_err(onnx)?;
            (shape.to_vec(), data.to_vec())
        };

        self.audio = Some(audio);
        self.restart();
        Ok(())
    }

    fn step(&mut self, tokens: &[u32]) -> Result<Vec<f32>> {
        let (audio_shape, audio_data) = self
            .audio
            .as_ref()
            .ok_or_else(|| SttError::Config("step before encode".into()))?;

        // The first call after `restart` has no cache, so it takes the no-cache
        // branch and produces the cross-attention entries the rest reuse.
        let first = self.self_kv.is_empty();

        let ids: Vec<i64> = tokens.iter().map(|&t| t as i64).collect();
        let mut inputs = ort::inputs![
            "input_ids" => Tensor::from_array((vec![1i64, ids.len() as i64], ids)).map_err(onnx)?,
            "encoder_hidden_states" =>
                Tensor::from_array((audio_shape.clone(), audio_data.clone())).map_err(onnx)?,
            "use_cache_branch" => Tensor::from_array((vec![1i64], vec![!first])).map_err(onnx)?,
        ];
        let allocator = self.decoder.allocator();
        for i in 0..self.layers {
            let mut feed = |slot: &str, kv: Option<&Kv>| -> Result<()> {
                let value = match kv {
                    Some(kv) => kv.value()?,
                    None => empty_kv(allocator, self.heads, self.head_dim)?,
                };
                inputs.push((format!("past_key_values.{i}.{slot}").into(), value.into()));
                Ok(())
            };
            let (self_k, self_v) = split(self.self_kv.get(i));
            let (cross_k, cross_v) = split(self.cross.get(i));
            feed("decoder.key", self_k)?;
            feed("decoder.value", self_v)?;
            feed("encoder.key", cross_k)?;
            feed("encoder.value", cross_v)?;
        }

        let outputs = self.decoder.run(inputs).map_err(onnx)?;

        let take = |name: String| -> Result<Kv> {
            let (shape, data) = outputs[name.as_str()]
                .try_extract_tensor::<f32>()
                .map_err(onnx)?;
            Ok(Kv {
                shape: shape.to_vec(),
                data: data.to_vec(),
            })
        };
        let mut self_kv = Vec::with_capacity(self.layers);
        for i in 0..self.layers {
            self_kv.push((
                take(format!("present.{i}.decoder.key"))?,
                take(format!("present.{i}.decoder.value"))?,
            ));
        }
        // Only the no-cache branch computes these; afterwards the graph passes
        // the inputs straight through, so recapturing them would be a no-op at
        // best and is skipped.
        if first {
            let mut cross = Vec::with_capacity(self.layers);
            for i in 0..self.layers {
                cross.push((
                    take(format!("present.{i}.encoder.key"))?,
                    take(format!("present.{i}.encoder.value"))?,
                ));
            }
            self.cross = cross;
        }
        self.self_kv = self_kv;

        let (shape, data) = outputs["logits"]
            .try_extract_tensor::<f32>()
            .map_err(onnx)?;
        let vocab = shape[2] as usize;
        Ok(data[data.len() - vocab..].to_vec())
    }

    fn restart(&mut self) {
        // Both caches go: without the decoder cache the graph must take the
        // no-cache branch, and that branch recomputes the cross-attention
        // entries anyway.
        self.self_kv.clear();
        self.cross.clear();
    }
}

/// Build a session with CUDA-then-CPU execution providers.
///
/// `rvc-core` has the same six lines. They are not shared because engines do not
/// depend on each other; if a third one needs it, that is the moment to lift it
/// into a kit crate rather than now.
fn build_session(path: &Path) -> Result<Session> {
    use ort::execution_providers::{CPUExecutionProvider, CUDAExecutionProvider};

    Session::builder()
        .map_err(onnx)?
        .with_execution_providers([
            CUDAExecutionProvider::default().build(),
            CPUExecutionProvider::default().build(),
        ])
        .map_err(|e| SttError::Weights(e.to_string()))?
        .commit_from_file(path)
        .map_err(onnx)
}

/// Borrow a layer's key and value halves, present or not.
fn split(kv: Option<&(Kv, Kv)>) -> (Option<&Kv>, Option<&Kv>) {
    match kv {
        Some((k, v)) => (Some(k), Some(v)),
        None => (None, None),
    }
}

fn onnx(e: ort::Error) -> SttError {
    SttError::Weights(format!("onnxruntime: {e}"))
}
