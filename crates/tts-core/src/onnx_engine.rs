//! The ONNX Runtime implementation of [`Engine`].
//!
//! Consumes the four graphs `export/export_gptsovits.py` writes — `reference`,
//! `s1_prompt`, `s1_step` and `s2` — whose contracts are documented in
//! `export/README.md`. Everything above this file is shared with the Burn path,
//! including sampling, the repetition penalty and the stop rule, so the two
//! runtimes differ only in the arithmetic.
//!
//! `s1` is split into a prompt graph and a step graph, which is the same shape
//! upstream uses and the same shape the Whisper export `stt-core` reads has
//! (there as one merged graph selected by a flag). The prompt pass returns the
//! whole key/value cache; each step is handed it back and returns it one
//! position longer.

use std::path::{Path, PathBuf};

use ort::session::Session;
use ort::value::Tensor;

use crate::engine::{Engine, PRIOR_CHANNELS};
use crate::error::{Result, TtsError};

/// One tensor on its way back into a graph: flat data with its shape.
type Cached = (Vec<i64>, Vec<f32>);

pub struct OnnxEngine {
    reference: Session,
    s1_prompt: Session,
    s1_step: Session,
    s2: Session,
    /// `2 * n_layer` entries, key and value interleaved as the graph names them.
    cache: Vec<Cached>,
    layers: usize,
}

impl OnnxEngine {
    /// The directory holding an export, if it has one.
    ///
    /// Both layouts the exporter's own README suggests: the graphs directly in
    /// `dir`, or under `dir/onnx`. Used by `auto` backend selection, so it must
    /// answer without loading anything.
    pub fn find(dir: &Path) -> Option<PathBuf> {
        [dir.join("onnx"), dir.to_path_buf()]
            .into_iter()
            .find(|d| GRAPHS.iter().all(|name| d.join(name).exists()))
    }

    /// Load all four graphs.
    pub fn load(dir: &Path) -> Result<Self> {
        let dir = Self::find(dir).ok_or_else(|| {
            TtsError::Weights(format!(
                "no GPT-SoVITS ONNX export under {} or {}/onnx — expected {} \
                 (write them with `export/export_gptsovits.py`)",
                dir.display(),
                dir.display(),
                GRAPHS.join(", ")
            ))
        })?;

        let s1_prompt = session(&dir.join("s1_prompt.onnx"))?;
        // Read the depth from the graph rather than from a config: an export of
        // a different `s1` would still load, and a hardcoded 24 would quietly
        // drop half its cache.
        let layers = s1_prompt
            .outputs()
            .iter()
            .filter(|o| o.name().starts_with("present."))
            .count()
            / 2;
        if layers == 0 {
            return Err(TtsError::Weights(
                "s1_prompt.onnx has no `present.*` outputs — it is not a graph \
                 this can drive"
                    .into(),
            ));
        }

        Ok(Self {
            reference: session(&dir.join("reference.onnx"))?,
            s1_prompt,
            s1_step: session(&dir.join("s1_step.onnx"))?,
            s2: session(&dir.join("s2.onnx"))?,
            cache: Vec::new(),
            layers,
        })
    }
}

/// Take `logits` and the whole `present.*` cache out of an `s1` run.
///
/// A free function rather than a method because `SessionOutputs` holds the
/// session borrowed, and the cache has to be *owned* before it can be stored
/// back on the engine that owns the session.
fn absorb(
    outputs: &ort::session::SessionOutputs<'_>,
    layers: usize,
) -> Result<(Vec<f32>, Vec<Cached>)> {
    let mut cache = Vec::with_capacity(2 * layers);
    for i in 0..layers {
        for slot in ["key", "value"] {
            let (shape, data) = outputs[format!("present.{i}.{slot}").as_str()]
                .try_extract_tensor::<f32>()
                .map_err(onnx)?;
            cache.push((shape.to_vec(), data.to_vec()));
        }
    }
    let (_, logits) = outputs["logits"]
        .try_extract_tensor::<f32>()
        .map_err(onnx)?;
    Ok((logits.to_vec(), cache))
}

impl Engine for OnnxEngine {
    fn analyse(&mut self, audio: &[f32]) -> Result<(Vec<u32>, Vec<f32>)> {
        let input =
            Tensor::from_array((vec![1i64, audio.len() as i64], audio.to_vec())).map_err(onnx)?;
        let outputs = self
            .reference
            .run(ort::inputs!["audio" => input])
            .map_err(onnx)?;

        let (_, codes) = outputs["codes"].try_extract_tensor::<i64>().map_err(onnx)?;
        let (_, speaker) = outputs["speaker"]
            .try_extract_tensor::<f32>()
            .map_err(onnx)?;
        Ok((codes.iter().map(|&c| c as u32).collect(), speaker.to_vec()))
    }

    fn s1_prompt(
        &mut self,
        phones: &[u32],
        bert: &[f32],
        hidden: usize,
        prompt: &[u32],
    ) -> Result<Vec<f32>> {
        let (logits, cache) = {
            let outputs = self
                .s1_prompt
                .run(ort::inputs![
                    "phones" => ids(phones)?,
                    "bert" => Tensor::from_array((
                        vec![1i64, phones.len() as i64, hidden as i64],
                        bert.to_vec(),
                    )).map_err(onnx)?,
                    "prompt" => ids(prompt)?,
                ])
                .map_err(onnx)?;
            absorb(&outputs, self.layers)?
        };
        self.cache = cache;
        Ok(logits)
    }

    fn s1_step(&mut self, token: u32, position: usize) -> Result<Vec<f32>> {
        if self.cache.is_empty() {
            return Err(TtsError::Weights("s1 step before prompt".into()));
        }
        let mut inputs = ort::inputs![
            "token" => ids(&[token])?,
            "position" => Tensor::from_array((vec![1i64], vec![position as i64])).map_err(onnx)?,
        ];
        for (i, (shape, data)) in self.cache.iter().enumerate() {
            let slot = if i % 2 == 0 { "key" } else { "value" };
            let value = Tensor::from_array((shape.clone(), data.clone())).map_err(onnx)?;
            inputs.push((format!("past.{}.{slot}", i / 2).into(), value.into()));
        }

        let (logits, cache) = {
            let outputs = self.s1_step.run(inputs).map_err(onnx)?;
            absorb(&outputs, self.layers)?
        };
        self.cache = cache;
        Ok(logits)
    }

    fn s2(
        &mut self,
        codes: &[u32],
        text: &[u32],
        speaker: &[f32],
        noise: &[f32],
    ) -> Result<Vec<f32>> {
        let frames = codes.len() as i64 * 2;
        let outputs = self
            .s2
            .run(ort::inputs![
                "codes" => ids(codes)?,
                "text" => ids(text)?,
                "speaker" => Tensor::from_array((
                    vec![1i64, speaker.len() as i64, 1],
                    speaker.to_vec(),
                )).map_err(onnx)?,
                "noise" => Tensor::from_array((
                    vec![1i64, PRIOR_CHANNELS as i64, frames],
                    noise.to_vec(),
                )).map_err(onnx)?,
            ])
            .map_err(onnx)?;
        let (_, audio) = outputs["audio"].try_extract_tensor::<f32>().map_err(onnx)?;
        Ok(audio.to_vec())
    }
}

/// The four graphs, and the names `find` looks for.
const GRAPHS: [&str; 4] = [
    "reference.onnx",
    "s1_prompt.onnx",
    "s1_step.onnx",
    "s2.onnx",
];

/// A `[1, n]` int64 tensor — every id sequence these graphs take.
fn ids(values: &[u32]) -> Result<Tensor<i64>> {
    let data: Vec<i64> = values.iter().map(|&v| v as i64).collect();
    Tensor::from_array((vec![1i64, values.len() as i64], data)).map_err(onnx)
}

/// Build a session with CUDA-then-CPU execution providers.
///
/// `stt-core` and `rvc-core` each have the same six lines. They are not shared
/// because engines do not depend on each other, and a third copy is the moment
/// to consider a kit crate rather than the moment to reach across.
fn session(path: &Path) -> Result<Session> {
    use ort::execution_providers::{CPU, CUDA};

    Session::builder()
        .map_err(onnx)?
        .with_execution_providers([CUDA::default().build(), CPU::default().build()])
        .map_err(|e| TtsError::Weights(e.to_string()))?
        .commit_from_file(path)
        .map_err(|e| TtsError::Weights(format!("{}: {e}", path.display())))
}

fn onnx(e: ort::Error) -> TtsError {
    TtsError::Weights(format!("onnxruntime: {e}"))
}
