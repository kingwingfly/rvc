//! The native Burn implementation of [`Engine`].

use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor, TensorData};
use burn_whisper::{DecodeState, Whisper, WhisperConfig};

use crate::engine::Engine;
use crate::error::{Result, SttError};

pub struct BurnEngine<B: Backend> {
    model: Whisper<B>,
    device: B::Device,
    mel_bins: usize,
    /// Encoder output for the current window, `[1, frames, d_model]`.
    audio: Option<Tensor<B, 3>>,
    state: DecodeState<B>,
}

impl<B: Backend> BurnEngine<B> {
    /// Load `model.safetensors` from a Hugging Face Whisper directory.
    pub fn load(cfg: &WhisperConfig, path: &std::path::Path, device: &B::Device) -> Result<Self> {
        // Named explicitly, because the most likely way to get here is pointing a
        // Burn backend at an ONNX export, and "No such file or directory" alone
        // does not say that.
        if !path.exists() {
            return Err(SttError::Weights(format!(
                "{} not found — the Burn backends need a `model.safetensors`; \
                 for an ONNX export use `--backend onnx`",
                path.display()
            )));
        }
        let mut model = Whisper::<B>::new(cfg, device);
        let applied = model
            .load_safetensors(path)
            .map_err(|e| SttError::Weights(e.to_string()))?;
        if !applied.missing.is_empty() {
            return Err(SttError::Weights(format!(
                "{} parameters had no tensor in the checkpoint (first: {})",
                applied.missing.len(),
                applied.missing[0].0
            )));
        }
        Ok(Self {
            state: model.decoder.state(),
            model,
            device: device.clone(),
            mel_bins: cfg.num_mel_bins,
            audio: None,
        })
    }
}

impl<B: Backend> Engine for BurnEngine<B> {
    fn mel_bins(&self) -> usize {
        self.mel_bins
    }

    fn encode(&mut self, mel: &[f32], frames: usize) -> Result<()> {
        let mel: Tensor<B, 3> = Tensor::from_data(
            TensorData::new(mel.to_vec(), [1, self.mel_bins, frames]),
            &self.device,
        );
        self.audio = Some(self.model.encoder.forward(mel));
        self.restart();
        Ok(())
    }

    fn step(&mut self, tokens: &[u32]) -> Result<Vec<f32>> {
        let audio = self
            .audio
            .clone()
            .ok_or_else(|| SttError::Config("step before encode".into()))?;

        let ids: Vec<i32> = tokens.iter().map(|&i| i as i32).collect();
        let input: Tensor<B, 2, Int> =
            Tensor::from_data(TensorData::new(ids, [1, tokens.len()]), &self.device);

        let logits = self.model.decoder.forward(input, audio, &mut self.state);
        let [_, seq, vocab] = logits.dims();
        Ok(logits
            .slice([0..1, seq - 1..seq, 0..vocab])
            .into_data()
            .to_vec()
            .expect("logits are f32"))
    }

    fn restart(&mut self) {
        self.state = self.model.decoder.state();
    }
}
