//! Streaming and batch conversion built on top of [`RvcModel`].
//!
//! Audio is processed in fixed input blocks with a left "look-back" context so
//! each block has enough history for stable content/F0 estimation. The context
//! portion is dropped from the output, and consecutive outputs are joined with a
//! short linear crossfade to suppress seams. Everything is mono `f32`: input at
//! 16 kHz (the analysis rate), output at the generator's sample rate.

use audio_kit::Samples;
use futures::{Stream, StreamExt};
use tokio::sync::mpsc;

use crate::backend::Generator;
use crate::config::{ANALYSIS_SR, ConvertParams};
use audio_kit::{DenoiseParams, Denoiser};
use crate::error::Result;

/// Block/overlap parameters, expressed in 16 kHz input samples.
#[derive(Debug, Clone, Copy)]
pub struct StreamParams {
    /// Input samples emitted per processed block.
    pub block: usize,
    /// Left look-back context prepended to each block before inference.
    pub context: usize,
    /// Crossfade length used to join consecutive output blocks.
    pub crossfade: usize,
}

impl StreamParams {
    /// Low-latency preset for the streaming filter (~0.5 s blocks).
    pub fn realtime() -> Self {
        Self {
            block: ANALYSIS_SR as usize / 2,
            context: ANALYSIS_SR as usize / 4,
            crossfade: ANALYSIS_SR as usize / 20,
        }
    }

    /// Large-block preset for batch `convert` (higher quality, more latency).
    pub fn batch() -> Self {
        Self {
            block: ANALYSIS_SR as usize * 15,
            context: ANALYSIS_SR as usize / 2,
            crossfade: ANALYSIS_SR as usize / 20,
        }
    }
}

/// Stateful block processor over any [`Generator`] backend (ONNX or Burn). Not
/// `Send`-friendly to share across threads; it is intended to live on a single
/// (blocking) worker.
pub struct Converter {
    generator: Box<dyn Generator>,
    params: StreamParams,
    conv_params: ConvertParams,
    /// Pending, not-yet-processed input samples (16 kHz).
    buf: Vec<f32>,
    /// Left context carried into the next block (16 kHz).
    context: Vec<f32>,
    /// Withheld tail of the last emitted output, awaiting crossfade.
    prev_tail: Vec<f32>,
    /// Output-domain crossfade length (samples at model_sr).
    xf_out: usize,
    /// Output samples per input sample.
    ratio: f32,
    /// Optional post de-hiss stage, applied to the emitted output stream.
    denoiser: Option<Denoiser>,
}

impl Converter {
    /// Build a converter around any loaded [`Generator`] backend.
    pub fn new(
        generator: impl Generator + 'static,
        params: StreamParams,
        conv_params: ConvertParams,
    ) -> Self {
        let ratio = generator.output_sr() as f32 / ANALYSIS_SR as f32;
        let xf_out = (params.crossfade as f32 * ratio).round() as usize;
        Self {
            generator: Box::new(generator),
            params,
            conv_params,
            buf: Vec::new(),
            context: Vec::new(),
            prev_tail: Vec::new(),
            xf_out,
            ratio,
            denoiser: None,
        }
    }

    /// Enable the optional post de-hiss stage with the given [`DenoiseParams`]
    /// (off by default; pass `None` to leave it disabled). See [`crate::Denoiser`].
    pub fn with_denoise(mut self, params: Option<DenoiseParams>) -> Self {
        self.denoiser = params.map(|p| Denoiser::new(self.generator.output_sr(), p));
        self
    }

    /// The generator's output sample rate.
    pub fn output_sr(&self) -> u32 {
        self.generator.output_sr()
    }

    /// Clear all streaming state (pending input, look-back context, crossfade
    /// tail) so the same loaded model can process an independent input next,
    /// e.g. the next file in a batch.
    pub fn reset(&mut self) {
        self.buf.clear();
        self.context.clear();
        self.prev_tail.clear();
        if let Some(d) = &mut self.denoiser {
            d.reset();
        }
    }

    /// Route an emitted output chunk through the optional de-hiss stage.
    fn emit(&mut self, chunk: Samples, outs: &mut Vec<Samples>) {
        match &mut self.denoiser {
            Some(d) => {
                let den = d.process(&chunk);
                if !den.is_empty() {
                    outs.push(den);
                }
            }
            None => outs.push(chunk),
        }
    }

    /// Feed input samples; returns zero or more converted output chunks.
    pub fn push(&mut self, input: &[f32]) -> Result<Vec<Samples>> {
        self.buf.extend_from_slice(input);
        let mut outs = Vec::new();
        while self.buf.len() >= self.params.block {
            let block: Vec<f32> = self.buf.drain(..self.params.block).collect();
            if let Some(chunk) = self.process_block(&block)? {
                self.emit(chunk, &mut outs);
            }
        }
        Ok(outs)
    }

    /// Flush remaining buffered input, the withheld crossfade tail, and the
    /// de-hiss stage's look-ahead.
    pub fn flush(&mut self) -> Result<Vec<Samples>> {
        let mut outs = Vec::new();
        if !self.buf.is_empty() {
            let block: Vec<f32> = std::mem::take(&mut self.buf);
            if let Some(chunk) = self.process_block(&block)? {
                self.emit(chunk, &mut outs);
            }
        }
        if !self.prev_tail.is_empty() {
            let tail = std::mem::take(&mut self.prev_tail);
            self.emit(tail, &mut outs);
        }
        if let Some(d) = &mut self.denoiser {
            let tail = d.flush();
            if !tail.is_empty() {
                outs.push(tail);
            }
        }
        Ok(outs)
    }

    /// Convert one input block (with context) and return the emittable output,
    /// withholding a crossfade tail for the next join.
    fn process_block(&mut self, block: &[f32]) -> Result<Option<Samples>> {
        // Prepend look-back context for stable analysis.
        let mut input = Vec::with_capacity(self.context.len() + block.len());
        input.extend_from_slice(&self.context);
        input.extend_from_slice(block);

        let conv = self.generator.convert_segment(&input, self.conv_params)?;

        // Update context to the tail of this block's raw input for next time.
        let ctx_n = self.params.context.min(block.len());
        self.context = block[block.len() - ctx_n..].to_vec();

        if conv.is_empty() {
            return Ok(None);
        }

        // Keep this block's output (drop the context prefix) plus an extra
        // `xf_out` samples of look-back, so this block's head OVERLAPS the
        // previous block's withheld tail — the same output-time region the
        // crossfade below blends. Without the overlap the two regions are merely
        // adjacent, and blending would delete `xf_out` samples per block: ~10% of
        // each 0.5 s realtime block (audible speed-up + seam clicks), negligible
        // for batch's 15 s blocks. Saturates to 0 on the first block.
        let block_out = (block.len() as f32 * self.ratio).round() as usize;
        let start = conv.len().saturating_sub(block_out + self.xf_out);
        let mut kept = conv[start..].to_vec();

        // Crossfade the head of `kept` against the previously withheld tail.
        if !self.prev_tail.is_empty() {
            let xf = self.xf_out.min(kept.len()).min(self.prev_tail.len());
            for (i, (k, &p)) in kept[..xf].iter_mut().zip(self.prev_tail.iter()).enumerate() {
                let w = i as f32 / xf as f32;
                *k = p * (1.0 - w) + *k * w;
            }
            // If prev_tail was longer than what we blended, prepend the remainder
            // so no samples are lost.
            if self.prev_tail.len() > xf {
                let mut head = self.prev_tail[xf..].to_vec();
                head.extend_from_slice(&kept);
                kept = head;
            }
        }

        // Withhold a new tail for the next crossfade.
        let tail_n = self.xf_out.min(kept.len());
        self.prev_tail = kept.split_off(kept.len() - tail_n);

        if kept.is_empty() {
            Ok(None)
        } else {
            Ok(Some(kept))
        }
    }

    /// Convert an entire 16 kHz buffer to a single output vector at `output_sr()`.
    pub fn convert_all(&mut self, wav16k: &[f32]) -> Result<Vec<f32>> {
        let mut out = Vec::new();
        for chunk in self.push(wav16k)? {
            out.extend(chunk);
        }
        for chunk in self.flush()? {
            out.extend(chunk);
        }
        Ok(out)
    }
}

/// Drive a [`Converter`] over an async input stream, yielding an async output
/// stream. The model runs on a dedicated blocking worker so `ort` never touches
/// the async runtime threads; back-pressure flows through bounded channels.
pub fn convert_stream<S>(
    mut converter: Converter,
    mut input: S,
) -> impl Stream<Item = Result<Samples>>
where
    S: Stream<Item = std::result::Result<Samples, audio_kit::AudioError>> + Unpin + Send + 'static,
{
    let (in_tx, mut in_rx) = mpsc::channel::<Samples>(32);
    let (out_tx, mut out_rx) = mpsc::channel::<Result<Samples>>(32);

    // Forward the async input stream into the worker's input channel, surfacing
    // decode errors on the output side.
    let err_tx = out_tx.clone();
    tokio::spawn(async move {
        while let Some(item) = input.next().await {
            match item {
                Ok(s) => {
                    if in_tx.send(s).await.is_err() {
                        break;
                    }
                }
                Err(e) => {
                    let _ = err_tx.send(Err(e.into())).await;
                    break;
                }
            }
        }
        // Dropping in_tx signals end-of-input to the worker.
    });

    // Blocking worker owns the model.
    let worker_out = out_tx;
    tokio::task::spawn_blocking(move || {
        while let Some(samples) = in_rx.blocking_recv() {
            match converter.push(&samples) {
                Ok(chunks) => {
                    for c in chunks {
                        if worker_out.blocking_send(Ok(c)).is_err() {
                            return;
                        }
                    }
                }
                Err(e) => {
                    let _ = worker_out.blocking_send(Err(e));
                    return;
                }
            }
        }
        match converter.flush() {
            Ok(chunks) => {
                for c in chunks {
                    if worker_out.blocking_send(Ok(c)).is_err() {
                        return;
                    }
                }
            }
            Err(e) => {
                let _ = worker_out.blocking_send(Err(e));
            }
        }
    });

    async_stream::stream! {
        while let Some(item) = out_rx.recv().await {
            yield item;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ConvertParams;

    /// A stand-in generator that upsamples the input by an integer `ratio` with
    /// sample-hold, so streaming behaviour can be checked without ONNX/GPU.
    struct FakeGen {
        sr: u32,
        ratio: usize,
    }

    impl Generator for FakeGen {
        fn output_sr(&self) -> u32 {
            self.sr
        }
        fn convert_segment(&mut self, wav16k: &[f32], _p: ConvertParams) -> Result<Vec<f32>> {
            let mut out = Vec::with_capacity(wav16k.len() * self.ratio);
            for &s in wav16k {
                for _ in 0..self.ratio {
                    out.push(s);
                }
            }
            Ok(out)
        }
    }

    /// Streaming must preserve the sample-rate ratio (output length ≈
    /// `input * ratio`). The pre-fix crossfade deleted `xf_out` samples per
    /// block — ~10% of each realtime block (the audible speed-up).
    #[test]
    fn streaming_preserves_length() {
        let ratio = 3usize; // 48 kHz out / 16 kHz in
        let params = StreamParams::realtime();
        let mut conv = Converter::new(
            FakeGen {
                sr: ANALYSIS_SR * ratio as u32,
                ratio,
            },
            params,
            ConvertParams { transpose: 0 },
        );
        // ~4 s of input → many realtime blocks, so any per-block loss compounds.
        let input: Vec<f32> = (0..ANALYSIS_SR as usize * 4)
            .map(|i| (i as f32 * 0.001).sin() * 0.5)
            .collect();
        let out = conv.convert_all(&input).unwrap();
        let expected = input.len() * ratio;
        let drift = (out.len() as isize - expected as isize).unsigned_abs();
        // Allow one crossfade's worth of slack for edge handling.
        assert!(
            drift <= conv.xf_out * 2,
            "output length {} drifted from expected {} by {} (> {})",
            out.len(),
            expected,
            drift,
            conv.xf_out * 2
        );
    }
}
