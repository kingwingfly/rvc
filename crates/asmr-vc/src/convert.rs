//! Streaming and batch conversion built on top of [`RvcModel`].
//!
//! Audio is processed in fixed input blocks with a left "look-back" context so
//! each block has enough history for stable content/F0 estimation. The context
//! portion is dropped from the output, and consecutive outputs are joined with a
//! short linear crossfade to suppress seams. Everything is mono `f32`: input at
//! 16 kHz (the analysis rate), output at the generator's sample rate.

use asmr_audio::Samples;
use futures::{Stream, StreamExt};
use tokio::sync::mpsc;

use crate::config::{ANALYSIS_SR, ConvertParams};
use crate::error::Result;
use crate::rvc::RvcModel;

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
    /// Low-latency preset for `serve` (~0.5 s blocks).
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

/// Stateful block processor. Not `Send`-friendly to share across threads; it is
/// intended to live on a single (blocking) worker.
pub struct Converter {
    model: RvcModel,
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
}

impl Converter {
    /// Build a converter around a loaded model.
    pub fn new(model: RvcModel, params: StreamParams, conv_params: ConvertParams) -> Self {
        let ratio = model.output_sr() as f32 / ANALYSIS_SR as f32;
        let xf_out = (params.crossfade as f32 * ratio).round() as usize;
        Self {
            model,
            params,
            conv_params,
            buf: Vec::new(),
            context: Vec::new(),
            prev_tail: Vec::new(),
            xf_out,
            ratio,
        }
    }

    /// The generator's output sample rate.
    pub fn output_sr(&self) -> u32 {
        self.model.output_sr()
    }

    /// Clear all streaming state (pending input, look-back context, crossfade
    /// tail) so the same loaded model can process an independent input next,
    /// e.g. the next file in a batch.
    pub fn reset(&mut self) {
        self.buf.clear();
        self.context.clear();
        self.prev_tail.clear();
    }

    /// Feed input samples; returns zero or more converted output chunks.
    pub fn push(&mut self, input: &[f32]) -> Result<Vec<Samples>> {
        self.buf.extend_from_slice(input);
        let mut outs = Vec::new();
        while self.buf.len() >= self.params.block {
            let block: Vec<f32> = self.buf.drain(..self.params.block).collect();
            if let Some(chunk) = self.process_block(&block)? {
                outs.push(chunk);
            }
        }
        Ok(outs)
    }

    /// Flush remaining buffered input and the withheld crossfade tail.
    pub fn flush(&mut self) -> Result<Vec<Samples>> {
        let mut outs = Vec::new();
        if !self.buf.is_empty() {
            let block: Vec<f32> = std::mem::take(&mut self.buf);
            if let Some(chunk) = self.process_block(&block)? {
                outs.push(chunk);
            }
        }
        if !self.prev_tail.is_empty() {
            outs.push(std::mem::take(&mut self.prev_tail));
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

        let conv = self.model.convert_segment(&input, self.conv_params)?;

        // Update context to the tail of this block's raw input for next time.
        let ctx_n = self.params.context.min(block.len());
        self.context = block[block.len() - ctx_n..].to_vec();

        if conv.is_empty() {
            return Ok(None);
        }

        // Keep only the portion corresponding to this block (drop the context's
        // output prefix), estimated via the sample-rate ratio.
        let block_out = (block.len() as f32 * self.ratio).round() as usize;
        let start = conv.len().saturating_sub(block_out);
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
    S: Stream<Item = std::result::Result<Samples, asmr_audio::AudioError>> + Unpin + Send + 'static,
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
