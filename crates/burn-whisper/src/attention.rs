//! Multi-head attention, and the incremental key/value cache decoding needs.
//!
//! Field names are Hugging Face's (`q_proj`, `k_proj`, `v_proj`, `out_proj`)
//! because those are the names in the checkpoint. One shape to remember:
//! **`k_proj` has no bias** — Whisper omits it, since a constant added to every
//! key shifts all logits in a row equally and softmax cancels it.

use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::tensor::activation::softmax;
use burn::tensor::backend::Backend;
use burn::tensor::{Bool, Tensor};

/// Keys and values already computed for earlier positions, `[batch, seq, d_model]`.
///
/// Held outside the module rather than as state inside it: a `Module` is cloned
/// per device and saved to checkpoints, and neither should carry a decode in
/// progress.
#[derive(Debug, Clone)]
pub struct KvCache<B: Backend> {
    pub k: Tensor<B, 3>,
    pub v: Tensor<B, 3>,
}

#[derive(Module, Debug)]
pub struct Attention<B: Backend> {
    q_proj: Linear<B>,
    k_proj: Linear<B>,
    v_proj: Linear<B>,
    out_proj: Linear<B>,
    n_head: usize,
}

impl<B: Backend> Attention<B> {
    pub fn new(d_model: usize, n_head: usize, device: &B::Device) -> Self {
        Self {
            q_proj: LinearConfig::new(d_model, d_model).init(device),
            k_proj: LinearConfig::new(d_model, d_model)
                .with_bias(false)
                .init(device),
            v_proj: LinearConfig::new(d_model, d_model).init(device),
            out_proj: LinearConfig::new(d_model, d_model).init(device),
            n_head,
        }
    }

    /// Self-attention over `x`, `[batch, seq, d_model]`.
    pub fn forward(&self, x: Tensor<B, 3>, mask: Option<Tensor<B, 2, Bool>>) -> Tensor<B, 3> {
        let k = self.k_proj.forward(x.clone());
        let v = self.v_proj.forward(x.clone());
        self.attend(x, k, v, mask)
    }

    /// Self-attention that appends this step's keys and values to `cache` and
    /// attends over everything cached so far — the decoder's inner loop, where
    /// `x` is usually a single token.
    pub fn forward_cached(
        &self,
        x: Tensor<B, 3>,
        mask: Option<Tensor<B, 2, Bool>>,
        cache: &mut Option<KvCache<B>>,
    ) -> Tensor<B, 3> {
        let k = self.k_proj.forward(x.clone());
        let v = self.v_proj.forward(x.clone());
        let (k, v) = match cache.take() {
            Some(prev) => (
                Tensor::cat(vec![prev.k, k], 1),
                Tensor::cat(vec![prev.v, v], 1),
            ),
            None => (k, v),
        };
        *cache = Some(KvCache {
            k: k.clone(),
            v: v.clone(),
        });
        self.attend(x, k, v, mask)
    }

    /// Cross-attention from `x` onto the encoder output `xa`.
    ///
    /// `xa` does not change across decoding steps, so its projections are
    /// computed on the first call and reused — worth far more than the
    /// self-attention cache here, since `xa` is 1500 frames wide.
    pub fn forward_cross(
        &self,
        x: Tensor<B, 3>,
        xa: Tensor<B, 3>,
        cache: &mut Option<KvCache<B>>,
    ) -> Tensor<B, 3> {
        let KvCache { k, v } = cache
            .get_or_insert_with(|| KvCache {
                k: self.k_proj.forward(xa.clone()),
                v: self.v_proj.forward(xa),
            })
            .clone();
        self.attend(x, k, v, None)
    }

    /// Scaled dot-product attention. `q_src` is projected here; `k` and `v`
    /// arrive already projected, because where they came from is the only thing
    /// that differs between the three entry points above.
    fn attend(
        &self,
        q_src: Tensor<B, 3>,
        k: Tensor<B, 3>,
        v: Tensor<B, 3>,
        mask: Option<Tensor<B, 2, Bool>>,
    ) -> Tensor<B, 3> {
        let q = self.q_proj.forward(q_src);
        let [batch, n_q, d_model] = q.dims();
        let n_kv = k.dims()[1];
        let d_head = d_model / self.n_head;

        // [batch, seq, d_model] -> [batch, head, seq, d_head]
        let heads = |t: Tensor<B, 3>, seq: usize| {
            t.reshape([batch, seq, self.n_head, d_head]).swap_dims(1, 2)
        };
        // The reference scales q and k by `d_head^-0.25` each, which is an fp16
        // overflow guard; the product is the same and this crate is fp32.
        let q = heads(q, n_q) * (d_head as f64).powf(-0.5);
        let k = heads(k, n_kv);
        let v = heads(v, n_kv);

        let mut scores = q.matmul(k.swap_dims(2, 3)); // [batch, head, n_q, n_kv]
        if let Some(mask) = mask {
            scores = scores.mask_fill(mask.unsqueeze::<4>(), f32::NEG_INFINITY);
        }
        let out = softmax(scores, 3)
            .matmul(v)
            .swap_dims(1, 2)
            .reshape([batch, n_q, d_model]);
        self.out_proj.forward(out)
    }
}

/// The decoder's causal mask: `true` where attention must be blocked, so a
/// query may not see keys from its own future.
///
/// Built for the `[n_q, n_kv]` slice actually being computed. With a cache of
/// `n_kv - n_q` earlier tokens, query `i` sits at absolute position
/// `n_kv - n_q + i`, which is exactly `tril_mask`'s offset. During incremental
/// decoding `n_q` is 1 and the mask comes out all-false — one new token
/// legitimately sees the whole prefix.
///
/// It is [`Tensor::tril_mask`], not `triu_mask`, and the names invite the
/// opposite: Burn's masks say which triangle to *keep*, so `triu_mask` blocks
/// the lower triangle. Using it here reverses time, and at `n_q == n_kv == 1` it
/// blocks the only position there is — a whole row of `-inf` into softmax, which
/// is `NaN` rather than an error.
pub fn causal_mask<B: Backend>(n_q: usize, n_kv: usize, device: &B::Device) -> Tensor<B, 2, Bool> {
    Tensor::tril_mask([n_q, n_kv], (n_kv - n_q) as i64, device)
}
