//! The VITS attention stack (`attentions.Encoder`).
//!
//! Relative-position multi-head attention, exactly as the reference implements
//! it, so weights load unchanged into either model built on it. Sequence masking
//! is omitted: inference runs one clip at a time (full length), and batched
//! training pads to equal lengths — add masks there if batches mix lengths.

use burn::module::{Module, Param};
use burn::nn::PaddingConfig1d;
use burn::nn::conv::{Conv1d, Conv1dConfig};
use burn::tensor::activation::{relu, softmax};
use burn::tensor::backend::Backend;
use burn::tensor::{Distribution, Tensor};

use crate::nn::VitsLayerNorm;

/// Multi-head self-attention with relative position embeddings.
#[derive(Module, Debug)]
pub struct MultiHeadAttention<B: Backend> {
    conv_q: Conv1d<B>,
    conv_k: Conv1d<B>,
    conv_v: Conv1d<B>,
    conv_o: Conv1d<B>,
    emb_rel_k: Param<Tensor<B, 3>>,
    emb_rel_v: Param<Tensor<B, 3>>,
    n_heads: usize,
    k_channels: usize,
    window_size: usize,
}

impl<B: Backend> MultiHeadAttention<B> {
    pub fn new(channels: usize, n_heads: usize, window_size: usize, device: &B::Device) -> Self {
        let k_channels = channels / n_heads;
        let conv = || Conv1dConfig::new(channels, channels, 1).init(device);
        let rel_stddev = (k_channels as f64).powf(-0.5);
        let rel = || {
            Param::from_tensor(Tensor::random(
                [1, window_size * 2 + 1, k_channels],
                Distribution::Normal(0.0, rel_stddev),
                device,
            ))
        };
        Self {
            conv_q: conv(),
            conv_k: conv(),
            conv_v: conv(),
            conv_o: conv(),
            emb_rel_k: rel(),
            emb_rel_v: rel(),
            n_heads,
            k_channels,
            window_size,
        }
    }

    /// `x`: `[batch, channels, time]` self-attention → `[batch, channels, time]`.
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let q = self.conv_q.forward(x.clone());
        let k = self.conv_k.forward(x.clone());
        let v = self.conv_v.forward(x);
        let out = self.attention(q, k, v);
        self.conv_o.forward(out)
    }

    fn attention(&self, q: Tensor<B, 3>, k: Tensor<B, 3>, v: Tensor<B, 3>) -> Tensor<B, 3> {
        let [b, d, t] = q.dims();
        let (h, dk) = (self.n_heads, self.k_channels);
        // [b, d, t] -> [b, h, t, dk]
        let reshape = |x: Tensor<B, 3>| x.reshape([b, h, dk, t]).swap_dims(2, 3);
        let q = reshape(q);
        let k = reshape(k);
        let v = reshape(v);

        let scale = (dk as f64).sqrt();
        let q_scaled = q / scale;
        // content scores: [b, h, t, t]
        let mut scores = q_scaled.clone().matmul(k.swap_dims(2, 3));
        // relative-key scores
        let key_rel = self.get_relative_embeddings(self.emb_rel_k.val(), t); // [1, 2t-1, dk]
        let rel_logits = q_scaled.matmul(key_rel.swap_dims(1, 2).unsqueeze::<4>()); // [b,h,t,2t-1]
        scores = scores + Self::rel_to_abs(rel_logits);

        let p_attn = softmax(scores, 3); // [b, h, t, t]
        let mut output = p_attn.clone().matmul(v); // [b, h, t, dk]
        let rel_weights = Self::abs_to_rel(p_attn); // [b, h, t, 2t-1]
        let value_rel = self.get_relative_embeddings(self.emb_rel_v.val(), t); // [1, 2t-1, dk]
        output = output + rel_weights.matmul(value_rel.unsqueeze::<4>()); // [b,h,t,dk]

        output.swap_dims(2, 3).reshape([b, d, t])
    }

    /// Slice/pad the `[1, 2*window+1, dk]` table to `[1, 2*length-1, dk]`.
    fn get_relative_embeddings(&self, emb: Tensor<B, 3>, length: usize) -> Tensor<B, 3> {
        let w = self.window_size;
        let pad = (length as isize - (w as isize + 1)).max(0) as usize;
        let slice_start = ((w as isize + 1) - length as isize).max(0) as usize;
        let slice_end = slice_start + 2 * length - 1;
        let emb = if pad > 0 {
            pad_dim1(emb, pad, pad)
        } else {
            emb
        };
        let [_, n, dk] = emb.dims();
        let _ = n;
        emb.slice([0..1, slice_start..slice_end, 0..dk])
    }

    /// `[b, h, l, 2l-1]` relative → `[b, h, l, l]` absolute indexing.
    fn rel_to_abs(x: Tensor<B, 4>) -> Tensor<B, 4> {
        let [b, h, l, _] = x.dims();
        let x = pad_last(x, 0, 1); // [b,h,l,2l]
        let x = x.reshape([b, h, l * 2 * l]);
        let x = pad_last(x, 0, l - 1); // [b,h, 2l*l + l-1]
        x.reshape([b, h, l + 1, 2 * l - 1])
            .slice([0..b, 0..h, 0..l, (l - 1)..(2 * l - 1)])
    }

    /// `[b, h, l, l]` absolute → `[b, h, l, 2l-1]` relative indexing.
    fn abs_to_rel(x: Tensor<B, 4>) -> Tensor<B, 4> {
        let [b, h, l, _] = x.dims();
        let x = pad_last(x, 0, l - 1); // [b,h,l,2l-1]
        let x = x.reshape([b, h, l * l + l * (l - 1)]);
        let x = pad_last(x, l, 0); // prepend l zeros
        x.reshape([b, h, l, 2 * l])
            .slice([0..b, 0..h, 0..l, 1..(2 * l)])
    }
}

/// Zero-pad the last dim of a rank-3 or rank-4 tensor by `(left, right)`.
fn pad_last<B: Backend, const D: usize>(
    x: Tensor<B, D>,
    left: usize,
    right: usize,
) -> Tensor<B, D> {
    if left == 0 && right == 0 {
        return x;
    }
    let mut parts = Vec::new();
    let mut dims = x.dims();
    let device = x.device();
    if left > 0 {
        dims[D - 1] = left;
        parts.push(Tensor::zeros(dims, &device));
    }
    parts.push(x.clone());
    if right > 0 {
        let mut dims = x.dims();
        dims[D - 1] = right;
        parts.push(Tensor::zeros(dims, &device));
    }
    Tensor::cat(parts, D - 1)
}

/// Zero-pad dim 1 of a `[1, n, dk]` tensor by `(left, right)`.
fn pad_dim1<B: Backend>(x: Tensor<B, 3>, left: usize, right: usize) -> Tensor<B, 3> {
    let [a, _, c] = x.dims();
    let device = x.device();
    let l = Tensor::zeros([a, left, c], &device);
    let r = Tensor::zeros([a, right, c], &device);
    Tensor::cat(vec![l, x, r], 1)
}

/// Position-wise feed-forward (`FFN`): conv → relu → conv, "same" padded.
#[derive(Module, Debug)]
pub struct Ffn<B: Backend> {
    conv_1: Conv1d<B>,
    conv_2: Conv1d<B>,
}

impl<B: Backend> Ffn<B> {
    pub fn new(channels: usize, filter_channels: usize, kernel: usize, device: &B::Device) -> Self {
        // VITS's "same" padding: (left, right) = ((k-1)/2, k/2).
        let pad = PaddingConfig1d::Explicit((kernel - 1) / 2, kernel / 2);
        Self {
            conv_1: Conv1dConfig::new(channels, filter_channels, kernel)
                .with_padding(pad.clone())
                .init(device),
            conv_2: Conv1dConfig::new(filter_channels, channels, kernel)
                .with_padding(pad)
                .init(device),
        }
    }

    /// `x`: `[batch, channels, time]`.
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let x = relu(self.conv_1.forward(x));
        self.conv_2.forward(x)
    }
}

/// One transformer block: attention + FFN with pre-residual LayerNorms.
#[derive(Module, Debug)]
struct EncoderLayer<B: Backend> {
    attn: MultiHeadAttention<B>,
    norm_1: VitsLayerNorm<B>,
    ffn: Ffn<B>,
    norm_2: VitsLayerNorm<B>,
}

/// How wide and how deep an [`Encoder`] is.
///
/// Its own struct rather than a borrow of some model's config, because both
/// models that use this stack have their own and neither should have to know
/// about the other's.
#[derive(Debug, Clone, Copy)]
pub struct EncoderConfig {
    pub hidden_channels: usize,
    pub filter_channels: usize,
    pub n_heads: usize,
    pub n_layers: usize,
    pub kernel_size: usize,
    /// Relative-position attention window; 4 in every configuration shipped so
    /// far, on both sides of the family.
    pub window_size: usize,
}

/// The transformer stack (`attentions.Encoder`).
#[derive(Module, Debug)]
pub struct Encoder<B: Backend> {
    layers: Vec<EncoderLayer<B>>,
}

impl<B: Backend> Encoder<B> {
    pub fn new(cfg: &EncoderConfig, device: &B::Device) -> Self {
        let layers = (0..cfg.n_layers)
            .map(|_| EncoderLayer {
                attn: MultiHeadAttention::new(
                    cfg.hidden_channels,
                    cfg.n_heads,
                    cfg.window_size,
                    device,
                ),
                norm_1: VitsLayerNorm::new(cfg.hidden_channels, device),
                ffn: Ffn::new(
                    cfg.hidden_channels,
                    cfg.filter_channels,
                    cfg.kernel_size,
                    device,
                ),
                norm_2: VitsLayerNorm::new(cfg.hidden_channels, device),
            })
            .collect();
        Self { layers }
    }

    /// `x`: `[batch, hidden, time]`.
    pub fn forward(&self, mut x: Tensor<B, 3>) -> Tensor<B, 3> {
        for layer in &self.layers {
            let y = layer.attn.forward(x.clone());
            x = layer.norm_1.forward(x + y);
            let y = layer.ffn.forward(x.clone());
            x = layer.norm_2.forward(x + y);
        }
        x
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    type B = burn::backend::NdArray;
    type Dev = burn::backend::ndarray::NdArrayDevice;

    fn flat<const D: usize>(t: Tensor<B, D>) -> Vec<f32> {
        t.into_data().to_vec().unwrap()
    }
    fn worst(got: &[f32], want: &[f32]) -> f32 {
        assert_eq!(got.len(), want.len(), "{got:?} vs {want:?}");
        assert!(got.iter().all(|v| v.is_finite()), "output must be finite");
        got.iter()
            .zip(want)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max)
    }

    /// `[1, n, 1]` holding `1..=n`, so a padded zero is distinguishable from
    /// row 0 — which is the whole difficulty in reading the table below.
    fn ramp(n: usize, device: &Dev) -> Tensor<B, 3> {
        let v: Vec<f32> = (1..=n).map(|i| i as f32).collect();
        Tensor::<B, 1>::from_floats(v.as_slice(), device).reshape([1, n, 1])
    }

    fn head(window_size: usize, device: &Dev) -> MultiHeadAttention<B> {
        MultiHeadAttention::new(2, 1, window_size, device)
    }

    /// Hand-computed index table for the `[1, 2w+1, dk]` → `[1, 2l-1, dk]`
    /// window slice, at a `window_size` of 4 (every configuration in the family
    /// uses 4). Rows carry `1..=9`, so a `0` in an expectation is a pad.
    ///
    /// Both regimes, because they take different branches: `l <= w + 1` slices
    /// out of the middle with no padding at all (the `pad == 0` path, which is
    /// the one that hands out a bare view of the live `Param`), and `l > w + 1`
    /// zero-pads first and then takes the whole thing.
    #[test]
    fn relative_embeddings_match_a_hand_written_index_table() {
        let device = Default::default();
        let mha = head(4, &device);
        let emb = ramp(9, &device);

        // pad == 0, sliced from the middle: start = (w + 1) - l = 2, five rows.
        let got = flat(mha.get_relative_embeddings(emb.clone(), 3));
        assert!(worst(&got, &[3.0, 4.0, 5.0, 6.0, 7.0]) == 0.0, "{got:?}");

        // pad == 0 and l == w + 1: the whole table, start = 0, nine rows.
        let got = flat(mha.get_relative_embeddings(emb.clone(), 5));
        let all: Vec<f32> = (1..=9).map(|i| i as f32).collect();
        assert!(worst(&got, &all) == 0.0, "{got:?}");

        // pad == 1: one zero row each side, all eleven rows taken.
        let got = flat(mha.get_relative_embeddings(emb.clone(), 6));
        let mut want = vec![0.0];
        want.extend_from_slice(&all);
        want.push(0.0);
        assert!(worst(&got, &want) == 0.0, "{got:?}");

        // pad == 2: two zero rows each side, all thirteen rows taken.
        let got = flat(mha.get_relative_embeddings(emb, 7));
        let mut want = vec![0.0, 0.0];
        want.extend_from_slice(&all);
        want.extend_from_slice(&[0.0, 0.0]);
        assert!(worst(&got, &want) == 0.0, "{got:?}");
    }

    /// The output width is `2l - 1` for every length, on both sides of the
    /// branch — the invariant `rel_to_abs` then depends on, and the one a
    /// wrong `slice_end` breaks without changing any value in view.
    #[test]
    fn relative_embeddings_are_always_two_l_minus_one_wide() {
        let device = Default::default();
        for w in [1usize, 4, 6] {
            let mha = head(w, &device);
            let emb = ramp(2 * w + 1, &device);
            for l in 1..=(2 * w + 3) {
                let dims = mha.get_relative_embeddings(emb.clone(), l).dims();
                assert_eq!(dims, [1, 2 * l - 1, 1], "window {w}, length {l}");
            }
        }
    }

    /// Relative → absolute, as a literal table and then as the index formula
    /// that produced it.
    ///
    /// `x[i][j]` holds the score for query `i` at relative offset `j - (l - 1)`,
    /// so the absolute entry `out[i][k]` must be `x[i][k - i + l - 1]`. That
    /// `l - 1` is the off-by-one this family gets wrong: shifting the final
    /// slice by one column still returns the right shape and finite numbers,
    /// and simply reads every score off by one position.
    #[test]
    fn rel_to_abs_matches_a_hand_written_table() {
        let device = Default::default();
        let l = 3;
        let v: Vec<f32> = (1..=15).map(|i| i as f32).collect();
        let x = Tensor::<B, 1>::from_floats(v.as_slice(), &device).reshape([1, 1, l, 2 * l - 1]);
        let got = flat(MultiHeadAttention::<B>::rel_to_abs(x));
        // Rows 1..5, 6..10, 11..15; row i keeps the window starting at l-1-i.
        let want = [3.0, 4.0, 5.0, 7.0, 8.0, 9.0, 11.0, 12.0, 13.0];
        assert!(worst(&got, &want) == 0.0, "{got:?}");

        // The same claim as a formula, over a wider case, so the table above is
        // an instance of a rule rather than a memorised answer.
        let l = 5;
        let (b, h) = (2, 3);
        let n = b * h * l * (2 * l - 1);
        let v: Vec<f32> = (1..=n).map(|i| i as f32).collect();
        let x = Tensor::<B, 1>::from_floats(v.as_slice(), &device).reshape([b, h, l, 2 * l - 1]);
        let got = flat(MultiHeadAttention::<B>::rel_to_abs(x));
        let mut want = Vec::with_capacity(b * h * l * l);
        for bh in 0..(b * h) {
            for i in 0..l {
                for k in 0..l {
                    want.push(v[(bh * l + i) * (2 * l - 1) + (k + l - 1 - i)]);
                }
            }
        }
        assert!(worst(&got, &want) == 0.0);
    }

    /// Absolute → relative, the inverse skew, as a literal table. The zeros are
    /// the entries no relative offset addresses at that query position, and
    /// where they fall is the whole content of the transform.
    #[test]
    fn abs_to_rel_matches_a_hand_written_table() {
        let device = Default::default();
        let l = 3;
        let v: Vec<f32> = (1..=9).map(|i| i as f32).collect();
        let x = Tensor::<B, 1>::from_floats(v.as_slice(), &device).reshape([1, 1, l, l]);
        let got = flat(MultiHeadAttention::<B>::abs_to_rel(x));
        let want = [
            0.0, 0.0, 1.0, 2.0, 3.0, //
            0.0, 4.0, 5.0, 6.0, 0.0, //
            7.0, 8.0, 9.0, 0.0, 0.0,
        ];
        assert!(worst(&got, &want) == 0.0, "{got:?}");
    }

    /// A round trip that must return its input exactly: `abs_to_rel` skews an
    /// absolute score matrix into relative offsets and `rel_to_abs` skews it
    /// back, so the composition in *that* order is the identity.
    ///
    /// The other order is not, and the second half of this test says so — the
    /// relative form has `2l - 1` columns per row of which only `l` are ever
    /// addressed, so a rel → abs → rel round trip zeroes the rest. That
    /// asymmetry is what makes the pair's direction load-bearing rather than
    /// decorative.
    #[test]
    fn rel_to_abs_undoes_abs_to_rel() {
        let device = Default::default();
        let (b, h, l) = (2, 3, 5);
        let x = Tensor::<B, 4>::random(
            [b, h, l, l],
            Distribution::Normal(0.0, 1.0),
            &device,
        );
        let before = flat(x.clone());
        let after = flat(MultiHeadAttention::<B>::rel_to_abs(
            MultiHeadAttention::<B>::abs_to_rel(x),
        ));
        assert!(worst(&after, &before) == 0.0);

        let y = Tensor::<B, 4>::random(
            [b, h, l, 2 * l - 1],
            Distribution::Normal(0.0, 1.0),
            &device,
        );
        let before = flat(y.clone());
        let after = flat(MultiHeadAttention::<B>::abs_to_rel(
            MultiHeadAttention::<B>::rel_to_abs(y),
        ));
        let changed = before
            .iter()
            .zip(&after)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(changed > 0.1, "the two directions are not inverses either way");
    }

    /// Scalar softmax attention written by hand, against the tensor path with
    /// the relative-position tables zeroed out.
    ///
    /// The four 1×1 convolutions get four **independent random** weight
    /// matrices, and the reference does those projections itself by index — the
    /// shape `burn-rmvpe`'s GRU reference uses, and the reason for it is that
    /// identity convolutions would make `q = k = v = x`, whose score matrix is
    /// symmetric in query and key, so a `conv_q`/`conv_k` mix-up would be
    /// invisible. That copy-paste is exactly the class that loads at 100%
    /// coverage and produces garbage.
    ///
    /// Four things it pins that a shape check cannot: each projection is wired
    /// to its own place; the scale is `sqrt(dk)` and not `dk` or
    /// `sqrt(channels)`; the softmax runs over the **key** axis, which in
    /// self-attention has the same extent as the query axis and is therefore
    /// shape-identical when wrong; and the heads are contiguous channel blocks
    /// (`head * dk + d`), which an interleaved split would also survive.
    #[test]
    fn attention_matches_a_scalar_softmax_reference() {
        let device = Default::default();
        let (c, h, t) = (4usize, 2usize, 5usize);
        let dk = c / h;
        let mut mha = MultiHeadAttention::<B>::new(c, h, 4, &device);

        let normal = Distribution::Normal(0.0, 1.0);
        let draw = || Tensor::<B, 3>::random([c, c, 1], normal, &device);
        let (wq, wk, wv, wo) = (draw(), draw(), draw(), draw());
        for (conv, w) in [
            (&mut mha.conv_q, &wq),
            (&mut mha.conv_k, &wk),
            (&mut mha.conv_v, &wv),
            (&mut mha.conv_o, &wo),
        ] {
            conv.weight = Param::from_tensor(w.clone());
            conv.bias = Some(Param::from_tensor(Tensor::zeros([c], &device)));
        }
        let zeros = || Param::from_tensor(Tensor::<B, 3>::zeros([1, 9, dk], &device));
        mha.emb_rel_k = zeros();
        mha.emb_rel_v = zeros();

        let x = Tensor::<B, 3>::random([1, c, t], normal, &device);
        let got = flat(mha.forward(x.clone()));

        // Everything below is scalar and indexed by hand: `[channel][time]`
        // flattened channel-major, which is how the tensors are laid out.
        let xs: Vec<f64> = flat(x).iter().map(|v| *v as f64).collect();
        let proj = |w: &Tensor<B, 3>, src: &[f64]| -> Vec<f64> {
            let w: Vec<f32> = flat(w.clone());
            let mut y = vec![0.0f64; c * t];
            for o in 0..c {
                for i in 0..t {
                    y[o * t + i] = (0..c).map(|k| w[o * c + k] as f64 * src[k * t + i]).sum();
                }
            }
            y
        };
        let (q, k, v) = (proj(&wq, &xs), proj(&wk, &xs), proj(&wv, &xs));

        let mut att = vec![0.0f64; c * t];
        for head in 0..h {
            let ch = |d: usize| head * dk + d;
            for i in 0..t {
                let logits: Vec<f64> = (0..t)
                    .map(|j| {
                        let dot: f64 = (0..dk)
                            .map(|d| q[ch(d) * t + i] * k[ch(d) * t + j])
                            .sum();
                        dot / (dk as f64).sqrt()
                    })
                    .collect();
                let max = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let exp: Vec<f64> = logits.iter().map(|l| (l - max).exp()).collect();
                let sum: f64 = exp.iter().sum();
                for d in 0..dk {
                    att[ch(d) * t + i] = (0..t).map(|j| exp[j] / sum * v[ch(d) * t + j]).sum();
                }
            }
        }
        let want: Vec<f32> = proj(&wo, &att).iter().map(|v| *v as f32).collect();
        assert!(worst(&got, &want) < 1e-5, "{got:?} vs {want:?}");
    }

    /// The two relative-position tables are not interchangeable: `emb_rel_k`
    /// biases the scores before the softmax and `emb_rel_v` mixes into the
    /// values after it. Reading one of them twice — the easiest possible slip,
    /// since the two call sites differ by one letter — changes no shape and
    /// keeps everything finite.
    ///
    /// Three runs on one input: neither table live, then each alone. All three
    /// must differ, and a table read twice collapses two of them together.
    #[test]
    fn the_key_and_value_relative_tables_are_not_interchangeable() {
        let device = Default::default();
        let (c, h, t) = (4usize, 2usize, 6usize);
        let dk = c / h;
        let normal = Distribution::Normal(0.0, 1.0);
        let table = Tensor::<B, 3>::random([1, 9, dk], normal, &device);
        let x = Tensor::<B, 3>::random([1, c, t], normal, &device);

        let mut mha = MultiHeadAttention::<B>::new(c, h, 4, &device);
        let run = |mha: &MultiHeadAttention<B>| flat(mha.forward(x.clone()));
        let zeros = || Param::from_tensor(Tensor::<B, 3>::zeros([1, 9, dk], &device));

        mha.emb_rel_k = zeros();
        mha.emb_rel_v = zeros();
        let neither = run(&mha);
        mha.emb_rel_k = Param::from_tensor(table.clone());
        let key_only = run(&mha);
        mha.emb_rel_k = zeros();
        mha.emb_rel_v = Param::from_tensor(table);
        let value_only = run(&mha);

        let apart = |a: &[f32], b: &[f32]| {
            assert!(a.iter().chain(b).all(|v| v.is_finite()));
            a.iter().zip(b).map(|(p, q)| (p - q).abs()).fold(0.0f32, f32::max)
        };
        assert!(apart(&neither, &key_only) > 1e-4, "the key table does nothing");
        assert!(apart(&neither, &value_only) > 1e-4, "the value table does nothing");
        assert!(
            apart(&key_only, &value_only) > 1e-4,
            "the two tables are being read from the same place"
        );
    }

    /// Finiteness on its own, on both sides of the padding branch and with the
    /// relative tables live. A softmax row that has gone entirely to `-inf`
    /// yields `NaN` rather than an error, and `assert_approx_eq` compares `NaN`
    /// to `NaN` without complaint — so this is asserted separately from any
    /// comparison, which is how a reversed mask once shipped at 100% coverage.
    #[test]
    fn attention_output_is_finite_at_every_length() {
        let device = Default::default();
        let mha = MultiHeadAttention::<B>::new(8, 2, 4, &device);
        for t in [1usize, 4, 5, 6, 17] {
            let x = Tensor::<B, 3>::random([2, 8, t], Distribution::Normal(0.0, 1.0), &device);
            let y = mha.forward(x);
            assert_eq!(y.dims(), [2, 8, t]);
            assert!(flat(y).iter().all(|v| v.is_finite()), "length {t}");
        }
    }

    /// VITS's "same" padding is asymmetric — `(k-1)/2` left and `k/2` right —
    /// and this pins the alignment rather than only the length.
    ///
    /// Both convolutions are a delta at kernel index 0, so each shifts its
    /// input right by exactly its left pad and the whole FFN is a shift by two.
    /// An even kernel is the discriminating case: `k = 4` needs `(1, 2)`, and
    /// `(2, 1)` — the same total, the same output length — shifts by four.
    /// `k = 3` is the control, where the two spellings agree.
    #[test]
    fn ffn_same_padding_keeps_the_alignment_for_even_kernels() {
        let device = Default::default();
        for kernel in [3usize, 4] {
            let mut ffn = Ffn::<B>::new(1, 1, kernel, &device);
            let mut delta = vec![0.0f32; kernel];
            delta[0] = 1.0;
            for conv in [&mut ffn.conv_1, &mut ffn.conv_2] {
                conv.weight = Param::from_tensor(
                    Tensor::<B, 1>::from_floats(delta.as_slice(), &device).reshape([1, 1, kernel]),
                );
                conv.bias = Some(Param::from_tensor(Tensor::zeros([1], &device)));
            }
            // All positive, so the ReLU between the two convolutions is the
            // identity and the shift is all that is left.
            let x = Tensor::<B, 1>::from_floats([1.0, 2.0, 3.0, 4.0, 5.0], &device)
                .reshape([1, 1, 5]);
            let got = flat(ffn.forward(x));
            assert!(
                worst(&got, &[0.0, 0.0, 1.0, 2.0, 3.0]) == 0.0,
                "kernel {kernel}: {got:?}"
            );
        }
    }

    /// The stack keeps `[batch, hidden, time]` and stays finite through two
    /// layers of attention, FFN and two LayerNorms — the composition the two
    /// models actually run, at a length past the relative window so the padded
    /// branch is the one exercised.
    #[test]
    fn encoder_preserves_shape_and_stays_finite() {
        let device = Default::default();
        let cfg = EncoderConfig {
            hidden_channels: 8,
            filter_channels: 16,
            n_heads: 2,
            n_layers: 2,
            kernel_size: 3,
            window_size: 4,
        };
        let enc = Encoder::<B>::new(&cfg, &device);
        let x = Tensor::<B, 3>::random([2, 8, 12], Distribution::Normal(0.0, 1.0), &device);
        let y = enc.forward(x);
        assert_eq!(y.dims(), [2, 8, 12]);
        assert!(flat(y).iter().all(|v| v.is_finite()));
    }
}
