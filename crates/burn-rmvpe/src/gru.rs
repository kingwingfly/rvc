//! One bidirectional GRU layer, laid out exactly as `torch.nn.GRU` stores it.
//!
//! **This is the piece of RMVPE most likely to be silently wrong**, which is why
//! it is written here rather than reached for from `burn::nn`. Three details of
//! PyTorch's GRU have to be reproduced, and getting any of them wrong yields a
//! network that runs, produces finite salience, and reports perfect weight
//! coverage — while predicting the wrong pitch:
//!
//! - **The parameter layout.** `torch.nn.GRU` concatenates the three gates into
//!   one `weight_ih_l0` of `[3 * hidden, input]` rather than keeping them apart.
//!   Burn's [`burn::nn::Gru`] holds three separate `GateController`s, so loading
//!   the checkpoint into it would mean *slicing* a tensor — and a
//!   [`burn_store::KeyRemapper`] renames keys, it cannot cut one into three. The
//!   fields below are therefore spelled `weight_ih_l0`, `weight_hh_l0`,
//!   `bias_ih_l0`, `bias_hh_l0` and their `_reverse` twins, which is what
//!   `rmvpe.pt` says, so nothing has to be translated at all.
//! - **The gate order is `r, z, n`** — reset, update, candidate — over the rows
//!   of those concatenated tensors. Swapping `r` and `z` is the classic port bug
//!   and is invisible to every structural check.
//! - **`bias_ih` and `bias_hh` stay separate.** For `r` and `z` their sum is all
//!   that matters, so fusing them looks safe; for the candidate gate it is not,
//!   because PyTorch computes
//!   `n = tanh(W_in x + b_in + r * (W_hn h + b_hn))` — `b_hn` is inside the
//!   reset gating. A fused bias moves it outside and changes the answer wherever
//!   `r != 1`, which is everywhere.
//!
//! Burn's own `Gru` does implement this convention (`reset_after = true`), so
//! the arithmetic here is not a departure from it — only the storage is.

use burn::module::{Module, Param};
use burn::tensor::Tensor;
use burn::tensor::activation::{sigmoid, tanh};
use burn::tensor::backend::Backend;

/// A single-layer bidirectional GRU.
///
/// The field names are `torch.nn.GRU`'s own, so the checkpoint's
/// `fc.0.gru.weight_ih_l0` lands here after nothing more than the `fc.0.gru.` →
/// `gru.` prefix rename that flattens upstream's `nn.Sequential`.
#[derive(Module, Debug)]
pub struct BiGru<B: Backend> {
    weight_ih_l0: Param<Tensor<B, 2>>,
    weight_hh_l0: Param<Tensor<B, 2>>,
    bias_ih_l0: Param<Tensor<B, 1>>,
    bias_hh_l0: Param<Tensor<B, 1>>,
    weight_ih_l0_reverse: Param<Tensor<B, 2>>,
    weight_hh_l0_reverse: Param<Tensor<B, 2>>,
    bias_ih_l0_reverse: Param<Tensor<B, 1>>,
    bias_hh_l0_reverse: Param<Tensor<B, 1>>,
    hidden: usize,
}

impl<B: Backend> BiGru<B> {
    /// A zero-initialised layer of the given shape; the weights come from a
    /// checkpoint, so the initialiser only has to make the tree the right shape.
    pub fn new(input: usize, hidden: usize, device: &B::Device) -> Self {
        let w_ih = || Param::from_tensor(Tensor::zeros([3 * hidden, input], device));
        let w_hh = || Param::from_tensor(Tensor::zeros([3 * hidden, hidden], device));
        let bias = || Param::from_tensor(Tensor::zeros([3 * hidden], device));
        Self {
            weight_ih_l0: w_ih(),
            weight_hh_l0: w_hh(),
            bias_ih_l0: bias(),
            bias_hh_l0: bias(),
            weight_ih_l0_reverse: w_ih(),
            weight_hh_l0_reverse: w_hh(),
            bias_ih_l0_reverse: bias(),
            bias_hh_l0_reverse: bias(),
            hidden,
        }
    }

    /// `[batch, time, input]` → `[batch, time, 2 * hidden]`.
    ///
    /// The two directions are concatenated on the feature axis, forward first,
    /// which is what `torch.nn.GRU(bidirectional=True)` returns and therefore
    /// what the `Linear(512, 360)` above expects.
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let forward = self.direction(
            x.clone(),
            self.weight_ih_l0.val(),
            self.weight_hh_l0.val(),
            self.bias_ih_l0.val(),
            self.bias_hh_l0.val(),
            false,
        );
        let backward = self.direction(
            x,
            self.weight_ih_l0_reverse.val(),
            self.weight_hh_l0_reverse.val(),
            self.bias_ih_l0_reverse.val(),
            self.bias_hh_l0_reverse.val(),
            true,
        );
        Tensor::cat(vec![forward, backward], 2)
    }

    /// Run one direction over the whole sequence.
    ///
    /// The input projection is lifted out of the loop: `W_ih x_t + b_ih` does not
    /// depend on the recurrence, so the whole sequence goes through one matmul
    /// and each of the `2 * time` steps is left with a single `[batch, hidden] ×
    /// [hidden, 3 * hidden]` product. On a ten-second clip that is a thousand
    /// steps per direction, so it is the difference between a warm loop and a
    /// cold one.
    fn direction(
        &self,
        x: Tensor<B, 3>,
        w_ih: Tensor<B, 2>,
        w_hh: Tensor<B, 2>,
        b_ih: Tensor<B, 1>,
        b_hh: Tensor<B, 1>,
        reverse: bool,
    ) -> Tensor<B, 3> {
        let [batch, time, _] = x.dims();
        let h_size = self.hidden;
        let device = x.device();

        // `[batch, time, 3 * hidden]`: `W_ih` is stored `[3 * hidden, input]`,
        // PyTorch's layout, hence the transpose here rather than at load time.
        //
        // Both transposes below are of a `Param::val()` clone taken in
        // `forward`, so the parameter itself stays live in the module for the
        // whole call — and `w_hh`'s view is then held across every timestep of
        // the loop, which is the longest such window in this workspace. On
        // LibTorch a transposed view carries a storage handle burn-tch believes
        // is exclusive (CLAUDE.md, **`swap_dims` on LibTorch returns a view
        // burn-tch forgets the provenance of**), so the only thing standing
        // between these two lines and a scribbled-on weight is that `matmul` —
        // their sole consumer — allocates its output and mutates neither
        // operand. `tch_aliasing` below pins exactly that.
        let gates_x =
            x.matmul(w_ih.transpose().unsqueeze::<3>()) + b_ih.reshape([1, 1, 3 * h_size]);
        let w_hh = w_hh.transpose();
        let b_hh = b_hh.reshape([1, 3 * h_size]);

        let mut h = Tensor::zeros([batch, h_size], &device);
        let mut outputs = Vec::with_capacity(time);
        for step in 0..time {
            let t = if reverse { time - 1 - step } else { step };
            let gx = gates_x.clone().narrow(1, t, 1).reshape([batch, 3 * h_size]);
            let gh = h.clone().matmul(w_hh.clone()) + b_hh.clone();

            let r = sigmoid(gx.clone().narrow(1, 0, h_size) + gh.clone().narrow(1, 0, h_size));
            let z = sigmoid(
                gx.clone().narrow(1, h_size, h_size) + gh.clone().narrow(1, h_size, h_size),
            );
            // `b_hh`'s candidate slice sits inside the reset gating — see the
            // module docs; this line is the one a fused bias would break.
            let n = tanh(gx.narrow(1, 2 * h_size, h_size) + r * gh.narrow(1, 2 * h_size, h_size));

            h = n * z.clone().neg().add_scalar(1.0) + z * h;
            outputs.push(h.clone().unsqueeze_dim::<3>(1));
        }
        if reverse {
            // Collected newest-first; a reverse GRU still reports its state at
            // the timestep it belongs to.
            outputs.reverse();
        }
        Tensor::cat(outputs, 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::Distribution;

    type B = burn_ndarray::NdArray;

    /// A scalar GRU written straight from `torch.nn.GRU`'s documented equations,
    /// for one direction. Independent of the tensor code above — different
    /// indexing, no batching, no lifted projection — so it catches a wrong gate
    /// order, a wrong slice and a fused bias, which is the whole point.
    #[allow(clippy::too_many_arguments)]
    fn reference(
        x: &[f32],
        w_ih: &[f32],
        w_hh: &[f32],
        b_ih: &[f32],
        b_hh: &[f32],
        input: usize,
        hidden: usize,
        time: usize,
        reverse: bool,
    ) -> Vec<f32> {
        let sig = |v: f32| 1.0 / (1.0 + (-v).exp());
        let mut h = vec![0.0f32; hidden];
        let mut out = vec![0.0f32; time * hidden];
        for step in 0..time {
            let t = if reverse { time - 1 - step } else { step };
            // Gate g occupies rows [g * hidden, (g + 1) * hidden) in r, z, n order.
            let gate = |g: usize| -> Vec<f32> {
                (0..hidden)
                    .map(|i| {
                        let row = g * hidden + i;
                        let xi: f32 = (0..input)
                            .map(|j| w_ih[row * input + j] * x[t * input + j])
                            .sum();
                        let hi: f32 = (0..hidden).map(|j| w_hh[row * hidden + j] * h[j]).sum();
                        // Returned as (input part, hidden part): the candidate
                        // gate needs them apart.
                        xi + b_ih[row] + hi + b_hh[row]
                    })
                    .collect()
            };
            let r: Vec<f32> = gate(0).into_iter().map(sig).collect();
            let z: Vec<f32> = gate(1).into_iter().map(sig).collect();
            let n: Vec<f32> = (0..hidden)
                .map(|i| {
                    let row = 2 * hidden + i;
                    let xi: f32 = (0..input)
                        .map(|j| w_ih[row * input + j] * x[t * input + j])
                        .sum();
                    let hi: f32 = (0..hidden).map(|j| w_hh[row * hidden + j] * h[j]).sum();
                    (xi + b_ih[row] + r[i] * (hi + b_hh[row])).tanh()
                })
                .collect();
            for i in 0..hidden {
                h[i] = (1.0 - z[i]) * n[i] + z[i] * h[i];
            }
            out[t * hidden..(t + 1) * hidden].copy_from_slice(&h);
        }
        out
    }

    #[test]
    fn matches_a_scalar_reference_in_both_directions() {
        let (input, hidden, time) = (5usize, 4usize, 7usize);
        let device = Default::default();
        let normal = Distribution::Normal(0.0, 1.0);

        let mut gru = BiGru::<B>::new(input, hidden, &device);
        let draw2 = |rows, cols| Tensor::<B, 2>::random([rows, cols], normal, &device);
        let w_ih_f = draw2(3 * hidden, input);
        let w_hh_f = draw2(3 * hidden, hidden);
        let w_ih_b = draw2(3 * hidden, input);
        let w_hh_b = draw2(3 * hidden, hidden);
        let draw1 = || Tensor::<B, 1>::random([3 * hidden], normal, &device);
        let (b_ih_f, b_hh_f, b_ih_b, b_hh_b) = (draw1(), draw1(), draw1(), draw1());

        gru.weight_ih_l0 = Param::from_tensor(w_ih_f.clone());
        gru.weight_hh_l0 = Param::from_tensor(w_hh_f.clone());
        gru.bias_ih_l0 = Param::from_tensor(b_ih_f.clone());
        gru.bias_hh_l0 = Param::from_tensor(b_hh_f.clone());
        gru.weight_ih_l0_reverse = Param::from_tensor(w_ih_b.clone());
        gru.weight_hh_l0_reverse = Param::from_tensor(w_hh_b.clone());
        gru.bias_ih_l0_reverse = Param::from_tensor(b_ih_b.clone());
        gru.bias_hh_l0_reverse = Param::from_tensor(b_hh_b.clone());

        let x = Tensor::<B, 3>::random([1, time, input], normal, &device);
        let got: Vec<f32> = gru.forward(x.clone()).into_data().to_vec().unwrap();

        let vec1 = |t: Tensor<B, 1>| -> Vec<f32> { t.into_data().to_vec().unwrap() };
        let vec2 = |t: Tensor<B, 2>| -> Vec<f32> { t.into_data().to_vec().unwrap() };
        let xs = vec1(x.reshape([time * input]));
        let fwd = reference(
            &xs,
            &vec2(w_ih_f),
            &vec2(w_hh_f),
            &vec1(b_ih_f),
            &vec1(b_hh_f),
            input,
            hidden,
            time,
            false,
        );
        let bwd = reference(
            &xs,
            &vec2(w_ih_b),
            &vec2(w_hh_b),
            &vec1(b_ih_b),
            &vec1(b_hh_b),
            input,
            hidden,
            time,
            true,
        );

        // Interleaved as the concatenation lays them out: forward's `hidden`
        // features then backward's, per timestep.
        let mut want = Vec::with_capacity(time * 2 * hidden);
        for t in 0..time {
            want.extend_from_slice(&fwd[t * hidden..(t + 1) * hidden]);
            want.extend_from_slice(&bwd[t * hidden..(t + 1) * hidden]);
        }
        // Finiteness first, and the comparison by hand rather than through
        // `assert_approx_eq`: that helper treats NaN as equal to NaN, so a
        // network that had gone numerically wrong would pass it silently.
        assert!(
            got.iter().all(|v| v.is_finite()),
            "GRU output must be finite"
        );
        assert_eq!(got.len(), want.len());
        let worst = got
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            worst < 1e-5,
            "gate order or bias placement is wrong: {worst}"
        );
    }

    /// The reverse direction must actually see the sequence backwards. A port
    /// that runs it forwards and stores it at the same index passes every shape
    /// check and every finiteness check; this is what separates the halves.
    #[test]
    fn the_two_directions_differ() {
        let (input, hidden, time) = (5usize, 4usize, 9usize);
        let device = Default::default();
        let normal = Distribution::Normal(0.0, 1.0);
        let mut gru = BiGru::<B>::new(input, hidden, &device);
        // The *same* weights in both directions, so the only thing that can make
        // the halves differ is the direction of travel.
        let w_ih = Tensor::<B, 2>::random([3 * hidden, input], normal, &device);
        let w_hh = Tensor::<B, 2>::random([3 * hidden, hidden], normal, &device);
        gru.weight_ih_l0 = Param::from_tensor(w_ih.clone());
        gru.weight_hh_l0 = Param::from_tensor(w_hh.clone());
        gru.weight_ih_l0_reverse = Param::from_tensor(w_ih);
        gru.weight_hh_l0_reverse = Param::from_tensor(w_hh);

        let x = Tensor::<B, 3>::random([1, time, input], normal, &device);
        let out = gru.forward(x);
        let fwd: Vec<f32> = out
            .clone()
            .narrow(2, 0, hidden)
            .into_data()
            .to_vec()
            .unwrap();
        let bwd: Vec<f32> = out.narrow(2, hidden, hidden).into_data().to_vec().unwrap();
        assert!(
            fwd.iter().zip(&bwd).any(|(a, b)| (a - b).abs() > 1e-4),
            "the reverse direction produced the forward one"
        );
    }
}

/// What `direction`'s two transposes rest on, pinned against the backend that
/// can break them.
///
/// `burn-tch`'s `swap_dims`/`permute` build their result with `TchTensor::new`,
/// which stamps a fresh `Storage::Owned` on what is still torch's view of
/// somebody else's buffer. `can_mut()` then answers true and the next
/// in-place-capable op writes straight through the view into the tensor the
/// caller is still holding — CLAUDE.md's **`swap_dims` on LibTorch returns a
/// view burn-tch forgets the provenance of**, which cost `burn-mdx` its
/// separation for weeks while every structural check stayed green.
///
/// This file is the workspace's longest exposure to that: `w_hh.transpose()` is
/// a view of a `Param::val()` clone held across *every timestep* of the loop,
/// and `w_ih.transpose()` is the same shape. Nothing about the module tree, the
/// coverage report or the scalar reference above can see a scribbled-on weight
/// — the reference runs on `ndarray`, which has no such defect. So the two
/// facts that make these lines safe are asserted here directly, on LibTorch,
/// where they are neither obvious nor free.
///
/// Gated on `--features tch` because a default `cargo test` cannot reach the
/// backend the whole module exists to exercise.
#[cfg(all(test, feature = "tch"))]
mod tch_aliasing {
    use super::*;
    use burn::tensor::{Int, TensorData};

    type Nd = burn_ndarray::NdArray;
    type Tch = burn::backend::LibTorch<f32>;

    /// Subtracting a broadcast one is in-place-capable in burn-tch whenever
    /// `can_mut()` says the buffer is exclusively owned. Returns how far the
    /// still-live `keep` moved, which is `0.0` exactly when the view respected
    /// its parent's storage.
    fn wrote_through(view: Tensor<Tch, 3>, keep: Tensor<Tch, 3>) -> f32 {
        let device = view.device();
        let _ = view - Tensor::<Tch, 3>::ones([1, 1, 1], &device);
        keep.into_data()
            .to_vec::<f32>()
            .unwrap()
            .iter()
            .map(|v| (v - 1.0).abs())
            .fold(0.0f32, f32::max)
    }

    fn live() -> (Tensor<Tch, 3>, Tensor<Tch, 3>) {
        let device = Default::default();
        let src = Tensor::<Tch, 3>::ones([1, 4, 6], &device);
        let keep = src.clone();
        (src, keep)
    }

    /// The line between the safe view ops and the dangerous ones, in one place.
    ///
    /// `slice`, `narrow`, `reshape` and `gather` go through `from_existing`,
    /// which compares data pointers and shares the parent's storage, so
    /// `can_mut()` stays false while the parent lives. `swap_dims` does not.
    /// The contrast is the whole audit rule, and it is easy to doubt from the
    /// outside — a `swap_dims` *followed by* a `reshape` writes through, which
    /// invites the reading that `reshape` is unsafe too. It is not: the second
    /// case below shows the same `reshape` is harmless on its own, so what
    /// survives the pair is the `swap_dims`'s broken provenance rather than
    /// anything the `reshape` did.
    ///
    /// **If a `wrote_through` assertion of 0.0 ever fails, that is a new
    /// unsafe op** and every audit that leaned on it is void. **If the
    /// `swap_dims` assertion fails, read it as news rather than as a
    /// regression**: burn-tch has fixed the defect upstream and this whole
    /// module has become free.
    #[test]
    fn only_the_transpose_family_loses_provenance() {
        let (src, keep) = live();
        assert_eq!(wrote_through(src.reshape([1, 6, 4]), keep), 0.0, "reshape");

        let (src, keep) = live();
        let sliced = src.slice([0..1, 0..2, 0..6]);
        assert_eq!(wrote_through(sliced, keep), 0.0, "slice");

        let (src, keep) = live();
        assert_eq!(wrote_through(src.narrow(1, 0, 2), keep), 0.0, "narrow");

        let (src, keep) = live();
        let idx = Tensor::<Tch, 3, Int>::zeros([1, 4, 6], &keep.device());
        assert_eq!(wrote_through(src.gather(1, idx), keep), 0.0, "gather");

        // And the one that does. Asserted as *not* zero, so the day burn-tch
        // fixes it this test says so out loud.
        let (src, keep) = live();
        let moved = wrote_through(src.swap_dims(1, 2), keep);
        assert!(
            moved > 0.0,
            "burn-tch's swap_dims no longer loses its parent's storage — \
             this is news, not a regression: re-read CLAUDE.md's entry and \
             the comments in this crate, `burn-whisper` and `burn-mdx` that \
             cite it"
        );
    }

    /// The property both transposes in [`BiGru::direction`] are safe *by*, and
    /// the one three further sites across the workspace rest on:
    /// `burn-whisper`'s tied vocabulary projection and its KV-cache-aliasing
    /// head views. `matmul` allocates its output and mutates neither operand,
    /// so a view with a bogus `Storage::Owned` is inert there.
    #[test]
    fn matmul_mutates_neither_operand_through_a_transposed_view() {
        let device = Default::default();

        // Right-hand side: `w_ih.transpose()`, `w_hh.transpose()`, and
        // `burn-whisper`'s `embed_tokens.weight.val().swap_dims(0, 1)`.
        let param = Tensor::<Tch, 2>::ones([6, 4], &device);
        let keep = param.clone();
        let _ = Tensor::<Tch, 2>::ones([3, 4], &device).matmul(param.transpose());
        let moved = keep
            .into_data()
            .to_vec::<f32>()
            .unwrap()
            .iter()
            .map(|v| (v - 1.0).abs())
            .fold(0.0f32, f32::max);
        assert_eq!(moved, 0.0, "matmul mutated its right-hand operand");

        // Left-hand side: `burn-whisper`'s `q`/`k`/`v` head views, which are
        // views of tensors its caller's `KvCache` is still holding.
        let (src, keep) = live();
        let _ = src
            .swap_dims(1, 2)
            .matmul(Tensor::<Tch, 3>::ones([1, 4, 2], &device));
        let moved = keep
            .into_data()
            .to_vec::<f32>()
            .unwrap()
            .iter()
            .map(|v| (v - 1.0).abs())
            .fold(0.0f32, f32::max);
        assert_eq!(moved, 0.0, "matmul mutated its left-hand operand");
    }

    /// The end-to-end reading, and the cheapest check this repository has: two
    /// independent implementations of the same arithmetic over one set of
    /// weights have to agree. The scalar reference above runs on `ndarray`
    /// alone, so it cannot see a weight LibTorch scribbled on; this can, and it
    /// exercises the transposes through the real forward pass rather than
    /// through a stand-in tensor.
    #[test]
    fn libtorch_agrees_with_ndarray_over_the_whole_layer() {
        let (input, hidden, time) = (5usize, 4usize, 7usize);
        let (nd_device, tch_device) = (Default::default(), Default::default());

        // A deterministic fill rather than `random`: the two backends do not
        // share an RNG, and the comparison is only worth anything on identical
        // weights. Zeros would not do — a zero `w_hh` makes the recurrence
        // vanish, which is the one fill that passes while saying nothing.
        let mut state = 0x2545_F491_4F6C_DD1Du64;
        let mut fill = |n: usize| -> Vec<f32> {
            (0..n)
                .map(|_| {
                    state = state
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    ((state >> 40) as f32 / 8388608.0) - 1.0
                })
                .collect()
        };

        let mut nd = BiGru::<Nd>::new(input, hidden, &nd_device);
        let mut tch = BiGru::<Tch>::new(input, hidden, &tch_device);

        let pair2 = |rows: usize, cols: usize, f: &mut dyn FnMut(usize) -> Vec<f32>| {
            let data = TensorData::new(f(rows * cols), [rows, cols]);
            (
                Tensor::<Nd, 2>::from_data(data.clone(), &nd_device),
                Tensor::<Tch, 2>::from_data(data, &tch_device),
            )
        };
        let (w_ih_f_n, w_ih_f_t) = pair2(3 * hidden, input, &mut fill);
        let (w_hh_f_n, w_hh_f_t) = pair2(3 * hidden, hidden, &mut fill);
        let (w_ih_b_n, w_ih_b_t) = pair2(3 * hidden, input, &mut fill);
        let (w_hh_b_n, w_hh_b_t) = pair2(3 * hidden, hidden, &mut fill);

        let pair1 = |f: &mut dyn FnMut(usize) -> Vec<f32>| {
            let data = TensorData::new(f(3 * hidden), [3 * hidden]);
            (
                Tensor::<Nd, 1>::from_data(data.clone(), &nd_device),
                Tensor::<Tch, 1>::from_data(data, &tch_device),
            )
        };
        let (b_ih_f_n, b_ih_f_t) = pair1(&mut fill);
        let (b_hh_f_n, b_hh_f_t) = pair1(&mut fill);
        let (b_ih_b_n, b_ih_b_t) = pair1(&mut fill);
        let (b_hh_b_n, b_hh_b_t) = pair1(&mut fill);

        nd.weight_ih_l0 = Param::from_tensor(w_ih_f_n);
        nd.weight_hh_l0 = Param::from_tensor(w_hh_f_n);
        nd.bias_ih_l0 = Param::from_tensor(b_ih_f_n);
        nd.bias_hh_l0 = Param::from_tensor(b_hh_f_n);
        nd.weight_ih_l0_reverse = Param::from_tensor(w_ih_b_n);
        nd.weight_hh_l0_reverse = Param::from_tensor(w_hh_b_n);
        nd.bias_ih_l0_reverse = Param::from_tensor(b_ih_b_n);
        nd.bias_hh_l0_reverse = Param::from_tensor(b_hh_b_n);

        tch.weight_ih_l0 = Param::from_tensor(w_ih_f_t);
        tch.weight_hh_l0 = Param::from_tensor(w_hh_f_t);
        tch.bias_ih_l0 = Param::from_tensor(b_ih_f_t);
        tch.bias_hh_l0 = Param::from_tensor(b_hh_f_t);
        tch.weight_ih_l0_reverse = Param::from_tensor(w_ih_b_t);
        tch.weight_hh_l0_reverse = Param::from_tensor(w_hh_b_t);
        tch.bias_ih_l0_reverse = Param::from_tensor(b_ih_b_t);
        tch.bias_hh_l0_reverse = Param::from_tensor(b_hh_b_t);

        let x = TensorData::new(fill(time * input), [1, time, input]);
        let want: Vec<f32> = nd
            .forward(Tensor::<Nd, 3>::from_data(x.clone(), &nd_device))
            .into_data()
            .to_vec()
            .unwrap();
        let got: Vec<f32> = tch
            .forward(Tensor::<Tch, 3>::from_data(x, &tch_device))
            .into_data()
            .to_vec()
            .unwrap();

        // Finiteness separately: Burn's `assert_approx_eq` compares NaN to NaN
        // without complaint, so a NaN would sail through the difference below.
        assert!(
            got.iter().all(|v| v.is_finite()),
            "LibTorch output must be finite"
        );
        assert_eq!(got.len(), want.len());
        let worst = want
            .iter()
            .zip(&got)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(worst < 1e-5, "the two backends differ by {worst}");
    }
}
