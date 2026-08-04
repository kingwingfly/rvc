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
