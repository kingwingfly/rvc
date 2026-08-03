//! Conditional flow matching — the sampler that drives the transformer.
//!
//! Flow matching learns a **velocity field** rather than a denoiser: the
//! transformer is asked, at time `t`, which direction the mel is moving in, and
//! sampling is then an ordinary initial-value problem — start at noise at
//! `t = 0`, integrate the field, arrive at a mel at `t = 1`. Everything in this
//! module is that integration and the guidance arithmetic around it. There are
//! **no weights here at all**: the checkpoint's `net.cfm.module.estimator.*`
//! tensors are the transformer's, and this file is only the loop that calls it.
//!
//! That is why the module is written against [`Estimator`] rather than against
//! the transformer's concrete type. The solver is pure arithmetic, so it can be
//! tested *exactly*, against fields whose flow can be written down by hand —
//! which is a stronger check than the weight-coverage harnesses the rest of the
//! crate leans on, not a weaker one.
//!
//! # The scheme is explicit Euler, and nothing better
//!
//! Upstream's `solve_euler` is named for what it is, and this port keeps it:
//! `x ← x + Δt · v(x, t)` over a uniform grid from 0 to 1. **That is first-order
//! accurate, so the error falls like `1/steps` and no faster** — doubling the
//! step count roughly halves it. Do not read the step count as if it bought a
//! Runge–Kutta rate. The tests below pin the measured behaviour: integrating
//! `dx/dt = x` from 1 towards the analytic `e`, the error is 0.2769 at 4 steps,
//! 0.1245 at 10 and 0.0267 at 50.
//!
//! The trade is therefore linear in both directions and the choice is a taste
//! one: 4–10 steps is the real-time range, 25–50 buys audible polish, and
//! upstream's own CLI defaults to 30 — which is what [`Sampler::default`] uses.
//!
//! # Classifier-free guidance
//!
//! The estimator is evaluated twice, once with the conditioning it was given and
//! once with that conditioning zeroed, and the two velocities are combined as
//!
//! ```text
//! v = (1 + w) · v_conditioned − w · v_unconditional
//! ```
//!
//! **At `w = 0` this is exactly the conditioned velocity, not the unconditional
//! one.** The scale measures how far *past* the conditioned prediction to
//! extrapolate, away from the unconditional one, so zero means "no
//! extrapolation" rather than "no conditioning". Upstream skips the second
//! evaluation entirely in that case and so does this port, which is what makes
//! the reduction exact rather than merely equal to within rounding.
//!
//! The unconditional branch is produced by **zeroing the raw conditioning** —
//! the reference mel, the timbre vector and the content stream — rather than by
//! a learned null embedding. Training's counterpart is the transformer's own
//! `class_dropout_prob: 0.1`, which zeroes the same three signals but *after*
//! the content projection has run. Those are not quite the same operation: that
//! projection carries a bias, so a zeroed content stream reaches the transformer
//! as the bias rather than as zeros. The asymmetry is upstream's, and it is
//! reproduced deliberately — the released weights are only ever driven the
//! inference way, and "fixing" it would put this port somewhere the checkpoint
//! has never been evaluated.
//!
//! # The prompt is a pinned region, not a separate input
//!
//! The reference mel occupies the leading frames of the window. Those frames of
//! the state are **held at zero for the whole integration** while the reference
//! itself is fed alongside as `prompt_x`; only the frames after it are
//! generated. This module returns the full window, prompt region included and
//! still zero, exactly as upstream does — slicing it off is the caller's job.

use burn::tensor::{Int, Tensor, backend::Backend};

/// The velocity field the sampler integrates.
///
/// One method, and it is the entire boundary between this solver and the
/// diffusion transformer: given the current state, the time and the
/// conditioning, return a velocity shaped like the state.
///
/// Shapes, matching upstream's `DiT.forward(x, prompt_x, x_lens, t, style, cond)`:
///
/// - `x` — `[batch, n_mels, frames]`, the state being integrated,
/// - `prompt_x` — `[batch, n_mels, frames]`, the reference mel in the leading
///   frames and zeros after it,
/// - `x_lens` — `[batch]`, valid frames per batch element, for the attention mask,
/// - `t` — `[batch]`, the current time in `[0, 1]`, the same value in every entry,
/// - `style` — `[batch, style_dim]`, the timbre vector, which on the released
///   inference path comes from CAMPPlus and not from this crate's
///   `style_encoder` — see that module for why the distinction matters,
/// - `cond` — `[batch, frames, hidden_dim]`, the length-regulated content stream.
///
/// The return is `[batch, n_mels, frames]`.
///
/// **Batch entries must be independent.** With guidance on, the sampler stacks
/// the conditioned and unconditional inputs into one call of twice the batch to
/// pay for a single forward pass, so an estimator that mixed across the batch
/// dimension would silently blend the two branches into each other.
pub trait Estimator<B: Backend> {
    fn velocity(
        &self,
        x: Tensor<B, 3>,
        prompt_x: Tensor<B, 3>,
        x_lens: Tensor<B, 1, Int>,
        t: Tensor<B, 1>,
        style: Tensor<B, 2>,
        cond: Tensor<B, 3>,
    ) -> Tensor<B, 3>;
}

/// Upstream's `CFM`: the Euler solver and its guidance scale.
#[derive(Debug, Clone, Copy)]
pub struct Sampler {
    /// Euler steps from noise to mel. Linear in both cost and accuracy — see the
    /// module docs.
    pub steps: usize,
    /// Classifier-free guidance scale (upstream's `inference_cfg_rate`).
    /// Anything at or below zero skips the unconditional evaluation entirely,
    /// halving the work per step — upstream's own `> 0` test, kept rather than
    /// tightened, so a negative scale is a no-op here exactly as it is there.
    pub guidance: f64,
}

impl Default for Sampler {
    /// Upstream's `inference.py` defaults, which are what a user actually runs —
    /// not the lower `inference_cfg_rate=0.5` that `BASECFM.inference` carries as
    /// a function default and that nothing calls it with.
    fn default() -> Self {
        Self {
            steps: 30,
            guidance: 0.7,
        }
    }
}

impl Sampler {
    /// Integrate from `noise` at `t = 0` to a mel at `t = 1`.
    ///
    /// `noise` is drawn by the caller rather than here, for the same reason the
    /// exported GPT-SoVITS graphs sample on the host: a solver that is a pure
    /// function of its inputs can be compared run to run and runtime to runtime.
    /// Upstream's `temperature` is a scaling of that draw, so it belongs with
    /// whoever draws it.
    ///
    /// Shapes: `noise` and the result are `[batch, n_mels, frames]`, `prompt` is
    /// `[batch, n_mels, prompt_frames]`, `cond` is `[batch, frames, hidden_dim]`,
    /// `style` is `[batch, style_dim]` and `x_lens` is `[batch]`.
    pub fn sample<B: Backend, E: Estimator<B>>(
        &self,
        estimator: &E,
        noise: Tensor<B, 3>,
        prompt: Tensor<B, 3>,
        cond: Tensor<B, 3>,
        style: Tensor<B, 2>,
        x_lens: Tensor<B, 1, Int>,
    ) -> Tensor<B, 3> {
        let device = noise.device();
        let [batch, channels, frames] = noise.dims();
        let prompt_frames = prompt.dims()[2];
        assert!(
            prompt_frames > 0 && prompt_frames < frames,
            "the reference clip is the whole speaker specification, so a \
             zero-length prompt conditions on nothing, and one filling the whole \
             {frames}-frame window leaves nothing to generate (got {prompt_frames})"
        );

        // The reference sits in the leading frames of a full-width tensor, and
        // the matching frames of the state stay zero from here to the end.
        let pinned = Tensor::zeros([batch, channels, prompt_frames], &device);
        let prompt_x = Tensor::zeros([batch, channels, frames], &device)
            .slice_assign([0..batch, 0..channels, 0..prompt_frames], prompt);
        let mut x = noise.slice_assign([0..batch, 0..channels, 0..prompt_frames], pinned.clone());

        let dt = 1.0 / self.steps as f64;
        for step in 0..self.steps {
            let t = Tensor::<B, 1>::full([batch], step as f64 * dt, &device);

            let velocity = if self.guidance > 0.0 {
                // One forward pass of twice the batch rather than two passes of
                // the batch: the branches then share every kernel launch, which
                // is most of the cost at these sequence lengths.
                let stacked = estimator.velocity(
                    Tensor::cat(vec![x.clone(), x.clone()], 0),
                    Tensor::cat(vec![prompt_x.clone(), prompt_x.zeros_like()], 0),
                    Tensor::cat(vec![x_lens.clone(), x_lens.clone()], 0),
                    Tensor::cat(vec![t.clone(), t], 0),
                    Tensor::cat(vec![style.clone(), style.zeros_like()], 0),
                    Tensor::cat(vec![cond.clone(), cond.zeros_like()], 0),
                );
                let conditioned = stacked.clone().narrow(0, 0, batch);
                let unconditional = stacked.narrow(0, batch, batch);
                conditioned.mul_scalar(1.0 + self.guidance)
                    - unconditional.mul_scalar(self.guidance)
            } else {
                estimator.velocity(
                    x.clone(),
                    prompt_x.clone(),
                    x_lens.clone(),
                    t,
                    style.clone(),
                    cond.clone(),
                )
            };

            x = (x + velocity.mul_scalar(dt))
                .slice_assign([0..batch, 0..channels, 0..prompt_frames], pinned.clone());
        }

        x
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    type B = burn_ndarray::NdArray;

    const BATCH: usize = 1;
    const CHANNELS: usize = 2;
    const FRAMES: usize = 4;
    const PROMPT: usize = 1;
    const STYLE: usize = 3;
    const HIDDEN: usize = 5;

    /// Every element finite, **reported rather than asserted**.
    ///
    /// Burn's `assert_approx_eq` compares `NaN` to `NaN` without complaint, so a
    /// silent `NaN` survives a suite that only checks values — `burn-whisper`'s
    /// tests carry the same separate check for the same reason. Returning a
    /// `bool` also lets one test below confirm that the check itself bites.
    fn all_finite(t: &Tensor<B, 3>) -> bool {
        t.clone().into_data().iter::<f32>().all(|v| v.is_finite())
    }

    fn values(t: Tensor<B, 3>) -> Vec<f32> {
        t.into_data().iter::<f32>().collect()
    }

    type Inputs = (
        Tensor<B, 3>,
        Tensor<B, 3>,
        Tensor<B, 3>,
        Tensor<B, 2>,
        Tensor<B, 1, Int>,
    );

    /// Solver inputs of the shapes above, with the initial state supplied per test.
    fn inputs(noise: f64) -> Inputs {
        let device = Default::default();
        (
            Tensor::ones([BATCH, CHANNELS, FRAMES], &device).mul_scalar(noise),
            Tensor::ones([BATCH, CHANNELS, PROMPT], &device),
            Tensor::ones([BATCH, FRAMES, HIDDEN], &device),
            Tensor::ones([BATCH, STYLE], &device),
            Tensor::from_ints([FRAMES as i32], &device),
        )
    }

    /// `dx/dt = c`. Flow: `x(1) = x(0) + c`, and Euler is exact on it at any
    /// step count.
    struct Constant(f64);

    impl<B: Backend> Estimator<B> for Constant {
        fn velocity(
            &self,
            x: Tensor<B, 3>,
            _prompt_x: Tensor<B, 3>,
            _x_lens: Tensor<B, 1, Int>,
            _t: Tensor<B, 1>,
            _style: Tensor<B, 2>,
            _cond: Tensor<B, 3>,
        ) -> Tensor<B, 3> {
            x.zeros_like().add_scalar(self.0)
        }
    }

    /// `dx/dt = x`. Flow: `x(1) = x(0)·e`, which Euler only approaches.
    struct Linear;

    impl<B: Backend> Estimator<B> for Linear {
        fn velocity(
            &self,
            x: Tensor<B, 3>,
            _prompt_x: Tensor<B, 3>,
            _x_lens: Tensor<B, 1, Int>,
            _t: Tensor<B, 1>,
            _style: Tensor<B, 2>,
            _cond: Tensor<B, 3>,
        ) -> Tensor<B, 3> {
            x
        }
    }

    /// `1 + Σstyle + Σcond + Σprompt_x`, summed **per batch element** so the two
    /// stacked guidance branches cannot leak into each other. Zeroing the
    /// conditioning drops it to a constant 1, which is what makes the guidance
    /// combination readable in closed form.
    struct ConditioningSum;

    impl<B: Backend> Estimator<B> for ConditioningSum {
        fn velocity(
            &self,
            x: Tensor<B, 3>,
            prompt_x: Tensor<B, 3>,
            _x_lens: Tensor<B, 1, Int>,
            _t: Tensor<B, 1>,
            style: Tensor<B, 2>,
            cond: Tensor<B, 3>,
        ) -> Tensor<B, 3> {
            let total = style.sum_dim(1).unsqueeze_dim::<3>(2)
                + prompt_x.sum_dim(2).sum_dim(1)
                + cond.sum_dim(2).sum_dim(1);
            x.zeros_like().add_scalar(1.0) + total
        }
    }

    /// `∞ × 0`, which is how a real `NaN` arrives — a fully masked softmax row,
    /// never a literal.
    struct NotANumber;

    impl<B: Backend> Estimator<B> for NotANumber {
        fn velocity(
            &self,
            x: Tensor<B, 3>,
            _prompt_x: Tensor<B, 3>,
            _x_lens: Tensor<B, 1, Int>,
            _t: Tensor<B, 1>,
            _style: Tensor<B, 2>,
            _cond: Tensor<B, 3>,
        ) -> Tensor<B, 3> {
            x.zeros_like().add_scalar(f64::INFINITY).mul_scalar(0.0)
        }
    }

    /// Integrating `dx/dt = 3` from `x(0) = 2` must land on 5 exactly, at every
    /// step count, because Euler is exact on a constant field. Anything else is
    /// a step size that does not sum to 1, or an off-by-one in the grid.
    #[test]
    fn a_constant_field_integrates_exactly() {
        for steps in [1, 3, 4, 30] {
            let (noise, prompt, cond, style, lens) = inputs(2.0);
            let out = Sampler {
                steps,
                guidance: 0.0,
            }
            .sample(&Constant(3.0), noise, prompt, cond, style, lens);

            assert!(all_finite(&out), "{steps} steps produced a non-finite mel");
            for (i, value) in values(out).iter().enumerate() {
                // Layout is [batch, channels, frames], so the pinned prompt is
                // the first PROMPT entries of each channel's run.
                let expected = if i % FRAMES < PROMPT { 0.0 } else { 5.0 };
                assert!(
                    (value - expected).abs() < 1e-5,
                    "{steps} steps, element {i}: {value} != {expected}"
                );
            }
        }
    }

    /// Integrating `dx/dt = x` from 1 must approach `e`, and the error must fall
    /// like `1/steps` — **first order, because the scheme is Euler**. The
    /// asserted numbers are the measured ones; a change to any of them is a
    /// change to the scheme, not a tolerance to widen.
    #[test]
    fn halving_the_step_size_halves_the_error() {
        let error = |steps: usize| {
            let (noise, prompt, cond, style, lens) = inputs(1.0);
            let out = Sampler {
                steps,
                guidance: 0.0,
            }
            .sample(&Linear, noise, prompt, cond, style, lens);
            assert!(all_finite(&out), "{steps} steps produced a non-finite mel");
            // Any generated frame will do: the field is elementwise, so they all
            // follow the same scalar ODE.
            (values(out)[PROMPT] as f64 - std::f64::consts::E).abs()
        };

        // Euler from 1 reaches (1 + 1/n)^n, so the shortfall against e is known
        // in closed form and these are it.
        for (steps, expected) in [(4, 0.276876), (10, 0.124539), (50, 0.026694)] {
            let got = error(steps);
            assert!(
                (got - expected).abs() < 1e-4,
                "{steps} steps: error {got} != {expected}"
            );
        }

        // First order means the ratio approaches 2 from below as the grid
        // refines, and never passes it. Second order would show ~4 here.
        let mut previous = error(4);
        for steps in [8, 16, 32] {
            let got = error(steps);
            let ratio = previous / got;
            assert!(
                (1.7..2.0).contains(&ratio),
                "{steps} steps: error ratio {ratio} is not the first-order ~2"
            );
            previous = got;
        }
    }

    /// Guidance zero must be the **conditioned** path — the estimator's own
    /// prediction, with no unconditional evaluation at all — and a positive
    /// scale must be the exact extrapolation away from the unconditional one.
    #[test]
    fn guidance_extrapolates_from_the_conditioned_velocity() {
        // ConditioningSum sees Σstyle = 3, Σcond = 4·5 = 20 and Σprompt_x = 2
        // (ones over 2 channels × 1 prompt frame, zero-padded to full width), so
        // the conditioned velocity is 26 and the unconditional one is 1. That
        // Σprompt_x is 2 rather than 0 is itself the check that the solver
        // placed the reference into the leading frames.
        let conditioned = 26.0;
        let unconditional = 1.0;

        // One step of Δt = 1 from a zero state, so the result *is* the velocity.
        let velocity = |guidance: f64| {
            let (noise, prompt, cond, style, lens) = inputs(0.0);
            let out = Sampler { steps: 1, guidance }.sample(
                &ConditioningSum,
                noise,
                prompt,
                cond,
                style,
                lens,
            );
            assert!(
                all_finite(&out),
                "guidance {guidance} produced a non-finite mel"
            );
            let v = values(out);
            assert!(
                v[..PROMPT].iter().all(|x| *x == 0.0),
                "the prompt region must stay pinned at zero, got {:?}",
                &v[..PROMPT]
            );
            v[PROMPT] as f64
        };

        let unguided = velocity(0.0);
        assert!(
            (unguided - conditioned).abs() < 1e-4,
            "guidance 0 must reduce to the conditioned velocity, got {unguided}"
        );

        for guidance in [0.5, 0.7, 3.0] {
            let expected = (1.0 + guidance) * conditioned - guidance * unconditional;
            let got = velocity(guidance);
            assert!(
                (got - expected).abs() < 1e-3,
                "guidance {guidance}: {got} != {expected}"
            );
        }
    }

    /// When the two branches agree — an estimator that ignores its conditioning
    /// — the combination `(1 + w)·v − w·v` must return that common velocity for
    /// every scale, with no drift out of the extrapolation arithmetic.
    #[test]
    fn guidance_is_a_no_op_when_the_branches_agree() {
        for guidance in [0.0, 0.7, 5.0] {
            let (noise, prompt, cond, style, lens) = inputs(0.0);
            let out = Sampler { steps: 1, guidance }.sample(
                &Constant(7.0),
                noise,
                prompt,
                cond,
                style,
                lens,
            );
            assert!(
                all_finite(&out),
                "guidance {guidance} produced a non-finite mel"
            );
            assert!(
                (values(out)[PROMPT] - 7.0).abs() < 1e-5,
                "guidance {guidance} moved a velocity both branches agreed on"
            );
        }
    }

    /// The finiteness check must itself detect a `NaN`, or every other test's use
    /// of it is decoration.
    #[test]
    fn the_finiteness_check_bites() {
        let (noise, prompt, cond, style, lens) = inputs(1.0);
        let out = Sampler {
            steps: 2,
            guidance: 0.7,
        }
        .sample(&NotANumber, noise, prompt, cond, style, lens);
        assert!(!all_finite(&out), "a NaN velocity must not pass as finite");
    }
}
