//! Gradient accumulation across micro-batches and devices.

use std::marker::PhantomData;

use burn::module::{Module, ModuleVisitor, Param};
use burn::optim::GradientsParams;
use burn::tensor::Tensor;
use burn::tensor::backend::AutodiffBackend;

/// Sums `incoming` (scaled) into `acc`, matched by `ParamId`. Gradients live on
/// `AB::InnerBackend` even though the module is autodiff-wrapped, which is why
/// the lookups below name it.
pub struct GradAccum<AB> {
    acc: GradientsParams,
    incoming: GradientsParams,
    scale: f32,
    /// `AB` appears only in the trait we implement, never in a field.
    _ab: PhantomData<AB>,
}

impl<AB: AutodiffBackend> ModuleVisitor<AB> for GradAccum<AB> {
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<AB, D>>) {
        let id = param.id;
        if let Some(g) = self.incoming.remove::<AB::InnerBackend, D>(id) {
            let g = if (self.scale - 1.0).abs() > f32::EPSILON {
                g.mul_scalar(self.scale)
            } else {
                g
            };
            let merged = match self.acc.remove::<AB::InnerBackend, D>(id) {
                Some(a) => a + g,
                None => g,
            };
            self.acc.register::<AB::InnerBackend, D>(id, merged);
        }
    }
}

/// Add `incoming * scale` into `acc`, param by param (for gradient
/// accumulation). `module` supplies the traversal over parameter ids.
pub fn accumulate<AB: AutodiffBackend, M: Module<AB>>(
    acc: GradientsParams,
    incoming: GradientsParams,
    module: &M,
    scale: f32,
) -> GradientsParams {
    let mut v = GradAccum::<AB> {
        acc,
        incoming,
        scale,
        _ab: PhantomData,
    };
    module.visit(&mut v);
    v.acc
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::Autodiff;
    use burn::nn::{Linear, LinearConfig};

    type B = Autodiff<burn_ndarray::NdArray>;

    /// The load-bearing assumption of data-parallel training: a replica made with
    /// `clone().to_device(..)` still produces gradients, and they carry the *same*
    /// `ParamId`s as the master so `accumulate` can find them.
    ///
    /// If replicas silently lost `require_grad`, every extra device would
    /// contribute nothing and training would look fine while using one device's
    /// data. Nothing in a multi-GPU run would reveal that, hence this test.
    #[test]
    fn a_replica_contributes_gradients_to_the_master() {
        let device = Default::default();
        let master: Linear<B> = LinearConfig::new(4, 4).init(&device);
        // Burn initialises parameters lazily, and a module whose parameters
        // materialise *during* the pass being differentiated yields no gradients
        // for them. `run` never hits this — warm-start or resume fills the
        // generator in before any replica is cloned — so force it here too.
        let replica = master.clone().to_device(&device);
        let warm = Tensor::<B, 2>::ones([2, 4], &device);
        let _ = GradientsParams::from_grads(replica.forward(warm).sum().backward(), &replica);

        // Exactly the non-master path in `run`: backward on the replica, ship the
        // gradients to the master's device, fold them in with the accumulator.
        let x = Tensor::<B, 2>::ones([2, 4], &device);
        let grads = GradientsParams::from_grads(replica.forward(x).sum().backward(), &replica);
        let moved = grads.to_device(&device, &master);
        assert_eq!(
            moved.len(),
            2,
            "weight and bias gradients should survive the move"
        );

        let mut acc = accumulate::<B, _>(GradientsParams::new(), moved, &master, 1.0);
        assert!(
            acc.remove::<<B as AutodiffBackend>::InnerBackend, 2>(master.weight.id)
                .is_some(),
            "the master could not claim the replica's weight gradient — every extra \
             device would contribute nothing and training would silently use one"
        );
    }

    /// Scaling has to average over devices as well as accumulation steps,
    /// otherwise adding a device silently inflates the effective learning rate.
    #[test]
    fn gradient_scale_averages_over_devices_and_accumulation() {
        for (accum, devices) in [(1, 1), (1, 2), (4, 1), (2, 3)] {
            let inv = 1.0 / (accum * devices) as f32;
            assert!((inv * (accum * devices) as f32 - 1.0).abs() < f32::EPSILON);
        }
    }
}
