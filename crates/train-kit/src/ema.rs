//! An exponential moving average of a module's weights.
//!
//! The averaged copy is what gets deployed: it is markedly steadier than the
//! live weights, which bounce around under an adversarial objective.

use burn::module::{AutodiffModule, Module, ModuleMapper, ModuleVisitor, Param};
use burn::tensor::backend::{AutodiffBackend, Backend};
use burn::tensor::{Tensor, TensorPrimitive};

/// Collects a module's float parameters (in traversal order) as rank-erased
/// primitives, so a same-typed module can be blended against them element-wise.
pub struct ParamCollector<B: Backend> {
    prims: Vec<TensorPrimitive<B>>,
}

impl<B: Backend> ModuleVisitor<B> for ParamCollector<B> {
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
        self.prims.push(param.val().into_primitive());
    }
}

/// [`ModuleMapper`] applying `ema = keep*ema + (1-keep)*src`, consuming the
/// collected source primitives in the same traversal order.
pub struct EmaBlend<B: Backend> {
    prims: Vec<TensorPrimitive<B>>,
    keep: f64,
}

impl<B: Backend> ModuleMapper<B> for EmaBlend<B> {
    fn map_float<const D: usize>(&mut self, param: Param<Tensor<B, D>>) -> Param<Tensor<B, D>> {
        let (id, dst, mapper) = param.consume();
        let src = Tensor::<B, D>::from_primitive(self.prims.pop().expect("ema source underflow"));
        let blended = dst.mul_scalar(self.keep) + src.mul_scalar(1.0 - self.keep);
        Param::from_mapped_value(id, blended, mapper)
    }
}

/// Move `ema` a step toward `live`, returning the blended weights.
///
/// Pairing is **positional**: parameters are collected by `visit` and re-applied
/// by `map`, which traverse in the same order. That is why the assertion at the
/// end matters — a drift between the two would blend the wrong weights together
/// and produce a model that is subtly, silently wrong.
pub fn ema_update<AB, M>(ema: M::InnerModule, live: &M, keep: f64) -> M::InnerModule
where
    AB: AutodiffBackend,
    M: AutodiffModule<AB>,
{
    let src = live.valid();
    let mut collect = ParamCollector { prims: Vec::new() };
    src.visit(&mut collect);
    // `map` traverses in the same order as `visit`; reverse so `pop()` yields
    // the source params in that order.
    collect.prims.reverse();
    let mut blend = EmaBlend {
        prims: collect.prims,
        keep,
    };
    let blended = ema.map(&mut blend);
    // Pairing is positional, so a `map`/`visit` order drift would silently blend
    // the wrong weights together. Leftovers prove the two disagreed on the count.
    assert!(blend.prims.is_empty(), "ema source overflow");
    blended
}

/// Force Burn's lazily-initialised parameters to materialise.
///
/// Load-bearing before data-parallel training starts. Burn defers parameter
/// allocation, and two things go wrong if a module is still lazy: a clone taken
/// beforehand gets *fresh* `ParamId`s, and parameters that materialise during
/// the pass being differentiated yield no gradients at all. Either way a
/// replica's gradients stop matching the master's and are dropped in silence —
/// every extra device contributes nothing while the run looks healthy.
///
/// Warm-start and resume materialise on load; training from scratch does not.
pub fn materialize<B: Backend, M: Module<B>>(module: &M) {
    let mut collector = ParamCollector::<B> { prims: Vec::new() };
    module.visit(&mut collector);
}
