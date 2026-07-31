//! A tiny xorshift, so drawing clips and segments needs no `rand` dependency.

/// xorshift64. Deterministic from its seed, which is what makes a run
/// reproducible: two trainers given the same seed draw the same corpus order.
pub struct Rng(u64);

impl Rng {
    /// Seed the RNG. The low bit is forced because xorshift is dead at zero.
    pub fn new(seed: u64) -> Self {
        Self(seed | 1)
    }

    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    /// Uniform integer in `0..n`.
    pub fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }

    /// Uniform `f64` in `[0, 1)`.
    pub fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
}
