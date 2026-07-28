//! Turning `s1`'s logits into a token.
//!
//! Sampling rather than argmax, because the model is predicting one of a
//! thousand codebook entries and the greedy choice collapses into repeating a
//! single sound. The repetition penalty is the other half of that: without it a
//! run of the same token becomes self-reinforcing and the utterance never ends.

/// How adventurous generation is.
#[derive(Debug, Clone, Copy)]
pub struct SampleOptions {
    /// Keep only the `k` highest-scoring tokens. Upstream's default is 15.
    pub top_k: usize,
    /// Keep the smallest set of tokens whose probability sums past this. `1.0`
    /// disables it.
    pub top_p: f32,
    /// Below 1 sharpens the distribution, above 1 flattens it.
    pub temperature: f32,
    /// Divides the score of tokens already generated — 1.35 upstream. Applied
    /// before the cuts, so a penalised token can drop out of the top-k entirely.
    pub repetition_penalty: f32,
}

impl Default for SampleOptions {
    fn default() -> Self {
        Self {
            top_k: 15,
            top_p: 1.0,
            temperature: 1.0,
            repetition_penalty: 1.35,
        }
    }
}

/// A small deterministic generator.
///
/// Its own rather than a dependency: sampling needs uniform floats and nothing
/// else, and a seed that can be written down makes a synthesis reproducible.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed | 1)
    }

    fn next_f32(&mut self) -> f32 {
        // xorshift64*
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        let x = self.0.wrapping_mul(0x2545_F491_4F6C_DD1D);
        (x >> 40) as f32 / (1u32 << 24) as f32
    }
}

/// Pick the next token.
///
/// `previous` is every token generated so far, for the repetition penalty.
pub fn sample(logits: &[f32], previous: &[u32], opts: &SampleOptions, rng: &mut Rng) -> u32 {
    let mut scores = logits.to_vec();

    // Penalise what has already been said. Note the asymmetry, which is
    // upstream's: a positive score is divided and a negative one multiplied, so
    // both move *down*.
    if opts.repetition_penalty != 1.0 {
        for &token in previous {
            if let Some(s) = scores.get_mut(token as usize) {
                *s = if *s < 0.0 {
                    *s * opts.repetition_penalty
                } else {
                    *s / opts.repetition_penalty
                };
            }
        }
    }
    if opts.temperature > 0.0 && opts.temperature != 1.0 {
        for s in &mut scores {
            *s /= opts.temperature;
        }
    }

    let mut order: Vec<usize> = (0..scores.len()).collect();
    order.sort_unstable_by(|a, b| scores[*b].total_cmp(&scores[*a]));

    let keep = opts.top_k.clamp(1, order.len());
    let order = &order[..keep];

    // Softmax over the survivors only.
    let max = scores[order[0]];
    let mut probs: Vec<f32> = order.iter().map(|&i| (scores[i] - max).exp()).collect();
    let total: f32 = probs.iter().sum();
    for p in &mut probs {
        *p /= total;
    }

    // Nucleus: drop the tail once enough probability has accumulated, keeping at
    // least one candidate.
    let mut cutoff = probs.len();
    if opts.top_p < 1.0 {
        let mut acc = 0.0;
        for (i, p) in probs.iter().enumerate() {
            acc += p;
            if acc >= opts.top_p {
                cutoff = i + 1;
                break;
            }
        }
    }
    let (order, probs) = (&order[..cutoff], &probs[..cutoff]);
    let total: f32 = probs.iter().sum();

    let mut target = rng.next_f32() * total;
    for (&idx, &p) in order.iter().zip(probs) {
        target -= p;
        if target <= 0.0 {
            return idx as u32;
        }
    }
    order[order.len() - 1] as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn top_k_confines_the_choice() {
        // Anything outside the k best must be unreachable, however many draws.
        let logits: Vec<f32> = (0..100).map(|i| i as f32).collect();
        let opts = SampleOptions {
            top_k: 3,
            repetition_penalty: 1.0,
            ..Default::default()
        };
        let mut rng = Rng::new(7);
        for _ in 0..200 {
            let t = sample(&logits, &[], &opts, &mut rng);
            assert!((97..100).contains(&t), "sampled {t} outside the top 3");
        }
    }

    #[test]
    fn the_repetition_penalty_pushes_a_token_down_whatever_its_sign() {
        // Upstream divides positive scores and multiplies negative ones, so both
        // fall. Applying one rule to both would *raise* the negative case and
        // make repetition more likely — the opposite of the intent.
        let opts = SampleOptions {
            top_k: 2,
            repetition_penalty: 2.0,
            ..Default::default()
        };
        let mut rng = Rng::new(3);

        // Positive: 10 penalised to 5, so token 1 (score 6) should now win.
        let picks: Vec<u32> = (0..50)
            .map(|_| sample(&[10.0, 6.0, 0.0], &[0], &opts, &mut rng))
            .collect();
        assert!(picks.iter().filter(|&&t| t == 1).count() > 25);

        // Negative: -1 penalised to -2, so it stays below token 1 (-1.5).
        let picks: Vec<u32> = (0..50)
            .map(|_| sample(&[-1.0, -1.5, -9.0], &[0], &opts, &mut rng))
            .collect();
        assert!(picks.iter().filter(|&&t| t == 1).count() > 25);
    }

    #[test]
    fn sampling_is_reproducible_from_its_seed() {
        let logits: Vec<f32> = (0..50).map(|i| (i % 7) as f32).collect();
        let opts = SampleOptions::default();
        let run = || {
            let mut rng = Rng::new(42);
            (0..20)
                .map(|_| sample(&logits, &[], &opts, &mut rng))
                .collect::<Vec<_>>()
        };
        assert_eq!(run(), run());
    }
}
