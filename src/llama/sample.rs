//! Token sampling — mirrors llama.cpp sampler chain basics:
//! temperature → top-k → top-p (nucleus) → multinomial / greedy.

use std::cmp::Ordering;

#[derive(Debug, Clone)]
pub struct SamplerParams {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub repeat_penalty: f32,
    pub seed: u64,
}

impl Default for SamplerParams {
    fn default() -> Self {
        Self {
            // Low temp: small models follow tool tags more reliably
            temperature: 0.2,
            top_k: 40,
            top_p: 0.9,
            repeat_penalty: 1.15,
            seed: 0xC0FFEE,
        }
    }
}

/// Simple xorshift64* RNG so sampling is deterministic without extra crates.
#[derive(Clone)]
pub struct Rng64 {
    state: u64,
}

impl Rng64 {
    pub fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 { 0xDEAD_BEEF_CAFE_BABE } else { seed },
        }
    }

    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    pub fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
}

#[derive(Clone)]
struct Candidate {
    id: u32,
    logit: f32,
}

/// Apply repetition penalty to tokens already present in `ctx`.
pub fn apply_repeat_penalty(logits: &mut [f32], ctx: &[u32], penalty: f32) {
    if (penalty - 1.0).abs() < 1e-6 || ctx.is_empty() {
        return;
    }
    let mut seen = vec![false; logits.len()];
    for &t in ctx {
        let i = t as usize;
        if i < seen.len() {
            seen[i] = true;
        }
    }
    for (i, logit) in logits.iter_mut().enumerate() {
        if seen[i] {
            if *logit > 0.0 {
                *logit /= penalty;
            } else {
                *logit *= penalty;
            }
        }
    }
}

/// Sample next token id from logits.
pub fn sample_token(logits: &[f32], params: &SamplerParams, rng: &mut Rng64) -> u32 {
    if logits.is_empty() {
        return 0;
    }

    // Greedy path
    if params.temperature <= 0.0 {
        return argmax(logits);
    }

    let mut cands: Vec<Candidate> = logits
        .iter()
        .enumerate()
        .map(|(i, &l)| Candidate {
            id: i as u32,
            logit: l,
        })
        .collect();

    // Temperature
    let temp = params.temperature.max(1e-5);
    for c in &mut cands {
        c.logit /= temp;
    }

    // Top-k
    if params.top_k > 0 && params.top_k < cands.len() {
        cands.select_nth_unstable_by(params.top_k, |a, b| {
            b.logit.partial_cmp(&a.logit).unwrap_or(Ordering::Equal)
        });
        cands.truncate(params.top_k);
    }

    // Softmax
    let max_logit = cands
        .iter()
        .map(|c| c.logit)
        .fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for c in &mut cands {
        c.logit = (c.logit - max_logit).exp();
        sum += c.logit;
    }
    for c in &mut cands {
        c.logit /= sum.max(1e-12);
    }

    // Top-p (nucleus)
    if params.top_p > 0.0 && params.top_p < 1.0 {
        cands.sort_by(|a, b| b.logit.partial_cmp(&a.logit).unwrap_or(Ordering::Equal));
        let mut cum = 0.0f32;
        let mut cut = cands.len();
        for (i, c) in cands.iter().enumerate() {
            cum += c.logit;
            if cum >= params.top_p {
                cut = i + 1;
                break;
            }
        }
        cands.truncate(cut.max(1));
        // renorm
        let s: f32 = cands.iter().map(|c| c.logit).sum();
        for c in &mut cands {
            c.logit /= s.max(1e-12);
        }
    }

    // Multinomial
    let r = rng.next_f32();
    let mut cum = 0.0f32;
    for c in &cands {
        cum += c.logit;
        if r <= cum {
            return c.id;
        }
    }
    cands.last().map(|c| c.id).unwrap_or(0)
}

pub fn argmax(logits: &[f32]) -> u32 {
    let mut best_i = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best_i = i;
        }
    }
    best_i as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_argmax() {
        assert_eq!(argmax(&[1.0, 5.0, 3.0]), 1);
    }

    #[test]
    fn test_sample_greedy() {
        let mut rng = Rng64::new(1);
        let params = SamplerParams {
            temperature: 0.0,
            ..Default::default()
        };
        assert_eq!(sample_token(&[0.1, 9.0, 0.2], &params, &mut rng), 1);
    }
}
