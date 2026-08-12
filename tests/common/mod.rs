//! E2E Test Suite Common Infrastructure
//!
//! Provides synthetic matrix/vector generators, double-precision `f64` reference
//! calculation oracle, mathematical metric evaluators, and synthetic GGUF model builders.

#![allow(dead_code)]

use hercules_agent::llama::gguf::{dequant_buffer, GgmlType};
use std::fs::File;
use std::io::Write;
use std::path::Path;

// ============================================================================
// 1. Half-Precision (f16) Utilities
// ============================================================================

pub fn f16_to_f32_bits(val: u16) -> f32 {
    let sign = (val >> 15) & 0x0001;
    let exp = (val >> 10) & 0x001F;
    let frac = val & 0x03FF;
    if exp == 0 {
        if frac == 0 {
            f32::from_bits((sign as u32) << 31)
        } else {
            let mut m = frac as u32;
            let mut e = 0i32;
            while (m & 0x0400) == 0 {
                m <<= 1;
                e -= 1;
            }
            let exp_f32 = ((127 - 15 + 1 + e) as u32) << 23;
            let frac_f32 = (m & 0x03FF) << 13;
            f32::from_bits(((sign as u32) << 31) | exp_f32 | frac_f32)
        }
    } else if exp == 0x1F {
        if frac == 0 {
            f32::from_bits(((sign as u32) << 31) | 0x7F800000)
        } else {
            f32::from_bits(((sign as u32) << 31) | 0x7F800000 | ((frac as u32) << 13))
        }
    } else {
        let exp_f32 = (((exp as i32) - 15 + 127) as u32) << 23;
        let frac_f32 = (frac as u32) << 13;
        f32::from_bits(((sign as u32) << 31) | exp_f32 | frac_f32)
    }
}

pub fn f32_to_f16_bits(f: f32) -> u16 {
    let bits = f.to_bits();
    let sign = (bits >> 16) & 0x8000;
    let exp = ((bits >> 23) & 0xFF) as i32 - 127 + 15;
    let frac = bits & 0x007FFFFF;
    if exp <= 0 {
        0
    } else if exp >= 31 {
        (sign as u16) | 0x7C00
    } else {
        (sign as u16) | ((exp as u16) << 10) | ((frac >> 13) as u16)
    }
}

// ============================================================================
// 2. Deterministic Synthetic Buffer Generators
// ============================================================================

/// Linear Congruential Generator for reproducible pseudo-random data.
struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u32(&mut self) -> u32 {
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.state >> 32) as u32
    }

    fn next_f32_range(&mut self, min: f32, max: f32) -> f32 {
        let norm = (self.next_u32() as f64) / (u32::MAX as f64);
        (min as f64 + norm * (max as f64 - min as f64)) as f32
    }
}

/// Generates a synthetic input vector `x` of length `n`.
pub fn generate_synthetic_vector(n: usize, seed: u64) -> Vec<f32> {
    let mut rng = Lcg::new(seed);
    (0..n).map(|_| rng.next_f32_range(-1.5, 1.5)).collect()
}

/// Generates raw `F32` matrix payload (`rows × cols × 4` bytes).
pub fn generate_synthetic_f32(rows: usize, cols: usize, seed: u64) -> Vec<u8> {
    let n = rows * cols;
    let mut rng = Lcg::new(seed);
    let mut raw = Vec::with_capacity(n * 4);
    for _ in 0..n {
        let val = rng.next_f32_range(-2.0, 2.0);
        raw.extend_from_slice(&val.to_le_bytes());
    }
    raw
}

/// Generates raw `F16` matrix payload (`rows × cols × 2` bytes).
pub fn generate_synthetic_f16(rows: usize, cols: usize, seed: u64) -> Vec<u8> {
    let n = rows * cols;
    let mut rng = Lcg::new(seed);
    let mut raw = Vec::with_capacity(n * 2);
    for _ in 0..n {
        let val = rng.next_f32_range(-2.0, 2.0);
        let u16_val = f32_to_f16_bits(val);
        raw.extend_from_slice(&u16_val.to_le_bytes());
    }
    raw
}

/// Generates raw `Q8_0` matrix payload (34 bytes per 32 weights).
pub fn generate_synthetic_q8_0(rows: usize, cols: usize, seed: u64) -> Vec<u8> {
    let n = rows * cols;
    let num_blocks = (n + 31) / 32;
    let mut rng = Lcg::new(seed);
    let mut raw = Vec::with_capacity(num_blocks * 34);

    for _ in 0..num_blocks {
        let scale = rng.next_f32_range(0.01, 0.1);
        let scale_f16 = f32_to_f16_bits(scale);
        raw.extend_from_slice(&scale_f16.to_le_bytes());
        for _ in 0..32 {
            let q = (rng.next_u32() % 255) as i8;
            raw.push(q as u8);
        }
    }
    raw
}

/// Generates raw `Q4_0` matrix payload (18 bytes per 32 weights).
pub fn generate_synthetic_q4_0(rows: usize, cols: usize, seed: u64) -> Vec<u8> {
    let n = rows * cols;
    let num_blocks = (n + 31) / 32;
    let mut rng = Lcg::new(seed);
    let mut raw = Vec::with_capacity(num_blocks * 18);

    for _ in 0..num_blocks {
        let scale = rng.next_f32_range(0.01, 0.1);
        let scale_f16 = f32_to_f16_bits(scale);
        raw.extend_from_slice(&scale_f16.to_le_bytes());
        for _ in 0..16 {
            let nib0 = (rng.next_u32() & 0x0F) as u8;
            let nib1 = (rng.next_u32() & 0x0F) as u8;
            raw.push((nib1 << 4) | nib0);
        }
    }
    raw
}

/// Generates raw `Q4_K` matrix payload (144 bytes per 256 weights).
pub fn generate_synthetic_q4_k(rows: usize, cols: usize, seed: u64) -> Vec<u8> {
    let n = rows * cols;
    let num_superblocks = (n + 255) / 256;
    let mut rng = Lcg::new(seed);
    let mut raw = Vec::with_capacity(num_superblocks * 144);

    for _ in 0..num_superblocks {
        let d = f32_to_f16_bits(rng.next_f32_range(0.005, 0.05));
        let dmin = f32_to_f16_bits(rng.next_f32_range(0.001, 0.01));
        raw.extend_from_slice(&d.to_le_bytes());
        raw.extend_from_slice(&dmin.to_le_bytes());

        // 12 bytes packed scales/mins
        for _ in 0..12 {
            raw.push((rng.next_u32() & 0xFF) as u8);
        }
        // 128 bytes qs nibbles
        for _ in 0..128 {
            raw.push((rng.next_u32() & 0xFF) as u8);
        }
    }
    raw
}

/// Generates raw `Q5_K` matrix payload (176 bytes per 256 weights).
pub fn generate_synthetic_q5_k(rows: usize, cols: usize, seed: u64) -> Vec<u8> {
    let n = rows * cols;
    let num_superblocks = (n + 255) / 256;
    let mut rng = Lcg::new(seed);
    let mut raw = Vec::with_capacity(num_superblocks * 176);

    for _ in 0..num_superblocks {
        let d = f32_to_f16_bits(rng.next_f32_range(0.005, 0.05));
        let dmin = f32_to_f16_bits(rng.next_f32_range(0.001, 0.01));
        raw.extend_from_slice(&d.to_le_bytes());
        raw.extend_from_slice(&dmin.to_le_bytes());

        // 12 bytes scales/mins
        for _ in 0..12 {
            raw.push((rng.next_u32() & 0xFF) as u8);
        }
        // 32 bytes qh
        for _ in 0..32 {
            raw.push((rng.next_u32() & 0xFF) as u8);
        }
        // 128 bytes qs nibbles
        for _ in 0..128 {
            raw.push((rng.next_u32() & 0xFF) as u8);
        }
    }
    raw
}

/// Dynamic matrix generator helper for any supported `GgmlType`.
pub fn generate_quant_matrix(rows: usize, cols: usize, ggml_type: GgmlType, seed: u64) -> Vec<u8> {
    match ggml_type {
        GgmlType::F32 => generate_synthetic_f32(rows, cols, seed),
        GgmlType::F16 => generate_synthetic_f16(rows, cols, seed),
        GgmlType::Q8_0 => generate_synthetic_q8_0(rows, cols, seed),
        GgmlType::Q4_0 => generate_synthetic_q4_0(rows, cols, seed),
        GgmlType::Q4_K => generate_synthetic_q4_k(rows, cols, seed),
        GgmlType::Q5_K => generate_synthetic_q5_k(rows, cols, seed),
        _ => generate_synthetic_f32(rows, cols, seed),
    }
}

// ============================================================================
// 3. Double-Precision (f64) Reference Math Oracle
// ============================================================================

/// Computes high-precision `f64` reference GEMV ground truth: `y = W · x`.
pub fn f64_reference_gemv(
    raw: &[u8],
    ggml_type: GgmlType,
    rows: usize,
    cols: usize,
    x: &[f32],
) -> Vec<f64> {
    let n_total = rows * cols;
    let dequant_f32 = dequant_buffer(raw, ggml_type, n_total).expect("dequant_buffer reference failed");

    let mut y_ref = vec![0.0f64; rows];
    for r in 0..rows {
        let mut sum = 0.0f64;
        for c in 0..cols {
            let idx = r * cols + c;
            if idx < dequant_f32.len() {
                let w_val = dequant_f32[idx] as f64;
                let x_val = x[c] as f64;
                sum += w_val * x_val;
            }
        }
        y_ref[r] = sum;
    }
    y_ref
}

/// Computes high-precision `f64` reference RMSNorm ground truth.
pub fn f64_reference_rms_norm(x: &[f32], weight: &[f32], eps: f32) -> Vec<f64> {
    let n = x.len().min(weight.len());
    let mut ss = 0.0f64;
    for i in 0..n {
        let xi = x[i] as f64;
        ss += xi * xi;
    }
    let scale = 1.0 / (ss / n as f64 + eps as f64).sqrt();
    let mut out = vec![0.0f64; n];
    for i in 0..n {
        out[i] = (x[i] as f64) * scale * (weight[i] as f64);
    }
    out
}

// ============================================================================
// 4. Mathematical Metric Evaluator & Tolerance Checks
// ============================================================================

#[derive(Debug, Clone, Copy)]
pub struct MetricsResult {
    pub max_abs_err: f64,
    pub max_rel_err: f64,
    pub rmse: f64,
    pub cosine_sim: f64,
}

pub fn evaluate_metrics(actual: &[f32], reference: &[f64]) -> MetricsResult {
    let n = actual.len().min(reference.len());
    if n == 0 {
        return MetricsResult {
            max_abs_err: 0.0,
            max_rel_err: 0.0,
            rmse: 0.0,
            cosine_sim: 1.0,
        };
    }

    let mut max_abs_err: f64 = 0.0;
    let mut max_rel_err: f64 = 0.0;
    let mut sum_sq_err: f64 = 0.0;
    let mut dot_prod: f64 = 0.0;
    let mut norm_actual_sq: f64 = 0.0;
    let mut norm_ref_sq: f64 = 0.0;

    for i in 0..n {
        let a = actual[i] as f64;
        let r = reference[i];
        let diff = (a - r).abs();
        if diff > max_abs_err {
            max_abs_err = diff;
        }

        let rel = diff / r.abs().max(1e-6);
        if rel > max_rel_err {
            max_rel_err = rel;
        }

        sum_sq_err += diff * diff;
        dot_prod += a * r;
        norm_actual_sq += a * a;
        norm_ref_sq += r * r;
    }

    let rmse = (sum_sq_err / n as f64).sqrt();
    let norm_product = norm_actual_sq.sqrt() * norm_ref_sq.sqrt();
    let cosine_sim = if norm_product < 1e-12 {
        1.0
    } else {
        (dot_prod / norm_product).clamp(-1.0, 1.0)
    };

    MetricsResult {
        max_abs_err,
        max_rel_err,
        rmse,
        cosine_sim,
    }
}

pub fn assert_metrics_within_tolerance(actual: &[f32], reference: &[f64], quant_type: GgmlType) {
    let metrics = evaluate_metrics(actual, reference);
    match quant_type {
        GgmlType::F32 => {
            assert!(
                metrics.max_abs_err < 1e-4 || metrics.max_rel_err < 1e-4,
                "F32 Max Error threshold breached: {:?}",
                metrics
            );
            assert!(
                metrics.cosine_sim >= 0.9999,
                "F32 Cosine Similarity threshold breached: {:?}",
                metrics
            );
        }
        GgmlType::F16 => {
            assert!(
                metrics.max_abs_err < 1e-3 || metrics.max_rel_err < 1e-3,
                "F16 Max Error threshold breached: {:?}",
                metrics
            );
            assert!(
                metrics.cosine_sim >= 0.999,
                "F16 Cosine Similarity threshold breached: {:?}",
                metrics
            );
        }
        _ => {
            // Quantized types (Q4_K, Q5_K, Q8_0, Q4_0)
            assert!(
                metrics.max_abs_err < 1e-2 || metrics.max_rel_err < 1e-2,
                "Quantized Max Error threshold breached: {:?}",
                metrics
            );
            assert!(
                metrics.cosine_sim >= 0.99,
                "Quantized Cosine Similarity threshold breached: {:?}",
                metrics
            );
        }
    }
}

// ============================================================================
// 5. Synthetic GGUF Model Builder for Tier 4 Tests
// ============================================================================

/// Creates a minimal valid GGUF model binary file for testing engine loading.
pub fn create_synthetic_gguf_file(path: &Path, vocab_size: usize, hidden_dim: usize, num_layers: usize) {
    let mut file = File::create(path).expect("Failed to create synthetic GGUF file");

    // Header: Magic "GGUF" (0x46554747), version 3, tensor_count 0, kv_count 4
    file.write_all(b"GGUF").unwrap();
    file.write_all(&3u32.to_le_bytes()).unwrap(); // Version 3
    file.write_all(&(0u64).to_le_bytes()).unwrap(); // 0 tensors for basic mock
    file.write_all(&(4u64).to_le_bytes()).unwrap(); // 4 KV metadata pairs

    // KV 1: general.architecture = "llama"
    write_gguf_kv_string(&mut file, "general.architecture", "llama");
    // KV 2: llama.embedding_length = hidden_dim
    write_gguf_kv_u32(&mut file, "llama.embedding_length", hidden_dim as u32);
    // KV 3: llama.block_count = num_layers
    write_gguf_kv_u32(&mut file, "llama.block_count", num_layers as u32);
    // KV 4: tokenizer.ggml.tokens = string array of vocab_size
    write_gguf_kv_vocab(&mut file, "tokenizer.ggml.tokens", vocab_size);

    file.flush().unwrap();
}

fn write_gguf_kv_string(file: &mut File, key: &str, val: &str) {
    // Key string
    file.write_all(&(key.len() as u64).to_le_bytes()).unwrap();
    file.write_all(key.as_bytes()).unwrap();
    // Value type string = 8
    file.write_all(&8u32.to_le_bytes()).unwrap();
    // Val string
    file.write_all(&(val.len() as u64).to_le_bytes()).unwrap();
    file.write_all(val.as_bytes()).unwrap();
}

fn write_gguf_kv_u32(file: &mut File, key: &str, val: u32) {
    file.write_all(&(key.len() as u64).to_le_bytes()).unwrap();
    file.write_all(key.as_bytes()).unwrap();
    file.write_all(&4u32.to_le_bytes()).unwrap(); // Value type u32 = 4
    file.write_all(&val.to_le_bytes()).unwrap();
}

fn write_gguf_kv_vocab(file: &mut File, key: &str, vocab_size: usize) {
    file.write_all(&(key.len() as u64).to_le_bytes()).unwrap();
    file.write_all(key.as_bytes()).unwrap();
    file.write_all(&9u32.to_le_bytes()).unwrap(); // Value type Array = 9
    file.write_all(&8u32.to_le_bytes()).unwrap(); // Array element type String = 8
    file.write_all(&(vocab_size as u64).to_le_bytes()).unwrap();

    for i in 0..vocab_size {
        let tok = format!("<tok_{}>", i);
        file.write_all(&(tok.len() as u64).to_le_bytes()).unwrap();
        file.write_all(tok.as_bytes()).unwrap();
    }
}
