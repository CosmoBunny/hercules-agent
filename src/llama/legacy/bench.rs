//! Kernel-level micro-benchmarks and end-to-end throughput tests.
//!
//! ## Per-kernel benchmarks (no GGUF needed)
//! ```text
//! cargo test --release -p hercules-agent llama::bench -- --nocapture
//! ```
//!
//! ## End-to-end benchmark (requires a local GGUF)
//! ```text
//! HERCULES_TEST_GGUF=~/.local/hercules/model/qwen2.5-1.5b-instruct-q4_k_m.gguf \
//!   cargo test --release -p hercules-agent llama::bench -- --nocapture
//! ```

#[cfg(test)]
use crate::llama::compute::{build_default_backend, ComputeBackend, ComputePrefs, ScalarBackend};
#[cfg(test)]
use crate::llama::compute::simd::SimdBackend;
#[cfg(test)]
use crate::llama::legacy::infer::LlamaRsEngine;
#[cfg(test)]
use crate::llama::gguf::GgmlType;
#[cfg(test)]
use crate::llama::kernels::gemv_quant_fused;
#[cfg(test)]
use std::path::PathBuf;
#[cfg(test)]
use std::time::Instant;

#[cfg(test)]
fn test_gguf_path() -> Option<PathBuf> {
    if let Some(p) = std::env::var_os("HERCULES_TEST_GGUF") {
        let p = PathBuf::from(p);
        if p.is_file() {
            return Some(p);
        }
    }
    let home = std::env::var_os("HOME")?;
    let p = PathBuf::from(home)
        .join(".local/hercules/model/qwen2.5-1.5b-instruct-q4_k_m.gguf");
    if p.is_file() {
        Some(p)
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// Helpers to build synthetic quant blocks for benchmarking
// ---------------------------------------------------------------------------

#[cfg(test)]
/// Build N rows × cols of Q8_0 synthetic data.
fn make_q8_0(rows: usize, cols: usize) -> Vec<u8> {
    // Q8_0: 2B f16 scale + 32 i8 qs per block
    let blocks_per_row = (cols + 31) / 32;
    let block = 34;
    let mut raw = vec![0u8; rows * blocks_per_row * block];
    for (i, chunk) in raw.chunks_mut(block).enumerate() {
        // scale = 1.0 f16
        chunk[0] = 0x00;
        chunk[1] = 0x3C;
        // weights cycle 0..32
        for j in 0..32 {
            chunk[2 + j] = ((i * 32 + j) % 128) as u8;
        }
    }
    raw
}

#[cfg(test)]
/// Build N rows × cols of Q4_K synthetic data.
fn make_q4_k(rows: usize, cols: usize) -> Vec<u8> {
    // Q4_K: 144B per block of 256 elements
    let blocks_per_row = (cols + 255) / 256;
    let block = 144;
    let mut raw = vec![0u8; rows * blocks_per_row * block];
    for chunk in raw.chunks_mut(block) {
        chunk[0] = 0x00; chunk[1] = 0x3C; // d = 1.0
        chunk[2] = 0x00; chunk[3] = 0x38; // dmin = 0.5
        for i in 4..16 { chunk[i] = (i * 3) as u8; }
        for i in 16..144 { chunk[i] = (i * 11) as u8; }
    }
    raw
}

#[cfg(test)]
/// Time a closure N iterations, return (total_ns, per_iter_us).
fn time_iters(n: usize, mut f: impl FnMut()) -> (f64, f64) {
    let t0 = Instant::now();
    for _ in 0..n {
        f();
    }
    let total_ns = t0.elapsed().as_nanos() as f64;
    let per_us = total_ns / n as f64 / 1000.0;
    (total_ns, per_us)
}

#[cfg(test)]
/// Print throughput in GFLOPs (2 * rows * cols MACs per GEMV).
fn print_gflops(label: &str, rows: usize, cols: usize, iters: usize, total_ns: f64) {
    let macs = 2.0 * rows as f64 * cols as f64;
    let total_ops = macs * iters as f64;
    let gflops = total_ops / total_ns; // (GFLOPs = ops / ns)
    eprintln!("  [{label}] {rows}\u{d7}{cols}: {gflops:.3} GFLOPS");
}

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Kernel micro-benchmarks (always run; no GGUF required)
    // -----------------------------------------------------------------------

    #[test]
    fn bench_q8_0_kernels() {
        let rows = 512;
        let cols = 4096;
        let n = rows * cols;
        let raw = make_q8_0(rows, cols);
        let x: Vec<f32> = (0..cols).map(|i| i as f32 * 0.001).collect();
        let mut y = vec![0.0f32; rows];

        let iters = 10;

        // Scalar fused
        let (total_ns, per_us) = time_iters(iters, || {
            gemv_quant_fused(GgmlType::Q8_0, &raw, rows, cols, n, &x, &mut y).unwrap();
        });
        eprintln!("[bench] Q8_0 scalar-fused {rows}×{cols}: {per_us:.1} µs/iter");
        print_gflops("scalar-fused", rows, cols, iters, total_ns);

        // SIMD (if available)
        let backend = build_default_backend(&ComputePrefs::from_settings());
        eprintln!("[bench] default backend: {}", backend.name());
        let (total_ns, per_us) = time_iters(iters, || {
            backend.gemv_quant(GgmlType::Q8_0, &raw, rows, cols, n, &x, &mut y).unwrap();
        });
        eprintln!("[bench] Q8_0 {} {rows}×{cols}: {per_us:.1} µs/iter", backend.name());
        print_gflops(backend.name(), rows, cols, iters, total_ns);
    }

    #[test]
    fn bench_q4_k_kernels() {
        let rows = 256;
        let cols = 4096;
        let n = rows * cols;
        let raw = make_q4_k(rows, cols);
        let x: Vec<f32> = (0..cols).map(|i| i as f32 * 0.001).collect();
        let mut y = vec![0.0f32; rows];

        let iters = 10;

        let (total_ns, per_us) = time_iters(iters, || {
            gemv_quant_fused(GgmlType::Q4_K, &raw, rows, cols, n, &x, &mut y).unwrap();
        });
        eprintln!("[bench] Q4_K scalar-fused {rows}×{cols}: {per_us:.1} µs/iter");
        print_gflops("scalar-fused", rows, cols, iters, total_ns);

        let backend = build_default_backend(&ComputePrefs::from_settings());
        let (total_ns, per_us) = time_iters(iters, || {
            backend.gemv_quant(GgmlType::Q4_K, &raw, rows, cols, n, &x, &mut y).unwrap();
        });
        eprintln!("[bench] Q4_K {} {rows}×{cols}: {per_us:.1} µs/iter", backend.name());
        print_gflops(backend.name(), rows, cols, iters, total_ns);
    }

    #[test]
    fn bench_rms_norm() {
        let n = 4096;
        let x: Vec<f32> = (0..n).map(|i| i as f32 * 0.001 + 0.1).collect();
        let w = vec![1.0f32; n];
        let mut out = vec![0.0f32; n];
        let eps = 1e-6f32;

        let iters = 10_000;

        // Scalar
        let (_, per_us) = time_iters(iters, || {
            crate::llama::compute::default_rms_norm(&x, &w, eps, &mut out);
        });
        eprintln!("[bench] RMSNorm scalar n={n}: {per_us:.3} µs/iter");

        // SIMD backend
        let backend = build_default_backend(&ComputePrefs::from_settings());
        let (_, per_us) = time_iters(iters, || {
            backend.rms_norm(&x, &w, eps, &mut out);
        });
        eprintln!("[bench] RMSNorm {} n={n}: {per_us:.3} µs/iter", backend.name());
    }

    #[test]
    fn bench_backend_comparison() {
        let rows = 128;
        let cols = 2048;
        let n = rows * cols;
        let raw = make_q8_0(rows, cols);
        let x: Vec<f32> = (0..cols).map(|i| i as f32 * 0.001).collect();
        let mut y = vec![0.0f32; rows];
        let iters = 50;

        eprintln!("[bench] === Backend Comparison Q8_0 {rows}×{cols} ({iters} iters) ===");

        // Scalar
        {
            let b = ScalarBackend::new();
            let (total_ns, per_us) = time_iters(iters, || {
                b.gemv_quant(GgmlType::Q8_0, &raw, rows, cols, n, &x, &mut y).unwrap();
            });
            eprintln!("  [scalar-fused]   {per_us:.1} µs   ({:.2} GFLOPS)",
                2.0 * rows as f64 * cols as f64 * iters as f64 / total_ns);
        }

        // SIMD (if available)
        if SimdBackend::is_supported() {
            let b = SimdBackend::new(1);
            let (total_ns, per_us) = time_iters(iters, || {
                b.gemv_quant(GgmlType::Q8_0, &raw, rows, cols, n, &x, &mut y).unwrap();
            });
            eprintln!("  [{}]   {per_us:.1} µs   ({:.2} GFLOPS)", b.name(),
                2.0 * rows as f64 * cols as f64 * iters as f64 / total_ns);
        }

        // Default (best available)
        {
            let b = build_default_backend(&ComputePrefs::from_settings());
            let (total_ns, per_us) = time_iters(iters, || {
                b.gemv_quant(GgmlType::Q8_0, &raw, rows, cols, n, &x, &mut y).unwrap();
            });
            eprintln!("  [{}] {per_us:.1} µs   ({:.2} GFLOPS)", b.name(),
                2.0 * rows as f64 * cols as f64 * iters as f64 / total_ns);
        }
    }

    // -----------------------------------------------------------------------
    // End-to-end benchmark (requires GGUF)
    // -----------------------------------------------------------------------

    #[test]
    #[ignore = "slow end-to-end benchmark on unoptimized debug build"]
    fn baseline_decode_tok_s() {
        let Some(path) = test_gguf_path() else {
            eprintln!("[bench] skip: no GGUF (set HERCULES_TEST_GGUF)");
            return;
        };

        let prefs = ComputePrefs::from_settings();
        let backend = build_default_backend(&prefs);
        eprintln!(
            "[bench] compute={} threads={}",
            backend.name(),
            backend.num_threads()
        );

        let t0 = Instant::now();
        let engine = LlamaRsEngine::load(&path).expect("load");
        let load_ms = t0.elapsed().as_secs_f64() * 1000.0;
        eprintln!("[bench] load_ms={load_ms:.1} | {}", engine.summary());
        eprintln!("[bench] backend={}", engine.compute.name());

        let system = "You are a concise assistant.";
        let user = "Say hello in one short sentence.";

        // Warm / prefill+decode
        let t1 = Instant::now();
        let out = engine
            .generate_stream(system, user, None, None, Some(16))
            .expect("generate");
        let elapsed = t1.elapsed().as_secs_f64();
        let n_out = out.split_whitespace().count().max(1);
        // Better: use token count — approximate via chars
        let approx_tok = (out.len() / 4).max(1) as f64;
        let tok_s = approx_tok / elapsed.max(1e-6);
        eprintln!(
            "[bench] elapsed_s={elapsed:.3} approx_out_tok={approx_tok:.0} \
             decode≈{tok_s:.3} tok/s (rough) out_chars={} preview={:?}",
            out.len(),
            out.chars().take(80).collect::<String>()
        );
        let _ = n_out;
        assert!(!out.is_empty());
    }
}
