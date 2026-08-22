//! Tier 1 & Tier 3: ComputeBackend Lifecycle & Fallback Verification
//!
//! Tests backend selection hierarchy, thread pool scaling, concurrent execution,
//! and fallback mechanisms across Scalar, Parallel, and default backends.

mod common;

use common::*;
use hercules_agent::llama::gguf::GgmlType;
use hercules_agent::llama::{
    build_default_backend, ComputeBackend, ComputePrefs, ParallelBackend, ScalarBackend,
};
use hercules_agent::settings::PowerMode;
use std::sync::Arc;
use std::thread;

#[test]
fn test_tier1_backend_name_scalar() {
    let backend = ScalarBackend::with_threads(1);
    assert_eq!(backend.name(), "scalar-fused");
}

#[test]
fn test_tier1_backend_name_parallel() {
    let backend = ParallelBackend::new(4);
    assert!(backend.name() == "parallel-fused" || backend.name() == "parallel-simd");
}

#[test]
fn test_tier1_backend_num_threads() {
    let scalar = ScalarBackend::with_threads(2);
    assert_eq!(scalar.num_threads(), 2);

    let parallel = ParallelBackend::new(8);
    assert_eq!(parallel.num_threads(), 8);
}

#[test]
fn test_tier1_build_default_backend_embedded() {
    let prefs = ComputePrefs::embedded();
    let backend = build_default_backend(&prefs);
    assert!(!backend.name().is_empty());
    assert_eq!(backend.num_threads(), 1);
}

#[test]
fn test_tier1_build_default_backend_custom_prefs() {
    let prefs = ComputePrefs {
        power: PowerMode::Normal,
        allow_gpu: false,
        allow_simd: true,
        allow_parallel: true,
        max_threads: 4,
    };
    let backend = build_default_backend(&prefs);
    assert!(!backend.name().is_empty());
}

#[test]
fn test_tier3_thread_scaling_parallel() {
    let rows = 8;
    let cols = 256;
    let raw = generate_synthetic_q4_k(rows, cols, 5001);
    let x = generate_synthetic_vector(cols, 5002);

    let mut y_1thread = vec![0.0f32; rows];
    let mut y_4thread = vec![0.0f32; rows];

    let b1 = ParallelBackend::new(1);
    b1.gemv_quant(GgmlType::Q4_K, &raw, rows, cols, rows * cols, &x, &mut y_1thread)
        .unwrap();

    let b4 = ParallelBackend::new(4);
    b4.gemv_quant(GgmlType::Q4_K, &raw, rows, cols, rows * cols, &x, &mut y_4thread)
        .unwrap();

    // Verify 1 thread and 4 threads produce bitwise or within-tolerance identical outputs
    for r in 0..rows {
        assert!(
            (y_1thread[r] - y_4thread[r]).abs() < 1e-5,
            "Thread scaling output mismatch at row {}: t1={} t4={}",
            r,
            y_1thread[r],
            y_4thread[r]
        );
    }
}

#[test]
fn test_tier3_backend_fallback_unsupported_type() {
    // Q3_K fallback to full dequant buffer path inside backend
    let rows = 2;
    let cols = 256;
    // Generate dummy buffer of Q3_K size (110 bytes per superblock)
    let raw = vec![0u8; 110 * 2];
    let x = vec![1.0f32; cols];
    let mut y = vec![0.0f32; rows];

    let backend = ScalarBackend::with_threads(1);
    let res = backend.gemv_quant(GgmlType::Q3_K, &raw, rows, cols, rows * cols, &x, &mut y);
    // Should return result cleanly without panic
    assert!(res.is_ok());
}

#[test]
fn test_tier3_backend_concurrent_gemv() {
    let rows = 4;
    let cols = 128;
    let raw = Arc::new(generate_synthetic_q8_0(rows, cols, 6001));
    let x = Arc::new(generate_synthetic_vector(cols, 6002));
    let backend: Arc<Box<dyn ComputeBackend>> = Arc::new(build_default_backend(&ComputePrefs::default()));

    let mut handles = Vec::new();
    for _ in 0..4 {
        let raw_c = Arc::clone(&raw);
        let x_c = Arc::clone(&x);
        let b_c = Arc::clone(&backend);
        handles.push(thread::spawn(move || {
            let mut y = vec![0.0f32; rows];
            b_c.gemv_quant(GgmlType::Q8_0, &raw_c, rows, cols, rows * cols, &x_c, &mut y)
                .expect("Concurrent GEMV failed");
            y
        }));
    }

    let mut results = Vec::new();
    for h in handles {
        results.push(h.join().unwrap());
    }

    // All threads must return identical output
    for i in 1..results.len() {
        assert_eq!(results[0], results[i]);
    }
}
