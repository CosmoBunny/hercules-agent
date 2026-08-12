//! Tier 3: Cross-Feature Pairwise Interaction Integration Tests
//!
//! Tests interactions between quantization types, compute backends, thread counts,
//! fused vs fallback execution paths, and RMSNorm integration.

mod common;

use common::*;
use hercules_agent::llama::gguf::GgmlType;
use hercules_agent::llama::{default_rms_norm, ComputeBackend, ParallelBackend, ScalarBackend};

#[test]
fn test_tier3_combo_q4_k_scalar_vs_parallel() {
    let rows = 8;
    let cols = 512;
    let raw = generate_synthetic_q4_k(rows, cols, 8001);
    let x = generate_synthetic_vector(cols, 8002);

    let mut y_scalar = vec![0.0f32; rows];
    let mut y_parallel = vec![0.0f32; rows];

    let scalar = ScalarBackend::with_threads(1);
    scalar
        .gemv_quant(GgmlType::Q4_K, &raw, rows, cols, rows * cols, &x, &mut y_scalar)
        .unwrap();

    let parallel = ParallelBackend::new(4);
    parallel
        .gemv_quant(GgmlType::Q4_K, &raw, rows, cols, rows * cols, &x, &mut y_parallel)
        .unwrap();

    let y_ref = f64_reference_gemv(&raw, GgmlType::Q4_K, rows, cols, &x);
    assert_metrics_within_tolerance(&y_scalar, &y_ref, GgmlType::Q4_K);
    assert_metrics_within_tolerance(&y_parallel, &y_ref, GgmlType::Q4_K);
}

#[test]
fn test_tier3_combo_q5_k_scalar_vs_parallel() {
    let rows = 8;
    let cols = 512;
    let raw = generate_synthetic_q5_k(rows, cols, 8003);
    let x = generate_synthetic_vector(cols, 8004);

    let mut y_scalar = vec![0.0f32; rows];
    let mut y_parallel = vec![0.0f32; rows];

    let scalar = ScalarBackend::with_threads(1);
    scalar
        .gemv_quant(GgmlType::Q5_K, &raw, rows, cols, rows * cols, &x, &mut y_scalar)
        .unwrap();

    let parallel = ParallelBackend::new(4);
    parallel
        .gemv_quant(GgmlType::Q5_K, &raw, rows, cols, rows * cols, &x, &mut y_parallel)
        .unwrap();

    let y_ref = f64_reference_gemv(&raw, GgmlType::Q5_K, rows, cols, &x);
    assert_metrics_within_tolerance(&y_scalar, &y_ref, GgmlType::Q5_K);
    assert_metrics_within_tolerance(&y_parallel, &y_ref, GgmlType::Q5_K);
}

#[test]
fn test_tier3_combo_q8_0_scalar_vs_parallel() {
    let rows = 8;
    let cols = 256;
    let raw = generate_synthetic_q8_0(rows, cols, 8005);
    let x = generate_synthetic_vector(cols, 8006);

    let mut y_scalar = vec![0.0f32; rows];
    let mut y_parallel = vec![0.0f32; rows];

    let scalar = ScalarBackend::with_threads(1);
    scalar
        .gemv_quant(GgmlType::Q8_0, &raw, rows, cols, rows * cols, &x, &mut y_scalar)
        .unwrap();

    let parallel = ParallelBackend::new(4);
    parallel
        .gemv_quant(GgmlType::Q8_0, &raw, rows, cols, rows * cols, &x, &mut y_parallel)
        .unwrap();

    let y_ref = f64_reference_gemv(&raw, GgmlType::Q8_0, rows, cols, &x);
    assert_metrics_within_tolerance(&y_scalar, &y_ref, GgmlType::Q8_0);
    assert_metrics_within_tolerance(&y_parallel, &y_ref, GgmlType::Q8_0);
}

#[test]
fn test_tier3_combo_gemv_followed_by_rmsnorm() {
    let rows = 64;
    let cols = 128;
    let raw = generate_synthetic_q8_0(rows, cols, 8007);
    let x = generate_synthetic_vector(cols, 8008);
    let mut gemv_out = vec![0.0f32; rows];

    let backend = ParallelBackend::new(2);
    backend
        .gemv_quant(GgmlType::Q8_0, &raw, rows, cols, rows * cols, &x, &mut gemv_out)
        .unwrap();

    let weight = generate_synthetic_vector(rows, 8009);
    let mut norm_out = vec![0.0f32; rows];
    default_rms_norm(&gemv_out, &weight, 1e-5, &mut norm_out);

    // Verify norm output is non-empty and bounded
    assert_eq!(norm_out.len(), rows);
    for v in norm_out {
        assert!(v.is_finite());
    }
}

#[test]
fn test_tier3_combo_thread_sweep_parallel() {
    let rows = 16;
    let cols = 512;
    let raw = generate_synthetic_q4_k(rows, cols, 8010);
    let x = generate_synthetic_vector(cols, 8011);

    let thread_counts = [1, 2, 4, 8];
    let mut outputs = Vec::new();

    for &t in &thread_counts {
        let backend = ParallelBackend::new(t);
        let mut y = vec![0.0f32; rows];
        backend
            .gemv_quant(GgmlType::Q4_K, &raw, rows, cols, rows * cols, &x, &mut y)
            .unwrap();
        outputs.push(y);
    }

    // Verify outputs match across thread counts
    for i in 1..outputs.len() {
        for r in 0..rows {
            assert!(
                (outputs[0][r] - outputs[i][r]).abs() < 1e-5,
                "Thread sweep mismatch at thread_count={} row {}: {} vs {}",
                thread_counts[i],
                r,
                outputs[0][r],
                outputs[i][r]
            );
        }
    }
}
