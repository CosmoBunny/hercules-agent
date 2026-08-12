//! Tier 2: Boundary & Corner Cases Integration Tests
//!
//! Tests kernel behavior under extreme dimensions, zero vectors, subnormals,
//! NaN/Inf propagation, truncated byte buffers, and shape mismatches.

mod common;

use common::*;
use hercules_agent::llama::gguf::GgmlType;
use hercules_agent::llama::{default_rms_norm, ComputeBackend, ComputePrefs, ParallelBackend, ScalarBackend};

#[test]
fn test_tier2_edge_shape_mismatch_x() {
    let rows = 4;
    let cols = 128;
    let raw = generate_synthetic_q8_0(rows, cols, 7001);
    let x_short = vec![1.0f32; 64]; // Should be 128
    let mut y = vec![0.0f32; rows];

    let backend = ScalarBackend::with_threads(1);
    let err = backend.gemv_quant(GgmlType::Q8_0, &raw, rows, cols, rows * cols, &x_short, &mut y);
    assert!(err.is_err(), "Expected shape mismatch error for short x vector");
}

#[test]
fn test_tier2_edge_shape_mismatch_y() {
    let rows = 4;
    let cols = 128;
    let raw = generate_synthetic_q8_0(rows, cols, 7002);
    let x = vec![1.0f32; cols];
    let mut y_short = vec![0.0f32; 2]; // Should be 4

    let backend = ScalarBackend::with_threads(1);
    let err = backend.gemv_quant(GgmlType::Q8_0, &raw, rows, cols, rows * cols, &x, &mut y_short);
    assert!(err.is_err(), "Expected shape mismatch error for short y vector");
}

#[test]
fn test_tier2_edge_truncated_raw_buffer_q8_0() {
    let rows = 2;
    let cols = 64; // 2 blocks of 34 bytes per row = 136 bytes total
    let truncated_raw = vec![0u8; 50]; // Truncated to 50 bytes
    let x = vec![1.0f32; cols];
    let mut y = vec![0.0f32; rows];

    let backend = ScalarBackend::with_threads(1);
    let err = backend.gemv_quant(GgmlType::Q8_0, &truncated_raw, rows, cols, rows * cols, &x, &mut y);
    assert!(err.is_err(), "Expected error on truncated raw buffer");
}

#[test]
fn test_tier2_edge_truncated_raw_buffer_q4_k() {
    let rows = 2;
    let cols = 256; // 1 superblock per row = 288 bytes total
    let truncated_raw = vec![0u8; 100];
    let x = vec![1.0f32; cols];
    let mut y = vec![0.0f32; rows];

    let backend = ScalarBackend::with_threads(1);
    let err = backend.gemv_quant(GgmlType::Q4_K, &truncated_raw, rows, cols, rows * cols, &x, &mut y);
    assert!(err.is_err(), "Expected error on truncated Q4_K buffer");
}

#[test]
fn test_tier2_edge_nan_input_vector() {
    let rows = 2;
    let cols = 128;
    let raw = generate_synthetic_f32(rows, cols, 7003);
    let mut x = vec![1.0f32; cols];
    x[0] = f32::NAN;
    let mut y = vec![0.0f32; rows];

    let backend = ScalarBackend::with_threads(1);
    let res = backend.gemv_quant(GgmlType::F32, &raw, rows, cols, rows * cols, &x, &mut y);
    assert!(res.is_ok());
    // Output must contain NaN without kernel panic
    assert!(y[0].is_nan());
}

#[test]
fn test_tier2_edge_inf_input_vector() {
    let rows = 2;
    let cols = 128;
    let raw = generate_synthetic_f32(rows, cols, 7004);
    let mut x = vec![1.0f32; cols];
    x[0] = f32::INFINITY;
    let mut y = vec![0.0f32; rows];

    let backend = ScalarBackend::with_threads(1);
    let res = backend.gemv_quant(GgmlType::F32, &raw, rows, cols, rows * cols, &x, &mut y);
    assert!(res.is_ok());
    assert!(y[0].is_infinite() || y[0].is_nan());
}

#[test]
fn test_tier2_edge_zero_vector_x() {
    let rows = 4;
    let cols = 256;
    let raw = generate_synthetic_q4_k(rows, cols, 7005);
    let x = vec![0.0f32; cols];
    let mut y = vec![999.0f32; rows];

    let backend = ScalarBackend::with_threads(1);
    backend
        .gemv_quant(GgmlType::Q4_K, &raw, rows, cols, rows * cols, &x, &mut y)
        .expect("Zero vector GEMV failed");

    for val in y {
        assert_eq!(val, 0.0, "Zero input vector must yield exact 0.0 output");
    }
}

#[test]
fn test_tier2_edge_single_row_gemv() {
    let rows = 1;
    let cols = 512;
    let raw = generate_synthetic_q8_0(rows, cols, 7006);
    let x = generate_synthetic_vector(cols, 7007);
    let mut y = vec![0.0f32; rows];

    let backend = ParallelBackend::new(2);
    backend
        .gemv_quant(GgmlType::Q8_0, &raw, rows, cols, rows * cols, &x, &mut y)
        .expect("Single row GEMV failed");

    let y_ref = f64_reference_gemv(&raw, GgmlType::Q8_0, rows, cols, &x);
    assert_metrics_within_tolerance(&y, &y_ref, GgmlType::Q8_0);
}

#[test]
fn test_tier2_edge_single_column_gemv() {
    let rows = 128;
    let cols = 1;
    let raw = generate_synthetic_f32(rows, cols, 7008);
    let x = vec![2.5f32];
    let mut y = vec![0.0f32; rows];

    let backend = ScalarBackend::with_threads(1);
    backend
        .gemv_quant(GgmlType::F32, &raw, rows, cols, rows * cols, &x, &mut y)
        .expect("Single column GEMV failed");

    let y_ref = f64_reference_gemv(&raw, GgmlType::F32, rows, cols, &x);
    assert_metrics_within_tolerance(&y, &y_ref, GgmlType::F32);
}

#[test]
fn test_tier2_edge_rms_norm_all_zeros() {
    let n = 128;
    let x = vec![0.0f32; n];
    let weight = vec![1.0f32; n];
    let eps = 1e-5f32;
    let mut out = vec![99.0f32; n];

    default_rms_norm(&x, &weight, eps, &mut out);

    // RMS of 0s is 0; scale is 1/sqrt(eps); 0 * scale * 1 = 0
    for val in out {
        assert_eq!(val, 0.0);
    }
}

#[test]
fn test_tier2_edge_excessive_thread_request() {
    let prefs = ComputePrefs {
        power: hercules_agent::settings::PowerMode::Extreme,
        allow_gpu: false,
        allow_simd: true,
        allow_parallel: true,
        max_threads: 1024,
    };
    let backend = hercules_agent::llama::build_default_backend(&prefs);
    assert!(backend.num_threads() >= 1);
}
