//! Tier 1 & Tier 2: Mathematical Correctness Validation Framework
//!
//! Validates quantized matrix-vector multiplication (Q4_K, Q5_K, Q8_0, Q4_0, F16, F32)
//! and RMSNorm layers against f64 precision reference calculation.

mod common;

use common::*;
use hercules_agent::llama::gguf::GgmlType;
use hercules_agent::llama::{default_rms_norm, ComputeBackend, ParallelBackend, ScalarBackend};

#[test]
fn test_tier1_correctness_q8_0_scalar() {
    let rows = 4;
    let cols = 128; // 4 blocks of 32
    let raw = generate_synthetic_q8_0(rows, cols, 1001);
    let x = generate_synthetic_vector(cols, 2001);
    let mut y_actual = vec![0.0f32; rows];

    let backend = ScalarBackend::with_threads(1);
    backend
        .gemv_quant(GgmlType::Q8_0, &raw, rows, cols, rows * cols, &x, &mut y_actual)
        .expect("Q8_0 Scalar GEMV failed");

    let y_ref = f64_reference_gemv(&raw, GgmlType::Q8_0, rows, cols, &x);
    assert_metrics_within_tolerance(&y_actual, &y_ref, GgmlType::Q8_0);
}

#[test]
fn test_tier1_correctness_q4_0_scalar() {
    let rows = 4;
    let cols = 128;
    let raw = generate_synthetic_q4_0(rows, cols, 1002);
    let x = generate_synthetic_vector(cols, 2002);
    let mut y_actual = vec![0.0f32; rows];

    let backend = ScalarBackend::with_threads(1);
    backend
        .gemv_quant(GgmlType::Q4_0, &raw, rows, cols, rows * cols, &x, &mut y_actual)
        .expect("Q4_0 Scalar GEMV failed");

    let y_ref = f64_reference_gemv(&raw, GgmlType::Q4_0, rows, cols, &x);
    assert_metrics_within_tolerance(&y_actual, &y_ref, GgmlType::Q4_0);
}

#[test]
fn test_tier1_correctness_q4_k_scalar() {
    let rows = 4;
    let cols = 256; // 1 superblock per row
    let raw = generate_synthetic_q4_k(rows, cols, 1003);
    let x = generate_synthetic_vector(cols, 2003);
    let mut y_actual = vec![0.0f32; rows];

    let backend = ScalarBackend::with_threads(1);
    backend
        .gemv_quant(GgmlType::Q4_K, &raw, rows, cols, rows * cols, &x, &mut y_actual)
        .expect("Q4_K Scalar GEMV failed");

    let y_ref = f64_reference_gemv(&raw, GgmlType::Q4_K, rows, cols, &x);
    assert_metrics_within_tolerance(&y_actual, &y_ref, GgmlType::Q4_K);
}

#[test]
fn test_tier1_correctness_q5_k_scalar() {
    let rows = 4;
    let cols = 256; // 1 superblock per row
    let raw = generate_synthetic_q5_k(rows, cols, 1004);
    let x = generate_synthetic_vector(cols, 2004);
    let mut y_actual = vec![0.0f32; rows];

    let backend = ScalarBackend::with_threads(1);
    backend
        .gemv_quant(GgmlType::Q5_K, &raw, rows, cols, rows * cols, &x, &mut y_actual)
        .expect("Q5_K Scalar GEMV failed");

    let y_ref = f64_reference_gemv(&raw, GgmlType::Q5_K, rows, cols, &x);
    assert_metrics_within_tolerance(&y_actual, &y_ref, GgmlType::Q5_K);
}

#[test]
fn test_tier1_correctness_f16_scalar() {
    let rows = 4;
    let cols = 128;
    let raw = generate_synthetic_f16(rows, cols, 1005);
    let x = generate_synthetic_vector(cols, 2005);
    let mut y_actual = vec![0.0f32; rows];

    let backend = ScalarBackend::with_threads(1);
    backend
        .gemv_quant(GgmlType::F16, &raw, rows, cols, rows * cols, &x, &mut y_actual)
        .expect("F16 Scalar GEMV failed");

    let y_ref = f64_reference_gemv(&raw, GgmlType::F16, rows, cols, &x);
    assert_metrics_within_tolerance(&y_actual, &y_ref, GgmlType::F16);
}

#[test]
fn test_tier1_correctness_f32_scalar() {
    let rows = 4;
    let cols = 128;
    let raw = generate_synthetic_f32(rows, cols, 1006);
    let x = generate_synthetic_vector(cols, 2006);
    let mut y_actual = vec![0.0f32; rows];

    let backend = ScalarBackend::with_threads(1);
    backend
        .gemv_quant(GgmlType::F32, &raw, rows, cols, rows * cols, &x, &mut y_actual)
        .expect("F32 Scalar GEMV failed");

    let y_ref = f64_reference_gemv(&raw, GgmlType::F32, rows, cols, &x);
    assert_metrics_within_tolerance(&y_actual, &y_ref, GgmlType::F32);
}

#[test]
fn test_tier1_correctness_q8_0_parallel() {
    let rows = 8;
    let cols = 256;
    let raw = generate_synthetic_q8_0(rows, cols, 1007);
    let x = generate_synthetic_vector(cols, 2007);
    let mut y_actual = vec![0.0f32; rows];

    let backend = ParallelBackend::new(4);
    backend
        .gemv_quant(GgmlType::Q8_0, &raw, rows, cols, rows * cols, &x, &mut y_actual)
        .expect("Q8_0 Parallel GEMV failed");

    let y_ref = f64_reference_gemv(&raw, GgmlType::Q8_0, rows, cols, &x);
    assert_metrics_within_tolerance(&y_actual, &y_ref, GgmlType::Q8_0);
}

#[test]
fn test_tier1_correctness_q4_k_parallel() {
    let rows = 8;
    let cols = 512;
    let raw = generate_synthetic_q4_k(rows, cols, 1008);
    let x = generate_synthetic_vector(cols, 2008);
    let mut y_actual = vec![0.0f32; rows];

    let backend = ParallelBackend::new(4);
    backend
        .gemv_quant(GgmlType::Q4_K, &raw, rows, cols, rows * cols, &x, &mut y_actual)
        .expect("Q4_K Parallel GEMV failed");

    let y_ref = f64_reference_gemv(&raw, GgmlType::Q4_K, rows, cols, &x);
    assert_metrics_within_tolerance(&y_actual, &y_ref, GgmlType::Q4_K);
}

#[test]
fn test_tier1_correctness_q5_k_parallel() {
    let rows = 8;
    let cols = 512;
    let raw = generate_synthetic_q5_k(rows, cols, 1009);
    let x = generate_synthetic_vector(cols, 2009);
    let mut y_actual = vec![0.0f32; rows];

    let backend = ParallelBackend::new(4);
    backend
        .gemv_quant(GgmlType::Q5_K, &raw, rows, cols, rows * cols, &x, &mut y_actual)
        .expect("Q5_K Parallel GEMV failed");

    let y_ref = f64_reference_gemv(&raw, GgmlType::Q5_K, rows, cols, &x);
    assert_metrics_within_tolerance(&y_actual, &y_ref, GgmlType::Q5_K);
}

#[test]
fn test_tier1_correctness_rms_norm_scalar() {
    let n = 256;
    let x = generate_synthetic_vector(n, 3001);
    let weight = generate_synthetic_vector(n, 3002);
    let eps = 1e-5f32;
    let mut out_actual = vec![0.0f32; n];

    default_rms_norm(&x, &weight, eps, &mut out_actual);
    let ref_out = f64_reference_rms_norm(&x, &weight, eps);

    assert_metrics_within_tolerance(&out_actual, &ref_out, GgmlType::F32);
}

#[test]
fn test_tier2_correctness_single_block_q8_0() {
    let rows = 1;
    let cols = 32; // Exactly 1 block
    let raw = generate_synthetic_q8_0(rows, cols, 4001);
    let x = generate_synthetic_vector(cols, 4002);
    let mut y_actual = vec![0.0f32; rows];

    let backend = ScalarBackend::with_threads(1);
    backend
        .gemv_quant(GgmlType::Q8_0, &raw, rows, cols, rows * cols, &x, &mut y_actual)
        .expect("Single block Q8_0 GEMV failed");

    let y_ref = f64_reference_gemv(&raw, GgmlType::Q8_0, rows, cols, &x);
    assert_metrics_within_tolerance(&y_actual, &y_ref, GgmlType::Q8_0);
}

#[test]
fn test_tier2_correctness_single_block_q4_k() {
    let rows = 1;
    let cols = 256; // Exactly 1 superblock
    let raw = generate_synthetic_q4_k(rows, cols, 4003);
    let x = generate_synthetic_vector(cols, 4004);
    let mut y_actual = vec![0.0f32; rows];

    let backend = ScalarBackend::with_threads(1);
    backend
        .gemv_quant(GgmlType::Q4_K, &raw, rows, cols, rows * cols, &x, &mut y_actual)
        .expect("Single block Q4_K GEMV failed");

    let y_ref = f64_reference_gemv(&raw, GgmlType::Q4_K, rows, cols, &x);
    assert_metrics_within_tolerance(&y_actual, &y_ref, GgmlType::Q4_K);
}

#[test]
fn test_tier2_correctness_large_matrix_q4_k() {
    let rows = 16;
    let cols = 1024;
    let raw = generate_synthetic_q4_k(rows, cols, 4005);
    let x = generate_synthetic_vector(cols, 4006);
    let mut y_actual = vec![0.0f32; rows];

    let backend = ScalarBackend::with_threads(1);
    backend
        .gemv_quant(GgmlType::Q4_K, &raw, rows, cols, rows * cols, &x, &mut y_actual)
        .expect("Large matrix Q4_K GEMV failed");

    let y_ref = f64_reference_gemv(&raw, GgmlType::Q4_K, rows, cols, &x);
    assert_metrics_within_tolerance(&y_actual, &y_ref, GgmlType::Q4_K);
}
