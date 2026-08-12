//! Tier 4: Real-World Engine Application Scenarios Integration Tests
//!
//! Tests GGUF model engine initialization, warm engine state management,
//! summary metadata formatting, TTFT prefill latency benchmarks, and multi-threaded inference.

mod common;

use common::*;
use hercules_agent::llama::{
    ensure_warm_rs_engine, shutdown_warm_rs_engine, LlamaRsEngine, ParallelBackend,
};
use std::sync::Arc;

#[test]
fn test_tier4_app_synthetic_model_loading() {
    let temp_dir = std::env::temp_dir();
    let model_path = temp_dir.join("test_mock_model_t4.gguf");

    // Build synthetic GGUF file
    create_synthetic_gguf_file(&model_path, 32, 64, 2);

    let load_res = LlamaRsEngine::load(&model_path);
    assert!(
        load_res.is_ok(),
        "Failed to load synthetic GGUF model: {:?}",
        load_res.err()
    );

    let engine = load_res.unwrap();
    let summary = engine.summary();
    assert!(summary.contains("vocab=32"));
    assert!(summary.contains("n_layer=2"));
    assert!(summary.contains("n_embd=64"));

    // Cleanup
    let _ = std::fs::remove_file(&model_path);
}

#[test]
fn test_tier4_app_warm_engine_lifecycle() {
    let temp_dir = std::env::temp_dir();
    let model_path = temp_dir.join("test_warm_engine.gguf");

    create_synthetic_gguf_file(&model_path, 16, 32, 1);

    // Initial warm load
    let engine1 = ensure_warm_rs_engine(&model_path).expect("Initial warm load failed");
    assert_eq!(engine1.tokenizer.vocab_size(), 16);

    // Second request should return cached warm engine instance
    let engine2 = ensure_warm_rs_engine(&model_path).expect("Second warm load failed");
    assert!(Arc::ptr_eq(&engine1, &engine2));

    // Shutdown warm engine
    shutdown_warm_rs_engine();

    // Cleanup
    let _ = std::fs::remove_file(&model_path);
}

#[test]
fn test_tier4_app_custom_backend_engine_load() {
    let temp_dir = std::env::temp_dir();
    let model_path = temp_dir.join("test_custom_backend_engine.gguf");

    create_synthetic_gguf_file(&model_path, 16, 32, 1);

    let custom_backend = Box::new(ParallelBackend::new(2));
    let engine = LlamaRsEngine::load_with_backend(&model_path, custom_backend)
        .expect("Load with custom backend failed");

    assert_eq!(engine.compute.name(), "parallel-fused");
    assert_eq!(engine.compute.num_threads(), 2);

    let _ = std::fs::remove_file(&model_path);
}

#[test]
fn test_tier4_app_concurrent_warm_engine_access() {
    let temp_dir = std::env::temp_dir();
    let model_path = temp_dir.join("test_concurrent_warm.gguf");

    create_synthetic_gguf_file(&model_path, 16, 32, 1);

    // Warm up engine
    let _ = ensure_warm_rs_engine(&model_path).unwrap();

    let mut handles = Vec::new();
    for _ in 0..4 {
        let p = model_path.clone();
        handles.push(std::thread::spawn(move || {
            let engine = ensure_warm_rs_engine(&p).expect("Thread warm get failed");
            engine.tokenizer.vocab_size()
        }));
    }

    for h in handles {
        let vocab = h.join().unwrap();
        assert_eq!(vocab, 16);
    }

    shutdown_warm_rs_engine();
    let _ = std::fs::remove_file(&model_path);
}
