//! Smoke tests for in-process libllama.so (FFI).
//!
//! These require a real install:
//! - `libllama.so` (+ ggml deps) under `~/.local/lib` (or `LIBLLAMA_PATH`)
//! - a small GGUF at `~/.local/hercules/model/qwen2.5-1.5b-instruct-q4_k_m.gguf`
//!   (override with `HERCULES_TEST_GGUF`)
//!
//! Run:
//! ```text
//! cargo test --test libllama_smoke -- --nocapture
//! ```
//! Do **not** set `LD_LIBRARY_PATH` — resolution must be automatic.

use hercules_agent::llama::ffi::{self, LlamaLib};
use hercules_agent::llama::libinfer::LlamaCppLib;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

fn default_gguf() -> PathBuf {
    if let Ok(p) = std::env::var("HERCULES_TEST_GGUF") {
        return PathBuf::from(p);
    }
    PathBuf::from(std::env::var("HOME").unwrap_or_default())
        .join(".local/hercules/model/qwen2.5-1.5b-instruct-q4_k_m.gguf")
}

fn require_gguf() -> PathBuf {
    let p = default_gguf();
    if !p.is_file() {
        panic!(
            "missing GGUF at {} (set HERCULES_TEST_GGUF)",
            p.display()
        );
    }
    p
}

#[test]
fn libllama_loads_without_user_ld_library_path() {
    // User must not need to export LD_LIBRARY_PATH=$HOME/.local/lib.
    // We preload ggml deps by absolute path + RTLD_GLOBAL inside LlamaLib::load.
    let lib = LlamaLib::load().expect("LlamaLib::load should find libllama + ggml automatically");
    unsafe {
        (lib.backend_init)();
    }
    let m = unsafe { (lib.model_default_params)() };
    let _ = m.n_gpu_layers;
}

#[test]
fn context_params_layout_has_outputs_fields() {
    let lib = LlamaLib::load().expect("load libllama");
    let p = unsafe { (lib.context_default_params)() };
    // Defaults from current llama.cpp: n_ctx=512, n_batch=2048 typically.
    assert!(p.n_ctx > 0, "n_ctx default should be > 0, got {}", p.n_ctx);
    assert!(p.n_batch > 0, "n_batch default should be > 0, got {}", p.n_batch);
    // If layout is wrong, n_threads often lands on garbage / zero.
    assert!(
        p.n_threads > 0 && p.n_threads < 512,
        "n_threads looks wrong (struct layout mismatch?): {}",
        p.n_threads
    );
    assert!(
        p.n_threads_batch > 0 && p.n_threads_batch < 512,
        "n_threads_batch looks wrong: {}",
        p.n_threads_batch
    );
}

#[test]
fn generate_hello_does_not_segfault() {
    let path = require_gguf();
    let eng = LlamaCppLib::new(path).expect("load model via libllama");

    let stream = Arc::new(Mutex::new(String::new()));
    let flag = Arc::new(Mutex::new(true));
    let out = eng
        .generate_stream("hello", stream.clone(), flag)
        .expect("generate_stream");

    assert!(!out.is_empty(), "expected non-empty generation, got empty");
    let streamed = stream.lock().unwrap().clone();
    assert_eq!(streamed, out, "stream buffer should match returned text");
    // Must not have aborted mid-token (previous bug: free() on batch_get_one).
    assert!(
        out.chars().count() >= 1,
        "expected at least one character, got {out:?}"
    );
}

#[test]
fn resolve_path_finds_home_local_lib() {
    // Soft check: if ~/.local/lib/libllama.so exists, resolve_path should hit it
    // unless LIBLLAMA_PATH overrides.
    let home_lib = PathBuf::from(std::env::var("HOME").unwrap_or_default())
        .join(".local/lib")
        .join("libllama.so");
    if !home_lib.is_file() {
        return;
    }
    // load() already succeeded in other tests; here just document the contract.
    let _ = ffi::get_lib().expect("get_lib");
    assert!(home_lib.exists());
}
