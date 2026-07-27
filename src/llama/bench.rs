//! Baseline benchmarks for pure-Rust llama.rs (no llama.cpp).
//!
//! Run with a local GGUF:
//! ```text
//! HERCULES_TEST_GGUF=~/.local/hercules/model/qwen2.5-1.5b-instruct-q4_k_m.gguf \
//!   cargo test --release -p hercules-agent llama::bench -- --nocapture
//! ```

use crate::llama::compute::{build_default_backend, ComputePrefs};
use crate::llama::infer::LlamaRsEngine;
use std::path::PathBuf;
use std::time::Instant;

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

/// Load + short generate; prints tok/s for baseline comparison.
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
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
