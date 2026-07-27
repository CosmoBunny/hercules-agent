//! Example: plug a custom pure-Rust [`ComputeBackend`] (e.g. for embedded).
//!
//! ```text
//! cargo run --example custom_compute_backend --features llama-rs
//! ```
//!
//! This does not load a real model; it only shows the trait surface.

use hercules_agent::llama::gguf::GgmlType;
use hercules_agent::llama::{
    build_default_backend, ComputeBackend, ComputeError, ComputePrefs, ScalarBackend,
};

/// Toy backend that logs each GEMV then defers to scalar.
struct LoggingBackend {
    inner: ScalarBackend,
    calls: std::sync::atomic::AtomicUsize,
}

impl ComputeBackend for LoggingBackend {
    fn name(&self) -> &str {
        "logging-scalar"
    }

    fn num_threads(&self) -> usize {
        1
    }

    fn gemv_quant(
        &self,
        quant: GgmlType,
        raw: &[u8],
        rows: usize,
        cols: usize,
        n_elements: usize,
        x: &[f32],
        y: &mut [f32],
    ) -> Result<(), ComputeError> {
        self.calls
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.inner
            .gemv_quant(quant, raw, rows, cols, n_elements, x, y)
    }
}

fn main() {
    let prefs = ComputePrefs::embedded();
    let default = build_default_backend(&prefs);
    println!(
        "default backend: {} threads={}",
        default.name(),
        default.num_threads()
    );

    let log = LoggingBackend {
        inner: ScalarBackend::new(),
        calls: std::sync::atomic::AtomicUsize::new(0),
    };

    // 2×2 identity F32
    let mut raw = Vec::new();
    for v in [1.0f32, 0.0, 0.0, 1.0] {
        raw.extend_from_slice(&v.to_le_bytes());
    }
    let x = [2.0f32, 5.0];
    let mut y = [0.0f32; 2];
    log.gemv_quant(GgmlType::F32, &raw, 2, 2, 4, &x, &mut y)
        .expect("gemv");
    println!("y = {:?} after {} gemv call(s)", y, log.calls.load(std::sync::atomic::Ordering::Relaxed));
    println!("Load a GGUF with: LlamaRsEngine::load_with_backend(path, Box::new(your_backend))");
}
