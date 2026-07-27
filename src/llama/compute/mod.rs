//! Pluggable pure-Rust compute backends for llama.rs.
//!
//! **No C/FFI here.** Desktop SIMD and rayon are optional pure-Rust speedups.
//! Embedded / custom devices implement [`ComputeBackend`] themselves.

use crate::llama::gguf::GgmlType;
use crate::settings::{get_settings, PowerMode};

#[cfg(feature = "parallel")]
mod parallel;
mod scalar;

#[cfg(feature = "parallel")]
pub use parallel::ParallelBackend;
pub use scalar::ScalarBackend;

/// Error from a compute kernel.
#[derive(Debug, Clone)]
pub struct ComputeError(pub String);

impl std::fmt::Display for ComputeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for ComputeError {}

impl From<String> for ComputeError {
    fn from(s: String) -> Self {
        Self(s)
    }
}

impl From<&str> for ComputeError {
    fn from(s: &str) -> Self {
        Self(s.to_string())
    }
}

/// Preferences when selecting / building a backend.
#[derive(Debug, Clone)]
pub struct ComputePrefs {
    pub power: PowerMode,
    pub allow_simd: bool,
    pub allow_parallel: bool,
    pub max_threads: usize,
}

impl Default for ComputePrefs {
    fn default() -> Self {
        let power = get_settings().power_mode;
        let embedded = cfg!(feature = "embedded");
        Self {
            power,
            allow_simd: cfg!(feature = "simd") && !embedded,
            allow_parallel: cfg!(feature = "parallel") && !embedded,
            max_threads: if embedded {
                1
            } else {
                power.threads()
            },
        }
    }
}

impl ComputePrefs {
    pub fn from_settings() -> Self {
        Self::default()
    }

    pub fn embedded() -> Self {
        Self {
            power: PowerMode::PowerSaver,
            allow_simd: false,
            allow_parallel: false,
            max_threads: 1,
        }
    }
}

/// Matmul / elementwise primitives used by the pure-Rust decoder.
///
/// Implement this for desktop SIMD, scalar MCU, or a custom accelerator.
/// The graph in `model.rs` must not call C libraries — only this trait.
pub trait ComputeBackend: Send + Sync {
    fn name(&self) -> &str;

    /// Preferred parallel workers (1 on embedded).
    fn num_threads(&self) -> usize {
        1
    }

    /// `y = W_q · x` where `W` stays quantized in `raw`.
    fn gemv_quant(
        &self,
        quant: GgmlType,
        raw: &[u8],
        rows: usize,
        cols: usize,
        n_elements: usize,
        x: &[f32],
        y: &mut [f32],
    ) -> Result<(), ComputeError>;

    /// RMSNorm: `out = (x / rms) * w`.
    fn rms_norm(&self, x: &[f32], weight: &[f32], eps: f32, out: &mut [f32]) {
        default_rms_norm(x, weight, eps, out);
    }
}

/// Portable RMSNorm (used as default and by ScalarBackend).
pub fn default_rms_norm(x: &[f32], weight: &[f32], eps: f32, out: &mut [f32]) {
    let n = x.len().min(out.len()).min(weight.len());
    let mut ss = 0.0f32;
    for i in 0..n {
        ss += x[i] * x[i];
    }
    let scale = (ss / n as f32 + eps).sqrt().recip();
    for i in 0..n {
        out[i] = x[i] * scale * weight[i];
    }
    for i in n..out.len() {
        out[i] = 0.0;
    }
}

/// Build the best pure-Rust backend for this process / features.
///
/// Never links C/llama.cpp. Prefer parallel fused when `parallel` feature is on.
pub fn build_default_backend(prefs: &ComputePrefs) -> Box<dyn ComputeBackend> {
    let t = prefs.max_threads.max(1);
    #[cfg(feature = "parallel")]
    {
        if prefs.allow_parallel && t > 1 {
            return Box::new(ParallelBackend::new(t));
        }
    }
    let _ = prefs.allow_simd; // Phase: SIMD backends later
    Box::new(ScalarBackend::with_threads(t))
}

/// Convenience: settings-driven default backend.
pub fn default_backend() -> Box<dyn ComputeBackend> {
    build_default_backend(&ComputePrefs::from_settings())
}
