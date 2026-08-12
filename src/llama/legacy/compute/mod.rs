//! Pluggable pure-Rust compute backends for llama.rs.
//!
//! **No C/FFI here.** Desktop SIMD, rayon, and wgpu (GPU) are optional
//! pure-Rust speedups. Embedded / custom devices implement [`ComputeBackend`]
//! themselves.
//!
//! ## Backend selection priority (highest first)
//! 1. `gpu`   — wgpu GPU (Vulkan/Metal/DX12/WebGPU) when `gpu` feature is on
//!              and a suitable adapter is detected at runtime.
//! 2. `simd`  — AVX-512 / AVX2+FMA / NEON (dynamic dispatch at runtime).
//! 3. `parallel` — rayon multi-thread scalar (bandwidth-bound fallback).
//! 4. scalar  — single-thread portable (always available).

use crate::llama::gguf::GgmlType;
use crate::settings::{get_settings, PowerMode};

#[cfg(feature = "parallel")]
mod parallel;
pub mod simd;
mod scalar;
#[cfg(feature = "gpu")]
pub mod gpu;

#[cfg(feature = "parallel")]
pub use parallel::ParallelBackend;
pub use scalar::ScalarBackend;
pub use simd::SimdBackend;
#[cfg(feature = "gpu")]
pub use gpu::GpuBackend;

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
    pub allow_gpu: bool,
    pub allow_simd: bool,
    pub allow_parallel: bool,
    pub max_threads: usize,
}

impl Default for ComputePrefs {
    fn default() -> Self {
        let power = get_settings().power_mode;
        let embedded = cfg!(feature = "embedded");

        let allow_gpu = if let Ok(v) = std::env::var("HERCULES_ALLOW_GPU") {
            v != "0" && v.to_lowercase() != "false"
        } else {
            cfg!(feature = "gpu") && !embedded
        };

        let allow_simd = if let Ok(v) = std::env::var("HERCULES_ALLOW_SIMD") {
            v != "0" && v.to_lowercase() != "false"
        } else {
            cfg!(feature = "simd") && !embedded
        };

        let allow_parallel = if let Ok(v) = std::env::var("HERCULES_ALLOW_PARALLEL") {
            v != "0" && v.to_lowercase() != "false"
        } else {
            cfg!(feature = "parallel") && !embedded
        };

        let max_threads = if let Ok(v) = std::env::var("HERCULES_THREADS") {
            v.parse::<usize>().unwrap_or_else(|_| power.threads()).max(1)
        } else if embedded {
            1
        } else {
            power.threads()
        };

        Self {
            power,
            allow_gpu,
            allow_simd,
            allow_parallel,
            max_threads,
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
            allow_gpu: false,
            allow_simd: false,
            allow_parallel: false,
            max_threads: 1,
        }
    }
}

/// Matmul / elementwise primitives used by the pure-Rust decoder.
///
/// Implement this for desktop SIMD, GPU, scalar MCU, or a custom accelerator.
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

    /// Batched 2D/3D Tensor Matrix Multiplication:
    /// `Y = W_q · X` where `X` is [cols × batch_size], `Y` is [rows × batch_size].
    fn gemm_quant(
        &self,
        quant: GgmlType,
        raw: &[u8],
        rows: usize,
        cols: usize,
        n_elements: usize,
        x_batch: &[f32],
        batch_size: usize,
        y_batch: &mut [f32],
    ) -> Result<(), ComputeError> {
        for b in 0..batch_size {
            let x_slice = &x_batch[b * cols..(b + 1) * cols];
            let y_slice = &mut y_batch[b * rows..(b + 1) * rows];
            self.gemv_quant(quant, raw, rows, cols, n_elements, x_slice, y_slice)?;
        }
        Ok(())
    }

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
/// Priority: Explicit Env Override -> GPU → SIMD → Parallel → Scalar.
/// Never links C/llama.cpp.
pub fn build_default_backend(prefs: &ComputePrefs) -> Box<dyn ComputeBackend> {
    let t = prefs.max_threads.max(1);

    if let Ok(requested) = std::env::var("HERCULES_COMPUTE_BACKEND") {
        match requested.to_lowercase().trim() {
            "gpu" | "wgpu" | "vulkan" | "metal" | "dx12" => {
                #[cfg(feature = "gpu")]
                if let Some(gpu) = gpu::try_build_gpu_backend(t) {
                    return gpu;
                }
            }
            "simd" | "avx" | "avx2" | "avx512" | "neon" => {
                if SimdBackend::is_supported() {
                    return Box::new(SimdBackend::new(t));
                }
            }
            "parallel" | "rayon" => {
                #[cfg(feature = "parallel")]
                {
                    return Box::new(ParallelBackend::new(t));
                }
            }
            "scalar" => {
                return Box::new(ScalarBackend::with_threads(t));
            }
            _ => {}
        }
    }

    // 1. Multi-Threaded SIMD (Rayon + AVX2 SIMD across all CPU cores — highest throughput)
    #[cfg(feature = "parallel")]
    if prefs.allow_parallel && t > 1 {
        return Box::new(ParallelBackend::new(t));
    }

    // 2. Single-thread SIMD (AVX-512 / AVX2 / NEON)
    if prefs.allow_simd && SimdBackend::is_supported() {
        return Box::new(SimdBackend::new(t));
    }

    // 3. GPU (wgpu) — available when explicitly requested or enabled
    #[cfg(feature = "gpu")]
    if prefs.allow_gpu {
        if let Some(gpu) = gpu::try_build_gpu_backend(t) {
            return gpu;
        }
    }

    // 4. Scalar fallback (always available)
    Box::new(ScalarBackend::with_threads(t))
}

/// Convenience: settings-driven default backend.
pub fn default_backend() -> Box<dyn ComputeBackend> {
    build_default_backend(&ComputePrefs::from_settings())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_default_backend_selection() {
        let mut prefs = ComputePrefs::embedded();
        prefs.allow_simd = false;
        prefs.allow_parallel = false;
        prefs.allow_gpu = false;
        let backend = build_default_backend(&prefs);
        assert_eq!(backend.name(), "scalar-fused");

        let mut prefs_simd = ComputePrefs::embedded();
        prefs_simd.allow_simd = true;
        let backend_simd = build_default_backend(&prefs_simd);
        if SimdBackend::is_supported() {
            assert!(backend_simd.name().starts_with("simd-"));
        } else {
            assert_eq!(backend_simd.name(), "scalar-fused");
        }
    }

    #[test]
    fn test_default_rms_norm() {
        let x = [1.0f32, 2.0, 3.0, 4.0];
        let w = [1.0f32; 4];
        let mut out = [0.0f32; 4];
        default_rms_norm(&x, &w, 1e-6, &mut out);
        // All outputs should be finite and scaled.
        for &v in &out {
            assert!(v.is_finite(), "rms_norm output not finite");
        }
        // Verify monotonicity: x[3] > x[2] > x[1] > x[0] → same order in out.
        assert!(out[3] > out[2]);
        assert!(out[2] > out[1]);
        assert!(out[1] > out[0]);
    }
}
