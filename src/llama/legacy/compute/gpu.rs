//! GPU compute backend for llama.rs via wgpu (cross-platform: Vulkan/Metal/DX12/WebGPU).
//!
//! ## Design
//! - Uses `burn` 0.21 with the `wgpu` backend for GPU-accelerated tensor ops.
//! - GEMV is implemented as a batched matrix-vector multiply via burn's `Tensor` API.
//! - Weights are kept **quantized on CPU** (same as SIMD path); dequantization happens
//!   in a thin host-side buffer before uploading to GPU for each GEMV call.
//!   This means GPU throughput kicks in only for large matrices where the upload
//!   cost is amortized — for tiny matrices the CPU SIMD path is faster.
//! - Falls back to CPU if wgpu device init fails (e.g. headless CI, embedded).
//!
//! ## Feature flag
//! Enabled by default via the `burn` dep in Cargo.toml. GPU path is only taken
//! when `GpuBackend::try_new()` succeeds (GPU device present).
//!
//! ## Thread safety
//! `burn` tensor handles are `Send + Sync`; `GpuBackend` is `Send + Sync`.

use crate::llama::gguf::{dequant_buffer, GgmlType};
use crate::llama::compute::{ComputeBackend, ComputeError, default_rms_norm};
use crate::llama::kernels::gemv_quant_fused;

// -------------------------------------------------------------------
// burn / wgpu imports — only when the crate is available
// -------------------------------------------------------------------
use burn::backend::Wgpu;
use burn::prelude::*;
use burn::tensor::Tensor;

type B = Wgpu;

use std::collections::HashMap;
use std::sync::Mutex;

/// Minimum number of output rows to prefer GPU GEMV over CPU.
/// Below this, PCIe/shared-memory upload latency dominates.
const GPU_THRESHOLD_ROWS: usize = 256;

/// GPU compute backend (wgpu via burn).
///
/// Use [`GpuBackend::try_new`] to construct — returns `None` if no GPU
/// device is available (so the caller can fall back to SIMD).
pub struct GpuBackend {
    device: burn::backend::wgpu::WgpuDevice,
    threads: usize,
    vram_cache: Mutex<HashMap<usize, Tensor<B, 2>>>,
}

impl std::fmt::Debug for GpuBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuBackend")
            .field("threads", &self.threads)
            .finish()
    }
}

impl GpuBackend {
    /// Try to initialise a wgpu device (GPU).
    ///
    /// Returns `None` when no compatible GPU adapter is available (headless,
    /// embedded, etc.) so callers can gracefully fall back to SIMD/scalar.
    pub fn try_new(threads: usize) -> Option<Self> {
        // `WgpuDevice::default()` picks the best available adapter at runtime.
        // On systems with no Vulkan/Metal/DX12/WebGPU it returns an error.
        let device = burn::backend::wgpu::WgpuDevice::default();
        // Probe by allocating a tiny tensor — if this fails there is no GPU.
        let probe = std::panic::catch_unwind(|| {
            let _t: Tensor<B, 1> = Tensor::zeros([4], &device);
        });
        if probe.is_err() {
            return None;
        }
        Some(Self {
            device,
            threads: threads.max(1),
            vram_cache: Mutex::new(HashMap::new()),
        })
    }

    /// GPU GEMV with VRAM Tensor Caching:
    /// Uploads weight matrix `W` to GPU VRAM once on token 1, then reuses `W` in VRAM
    /// for all subsequent tokens using WGSL compute shader `w.matmul(xv)`.
    fn gemv_gpu_cached(
        &self,
        raw: &[u8],
        quant: GgmlType,
        rows: usize,
        cols: usize,
        n_elements: usize,
        x: &[f32],
    ) -> Result<Vec<f32>, ComputeError> {
        let key = raw.as_ptr() as usize;
        let dev = &self.device;

        let w = {
            let mut cache = self
                .vram_cache
                .lock()
                .map_err(|e| ComputeError(format!("VRAM cache lock: {}", e)))?;

            if let Some(existing) = cache.get(&key) {
                existing.clone()
            } else {
                let weights_f32 = dequant_buffer(raw, quant, n_elements)
                    .map_err(|e| ComputeError(e.to_string()))?;
                if weights_f32.len() < rows * cols {
                    return Err(ComputeError(format!(
                        "GPU gemv: weight buffer {} < rows*cols {}",
                        weights_f32.len(),
                        rows * cols
                    )));
                }
                let tensor: Tensor<B, 2> = Tensor::from_floats(
                    burn::tensor::TensorData::new(
                        weights_f32[..rows * cols].to_vec(),
                        [rows, cols],
                    ),
                    dev,
                );
                cache.insert(key, tensor.clone());
                tensor
            }
        };

        // Upload only vector x [cols, 1] (6 KB)
        let xv: Tensor<B, 2> = Tensor::from_floats(
            burn::tensor::TensorData::new(x[..cols].to_vec(), [cols, 1]),
            dev,
        );

        // Compute matmul W @ x on GPU via WGPU WGSL compute shader
        let out: Tensor<B, 2> = w.matmul(xv);

        // Read back output y [rows, 1] (6 KB)
        let data = out.into_data();
        let flat: Vec<f32> = data
            .to_vec::<f32>()
            .map_err(|e| ComputeError(format!("GPU read-back: {:?}", e)))?;
        Ok(flat)
    }

    /// 2D/3D Batched Tensor GPU GEMM:
    /// Computes W [rows × cols] @ X_batch [cols × batch_size] -> Y [rows × batch_size]
    /// in a single WGPU compute shader pipeline execution.
    fn gemm_gpu_cached(
        &self,
        raw: &[u8],
        quant: GgmlType,
        rows: usize,
        cols: usize,
        n_elements: usize,
        x_batch: &[f32],
        batch_size: usize,
    ) -> Result<Vec<f32>, ComputeError> {
        let key = raw.as_ptr() as usize;
        let dev = &self.device;

        let w = {
            let mut cache = self
                .vram_cache
                .lock()
                .map_err(|e| ComputeError(format!("VRAM cache lock: {}", e)))?;

            if let Some(existing) = cache.get(&key) {
                existing.clone()
            } else {
                let weights_f32 = dequant_buffer(raw, quant, n_elements)
                    .map_err(|e| ComputeError(e.to_string()))?;
                if weights_f32.len() < rows * cols {
                    return Err(ComputeError(format!(
                        "GPU gemm: weight buffer {} < rows*cols {}",
                        weights_f32.len(),
                        rows * cols
                    )));
                }
                let tensor: Tensor<B, 2> = Tensor::from_floats(
                    burn::tensor::TensorData::new(
                        weights_f32[..rows * cols].to_vec(),
                        [rows, cols],
                    ),
                    dev,
                );
                cache.insert(key, tensor.clone());
                tensor
            }
        };

        if x_batch.len() < batch_size * cols {
            return Err(ComputeError(format!(
                "GPU gemm: x_batch len {} < batch_size * cols {}",
                x_batch.len(),
                batch_size * cols
            )));
        }

        // X_batch tensor [batch_size, cols]
        let xv: Tensor<B, 2> = Tensor::from_floats(
            burn::tensor::TensorData::new(
                x_batch[..batch_size * cols].to_vec(),
                [batch_size, cols],
            ),
            dev,
        );

        // Y [batch_size, rows] = X [batch_size, cols] @ W^T [cols, rows]
        let out: Tensor<B, 2> = xv.matmul(w.transpose());

        let data = out.into_data();
        let flat: Vec<f32> = data
            .to_vec::<f32>()
            .map_err(|e| ComputeError(format!("GPU batched read-back: {:?}", e)))?;
        Ok(flat)
    }
}

impl ComputeBackend for GpuBackend {
    fn name(&self) -> &str {
        "gpu-wgpu"
    }

    fn num_threads(&self) -> usize {
        self.threads
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
        gemv_quant_fused(quant, raw, rows, cols, n_elements, x, y)
            .map_err(ComputeError)
    }

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
        if batch_size <= 1 {
            let x_slice = &x_batch[..cols.min(x_batch.len())];
            return self.gemv_quant(quant, raw, rows, cols, n_elements, x_slice, y_batch);
        }

        let out = self.gemm_gpu_cached(raw, quant, rows, cols, n_elements, x_batch, batch_size)?;
        let take = out.len().min(y_batch.len());
        y_batch[..take].copy_from_slice(&out[..take]);
        Ok(())
    }

    /// RMSNorm on CPU (fast SIMD math, avoids CPU-GPU pipeline fence stalls).
    fn rms_norm(&self, x: &[f32], weight: &[f32], eps: f32, out: &mut [f32]) {
        default_rms_norm(x, weight, eps, out);
    }
}

// ---------------------------------------------------------------------------
// Send + Sync — burn::backend::Wgpu types are Send but not always Sync via
// their internal Arc-ed state. We gate on the probe above ensuring a live GPU.
// ---------------------------------------------------------------------------
unsafe impl Send for GpuBackend {}
unsafe impl Sync for GpuBackend {}

// ---------------------------------------------------------------------------
// Build helper
// ---------------------------------------------------------------------------

/// Build a GPU backend, or return None if no GPU is available.
pub fn try_build_gpu_backend(threads: usize) -> Option<Box<dyn ComputeBackend>> {
    GpuBackend::try_new(threads).map(|b| Box::new(b) as Box<dyn ComputeBackend>)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_backend_init() {
        // If no GPU available, skip gracefully (don't panic).
        match GpuBackend::try_new(1) {
            Some(gpu) => {
                assert_eq!(gpu.name(), "gpu-wgpu");
                println!("[gpu test] GPU backend initialised successfully");
            }
            None => {
                println!("[gpu test] No GPU available — skipped (expected in headless CI)");
            }
        }
    }

    #[test]
    fn test_gpu_fallback_small_matrix() {
        let Some(gpu) = GpuBackend::try_new(1) else {
            return;
        };
        // Small matrix (below threshold) — goes through CPU fused path.
        let rows = 2;
        let cols = 32;
        let n = rows * cols;
        let mut raw = Vec::new();
        for _ in 0..2 {
            raw.extend_from_slice(&0x3C00u16.to_le_bytes()); // scale = 1.0 f16
            for i in 0i8..32 {
                raw.push(i as u8);
            }
        }
        let x: Vec<f32> = (0..cols).map(|i| i as f32 * 0.01 + 0.5).collect();
        let mut y = vec![0.0f32; rows];
        gpu.gemv_quant(GgmlType::Q8_0, &raw, rows, cols, n, &x, &mut y)
            .expect("gpu gemv_quant (small, cpu-fused path)");
        // Just check the output is finite.
        for &v in &y {
            assert!(v.is_finite(), "output {v} is not finite");
        }
    }
}
