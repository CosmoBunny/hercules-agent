//! Multi-thread pure-Rust backend (rayon). Still no C/FFI.

use super::scalar::ScalarBackend;
use super::{ComputeBackend, ComputeError};
use crate::llama::gguf::GgmlType;
use crate::llama::kernels::{gemv_quant_fused, supports_fused_gemv};

/// Wraps fused GEMV; splits work across threads for large matrices.
pub struct ParallelBackend {
    threads: usize,
    inner: ScalarBackend,
}

impl ParallelBackend {
    pub fn new(threads: usize) -> Self {
        let threads = threads.max(1);
        Self {
            threads,
            inner: ScalarBackend::with_threads(threads),
        }
    }
}

impl ComputeBackend for ParallelBackend {
    fn name(&self) -> &str {
        "parallel-fused"
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
        // Small mats: single-thread fused (less overhead)
        if rows < 64 || self.threads <= 1 || !cfg!(feature = "parallel") {
            return self
                .inner
                .gemv_quant(quant, raw, rows, cols, n_elements, x, y);
        }

        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            // Fused path is sequential stream over weights; for parallel we
            // still use fused single-thread (streaming is already bandwidth-
            // bound). Parallel helps fallback dequant path.
            if supports_fused_gemv(quant) {
                return gemv_quant_fused(quant, raw, rows, cols, n_elements, x, y)
                    .map_err(ComputeError);
            }

            // Parallel row dequant-dot for unsupported types
            let data = crate::llama::gguf::dequant_buffer(raw, quant, n_elements)
                .map_err(|e| ComputeError(e.to_string()))?;
            if data.len() < rows * cols {
                return Err(ComputeError("dequant short".into()));
            }
            y.par_iter_mut().enumerate().for_each(|(r, yr)| {
                let mut sum = 0.0f32;
                let row = &data[r * cols..(r + 1) * cols];
                for c in 0..cols {
                    sum += row[c] * x[c];
                }
                *yr = sum;
            });
            Ok(())
        }

        #[cfg(not(feature = "parallel"))]
        {
            self.inner
                .gemv_quant(quant, raw, rows, cols, n_elements, x, y)
        }
    }
}
