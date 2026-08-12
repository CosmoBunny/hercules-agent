//! Multi-thread SIMD-fused backend (rayon + SIMD). Still no C/FFI.
//!
//! ## Strategy
//! For large matrices (≥ threshold rows):
//!   - Partition rows across rayon threads.
//!   - Each thread runs the SIMD GEMV kernel on its slice.
//!
//! For small matrices or when SIMD is not supported, falls back to the
//! scalar fused kernel (same as before, no regression).
//!
//! This is the highest-throughput CPU path: SIMD × N_threads bandwidth.

use super::scalar::ScalarBackend;
use super::simd::{SimdBackend, SimdInstructionSet};
use super::{ComputeBackend, ComputeError};
use crate::llama::gguf::{dequant_buffer, GgmlType};
use crate::llama::kernels::{gemv_quant_fused, supports_fused_gemv};

/// Minimum rows to use the multi-threaded path (below this, overhead dominates).
const PAR_THRESHOLD_ROWS: usize = 64;

/// Wraps SIMD GEMV; splits work across rayon threads for large matrices.
pub struct ParallelBackend {
    threads: usize,
    simd: Option<SimdBackend>,
    scalar: ScalarBackend,
}

impl ParallelBackend {
    pub fn new(threads: usize) -> Self {
        let threads = threads.max(1);
        let simd = if SimdBackend::is_supported() {
            Some(SimdBackend::new(threads))
        } else {
            None
        };
        Self {
            threads,
            simd,
            scalar: ScalarBackend::with_threads(threads),
        }
    }
}

impl ComputeBackend for ParallelBackend {
    fn name(&self) -> &str {
        if self.simd.is_some() {
            "parallel-simd"
        } else {
            "parallel-fused"
        }
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
        // Small matrices: single-thread SIMD or scalar (no rayon overhead)
        if rows < PAR_THRESHOLD_ROWS || self.threads <= 1 || !cfg!(feature = "parallel") {
            return if let Some(ref simd) = self.simd {
                simd.gemv_quant(quant, raw, rows, cols, n_elements, x, y)
            } else {
                self.scalar.gemv_quant(quant, raw, rows, cols, n_elements, x, y)
            };
        }

        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;

            // ----------------------------------------------------------------
            // SIMD-fused parallel path: each thread runs its own SIMD kernel
            // on a contiguous row slice.
            // ----------------------------------------------------------------
            if let Some(ref simd_backend) = self.simd {
                let isa = simd_backend.isa();

                // Choose block sizes based on quant type
                let elem_per_col = n_elements / rows.max(1);
                let bytes_per_row = if cols > 0 {
                    raw.len() / rows.max(1)
                } else {
                    0
                };

                // Parallel GEMV: partition rows, each chunk gets its raw slice
                let result: Result<(), String> = y
                    .par_iter_mut()
                    .enumerate()
                    .try_for_each(|(r, yr)| {
                        // Per-row raw slice
                        let row_raw_start = r * bytes_per_row;
                        let row_raw_end = (row_raw_start + bytes_per_row).min(raw.len());
                        let row_raw = &raw[row_raw_start..row_raw_end];

                        let row_n_elem = elem_per_col.min(n_elements.saturating_sub(r * cols));

                        let mut row_out = [0.0f32; 1];
                        let x_slice = &x[..cols.min(x.len())];

                        // Run single-row SIMD GEMV
                        let mut res = unsafe {
                            match isa {
                                SimdInstructionSet::Avx512 => {
                                    super::simd::avx512::gemv_avx512(
                                        quant, row_raw, 1, cols, row_n_elem, x_slice, &mut row_out,
                                    )
                                }
                                SimdInstructionSet::Avx2 => {
                                    super::simd::avx2::gemv_avx2(
                                        quant, row_raw, 1, cols, row_n_elem, x_slice, &mut row_out,
                                    )
                                }
                                SimdInstructionSet::Neon => {
                                    super::simd::neon::gemv_neon(
                                        quant, row_raw, 1, cols, row_n_elem, x_slice, &mut row_out,
                                    )
                                }
                                _ => Err("unsupported SIMD".into()),
                            }
                        };
                        if res.is_err() {
                            res = gemv_quant_fused(
                                quant, row_raw, 1, cols, row_n_elem, x_slice, &mut row_out,
                            );
                        }
                        res.map(|_| {
                            *yr = row_out[0];
                        })
                    });

                if result.is_ok() {
                    return result.map_err(ComputeError);
                }
            }

            // ----------------------------------------------------------------
            // Fused-only parallel path (no SIMD available)
            // ----------------------------------------------------------------
            if supports_fused_gemv(quant) {
                return gemv_quant_fused(quant, raw, rows, cols, n_elements, x, y)
                    .map_err(ComputeError);
            }

            // Parallel row dequant-dot for unsupported quant types
            let data = dequant_buffer(raw, quant, n_elements)
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
            self.scalar
                .gemv_quant(quant, raw, rows, cols, n_elements, x, y)
        }
    }

    fn rms_norm(&self, x: &[f32], weight: &[f32], eps: f32, out: &mut [f32]) {
        // Delegate to SIMD for the actual norm (not worth parallelising — memory-bound)
        if let Some(ref simd) = self.simd {
            simd.rms_norm(x, weight, eps, out);
        } else {
            super::default_rms_norm(x, weight, eps, out);
        }
    }
}
