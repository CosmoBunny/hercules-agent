//! Portable scalar compute backend — works on every device.
//!
//! Uses fused quant GEMV when available (no full f32 weight matrix).

use super::{ComputeBackend, ComputeError};
use crate::llama::gguf::{dequant_buffer, GgmlType};
use crate::llama::kernels::{gemv_quant_fused, supports_fused_gemv};

/// Single-thread (or thread-count reserved) pure-Rust backend.
#[derive(Debug, Clone)]
pub struct ScalarBackend {
    threads: usize,
}

impl Default for ScalarBackend {
    fn default() -> Self {
        Self { threads: 1 }
    }
}

impl ScalarBackend {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_threads(threads: usize) -> Self {
        Self {
            threads: threads.max(1),
        }
    }
}

impl ComputeBackend for ScalarBackend {
    fn name(&self) -> &str {
        "scalar-fused"
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
        if x.len() != cols {
            return Err(ComputeError(format!(
                "gemv x len {} != cols {}",
                x.len(),
                cols
            )));
        }
        if y.len() != rows {
            return Err(ComputeError(format!(
                "gemv y len {} != rows {}",
                y.len(),
                rows
            )));
        }

        if supports_fused_gemv(quant) {
            return gemv_quant_fused(quant, raw, rows, cols, n_elements, x, y)
                .map_err(ComputeError);
        }

        // Fallback: full dequant (rare types)
        let data = dequant_buffer(raw, quant, n_elements)
            .map_err(|e| ComputeError(e.to_string()))?;
        if data.len() != rows * cols {
            return Err(ComputeError(format!(
                "dequant size {} != rows*cols {}",
                data.len(),
                rows * cols
            )));
        }
        for r in 0..rows {
            let mut sum = 0.0f32;
            let row = &data[r * cols..(r + 1) * cols];
            for c in 0..cols {
                sum += row[c] * x[c];
            }
            y[r] = sum;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalar_f32_gemv_identity() {
        let mut raw = Vec::new();
        for v in [1.0f32, 0.0, 0.0, 1.0] {
            raw.extend_from_slice(&v.to_le_bytes());
        }
        let x = [3.0f32, 4.0];
        let mut y = [0.0f32; 2];
        let b = ScalarBackend::new();
        b.gemv_quant(GgmlType::F32, &raw, 2, 2, 4, &x, &mut y)
            .unwrap();
        assert!((y[0] - 3.0).abs() < 1e-5);
        assert!((y[1] - 4.0).abs() < 1e-5);
    }
}
