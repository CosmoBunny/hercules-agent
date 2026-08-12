//! Dynamic CPU feature detection & SIMD acceleration backend.

pub mod avx2;
pub mod avx512;
pub mod neon;

use super::scalar::ScalarBackend;
use super::{ComputeBackend, ComputeError};
use crate::llama::gguf::GgmlType;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimdInstructionSet {
    Avx512,
    Avx2,
    Neon,
    Scalar,
}

impl SimdInstructionSet {
    pub fn detect() -> Self {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if std::is_x86_feature_detected!("avx512f")
                && std::is_x86_feature_detected!("avx512bw")
            {
                return Self::Avx512;
            }
            if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
                return Self::Avx2;
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            return Self::Neon;
        }
        #[allow(unreachable_code)]
        Self::Scalar
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::Avx512 => "simd-avx512",
            Self::Avx2 => "simd-avx2",
            Self::Neon => "simd-neon",
            Self::Scalar => "simd-scalar",
        }
    }
}

#[derive(Debug, Clone)]
pub struct SimdBackend {
    threads: usize,
    isa: SimdInstructionSet,
    scalar: ScalarBackend,
}

impl SimdBackend {
    pub fn new(threads: usize) -> Self {
        let threads = threads.max(1);
        let isa = SimdInstructionSet::detect();
        Self {
            threads,
            isa,
            scalar: ScalarBackend::with_threads(threads),
        }
    }

    pub fn with_isa(threads: usize, isa: SimdInstructionSet) -> Self {
        let threads = threads.max(1);
        Self {
            threads,
            isa,
            scalar: ScalarBackend::with_threads(threads),
        }
    }

    pub fn is_supported() -> bool {
        SimdInstructionSet::detect() != SimdInstructionSet::Scalar
    }

    pub fn isa(&self) -> SimdInstructionSet {
        self.isa
    }
}

impl ComputeBackend for SimdBackend {
    fn name(&self) -> &str {
        self.isa.name()
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
        match self.isa {
            SimdInstructionSet::Avx512 => unsafe {
                avx512::gemv_avx512(quant, raw, rows, cols, n_elements, x, y)
                    .map_err(ComputeError)
            },
            SimdInstructionSet::Avx2 => unsafe {
                avx2::gemv_avx2(quant, raw, rows, cols, n_elements, x, y)
                    .map_err(ComputeError)
            },
            SimdInstructionSet::Neon => unsafe {
                neon::gemv_neon(quant, raw, rows, cols, n_elements, x, y)
                    .map_err(ComputeError)
            },
            SimdInstructionSet::Scalar => {
                self.scalar.gemv_quant(quant, raw, rows, cols, n_elements, x, y)
            }
        }
    }

    /// SIMD-accelerated RMSNorm. Uses AVX2/AVX-512 horizontal reduction on x86,
    /// NEON on AArch64, and falls back to scalar on unsupported ISAs.
    fn rms_norm(&self, x: &[f32], weight: &[f32], eps: f32, out: &mut [f32]) {
        match self.isa {
            SimdInstructionSet::Avx512 | SimdInstructionSet::Avx2 => {
                #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                {
                    // Safety: feature already confirmed by SimdInstructionSet::detect()
                    if self.isa == SimdInstructionSet::Avx2
                        || self.isa == SimdInstructionSet::Avx512
                    {
                        unsafe { avx2::rms_norm_avx2(x, weight, eps, out) };
                        return;
                    }
                }
                super::default_rms_norm(x, weight, eps, out);
            }
            SimdInstructionSet::Neon => {
                #[cfg(target_arch = "aarch64")]
                {
                    unsafe { neon::rms_norm_neon(x, weight, eps, out) };
                    return;
                }
                #[allow(unreachable_code)]
                super::default_rms_norm(x, weight, eps, out);
            }
            SimdInstructionSet::Scalar => {
                super::default_rms_norm(x, weight, eps, out);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llama::gguf::dequant_buffer;

    fn verify_equivalence(backend: &dyn ComputeBackend, quant: GgmlType, raw: &[u8], rows: usize, cols: usize) {
        let n = rows * cols;
        let x: Vec<f32> = (0..cols).map(|i| (i as f32 * 0.01) + 0.5).collect();
        
        let scalar_backend = ScalarBackend::new();
        let mut y_scalar = vec![0.0f32; rows];
        scalar_backend.gemv_quant(quant, raw, rows, cols, n, &x, &mut y_scalar).unwrap();

        let mut y_simd = vec![0.0f32; rows];
        backend.gemv_quant(quant, raw, rows, cols, n, &x, &mut y_simd).unwrap();

        let ref_weights = dequant_buffer(raw, quant, n).expect("dequant");
        let mut y_ref = vec![0.0f32; rows];
        for r in 0..rows {
            let mut sum = 0.0f32;
            for c in 0..cols {
                sum += ref_weights[r * cols + c] * x[c];
            }
            y_ref[r] = sum;
        }

        for r in 0..rows {
            let err_simd = (y_simd[r] - y_scalar[r]).abs();
            let err_ref = (y_simd[r] - y_ref[r]).abs();
            assert!(
                err_simd < 1e-3 * (1.0 + y_scalar[r].abs()),
                "row {r}: simd {} vs scalar {} (err {})",
                y_simd[r], y_scalar[r], err_simd
            );
            assert!(
                err_ref < 1e-3 * (1.0 + y_ref[r].abs()),
                "row {r}: simd {} vs ref {} (err {})",
                y_simd[r], y_ref[r], err_ref
            );
        }
    }

    #[test]
    fn test_simd_backend_q8_0() {
        let rows = 2;
        let cols = 64;
        let mut raw = Vec::new();
        for b in 0..4i8 {
            raw.extend_from_slice(&0x3C00u16.to_le_bytes()); // scale 1.0
            for i in 0..32i8 {
                raw.push(i.wrapping_add(b) as u8);
            }
        }
        let backend = SimdBackend::new(1);
        verify_equivalence(&backend, GgmlType::Q8_0, &raw, rows, cols);
    }

    #[test]
    fn test_simd_backend_q4_k() {
        let rows = 1;
        let cols = 256;
        let mut raw = vec![0u8; 144];
        raw[0] = 0x00; raw[1] = 0x3C; // d = 1.0
        raw[2] = 0x00; raw[3] = 0x38; // dmin = 0.5
        for i in 4..16 { raw[i] = (i * 3) as u8; }
        for i in 16..144 { raw[i] = (i * 11) as u8; }
        let backend = SimdBackend::new(1);
        verify_equivalence(&backend, GgmlType::Q4_K, &raw, rows, cols);
    }

    #[test]
    fn test_simd_backend_q5_k() {
        let rows = 1;
        let cols = 256;
        let mut raw = vec![0u8; 176];
        raw[0] = 0x00; raw[1] = 0x3C; // d = 1.0
        raw[2] = 0x00; raw[3] = 0x38; // dmin = 0.5
        for i in 4..16 { raw[i] = (i * 7) as u8; }
        for i in 16..48 { raw[i] = (i * 13) as u8; }
        for i in 48..176 { raw[i] = (i * 17) as u8; }
        let backend = SimdBackend::new(1);
        verify_equivalence(&backend, GgmlType::Q5_K, &raw, rows, cols);
    }

    #[test]
    fn test_simd_backend_f16() {
        let rows = 2;
        let cols = 16;
        let mut raw = Vec::new();
        for i in 0..(rows * cols) {
            let u = 0x3C00u16 + (i as u16 * 0x0040); // valid f16 numbers starting at 1.0
            raw.extend_from_slice(&u.to_le_bytes());
        }
        let backend = SimdBackend::new(1);
        verify_equivalence(&backend, GgmlType::F16, &raw, rows, cols);
    }

    #[test]
    fn test_simd_backend_f32() {
        let rows = 2;
        let cols = 16;
        let mut raw = Vec::new();
        for i in 0..(rows * cols) {
            raw.extend_from_slice(&(i as f32 * 0.1).to_le_bytes());
        }
        let backend = SimdBackend::new(1);
        verify_equivalence(&backend, GgmlType::F32, &raw, rows, cols);
    }
}
