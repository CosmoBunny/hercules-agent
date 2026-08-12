//! AVX2 + FMA 256-bit SIMD GEMV kernels for Q4_K, Q5_K, Q8_0, F16, F32.

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use std::arch::x86_64::*;

use crate::llama::gguf::{f16_to_f32, GgmlType};
use crate::llama::kernels::gemv_quant_fused;

const QK_K: usize = 256;

#[inline]
fn read_f16_le(raw: &[u8], idx: usize) -> f32 {
    f16_to_f32(u16::from_le_bytes([raw[idx], raw[idx + 1]]))
}

#[inline]
fn get_scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        let d = (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4);
        let m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
        (d, m)
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn hsum_avx2(v: __m256) -> f32 {
    unsafe {
        let vlow = _mm256_castps256_ps128(v);
        let vhigh = _mm256_extractf128_ps(v, 1);
        let v128 = _mm_add_ps(vlow, vhigh);
        let shuf = _mm_movehdup_ps(v128);
        let sums = _mm_add_ps(v128, shuf);
        let shuf2 = _mm_movehl_ps(sums, sums);
        let sums2 = _mm_add_ps(sums, shuf2);
        _mm_cvtss_f32(sums2)
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn gemv_avx2(
    quant: GgmlType,
    raw: &[u8],
    rows: usize,
    cols: usize,
    n_elements: usize,
    x: &[f32],
    y: &mut [f32],
) -> Result<(), String> {
    unsafe {
        match quant {
            GgmlType::Q8_0 => gemv_q8_0_avx2(raw, rows, cols, n_elements, x, y),
            GgmlType::Q4_K => gemv_q4_k_avx2(raw, rows, cols, n_elements, x, y),
            GgmlType::Q5_K => gemv_q5_k_avx2(raw, rows, cols, n_elements, x, y),
            GgmlType::F16 => gemv_f16_avx2(raw, rows, cols, n_elements, x, y),
            GgmlType::F32 => gemv_f32_avx2(raw, rows, cols, n_elements, x, y),
            _ => gemv_quant_fused(quant, raw, rows, cols, n_elements, x, y),
        }
    }
}

#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
pub unsafe fn gemv_avx2(
    quant: GgmlType,
    raw: &[u8],
    rows: usize,
    cols: usize,
    n_elements: usize,
    x: &[f32],
    y: &mut [f32],
) -> Result<(), String> {
    gemv_quant_fused(quant, raw, rows, cols, n_elements, x, y)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn gemv_f32_avx2(
    raw: &[u8],
    rows: usize,
    cols: usize,
    n: usize,
    x: &[f32],
    y: &mut [f32],
) -> Result<(), String> {
    if raw.len() < n * 4 {
        return Err("F32 raw too short".into());
    }
    let raw_ptr = raw.as_ptr() as *const f32;
    for r in 0..rows {
        let base = r * cols;
        let row_n = cols.min(if n > base { n - base } else { 0 });
        let mut acc = unsafe { _mm256_setzero_ps() };
        let mut c = 0;
        while c + 8 <= row_n {
            unsafe {
                let w_v = _mm256_loadu_ps(raw_ptr.add(base + c));
                let x_v = _mm256_loadu_ps(x.as_ptr().add(c));
                acc = _mm256_fmadd_ps(w_v, x_v, acc);
            }
            c += 8;
        }
        let mut sum = unsafe { hsum_avx2(acc) };
        while c < row_n {
            let off = (base + c) * 4;
            let w = f32::from_le_bytes([raw[off], raw[off + 1], raw[off + 2], raw[off + 3]]);
            sum += w * x[c];
            c += 1;
        }
        y[r] = sum;
    }
    Ok(())
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn gemv_f16_avx2(
    raw: &[u8],
    rows: usize,
    cols: usize,
    n: usize,
    x: &[f32],
    y: &mut [f32],
) -> Result<(), String> {
    if raw.len() < n * 2 {
        return Err("F16 raw too short".into());
    }
    for r in 0..rows {
        let base = r * cols;
        let row_n = cols.min(if n > base { n - base } else { 0 });
        let mut acc = unsafe { _mm256_setzero_ps() };
        let mut c = 0;
        while c + 8 <= row_n {
            let mut w_arr = [0.0f32; 8];
            for i in 0..8 {
                let off = (base + c + i) * 2;
                w_arr[i] = read_f16_le(raw, off);
            }
            unsafe {
                let w_v = _mm256_loadu_ps(w_arr.as_ptr());
                let x_v = _mm256_loadu_ps(x.as_ptr().add(c));
                acc = _mm256_fmadd_ps(w_v, x_v, acc);
            }
            c += 8;
        }
        let mut sum = unsafe { hsum_avx2(acc) };
        while c < row_n {
            let off = (base + c) * 2;
            let w = read_f16_le(raw, off);
            sum += w * x[c];
            c += 1;
        }
        y[r] = sum;
    }
    Ok(())
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn gemv_q8_0_avx2(
    raw: &[u8],
    rows: usize,
    cols: usize,
    n: usize,
    x: &[f32],
    y: &mut [f32],
) -> Result<(), String> {
    let block = 34;
    let blocks_per_row = cols / 32;
    if cols % 32 == 0 && n == rows * cols && raw.len() >= rows * blocks_per_row * block {
        for r in 0..rows {
            let mut acc = unsafe { _mm256_setzero_ps() };
            for b in 0..blocks_per_row {
                let sb_idx = (r * blocks_per_row + b) * block;
                let scale = read_f16_le(raw, sb_idx);
                let scale_v = unsafe { _mm256_set1_ps(scale) };
                let qs = &raw[sb_idx + 2..sb_idx + 34];
                let x_base = b * 32;

                for chunk in 0..4 {
                    let c = x_base + chunk * 8;
                    let q_slice = &qs[chunk * 8..(chunk + 1) * 8];
                    let q_arr = [
                        (q_slice[0] as i8) as f32,
                        (q_slice[1] as i8) as f32,
                        (q_slice[2] as i8) as f32,
                        (q_slice[3] as i8) as f32,
                        (q_slice[4] as i8) as f32,
                        (q_slice[5] as i8) as f32,
                        (q_slice[6] as i8) as f32,
                        (q_slice[7] as i8) as f32,
                    ];
                    unsafe {
                        let q_v = _mm256_loadu_ps(q_arr.as_ptr());
                        let w_v = _mm256_mul_ps(q_v, scale_v);
                        let x_v = _mm256_loadu_ps(x.as_ptr().add(c));
                        acc = _mm256_fmadd_ps(w_v, x_v, acc);
                    }
                }
            }
            y[r] = unsafe { hsum_avx2(acc) };
        }
        return Ok(());
    }

    gemv_quant_fused(GgmlType::Q8_0, raw, rows, cols, n, x, y)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn gemv_q4_k_avx2(
    raw: &[u8],
    rows: usize,
    cols: usize,
    n: usize,
    x: &[f32],
    y: &mut [f32],
) -> Result<(), String> {
    let block = 144;
    let blocks_per_row = cols / QK_K;
    if cols % QK_K == 0 && n == rows * cols && raw.len() >= rows * blocks_per_row * block {
        for r in 0..rows {
            let mut acc = unsafe { _mm256_setzero_ps() };
            for b in 0..blocks_per_row {
                let sb_idx = (r * blocks_per_row + b) * block;
                let d = read_f16_le(raw, sb_idx);
                let dmin = read_f16_le(raw, sb_idx + 2);
                let scales = &raw[sb_idx + 4..sb_idx + 16];
                let mut q_ptr = &raw[sb_idx + 16..sb_idx + 144];
                let x_base = b * QK_K;

                let mut is = 0usize;
                for j in 0..4 {
                    let (sc1, m1) = get_scale_min_k4(is, scales);
                    let d1 = d * (sc1 as f32);
                    let m1 = dmin * (m1 as f32);
                    let d1_v = unsafe { _mm256_set1_ps(d1) };
                    let m1_v = unsafe { _mm256_set1_ps(m1) };

                    let (sc2, m2) = get_scale_min_k4(is + 1, scales);
                    let d2 = d * (sc2 as f32);
                    let m2 = dmin * (m2 as f32);
                    let d2_v = unsafe { _mm256_set1_ps(d2) };
                    let m2_v = unsafe { _mm256_set1_ps(m2) };

                    // Sub-block 1: 32 elements (low nibbles)
                    for chunk in 0..4 {
                        let c = x_base + j * 64 + chunk * 8;
                        let ql_slice = &q_ptr[chunk * 8..(chunk + 1) * 8];
                        let q_low = [
                            (ql_slice[0] & 0x0F) as f32,
                            (ql_slice[1] & 0x0F) as f32,
                            (ql_slice[2] & 0x0F) as f32,
                            (ql_slice[3] & 0x0F) as f32,
                            (ql_slice[4] & 0x0F) as f32,
                            (ql_slice[5] & 0x0F) as f32,
                            (ql_slice[6] & 0x0F) as f32,
                            (ql_slice[7] & 0x0F) as f32,
                        ];
                        unsafe {
                            let q_v = _mm256_loadu_ps(q_low.as_ptr());
                            let w_v = _mm256_sub_ps(_mm256_mul_ps(d1_v, q_v), m1_v);
                            let x_v = _mm256_loadu_ps(x.as_ptr().add(c));
                            acc = _mm256_fmadd_ps(w_v, x_v, acc);
                        }
                    }

                    // Sub-block 2: 32 elements (high nibbles)
                    for chunk in 0..4 {
                        let c = x_base + j * 64 + 32 + chunk * 8;
                        let ql_slice = &q_ptr[chunk * 8..(chunk + 1) * 8];
                        let q_high = [
                            (ql_slice[0] >> 4) as f32,
                            (ql_slice[1] >> 4) as f32,
                            (ql_slice[2] >> 4) as f32,
                            (ql_slice[3] >> 4) as f32,
                            (ql_slice[4] >> 4) as f32,
                            (ql_slice[5] >> 4) as f32,
                            (ql_slice[6] >> 4) as f32,
                            (ql_slice[7] >> 4) as f32,
                        ];
                        unsafe {
                            let q_v = _mm256_loadu_ps(q_high.as_ptr());
                            let w_v = _mm256_sub_ps(_mm256_mul_ps(d2_v, q_v), m2_v);
                            let x_v = _mm256_loadu_ps(x.as_ptr().add(c));
                            acc = _mm256_fmadd_ps(w_v, x_v, acc);
                        }
                    }

                    q_ptr = &q_ptr[32..];
                    is += 2;
                }
            }
            y[r] = unsafe { hsum_avx2(acc) };
        }
        return Ok(());
    }

    gemv_quant_fused(GgmlType::Q4_K, raw, rows, cols, n, x, y)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn gemv_q5_k_avx2(
    raw: &[u8],
    rows: usize,
    cols: usize,
    n: usize,
    x: &[f32],
    y: &mut [f32],
) -> Result<(), String> {
    let block = 176;
    let blocks_per_row = cols / QK_K;
    if cols % QK_K == 0 && n == rows * cols && raw.len() >= rows * blocks_per_row * block {
        for r in 0..rows {
            let mut acc = unsafe { _mm256_setzero_ps() };
            for b in 0..blocks_per_row {
                let sb_idx = (r * blocks_per_row + b) * block;
                let d = read_f16_le(raw, sb_idx);
                let dmin = read_f16_le(raw, sb_idx + 2);
                let scales = &raw[sb_idx + 4..sb_idx + 16];
                let qh = &raw[sb_idx + 16..sb_idx + 48];
                let mut ql = &raw[sb_idx + 48..sb_idx + 176];
                let x_base = b * QK_K;

                let mut is = 0usize;
                let mut u1: u8 = 1;
                let mut u2: u8 = 2;

                for j in 0..4 {
                    let (sc1, m1) = get_scale_min_k4(is, scales);
                    let d1 = d * (sc1 as f32);
                    let m1 = dmin * (m1 as f32);
                    let d1_v = unsafe { _mm256_set1_ps(d1) };
                    let m1_v = unsafe { _mm256_set1_ps(m1) };

                    let (sc2, m2) = get_scale_min_k4(is + 1, scales);
                    let d2 = d * (sc2 as f32);
                    let m2 = dmin * (m2 as f32);
                    let d2_v = unsafe { _mm256_set1_ps(d2) };
                    let m2_v = unsafe { _mm256_set1_ps(m2) };

                    // Sub-block 1: 32 elements
                    for chunk in 0..4 {
                        let c = x_base + j * 64 + chunk * 8;
                        let ql_slice = &ql[chunk * 8..(chunk + 1) * 8];
                        let qh_slice = &qh[chunk * 8..(chunk + 1) * 8];
                        let q_arr = [
                            ((ql_slice[0] & 0x0F) + if qh_slice[0] & u1 != 0 { 16 } else { 0 })
                                as f32,
                            ((ql_slice[1] & 0x0F) + if qh_slice[1] & u1 != 0 { 16 } else { 0 })
                                as f32,
                            ((ql_slice[2] & 0x0F) + if qh_slice[2] & u1 != 0 { 16 } else { 0 })
                                as f32,
                            ((ql_slice[3] & 0x0F) + if qh_slice[3] & u1 != 0 { 16 } else { 0 })
                                as f32,
                            ((ql_slice[4] & 0x0F) + if qh_slice[4] & u1 != 0 { 16 } else { 0 })
                                as f32,
                            ((ql_slice[5] & 0x0F) + if qh_slice[5] & u1 != 0 { 16 } else { 0 })
                                as f32,
                            ((ql_slice[6] & 0x0F) + if qh_slice[6] & u1 != 0 { 16 } else { 0 })
                                as f32,
                            ((ql_slice[7] & 0x0F) + if qh_slice[7] & u1 != 0 { 16 } else { 0 })
                                as f32,
                        ];
                        unsafe {
                            let q_v = _mm256_loadu_ps(q_arr.as_ptr());
                            let w_v = _mm256_sub_ps(_mm256_mul_ps(d1_v, q_v), m1_v);
                            let x_v = _mm256_loadu_ps(x.as_ptr().add(c));
                            acc = _mm256_fmadd_ps(w_v, x_v, acc);
                        }
                    }

                    // Sub-block 2: 32 elements
                    for chunk in 0..4 {
                        let c = x_base + j * 64 + 32 + chunk * 8;
                        let ql_slice = &ql[chunk * 8..(chunk + 1) * 8];
                        let qh_slice = &qh[chunk * 8..(chunk + 1) * 8];
                        let q_arr = [
                            ((ql_slice[0] >> 4) + if qh_slice[0] & u2 != 0 { 16 } else { 0 })
                                as f32,
                            ((ql_slice[1] >> 4) + if qh_slice[1] & u2 != 0 { 16 } else { 0 })
                                as f32,
                            ((ql_slice[2] >> 4) + if qh_slice[2] & u2 != 0 { 16 } else { 0 })
                                as f32,
                            ((ql_slice[3] >> 4) + if qh_slice[3] & u2 != 0 { 16 } else { 0 })
                                as f32,
                            ((ql_slice[4] >> 4) + if qh_slice[4] & u2 != 0 { 16 } else { 0 })
                                as f32,
                            ((ql_slice[5] >> 4) + if qh_slice[5] & u2 != 0 { 16 } else { 0 })
                                as f32,
                            ((ql_slice[6] >> 4) + if qh_slice[6] & u2 != 0 { 16 } else { 0 })
                                as f32,
                            ((ql_slice[7] >> 4) + if qh_slice[7] & u2 != 0 { 16 } else { 0 })
                                as f32,
                        ];
                        unsafe {
                            let q_v = _mm256_loadu_ps(q_arr.as_ptr());
                            let w_v = _mm256_sub_ps(_mm256_mul_ps(d2_v, q_v), m2_v);
                            let x_v = _mm256_loadu_ps(x.as_ptr().add(c));
                            acc = _mm256_fmadd_ps(w_v, x_v, acc);
                        }
                    }

                    ql = &ql[32..];
                    is += 2;
                    u1 <<= 2;
                    u2 <<= 2;
                }
            }
            y[r] = unsafe { hsum_avx2(acc) };
        }
        return Ok(());
    }

    gemv_quant_fused(GgmlType::Q5_K, raw, rows, cols, n, x, y)
}

/// AVX2+FMA accelerated RMSNorm: `out[i] = x[i] / rms(x) * weight[i]`.
///
/// Uses 8-wide SIMD to accumulate sum-of-squares, then a second pass to scale.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn rms_norm_avx2(x: &[f32], weight: &[f32], eps: f32, out: &mut [f32]) {
    let n = x.len().min(out.len()).min(weight.len());

    // --- Pass 1: sum of squares ---
    let mut acc = unsafe { _mm256_setzero_ps() };
    let mut i = 0;
    while i + 8 <= n {
        unsafe {
            let xv = _mm256_loadu_ps(x.as_ptr().add(i));
            acc = _mm256_fmadd_ps(xv, xv, acc);
        }
        i += 8;
    }
    let mut ss = unsafe { hsum_avx2(acc) };
    while i < n {
        ss += x[i] * x[i];
        i += 1;
    }

    let scale = (ss / n as f32 + eps).sqrt().recip();
    let scale_v = unsafe { _mm256_set1_ps(scale) };

    // --- Pass 2: scale × weight ---
    let mut j = 0;
    while j + 8 <= n {
        unsafe {
            let xv  = _mm256_loadu_ps(x.as_ptr().add(j));
            let wv  = _mm256_loadu_ps(weight.as_ptr().add(j));
            let res = _mm256_mul_ps(_mm256_mul_ps(xv, scale_v), wv);
            _mm256_storeu_ps(out.as_mut_ptr().add(j), res);
        }
        j += 8;
    }
    while j < n {
        out[j] = x[j] * scale * weight[j];
        j += 1;
    }
    for k in n..out.len() {
        out[k] = 0.0;
    }
}

#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
pub unsafe fn rms_norm_avx2(x: &[f32], weight: &[f32], eps: f32, out: &mut [f32]) {
    crate::llama::compute::default_rms_norm(x, weight, eps, out);
}
