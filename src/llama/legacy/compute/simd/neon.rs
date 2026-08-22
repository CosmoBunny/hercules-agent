//! ARM NEON 128-bit SIMD GEMV kernels for Q4_K, Q5_K, Q8_0, F16, F32.

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

use crate::llama::gguf::{f16_to_f32, GgmlType};
use crate::llama::kernels::gemv_quant_fused;

#[allow(dead_code)]
const QK_K: usize = 256;

#[inline]
#[allow(dead_code)]
fn read_f16_le(raw: &[u8], idx: usize) -> f32 {
    f16_to_f32(u16::from_le_bytes([raw[idx], raw[idx + 1]]))
}

#[inline]
#[allow(dead_code)]
fn get_scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        let d = (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4);
        let m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
        (d, m)
    }
}

#[cfg(target_arch = "aarch64")]
pub unsafe fn gemv_neon(
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
            GgmlType::Q8_0 => gemv_q8_0_neon(raw, rows, cols, n_elements, x, y),
            GgmlType::Q4_K => gemv_q4_k_neon(raw, rows, cols, n_elements, x, y),
            GgmlType::Q5_K => gemv_q5_k_neon(raw, rows, cols, n_elements, x, y),
            GgmlType::F16 => gemv_f16_neon(raw, rows, cols, n_elements, x, y),
            GgmlType::F32 => gemv_f32_neon(raw, rows, cols, n_elements, x, y),
            _ => gemv_quant_fused(quant, raw, rows, cols, n_elements, x, y),
        }
    }
}

#[cfg(not(target_arch = "aarch64"))]
pub unsafe fn gemv_neon(
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

#[cfg(target_arch = "aarch64")]
pub unsafe fn gemv_f32_neon(
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
        let mut acc = unsafe { vdupq_n_f32(0.0) };
        let mut c = 0;
        while c + 4 <= row_n {
            unsafe {
                let w_v = vld1q_f32(raw_ptr.add(base + c));
                let x_v = vld1q_f32(x.as_ptr().add(c));
                acc = vfmaq_f32(acc, w_v, x_v);
            }
            c += 4;
        }
        let mut sum = unsafe { vaddvq_f32(acc) };
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

#[cfg(target_arch = "aarch64")]
pub unsafe fn gemv_f16_neon(
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
        let mut acc = unsafe { vdupq_n_f32(0.0) };
        let mut c = 0;
        while c + 4 <= row_n {
            let w_arr = [
                read_f16_le(raw, (base + c) * 2),
                read_f16_le(raw, (base + c + 1) * 2),
                read_f16_le(raw, (base + c + 2) * 2),
                read_f16_le(raw, (base + c + 3) * 2),
            ];
            unsafe {
                let w_v = vld1q_f32(w_arr.as_ptr());
                let x_v = vld1q_f32(x.as_ptr().add(c));
                acc = vfmaq_f32(acc, w_v, x_v);
            }
            c += 4;
        }
        let mut sum = unsafe { vaddvq_f32(acc) };
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

#[cfg(target_arch = "aarch64")]
pub unsafe fn gemv_q8_0_neon(
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
            let mut acc = unsafe { vdupq_n_f32(0.0) };
            for b in 0..blocks_per_row {
                let sb_idx = (r * blocks_per_row + b) * block;
                let scale = read_f16_le(raw, sb_idx);
                let scale_v = unsafe { vdupq_n_f32(scale) };
                let qs = &raw[sb_idx + 2..sb_idx + 34];
                let x_base = b * 32;

                for chunk in 0..8 {
                    let c = x_base + chunk * 4;
                    let q_slice = &qs[chunk * 4..(chunk + 1) * 4];
                    let q_arr = [
                        (q_slice[0] as i8) as f32,
                        (q_slice[1] as i8) as f32,
                        (q_slice[2] as i8) as f32,
                        (q_slice[3] as i8) as f32,
                    ];
                    unsafe {
                        let q_v = vld1q_f32(q_arr.as_ptr());
                        let w_v = vmulq_f32(q_v, scale_v);
                        let x_v = vld1q_f32(x.as_ptr().add(c));
                        acc = vfmaq_f32(acc, w_v, x_v);
                    }
                }
            }
            y[r] = unsafe { vaddvq_f32(acc) };
        }
        return Ok(());
    }

    gemv_quant_fused(GgmlType::Q8_0, raw, rows, cols, n, x, y)
}

#[cfg(target_arch = "aarch64")]
pub unsafe fn gemv_q4_k_neon(
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
            let mut acc = unsafe { vdupq_n_f32(0.0) };
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
                    let d1_v = unsafe { vdupq_n_f32(d1) };
                    let m1_v = unsafe { vdupq_n_f32(m1) };

                    let (sc2, m2) = get_scale_min_k4(is + 1, scales);
                    let d2 = d * (sc2 as f32);
                    let m2 = dmin * (m2 as f32);
                    let d2_v = unsafe { vdupq_n_f32(d2) };
                    let m2_v = unsafe { vdupq_n_f32(m2) };

                    // Sub-block 1: 32 elements (8 chunks of 4)
                    for chunk in 0..8 {
                        let c = x_base + j * 64 + chunk * 4;
                        let ql_slice = &q_ptr[chunk * 4..(chunk + 1) * 4];
                        let q_low = [
                            (ql_slice[0] & 0x0F) as f32,
                            (ql_slice[1] & 0x0F) as f32,
                            (ql_slice[2] & 0x0F) as f32,
                            (ql_slice[3] & 0x0F) as f32,
                        ];
                        unsafe {
                            let q_v = vld1q_f32(q_low.as_ptr());
                            let w_v = vsubq_f32(vmulq_f32(d1_v, q_v), m1_v);
                            let x_v = vld1q_f32(x.as_ptr().add(c));
                            acc = vfmaq_f32(acc, w_v, x_v);
                        }
                    }

                    // Sub-block 2: 32 elements (8 chunks of 4)
                    for chunk in 0..8 {
                        let c = x_base + j * 64 + 32 + chunk * 4;
                        let ql_slice = &q_ptr[chunk * 4..(chunk + 1) * 4];
                        let q_high = [
                            (ql_slice[0] >> 4) as f32,
                            (ql_slice[1] >> 4) as f32,
                            (ql_slice[2] >> 4) as f32,
                            (ql_slice[3] >> 4) as f32,
                        ];
                        unsafe {
                            let q_v = vld1q_f32(q_high.as_ptr());
                            let w_v = vsubq_f32(vmulq_f32(d2_v, q_v), m2_v);
                            let x_v = vld1q_f32(x.as_ptr().add(c));
                            acc = vfmaq_f32(acc, w_v, x_v);
                        }
                    }

                    q_ptr = &q_ptr[32..];
                    is += 2;
                }
            }
            y[r] = unsafe { vaddvq_f32(acc) };
        }
        return Ok(());
    }

    gemv_quant_fused(GgmlType::Q4_K, raw, rows, cols, n, x, y)
}

#[cfg(target_arch = "aarch64")]
pub unsafe fn gemv_q5_k_neon(
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
            let mut acc = unsafe { vdupq_n_f32(0.0) };
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
                    let d1_v = unsafe { vdupq_n_f32(d1) };
                    let m1_v = unsafe { vdupq_n_f32(m1) };

                    let (sc2, m2) = get_scale_min_k4(is + 1, scales);
                    let d2 = d * (sc2 as f32);
                    let m2 = dmin * (m2 as f32);
                    let d2_v = unsafe { vdupq_n_f32(d2) };
                    let m2_v = unsafe { vdupq_n_f32(m2) };

                    // Sub-block 1: 32 elements (8 chunks of 4)
                    for chunk in 0..8 {
                        let c = x_base + j * 64 + chunk * 4;
                        let ql_slice = &ql[chunk * 4..(chunk + 1) * 4];
                        let qh_slice = &qh[chunk * 4..(chunk + 1) * 4];
                        let q_arr = [
                            ((ql_slice[0] & 0x0F) + if qh_slice[0] & u1 != 0 { 16 } else { 0 })
                                as f32,
                            ((ql_slice[1] & 0x0F) + if qh_slice[1] & u1 != 0 { 16 } else { 0 })
                                as f32,
                            ((ql_slice[2] & 0x0F) + if qh_slice[2] & u1 != 0 { 16 } else { 0 })
                                as f32,
                            ((ql_slice[3] & 0x0F) + if qh_slice[3] & u1 != 0 { 16 } else { 0 })
                                as f32,
                        ];
                        unsafe {
                            let q_v = vld1q_f32(q_arr.as_ptr());
                            let w_v = vsubq_f32(vmulq_f32(d1_v, q_v), m1_v);
                            let x_v = vld1q_f32(x.as_ptr().add(c));
                            acc = vfmaq_f32(acc, w_v, x_v);
                        }
                    }

                    // Sub-block 2: 32 elements (8 chunks of 4)
                    for chunk in 0..8 {
                        let c = x_base + j * 64 + 32 + chunk * 4;
                        let ql_slice = &ql[chunk * 4..(chunk + 1) * 4];
                        let qh_slice = &qh[chunk * 4..(chunk + 1) * 4];
                        let q_arr = [
                            ((ql_slice[0] >> 4) + if qh_slice[0] & u2 != 0 { 16 } else { 0 })
                                as f32,
                            ((ql_slice[1] >> 4) + if qh_slice[1] & u2 != 0 { 16 } else { 0 })
                                as f32,
                            ((ql_slice[2] >> 4) + if qh_slice[2] & u2 != 0 { 16 } else { 0 })
                                as f32,
                            ((ql_slice[3] >> 4) + if qh_slice[3] & u2 != 0 { 16 } else { 0 })
                                as f32,
                        ];
                        unsafe {
                            let q_v = vld1q_f32(q_arr.as_ptr());
                            let w_v = vsubq_f32(vmulq_f32(d2_v, q_v), m2_v);
                            let x_v = vld1q_f32(x.as_ptr().add(c));
                            acc = vfmaq_f32(acc, w_v, x_v);
                        }
                    }

                    ql = &ql[32..];
                    is += 2;
                    u1 <<= 2;
                    u2 <<= 2;
                }
            }
            y[r] = unsafe { vaddvq_f32(acc) };
        }
        return Ok(());
    }

    gemv_quant_fused(GgmlType::Q5_K, raw, rows, cols, n, x, y)
}

/// ARM NEON accelerated RMSNorm: `out[i] = x[i] / rms(x) * weight[i]`.
#[cfg(target_arch = "aarch64")]
pub unsafe fn rms_norm_neon(x: &[f32], weight: &[f32], eps: f32, out: &mut [f32]) {
    let n = x.len().min(out.len()).min(weight.len());

    // --- Pass 1: sum of squares ---
    let mut acc = unsafe { vdupq_n_f32(0.0) };
    let mut i = 0;
    while i + 4 <= n {
        unsafe {
            let xv = vld1q_f32(x.as_ptr().add(i));
            acc = vfmaq_f32(acc, xv, xv);
        }
        i += 4;
    }
    let mut ss = unsafe { vaddvq_f32(acc) };
    while i < n {
        ss += x[i] * x[i];
        i += 1;
    }

    let scale = (ss / n as f32 + eps).sqrt().recip();
    let scale_v = unsafe { vdupq_n_f32(scale) };

    // --- Pass 2: scale × weight ---
    let mut j = 0;
    while j + 4 <= n {
        unsafe {
            let xv  = vld1q_f32(x.as_ptr().add(j));
            let wv  = vld1q_f32(weight.as_ptr().add(j));
            let res = vmulq_f32(vmulq_f32(xv, scale_v), wv);
            vst1q_f32(out.as_mut_ptr().add(j), res);
        }
        j += 4;
    }
    while j < n {
        out[j] = x[j] * scale * weight[j];
        j += 1;
    }
    for k in n..out.len() {
        out[k] = 0.0;
    }
}

#[cfg(not(target_arch = "aarch64"))]
pub unsafe fn rms_norm_neon(x: &[f32], weight: &[f32], eps: f32, out: &mut [f32]) {
    crate::llama::compute::default_rms_norm(x, weight, eps, out);
}
