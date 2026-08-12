//! Fused quant GEMV: dequant block → accumulate, no full-matrix alloc.

use crate::llama::gguf::{f16_to_f32, GgmlType};

const QK_K: usize = 256;
const K_SCALE_SIZE: usize = 12;

#[inline]
fn read_f16_le(raw: &[u8], idx: usize) -> f32 {
    f16_to_f32(u16::from_le_bytes([raw[idx], raw[idx + 1]]))
}

/// Packed 6-bit scale/min (must match `gguf::get_scale_min_k4`).
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

pub fn supports_fused_gemv(t: GgmlType) -> bool {
    matches!(
        t,
        GgmlType::Q8_0
            | GgmlType::Q4_0
            | GgmlType::Q5_0
            | GgmlType::Q4_K
            | GgmlType::Q5_K
            | GgmlType::Q6_K
            | GgmlType::F32
            | GgmlType::F16
    )
}

/// `y = W · x` with W quantized row-major (`rows × cols`).
pub fn gemv_quant_fused(
    quant: GgmlType,
    raw: &[u8],
    rows: usize,
    cols: usize,
    n_elements: usize,
    x: &[f32],
    y: &mut [f32],
) -> Result<(), String> {
    if x.len() != cols || y.len() != rows {
        return Err(format!(
            "gemv shape: x={} cols={} y={} rows={}",
            x.len(),
            cols,
            y.len(),
            rows
        ));
    }
    let n = n_elements.min(rows * cols);
    y.fill(0.0);

    match quant {
        GgmlType::Q8_0 => gemv_q8_0(raw, rows, cols, n, x, y),
        GgmlType::Q4_0 => gemv_q4_0(raw, rows, cols, n, x, y),
        GgmlType::Q5_0 => gemv_q5_0(raw, rows, cols, n, x, y),
        GgmlType::Q4_K => gemv_q4_k(raw, rows, cols, n, x, y),
        GgmlType::Q5_K => gemv_q5_k(raw, rows, cols, n, x, y),
        GgmlType::Q6_K => gemv_q6_k(raw, rows, cols, n, x, y),
        GgmlType::F32 => gemv_f32(raw, rows, cols, n, x, y),
        GgmlType::F16 => gemv_f16(raw, rows, cols, n, x, y),
        _ => {
            let data = crate::llama::gguf::dequant_buffer(raw, quant, n)
                .map_err(|e| e.to_string())?;
            if data.len() < rows * cols {
                return Err(format!(
                    "dequant size {} < rows*cols {}",
                    data.len(),
                    rows * cols
                ));
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
}

fn gemv_f32(
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
    for r in 0..rows {
        let mut sum = 0.0f32;
        let base = r * cols;
        for c in 0..cols {
            let i = base + c;
            if i >= n {
                break;
            }
            let off = i * 4;
            let w = f32::from_le_bytes([raw[off], raw[off + 1], raw[off + 2], raw[off + 3]]);
            sum += w * x[c];
        }
        y[r] = sum;
    }
    Ok(())
}

fn gemv_f16(
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
        let mut sum = 0.0f32;
        let base = r * cols;
        for c in 0..cols {
            let i = base + c;
            if i >= n {
                break;
            }
            let off = i * 2;
            let w = read_f16_le(raw, off);
            sum += w * x[c];
        }
        y[r] = sum;
    }
    Ok(())
}

/// Stream Q8_0 blocks (32 qs + f16 scale).
fn gemv_q8_0(
    raw: &[u8],
    rows: usize,
    cols: usize,
    n: usize,
    x: &[f32],
    y: &mut [f32],
) -> Result<(), String> {
    let block = 34;
    let mut idx = 0usize;
    let mut o = 0usize;
    while o < n {
        if idx + block > raw.len() {
            return Err("Q8_0 truncated".into());
        }
        let scale = read_f16_le(raw, idx);
        idx += 2;
        for _ in 0..32 {
            if o >= n {
                break;
            }
            let q = raw[idx] as i8;
            idx += 1;
            let r = o / cols;
            let c = o % cols;
            if r < rows && c < cols {
                y[r] += scale * (q as f32) * x[c];
            }
            o += 1;
        }
    }
    Ok(())
}

fn gemv_q4_0(
    raw: &[u8],
    rows: usize,
    cols: usize,
    n: usize,
    x: &[f32],
    y: &mut [f32],
) -> Result<(), String> {
    let block = 18;
    let mut idx = 0usize;
    let mut o = 0usize;
    while o < n {
        if idx + block > raw.len() {
            return Err("Q4_0 truncated".into());
        }
        let scale = read_f16_le(raw, idx);
        idx += 2;
        for j in 0..16 {
            if o >= n {
                break;
            }
            let byte = raw[idx + j];
            let x0 = (byte & 0x0F) as i8 - 8;
            let x1 = (byte >> 4) as i8 - 8;
            {
                let r = o / cols;
                let c = o % cols;
                if r < rows && c < cols {
                    y[r] += scale * (x0 as f32) * x[c];
                }
                o += 1;
            }
            if o < n {
                let r = o / cols;
                let c = o % cols;
                if r < rows && c < cols {
                    y[r] += scale * (x1 as f32) * x[c];
                }
                o += 1;
            }
        }
        idx += 16;
    }
    Ok(())
}

/// Q5_0 blocks of 32 (d f16 + qh u32 + qs[16]) = 22B/block.
/// Each element: 5-bit unsigned (low nibble | high bit << 4), bias -16.
/// y[j+0 ] = d * ((qs[j] & 0x0F | ((qh>>j      & 1) << 4)) - 16)
/// y[j+16] = d * ((qs[j] >> 4  | ((qh>>(j+16)  & 1) << 4)) - 16)
fn gemv_q5_0(
    raw: &[u8],
    rows: usize,
    cols: usize,
    n: usize,
    x: &[f32],
    y: &mut [f32],
) -> Result<(), String> {
    let block = 22; // 2B d + 4B qh + 16B qs
    let mut idx = 0usize;
    let mut base = 0usize; // element offset into the full weight matrix

    while base < n {
        if idx + block > raw.len() {
            return Err(format!(
                "Q5_0 truncated at block offset {} (raw len={})",
                idx, raw.len()
            ));
        }
        let d = read_f16_le(raw, idx);
        let qh = u32::from_le_bytes([raw[idx+2], raw[idx+3], raw[idx+4], raw[idx+5]]);
        let qs = &raw[idx+6..idx+22];
        idx += block;

        for j in 0..16usize {
            // Lower half element (offset j)
            {
                let elem = base + j;
                if elem < n {
                    let xh = ((qh >> j) & 1) as i32;
                    let v = ((qs[j] & 0x0F) as i32) | (xh << 4);
                    let w = d * ((v - 16) as f32);
                    let r = elem / cols;
                    let c = elem % cols;
                    if r < rows && c < cols {
                        y[r] += w * x[c];
                    }
                }
            }
            // Upper half element (offset j+16)
            {
                let elem = base + 16 + j;
                if elem < n {
                    let xh = ((qh >> (j + 16)) & 1) as i32;
                    let v = ((qs[j] >> 4) as i32) | (xh << 4);
                    let w = d * ((v - 16) as f32);
                    let r = elem / cols;
                    let c = elem % cols;
                    if r < rows && c < cols {
                        y[r] += w * x[c];
                    }
                }
            }
        }
        base += 32;
    }
    Ok(())
}

/// Q4_K superblocks of 256 (matches `dequant_q4_k` layout).
fn gemv_q4_k(
    raw: &[u8],
    rows: usize,
    cols: usize,
    n: usize,
    x: &[f32],
    y: &mut [f32],
) -> Result<(), String> {
    let block = 144;
    let nb = (n + QK_K - 1) / QK_K;
    let mut idx = 0usize;
    let mut o = 0usize;
    for _ in 0..nb {
        if idx + block > raw.len() {
            return Err("Q4_K truncated".into());
        }
        let d = read_f16_le(raw, idx);
        let dmin = read_f16_le(raw, idx + 2);
        let scales = &raw[idx + 4..idx + 4 + K_SCALE_SIZE];
        let mut q = &raw[idx + 16..idx + 16 + 128];
        idx += block;

        let mut is = 0usize;
        for _j in 0..4 {
            let (sc, m) = get_scale_min_k4(is, scales);
            let d1 = d * (sc as f32);
            let m1 = dmin * (m as f32);
            let (sc, m) = get_scale_min_k4(is + 1, scales);
            let d2 = d * (sc as f32);
            let m2 = dmin * (m as f32);
            for l in 0..32 {
                if o >= n {
                    return Ok(());
                }
                let w = d1 * ((q[l] & 0x0F) as f32) - m1;
                let r = o / cols;
                let c = o % cols;
                if r < rows && c < cols {
                    y[r] += w * x[c];
                }
                o += 1;
            }
            for l in 0..32 {
                if o >= n {
                    return Ok(());
                }
                let w = d2 * ((q[l] >> 4) as f32) - m2;
                let r = o / cols;
                let c = o % cols;
                if r < rows && c < cols {
                    y[r] += w * x[c];
                }
                o += 1;
            }
            q = &q[32..];
            is += 2;
        }
    }
    Ok(())
}

/// Q5_K superblocks of 256 (matches `dequant_q5_k` layout).
fn gemv_q5_k(
    raw: &[u8],
    rows: usize,
    cols: usize,
    n: usize,
    x: &[f32],
    y: &mut [f32],
) -> Result<(), String> {
    let block = 176;
    let nb = (n + QK_K - 1) / QK_K;
    let mut idx = 0usize;
    let mut o = 0usize;
    for _ in 0..nb {
        if idx + block > raw.len() {
            return Err("Q5_K truncated".into());
        }
        let d = read_f16_le(raw, idx);
        let dmin = read_f16_le(raw, idx + 2);
        let scales = &raw[idx + 4..idx + 4 + K_SCALE_SIZE];
        let qh = &raw[idx + 16..idx + 16 + 32];
        let mut ql = &raw[idx + 48..idx + 48 + 128];
        idx += block;

        let mut is = 0usize;
        let mut u1: u8 = 1;
        let mut u2: u8 = 2;
        for _j in 0..4 {
            let (sc, m) = get_scale_min_k4(is, scales);
            let d1 = d * (sc as f32);
            let m1 = dmin * (m as f32);
            let (sc, m) = get_scale_min_k4(is + 1, scales);
            let d2 = d * (sc as f32);
            let m2 = dmin * (m as f32);
            for l in 0..32 {
                if o >= n {
                    return Ok(());
                }
                let qv = (ql[l] & 0x0F) as u32 + if (qh[l] & u1) != 0 { 16 } else { 0 };
                let w = d1 * (qv as f32) - m1;
                let r = o / cols;
                let c = o % cols;
                if r < rows && c < cols {
                    y[r] += w * x[c];
                }
                o += 1;
            }
            for l in 0..32 {
                if o >= n {
                    return Ok(());
                }
                let qv = (ql[l] >> 4) as u32 + if (qh[l] & u2) != 0 { 16 } else { 0 };
                let w = d2 * (qv as f32) - m2;
                let r = o / cols;
                let c = o % cols;
                if r < rows && c < cols {
                    y[r] += w * x[c];
                }
                o += 1;
            }
            ql = &ql[32..];
            is += 2;
            u1 <<= 2;
            u2 <<= 2;
        }
    }
    Ok(())
}

/// Q6_K superblocks of 256 (ql[128], qh[64], i8 scales[16], f16 d → 210B/block).
fn gemv_q6_k(
    raw: &[u8],
    rows: usize,
    cols: usize,
    n: usize,
    x: &[f32],
    y: &mut [f32],
) -> Result<(), String> {
    let block = 210;
    let nb = (n + QK_K - 1) / QK_K;
    let mut idx = 0usize;
    let mut o = 0usize;
    for _ in 0..nb {
        if idx + block > raw.len() {
            return Err("Q6_K truncated".into());
        }
        // ql[0..128], qh[128..192], sc[192..208], d at [208..210]
        let d = read_f16_le(raw, idx + 208);
        let mut ql = &raw[idx..idx + 128];
        let mut qh = &raw[idx + 128..idx + 192];
        let mut sc = &raw[idx + 192..idx + 208];
        idx += block;

        // Two passes of 128 elements each (matches llama.cpp layout)
        for _n128 in 0..2 {
            for l in 0..32 {
                // Four values per index: q1,q2,q3,q4
                let q1 = ((ql[l] & 0x0F) | (((qh[l] >> 0) & 3) << 4)) as i8 as i32 - 32;
                let q2 = ((ql[l + 32] & 0x0F) | (((qh[l] >> 2) & 3) << 4)) as i8 as i32 - 32;
                let q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) as i8 as i32 - 32;
                let q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) as i8 as i32 - 32;

                let is = l / 16;
                let s0 = sc[is] as i8 as f32;
                let s2 = sc[is + 2] as i8 as f32;
                let s4 = sc[is + 4] as i8 as f32;
                let s6 = sc[is + 6] as i8 as f32;

                for (q, s, off) in [
                    (q1, s0, 0usize),
                    (q2, s2, 32usize),
                    (q3, s4, 64usize),
                    (q4, s6, 96usize),
                ] {
                    let elem_idx = o + off + l;
                    if elem_idx >= n {
                        continue;
                    }
                    let r = elem_idx / cols;
                    let c = elem_idx % cols;
                    if r < rows && c < cols {
                        y[r] += d * s * (q as f32) * x[c];
                    }
                }
            }
            o += 128;
            ql = &ql[64..];
            qh = &qh[32..];
            sc = &sc[8..];
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llama::gguf::dequant_buffer;

    fn check_vs_dequant(quant: GgmlType, raw: &[u8], rows: usize, cols: usize) {
        let n = rows * cols;
        let w = dequant_buffer(raw, quant, n).expect("dequant");
        let x: Vec<f32> = (0..cols).map(|i| (i as f32 * 0.01) + 0.5).collect();
        let mut y_ref = vec![0.0f32; rows];
        for r in 0..rows {
            let mut s = 0.0f32;
            for c in 0..cols {
                s += w[r * cols + c] * x[c];
            }
            y_ref[r] = s;
        }
        let mut y = vec![0.0f32; rows];
        gemv_quant_fused(quant, raw, rows, cols, n, &x, &mut y).unwrap();
        for r in 0..rows {
            let err = (y[r] - y_ref[r]).abs();
            assert!(
                err < 1e-3 * (1.0 + y_ref[r].abs()),
                "row {r}: fused {} ref {} err {}",
                y[r],
                y_ref[r],
                err
            );
        }
    }

    #[test]
    fn q8_0_matches_dequant_gemv() {
        // 2 rows × 32 cols = 64 weights = 2 Q8_0 blocks
        let rows = 2;
        let cols = 32;
        let mut raw = Vec::new();
        for b in 0..2i8 {
            raw.extend_from_slice(&0x3C00u16.to_le_bytes()); // scale = 1.0 f16
            for i in 0..32i8 {
                raw.push(i.wrapping_add(b) as u8);
            }
        }
        check_vs_dequant(GgmlType::Q8_0, &raw, rows, cols);
    }

    #[test]
    fn q5_k_matches_dequant_gemv() {
        // 1 row x 256 cols = 256 weights = 1 Q5_K block (176 bytes)
        let rows = 1;
        let cols = 256;
        let mut raw = vec![0u8; 176];
        // d = 1.0 f16 (0x3C00)
        raw[0] = 0x00;
        raw[1] = 0x3C;
        // dmin = 0.5 f16 (0x3800)
        raw[2] = 0x00;
        raw[3] = 0x38;
        // scales (12 bytes)
        for i in 4..16 {
            raw[i] = (i * 7) as u8;
        }
        // qh (32 bytes)
        for i in 16..48 {
            raw[i] = (i * 13) as u8;
        }
        // ql (128 bytes)
        for i in 48..176 {
            raw[i] = (i * 17) as u8;
        }
        check_vs_dequant(GgmlType::Q5_K, &raw, rows, cols);
    }

    #[test]
    fn f32_matches() {
        let rows = 3;
        let cols = 4;
        let mut raw = Vec::new();
        for i in 0..(rows * cols) {
            raw.extend_from_slice(&(i as f32 * 0.1).to_le_bytes());
        }
        check_vs_dequant(GgmlType::F32, &raw, rows, cols);
    }

    #[test]
    fn q6_k_matches_dequant_gemv() {
        // 1 row × 256 cols = 256 weights = 1 Q6_K block (210 bytes)
        // Layout: ql[0..128], qh[128..192], sc[192..208], d f16 at [208..210]
        let rows = 1;
        let cols = 256;
        let mut raw = vec![0u8; 210];
        // d = 1.0 f16 (0x3C00)
        raw[208] = 0x00;
        raw[209] = 0x3C;
        // scales (i8): use varied values
        for i in 192..208 {
            raw[i] = (i as u8).wrapping_mul(7).wrapping_sub(50);
        }
        // ql (128 bytes): random-ish pattern
        for i in 0..128 {
            raw[i] = (i as u8).wrapping_mul(13).wrapping_add(3);
        }
        // qh (64 bytes)
        for i in 128..192 {
            raw[i] = (i as u8).wrapping_mul(5).wrapping_add(7);
        }
        check_vs_dequant(GgmlType::Q6_K, &raw, rows, cols);
    }

    #[test]
    fn q5_0_matches_dequant_gemv() {
        // 1 row × 32 cols = 32 weights = 1 Q5_0 block (22 bytes)
        let rows = 1;
        let cols = 32;
        let mut raw = vec![0u8; 22];
        // scale d = 1.0 f16 (0x3C00)
        raw[0] = 0x00;
        raw[1] = 0x3C;
        // qh (u32)
        raw[2] = 0xAA;
        raw[3] = 0x55;
        raw[4] = 0x12;
        raw[5] = 0x34;
        // qs (16 bytes)
        for i in 0..16 {
            raw[6 + i] = (i as u8).wrapping_mul(17);
        }
        check_vs_dequant(GgmlType::Q5_0, &raw, rows, cols);
    }
}

