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
        GgmlType::Q8_0 | GgmlType::Q4_0 | GgmlType::Q4_K | GgmlType::F32 | GgmlType::F16
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
        GgmlType::Q4_K => gemv_q4_k(raw, rows, cols, n, x, y),
        GgmlType::F32 => gemv_f32(raw, rows, cols, n, x, y),
        GgmlType::F16 => gemv_f16(raw, rows, cols, n, x, y),
        other => Err(format!("no fused kernel for {}", other.name())),
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
    fn f32_matches() {
        let rows = 3;
        let cols = 4;
        let mut raw = Vec::new();
        for i in 0..(rows * cols) {
            raw.extend_from_slice(&(i as f32 * 0.1).to_le_bytes());
        }
        check_vs_dequant(GgmlType::F32, &raw, rows, cols);
    }
}
