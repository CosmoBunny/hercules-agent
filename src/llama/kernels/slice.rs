//! Dequantize a contiguous element range without expanding the full tensor.

use crate::llama::gguf::{dequant_buffer, f16_to_f32, GgmlType};

const QK_K: usize = 256;
const K_SCALE_SIZE: usize = 12;

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

/// Dequantize elements `[start, start+len)` into a new `Vec` (length `len`).
pub fn dequant_slice(
    raw: &[u8],
    quant: GgmlType,
    n_elements: usize,
    start: usize,
    len: usize,
) -> Result<Vec<f32>, String> {
    if len == 0 {
        return Ok(Vec::new());
    }
    if start.saturating_add(len) > n_elements {
        return Err(format!(
            "dequant_slice OOB start={start} len={len} n={n_elements}"
        ));
    }
    let end = start + len;
    let mut out = vec![0.0f32; len];

    match quant {
        GgmlType::F32 => {
            for i in 0..len {
                let off = (start + i) * 4;
                out[i] = f32::from_le_bytes([raw[off], raw[off + 1], raw[off + 2], raw[off + 3]]);
            }
            Ok(out)
        }
        GgmlType::F16 | GgmlType::BF16 => {
            for i in 0..len {
                let off = (start + i) * 2;
                out[i] = if quant == GgmlType::F16 {
                    read_f16_le(raw, off)
                } else {
                    let bits = (u16::from_le_bytes([raw[off], raw[off + 1]]) as u32) << 16;
                    f32::from_bits(bits)
                };
            }
            Ok(out)
        }
        GgmlType::Q8_0 => {
            stream_q8_0(raw, start, end, &mut out)?;
            Ok(out)
        }
        GgmlType::Q4_0 => {
            stream_q4_0(raw, start, end, &mut out)?;
            Ok(out)
        }
        GgmlType::Q4_K => {
            stream_q4_k(raw, start, end, &mut out)?;
            Ok(out)
        }
        other => {
            // Fallback: full dequant then slice (rare emb types)
            let full = dequant_buffer(raw, other, n_elements).map_err(|e| e.to_string())?;
            Ok(full[start..end].to_vec())
        }
    }
}

fn stream_q8_0(raw: &[u8], start: usize, end: usize, out: &mut [f32]) -> Result<(), String> {
    let block = 34;
    let mut idx = 0usize;
    let mut o = 0usize;
    while o < end {
        if idx + block > raw.len() {
            return Err("Q8_0 truncated".into());
        }
        let scale = read_f16_le(raw, idx);
        idx += 2;
        for _ in 0..32 {
            if o >= end {
                return Ok(());
            }
            let q = raw[idx] as i8;
            idx += 1;
            if o >= start {
                out[o - start] = scale * (q as f32);
            }
            o += 1;
        }
    }
    Ok(())
}

fn stream_q4_0(raw: &[u8], start: usize, end: usize, out: &mut [f32]) -> Result<(), String> {
    let block = 18;
    let mut idx = 0usize;
    let mut o = 0usize;
    while o < end {
        if idx + block > raw.len() {
            return Err("Q4_0 truncated".into());
        }
        let scale = read_f16_le(raw, idx);
        idx += 2;
        for j in 0..16 {
            if o >= end {
                return Ok(());
            }
            let byte = raw[idx + j];
            let x0 = (byte & 0x0F) as i8 - 8;
            let x1 = (byte >> 4) as i8 - 8;
            if o >= start {
                out[o - start] = scale * (x0 as f32);
            }
            o += 1;
            if o >= end {
                return Ok(());
            }
            if o >= start {
                out[o - start] = scale * (x1 as f32);
            }
            o += 1;
        }
        idx += 16;
    }
    Ok(())
}

fn stream_q4_k(raw: &[u8], start: usize, end: usize, out: &mut [f32]) -> Result<(), String> {
    let block = 144;
    let n = end; // we only need up to end
    let nb = (n + QK_K - 1) / QK_K;
    let mut idx = 0usize;
    let mut o = 0usize;
    for _ in 0..nb {
        if o >= end {
            break;
        }
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
                if o >= end {
                    return Ok(());
                }
                if o >= start {
                    out[o - start] = d1 * ((q[l] & 0x0F) as f32) - m1;
                }
                o += 1;
            }
            for l in 0..32 {
                if o >= end {
                    return Ok(());
                }
                if o >= start {
                    out[o - start] = d2 * ((q[l] >> 4) as f32) - m2;
                }
                o += 1;
            }
            q = &q[32..];
            is += 2;
        }
    }
    Ok(())
}
