//! LLaMA-style model hyperparameters + weight tensors.
//!
//! Memory strategy (important on ≤8 GB machines):
//! - Keep **quantized** weights in RAM (≈ file size, ~1 GB for Q4_K 1.5B)
//! - Dequantize **one matrix at a time** inside `gemv`, then free the f32 buffer
//! - Cap context length so KV cache does not explode (e.g. 32k ctx → multi‑GB)
//!
//! Full dequant of all weights to f32 is **never** done (that was ~6 GB for 1.5B and OOM’d laptops).

use crate::llama::gguf::{load_tensor_f32, GgmlType, GgufFile, TensorInfo};
use std::sync::Arc;

/// Hard cap for pure-Rust KV cache (tokens). Full model ctx (e.g. 32k) is too large.
pub const LLAMA_RS_MAX_CTX: usize = 2048;

#[derive(Debug, Clone)]
pub struct ModelHyperparams {
    pub architecture: String,
    pub n_vocab: usize,
    pub n_embd: usize,
    pub n_layer: usize,
    pub n_ff: usize,
    pub n_head: usize,
    pub n_head_kv: usize,
    pub n_ctx: usize,
    pub rope_dim: usize,
    pub rope_freq_base: f32,
    pub rms_eps: f32,
}

impl ModelHyperparams {
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self, String> {
        let arch = gguf.architecture().unwrap_or("llama").to_string();
        let prefix = arch.as_str();

        let n_embd = gguf
            .meta_u64(&format!("{}.embedding_length", prefix))
            .or_else(|| gguf.meta_u64("llama.embedding_length"))
            .ok_or("missing embedding_length")? as usize;
        let n_layer = gguf
            .meta_u64(&format!("{}.block_count", prefix))
            .or_else(|| gguf.meta_u64("llama.block_count"))
            .ok_or("missing block_count")? as usize;
        let n_ff = gguf
            .meta_u64(&format!("{}.feed_forward_length", prefix))
            .or_else(|| gguf.meta_u64("llama.feed_forward_length"))
            .unwrap_or((n_embd * 4) as u64) as usize;
        let n_head = gguf
            .meta_u64(&format!("{}.attention.head_count", prefix))
            .or_else(|| gguf.meta_u64("llama.attention.head_count"))
            .ok_or("missing attention.head_count")? as usize;
        let n_head_kv = gguf
            .meta_u64(&format!("{}.attention.head_count_kv", prefix))
            .or_else(|| gguf.meta_u64("llama.attention.head_count_kv"))
            .unwrap_or(n_head as u64) as usize;
        let n_ctx_model = gguf
            .meta_u64(&format!("{}.context_length", prefix))
            .or_else(|| gguf.meta_u64("llama.context_length"))
            .unwrap_or(2048) as usize;
        // Cap ctx for pure-Rust KV (user can use llama.cpp for long context)
        let n_ctx = n_ctx_model.min(LLAMA_RS_MAX_CTX);
        let rope_dim = gguf
            .meta_u64(&format!("{}.rope.dimension_count", prefix))
            .or_else(|| gguf.meta_u64("llama.rope.dimension_count"))
            .unwrap_or((n_embd / n_head.max(1)) as u64) as usize;
        let rope_freq_base = gguf
            .meta_f32(&format!("{}.rope.freq_base", prefix))
            .or_else(|| gguf.meta_f32("llama.rope.freq_base"))
            .unwrap_or(10000.0);
        let rms_eps = gguf
            .meta_f32(&format!("{}.attention.layer_norm_rms_epsilon", prefix))
            .or_else(|| gguf.meta_f32("llama.attention.layer_norm_rms_epsilon"))
            .unwrap_or(1e-5);

        let n_vocab = gguf
            .metadata
            .get("tokenizer.ggml.tokens")
            .and_then(|v| match v {
                crate::llama::gguf::MetaValue::Array(a) => Some(a.len()),
                _ => None,
            })
            .or_else(|| {
                gguf.tensor("output.weight")
                    .or_else(|| gguf.tensor("token_embd.weight"))
                    .map(|t| t.dims.last().copied().unwrap_or(0) as usize)
                    .filter(|&n| n > 0)
            })
            .unwrap_or(0);

        if n_vocab == 0 {
            return Err("Could not determine vocabulary size".into());
        }

        Ok(Self {
            architecture: arch,
            n_vocab,
            n_embd,
            n_layer,
            n_ff,
            n_head,
            n_head_kv,
            n_ctx,
            rope_dim,
            rope_freq_base,
            rms_eps,
        })
    }

    pub fn head_dim(&self) -> usize {
        self.n_embd / self.n_head.max(1)
    }

    /// Rough bytes if everything were expanded to f32 (what used to OOM laptops).
    pub fn estimate_full_f32_bytes(&self) -> u64 {
        let emb = (self.n_vocab as u64) * (self.n_embd as u64) * 4;
        // per layer rough: q,k,v,o + gate,up,down
        let attn = 4u64 * (self.n_embd as u64) * (self.n_embd as u64) * 4;
        let ffn = 3u64 * (self.n_embd as u64) * (self.n_ff as u64) * 4;
        emb * 2 + (self.n_layer as u64) * (attn + ffn)
    }
}

/// Dense f32 matrix stored row-major: shape (rows, cols).
#[derive(Debug, Clone)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f32>,
}

impl Matrix {
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: vec![0.0; rows * cols],
        }
    }

    #[inline]
    pub fn get(&self, r: usize, c: usize) -> f32 {
        self.data[r * self.cols + c]
    }

    /// y = W * x  with W (rows, cols), x len=cols, y len=rows
    pub fn gemv(&self, x: &[f32], y: &mut [f32]) {
        assert_eq!(x.len(), self.cols);
        assert_eq!(y.len(), self.rows);
        for r in 0..self.rows {
            let mut sum = 0.0f32;
            let row = &self.data[r * self.cols..(r + 1) * self.cols];
            for c in 0..self.cols {
                sum += row[c] * x[c];
            }
            y[r] = sum;
        }
    }
}

/// Weight matrix kept quantized in RAM; dequantized only for the duration of one `gemv`.
#[derive(Debug, Clone)]
pub struct QuantMatrix {
    pub rows: usize,
    pub cols: usize,
    pub ggml_type: GgmlType,
    /// Packed quantized payload (or dense f32 little-endian if type is F32).
    pub raw: Arc<Vec<u8>>,
    pub n_elements: usize,
}

impl QuantMatrix {
    pub fn from_gguf(gguf: &GgufFile, name: &str) -> Result<Self, String> {
        let info = gguf
            .tensor(name)
            .ok_or_else(|| format!("Missing tensor '{}'", name))?;
        Self::from_tensor(gguf, info)
    }

    pub fn from_tensor(gguf: &GgufFile, info: &TensorInfo) -> Result<Self, String> {
        if !info.ggml_type.is_loadable() {
            return Err(format!(
                "Tensor '{}' type {} not supported in llama.rs",
                info.name,
                info.ggml_type.name()
            ));
        }
        let dims = &info.dims;
        let (rows, cols) = match dims.len() {
            1 => (dims[0] as usize, 1),
            2 => (dims[1] as usize, dims[0] as usize),
            _ => {
                let cols = dims[0] as usize;
                let rows = dims[1..].iter().product::<u64>() as usize;
                (rows, cols)
            }
        };
        let n_elements = info.n_elements() as usize;
        if n_elements != rows * cols && n_elements != cols * rows {
            return Err(format!(
                "Tensor '{}' shape mismatch: dims={:?} n={}",
                info.name, dims, n_elements
            ));
        }
        let (rows, cols) = if n_elements == rows * cols {
            (rows, cols)
        } else {
            (cols, rows)
        };

        let raw = gguf
            .tensor_bytes(info)
            .map_err(|e| e.to_string())?
            .to_vec();

        Ok(Self {
            rows,
            cols,
            ggml_type: info.ggml_type,
            raw: Arc::new(raw),
            n_elements,
        })
    }

    /// Dequantize full matrix to f32 (temporary — free after use).
    pub fn to_f32_matrix(&self) -> Result<Matrix, String> {
        // Build a synthetic TensorInfo for the dequant helpers
        let info = TensorInfo {
            name: String::new(),
            dims: vec![self.cols as u64, self.rows as u64],
            ggml_type: self.ggml_type,
            offset: 0,
        };
        let data = dequant_raw(&self.raw, &info, self.n_elements)?;
        if data.len() != self.rows * self.cols {
            return Err(format!(
                "Dequant size mismatch: got {} want {}",
                data.len(),
                self.rows * self.cols
            ));
        }
        Ok(Matrix {
            rows: self.rows,
            cols: self.cols,
            data,
        })
    }

    /// Matvec via pure-Rust [`crate::llama::ComputeBackend`] (no C/FFI).
    pub fn gemv(
        &self,
        x: &[f32],
        y: &mut [f32],
        compute: &dyn crate::llama::ComputeBackend,
    ) -> Result<(), String> {
        compute
            .gemv_quant(
                self.ggml_type,
                &self.raw,
                self.rows,
                self.cols,
                self.n_elements,
                x,
                y,
            )
            .map_err(|e| e.0)
    }

    /// Legacy full-dequant matvec (tests / fallback).
    pub fn gemv_dequant_fallback(&self, x: &[f32], y: &mut [f32]) -> Result<(), String> {
        let m = self.to_f32_matrix()?;
        m.gemv(x, y);
        Ok(())
    }

    /// Approx permanent RAM for this tensor (quantized).
    pub fn resident_bytes(&self) -> usize {
        self.raw.len()
    }
}

/// Dequantize from an in-memory raw slice (no file offset).
fn dequant_raw(raw: &[u8], info: &TensorInfo, n: usize) -> Result<Vec<f32>, String> {
    // Reuse load_tensor_f32 path by temporarily wrapping — call type-specific via public API
    // We construct a minimal path: duplicate logic by using a helper on GgufFile is awkward;
    // instead call the same functions through a thin fake.
    use crate::llama::gguf::dequant_buffer;
    dequant_buffer(raw, info.ggml_type, n).map_err(|e| e.to_string())
}

#[derive(Debug)]
pub struct TransformerLayer {
    pub attn_norm: Vec<f32>,
    pub wq: QuantMatrix,
    pub wk: QuantMatrix,
    pub wv: QuantMatrix,
    pub wo: QuantMatrix,
    pub ffn_norm: Vec<f32>,
    pub w_gate: QuantMatrix,
    pub w_up: QuantMatrix,
    pub w_down: QuantMatrix,
}

#[derive(Debug)]
pub struct LlamaModel {
    pub hparams: ModelHyperparams,
    /// Token embeddings kept as f32 rows when small enough; else quant + row dequant via full temp.
    pub tok_embeddings: Embeddings,
    pub output_norm: Vec<f32>,
    pub output: QuantMatrix,
    pub layers: Vec<TransformerLayer>,
    /// Sum of quantized weight bytes (for diagnostics).
    pub quant_bytes: usize,
}

/// Embedding table: prefer compact f32 table only if it fits a budget.
#[derive(Debug)]
pub enum Embeddings {
    /// rows = vocab, cols = n_embd
    Dense(Matrix),
    /// Quantized full table; lookup dequants whole table temporarily then takes one row (slow but low resident).
    Quant(QuantMatrix),
}

impl Embeddings {
    pub fn lookup(&self, token: u32, n_embd: usize) -> Result<Vec<f32>, String> {
        match self {
            Embeddings::Dense(m) => {
                if m.cols == n_embd {
                    let t = token as usize;
                    if t >= m.rows {
                        return Err(format!("Token id {} out of range", token));
                    }
                    Ok(m.data[t * n_embd..(t + 1) * n_embd].to_vec())
                } else if m.rows == n_embd {
                    let t = token as usize;
                    if t >= m.cols {
                        return Err(format!("Token id {} out of range", token));
                    }
                    let mut x = vec![0.0f32; n_embd];
                    for r in 0..n_embd {
                        x[r] = m.get(r, t);
                    }
                    Ok(x)
                } else {
                    Err(format!("Unexpected emb shape {}x{}", m.rows, m.cols))
                }
            }
            Embeddings::Quant(q) => {
                // Dequant only the needed row/column — never the full embedding table.
                let t = token as usize;
                if q.cols == n_embd {
                    if t >= q.rows {
                        return Err(format!("Token id {} out of range", token));
                    }
                    dequant_row_slice(q, t * n_embd, n_embd)
                } else if q.rows == n_embd {
                    if t >= q.cols {
                        return Err(format!("Token id {} out of range", token));
                    }
                    // Column t of a (n_embd × vocab) layout — gather via full dequant of
                    // only that column by streaming fused indices.
                    dequant_column_slice(q, t, n_embd)
                } else {
                    Err(format!("Unexpected emb shape {}x{}", q.rows, q.cols))
                }
            }
        }
    }
}

/// Soft budget for keeping embeddings permanently as f32 (~400 MB).
const EMB_F32_BUDGET: usize = 400 * 1024 * 1024;

/// Dequantize a contiguous row-major slice of a quant matrix (one emb row).
fn dequant_row_slice(q: &QuantMatrix, start: usize, len: usize) -> Result<Vec<f32>, String> {
    crate::llama::kernels::dequant_slice(&q.raw, q.ggml_type, q.n_elements, start, len)
}

/// Gather one column without materializing the full f32 table.
fn dequant_column_slice(q: &QuantMatrix, col: usize, n_rows: usize) -> Result<Vec<f32>, String> {
    // Column-major-ish emb: element at (r, col) is at linear index r if cols==vocab? 
    // Stored as rows=n_embd, cols=vocab → index = r * cols + col
    let mut out = vec![0.0f32; n_rows];
    for r in 0..n_rows {
        let idx = r * q.cols + col;
        let one = crate::llama::kernels::dequant_slice(
            &q.raw,
            q.ggml_type,
            q.n_elements,
            idx,
            1,
        )?;
        out[r] = one[0];
    }
    Ok(out)
}

impl LlamaModel {
    pub fn load(gguf: &GgufFile) -> Result<Self, String> {
        let hparams = ModelHyperparams::from_gguf(gguf)?;

        for t in &gguf.tensors {
            if !t.ggml_type.is_loadable() {
                return Err(format!(
                    "Tensor '{}' uses {} which llama.rs cannot dequantize yet. \
                     Use the llama.cpp engine for this quant type.",
                    t.name,
                    t.ggml_type.name()
                ));
            }
        }

        let emb_q = QuantMatrix::from_gguf(gguf, "token_embd.weight")
            .or_else(|_| QuantMatrix::from_gguf(gguf, "tok_embeddings.weight"))?;

        // Optional dense emb if small enough after dequant
        let emb_f32_size = emb_q.rows * emb_q.cols * 4;
        let tok_embeddings = if emb_f32_size <= EMB_F32_BUDGET {
            let m = emb_q.to_f32_matrix()?;
            Embeddings::Dense(m)
        } else {
            // Keep quantized (resident ~file slice); lookups are slower
            Embeddings::Quant(emb_q)
        };

        let output_norm = load_norm(gguf, "output_norm.weight")?;
        let output = QuantMatrix::from_gguf(gguf, "output.weight").or_else(|_| {
            // Tied embeddings: reuse emb quant if we still have it
            QuantMatrix::from_gguf(gguf, "token_embd.weight")
                .or_else(|_| QuantMatrix::from_gguf(gguf, "tok_embeddings.weight"))
        })?;

        let mut quant_bytes = output.resident_bytes();
        if let Embeddings::Quant(ref q) = tok_embeddings {
            quant_bytes += q.resident_bytes();
        }

        let mut layers = Vec::with_capacity(hparams.n_layer);
        for i in 0..hparams.n_layer {
            let attn_norm = load_norm(gguf, &format!("blk.{}.attn_norm.weight", i))?;
            let ffn_norm = load_norm(gguf, &format!("blk.{}.ffn_norm.weight", i))?;
            let wq = QuantMatrix::from_gguf(gguf, &format!("blk.{}.attn_q.weight", i))?;
            let wk = QuantMatrix::from_gguf(gguf, &format!("blk.{}.attn_k.weight", i))?;
            let wv = QuantMatrix::from_gguf(gguf, &format!("blk.{}.attn_v.weight", i))?;
            let wo = QuantMatrix::from_gguf(gguf, &format!("blk.{}.attn_output.weight", i))?;
            let w_gate = QuantMatrix::from_gguf(gguf, &format!("blk.{}.ffn_gate.weight", i))?;
            let w_up = QuantMatrix::from_gguf(gguf, &format!("blk.{}.ffn_up.weight", i))?;
            let w_down = QuantMatrix::from_gguf(gguf, &format!("blk.{}.ffn_down.weight", i))?;
            quant_bytes += wq.resident_bytes()
                + wk.resident_bytes()
                + wv.resident_bytes()
                + wo.resident_bytes()
                + w_gate.resident_bytes()
                + w_up.resident_bytes()
                + w_down.resident_bytes();
            layers.push(TransformerLayer {
                attn_norm,
                wq,
                wk,
                wv,
                wo,
                ffn_norm,
                w_gate,
                w_up,
                w_down,
            });
        }

        Ok(Self {
            hparams,
            tok_embeddings,
            output_norm,
            output,
            layers,
            quant_bytes,
        })
    }

    pub fn memory_summary(&self) -> String {
        let kv = self.hparams.n_layer
            * self.hparams.n_ctx
            * self.hparams.n_head_kv
            * self.hparams.head_dim()
            * 2
            * 4;
        let full_f32 = self.hparams.estimate_full_f32_bytes();
        format!(
            "llama.rs RAM: quant_weights≈{:.0} MB, kv_cache≈{:.0} MB (ctx={}), \
             full-f32-would-be≈{:.1} GB (NOT loaded)",
            self.quant_bytes as f64 / 1e6,
            kv as f64 / 1e6,
            self.hparams.n_ctx,
            full_f32 as f64 / 1e9
        )
    }
}

fn load_norm(gguf: &GgufFile, name: &str) -> Result<Vec<f32>, String> {
    let info = gguf
        .tensor(name)
        .ok_or_else(|| format!("Missing tensor '{}'", name))?;
    load_tensor_f32(gguf, info).map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn load_local_gguf_stays_under_2gb_quant() {
        // Portable: $HOME/.local/hercules/model or HERCULES_TEST_GGUF
        let path = std::env::var_os("HERCULES_TEST_GGUF")
            .map(PathBuf::from)
            .or_else(|| {
                std::env::var_os("HOME").map(|h| {
                    PathBuf::from(h)
                        .join(".local/hercules/model/qwen2.5-1.5b-instruct-q4_k_m.gguf")
                })
            });
        let Some(path) = path else { return };
        if !path.exists() {
            return;
        }
        let gguf = crate::llama::gguf::GgufFile::open(&path).expect("open gguf");
        let model = LlamaModel::load(&gguf).expect("load model");
        // Must NOT expand to full f32 (~6GB). Quant resident should track file size.
        assert!(
            model.quant_bytes < 1_800_000_000,
            "quant_bytes too high: {} ({})",
            model.quant_bytes,
            model.memory_summary()
        );
        assert!(model.hparams.n_ctx <= LLAMA_RS_MAX_CTX);
        eprintln!("{}", model.memory_summary());
    }
}

/// KV cache: per-layer key/value sequences (n_ctx × n_head_kv × head_dim).
#[derive(Debug)]
pub struct KvCache {
    pub n_layer: usize,
    pub n_head_kv: usize,
    pub head_dim: usize,
    pub n_ctx: usize,
    pub k: Vec<Vec<f32>>,
    pub v: Vec<Vec<f32>>,
    pub n_past: usize,
}

impl KvCache {
    pub fn new(n_layer: usize, n_head_kv: usize, head_dim: usize, n_ctx: usize) -> Self {
        let n_ctx = n_ctx.min(LLAMA_RS_MAX_CTX);
        let slot = n_ctx * n_head_kv.max(1) * head_dim.max(1);
        Self {
            n_layer,
            n_head_kv,
            head_dim,
            n_ctx,
            k: (0..n_layer).map(|_| vec![0.0; slot]).collect(),
            v: (0..n_layer).map(|_| vec![0.0; slot]).collect(),
            n_past: 0,
        }
    }

    pub fn clear(&mut self) {
        self.n_past = 0;
    }
}

// ---- Tensor ops used by the forward pass (llama.cpp style) ----

pub fn rms_norm(x: &[f32], weight: &[f32], eps: f32, out: &mut [f32]) {
    let mut ss = 0.0f32;
    for &v in x {
        ss += v * v;
    }
    ss = (ss / x.len() as f32 + eps).sqrt();
    let inv = 1.0 / ss;
    for i in 0..x.len() {
        out[i] = x[i] * inv * weight.get(i).copied().unwrap_or(1.0);
    }
}

pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

pub fn rope_inplace(
    q: &mut [f32],
    k: &mut [f32],
    head_dim: usize,
    n_head: usize,
    n_head_kv: usize,
    pos: usize,
    rope_dim: usize,
    freq_base: f32,
) {
    for h in 0..n_head {
        let off = h * head_dim;
        if off + head_dim <= q.len() {
            rope_vec(&mut q[off..off + head_dim], pos, rope_dim, freq_base);
        }
    }
    for h in 0..n_head_kv {
        let off = h * head_dim;
        if off + head_dim <= k.len() {
            rope_vec(&mut k[off..off + head_dim], pos, rope_dim, freq_base);
        }
    }
}

fn rope_vec(x: &mut [f32], pos: usize, rope_dim: usize, freq_base: f32) {
    let dim = rope_dim.min(x.len());
    for i in (0..dim).step_by(2) {
        let freq = 1.0 / freq_base.powf((i as f32) / (dim as f32).max(1.0));
        let val = pos as f32 * freq;
        let (sin, cos) = val.sin_cos();
        let x0 = x[i];
        let x1 = if i + 1 < x.len() { x[i + 1] } else { 0.0 };
        x[i] = x0 * cos - x1 * sin;
        if i + 1 < x.len() {
            x[i + 1] = x0 * sin + x1 * cos;
        }
    }
}

/// Forward one token position → logits (n_vocab).
pub fn forward_token(
    model: &LlamaModel,
    cache: &mut KvCache,
    token: u32,
    pos: usize,
    compute: &dyn crate::llama::ComputeBackend,
) -> Result<Vec<f32>, String> {
    let h = &model.hparams;
    let n_embd = h.n_embd;
    let head_dim = h.head_dim();
    if pos >= h.n_ctx {
        return Err(format!(
            "Position {} exceeds llama.rs context cap {} (use llama.cpp for longer ctx)",
            pos, h.n_ctx
        ));
    }

    let mut x = model.tok_embeddings.lookup(token, n_embd)?;

    let mut xb = vec![0.0f32; n_embd];
    let mut q = vec![0.0f32; n_embd];
    let mut k_cur = vec![0.0f32; h.n_head_kv.max(1) * head_dim];
    let mut v_cur = vec![0.0f32; h.n_head_kv.max(1) * head_dim];

    for (il, layer) in model.layers.iter().enumerate() {
        compute.rms_norm(&x, &layer.attn_norm, h.rms_eps, &mut xb);

        let mut q_full = vec![0.0f32; layer.wq.rows];
        let mut k_full = vec![0.0f32; layer.wk.rows];
        let mut v_full = vec![0.0f32; layer.wv.rows];
        layer.wq.gemv(&xb, &mut q_full, compute)?;
        layer.wk.gemv(&xb, &mut k_full, compute)?;
        layer.wv.gemv(&xb, &mut v_full, compute)?;

        q = q_full;
        k_cur = k_full;
        v_cur = v_full;

        rope_inplace(
            &mut q,
            &mut k_cur,
            head_dim,
            h.n_head,
            h.n_head_kv,
            pos,
            h.rope_dim,
            h.rope_freq_base,
        );

        let kv_head_dim = if h.n_head_kv > 0 {
            k_cur.len() / h.n_head_kv
        } else {
            head_dim
        };
        let cache_stride = h.n_head_kv.max(1) * kv_head_dim.max(1);
        if pos < h.n_ctx {
            let off = pos * cache_stride;
            if off + k_cur.len() <= cache.k[il].len() {
                cache.k[il][off..off + k_cur.len()].copy_from_slice(&k_cur);
                let vlen = v_cur.len().min(k_cur.len());
                cache.v[il][off..off + vlen].copy_from_slice(&v_cur[..vlen]);
            }
        }

        let seq_len = pos + 1;
        let mut attn_out = vec![0.0f32; n_embd.max(q.len())];
        let n_rep = if h.n_head_kv > 0 {
            h.n_head / h.n_head_kv.max(1)
        } else {
            1
        };

        for head in 0..h.n_head {
            let kv_head = head / n_rep.max(1);
            let q_off = head * head_dim;
            let scale = 1.0 / (head_dim as f32).sqrt();

            let mut scores = vec![0.0f32; seq_len];
            for t in 0..seq_len {
                let mut dot = 0.0f32;
                let k_base = t * cache_stride + kv_head * kv_head_dim;
                for d in 0..head_dim.min(kv_head_dim) {
                    let qv = q.get(q_off + d).copied().unwrap_or(0.0);
                    let kv = cache.k[il].get(k_base + d).copied().unwrap_or(0.0);
                    dot += qv * kv;
                }
                scores[t] = dot * scale;
            }
            let max_s = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for s in &mut scores {
                *s = (*s - max_s).exp();
                sum += *s;
            }
            for s in &mut scores {
                *s /= sum.max(1e-12);
            }
            for d in 0..head_dim.min(kv_head_dim) {
                let mut acc = 0.0f32;
                for t in 0..seq_len {
                    let v_base = t * cache_stride + kv_head * kv_head_dim;
                    acc += scores[t] * cache.v[il].get(v_base + d).copied().unwrap_or(0.0);
                }
                if q_off + d < attn_out.len() {
                    attn_out[q_off + d] = acc;
                }
            }
        }

        let mut att_proj = vec![0.0f32; layer.wo.rows];
        if attn_out.len() != layer.wo.cols {
            attn_out.resize(layer.wo.cols, 0.0);
        }
        layer.wo.gemv(&attn_out, &mut att_proj, compute)?;
        for i in 0..n_embd.min(att_proj.len()) {
            x[i] += att_proj[i];
        }

        compute.rms_norm(&x, &layer.ffn_norm, h.rms_eps, &mut xb);
        let mut gate = vec![0.0f32; layer.w_gate.rows];
        let mut up = vec![0.0f32; layer.w_up.rows];
        layer.w_gate.gemv(&xb, &mut gate, compute)?;
        layer.w_up.gemv(&xb, &mut up, compute)?;
        let ff_dim = gate.len().min(up.len());
        let mut hb = vec![0.0f32; ff_dim];
        for i in 0..ff_dim {
            hb[i] = silu(gate[i]) * up[i];
        }
        let mut hb2 = vec![0.0f32; layer.w_down.rows];
        if hb.len() != layer.w_down.cols {
            hb.resize(layer.w_down.cols, 0.0);
        }
        layer.w_down.gemv(&hb, &mut hb2, compute)?;
        for i in 0..n_embd.min(hb2.len()) {
            x[i] += hb2[i];
        }
    }

    cache.n_past = pos + 1;

    compute.rms_norm(&x, &model.output_norm, h.rms_eps, &mut xb);
    let mut logits = vec![0.0f32; model.output.rows.max(h.n_vocab)];
    if model.output.cols == n_embd {
        logits.resize(model.output.rows, 0.0);
        model.output.gemv(&xb, &mut logits, compute)?;
    } else if model.output.rows == n_embd {
        // Transposed — dequant once and multiply columns
        let m = model.output.to_f32_matrix()?;
        logits.resize(m.cols, 0.0);
        for c in 0..m.cols {
            let mut sum = 0.0f32;
            for r in 0..n_embd {
                sum += m.get(r, c) * xb[r];
            }
            logits[c] = sum;
        }
    } else {
        return Err(format!(
            "Unexpected output weight shape {}x{}",
            model.output.rows, model.output.cols
        ));
    }

    if logits.len() > h.n_vocab {
        logits.truncate(h.n_vocab);
    }
    Ok(logits)
}
