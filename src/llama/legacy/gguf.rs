//! GGUF (GPT-Generated Unified Format) reader.
//!
//! Mirrors how llama.cpp loads models: parse magic/version, metadata KV map,
//! tensor directory, then mmap-friendly weight region offsets.
//! Spec: https://github.com/ggml-org/ggml/blob/master/docs/gguf.md

use std::collections::HashMap;
use std::fs::File;
use std::io::{self, Read};
use std::path::{Path, PathBuf};

pub const GGUF_MAGIC: u32 = 0x4655_4747; // "GGUF" little-endian
pub const GGUF_VERSION: u32 = 3;
pub const DEFAULT_ALIGNMENT: u64 = 32;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
#[allow(non_camel_case_types)]
pub enum GgmlType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2_K = 10,
    Q3_K = 11,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    Q8_K = 15,
    I8 = 24,
    I16 = 25,
    I32 = 26,
    I64 = 27,
    F64 = 28,
    BF16 = 30,
    Unknown(u32),
}

impl GgmlType {
    pub fn from_u32(v: u32) -> Self {
        match v {
            0 => Self::F32,
            1 => Self::F16,
            2 => Self::Q4_0,
            3 => Self::Q4_1,
            6 => Self::Q5_0,
            7 => Self::Q5_1,
            8 => Self::Q8_0,
            9 => Self::Q8_1,
            10 => Self::Q2_K,
            11 => Self::Q3_K,
            12 => Self::Q4_K,
            13 => Self::Q5_K,
            14 => Self::Q6_K,
            15 => Self::Q8_K,
            24 => Self::I8,
            25 => Self::I16,
            26 => Self::I32,
            27 => Self::I64,
            28 => Self::F64,
            30 => Self::BF16,
            other => Self::Unknown(other),
        }
    }

    pub fn name(self) -> String {
        match self {
            Self::F32 => "F32".into(),
            Self::F16 => "F16".into(),
            Self::Q4_0 => "Q4_0".into(),
            Self::Q4_1 => "Q4_1".into(),
            Self::Q5_0 => "Q5_0".into(),
            Self::Q5_1 => "Q5_1".into(),
            Self::Q8_0 => "Q8_0".into(),
            Self::Q8_1 => "Q8_1".into(),
            Self::Q2_K => "Q2_K".into(),
            Self::Q3_K => "Q3_K".into(),
            Self::Q4_K => "Q4_K".into(),
            Self::Q5_K => "Q5_K".into(),
            Self::Q6_K => "Q6_K".into(),
            Self::Q8_K => "Q8_K".into(),
            Self::I8 => "I8".into(),
            Self::I16 => "I16".into(),
            Self::I32 => "I32".into(),
            Self::I64 => "I64".into(),
            Self::F64 => "F64".into(),
            Self::BF16 => "BF16".into(),
            Self::Unknown(v) => format!("Unknown({})", v),
        }
    }

    /// Whether llama.rs can dequantize / load this type into f32 weights today.
    pub fn is_loadable(self) -> bool {
        matches!(
            self,
            Self::F32
                | Self::F16
                | Self::BF16
                | Self::Q4_0
                | Self::Q4_1
                | Self::Q5_0
                | Self::Q5_1
                | Self::Q8_0
                | Self::Q2_K
                | Self::Q3_K
                | Self::Q4_K
                | Self::Q5_K
                | Self::Q6_K
                | Self::Q8_K
        )
    }
}

#[derive(Debug, Clone)]
pub enum MetaValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    String(String),
    Array(Vec<MetaValue>),
    U64(u64),
    I64(i64),
    F64(f64),
}

impl MetaValue {
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Self::U8(v) => Some(*v as u64),
            Self::U16(v) => Some(*v as u64),
            Self::U32(v) => Some(*v as u64),
            Self::U64(v) => Some(*v),
            Self::I8(v) if *v >= 0 => Some(*v as u64),
            Self::I16(v) if *v >= 0 => Some(*v as u64),
            Self::I32(v) if *v >= 0 => Some(*v as u64),
            Self::I64(v) if *v >= 0 => Some(*v as u64),
            _ => None,
        }
    }

    pub fn as_f32(&self) -> Option<f32> {
        match self {
            Self::F32(v) => Some(*v),
            Self::F64(v) => Some(*v as f32),
            Self::U32(v) => Some(*v as f32),
            Self::U64(v) => Some(*v as f32),
            _ => None,
        }
    }

    pub fn as_string_array(&self) -> Option<Vec<String>> {
        match self {
            Self::Array(items) => {
                let mut out = Vec::with_capacity(items.len());
                for it in items {
                    out.push(it.as_str()?.to_string());
                }
                Some(out)
            }
            _ => None,
        }
    }

    pub fn as_f32_array(&self) -> Option<Vec<f32>> {
        match self {
            Self::Array(items) => {
                let mut out = Vec::with_capacity(items.len());
                for it in items {
                    out.push(it.as_f32()?);
                }
                Some(out)
            }
            _ => None,
        }
    }

    pub fn as_i32_array(&self) -> Option<Vec<i32>> {
        match self {
            Self::Array(items) => {
                let mut out = Vec::with_capacity(items.len());
                for it in items {
                    match it {
                        MetaValue::I32(v) => out.push(*v),
                        MetaValue::U32(v) => out.push(*v as i32),
                        MetaValue::I64(v) => out.push(*v as i32),
                        MetaValue::U64(v) => out.push(*v as i32),
                        MetaValue::I8(v) => out.push(*v as i32),
                        MetaValue::U8(v) => out.push(*v as i32),
                        _ => return None,
                    }
                }
                Some(out)
            }
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub dims: Vec<u64>,
    pub ggml_type: GgmlType,
    /// Offset relative to the start of the tensor data section.
    pub offset: u64,
}

impl TensorInfo {
    pub fn n_elements(&self) -> u64 {
        self.dims.iter().product()
    }
}

#[derive(Debug)]
pub struct GgufFile {
    pub path: PathBuf,
    pub version: u32,
    pub alignment: u64,
    pub metadata: HashMap<String, MetaValue>,
    pub tensors: Vec<TensorInfo>,
    pub tensor_index: HashMap<String, usize>,
    /// Absolute file offset where tensor binary data begins.
    pub tensor_data_offset: u64,
    /// Raw file bytes for weight loading (owned for simplicity).
    data: Vec<u8>,
}

impl GgufFile {
    pub fn open(path: impl AsRef<Path>) -> io::Result<Self> {
        let path = path.as_ref().to_path_buf();
        let mut file = File::open(&path)?;
        let mut data = Vec::new();
        file.read_to_end(&mut data)?;
        Self::parse(path, data)
    }

    pub fn parse(path: PathBuf, data: Vec<u8>) -> io::Result<Self> {
        let mut cur = Cursor::new(&data);

        let magic = cur.read_u32()?;
        if magic != GGUF_MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "Not a GGUF file (magic=0x{:08X}, expected 0x{:08X})",
                    magic, GGUF_MAGIC
                ),
            ));
        }

        let version = cur.read_u32()?;
        if version < 2 || version > 3 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Unsupported GGUF version {}", version),
            ));
        }

        let tensor_count = cur.read_u64()?;
        let metadata_kv_count = cur.read_u64()?;

        let mut metadata = HashMap::with_capacity(metadata_kv_count as usize);
        for _ in 0..metadata_kv_count {
            let key = cur.read_string()?;
            let value = cur.read_meta_value()?;
            metadata.insert(key, value);
        }

        let alignment = metadata
            .get("general.alignment")
            .and_then(|v| v.as_u64())
            .unwrap_or(DEFAULT_ALIGNMENT);

        let mut tensors = Vec::with_capacity(tensor_count as usize);
        let mut tensor_index = HashMap::with_capacity(tensor_count as usize);
        for i in 0..tensor_count {
            let name = cur.read_string()?;
            let n_dims = cur.read_u32()?;
            let mut dims = Vec::with_capacity(n_dims as usize);
            for _ in 0..n_dims {
                dims.push(cur.read_u64()?);
            }
            let type_id = cur.read_u32()?;
            let offset = cur.read_u64()?;
            let info = TensorInfo {
                name: name.clone(),
                dims,
                ggml_type: GgmlType::from_u32(type_id),
                offset,
            };
            tensor_index.insert(name, i as usize);
            tensors.push(info);
        }

        // Tensor data section is aligned from current position.
        let tensor_data_offset = align_up(cur.pos, alignment);

        Ok(Self {
            path,
            version,
            alignment,
            metadata,
            tensors,
            tensor_index,
            tensor_data_offset,
            data,
        })
    }

    pub fn architecture(&self) -> Option<&str> {
        self.metadata
            .get("general.architecture")
            .and_then(|v| v.as_str())
    }

    pub fn model_name(&self) -> String {
        self.metadata
            .get("general.name")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .unwrap_or_else(|| {
                self.path
                    .file_name()
                    .map(|s| s.to_string_lossy().to_string())
                    .unwrap_or_else(|| "unknown".into())
            })
    }

    pub fn meta_u64(&self, key: &str) -> Option<u64> {
        self.metadata.get(key).and_then(|v| v.as_u64())
    }

    pub fn meta_f32(&self, key: &str) -> Option<f32> {
        self.metadata.get(key).and_then(|v| v.as_f32())
    }

    pub fn meta_str(&self, key: &str) -> Option<&str> {
        self.metadata.get(key).and_then(|v| v.as_str())
    }

    pub fn tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.tensor_index.get(name).map(|&i| &self.tensors[i])
    }

    /// Absolute byte range for a tensor's raw payload.
    pub fn tensor_bytes(&self, info: &TensorInfo) -> io::Result<&[u8]> {
        let start = (self.tensor_data_offset + info.offset) as usize;
        if start >= self.data.len() {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                format!("Tensor '{}' offset past EOF", info.name),
            ));
        }
        // Size is inferred by next tensor or EOF; for loading we use type-based size.
        let nbytes = tensor_nbytes(info)?;
        let end = start + nbytes;
        if end > self.data.len() {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                format!(
                    "Tensor '{}' needs {} bytes but only {} remain",
                    info.name,
                    nbytes,
                    self.data.len() - start
                ),
            ));
        }
        Ok(&self.data[start..end])
    }

    pub fn summary(&self) -> String {
        let arch = self.architecture().unwrap_or("unknown");
        let name = self.model_name();
        let n_tensors = self.tensors.len();
        let types: HashMap<String, usize> = {
            let mut m = HashMap::new();
            for t in &self.tensors {
                *m.entry(t.ggml_type.name()).or_insert(0) += 1;
            }
            m
        };
        let type_summary: Vec<String> = types.iter().map(|(k, v)| format!("{}×{}", k, v)).collect();
        format!(
            "GGUF v{} | {} | arch={} | tensors={} | types=[{}] | file={}",
            self.version,
            name,
            arch,
            n_tensors,
            type_summary.join(", "),
            self.path.display()
        )
    }
}

fn align_up(pos: u64, alignment: u64) -> u64 {
    if alignment == 0 {
        return pos;
    }
    let rem = pos % alignment;
    if rem == 0 {
        pos
    } else {
        pos + (alignment - rem)
    }
}

fn tensor_nbytes(info: &TensorInfo) -> io::Result<usize> {
    let n = info.n_elements() as usize;
    let bytes = match info.ggml_type {
        GgmlType::F32 | GgmlType::I32 => n * 4,
        GgmlType::F16 | GgmlType::BF16 | GgmlType::I16 => n * 2,
        GgmlType::F64 | GgmlType::I64 => n * 8,
        GgmlType::I8 => n,
        GgmlType::Q4_0 => {
            // block: 32 weights, 18 bytes
            let blocks = (n + 31) / 32;
            blocks * 18
        }
        GgmlType::Q4_1 => {
            // d+m f16 + 16 nibbles = 20
            let blocks = (n + 31) / 32;
            blocks * 20
        }
        GgmlType::Q5_0 => {
            // d + 4 qh + 16 qs = 22
            let blocks = (n + 31) / 32;
            blocks * 22
        }
        GgmlType::Q5_1 => {
            // d+m + 4 qh + 16 qs = 24
            let blocks = (n + 31) / 32;
            blocks * 24
        }
        GgmlType::Q8_0 => {
            // block: 32 weights, 34 bytes
            let blocks = (n + 31) / 32;
            blocks * 34
        }
        // K-quants: super-block of QK_K = 256 weights
        GgmlType::Q2_K => {
            let blocks = (n + 255) / 256;
            blocks * 84
        }
        GgmlType::Q3_K => {
            let blocks = (n + 255) / 256;
            blocks * 110
        }
        GgmlType::Q4_K => {
            let blocks = (n + 255) / 256;
            blocks * 144
        }
        GgmlType::Q5_K => {
            let blocks = (n + 255) / 256;
            blocks * 176
        }
        GgmlType::Q6_K => {
            let blocks = (n + 255) / 256;
            blocks * 210
        }
        GgmlType::Q8_K => {
            // intermediate: f32 d + 256 i8 + 16 i16 = 292
            let blocks = (n + 255) / 256;
            blocks * 292
        }
        other => {
            return Err(io::Error::new(
                io::ErrorKind::Unsupported,
                format!("Byte size not implemented for {}", other.name()),
            ));
        }
    };
    Ok(bytes)
}

struct Cursor<'a> {
    data: &'a [u8],
    pos: u64,
}

impl<'a> Cursor<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    fn need(&self, n: usize) -> io::Result<()> {
        if self.pos as usize + n > self.data.len() {
            Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "Unexpected EOF reading GGUF",
            ))
        } else {
            Ok(())
        }
    }

    fn read_exact(&mut self, buf: &mut [u8]) -> io::Result<()> {
        self.need(buf.len())?;
        let start = self.pos as usize;
        buf.copy_from_slice(&self.data[start..start + buf.len()]);
        self.pos += buf.len() as u64;
        Ok(())
    }

    fn read_u8(&mut self) -> io::Result<u8> {
        let mut b = [0u8; 1];
        self.read_exact(&mut b)?;
        Ok(b[0])
    }

    fn read_u16(&mut self) -> io::Result<u16> {
        let mut b = [0u8; 2];
        self.read_exact(&mut b)?;
        Ok(u16::from_le_bytes(b))
    }

    fn read_u32(&mut self) -> io::Result<u32> {
        let mut b = [0u8; 4];
        self.read_exact(&mut b)?;
        Ok(u32::from_le_bytes(b))
    }

    fn read_u64(&mut self) -> io::Result<u64> {
        let mut b = [0u8; 8];
        self.read_exact(&mut b)?;
        Ok(u64::from_le_bytes(b))
    }

    fn read_i8(&mut self) -> io::Result<i8> {
        Ok(self.read_u8()? as i8)
    }

    fn read_i16(&mut self) -> io::Result<i16> {
        Ok(self.read_u16()? as i16)
    }

    fn read_i32(&mut self) -> io::Result<i32> {
        Ok(self.read_u32()? as i32)
    }

    fn read_i64(&mut self) -> io::Result<i64> {
        Ok(self.read_u64()? as i64)
    }

    fn read_f32(&mut self) -> io::Result<f32> {
        Ok(f32::from_bits(self.read_u32()?))
    }

    fn read_f64(&mut self) -> io::Result<f64> {
        Ok(f64::from_bits(self.read_u64()?))
    }

    fn read_bool(&mut self) -> io::Result<bool> {
        match self.read_u8()? {
            0 => Ok(false),
            1 => Ok(true),
            other => Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Invalid bool value {}", other),
            )),
        }
    }

    fn read_string(&mut self) -> io::Result<String> {
        let len = self.read_u64()? as usize;
        self.need(len)?;
        let start = self.pos as usize;
        let s = String::from_utf8_lossy(&self.data[start..start + len]).into_owned();
        self.pos += len as u64;
        Ok(s)
    }

    fn read_meta_value(&mut self) -> io::Result<MetaValue> {
        let type_id = self.read_u32()?;
        self.read_meta_value_of_type(type_id)
    }

    fn read_meta_value_of_type(&mut self, type_id: u32) -> io::Result<MetaValue> {
        match type_id {
            0 => Ok(MetaValue::U8(self.read_u8()?)),
            1 => Ok(MetaValue::I8(self.read_i8()?)),
            2 => Ok(MetaValue::U16(self.read_u16()?)),
            3 => Ok(MetaValue::I16(self.read_i16()?)),
            4 => Ok(MetaValue::U32(self.read_u32()?)),
            5 => Ok(MetaValue::I32(self.read_i32()?)),
            6 => Ok(MetaValue::F32(self.read_f32()?)),
            7 => Ok(MetaValue::Bool(self.read_bool()?)),
            8 => Ok(MetaValue::String(self.read_string()?)),
            9 => {
                let elem_type = self.read_u32()?;
                let len = self.read_u64()?;
                // Guard against pathological / nested huge arrays during metadata load.
                if len > 50_000_000 {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("Metadata array too large: {}", len),
                    ));
                }
                let mut items = Vec::with_capacity(len as usize);
                for _ in 0..len {
                    items.push(self.read_meta_value_of_type(elem_type)?);
                }
                Ok(MetaValue::Array(items))
            }
            10 => Ok(MetaValue::U64(self.read_u64()?)),
            11 => Ok(MetaValue::I64(self.read_i64()?)),
            12 => Ok(MetaValue::F64(self.read_f64()?)),
            other => Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Unknown GGUF metadata type {}", other),
            )),
        }
    }
}

/// Super-block size for K-quants (matches llama.cpp `QK_K`).
const QK_K: usize = 256;
/// Packed scale/min table size for Q4_K / Q5_K.
const K_SCALE_SIZE: usize = 12;

/// Dequantize a tensor into a dense f32 buffer (row-major as stored).
pub fn load_tensor_f32(gguf: &GgufFile, info: &TensorInfo) -> io::Result<Vec<f32>> {
    let raw = gguf.tensor_bytes(info)?;
    let n = info.n_elements() as usize;
    dequant_buffer(raw, info.ggml_type, n)
}

/// Dequantize from an in-memory quantized buffer (no GGUF file offsets).
pub fn dequant_buffer(raw: &[u8], ggml_type: GgmlType, n: usize) -> io::Result<Vec<f32>> {
    match ggml_type {
        GgmlType::F32 => {
            let mut out = vec![0.0f32; n];
            for (i, chunk) in raw.chunks_exact(4).enumerate().take(n) {
                out[i] = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            }
            Ok(out)
        }
        GgmlType::F16 => {
            let mut out = vec![0.0f32; n];
            for (i, chunk) in raw.chunks_exact(2).enumerate().take(n) {
                out[i] = f16_to_f32(u16::from_le_bytes([chunk[0], chunk[1]]));
            }
            Ok(out)
        }
        GgmlType::BF16 => {
            let mut out = vec![0.0f32; n];
            for (i, chunk) in raw.chunks_exact(2).enumerate().take(n) {
                let bits = (u16::from_le_bytes([chunk[0], chunk[1]]) as u32) << 16;
                out[i] = f32::from_bits(bits);
            }
            Ok(out)
        }
        GgmlType::Q4_0 => dequant_q4_0(raw, n),
        GgmlType::Q4_1 => dequant_q4_1(raw, n),
        GgmlType::Q5_0 => dequant_q5_0(raw, n),
        GgmlType::Q5_1 => dequant_q5_1(raw, n),
        GgmlType::Q8_0 => dequant_q8_0(raw, n),
        GgmlType::Q2_K => dequant_q2_k(raw, n),
        GgmlType::Q3_K => dequant_q3_k(raw, n),
        GgmlType::Q4_K => dequant_q4_k(raw, n),
        GgmlType::Q5_K => dequant_q5_k(raw, n),
        GgmlType::Q6_K => dequant_q6_k(raw, n),
        GgmlType::Q8_K => dequant_q8_k(raw, n),
        other => Err(io::Error::new(
            io::ErrorKind::Unsupported,
            format!(
                "llama.rs cannot load tensor type {} yet. Use the llama.cpp backend for unsupported quants.",
                other.name()
            ),
        )),
    }
}

#[inline]
fn need_block(raw: &[u8], idx: usize, block: usize, name: &str) -> io::Result<()> {
    if idx + block > raw.len() {
        Err(io::Error::new(
            io::ErrorKind::UnexpectedEof,
            format!("Truncated {} data (need {} more bytes)", name, block),
        ))
    } else {
        Ok(())
    }
}

#[inline]
fn read_f16_le(raw: &[u8], idx: usize) -> f32 {
    f16_to_f32(u16::from_le_bytes([raw[idx], raw[idx + 1]]))
}

fn dequant_q8_0(raw: &[u8], n: usize) -> io::Result<Vec<f32>> {
    let mut out = vec![0.0f32; n];
    let block = 34; // 2 + 32
    let mut idx = 0usize;
    let mut o = 0usize;
    while o < n {
        need_block(raw, idx, block, "Q8_0")?;
        let scale = read_f16_le(raw, idx);
        idx += 2;
        for _ in 0..32 {
            if o >= n {
                break;
            }
            let q = raw[idx] as i8;
            idx += 1;
            out[o] = scale * (q as f32);
            o += 1;
        }
    }
    Ok(out)
}

fn dequant_q4_0(raw: &[u8], n: usize) -> io::Result<Vec<f32>> {
    let mut out = vec![0.0f32; n];
    let block = 18; // 2 + 16
    let mut idx = 0usize;
    let mut o = 0usize;
    while o < n {
        need_block(raw, idx, block, "Q4_0")?;
        let scale = read_f16_le(raw, idx);
        idx += 2;
        for j in 0..16 {
            if o >= n {
                break;
            }
            let byte = raw[idx + j];
            let x0 = (byte & 0x0F) as i8 - 8;
            let x1 = (byte >> 4) as i8 - 8;
            out[o] = scale * (x0 as f32);
            o += 1;
            if o < n {
                out[o] = scale * (x1 as f32);
                o += 1;
            }
        }
        idx += 16;
    }
    Ok(out)
}

fn dequant_q4_1(raw: &[u8], n: usize) -> io::Result<Vec<f32>> {
    // block_q4_1: d(f16), m(f16), qs[16]
    let mut out = vec![0.0f32; n];
    let block = 20;
    let mut idx = 0usize;
    let mut o = 0usize;
    while o < n {
        need_block(raw, idx, block, "Q4_1")?;
        let d = read_f16_le(raw, idx);
        let m = read_f16_le(raw, idx + 2);
        idx += 4;
        for j in 0..16 {
            if o >= n {
                break;
            }
            let byte = raw[idx + j];
            out[o] = d * ((byte & 0x0F) as f32) + m;
            o += 1;
            if o < n {
                out[o] = d * ((byte >> 4) as f32) + m;
                o += 1;
            }
        }
        idx += 16;
    }
    Ok(out)
}

fn dequant_q5_0(raw: &[u8], n: usize) -> io::Result<Vec<f32>> {
    // d(f16), qh[4], qs[16] — llama.cpp layout: y[j] and y[j+16]
    let mut out = vec![0.0f32; n];
    let block = 22;
    let mut idx = 0usize;
    let mut base = 0usize;
    while base < n {
        need_block(raw, idx, block, "Q5_0")?;
        let d = read_f16_le(raw, idx);
        idx += 2;
        let qh = u32::from_le_bytes([raw[idx], raw[idx + 1], raw[idx + 2], raw[idx + 3]]);
        idx += 4;
        let qs = &raw[idx..idx + 16];
        idx += 16;
        for j in 0..16 {
            let xh_0 = (((qh >> j) << 4) & 0x10) as i32;
            let xh_1 = ((qh >> (j + 12)) & 0x10) as i32;
            let x0 = ((qs[j] & 0x0F) as i32) | xh_0;
            let x1 = ((qs[j] >> 4) as i32) | xh_1;
            if base + j < n {
                out[base + j] = d * ((x0 - 16) as f32);
            }
            if base + 16 + j < n {
                out[base + 16 + j] = d * ((x1 - 16) as f32);
            }
        }
        base += 32;
    }
    Ok(out)
}

fn dequant_q5_1(raw: &[u8], n: usize) -> io::Result<Vec<f32>> {
    // d, m, qh[4], qs[16] — same split layout as Q5_0 with min offset
    let mut out = vec![0.0f32; n];
    let block = 24;
    let mut idx = 0usize;
    let mut base = 0usize;
    while base < n {
        need_block(raw, idx, block, "Q5_1")?;
        let d = read_f16_le(raw, idx);
        let m = read_f16_le(raw, idx + 2);
        idx += 4;
        let qh = u32::from_le_bytes([raw[idx], raw[idx + 1], raw[idx + 2], raw[idx + 3]]);
        idx += 4;
        let qs = &raw[idx..idx + 16];
        idx += 16;
        for j in 0..16 {
            let xh_0 = (((qh >> j) << 4) & 0x10) as u32;
            let xh_1 = (qh >> (j + 12)) & 0x10;
            let x0 = ((qs[j] & 0x0F) as u32) | xh_0;
            let x1 = ((qs[j] >> 4) as u32) | xh_1;
            if base + j < n {
                out[base + j] = d * (x0 as f32) + m;
            }
            if base + 16 + j < n {
                out[base + 16 + j] = d * (x1 as f32) + m;
            }
        }
        base += 32;
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// K-quants (super-block of 256) — layouts from llama.cpp ggml-common.h
// ---------------------------------------------------------------------------

/// Decode packed 6-bit scale/min pair for Q4_K / Q5_K sub-blocks.
#[inline]
fn get_scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    // q is K_SCALE_SIZE = 12 bytes
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        let d = (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4);
        let m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
        (d, m)
    }
}

fn dequant_q2_k(raw: &[u8], n: usize) -> io::Result<Vec<f32>> {
    // scales[16], qs[64], d(f16), dmin(f16)  → 84 bytes
    let mut out = vec![0.0f32; n];
    let block = 84;
    let nb = (n + QK_K - 1) / QK_K;
    let mut idx = 0usize;
    let mut o = 0usize;
    for _ in 0..nb {
        need_block(raw, idx, block, "Q2_K")?;
        let scales = &raw[idx..idx + 16];
        let qs = &raw[idx + 16..idx + 16 + 64];
        let d = read_f16_le(raw, idx + 80);
        let dmin = read_f16_le(raw, idx + 82);
        idx += block;

        let mut is = 0usize;
        let mut q_base = 0usize;
        // 2 groups of 128, each with 4 shift steps of 2 bits
        for _n128 in 0..2 {
            let mut shift = 0u32;
            for _j in 0..4 {
                let sc = scales[is];
                is += 1;
                let dl = d * ((sc & 0x0F) as f32);
                let ml = dmin * ((sc >> 4) as f32);
                for l in 0..16 {
                    if o >= n {
                        return Ok(out);
                    }
                    let qv = ((qs[q_base + l] >> shift) & 3) as i8;
                    out[o] = dl * (qv as f32) - ml;
                    o += 1;
                }

                let sc = scales[is];
                is += 1;
                let dl = d * ((sc & 0x0F) as f32);
                let ml = dmin * ((sc >> 4) as f32);
                for l in 0..16 {
                    if o >= n {
                        return Ok(out);
                    }
                    let qv = ((qs[q_base + 16 + l] >> shift) & 3) as i8;
                    out[o] = dl * (qv as f32) - ml;
                    o += 1;
                }
                shift += 2;
            }
            q_base += 32;
        }
    }
    Ok(out)
}

fn dequant_q3_k(raw: &[u8], n: usize) -> io::Result<Vec<f32>> {
    // hmask[32], qs[64], scales[12], d(f16) → 110
    let mut out = vec![0.0f32; n];
    let block = 110;
    let nb = (n + QK_K - 1) / QK_K;
    let mut idx = 0usize;
    let mut o = 0usize;

    let kmask1: u32 = 0x0303_0303;
    let kmask2: u32 = 0x0f0f_0f0f;

    for _ in 0..nb {
        need_block(raw, idx, block, "Q3_K")?;
        let hmask = &raw[idx..idx + 32];
        let qs = &raw[idx + 32..idx + 32 + 64];
        let scales_raw = &raw[idx + 96..idx + 108];
        let d_all = read_f16_le(raw, idx + 108);
        idx += block;

        // Unpack 6-bit scales into 16 signed values (offset 32)
        let mut aux = [0u32; 4];
        aux[0] = u32::from_le_bytes([
            scales_raw[0],
            scales_raw[1],
            scales_raw[2],
            scales_raw[3],
        ]);
        aux[1] = u32::from_le_bytes([
            scales_raw[4],
            scales_raw[5],
            scales_raw[6],
            scales_raw[7],
        ]);
        aux[2] = u32::from_le_bytes([
            scales_raw[8],
            scales_raw[9],
            scales_raw[10],
            scales_raw[11],
        ]);
        let tmp = aux[2];
        aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
        aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
        // reinterpret as 16 int8 scales
        let scales_bytes: [u8; 16] = {
            let mut b = [0u8; 16];
            for (i, a) in aux.iter().enumerate() {
                let bytes = a.to_le_bytes();
                b[i * 4..i * 4 + 4].copy_from_slice(&bytes);
            }
            b
        };
        let scales: [i8; 16] = scales_bytes.map(|b| b as i8);

        let mut is = 0usize;
        let mut q_base = 0usize;
        let mut m: u8 = 1;
        for _n128 in 0..2 {
            let mut shift = 0u32;
            for _j in 0..4 {
                let dl = d_all * ((scales[is] as i32 - 32) as f32);
                is += 1;
                for l in 0..16 {
                    if o >= n {
                        return Ok(out);
                    }
                    let lo = ((qs[q_base + l] >> shift) & 3) as i8;
                    let hi = if (hmask[l] & m) != 0 { 0 } else { 4 };
                    out[o] = dl * ((lo - hi) as f32);
                    o += 1;
                }

                let dl = d_all * ((scales[is] as i32 - 32) as f32);
                is += 1;
                for l in 0..16 {
                    if o >= n {
                        return Ok(out);
                    }
                    let lo = ((qs[q_base + 16 + l] >> shift) & 3) as i8;
                    let hi = if (hmask[16 + l] & m) != 0 { 0 } else { 4 };
                    out[o] = dl * ((lo - hi) as f32);
                    o += 1;
                }
                shift += 2;
                m <<= 1;
            }
            q_base += 32;
        }
    }
    Ok(out)
}

fn dequant_q4_k(raw: &[u8], n: usize) -> io::Result<Vec<f32>> {
    // d(f16), dmin(f16), scales[12], qs[128] → 144
    let mut out = vec![0.0f32; n];
    let block = 144;
    let nb = (n + QK_K - 1) / QK_K;
    let mut idx = 0usize;
    let mut o = 0usize;
    for _ in 0..nb {
        need_block(raw, idx, block, "Q4_K")?;
        let d = read_f16_le(raw, idx);
        let dmin = read_f16_le(raw, idx + 2);
        let scales = &raw[idx + 4..idx + 4 + K_SCALE_SIZE];
        let mut q = &raw[idx + 16..idx + 16 + 128];
        idx += block;

        let mut is = 0usize;
        // 4 × 64 elements
        for _j in 0..4 {
            let (sc, m) = get_scale_min_k4(is, scales);
            let d1 = d * (sc as f32);
            let m1 = dmin * (m as f32);
            let (sc, m) = get_scale_min_k4(is + 1, scales);
            let d2 = d * (sc as f32);
            let m2 = dmin * (m as f32);
            for l in 0..32 {
                if o >= n {
                    return Ok(out);
                }
                out[o] = d1 * ((q[l] & 0x0F) as f32) - m1;
                o += 1;
            }
            for l in 0..32 {
                if o >= n {
                    return Ok(out);
                }
                out[o] = d2 * ((q[l] >> 4) as f32) - m2;
                o += 1;
            }
            q = &q[32..];
            is += 2;
        }
    }
    Ok(out)
}

fn dequant_q5_k(raw: &[u8], n: usize) -> io::Result<Vec<f32>> {
    // d, dmin, scales[12], qh[32], qs[128] → 176
    let mut out = vec![0.0f32; n];
    let block = 176;
    let nb = (n + QK_K - 1) / QK_K;
    let mut idx = 0usize;
    let mut o = 0usize;
    for _ in 0..nb {
        need_block(raw, idx, block, "Q5_K")?;
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
                    return Ok(out);
                }
                let qv = (ql[l] & 0x0F) as u32 + if (qh[l] & u1) != 0 { 16 } else { 0 };
                out[o] = d1 * (qv as f32) - m1;
                o += 1;
            }
            for l in 0..32 {
                if o >= n {
                    return Ok(out);
                }
                let qv = (ql[l] >> 4) as u32 + if (qh[l] & u2) != 0 { 16 } else { 0 };
                out[o] = d2 * (qv as f32) - m2;
                o += 1;
            }
            ql = &ql[32..];
            is += 2;
            u1 <<= 2;
            u2 <<= 2;
        }
    }
    Ok(out)
}

fn dequant_q6_k(raw: &[u8], n: usize) -> io::Result<Vec<f32>> {
    // ql[128], qh[64], scales[16] i8, d(f16) → 210
    let mut out = vec![0.0f32; n];
    let block = 210;
    let nb = (n + QK_K - 1) / QK_K;
    let mut idx = 0usize;
    let mut o = 0usize;
    for _ in 0..nb {
        need_block(raw, idx, block, "Q6_K")?;
        let mut ql = &raw[idx..idx + 128];
        let mut qh = &raw[idx + 128..idx + 128 + 64];
        let mut sc = &raw[idx + 192..idx + 192 + 16];
        let d = read_f16_le(raw, idx + 208);
        idx += block;

        // Two passes of 128 elements
        for _n128 in 0..2 {
            // Write into a temp 128 then copy (matches llama.cpp y[l+…] layout)
            let mut y = [0.0f32; 128];
            for l in 0..32 {
                let is = l / 16;
                let q1 = ((ql[l] & 0x0F) | (((qh[l] >> 0) & 3) << 4)) as i8 as i32 - 32;
                let q2 = ((ql[l + 32] & 0x0F) | (((qh[l] >> 2) & 3) << 4)) as i8 as i32 - 32;
                let q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) as i8 as i32 - 32;
                let q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) as i8 as i32 - 32;
                let s0 = sc[is] as i8 as f32;
                let s2 = sc[is + 2] as i8 as f32;
                let s4 = sc[is + 4] as i8 as f32;
                let s6 = sc[is + 6] as i8 as f32;
                y[l] = d * s0 * (q1 as f32);
                y[l + 32] = d * s2 * (q2 as f32);
                y[l + 64] = d * s4 * (q3 as f32);
                y[l + 96] = d * s6 * (q4 as f32);
            }
            for v in y {
                if o >= n {
                    return Ok(out);
                }
                out[o] = v;
                o += 1;
            }
            ql = &ql[64..];
            qh = &qh[32..];
            sc = &sc[8..];
        }
    }
    Ok(out)
}

fn dequant_q8_k(raw: &[u8], n: usize) -> io::Result<Vec<f32>> {
    // f32 d + qs[256] i8 + bsums[16] i16 → 292 (bsums unused for dequant)
    let mut out = vec![0.0f32; n];
    let block = 292;
    let nb = (n + QK_K - 1) / QK_K;
    let mut idx = 0usize;
    let mut o = 0usize;
    for _ in 0..nb {
        need_block(raw, idx, block, "Q8_K")?;
        let d = f32::from_le_bytes([raw[idx], raw[idx + 1], raw[idx + 2], raw[idx + 3]]);
        let qs = &raw[idx + 4..idx + 4 + QK_K];
        idx += block;
        for j in 0..QK_K {
            if o >= n {
                break;
            }
            out[o] = d * (qs[j] as i8 as f32);
            o += 1;
        }
    }
    Ok(out)
}

/// IEEE half-precision to f32 (software).
pub fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1F) as u32;
    let frac = (h & 0x3FF) as u32;
    let bits = if exp == 0 {
        if frac == 0 {
            sign << 31
        } else {
            // subnormal
            let mut e = 127 - 15 + 1;
            let mut m = frac;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            m &= 0x3FF;
            (sign << 31) | ((e as u32) << 23) | (m << 13)
        }
    } else if exp == 31 {
        (sign << 31) | (0xFF << 23) | (frac << 13)
    } else {
        (sign << 31) | ((exp + (127 - 15)) << 23) | (frac << 13)
    };
    f32::from_bits(bits)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_f16_to_f32_one() {
        // 1.0 in f16 is 0x3C00
        let v = f16_to_f32(0x3C00);
        assert!((v - 1.0).abs() < 1e-3);
    }

    #[test]
    fn test_align_up() {
        assert_eq!(align_up(0, 32), 0);
        assert_eq!(align_up(1, 32), 32);
        assert_eq!(align_up(32, 32), 32);
        assert_eq!(align_up(33, 32), 64);
    }

    #[test]
    fn test_k_quants_loadable() {
        assert!(GgmlType::Q4_K.is_loadable());
        assert!(GgmlType::Q5_K.is_loadable());
        assert!(GgmlType::Q6_K.is_loadable());
        assert!(GgmlType::Q2_K.is_loadable());
        assert!(GgmlType::Q3_K.is_loadable());
    }

    #[test]
    fn test_q4_k_block_size() {
        let info = TensorInfo {
            name: "t".into(),
            dims: vec![256],
            ggml_type: GgmlType::Q4_K,
            offset: 0,
        };
        assert_eq!(tensor_nbytes(&info).unwrap(), 144);
        let info2 = TensorInfo {
            name: "t".into(),
            dims: vec![512],
            ggml_type: GgmlType::Q5_K,
            offset: 0,
        };
        assert_eq!(tensor_nbytes(&info2).unwrap(), 176 * 2);
        let info3 = TensorInfo {
            name: "t".into(),
            dims: vec![256],
            ggml_type: GgmlType::Q6_K,
            offset: 0,
        };
        assert_eq!(tensor_nbytes(&info3).unwrap(), 210);
    }

    #[test]
    fn test_dequant_q4_k_zero_block() {
        // All-zero qs/scales with d=1.0 → zeros after min*0
        // d = 1.0f16 = 0x3C00, dmin = 0
        let mut raw = vec![0u8; 144];
        raw[0] = 0x00;
        raw[1] = 0x3C; // little-endian f16 1.0
        // scales/qs zero → all outputs 0
        let out = dequant_q4_k(&raw, 256).unwrap();
        assert_eq!(out.len(), 256);
        assert!(out.iter().all(|&v| v.abs() < 1e-6));
    }

    #[test]
    fn test_get_scale_min_k4_low() {
        let q = [10u8, 20, 30, 40, 50, 60, 0, 0, 0, 0, 0, 0];
        let (d, m) = get_scale_min_k4(0, &q);
        assert_eq!(d, 10 & 63);
        assert_eq!(m, 50 & 63);
    }

    #[test]
    fn test_dequant_q6_k_size() {
        let raw = vec![0u8; 210];
        let out = dequant_q6_k(&raw, 256).unwrap();
        assert_eq!(out.len(), 256);
    }

    #[test]
    fn test_dequant_q5_k_size() {
        let raw = vec![0u8; 176];
        let out = dequant_q5_k(&raw, 256).unwrap();
        assert_eq!(out.len(), 256);
    }
}
