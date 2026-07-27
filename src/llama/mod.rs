//! Local LLM engines for Hercules Agent.
//!
//! # How llama.cpp handles models (reference architecture)
//!
//! 1. **GGUF load** — single-file format with metadata KV, tokenizer, and quantized tensors
//! 2. **Tokenizer** — encode text → token ids (BPE / SentencePiece from GGUF)
//! 3. **Embeddings** — token id rows from `token_embd.weight`
//! 4. **Transformer graph (ggml)** — per layer: RMSNorm → QKV → RoPE → attention (KV cache) → FFN (SiLU)
//! 5. **Logits** — final RMSNorm + `output.weight`
//! 6. **Sampling** — temperature / top-k / top-p / penalties → next token
//! 7. **Decode loop** — append token, reuse KV cache for O(1) per new token (amortized)
//!
//! # Backends in this crate
//!
//! | Backend    | Module        | Role |
//! |------------|---------------|------|
//! | **llama.rs**  | `infer`, `gguf`, `model`, `compute` | Pure Rust GGUF (**no** C/FFI; pluggable [`ComputeBackend`]) |
//! | **llama.cpp** | `cpp`, `server` | C++ runtime via CLI / managed `llama-server` (bindings track later) |
//! | **HTTP**      | `http`        | OpenAI-compatible client (llama.cpp server / remote) |

pub mod bench;
pub mod compute;
pub mod cpp;
pub mod gguf;
pub mod http;
pub mod infer;
pub mod kernels;
pub mod model;
pub mod sample;
pub mod server;
pub mod tokenizer;

pub use compute::{
    build_default_backend, default_backend, ComputeBackend, ComputeError, ComputePrefs,
    ScalarBackend,
};
pub use cpp::LlamaCppRuntime;
pub use http::HttpInferenceClient;
pub use infer::{
    ensure_warm_rs_engine, shutdown_warm_rs_engine, LlamaRsEngine, LlamaRsRuntime,
};

/// High-level engine choice exposed to the application backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LlamaEngineKind {
    /// Pure Rust implementation (this crate).
    LlamaRs,
    /// Official llama.cpp (CLI or server).
    LlamaCpp,
}

impl LlamaEngineKind {
    pub fn label(self) -> &'static str {
        match self {
            Self::LlamaRs => "llama.rs (Pure Rust)",
            Self::LlamaCpp => "llama.cpp (C/C++ runtime)",
        }
    }
}
