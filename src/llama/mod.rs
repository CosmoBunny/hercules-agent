//! Local LLM engines for Hercules Agent.
//!
//! # Backends
//!
//! | Backend       | Module            | Role                              |
//! |---------------|-------------------|-----------------------------------|
//! | **llama.cpp** | `ffi`, `libinfer` | In-process via libllama.so (FFI)  |
//! | **HTTP**      | `http`            | OpenAI-compatible client          |
//! | **Server**    | `server`, `cpp`   | Managed llama-server process      |

#[allow(dead_code, unused)]
pub mod cpp;
pub mod ffi;
pub mod http;
pub mod libinfer;

// ---------------------------------------------------------------------------
// Legacy pure-Rust modules (not active at runtime; kept for reference)
// Re-exported at crate::llama level so their internal cross-imports resolve.
// ---------------------------------------------------------------------------
#[allow(dead_code, unused, unused_imports)]
pub mod legacy {
    pub mod bench;
    pub mod compute;
    pub mod gguf;
    pub mod infer;
    pub mod kernels;
    pub mod model;
    pub mod sample;
    pub mod tokenizer;
}

// Re-export legacy modules at the top crate::llama level so legacy source
// files that do `use crate::llama::gguf` continue to compile.
#[allow(unused_imports)]
pub use legacy::bench;
#[allow(unused_imports)]
pub use legacy::compute;
#[allow(unused_imports)]
pub use legacy::gguf;
#[allow(unused_imports)]
pub use legacy::kernels;
#[allow(unused_imports)]
pub use legacy::model;
#[allow(unused_imports)]
pub use legacy::sample;
#[allow(unused_imports)]
pub use legacy::tokenizer;

// Re-export legacy compute traits at crate::llama level (legacy code uses
// `crate::llama::ComputeBackend` without the ::compute:: path component).
#[allow(unused_imports)]
pub use legacy::compute::{
    ComputeBackend, ComputeError, ComputePrefs, ScalarBackend, SimdBackend,
    build_default_backend, default_backend, default_rms_norm,
};
#[allow(unused_imports)]
pub use legacy::infer::{ensure_warm_rs_engine, shutdown_warm_rs_engine, LlamaRsEngine};
#[cfg(feature = "parallel")]
#[allow(unused_imports)]
pub use legacy::compute::ParallelBackend;

// Active public API
pub use http::HttpInferenceClient;
pub use libinfer::{
    ensure_warm_lib_engine, get_warm_lib_engine, shutdown_warm_lib_engine,
    LlamaCppLib, LlamaCppLibRuntime,
};

/// Engine choice exposed to the application.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LlamaEngineKind {
    /// In-process libllama engine (C FFI / static link) — primary path.
    LlamaCppLib,
}

impl LlamaEngineKind {
    pub fn label(self) -> &'static str {
        match self {
            Self::LlamaCppLib => "llama.cpp (in-process static engine)",
        }
    }
}
