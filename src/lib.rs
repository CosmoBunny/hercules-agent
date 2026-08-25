//! Hercules Agent — library surface.
//!
//! # Engines
//!
//! | Track | Module | C/FFI |
//! |-------|--------|-------|
//! | **llama.cpp lib** | `llama::ffi`, `llama::libinfer` | `libllama.so` (in-process C FFI) |
//! | **llama.cpp server** | `llama::server`, `llama::cpp` | Managed subprocess |
//! | **Ollama** | `backend::OllamaBackend` | HTTP |

#![allow(clippy::too_many_arguments)]

pub mod agent;
pub mod backend;
pub mod clipboard;
pub mod llama;
pub mod manager;
pub mod settings;
pub mod task_manager;
pub mod tool_panel;
pub mod diagram;
pub mod graphic;
pub mod markdown;
pub mod session;

// TUI lives with the binary but is part of the crate so `app` can use `crate::`.
pub mod app;

pub use llama::{
    ensure_warm_lib_engine, shutdown_warm_lib_engine,
    LlamaEngineKind, LlamaCppLib, LlamaCppLibRuntime,
};
pub use settings::{get_settings, PowerMode, RuntimeSettings};
