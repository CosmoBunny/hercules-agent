//! Hercules Agent — library surface.
//!
//! # Engines
//!
//! | Track | Module | C/FFI |
//! |-------|--------|-------|
//! | **llama.cpp lib** | `llama::ffi`, `llama::libinfer` | In-process static / direct FFI engine |
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
pub mod media;
pub mod ocr;
pub mod session;
pub mod smart_system;
pub mod mcp;
pub mod ask_mode;
pub mod code_graph;
pub mod lsp;

// TUI lives with the binary but is part of the crate so `app` can use `crate::`.
pub mod app;

pub use llama::{
    ensure_warm_lib_engine, shutdown_warm_lib_engine,
    LlamaEngineKind, LlamaCppLib, LlamaCppLibRuntime,
};
pub use settings::{get_settings, PowerMode, RuntimeSettings};
