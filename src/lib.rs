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
pub mod ask_mode;
pub mod backend;
pub mod clipboard;
pub mod code_graph;
pub mod diagram;
pub mod graphic;
pub mod llama;
pub mod lsp;
pub mod manager;
pub mod markdown;
pub mod mcp;
pub mod media;
pub mod model;
pub mod ocr;
pub mod run_timeline;
pub mod session;
pub mod settings;
pub mod smart_system;
pub mod task_manager;
pub mod tool_panel;

// TUI lives with the binary but is part of the crate so `app` can use `crate::`.
pub mod app;

pub use llama::{
    LlamaCppLib, LlamaCppLibRuntime, LlamaEngineKind, ensure_warm_lib_engine,
    shutdown_warm_lib_engine,
};
pub use settings::{PowerMode, RuntimeSettings, get_settings};
