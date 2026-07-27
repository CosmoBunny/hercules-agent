//! Hercules Agent — library surface.
//!
//! # Engines
//!
//! | Track | Feature / module | C/FFI |
//! |-------|------------------|--------|
//! | **llama.rs** | always (`llama` pure path) | **None** — portable pure Rust |
//! | **llama.cpp** | `server` / `cpp` today; `llama-cpp-bindings` later | Shared lib / process |
//!
//! External embedders can implement [`llama::ComputeBackend`] for custom devices
//! (MCU, NPU, …) without forking the decoder graph.

#![allow(clippy::too_many_arguments)]

pub mod agent;
pub mod backend;
pub mod clipboard;
pub mod llama;
pub mod manager;
pub mod settings;
pub mod task_manager;
pub mod tool_panel;

// TUI lives with the binary but is part of the crate so `app` can use `crate::`.
pub mod app;

pub use llama::{
    build_default_backend, ensure_warm_rs_engine, shutdown_warm_rs_engine, ComputeBackend,
    ComputePrefs, LlamaEngineKind, LlamaRsEngine, LlamaRsRuntime, ScalarBackend,
};
pub use settings::{get_settings, PowerMode, RuntimeSettings};
