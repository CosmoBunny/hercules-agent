//! Model/backend resolution foundation (Phase 1).
//!
//! Makes Hercules model/backend agnostic: a Hugging Face repository is a
//! model artifact described by architecture + formats + hardware needs,
//! resolved to the best available inference backend. This module only
//! defines the vocabulary (formats, backends, capabilities, hardware,
//! errors) and the provider registry — it changes no existing behavior.
//! Later phases plug the real resolvers and backends into it.

pub mod backend;
pub mod compatibility;
pub mod error;
pub mod format;
pub mod hardware;
pub mod huggingface;
pub mod manifest;
pub mod registry;
pub mod resolver;
pub mod runtime;
pub mod transformers;

pub use backend::{BackendCapabilities, BackendKind};
pub use compatibility::{Compatibility, normalize_arch};
pub use error::ModelError;
pub use format::{ModelFormat, ModelLayout};
pub use hardware::HardwareInfo;
pub use huggingface::{HfAuth, inspect_repository};
pub use manifest::{RepoFile, RepoManifest};
pub use registry::BackendRegistry;
pub use resolver::{ResolvedModel, resolve, resolve_with_preference};
pub use runtime::{
    ExistingRuntime, construct_existing_runtime, runtime_for_kind, validate_explicit_selection,
};
