//! Runtime factory boundary: selection → EXISTING Hercules runtimes.
//!
//! Strict separation, per the Phase 3 instruction:
//! - `BackendProvider` = discovery/capability/availability metadata.
//! - This module = mapping a selected `BackendKind` to the real,
//!   pre-existing implementation (`AgentBackend`).
//! - Execution state (models, tokenizers, contexts, HTTP clients, GPU
//!   handles) stays inside the existing runtimes. Nothing is duplicated
//!   here: the factory returns identities and validation verdicts, and
//!   the application constructs backends exactly as it does today.

use super::backend::BackendKind;
use super::error::ModelError;
use super::hardware::HardwareInfo;
use super::manifest::Artifact;
use super::registry::BackendRegistry;

/// The actually-existing Hercules inference implementations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExistingRuntime {
    LlamaCpp,
    Ollama,
    Transformers,
    BurnWgpu,
}

impl ExistingRuntime {
    pub fn kind(self) -> BackendKind {
        match self {
            Self::LlamaCpp => BackendKind::LlamaCpp,
            Self::Ollama => BackendKind::Ollama,
            Self::Transformers => BackendKind::Transformers,
            Self::BurnWgpu => BackendKind::BurnWgpu,
        }
    }

    /// Which existing code owns execution (for reports/diagnostics).
    pub fn implementation(self) -> &'static str {
        match self {
            Self::LlamaCpp => "LlamaCppLibBackend (in-process GGUF, cooperative cancel)",
            Self::Ollama => "OllamaBackend (local daemon HTTP, droppable streams)",
            Self::Transformers => "TransformersBackend (isolated Python worker, kill_on_drop)",
            Self::BurnWgpu => "BurnWgpuBackend (demo engine, gpu feature)",
        }
    }
}

impl BackendKind {
    /// Stable identity of a live backend. The single conversion point —
    /// never compare backend display names across the codebase.
    pub fn of_agent_backend(backend: &crate::backend::AgentBackend) -> BackendKind {
        match backend {
            crate::backend::AgentBackend::LlamaCppLib(_) => BackendKind::LlamaCpp,
            crate::backend::AgentBackend::Ollama(_) => BackendKind::Ollama,
            crate::backend::AgentBackend::Transformers(_) => BackendKind::Transformers,
            #[cfg(feature = "gpu")]
            crate::backend::AgentBackend::BurnWgpu(_) => BackendKind::BurnWgpu,
        }
    }
}

/// Map a selected kind to its existing runtime. Future kinds stay
/// Phase-gated errors — Phase 3 adds no engines.
pub fn runtime_for_kind(kind: BackendKind) -> Result<ExistingRuntime, ModelError> {
    match kind {
        BackendKind::LlamaCpp => Ok(ExistingRuntime::LlamaCpp),
        BackendKind::Ollama => Ok(ExistingRuntime::Ollama),
        #[cfg(feature = "gpu")]
        BackendKind::BurnWgpu => Ok(ExistingRuntime::BurnWgpu),
        #[cfg(not(feature = "gpu"))]
        BackendKind::BurnWgpu => Err(ModelError::BackendUnavailable {
            backend: kind.label().to_string(),
            reason: "Burn/WGPU needs the `gpu` build feature".to_string(),
        }),
        BackendKind::Transformers => Ok(ExistingRuntime::Transformers),
        BackendKind::Mlx => Err(ModelError::BackendUnavailable {
            backend: kind.label().to_string(),
            reason: "MLX runtime arrives in Phase 5 (Apple Silicon only)".to_string(),
        }),
        BackendKind::OpenAiCompatible => Err(ModelError::BackendUnavailable {
            backend: kind.label().to_string(),
            reason: "Remote endpoint support arrives in Phase 6".to_string(),
        }),
    }
}

/// Construct the EXISTING Hercules runtime for a selected kind.
/// Cheap and side-effect free: wires configuration only (a local path
/// or a model name). No model is loaded, no network is touched, no
/// inference runs — loading happens lazily inside the existing
/// generate paths, exactly as before. This is the proof that the
/// registry bridges to real implementations, not stubs.
pub fn construct_existing_runtime(
    kind: BackendKind,
    model_ref: &str,
) -> Result<crate::backend::AgentBackend, ModelError> {
    match runtime_for_kind(kind)? {
        ExistingRuntime::LlamaCpp => Ok(crate::backend::AgentBackend::LlamaCppLib(
            crate::backend::LlamaCppLibBackend::gguf(std::path::PathBuf::from(model_ref)),
        )),
        ExistingRuntime::Ollama => Ok(crate::backend::AgentBackend::Ollama(
            crate::backend::OllamaBackend::new(model_ref.to_string()),
        )),
        ExistingRuntime::Transformers => Ok(crate::backend::AgentBackend::Transformers(
            crate::model::transformers::TransformersBackend::new(std::path::PathBuf::from(
                model_ref,
            )),
        )),
        ExistingRuntime::BurnWgpu => {
            #[cfg(feature = "gpu")]
            {
                Ok(crate::backend::AgentBackend::BurnWgpu(
                    crate::backend::BurnWgpuBackend::with_model(model_ref.to_string()),
                ))
            }
            #[cfg(not(feature = "gpu"))]
            {
                Err(ModelError::BackendUnavailable {
                    backend: "Burn/WGPU".to_string(),
                    reason: "Burn/WGPU needs the `gpu` build feature".to_string(),
                })
            }
        }
    }
}

/// Validate an EXPLICIT user backend choice against one artifact.
/// Never silently switches: incompatible choices fail with a typed error
/// and a useful diagnostic. Each condition maps to its own variant so
/// callers never string-match.
pub fn validate_explicit_selection(
    registry: &BackendRegistry,
    kind: BackendKind,
    artifact: &Artifact,
    architecture: Option<&str>,
    hardware: &HardwareInfo,
) -> Result<(), ModelError> {
    let provider = registry
        .get(kind)
        .ok_or_else(|| ModelError::BackendUnavailable {
            backend: kind.label().to_string(),
            reason: "not registered".to_string(),
        })?;
    let caps = provider.capabilities();
    if !caps.available_on_this_host {
        return Err(ModelError::BackendUnavailable {
            backend: provider.name().to_string(),
            reason: "unavailable on this host".to_string(),
        });
    }
    if caps.requires_local_artifact && !caps.formats.contains(&artifact.format) {
        return Err(ModelError::FormatUnsupported {
            format: artifact.format.label().to_string(),
            backend: provider.name().to_string(),
        });
    }
    if caps.requires_local_artifact
        && !caps.layouts.is_empty()
        && !caps.layouts.contains(&artifact.layout)
    {
        return Err(ModelError::FormatUnsupported {
            format: format!("{} ({})", artifact.format.label(), artifact.layout.label()),
            backend: provider.name().to_string(),
        });
    }
    if !caps.architectures.is_empty() {
        match architecture {
            Some(arch) => {
                let family = super::compatibility::arch_family(arch);
                if !caps
                    .architectures
                    .iter()
                    .any(|a| super::compatibility::arch_family(a) == family)
                {
                    return Err(ModelError::ArchitectureUnsupported {
                        architecture: arch.to_string(),
                        backend: provider.name().to_string(),
                    });
                }
            }
            None => {
                return Err(ModelError::ArchitectureUnsupported {
                    architecture: "unknown".to_string(),
                    backend: provider.name().to_string(),
                });
            }
        }
    }
    let _ = hardware;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::format::{LayoutSource, ModelFormat, ModelLayout};
    use super::super::manifest::{Artifact, RepoFile, RepoManifest};
    use super::*;

    fn hw() -> HardwareInfo {
        HardwareInfo::detect()
    }

    fn gguf_artifact() -> Artifact {
        Artifact {
            format: ModelFormat::Gguf,
            files: vec!["model-Q4_K_M.gguf".to_string()],
            bytes: 1,
            layout: ModelLayout::Unknown,
            layout_source: LayoutSource::Undetermined,
            sharded: false,
        }
    }

    #[test]
    fn test_runtime_maps_to_existing_implementations() {
        assert_eq!(
            runtime_for_kind(BackendKind::LlamaCpp).unwrap(),
            ExistingRuntime::LlamaCpp
        );
        assert_eq!(
            runtime_for_kind(BackendKind::Ollama).unwrap(),
            ExistingRuntime::Ollama
        );
        // Phase 4: Transformers is a real runtime now.
        assert_eq!(
            runtime_for_kind(BackendKind::Transformers).unwrap(),
            ExistingRuntime::Transformers
        );
        assert!(runtime_for_kind(BackendKind::Mlx).is_err());
        assert!(runtime_for_kind(BackendKind::OpenAiCompatible).is_err());
    }

    #[test]
    fn test_explicit_selection_never_silently_switches() {
        let reg = BackendRegistry::default_registry();
        let art = gguf_artifact();
        // Compatible explicit choice succeeds.
        assert!(
            validate_explicit_selection(
                &reg,
                BackendKind::LlamaCpp,
                &art,
                Some("LlamaForCausalLM"),
                &hw()
            )
            .is_ok()
        );
        // Incompatible explicit choice fails typed, never switched.
        let st = Artifact {
            format: ModelFormat::SafeTensors,
            ..art.clone()
        };
        let err = validate_explicit_selection(
            &reg,
            BackendKind::LlamaCpp,
            &st,
            Some("LlamaForCausalLM"),
            &hw(),
        )
        .unwrap_err();
        assert!(
            matches!(err, ModelError::FormatUnsupported { .. }),
            "{err:?}"
        );
        let err = validate_explicit_selection(
            &reg,
            BackendKind::LlamaCpp,
            &art,
            Some("SomeNewModelForCausalLM"),
            &hw(),
        )
        .unwrap_err();
        assert!(
            matches!(err, ModelError::ArchitectureUnsupported { .. }),
            "{err:?}"
        );
    }

    #[test]
    fn test_resolve_then_factory_end_to_end() {
        // Instruction §19: resolve → factory chooses existing llama.cpp.
        let m = RepoManifest::from_files(
            "org/llama-gguf",
            "main",
            vec![RepoFile {
                name: "model-Q4_K_M.gguf".to_string(),
                size_bytes: Some(10),
            }],
            Some(r#"{"architectures":["LlamaForCausalLM"]}"#),
        );
        let reg = BackendRegistry::default_registry();
        let r = super::super::resolver::resolve(&m, &reg, &hw());
        let (id, kind) = r.selected.clone().expect("must select");
        assert_eq!(id, "gguf:model-q4_k_m");
        // Exact files behind the id — runtime loads Q4, never anything else.
        assert_eq!(
            r.artifact(&id).unwrap().files,
            vec!["model-Q4_K_M.gguf".to_string()]
        );
        let rt = runtime_for_kind(kind).unwrap();
        assert_eq!(rt, ExistingRuntime::LlamaCpp);
        assert!(rt.implementation().contains("LlamaCppLibBackend"));
    }

    #[test]
    fn test_cancellation_model_untouched() {
        // Registry integration adds no cancellation mechanism of its own:
        // providers only declare the capability bit.
        let reg = BackendRegistry::default_registry();
        let caps = reg.get(BackendKind::LlamaCpp).unwrap().capabilities();
        assert!(caps.cancellation);
    }

    #[test]
    fn test_llama_rejected_arch_still_reaches_transformers_caps() {
        // Instruction §28: an arch llama.cpp rejects must remain routable
        // to Transformers at the capability level (availability decides
        // separately via the real probe).
        use super::super::format::ModelLayout;
        let reg = BackendRegistry::default_registry();
        let tf_caps = reg.get(BackendKind::Transformers).unwrap().capabilities();
        assert!(
            tf_caps
                .formats
                .contains(&super::super::format::ModelFormat::SafeTensors)
        );
        assert!(tf_caps.layouts.contains(&ModelLayout::StandardHf));
        // llama.cpp rejects the exotic arch even as GGUF.
        let p = super::super::backend::LlamaCppProvider;
        let c = super::super::compatibility::check_compatibility(
            &p,
            super::super::format::ModelFormat::Gguf,
            ModelLayout::Unknown,
            Some("SomeNewModelForCausalLM"),
            &HardwareInfo::detect(),
        );
        assert!(!c.compatible);
    }

    #[test]
    fn test_explicit_transformers_never_silently_falls_back() {
        use super::super::format::{ModelFormat, ModelLayout};
        use super::super::manifest::Artifact;
        let reg = BackendRegistry::default_registry();
        let art = Artifact {
            format: ModelFormat::SafeTensors,
            files: vec!["model.safetensors".to_string()],
            bytes: 1,
            layout: ModelLayout::StandardHf,
            layout_source: super::super::format::LayoutSource::Metadata,
            sharded: false,
        };
        // On machines without torch this fails typed (BackendUnavailable);
        // on machines with torch it succeeds. Either way it NEVER returns
        // a different backend.
        let res = validate_explicit_selection(
            &reg,
            BackendKind::Transformers,
            &art,
            Some("LlamaForCausalLM"),
            &HardwareInfo::detect(),
        );
        match res {
            Ok(()) => {}
            Err(e) => assert!(
                matches!(
                    e,
                    crate::model::ModelError::BackendUnavailable { .. }
                        | crate::model::ModelError::FormatUnsupported { .. }
                        | crate::model::ModelError::ArchitectureUnsupported { .. }
                ),
                "{e:?}"
            ),
        }
    }

    #[test]
    fn test_construct_existing_runtime_round_trip() {
        // The bridge reaches REAL implementations: construct from a kind,
        // map back, and land on the same kind. Constructors are cheap
        // (config only — no model load, no network).
        let rt = construct_existing_runtime(BackendKind::LlamaCpp, "/models/m.gguf").unwrap();
        assert_eq!(BackendKind::of_agent_backend(&rt), BackendKind::LlamaCpp);
        let rt = construct_existing_runtime(BackendKind::Ollama, "llama3").unwrap();
        assert_eq!(BackendKind::of_agent_backend(&rt), BackendKind::Ollama);
        // Phase 4: Transformers constructs the real backend (config only —
        // no probe, no load, no network at construction).
        let rt = construct_existing_runtime(BackendKind::Transformers, "/models/m").unwrap();
        assert_eq!(
            BackendKind::of_agent_backend(&rt),
            BackendKind::Transformers
        );
        // Future kinds fail typed, never stubbed.
        assert!(construct_existing_runtime(BackendKind::Mlx, "x").is_err());
    }
}
