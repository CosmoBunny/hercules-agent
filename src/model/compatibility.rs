//! Compatibility verdicts: capabilities + model facts + hardware.
//!
//! Three distinct conditions are never conflated:
//! - "SafeTensors exists" (format inventory),
//! - "it could run via backend X" (format + architecture + host),
//! - "llama.cpp supports this architecture" (allow-list below).
//!
//! The allow-list is a heuristic snapshot of llama.cpp architecture
//! support, matched against normalized HF `architectures` strings. It is
//! data, not UI logic; unknown architectures resolve to
//! `ArchitectureUnsupported`, never to silent mis-execution.

use super::backend::{Availability, BackendProvider};
use super::format::{ModelFormat, ModelLayout};
use super::hardware::HardwareInfo;

/// llama.cpp-supported architecture families (lowercase).
/// Matched by EXACT `arch_family()` equality — never substring
/// containment. This list is a heuristic snapshot of llama.cpp support,
/// not authoritative runtime capability detection; do not grow it into
/// a detection project. The long-term source of truth is the installed
/// runtime's own capability query.
pub const LLAMACPP_SUPPORTED_ARCHS: &[&str] = &[
    "llama",
    "mistral",
    "mixtral",
    "qwen2",
    "qwen3",
    "qwen2moe",
    "gemma",
    "gemma2",
    "gemma3",
    "phi2",
    "phi3",
    "phi-3",
    "falcon",
    "gpt2",
    "gpt-j",
    "gpt-neox",
    "mpt",
    "bloom",
    "bert",
    "nomic-bert",
    "arctic",
    "deepseek",
    "deepseek2",
    "glm",
    "chatglm",
    "internlm2",
    "llava",
    "qwen2vl",
    "stablelm",
    "starcoder",
    "starcoder2",
    "openelm",
    "exaone",
    "granite",
    "bailingmoe",
    "command-r",
    "cohere",
    "dbrx",
    "olmo",
    "olmo2",
    "olmoe",
    "t5",
    "jais",
    "persimmon",
    "refact",
    "smollm",
];

/// Lowercase + trim + strip common separators. Kept for display and
/// fallback paths; compatibility itself uses [`arch_family`].
pub fn normalize_arch(raw: &str) -> String {
    raw.trim().to_lowercase().replace(['-', '_'], "")
}

/// Suffixes stripped to reach the architecture family
/// (`Qwen2ForCausalLM` → `qwen2`). Conservative list: task heads only.
const ARCH_TASK_SUFFIXES: &[&str] = &[
    "forcausallm",
    "forsequenceclassification",
    "fortokenclassification",
    "forquestionanswering",
    "formaskedlm",
    "forconditionalgeneration",
];

/// Normalize a raw HF `architectures` entry (or `model_type`) to an
/// explicit family string. Exact-equality compares on families — never
/// substring containment, which false-positives as the table grows.
pub fn arch_family(raw: &str) -> String {
    let mut s: String = raw
        .trim()
        .to_lowercase()
        .chars()
        .filter(|c| c.is_ascii_alphanumeric())
        .collect();
    for suffix in ARCH_TASK_SUFFIXES {
        if let Some(stripped) = s.strip_suffix(suffix) {
            if !stripped.is_empty() {
                s = stripped.to_string();
                break;
            }
        }
    }
    s
}

/// Verdict for one backend against one model artifact.
#[derive(Debug, Clone)]
pub struct Compatibility {
    pub compatible: bool,
    pub reasons: Vec<String>,
}

impl Compatibility {
    pub fn ok() -> Self {
        Self {
            compatible: true,
            reasons: Vec::new(),
        }
    }

    pub fn no(reason: impl Into<String>) -> Self {
        Self {
            compatible: false,
            reasons: vec![reason.into()],
        }
    }

    pub fn and(mut self, other: Compatibility) -> Self {
        if !other.compatible {
            self.compatible = false;
            self.reasons.extend(other.reasons);
        }
        self
    }
}

/// Check one provider against detected weights + layout + architecture.
pub fn check_compatibility(
    provider: &dyn BackendProvider,
    format: ModelFormat,
    layout: ModelLayout,
    architecture: Option<&str>,
    _hardware: &HardwareInfo,
) -> Compatibility {
    let caps = provider.capabilities();
    if !caps.available_on_this_host {
        return Compatibility::no(format!("{} is unavailable on this host", provider.name()));
    }
    if let Availability::Unavailable { reason } = provider.availability() {
        return Compatibility::no(reason);
    }
    // Remote/service backends run whatever the endpoint serves: repository
    // artifacts never decide their compatibility — endpoint configuration
    // (Phase 6) does. Local backends must match weights + layout.
    if caps.requires_local_artifact {
        if !caps.formats.contains(&format) {
            return Compatibility::no(format!(
                "{} does not run {} weights",
                provider.name(),
                format.label()
            ));
        }
        if !caps.layouts.is_empty() && !caps.layouts.contains(&layout) {
            return Compatibility::no(format!(
                "{} needs {} layout, model is {}",
                provider.name(),
                caps.layouts
                    .iter()
                    .map(|l| l.label())
                    .collect::<Vec<_>>()
                    .join("/"),
                layout.label()
            ));
        }
    }
    if !caps.architectures.is_empty() {
        match architecture {
            Some(arch) => {
                let family = arch_family(arch);
                let ok = caps.architectures.iter().any(|a| arch_family(a) == family);
                if !ok {
                    return Compatibility::no(format!(
                        "{} does not support architecture {arch} (family {family})",
                        provider.name()
                    ));
                }
            }
            None => {
                return Compatibility::no(format!(
                    "{} requires a known architecture, none detected",
                    provider.name()
                ));
            }
        }
    }
    Compatibility::ok()
}

#[cfg(test)]
mod tests {
    use super::super::backend::{
        LlamaCppProvider, MlxProvider, OllamaProvider, OpenAiCompatibleProvider,
        TransformersProvider,
    };
    use super::super::format::LayoutSource;
    use super::*;

    fn hw() -> HardwareInfo {
        HardwareInfo::detect()
    }

    fn check(
        p: &dyn BackendProvider,
        f: ModelFormat,
        l: ModelLayout,
        a: Option<&str>,
    ) -> Compatibility {
        check_compatibility(p, f, l, a, &hw())
    }

    #[test]
    fn test_llamacpp_gguf_llama_ok() {
        let p = LlamaCppProvider;
        let c = check(
            &p,
            ModelFormat::Gguf,
            ModelLayout::Unknown,
            Some("LlamaForCausalLM"),
        );
        assert!(c.compatible, "{:?}", c.reasons);
    }

    #[test]
    fn test_llamacpp_rejects_safetensors_and_unknown_arch() {
        let p = LlamaCppProvider;
        let c = check(
            &p,
            ModelFormat::SafeTensors,
            ModelLayout::StandardHf,
            Some("LlamaForCausalLM"),
        );
        assert!(!c.compatible);
        let c = check(
            &p,
            ModelFormat::Gguf,
            ModelLayout::Unknown,
            Some("SomeNewModelForCausalLM"),
        );
        assert!(!c.compatible);
        assert!(c.reasons.iter().any(|r| r.contains("architecture")));
    }

    #[test]
    fn test_transformers_availability_follows_probe() {
        // Phase 4: real provider — compatibility tracks the actual
        // dependency probe, not a phase gate.
        let p = TransformersProvider;
        let probe_ok = super::super::transformers::probe_cached("python3").is_ok();
        let c = check(
            &p,
            ModelFormat::SafeTensors,
            ModelLayout::StandardHf,
            Some("LlamaForCausalLM"),
        );
        assert_eq!(c.compatible, probe_ok, "{:?}", c.reasons);
    }

    #[test]
    fn test_ollama_is_format_generic() {
        let p = OllamaProvider;
        let c = check(&p, ModelFormat::SafeTensors, ModelLayout::StandardHf, None);
        assert!(c.compatible, "{:?}", c.reasons);
    }

    #[test]
    fn test_mlx_gated_by_host() {
        let p = MlxProvider;
        let c = check(
            &p,
            ModelFormat::SafeTensors,
            ModelLayout::Mlx,
            Some("LlamaForCausalLM"),
        );
        // This Linux host has no Metal: must be unavailable, never forced.
        assert!(!c.compatible);
    }

    #[test]
    fn test_openai_compatible_needs_endpoint() {
        let p = OpenAiCompatibleProvider;
        let c = check(&p, ModelFormat::Gguf, ModelLayout::Unknown, None);
        assert!(!c.compatible);
    }

    #[test]
    fn test_layout_selects_runtime_for_same_weights() {
        // Same SafeTensors bytes, different layouts → different runtimes.
        // (Providers are Phase-gated here, so assert on reasons instead.)
        let mlx = MlxProvider;
        let c = check(
            &mlx,
            ModelFormat::SafeTensors,
            ModelLayout::StandardHf,
            Some("LlamaForCausalLM"),
        );
        assert!(!c.compatible);
        assert!(
            c.reasons
                .iter()
                .any(|r| r.contains("layout") || r.contains("host"))
        );
        let tf = TransformersProvider;
        let c = check(
            &tf,
            ModelFormat::SafeTensors,
            ModelLayout::Mlx,
            Some("LlamaForCausalLM"),
        );
        assert!(!c.compatible);
        // Rejected by layout gate, host gate, or the real dependency probe.
        assert!(
            c.reasons.iter().any(|r| {
                let l = r.to_lowercase();
                l.contains("layout")
                    || l.contains("host")
                    || l.contains("transformers")
                    || l.contains("python")
                    || l.contains("torch")
            }),
            "{:?}",
            c.reasons
        );
    }

    #[test]
    fn test_layout_detection() {
        let mlx_files = vec!["model.safetensors".to_string()];
        let (layout, source) =
            ModelLayout::detect("mlx-community/Llama-3-8B-MLX", &mlx_files, false);
        assert_eq!(layout, ModelLayout::Mlx);
        assert_eq!(source, LayoutSource::Heuristic);
        // MLX repos still inventory as SafeTensors weights.
        assert!(ModelFormat::detect_in_files(&mlx_files).contains(&ModelFormat::SafeTensors));
        let hf_files = vec!["config.json".to_string(), "model.safetensors".to_string()];
        let (layout, source) = ModelLayout::detect("org/model", &hf_files, true);
        assert_eq!(layout, ModelLayout::StandardHf);
        assert_eq!(source, LayoutSource::Metadata);
        let gguf_files = vec!["model-Q4_K_M.gguf".to_string()];
        let (layout, source) = ModelLayout::detect("org/model-gguf", &gguf_files, false);
        assert_eq!(layout, ModelLayout::Unknown);
        assert_eq!(source, LayoutSource::Undetermined);
        // The review's dangerous example is now rejected: `mlx` buried
        // mid-name with no org/suffix/token evidence is NOT MLX.
        let (layout, source) = ModelLayout::detect("org/some-project-mlx-test", &hf_files, true);
        assert_eq!(layout, ModelLayout::StandardHf);
        assert_eq!(source, LayoutSource::Metadata);
    }

    #[test]
    fn test_arch_family_exact_match() {
        assert_eq!(arch_family("LlamaForCausalLM"), "llama");
        assert_eq!(arch_family("Qwen2ForCausalLM"), "qwen2");
        assert_eq!(arch_family("phi-3"), "phi3");
        // No substring luck: a superstring family must NOT match llama.
        assert_ne!(arch_family("SomeNewModelForCausalLM"), "llama");
        let p = LlamaCppProvider;
        let c = check(
            &p,
            ModelFormat::Gguf,
            ModelLayout::Unknown,
            Some("Qwen2ForCausalLM"),
        );
        assert!(c.compatible, "{:?}", c.reasons);
    }

    #[test]
    fn test_mlx_metadata_confirmation() {
        assert!(ModelLayout::confirm_mlx_metadata(
            r#"{"model_type":"llama","mlx_backend":"x"}"#
        ));
        assert!(!ModelLayout::confirm_mlx_metadata(
            r#"{"model_type":"llama"}"#
        ));
        // Prose VALUES mentioning MLX are not evidence — only JSON keys.
        assert!(!ModelLayout::confirm_mlx_metadata(
            r#"{"model_type":"llama","description":"This model is compatible with MLX"}"#
        ));
    }
}
