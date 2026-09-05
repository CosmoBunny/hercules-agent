//! Deterministic model resolution: manifest → ranked backends.
//!
//! Pipeline: `resolve(manifest)` → inspect → detect arch → detect
//! weights/layout → query capabilities → filter → rank → select.
//! Priority: explicit user choice → native local → local artifact →
//! hardware-optimized → generic → remote. Never prefers one backend
//! blindly; every rejection carries its reason for diagnostics.

use super::backend::BackendKind;
use super::compatibility::Compatibility;
use super::error::ModelError;
use super::format::{LayoutSource, ModelFormat, ModelLayout};
use super::hardware::HardwareInfo;
use super::manifest::{Artifact, RepoManifest};
use super::registry::BackendRegistry;

/// Deterministic rank: lower wins. User preference is handled by the
/// caller short-circuiting before `resolve` (explicit choice always wins).
fn backend_rank(kind: BackendKind, hardware: &HardwareInfo) -> u32 {
    match kind {
        // Native local GGUF engine first where it can run the artifact.
        BackendKind::LlamaCpp => 10,
        // Daemon-backed local inference second.
        BackendKind::Ollama => 20,
        // Hardware-optimized Apple path when present.
        BackendKind::Mlx => {
            if hardware.supports_metal {
                15
            } else {
                900
            }
        }
        BackendKind::BurnWgpu => 800,
        // Generic local fallback (Phase 4+).
        BackendKind::Transformers => 40,
        // Remote last: needs an endpoint and leaves the machine.
        BackendKind::OpenAiCompatible => 100,
    }
}

/// A resolved model: what it is, which artifact+backend pairs can run
/// it, what was chosen, why. A repository can hold several usable
/// artifacts — resolution selects an artifact AND a backend together,
/// never one format first and a backend second.
#[derive(Debug, Clone)]
pub struct ResolvedModel {
    pub repo: String,
    pub revision: String,
    pub architecture: Option<String>,
    /// Every detected artifact with its own files, layout and byte cost.
    pub artifacts: Vec<Artifact>,
    pub tokenizer_present: bool,
    /// Repo-wide byte upper bound (all artifacts — NOT the download plan).
    pub weight_bytes: u64,
    /// Byte cost of the selected artifact only (the actual download plan).
    pub selected_artifact_bytes: Option<u64>,
    /// Compatible (artifact id, backend) pairs, best first. The id pins
    /// the exact file set — never re-derive it from the format later.
    pub candidates: Vec<(String, BackendKind)>,
    /// The selected pair, if any.
    pub selected: Option<(String, BackendKind)>,
    /// Per-pair rejection reasons (for the diagnostic renderer).
    pub rejections: Vec<(String, BackendKind, Vec<String>)>,
}

impl ResolvedModel {
    /// Detected weight formats across artifacts.
    pub fn weight_formats(&self) -> Vec<ModelFormat> {
        self.artifacts.iter().map(|a| a.format).collect()
    }

    /// Look up the exact artifact behind a candidate/selected id.
    pub fn artifact(&self, id: &str) -> Option<&Artifact> {
        self.artifacts.iter().find(|a| a.id() == id)
    }

    /// Primary artifact for single-artifact displays.
    pub fn primary_weights(&self) -> ModelFormat {
        self.artifacts
            .first()
            .map(|a| a.format)
            .unwrap_or(ModelFormat::Unknown)
    }

    /// Human diagnostic in the review-specified shape. Never a bare
    /// "download failed": states formats, reasons and a suggested action.
    pub fn diagnose(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!("Model:\n    {}\n\n", self.repo));
        out.push_str(&format!(
            "Architecture:\n    {}\n\n",
            self.architecture.as_deref().unwrap_or("unknown")
        ));
        out.push_str("Artifacts:\n");
        if self.artifacts.is_empty() {
            out.push_str("    none\n");
        } else {
            for a in &self.artifacts {
                let layout_note = match a.layout_source {
                    LayoutSource::Heuristic => " (layout heuristic — unconfirmed)",
                    LayoutSource::Metadata => "",
                    LayoutSource::Undetermined => "",
                };
                out.push_str(&format!(
                    "    {} ({} files, {} bytes, layout={}{layout_note}{})\n",
                    a.format.label(),
                    a.files.len(),
                    a.bytes,
                    a.layout.label(),
                    if a.sharded { ", sharded" } else { "" },
                ));
            }
        }
        out.push_str(&format!(
            "\nTokenizer: {}\nRepo total: {} bytes\nSelected download: {} bytes\n\n",
            if self.tokenizer_present { "yes" } else { "no" },
            self.weight_bytes,
            self.selected_artifact_bytes
                .map(|b| b.to_string())
                .unwrap_or_else(|| "n/a".to_string()),
        ));
        out.push_str("Compatible artifact + backend pairs:\n");
        if self.candidates.is_empty() {
            out.push_str("    None\n");
        } else {
            for (id, k) in &self.candidates {
                out.push_str(&format!("    {id} via {}\n", k.label()));
            }
        }
        out.push_str("\nReasons:\n");
        for (id, kind, reasons) in &self.rejections {
            for r in reasons {
                out.push_str(&format!("    {id} via {}: {}\n", kind.label(), r));
            }
        }
        out.push_str("\nSuggested action:\n");
        if let Some((id, sel)) = &self.selected {
            let bytes = self.artifact(id).map(|a| a.bytes).unwrap_or(0);
            let files = self
                .artifact(id)
                .map(|a| a.files.join(", "))
                .unwrap_or_default();
            out.push_str(&format!(
                "    Selected:\n        artifact: {id}\n        files: {files}\n        backend: {}\n        downloads ~{bytes} bytes.\n",
                sel.label(),
            ));
        } else {
            out.push_str(&suggested_action(
                &self.weight_formats(),
                self.artifacts.iter().any(|a| a.layout == ModelLayout::Mlx),
                self.architecture.as_deref(),
            ));
        }
        out
    }
}

fn suggested_action(
    artifacts: &[ModelFormat],
    has_mlx_layout: bool,
    architecture: Option<&str>,
) -> String {
    // No weights at all dominates: architecture is moot without artifacts.
    if artifacts.is_empty() {
        return "    No recognizable weight files found; check the repository contents.\n"
            .to_string();
    }
    if architecture.is_none() {
        return "    Repository metadata has no recognizable architecture; \
                check config.json or choose a model with explicit metadata.\n"
            .to_string();
    }
    if artifacts.contains(&ModelFormat::Gguf) {
        return "    A GGUF artifact exists but no local backend supports this \
                architecture; try Ollama or wait for llama.cpp support.\n"
            .to_string();
    }
    if has_mlx_layout {
        return "    Use an Apple Silicon host with the MLX backend, \
             or use a Transformers-compatible build.\n"
            .to_string();
    }
    "    Install the Transformers backend, use a GGUF build, \
     or choose another supported model.\n"
        .to_string()
}

/// Format preference inside equal backend rank: a directly usable GGUF
/// beats downloading full SafeTensors (instruction §8).
fn format_rank(format: ModelFormat) -> u32 {
    match format {
        ModelFormat::Gguf => 0,
        ModelFormat::SafeTensors => 1,
        ModelFormat::PyTorch => 2,
        ModelFormat::Unknown => 3,
    }
}

/// Resolve a manifest against the registry: per-artifact compatibility,
/// ranked pairs. Pure and deterministic. Each artifact is matched with
/// its OWN layout — a GGUF sidecar never inherits an HF layout.
pub fn resolve(
    manifest: &RepoManifest,
    registry: &BackendRegistry,
    hardware: &HardwareInfo,
) -> ResolvedModel {
    let arch = manifest.config.primary_arch().map(|s| s.to_string());
    // (artifact id, backend, backend_rank, format_rank)
    let mut compatible: Vec<(String, BackendKind, u32, u32)> = Vec::new();
    let mut rejections: Vec<(String, BackendKind, Vec<String>)> = Vec::new();
    // No recognizable weights: nothing can run it, not even format-generic
    // backends. This keeps empty repos out of misleading selections.
    if manifest.artifacts.is_empty() {
        for kind in registry.kinds() {
            rejections.push((
                "none".to_string(),
                kind,
                vec!["no recognizable weight files in repository".to_string()],
            ));
        }
    } else {
        for artifact in &manifest.artifacts {
            let id = artifact.id();
            for kind in registry.kinds() {
                if let Some(provider) = registry.get(kind) {
                    let verdict: Compatibility = super::compatibility::check_compatibility(
                        provider,
                        artifact.format,
                        artifact.layout,
                        arch.as_deref(),
                        hardware,
                    );
                    if verdict.compatible {
                        compatible.push((
                            id.clone(),
                            kind,
                            backend_rank(kind, hardware),
                            format_rank(artifact.format),
                        ));
                    } else {
                        rejections.push((id.clone(), kind, verdict.reasons));
                    }
                }
            }
        }
    }
    compatible.sort_by_key(|(_, _, brank, frank)| (*brank, *frank));
    let candidates: Vec<(String, BackendKind)> = compatible
        .into_iter()
        .map(|(id, k, _, _)| (id, k))
        .collect();
    let selected = candidates.first().cloned();
    let selected_artifact_bytes = selected.as_ref().and_then(|(id, _)| {
        manifest
            .artifacts
            .iter()
            .find(|a| a.id() == *id)
            .map(|a| a.bytes)
    });
    ResolvedModel {
        repo: manifest.repo.clone(),
        revision: manifest.revision.clone(),
        architecture: arch,
        artifacts: manifest.artifacts.clone(),
        tokenizer_present: manifest.tokenizer_present,
        weight_bytes: manifest.weight_bytes(),
        selected_artifact_bytes,
        candidates,
        selected,
        rejections,
    }
}

/// Resolve with an explicit user backend choice. The explicit kind
/// bypasses automatic ranking but NEVER bypasses validation: the best
/// compatible artifact for that kind is selected, and an incompatible
/// choice fails typed instead of silently switching elsewhere.
pub fn resolve_with_preference(
    manifest: &RepoManifest,
    registry: &BackendRegistry,
    hardware: &HardwareInfo,
    preferred: Option<BackendKind>,
) -> Result<ResolvedModel, ModelError> {
    let Some(kind) = preferred else {
        return Ok(resolve(manifest, registry, hardware));
    };
    let arch = manifest.config.primary_arch().map(|s| s.to_string());
    let mut ranked: Vec<&Artifact> = manifest.artifacts.iter().collect();
    ranked.sort_by_key(|a| format_rank(a.format));
    let mut first_err: Option<ModelError> = None;
    for art in ranked {
        match super::runtime::validate_explicit_selection(
            registry,
            kind,
            art,
            arch.as_deref(),
            hardware,
        ) {
            Ok(()) => {
                let mut base = resolve(manifest, registry, hardware);
                base.candidates = vec![(art.id(), kind)];
                base.selected = Some((art.id(), kind));
                base.selected_artifact_bytes = Some(art.bytes);
                return Ok(base);
            }
            Err(e) => {
                if first_err.is_none() {
                    first_err = Some(e);
                }
            }
        }
    }
    Err(first_err.unwrap_or(ModelError::FileUnavailable {
        repo: manifest.repo.clone(),
        file: "no artifacts to validate against".to_string(),
    }))
}
#[cfg(test)]
mod tests {
    use super::super::manifest::RepoFile;
    use super::*;

    fn manifest(repo: &str, files: &[&str], config: Option<&str>) -> RepoManifest {
        RepoManifest::from_files(
            repo,
            "main",
            files
                .iter()
                .map(|n| RepoFile {
                    name: n.to_string(),
                    size_bytes: Some(10),
                })
                .collect(),
            config,
        )
    }

    fn hw() -> HardwareInfo {
        HardwareInfo::detect()
    }

    #[test]
    fn test_gguf_llama_selects_llamacpp_with_reason() {
        let m = manifest(
            "org/llama-gguf",
            &["model-Q4_K_M.gguf"],
            Some(r#"{"architectures":["LlamaForCausalLM"]}"#),
        );
        let r = resolve(&m, &BackendRegistry::default_registry(), &hw());
        assert_eq!(
            r.selected,
            Some(("gguf:model-q4_k_m".to_string(), BackendKind::LlamaCpp))
        );
        assert_eq!(r.primary_weights(), ModelFormat::Gguf);
        // The id resolves back to the exact file set — never re-derived.
        let art = r.artifact("gguf:model-q4_k_m").unwrap();
        assert_eq!(art.files, vec!["model-Q4_K_M.gguf".to_string()]);
        let d = r.diagnose();
        assert!(d.contains("llama.cpp"));
        assert!(d.contains("Suggested action"));
    }

    #[test]
    fn test_safetensors_only_gives_actionable_diagnostic() {
        // Mirrors the instruction's example: SafeTensors, no GGUF/MLX.
        let m = manifest(
            "org/model",
            &[
                "config.json",
                "model-00001-of-00004.safetensors",
                "model-00002-of-00004.safetensors",
            ],
            Some(r#"{"architectures":["LlamaForCausalLM"]}"#),
        );
        let r = resolve(&m, &BackendRegistry::default_registry(), &hw());
        // llama.cpp must not claim SafeTensors; Transformers is Phase-gated.
        assert!(!r.candidates.iter().any(|(id, k)| {
            r.artifact(id).map(|a| a.format) == Some(ModelFormat::SafeTensors)
                && *k == BackendKind::LlamaCpp
        }));
        let d = r.diagnose();
        assert!(d.contains("SafeTensors"), "{d}");
        assert!(!d.contains("download failed"), "{d}");
        assert!(d.contains("GGUF") || d.contains("Transformers"), "{d}");
    }

    #[test]
    fn test_exotic_arch_diagnostic_names_architecture() {
        let m = manifest(
            "org/new",
            &["model.safetensors", "config.json"],
            Some(r#"{"architectures":["SomeNewModelForCausalLM"]}"#),
        );
        let r = resolve(&m, &BackendRegistry::default_registry(), &hw());
        let d = r.diagnose();
        assert!(d.contains("SomeNewModelForCausalLM"), "{d}");
        assert!(d.contains("Reasons:"), "{d}");
    }

    #[test]
    fn test_empty_repo_diagnostic() {
        let m = manifest("org/empty", &["README.md"], None);
        let r = resolve(&m, &BackendRegistry::default_registry(), &hw());
        assert_eq!(r.selected, None);
        assert!(r.diagnose().contains("No recognizable weight files"),);
    }

    #[test]
    fn test_multi_format_repo_selects_artifact_pair() {
        // The review's key case: GGUF + SafeTensors in one repo must yield
        // per-artifact pairs, with the GGUF artifact preferred.
        let m = manifest(
            "org/both",
            &["model-Q4_K_M.gguf", "model.safetensors", "config.json"],
            Some(r#"{"architectures":["LlamaForCausalLM"]}"#),
        );
        let r = resolve(&m, &BackendRegistry::default_registry(), &hw());
        assert_eq!(r.artifacts.len(), 2);
        // Best pair first: GGUF via llama.cpp (rank 10 beats Ollama 20).
        assert_eq!(
            r.selected,
            Some(("gguf:model-q4_k_m".to_string(), BackendKind::LlamaCpp))
        );
        // Both artifacts appear across candidates — nothing was reduced away.
        let formats: Vec<ModelFormat> = r
            .candidates
            .iter()
            .filter_map(|(id, _)| r.artifact(id).map(|a| a.format))
            .collect();
        assert!(formats.contains(&ModelFormat::Gguf));
        assert!(formats.contains(&ModelFormat::SafeTensors));
        let d = r.diagnose();
        assert!(d.contains("gguf:model-q4_k_m via llama.cpp"), "{d}");
        assert!(d.contains("artifact: gguf:model-q4_k_m"), "{d}");
    }

    #[test]
    fn test_two_gguf_quants_stay_separate_candidates() {
        // Same format twice: Q4 and Q8 must not collapse into one pair.
        let m = manifest(
            "org/quants",
            &["model-Q4_K_M.gguf", "model-Q8_0.gguf", "config.json"],
            Some(r#"{"architectures":["LlamaForCausalLM"]}"#),
        );
        let r = resolve(&m, &BackendRegistry::default_registry(), &hw());
        assert_eq!(
            r.selected.as_ref().map(|(id, _)| id.as_str()),
            Some("gguf:model-q4_k_m")
        );
        let ids: Vec<&str> = r.candidates.iter().map(|(id, _)| id.as_str()).collect();
        assert!(ids.contains(&"gguf:model-q4_k_m"));
        assert!(ids.contains(&"gguf:model-q8_0"));
        // Exact files behind the selected id — runtime loads Q4, never Q8.
        let art = r.artifact("gguf:model-q4_k_m").unwrap();
        assert_eq!(art.files, vec!["model-Q4_K_M.gguf".to_string()]);
    }

    #[test]
    fn test_resolution_is_deterministic() {
        let m = manifest(
            "org/m",
            &["a.gguf", "config.json"],
            Some(r#"{"architectures":["MistralForCausalLM"]}"#),
        );
        let reg = BackendRegistry::default_registry();
        let a = resolve(&m, &reg, &hw());
        let b = resolve(&m, &reg, &hw());
        assert_eq!(a.selected, b.selected);
        assert_eq!(a.candidates, b.candidates);
    }

    #[test]
    fn test_explicit_preference_validated_not_overridden() {
        let m = manifest(
            "org/both",
            &["model-Q4_K_M.gguf", "model.safetensors", "config.json"],
            Some(r#"{"architectures":["LlamaForCausalLM"]}"#),
        );
        let reg = BackendRegistry::default_registry();
        // Explicit llama.cpp: best GGUF artifact, ranking bypassed.
        let r = resolve_with_preference(&m, &reg, &hw(), Some(BackendKind::LlamaCpp)).unwrap();
        assert_eq!(
            r.selected.as_ref().map(|(id, _)| id.as_str()),
            Some("gguf:model-q4_k_m")
        );
        assert_eq!(r.candidates.len(), 1);
        // None = automatic ranking (GGUF+llama.cpp still wins here).
        let r = resolve_with_preference(&m, &reg, &hw(), None).unwrap();
        assert_eq!(
            r.selected.as_ref().map(|(id, _)| id.as_str()),
            Some("gguf:model-q4_k_m")
        );
        // Explicit but impossible: typed error, never a silent switch.
        let m2 = manifest(
            "org/st",
            &["model.safetensors", "config.json"],
            Some(r#"{"architectures":["LlamaForCausalLM"]}"#),
        );
        let err =
            resolve_with_preference(&m2, &reg, &hw(), Some(BackendKind::LlamaCpp)).unwrap_err();
        assert!(
            matches!(err, crate::model::ModelError::FormatUnsupported { .. }),
            "{err:?}"
        );
    }
}
