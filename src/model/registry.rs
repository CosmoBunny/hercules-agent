//! Backend registry: one extension point for all backends.
//!
//! Adding a future backend (vLLM, TGI, …) means implementing
//! `BackendProvider` once and registering it here — never scattering
//! another `match` across the UI.

use super::backend::{
    BackendKind, BackendProvider, BurnWgpuProvider, LlamaCppProvider, MlxProvider, OllamaProvider,
    OpenAiCompatibleProvider, TransformersProvider,
};
use super::compatibility::{Compatibility, check_compatibility};
use super::format::{ModelFormat, ModelLayout};
use super::hardware::HardwareInfo;

/// All known backend providers, in registration order.
pub struct BackendRegistry {
    providers: Vec<Box<dyn BackendProvider>>,
}

impl BackendRegistry {
    pub fn new() -> Self {
        Self {
            providers: Vec::new(),
        }
    }

    pub fn register<P: BackendProvider + 'static>(&mut self, provider: P) {
        // One provider per kind: re-registering replaces (last wins).
        self.providers.retain(|p| p.kind() != provider.kind());
        self.providers.push(Box::new(provider));
    }

    /// Registry with every known provider, including not-yet-implemented
    /// ones (they report Unavailable with a reason instead of vanishing,
    /// so diagnostics can name them).
    pub fn default_registry() -> Self {
        let mut r = Self::new();
        r.register(LlamaCppProvider);
        r.register(OllamaProvider);
        r.register(BurnWgpuProvider);
        r.register(TransformersProvider);
        r.register(MlxProvider);
        r.register(OpenAiCompatibleProvider);
        r
    }

    pub fn kinds(&self) -> Vec<BackendKind> {
        self.providers.iter().map(|p| p.kind()).collect()
    }

    pub fn get(&self, kind: BackendKind) -> Option<&dyn BackendProvider> {
        self.providers
            .iter()
            .find(|p| p.kind() == kind)
            .map(|b| b.as_ref())
    }

    /// Providers usable on this host right now (for UI/API lists).
    pub fn available_backends(&self) -> Vec<BackendKind> {
        use super::backend::Availability;
        self.providers
            .iter()
            .filter(|p| matches!(p.availability(), Availability::Available))
            .map(|p| p.kind())
            .collect()
    }

    /// All providers compatible with detected weights + layout + arch.
    /// Returns (kind, verdict) pairs so callers can render reasons.
    pub fn compatible_backends(
        &self,
        format: ModelFormat,
        layout: ModelLayout,
        architecture: Option<&str>,
        hardware: &HardwareInfo,
    ) -> Vec<(BackendKind, Compatibility)> {
        self.providers
            .iter()
            .map(|p| {
                let verdict =
                    check_compatibility(p.as_ref(), format, layout, architecture, hardware);
                (p.kind(), verdict)
            })
            .filter(|(_, v)| v.compatible)
            .collect()
    }
}

impl Default for BackendRegistry {
    fn default() -> Self {
        Self::default_registry()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_registry_lists_all_kinds_once() {
        let r = BackendRegistry::default_registry();
        let kinds = r.kinds();
        assert_eq!(kinds.len(), 6);
        // Re-registering replaces instead of duplicating.
        let mut r = r;
        r.register(LlamaCppProvider);
        assert_eq!(r.kinds().len(), 6);
    }

    #[test]
    fn test_gguf_llama_resolves_to_llamacpp() {
        let r = BackendRegistry::default_registry();
        let hw = HardwareInfo::detect();
        let found = r.compatible_backends(
            ModelFormat::Gguf,
            ModelLayout::Unknown,
            Some("LlamaForCausalLM"),
            &hw,
        );
        let kinds: Vec<_> = found.iter().map(|(k, _)| *k).collect();
        assert!(kinds.contains(&BackendKind::LlamaCpp));
        // Transformers must NOT claim GGUF.
        assert!(!kinds.contains(&BackendKind::Transformers));
    }

    #[test]
    fn test_unknown_arch_has_no_silent_match() {
        let r = BackendRegistry::default_registry();
        let hw = HardwareInfo::detect();
        let found = r.compatible_backends(
            ModelFormat::SafeTensors,
            ModelLayout::StandardHf,
            Some("SomeNewModelForCausalLM"),
            &hw,
        );
        // Ollama is format-generic so it may match; llama.cpp must not.
        assert!(!found.iter().any(|(k, _)| *k == BackendKind::LlamaCpp));
        assert!(!found.iter().any(|(k, _)| *k == BackendKind::Mlx));
    }

    #[test]
    fn test_available_backends_reflects_host() {
        // This host: llama.cpp + Ollama available; BurnWgpu (no gpu
        // feature), MLX/OpenAI (Phase-gated) are not. Transformers tracks
        // the real dependency probe.
        let avail = BackendRegistry::default_registry().available_backends();
        assert!(avail.contains(&BackendKind::LlamaCpp));
        assert!(avail.contains(&BackendKind::Ollama));
        assert!(!avail.contains(&BackendKind::BurnWgpu));
        assert!(!avail.contains(&BackendKind::Mlx));
        assert!(!avail.contains(&BackendKind::OpenAiCompatible));
        let probe_ok = super::super::transformers::probe_cached("python3").is_ok();
        assert_eq!(avail.contains(&BackendKind::Transformers), probe_ok);
    }

    #[test]
    fn test_burnwgpu_never_auto_resolves_to_repo() {
        // The demo engine loads no repository artifacts: empty formats
        // means no repo can select it, on any host.
        let r = BackendRegistry::default_registry();
        let hw = HardwareInfo::detect();
        for format in [
            ModelFormat::Gguf,
            ModelFormat::SafeTensors,
            ModelFormat::Unknown,
        ] {
            for layout in [
                ModelLayout::StandardHf,
                ModelLayout::Mlx,
                ModelLayout::Unknown,
            ] {
                let found = r.compatible_backends(format, layout, Some("LlamaForCausalLM"), &hw);
                assert!(
                    !found.iter().any(|(k, _)| *k == BackendKind::BurnWgpu),
                    "BurnWgpu claimed {format:?}/{layout:?}"
                );
            }
        }
    }
}
