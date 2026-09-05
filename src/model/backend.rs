//! Backend kinds, capabilities, and provider definitions.
//!
//! Capabilities describe what a backend CAN do; the `compatibility`
//! module turns capabilities + model facts + hardware into verdicts.
//! No compatibility decision lives in the UI.

use super::format::{ModelFormat, ModelLayout};
use super::hardware::HardwareInfo;

/// Inference backends Hercules knows about. Only the first three have
/// implementations today; the rest are capability entries so the resolver
/// can explain "no compatible backend" precisely.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BackendKind {
    /// Native llama.cpp — GGUF, supported architectures only.
    LlamaCpp,
    /// Ollama daemon (local HTTP) — whatever the daemon serves.
    Ollama,
    /// Burn/WGPU demo engine (gpu feature).
    BurnWgpu,
    /// Hugging Face Transformers via isolated Python worker (Phase 4).
    Transformers,
    /// Apple MLX / mlx-lm, Apple Silicon only (Phase 5).
    Mlx,
    /// Any OpenAI-compatible HTTP server: vLLM, llama.cpp server,
    /// Ollama, LM Studio (Phase 6).
    OpenAiCompatible,
}

impl BackendKind {
    pub fn label(self) -> &'static str {
        match self {
            Self::LlamaCpp => "llama.cpp",
            Self::Ollama => "Ollama",
            Self::BurnWgpu => "Burn/WGPU",
            Self::Transformers => "Transformers",
            Self::Mlx => "MLX",
            Self::OpenAiCompatible => "OpenAI-compatible",
        }
    }
}

/// Whether a backend can currently be used on this machine.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Availability {
    Available,
    Unavailable { reason: String },
}

/// Static capabilities of one backend.
#[derive(Debug, Clone)]
pub struct BackendCapabilities {
    pub kind: BackendKind,
    pub formats: Vec<ModelFormat>,
    /// Empty = any layout (e.g. GGUF needs no HF layout; Ollama delegates).
    /// Otherwise the model layout must be listed: SafeTensors + MLX layout
    /// → MLX-LM, SafeTensors + Standard HF layout → Transformers.
    pub layouts: Vec<ModelLayout>,
    /// Empty = any architecture the format allows (e.g. Ollama serves
    /// whatever it was built with); otherwise an allow-list matched
    /// against the normalized `architectures` field of HF config.json.
    pub architectures: Vec<String>,
    pub streaming: bool,
    pub cancellation: bool,
    pub tool_calling: bool,
    pub vision: bool,
    pub quantization: bool,
    /// False on non-Apple-Silicon hosts (MLX); gates advertisement.
    pub available_on_this_host: bool,
    /// False for remote/service backends (OpenAI-compatible): they run
    /// whatever the endpoint serves, so repository artifacts never
    /// decide their compatibility — endpoint configuration does.
    pub requires_local_artifact: bool,
}

impl BackendCapabilities {
    pub fn llamacpp() -> Self {
        Self {
            kind: BackendKind::LlamaCpp,
            formats: vec![ModelFormat::Gguf],
            layouts: Vec::new(),
            architectures: super::compatibility::LLAMACPP_SUPPORTED_ARCHS
                .iter()
                .map(|s| s.to_string())
                .collect(),
            streaming: true,
            cancellation: true,
            tool_calling: true,
            vision: true,
            quantization: true,
            available_on_this_host: true,
            requires_local_artifact: true,
        }
    }

    pub fn ollama() -> Self {
        Self {
            kind: BackendKind::Ollama,
            // The daemon serves its own models; format/arch checks are
            // delegated to it (errors stay useful and specific).
            formats: vec![
                ModelFormat::Gguf,
                ModelFormat::SafeTensors,
                ModelFormat::Unknown,
            ],
            layouts: Vec::new(),
            architectures: Vec::new(),
            streaming: true,
            cancellation: true,
            tool_calling: true,
            vision: true,
            quantization: true,
            available_on_this_host: true,
            requires_local_artifact: true,
        }
    }

    pub fn burn_wgpu() -> Self {
        Self {
            kind: BackendKind::BurnWgpu,
            // Truthful: the existing demo engine loads no repository
            // artifacts at all, so it never auto-resolves to a repo.
            formats: Vec::new(),
            layouts: Vec::new(),
            architectures: Vec::new(),
            streaming: false,
            cancellation: true,
            tool_calling: false,
            vision: false,
            quantization: false,
            #[cfg(feature = "gpu")]
            available_on_this_host: true,
            #[cfg(not(feature = "gpu"))]
            available_on_this_host: false,
            requires_local_artifact: true,
        }
    }

    pub fn transformers() -> Self {
        Self {
            kind: BackendKind::Transformers,
            formats: vec![ModelFormat::SafeTensors, ModelFormat::PyTorch],
            layouts: vec![ModelLayout::StandardHf],
            architectures: Vec::new(),
            streaming: true,
            cancellation: true,
            tool_calling: false,
            vision: false,
            quantization: false,
            // Host gate is decided by the real dependency probe in
            // TransformersProvider::availability, not here.
            available_on_this_host: true,
            requires_local_artifact: true,
        }
    }

    pub fn mlx() -> Self {
        Self {
            kind: BackendKind::Mlx,
            // MLX weights ARE SafeTensors — the layout selects the runtime.
            formats: vec![ModelFormat::SafeTensors],
            layouts: vec![ModelLayout::Mlx],
            architectures: Vec::new(),
            streaming: true,
            cancellation: true,
            tool_calling: false,
            vision: false,
            quantization: true,
            available_on_this_host: HardwareInfo::detect().supports_metal,
            requires_local_artifact: true,
        }
    }

    pub fn openai_compatible() -> Self {
        Self {
            kind: BackendKind::OpenAiCompatible,
            formats: vec![
                ModelFormat::Gguf,
                ModelFormat::SafeTensors,
                ModelFormat::Unknown,
            ],
            layouts: Vec::new(),
            architectures: Vec::new(),
            streaming: true,
            cancellation: true,
            tool_calling: true,
            vision: false,
            quantization: true,
            // Needs a configured endpoint (Phase 6); generic entry only.
            // Remote: repository artifacts never decide compatibility.
            available_on_this_host: false,
            requires_local_artifact: false,
        }
    }
}

/// A backend provider: answers identity, availability, capabilities and
/// compatibility. Adding a future backend = implementing this once.
pub trait BackendProvider: Send + Sync {
    fn kind(&self) -> BackendKind;
    fn name(&self) -> &'static str;
    fn capabilities(&self) -> BackendCapabilities;
    fn availability(&self) -> Availability {
        if self.capabilities().available_on_this_host {
            Availability::Available
        } else {
            Availability::Unavailable {
                reason: format!("{} is not available on this host", self.name()),
            }
        }
    }
}

macro_rules! static_provider_caps {
    ($name:ident, $label:literal, $kind:ident, $ctor:ident) => {
        pub struct $name;
        impl BackendProvider for $name {
            fn kind(&self) -> BackendKind {
                BackendKind::$kind
            }
            fn name(&self) -> &'static str {
                $label
            }
            fn capabilities(&self) -> BackendCapabilities {
                BackendCapabilities::$ctor()
            }
        }
    };
}

static_provider_caps!(LlamaCppProvider, "llama.cpp", LlamaCpp, llamacpp);
static_provider_caps!(OllamaProvider, "Ollama", Ollama, ollama);
static_provider_caps!(BurnWgpuProvider, "Burn/WGPU", BurnWgpu, burn_wgpu);
/// Transformers provider with REAL availability: probes the configured
/// Python for `transformers` + `torch` (cached per interpreter) instead
/// of reporting phase-gated unavailability.
pub struct TransformersProvider;

impl BackendProvider for TransformersProvider {
    fn kind(&self) -> BackendKind {
        BackendKind::Transformers
    }
    fn name(&self) -> &'static str {
        "Transformers"
    }
    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities::transformers()
    }
    fn availability(&self) -> Availability {
        let python = crate::settings::get_transformers_python()
            .filter(|s| !s.trim().is_empty())
            .unwrap_or_else(|| {
                std::env::var("HERCULES_TRANSFORMERS_PYTHON")
                    .unwrap_or_else(|_| "python3".to_string())
            });
        match super::transformers::probe_cached(&python) {
            Ok(report) => {
                let _ = report;
                Availability::Available
            }
            Err(e) => Availability::Unavailable {
                reason: e.message(),
            },
        }
    }
}

static_provider_caps!(MlxProvider, "MLX", Mlx, mlx);
static_provider_caps!(
    OpenAiCompatibleProvider,
    "OpenAI-compatible",
    OpenAiCompatible,
    openai_compatible
);
