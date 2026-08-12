use ollama_rs::{generation::completion::request::GenerationRequest, Ollama};
#[cfg(feature = "gpu")]
use burn::backend::wgpu::WgpuDevice;
#[cfg(feature = "gpu")]
use burn::backend::Wgpu;
#[cfg(feature = "gpu")]
use burn::tensor::Tensor;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use crate::llama::{HttpInferenceClient, LlamaCppRuntime, LlamaCppLibRuntime};

#[derive(Clone)]
pub enum AgentBackend {
    #[cfg(feature = "gpu")]
    BurnWgpu(BurnWgpuBackend),
    /// In-process libllama.so engine (C FFI — no subprocess).
    LlamaCppLib(LlamaCppLibBackend),
    /// Official llama.cpp via CLI or managed llama-server.
    LlamaCpp(LlamaCppBackend),
    Ollama(OllamaBackend),
}

impl AgentBackend {
    pub async fn generate(&self, prompt: &str) -> Result<String, String> {
        match self {
            #[cfg(feature = "gpu")]
            Self::BurnWgpu(backend) => backend.generate(prompt).await,
            Self::LlamaCppLib(backend) => backend.generate(prompt).await,
            Self::LlamaCpp(backend) => backend.generate(prompt).await,
            Self::Ollama(backend) => backend.generate(prompt).await,
        }
    }

    /// Streaming generate — pushes tokens into stream_target as they arrive.
    pub async fn generate_stream(
        &self,
        prompt: &str,
        stream_target: Arc<Mutex<String>>,
        is_generating: Arc<Mutex<bool>>,
    ) -> Result<String, String> {
        match self {
            Self::LlamaCppLib(backend) => {
                backend
                    .generate_stream(prompt, stream_target, is_generating)
                    .await
            }
            Self::LlamaCpp(backend) => {
                backend
                    .generate_stream(prompt, stream_target, is_generating)
                    .await
            }
            Self::Ollama(backend) => {
                backend
                    .generate_stream(prompt, stream_target, is_generating)
                    .await
            }
            #[cfg(feature = "gpu")]
            Self::BurnWgpu(backend) => {
                let result = backend.generate(prompt).await?;
                if let Ok(mut target) = stream_target.lock() {
                    target.push_str(&result);
                }
                Ok(result)
            }
        }
    }

    pub fn current_model_path(&self) -> Option<PathBuf> {
        match self {
            Self::LlamaCppLib(b) => b.runtime.model_path.clone(),
            Self::LlamaCpp(b) => b.runtime.model_path(),
            _ => None,
        }
    }

    pub fn name(&self) -> String {
        match self {
            #[cfg(feature = "gpu")]
            Self::BurnWgpu(b) => format!("Burn/WGPU ({})", b.model_name),
            Self::LlamaCppLib(b) => b.name(),
            Self::LlamaCpp(b) => b.runtime.name(),
            Self::Ollama(b) => format!("Ollama ({})", b.model),
        }
    }
}

// ---------------------------------------------------------------------------
// llama.cpp in-process (libllama.so via C FFI)
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct LlamaCppLibBackend {
    pub runtime: LlamaCppLibRuntime,
}

impl LlamaCppLibBackend {
    pub fn http(endpoint: String, model_name: String) -> Self {
        Self {
            runtime: LlamaCppLibRuntime::with_endpoint(endpoint, model_name),
        }
    }

    pub fn gguf(path: impl Into<PathBuf>) -> Self {
        Self {
            runtime: LlamaCppLibRuntime::with_gguf(path),
        }
    }

    pub fn name(&self) -> String {
        if let Some(ref p) = self.runtime.model_path {
            format!(
                "llama.cpp lib ({})",
                p.file_name()
                    .map(|s| s.to_string_lossy().to_string())
                    .unwrap_or_else(|| p.display().to_string())
            )
        } else {
            format!("llama.cpp lib HTTP ({})", self.runtime.endpoint)
        }
    }

    pub async fn generate(&self, prompt: &str) -> Result<String, String> {
        self.runtime.generate(prompt).await
    }

    pub async fn generate_stream(
        &self,
        prompt: &str,
        stream_target: Arc<Mutex<String>>,
        is_generating: Arc<Mutex<bool>>,
    ) -> Result<String, String> {
        self.runtime
            .generate_stream(prompt, stream_target, is_generating)
            .await
    }
}

// Backward compat alias — old code may reference LlamaRsBackend
pub type LlamaRsBackend = LlamaCppLibBackend;

// ---------------------------------------------------------------------------
// llama.cpp — C/C++ server/CLI runtime
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct LlamaCppBackend {
    pub runtime: LlamaCppRuntime,
}

impl LlamaCppBackend {
    pub fn server(endpoint: String, model_name: String) -> Self {
        Self {
            runtime: LlamaCppRuntime::server(endpoint, model_name),
        }
    }

    pub fn cli(model_path: impl Into<PathBuf>) -> Self {
        Self {
            runtime: LlamaCppRuntime::cli(model_path),
        }
    }

    pub async fn generate(&self, prompt: &str) -> Result<String, String> {
        self.runtime.generate(prompt).await
    }

    pub async fn generate_stream(
        &self,
        prompt: &str,
        stream_target: Arc<Mutex<String>>,
        is_generating: Arc<Mutex<bool>>,
    ) -> Result<String, String> {
        self.runtime
            .generate_stream(prompt, stream_target, is_generating)
            .await
    }
}

// Keep old name working for any external references during transition.
pub type LlamaServerBackend = LlamaCppLibBackend;

// ---------------------------------------------------------------------------
// Ollama
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct OllamaBackend {
    ollama: Ollama,
    pub model: String,
}

impl OllamaBackend {
    pub fn new(model: String) -> Self {
        Self {
            ollama: Ollama::default(),
            model,
        }
    }

    pub async fn generate(&self, prompt: &str) -> Result<String, String> {
        let system = crate::agent::AgentEngine::system_prompt_for_cwd();
        let req = GenerationRequest::new(self.model.clone(), prompt.to_string()).system(system);
        match self.ollama.generate(req).await {
            Ok(res) => Ok(res.response),
            Err(e) => Err(format!(
                "[Ollama Error] Connection failed for model '{}': {}. Ensure local Ollama daemon is active.",
                self.model, e
            )),
        }
    }

    pub async fn generate_stream(
        &self,
        prompt: &str,
        stream_target: Arc<Mutex<String>>,
        is_generating: Arc<Mutex<bool>>,
    ) -> Result<String, String> {
        use futures_util::StreamExt;

        let system = crate::agent::AgentEngine::system_prompt_for_cwd();
        let req = GenerationRequest::new(self.model.clone(), prompt.to_string()).system(system);
        let mut stream = self.ollama.generate_stream(req).await.map_err(|e| {
            format!(
                "[Ollama Error] Stream failed for model '{}': {}. Ensure local Ollama daemon is active.",
                self.model, e
            )
        })?;

        let mut full_text = String::new();
        let mut thinking_active = false;

        while let Some(chunk_result) = stream.next().await {
            if let Ok(active_gen) = is_generating.lock() {
                if !*active_gen {
                    return Err("[Generation Cancelled by User (CTRL+C)]".to_string());
                }
            }
            match chunk_result {
                Ok(responses) => {
                    for resp in responses {
                        if let Some(ref think) = resp.thinking {
                            if !think.is_empty() {
                                if !thinking_active {
                                    thinking_active = true;
                                    full_text.push_str("<think>");
                                    if let Ok(mut target) = stream_target.lock() {
                                        target.push_str("<think>");
                                    }
                                }
                                full_text.push_str(think);
                                if let Ok(mut target) = stream_target.lock() {
                                    target.push_str(think);
                                }
                            }
                        }

                        if !resp.response.is_empty() {
                            if thinking_active {
                                thinking_active = false;
                                full_text.push_str("</think>\n");
                                if let Ok(mut target) = stream_target.lock() {
                                    target.push_str("</think>\n");
                                }
                            }
                            full_text.push_str(&resp.response);
                            if let Ok(mut target) = stream_target.lock() {
                                target.push_str(&resp.response);
                            }
                        }

                        if resp.done {
                            if thinking_active {
                                full_text.push_str("</think>\n");
                                if let Ok(mut target) = stream_target.lock() {
                                    target.push_str("</think>\n");
                                }
                            }
                            return Ok(full_text);
                        }
                    }
                }
                Err(e) => {
                    return Err(format!("[Ollama Stream Error] {}", e));
                }
            }
        }

        if full_text.is_empty() {
            Err("[Ollama Error] Stream completed with no tokens.".to_string())
        } else {
            Ok(full_text)
        }
    }
}

// ---------------------------------------------------------------------------
// Burn / WGPU (demo tensor path) — only when `gpu` feature is enabled
// ---------------------------------------------------------------------------

#[cfg(feature = "gpu")]
#[derive(Clone)]
pub struct BurnWgpuBackend {
    device: WgpuDevice,
    pub model_name: String,
}

#[cfg(feature = "gpu")]
impl BurnWgpuBackend {
    pub fn new() -> Self {
        Self {
            device: WgpuDevice::default(),
            model_name: "Native WGPU Hardware Engine".to_string(),
        }
    }

    pub fn with_model(model_name: String) -> Self {
        Self {
            device: WgpuDevice::default(),
            model_name,
        }
    }

    pub async fn generate(&self, prompt: &str) -> Result<String, String> {
        let tensor1 = Tensor::<Wgpu, 1>::from_data([1.0, 2.0, 3.0], &self.device);
        let tensor2 = Tensor::<Wgpu, 1>::from_data([4.0, 5.0, 6.0], &self.device);
        let _ = tensor1.add(tensor2);

        if prompt.contains("You just started up") {
            let cwd = std::env::current_dir()
                .unwrap_or_default()
                .display()
                .to_string();
            Ok(format!(
                "<think>Validating Vulkan/WGPU GPU hardware pipeline.</think>Hello! I am Hercules agent running on native Burn/WGPU hardware engine. Ready to assist with your project in {}.",
                cwd
            ))
        } else {
            Ok(format!(
                "<think>Analyzing prompt context.</think>[{}] Executed on Vulkan/WGPU GPU hardware pipeline.\n\n(Tip: select **llama.cpp lib** in Settings for real GGUF inference.)",
                self.model_name
            ))
        }
    }
}

// Silence unused import warning for HttpInferenceClient if only used via type alias paths
#[allow(dead_code)]
fn _http_type(_: HttpInferenceClient) {}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "gpu")]
    #[tokio::test]
    async fn test_burn_wgpu_backend() {
        let backend = BurnWgpuBackend::new();
        let response = backend.generate("Hello").await.unwrap();
        assert!(response.contains("Native WGPU Hardware Engine") || response.contains("WGPU"));
    }
}
