#[cfg(feature = "gpu")]
use burn::backend::Wgpu;
#[cfg(feature = "gpu")]
use burn::backend::wgpu::WgpuDevice;
#[cfg(feature = "gpu")]
use burn::tensor::Tensor;
use ollama_rs::{
    Ollama, generation::completion::request::GenerationRequest, generation::images::Image,
};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use crate::llama::{HttpInferenceClient, LlamaCppLibRuntime};

#[derive(Clone)]
pub enum AgentBackend {
    #[cfg(feature = "gpu")]
    BurnWgpu(BurnWgpuBackend),
    /// In-process static libllama engine (C FFI / static — no subprocess).
    LlamaCppLib(LlamaCppLibBackend),
    Ollama(OllamaBackend),
    /// Isolated Python Transformers worker (SafeTensors/PyTorch).
    Transformers(crate::model::transformers::TransformersBackend),
}

impl AgentBackend {
    pub async fn generate(&self, prompt: &str) -> Result<String, String> {
        match self {
            #[cfg(feature = "gpu")]
            Self::BurnWgpu(backend) => backend.generate(prompt).await,
            Self::LlamaCppLib(backend) => backend.generate(prompt).await,
            Self::Ollama(backend) => backend.generate(prompt).await,
            Self::Transformers(backend) => backend.generate(prompt).await,
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
            Self::Ollama(backend) => {
                backend
                    .generate_stream(prompt, Vec::new(), stream_target, is_generating)
                    .await
            }
            Self::Transformers(backend) => {
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
            Self::Transformers(b) => Some(b.model_dir.clone()),
            _ => None,
        }
    }

    pub fn name(&self) -> String {
        match self {
            #[cfg(feature = "gpu")]
            Self::BurnWgpu(b) => format!("Burn/WGPU ({})", b.model_name),
            Self::LlamaCppLib(b) => b.name(),
            Self::Ollama(b) => format!("Ollama ({})", b.model),
            Self::Transformers(b) => b.name(),
        }
    }

    pub fn with_model(&self, model_name: &str, manager: &crate::manager::ModelManager) -> Self {
        let trimmed = model_name.trim();
        if trimmed.is_empty() {
            return self.clone();
        }
        match self {
            Self::Ollama(_) => {
                let clean_name = trimmed.trim_start_matches("Ollama:").trim();
                Self::Ollama(OllamaBackend::new(clean_name.to_string()))
            }
            Self::LlamaCppLib(_) => {
                let entries = manager.list_installed_entries();
                if let Some(entry) = entries.iter().find(|e| {
                    e.name.eq_ignore_ascii_case(trimmed)
                        || e.filename.eq_ignore_ascii_case(trimmed)
                        || e.path.ends_with(trimmed)
                }) {
                    Self::LlamaCppLib(LlamaCppLibBackend::gguf_with_name(
                        std::path::PathBuf::from(&entry.path),
                        trimmed,
                    ))
                } else if std::path::Path::new(trimmed).exists() {
                    Self::LlamaCppLib(LlamaCppLibBackend::gguf_with_name(
                        std::path::PathBuf::from(trimmed),
                        trimmed,
                    ))
                } else {
                    Self::LlamaCppLib(LlamaCppLibBackend::http(
                        "http://localhost:8080".into(),
                        trimmed.into(),
                    ))
                }
            }
            #[cfg(feature = "gpu")]
            Self::BurnWgpu(_) => self.clone(),
            Self::Transformers(_) => {
                // Local model directory only in Phase 4 (no downloading here).
                let dir = std::path::PathBuf::from(trimmed);
                if dir.is_dir() {
                    Self::Transformers(crate::model::transformers::TransformersBackend::new(dir))
                } else {
                    // Not a local dir: keep current backend instead of
                    // inventing a model that does not exist.
                    self.clone()
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// llama.cpp in-process (static link / libllama via C FFI)
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

    pub fn gguf_with_name(path: impl Into<PathBuf>, name: impl Into<String>) -> Self {
        Self {
            runtime: LlamaCppLibRuntime::with_gguf_name(path, name),
        }
    }

    pub fn name(&self) -> String {
        if !self.runtime.model_name.is_empty()
            && self.runtime.model_name != "llama.cpp-lib-local"
            && self.runtime.model_name != "llama.cpp-lib"
        {
            format!("llama.cpp ({})", self.runtime.model_name)
        } else if let Some(ref p) = self.runtime.model_path {
            format!(
                "llama.cpp ({})",
                p.file_name()
                    .map(|s| s.to_string_lossy().to_string())
                    .unwrap_or_else(|| p.display().to_string())
            )
        } else {
            format!("llama.cpp HTTP ({})", self.runtime.endpoint)
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

    /// Get the actual model context limit (n_ctx) if using in-process llama.cpp
    pub fn actual_context_limit(&self) -> Option<usize> {
        crate::llama::libinfer::get_warm_lib_engine().map(|e| e.context_limit())
    }
}

// Backward compat alias
pub type LlamaRsBackend = LlamaCppLibBackend;
pub type LlamaCppBackend = LlamaCppLibBackend;
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
        images: Vec<Image>,
        stream_target: Arc<Mutex<String>>,
        is_generating: Arc<Mutex<bool>>,
    ) -> Result<String, String> {
        use futures_util::StreamExt;

        let system = crate::agent::AgentEngine::system_prompt_for_cwd();
        let mut req = GenerationRequest::new(self.model.clone(), prompt.to_string()).system(system);
        if !images.is_empty() {
            req = req.images(images);
        }
        let mut stream = self.ollama.generate_stream(req).await.map_err(|e| {
            format!(
                "[Ollama Error] Stream failed for model '{}': {}. Ensure local Ollama daemon is active.",
                self.model, e
            )
        })?;

        let mut full_text = String::new();
        let mut thinking_active = false;
        let mut token_count = 0usize;
        let gen_start_time = std::time::Instant::now();
        let mut first_token_time: Option<std::time::Instant> = None;

        while let Some(chunk_result) = stream.next().await {
            if let Ok(active_gen) = is_generating.lock() {
                if !*active_gen {
                    if thinking_active {
                        if let Ok(mut target) = stream_target.lock() {
                            target.push_str(" response\n");
                        }
                    }
                    return Err("[Generation Cancelled by User (CTRL+C)]".to_string());
                }
            }
            match chunk_result {
                Ok(responses) => {
                    for resp in responses {
                        if first_token_time.is_none() {
                            let now = std::time::Instant::now();
                            first_token_time = Some(now);
                            let ttft = (now - gen_start_time).as_secs_f64();
                            crate::llama::libinfer::update_inference_telemetry(|t| {
                                t.ttft_secs = ttft;
                            });
                        }
                        if let Some(ref think) = resp.thinking {
                            if !think.is_empty() {
                                token_count += 1;
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
                            token_count += 1;
                            if thinking_active {
                                thinking_active = false;
                                full_text.push_str(" response\n");
                                if let Ok(mut target) = stream_target.lock() {
                                    target.push_str(" response\n");
                                }
                            }
                            full_text.push_str(&resp.response);
                            if let Ok(mut target) = stream_target.lock() {
                                target.push_str(&resp.response);
                            }
                        }

                        crate::llama::libinfer::update_inference_telemetry(|t| {
                            t.generated_tokens = token_count;
                            let elapsed = gen_start_time.elapsed().as_secs_f64();
                            if elapsed > 0.0 {
                                t.decode_tok_per_sec = token_count as f64 / elapsed;
                            }
                        });

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
                    if let Ok(mut target) = stream_target.lock() {
                        target.push_str(" response\n");
                    }
                    return Err(format!("[Ollama Stream Error] {}", e));
                }
            }
        }

        if thinking_active {
            if let Ok(mut target) = stream_target.lock() {
                target.push_str(" response\n");
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
#[allow(unused_imports)]
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
