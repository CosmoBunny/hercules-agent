//! Autoregressive generation loop (pure-Rust llama.rs).
//!
//! Pipeline:
//! 1. Format chat prompt (system + user)
//! 2. Tokenize
//! 3. Prefill / decode each token through Transformer + KV cache
//! 4. Sample next token
//! 5. Detokenize streamed pieces
//!
//! GGUF is kept **warm in-process** (no llama.cpp / llama-server). First load
//! reads weights once; later prompts reuse the same engine.

use crate::llama::compute::{build_default_backend, ComputeBackend, ComputePrefs};
use crate::llama::gguf::GgufFile;
use crate::llama::model::{forward_token, KvCache, LlamaModel};
use crate::llama::sample::{apply_repeat_penalty, sample_token, Rng64, SamplerParams};
use crate::llama::tokenizer::{format_chat_prompt, Tokenizer};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

#[derive(Debug, Clone)]
pub struct GenerateConfig {
    pub n_predict: usize,
    pub sampler: SamplerParams,
}

impl Default for GenerateConfig {
    fn default() -> Self {
        Self {
            n_predict: 160,
            sampler: SamplerParams::default(),
        }
    }
}

/// Pure-Rust GGUF engine (llama.rs) — **no C/FFI**.
pub struct LlamaRsEngine {
    pub model_path: PathBuf,
    pub gguf: GgufFile,
    pub model: LlamaModel,
    pub tokenizer: Tokenizer,
    pub config: GenerateConfig,
    /// Pluggable pure-Rust compute (scalar / future SIMD / custom embedded).
    pub compute: Arc<dyn ComputeBackend>,
}

impl LlamaRsEngine {
    pub fn load(path: impl AsRef<Path>) -> Result<Self, String> {
        Self::load_with_backend(path, build_default_backend(&ComputePrefs::from_settings()))
    }

    /// Load GGUF with a custom [`ComputeBackend`] (e.g. embedded / NPU stub).
    pub fn load_with_backend(
        path: impl AsRef<Path>,
        compute: Box<dyn ComputeBackend>,
    ) -> Result<Self, String> {
        let path = path.as_ref().to_path_buf();
        let gguf = GgufFile::open(&path).map_err(|e| format!("Failed to open GGUF: {}", e))?;
        let tokenizer = Tokenizer::from_gguf(&gguf)?;
        let model = LlamaModel::load(&gguf)?;
        let compute: Arc<dyn ComputeBackend> = Arc::from(compute);
        // No eprintln here — it corrupts the ratatui alternate screen (input overlap).
        Ok(Self {
            model_path: path,
            gguf,
            model,
            tokenizer,
            config: GenerateConfig {
                n_predict: crate::settings::get_settings()
                    .power_mode
                    .pure_rust_n_predict(),
                ..GenerateConfig::default()
            },
            compute,
        })
    }

    pub fn summary(&self) -> String {
        format!(
            "{} | vocab={} | n_layer={} | n_embd={} | pure-rust compute={} | {}",
            self.gguf.summary(),
            self.tokenizer.vocab_size(),
            self.model.hparams.n_layer,
            self.model.hparams.n_embd,
            self.compute.name(),
            self.model.memory_summary()
        )
    }

    pub fn generate(&self, system: &str, user: &str) -> Result<String, String> {
        self.generate_stream(system, user, None, None, None)
    }

    /// Stream tokens into optional shared UI buffers.
    /// `n_predict_override` comes from power mode when set.
    pub fn generate_stream(
        &self,
        system: &str,
        user: &str,
        stream_target: Option<Arc<Mutex<String>>>,
        is_generating: Option<Arc<Mutex<bool>>>,
        n_predict_override: Option<usize>,
    ) -> Result<String, String> {
        let prompt = format_chat_prompt(Some(&self.gguf), system, user);
        let tokens = self.tokenizer.encode(&prompt, true);
        if tokens.is_empty() {
            return Err("Tokenizer produced empty token list".into());
        }
        if tokens.len() > self.model.hparams.n_ctx.saturating_sub(32) {
            return Err(format!(
                "[llama.rs] Prompt is {} tokens but ctx cap is {}. \
                 Shorten chat history (pure-Rust ctx is capped for RAM).",
                tokens.len(),
                self.model.hparams.n_ctx
            ));
        }

        let n_predict = n_predict_override
            .unwrap_or(self.config.n_predict)
            .max(1);

        let h = &self.model.hparams;
        let mut cache = KvCache::new(h.n_layer, h.n_head_kv, h.head_dim(), h.n_ctx);
        let mut rng = Rng64::new(self.config.sampler.seed);
        let mut all_tokens = tokens.clone();
        let mut full_text = String::new();

        let mut logits = Vec::new();
        for (pos, &tok) in tokens.iter().enumerate() {
            if let Some(ref flag) = is_generating {
                if let Ok(g) = flag.lock() {
                    if !*g {
                        return Err("[Generation Cancelled by User (CTRL+C)]".into());
                    }
                }
            }
            logits = forward_token(&self.model, &mut cache, tok, pos, self.compute.as_ref())?;
        }

        for _ in 0..n_predict {
            if let Some(ref flag) = is_generating {
                if let Ok(g) = flag.lock() {
                    if !*g {
                        return Err("[Generation Cancelled by User (CTRL+C)]".into());
                    }
                }
            }

            apply_repeat_penalty(
                &mut logits,
                &all_tokens,
                self.config.sampler.repeat_penalty,
            );
            let next = sample_token(&logits, &self.config.sampler, &mut rng);
            if self.tokenizer.is_eos(next) {
                break;
            }
            all_tokens.push(next);

            let piece = self.tokenizer.decode(&[next]);
            full_text.push_str(&piece);
            if let Some(ref target) = stream_target {
                if let Ok(mut t) = target.lock() {
                    t.push_str(&piece);
                }
            }

            let pos = all_tokens.len() - 1;
            if pos >= h.n_ctx {
                break;
            }
            logits =
                forward_token(&self.model, &mut cache, next, pos, self.compute.as_ref())?;
        }

        if full_text.is_empty() {
            Err("[llama.rs] Generation produced no tokens".into())
        } else {
            Ok(full_text)
        }
    }
}

// ---------------------------------------------------------------------------
// Warm in-process engine (custom "server" without llama.cpp)
// ---------------------------------------------------------------------------

struct WarmRsEngine {
    path: PathBuf,
    engine: Arc<LlamaRsEngine>,
}

static WARM_RS: Mutex<Option<WarmRsEngine>> = Mutex::new(None);

/// Ensure pure-Rust engine is loaded for this GGUF (reloads only if path changes).
pub fn ensure_warm_rs_engine(path: &Path) -> Result<Arc<LlamaRsEngine>, String> {
    let path = path
        .canonicalize()
        .unwrap_or_else(|_| path.to_path_buf());
    if !path.is_file() {
        return Err(format!("[llama.rs] Model not found: {}", path.display()));
    }

    let mut guard = WARM_RS
        .lock()
        .map_err(|e| format!("warm engine lock: {e}"))?;

    if let Some(ref warm) = *guard {
        if warm.path == path {
            return Ok(Arc::clone(&warm.engine));
        }
    }

    // Drop previous model before loading another (free RAM)
    *guard = None;
    let engine = Arc::new(LlamaRsEngine::load(&path)?);
    *guard = Some(WarmRsEngine {
        path: path.clone(),
        engine: Arc::clone(&engine),
    });
    Ok(engine)
}

/// Drop the warm pure-Rust engine (free RAM).
pub fn shutdown_warm_rs_engine() {
    if let Ok(mut g) = WARM_RS.lock() {
        *g = None;
    }
}

/// Info for UI: (model path, summary) if loaded.
pub fn warm_rs_info() -> Option<(PathBuf, String)> {
    WARM_RS.lock().ok().and_then(|g| {
        g.as_ref()
            .map(|w| (w.path.clone(), w.engine.summary()))
    })
}

/// Shared runtime state: optional GGUF path or HTTP fallback (remote only).
#[derive(Clone)]
pub struct LlamaRsRuntime {
    /// Path to GGUF when using local pure-Rust inference.
    pub model_path: Option<PathBuf>,
    /// Remote OpenAI-compatible endpoint (only when no GGUF path).
    pub endpoint: String,
    pub model_name: String,
}

impl Default for LlamaRsRuntime {
    fn default() -> Self {
        Self {
            model_path: None,
            endpoint: "http://localhost:8080".into(),
            model_name: "llama.rs".into(),
        }
    }
}

impl LlamaRsRuntime {
    pub fn with_gguf(path: impl Into<PathBuf>) -> Self {
        Self {
            model_path: Some(path.into()),
            endpoint: String::new(),
            model_name: "llama.rs-local".into(),
        }
    }

    pub fn with_endpoint(endpoint: impl Into<String>, model_name: impl Into<String>) -> Self {
        Self {
            model_path: None,
            endpoint: endpoint.into(),
            model_name: model_name.into(),
        }
    }

    pub async fn generate_stream(
        &self,
        user_prompt: &str,
        stream_target: Arc<Mutex<String>>,
        is_generating: Arc<Mutex<bool>>,
    ) -> Result<String, String> {
        // Local GGUF → pure-Rust warm engine (never llama-server / llama.cpp)
        if let Some(ref path) = self.model_path {
            let need_load = warm_rs_info()
                .map(|(p, _)| p != *path)
                .unwrap_or(true);
            if need_load {
                if let Ok(mut t) = stream_target.lock() {
                    t.push_str(
                        "[llama.rs] Loading pure-Rust engine (one-time; no llama.cpp)…\n",
                    );
                }
            }

            let path = path.clone();
            let user_prompt = user_prompt.to_string();
            let stream_target2 = stream_target.clone();
            let is_generating2 = is_generating.clone();
            let n_predict = crate::settings::get_settings()
                .power_mode
                .pure_rust_n_predict();

            let result = tokio::task::spawn_blocking(move || {
                let engine = ensure_warm_rs_engine(&path)?;
                if let Ok(mut t) = stream_target2.lock() {
                    if t.contains("[llama.rs] Loading") {
                        t.clear();
                    }
                }
                let system = crate::agent::AgentEngine::system_prompt_for_cwd();
                // Truncate very long user transcripts so pure-Rust ctx fits
                let user = truncate_user_for_ctx(&user_prompt, 3500);
                // Force-tool nudge on the last user turn for list/read/run
                let user = crate::agent::AgentEngine::with_tool_nudge(&user);
                engine.generate_stream(
                    &system,
                    &user,
                    Some(stream_target2),
                    Some(is_generating2),
                    Some(n_predict),
                )
            })
            .await
            .map_err(|e| format!("[llama.rs] task join: {e}"))?;

            return result;
        }

        // HTTP endpoint only (user-configured remote; not local GGUF)
        if self.endpoint.is_empty() {
            return Err(
                "[llama.rs] No GGUF path and no HTTP endpoint configured".into(),
            );
        }
        let client = crate::llama::http::HttpInferenceClient::new(
            self.endpoint.clone(),
            self.model_name.clone(),
        );
        client
            .generate_stream(user_prompt, stream_target, is_generating)
            .await
    }

    pub async fn generate(&self, user_prompt: &str) -> Result<String, String> {
        let target = Arc::new(Mutex::new(String::new()));
        let flag = Arc::new(Mutex::new(true));
        self.generate_stream(user_prompt, target, flag).await
    }
}

/// Keep the tail of a long chat so tool results + latest user stay in window.
fn truncate_user_for_ctx(s: &str, max_chars: usize) -> String {
    let s = s.trim();
    if s.chars().count() <= max_chars {
        return s.to_string();
    }
    let tail: String = s
        .chars()
        .rev()
        .take(max_chars)
        .collect::<String>()
        .chars()
        .rev()
        .collect();
    format!("…[earlier context trimmed]…\n{}", tail)
}
