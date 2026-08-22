//! High-level safe wrapper around `libllama.so` (in-process llama.cpp).
//!
//! Replaces the pure-Rust `llama.rs` inference path. Loads the model once,
//! caches it globally, and streams tokens into the shared UI buffer.

use crate::llama::ffi::{LlamaContext, LlamaModel, LlamaVocab, get_lib};
use std::ffi::CString;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

// ---------------------------------------------------------------------------
// LlamaCppLib — in-process engine
// ---------------------------------------------------------------------------

pub struct LlamaCppLib {
    pub model_path: PathBuf,
    model: *mut LlamaModel,
    ctx: *mut LlamaContext,
    vocab: *const LlamaVocab,
    n_ctx: u32,
    /// Max tokens per llama_decode call (must match context params).
    n_batch: u32,
}

// SAFETY: we only access model/ctx/vocab through a single Mutex at a time.
unsafe impl Send for LlamaCppLib {}
unsafe impl Sync for LlamaCppLib {}

impl LlamaCppLib {
    /// Load the model into process memory via libllama.so.
    pub fn new(model_path: PathBuf) -> Result<Self, String> {
        let lib = get_lib()?;

        // Initialise backend (idempotent)
        unsafe { (lib.backend_init)() };

        // Silence all libllama/ggml logs — they write directly to stderr and
        // trash the ratatui alternate screen with tensor load spam.
        unsafe extern "C" fn noop_log(
            _level: i32,
            _text: *const std::os::raw::c_char,
            _ud: *mut std::ffi::c_void,
        ) {
        }
        unsafe { (lib.log_set)(Some(noop_log), std::ptr::null_mut()) };

        let path_cstr = CString::new(model_path.to_str().unwrap_or(""))
            .map_err(|e| format!("Invalid model path: {}", e))?;

        // Model params — start from C defaults, then override only known-safe fields.
        let mut mparams = unsafe { (lib.model_default_params)() };
        mparams.n_gpu_layers = 0; // CPU-only safe default

        let model = unsafe { (lib.model_load_from_file)(path_cstr.as_ptr(), mparams) };
        if model.is_null() {
            return Err(format!(
                "[llama.cpp lib] Failed to load model: {}",
                model_path.display()
            ));
        }

        // Context params — must match installed libllama struct layout exactly.
        let mut cparams = unsafe { (lib.context_default_params)() };
        let n_ctx_train = unsafe { (lib.n_ctx_train)(model) };
        // Use min(4096, train_ctx) for sensible RAM usage
        cparams.n_ctx = (n_ctx_train.max(0) as u32).min(4096).max(512);
        // Prefill is chunked to this size — never pass more tokens than n_batch to decode.
        cparams.n_batch = 512;
        cparams.n_ubatch = 512;
        cparams.n_threads = num_cpus();
        cparams.n_threads_batch = num_cpus();
        // flash_attn_type: 0 = disabled (enum, not bool)
        cparams.flash_attn_type = 0;
        cparams.offload_kqv = false;
        cparams.no_perf = true;

        let ctx = unsafe { (lib.init_from_model)(model, cparams) };
        if ctx.is_null() {
            unsafe { (lib.model_free)(model) };
            return Err("[llama.cpp lib] Failed to create context".to_string());
        }

        let n_ctx = unsafe { (lib.n_ctx)(ctx) };
        let vocab = unsafe { (lib.model_get_vocab)(model) };
        if n_ctx == 0 {
            unsafe {
                (lib.context_free)(ctx);
                (lib.model_free)(model);
            }
            return Err("[llama.cpp lib] n_ctx=0 (context params layout mismatch?)".into());
        }

        Ok(Self {
            model_path,
            model,
            ctx,
            vocab,
            n_ctx,
            n_batch: 512,
        })
    }

    /// Tokenize text into a Vec of token ids.
    fn tokenize(&self, text: &str, add_special: bool) -> Result<Vec<i32>, String> {
        let lib = get_lib()?;
        let ctext = CString::new(text).map_err(|e| e.to_string())?;
        let max_tokens = (text.len() + 64) as i32;
        let mut tokens = vec![0i32; max_tokens as usize];
        let n = unsafe {
            (lib.tokenize)(
                self.vocab,
                ctext.as_ptr(),
                ctext.as_bytes().len() as i32,
                tokens.as_mut_ptr(),
                max_tokens,
                add_special,
                true,
            )
        };
        if n < 0 {
            // Retry with a larger buffer
            let needed = (-n) as usize + 8;
            tokens = vec![0i32; needed];
            let n2 = unsafe {
                (lib.tokenize)(
                    self.vocab,
                    ctext.as_ptr(),
                    ctext.as_bytes().len() as i32,
                    tokens.as_mut_ptr(),
                    needed as i32,
                    add_special,
                    true,
                )
            };
            if n2 < 0 {
                return Err(format!("[llama.cpp lib] tokenize failed: n={}", n2));
            }
            tokens.truncate(n2 as usize);
        } else {
            tokens.truncate(n as usize);
        }
        Ok(tokens)
    }

    /// Decode a single token id into its UTF-8 string piece.
    fn token_to_piece(&self, token: i32) -> String {
        let lib = match get_lib() {
            Ok(l) => l,
            Err(_) => return String::new(),
        };
        let mut buf = vec![0u8; 64];
        let n = unsafe {
            (lib.token_to_piece)(self.vocab, token, buf.as_mut_ptr() as *mut std::ffi::c_char, 64, 0, false)
        };
        if n <= 0 {
            return String::new();
        }
        buf.truncate(n as usize);
        String::from_utf8_lossy(&buf).into_owned()
    }

    /// Build a ChatML-format prompt.
    fn build_prompt(system: &str, user: &str) -> String {
        format!(
            "<|im_start|>system\n{}<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
            system, user
        )
    }

    /// Full generation with optional streaming.
    pub fn generate_stream(
        &self,
        user_prompt: &str,
        stream_target: Arc<Mutex<String>>,
        is_generating: Arc<Mutex<bool>>,
    ) -> Result<String, String> {
        // One generation at a time on the shared model/context.
        let _gen_guard = GEN_LOCK
            .lock()
            .map_err(|e| format!("[llama.cpp lib] gen lock: {e}"))?;

        let lib = get_lib()?;

        // Fresh KV each turn (warm engine reuses one context).
        // clear(data=false) is enough for a new prompt; data=true has crashed some builds.
        if let (Some(get_mem), Some(clear)) = (lib.get_memory, lib.memory_clear) {
            unsafe {
                let mem = get_mem(self.ctx);
                if !mem.is_null() {
                    clear(mem, false);
                }
            }
        }

        let system = crate::agent::AgentEngine::system_prompt_compact_for_cwd();
        // After tools already ran, the host sends "Tool results are above" — do NOT
        // re-apply tool-force nudge (that re-emits the same <read> and clears the answer).
        let already_has_tool_result = user_prompt.contains("Tool results are above")
            || user_prompt.contains("Do NOT re-call")
            || user_prompt.contains("Result:\n");
        let user = if already_has_tool_result {
            user_prompt.to_string()
        } else {
            crate::agent::AgentEngine::with_tool_nudge(user_prompt)
        };
        let prompt = Self::build_prompt(&system, &user);

        // Tokenize
        let add_bos = unsafe { (lib.vocab_get_add_bos)(self.vocab) };
        let mut tokens = self.tokenize(&prompt, add_bos)?;

        let max_ctx = self.n_ctx as usize;
        if tokens.len() >= max_ctx.saturating_sub(32) {
            // Trim prompt tokens from the front, keep BOS + last tokens
            let keep = max_ctx.saturating_sub(64);
            let skip = tokens.len() - keep;
            tokens.drain(1..=skip);
        }

        let n_predict = crate::settings::get_settings().power_mode.max_tokens() as usize;
        // Silence ggml/llama abort spam on the TUI alternate screen for this call.
        let _stderr_guard = StderrSilence::enter();

        // Prefill in chunks ≤ n_batch.
        // A single decode() with tokens.len() > n_batch triggers ggml abort
        // (looks like "Rust code" stack dump in the terminal after ls + re-prompt).
        // NOTE: llama_batch_get_one() does NOT allocate — do NOT llama_batch_free().
        let n_batch = self.n_batch.max(1) as usize;
        {
            let mut offset = 0usize;
            while offset < tokens.len() {
                if let Ok(is_gen) = is_generating.lock() {
                    if !*is_gen {
                        return Ok(String::new());
                    }
                }
                let end = (offset + n_batch).min(tokens.len());
                let chunk = &mut tokens[offset..end];
                let batch = unsafe { (lib.batch_get_one)(chunk.as_mut_ptr(), chunk.len() as i32) };
                let ret = unsafe { (lib.decode)(self.ctx, batch) };
                if ret != 0 {
                    return Err(format!(
                        "[llama.cpp lib] decode (prefill {}..{}) failed: {}",
                        offset, end, ret
                    ));
                }
                offset = end;
            }
        }

        // Build sampler chain
        let sampler_params = unsafe { (lib.sampler_chain_default_params)() };
        let chain = unsafe { (lib.sampler_chain_init)(sampler_params) };
        if chain.is_null() {
            return Err("[llama.cpp lib] sampler_chain_init returned null".into());
        }
        unsafe {
            (lib.sampler_chain_add)(chain, (lib.sampler_init_top_p)(0.9, 1));
            (lib.sampler_chain_add)(chain, (lib.sampler_init_temp)(0.7));
            (lib.sampler_chain_add)(chain, (lib.sampler_init_dist)(0));
        }

        // Decode loop
        let mut full_text = String::new();
        let mut n_generated = 0usize;

        // String-based stop sequences — same set as the HTTP backend.
        // The lib-mode loop only gets token_is_eog for native EOS; models like
        // DeepSeek Coder emit <|im_end|> as a *regular text token* and then
        // hallucinate the next user turn. We must stop on these strings too.
        const STOP_SEQS: &[&str] = &[
            "<|im_end|>",
            "<|im_start|>",
            "<|endoftext|>",
            "</s>",
            "\nYou:",
            "\nUser:",
            "\n### Instruction",
            "\nCRITICAL —",
            "\nCRITICAL -",
        ];

        loop {
            // Check cancellation
            if let Ok(is_gen) = is_generating.lock() {
                if !*is_gen {
                    break;
                }
            }

            let token = unsafe { (lib.sampler_sample)(chain, self.ctx, -1) };

            // Check EOS
            let is_eog = unsafe { (lib.token_is_eog)(self.vocab, token) };
            if is_eog || n_generated >= n_predict {
                break;
            }

            // Decode token to text
            let piece = self.token_to_piece(token);
            if !piece.is_empty() {
                full_text.push_str(&piece);

                // Check string-based stop sequences.
                // Only need to look at the tail (max stop-seq length ≈ 20 chars).
                // Use char_indices to avoid slicing inside a multi-byte UTF-8 character.
                let tail_start = full_text
                    .char_indices()
                    .rev()
                    .nth(31)
                    .map(|(i, _)| i)
                    .unwrap_or(0);
                let tail = &full_text[tail_start..];
                if let Some(stop) = STOP_SEQS.iter().find(|&&s| tail.contains(s)) {
                    // Strip the stop sequence and everything after it from full_text
                    if let Some(idx) = full_text.rfind(stop) {
                        full_text.truncate(idx);
                    }
                    // Update the stream buffer too
                    if let Ok(mut t) = stream_target.lock() {
                        *t = full_text.clone();
                    }
                    break;
                }

                if let Ok(mut t) = stream_target.lock() {
                    t.push_str(&piece);
                }
            }

            n_generated += 1;

            // Feed token back for next iteration (get_one batch — do not free)
            let mut tok = token;
            let batch = unsafe { (lib.batch_get_one)(&mut tok, 1) };
            let ret = unsafe { (lib.decode)(self.ctx, batch) };
            if ret != 0 {
                break;
            }
        }

        unsafe { (lib.sampler_free)(chain) };

        let cancelled = is_generating.lock().map(|g| !*g).unwrap_or(false);

        if full_text.is_empty() {
            if cancelled {
                // Ctrl+C before first token — not a hard failure (avoids noisy aborts).
                Ok(String::new())
            } else {
                Err("[llama.cpp lib] No tokens generated".to_string())
            }
        } else {
            Ok(full_text)
        }
    }

    /// Blocking generate (no streaming).
    pub fn generate(&self, user_prompt: &str) -> Result<String, String> {
        let target = Arc::new(Mutex::new(String::new()));
        let flag = Arc::new(Mutex::new(true));
        self.generate_stream(user_prompt, target, flag)
    }
}

impl Drop for LlamaCppLib {
    fn drop(&mut self) {
        // NOTE: Do NOT acquire GEN_LOCK here. Drop is only called when the last
        // Arc<LlamaCppLib> is released, which means generate_stream has already
        // finished and released its Arc clone. Acquiring GEN_LOCK here while
        // shutdown_warm_lib_engine holds it causes a deadlock (Mutex is not reentrant).
        if let Ok(lib) = get_lib() {
            unsafe {
                if !self.ctx.is_null() {
                    let ctx = self.ctx;
                    self.ctx = std::ptr::null_mut();
                    (lib.context_free)(ctx);
                }
                if !self.model.is_null() {
                    let model = self.model;
                    self.model = std::ptr::null_mut();
                    (lib.model_free)(model);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Warm engine cache (analogous to ensure_warm_rs_engine)
// ---------------------------------------------------------------------------

struct WarmEntry {
    path: PathBuf,
    engine: Arc<LlamaCppLib>,
}

static WARM_ENGINE: Mutex<Option<WarmEntry>> = Mutex::new(None);
/// Serialize all in-process generates (one ctx; concurrent decode → SIGABRT).
static GEN_LOCK: Mutex<()> = Mutex::new(());

/// Return a cached in-process engine, loading the model only when path changes.
pub fn ensure_warm_lib_engine(path: &Path) -> Result<Arc<LlamaCppLib>, String> {
    let mut guard = WARM_ENGINE
        .lock()
        .map_err(|e| format!("[llama.cpp lib] cache lock: {}", e))?;

    if let Some(ref e) = *guard {
        if e.path == path {
            return Ok(e.engine.clone());
        }
    }

    // Explicitly drop the old engine to free RAM/VRAM before allocating the new one.
    *guard = None;

    let engine = Arc::new(LlamaCppLib::new(path.to_path_buf())?);
    *guard = Some(WarmEntry {
        path: path.to_path_buf(),
        engine: engine.clone(),
    });
    Ok(engine)
}

/// Unload the cached model from memory.
pub fn shutdown_warm_lib_engine() {
    // Take the engine out while holding GEN_LOCK so no new generation starts.
    // Drop it AFTER releasing GEN_LOCK — otherwise Drop tries to re-acquire
    // GEN_LOCK on the same thread → deadlock (Mutex is not reentrant).
    let engine_to_drop = {
        let _guard = GEN_LOCK.lock();
        WARM_ENGINE.lock().ok().and_then(|mut g| g.take())
        // GEN_LOCK released here before engine_to_drop is dropped
    };
    drop(engine_to_drop); // Drop (llama_context_free / llama_model_free) runs here, outside GEN_LOCK
}

/// Retrieve the currently active cached engine if loaded.
pub fn get_warm_lib_engine() -> Option<Arc<LlamaCppLib>> {
    let guard = WARM_ENGINE.lock().ok()?;
    guard.as_ref().map(|e| e.engine.clone())
}

// ---------------------------------------------------------------------------
// LlamaCppLibRuntime — drop-in replacement for LlamaRsRuntime
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct LlamaCppLibRuntime {
    pub model_path: Option<PathBuf>,
    pub endpoint: String,
    pub model_name: String,
}

impl Default for LlamaCppLibRuntime {
    fn default() -> Self {
        Self {
            model_path: None,
            endpoint: "http://localhost:8080".into(),
            model_name: "llama.cpp-lib".into(),
        }
    }
}

impl LlamaCppLibRuntime {
    pub fn with_gguf(path: impl Into<PathBuf>) -> Self {
        Self {
            model_path: Some(path.into()),
            endpoint: String::new(),
            model_name: "llama.cpp-lib-local".into(),
        }
    }

    pub fn with_gguf_name(path: impl Into<PathBuf>, model_name: impl Into<String>) -> Self {
        Self {
            model_path: Some(path.into()),
            endpoint: String::new(),
            model_name: model_name.into(),
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
        if let Some(ref path) = self.model_path {
            let path = path.clone();
            let user_prompt = user_prompt.to_string();
            let stream_target2 = stream_target.clone();
            let is_generating2 = is_generating.clone();

            let result = tokio::task::spawn_blocking(move || {
                let engine = ensure_warm_lib_engine(&path)?;
                engine.generate_stream(&user_prompt, stream_target2, is_generating2)
            })
            .await
            .map_err(|e| format!("[llama.cpp lib] task join: {e}"))?;

            return result;
        }

        // HTTP fallback
        if self.endpoint.is_empty() {
            return Err("[llama.cpp lib] No model path and no HTTP endpoint configured".into());
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

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn num_cpus() -> i32 {
    std::thread::available_parallelism()
        .map(|n| n.get() as i32)
        .unwrap_or(2)
        .min(8)
}

/// Redirect stderr to /dev/null for the duration of llama.cpp calls so
/// ggml_abort backtraces do not paint over the ratatui alternate screen.
struct StderrSilence {
    saved: Option<i32>,
}

impl StderrSilence {
    fn enter() -> Self {
        #[cfg(unix)]
        {
            use std::os::fd::AsRawFd;
            // SAFETY: dup/dup2 on stderr for this process only.
            let saved = unsafe { libc::dup(2) };
            if saved < 0 {
                return Self { saved: None };
            }
            if let Ok(devnull) = std::fs::OpenOptions::new().write(true).open("/dev/null") {
                unsafe {
                    libc::dup2(devnull.as_raw_fd(), 2);
                }
                return Self { saved: Some(saved) };
            }
            unsafe {
                libc::close(saved);
            }
            Self { saved: None }
        }
        #[cfg(not(unix))]
        {
            Self { saved: None }
        }
    }
}

impl Drop for StderrSilence {
    fn drop(&mut self) {
        #[cfg(unix)]
        if let Some(fd) = self.saved.take() {
            unsafe {
                libc::dup2(fd, 2);
                libc::close(fd);
            }
        }
    }
}
