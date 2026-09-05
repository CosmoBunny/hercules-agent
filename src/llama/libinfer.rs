//! High-level safe wrapper around `libllama.so` (in-process llama.cpp).
//!
//! Replaces the pure-Rust `llama.rs` inference path. Loads the model once,
//! caches it globally, and streams tokens into the shared UI buffer.
//!
//! ## KV-cache warm-start
//!
//! After the first successful generation, the system-prompt portion of the
//! KV state is snapshotted via `llama_state_get_data`.  On every subsequent
//! call the snapshot is restored **instead of clearing and re-prefilling the
//! system prompt**, so only the new user-turn tokens need to be decoded.
//! This saves 0.5–5 seconds per turn on large system prompts.
//!
//! The snapshot is invalidated whenever the system prompt text changes
//! (e.g. different working directory) so it never goes stale.

use crate::llama::ffi::{LlamaContext, LlamaModel, LlamaVocab, get_lib};
use std::ffi::CString;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

// ---------------------------------------------------------------------------
// Inference Live & Session Diagnostics
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Default)]
pub struct InferenceTelemetry {
    pub prompt_tokens: usize,
    pub generated_tokens: usize,
    pub prefill_duration_secs: f64,
    pub decode_duration_secs: f64,
    pub ttft_secs: f64,
    pub prefill_tok_per_sec: f64,
    pub decode_tok_per_sec: f64,
    pub session_total_prompt_tokens: usize,
    pub session_total_gen_tokens: usize,
    pub session_total_inference_secs: f64,
}

static LIVE_TELEMETRY: Mutex<Option<InferenceTelemetry>> = Mutex::new(None);

pub fn get_inference_telemetry() -> InferenceTelemetry {
    LIVE_TELEMETRY.lock().unwrap().clone().unwrap_or_default()
}

pub fn update_inference_telemetry<F: FnOnce(&mut InferenceTelemetry)>(f: F) {
    let mut lock = LIVE_TELEMETRY.lock().unwrap();
    if lock.is_none() {
        *lock = Some(InferenceTelemetry::default());
    }
    if let Some(ref mut t) = *lock {
        f(t);
    }
}

// ---------------------------------------------------------------------------
// KV-cache system-prompt snapshot
// ---------------------------------------------------------------------------

/// Stores a serialised llama.cpp KV state taken immediately after the
/// system-prompt tokens have been decoded (but before any user turn).
struct SyspromptSnapshot {
    /// The exact system-prompt text this snapshot was built for.
    system_text: String,
    /// Number of tokens in the system-prompt batch (= KV position after prefill).
    n_sys_tokens: usize,
    /// Token IDs representing the prefilled system-prompt prefix.
    system_tokens: Vec<i32>,
    /// Raw serialised KV-cache bytes from `llama_state_get_data`.
    data: Vec<u8>,
}

// ---------------------------------------------------------------------------
// LlamaCppLib — in-process engine
// ---------------------------------------------------------------------------

pub struct LlamaCppLib {
    pub model_path: PathBuf,
    model: *mut LlamaModel,
    ctx: *mut LlamaContext,
    vocab: *const LlamaVocab,
    mtmd_ctx: Option<*mut crate::llama::ffi::MtmdContext>,
    n_ctx: u32,
    /// Max tokens per llama_decode call (must match context params).
    n_batch: u32,
    /// Cached KV state after system-prompt prefill — avoids re-encoding on every turn.
    sys_snapshot: Mutex<Option<SyspromptSnapshot>>,
}

// SAFETY: we only access model/ctx/vocab through a single Mutex at a time.
unsafe impl Send for LlamaCppLib {}
unsafe impl Sync for LlamaCppLib {}

impl LlamaCppLib {
    /// Get the actual model context limit (n_ctx)
    pub fn context_limit(&self) -> usize {
        self.n_ctx as usize
    }

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

        let _init_silence = StderrSilence::enter();

        // Model params — start from C defaults, then override only known-safe fields.
        let power_mode = crate::settings::get_settings().power_mode;
        let mut mparams = unsafe { (lib.model_default_params)() };
        mparams.n_gpu_layers = power_mode.n_gpu_layers();
        mparams.load_mtp = crate::settings::get_mtp_mode().is_native_mtp();

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
        let user_ctx = crate::settings::context_token_limit() as u32;
        let train_max = if n_ctx_train > 0 {
            n_ctx_train as u32
        } else {
            4096
        };
        cparams.n_ctx = user_ctx.min(train_max).max(512);
        // Prefill is chunked to this size — never pass more tokens than n_batch to decode.
        cparams.n_batch = 512;
        cparams.n_ubatch = 512;
        let th = power_mode.threads() as i32;
        cparams.n_threads = th;
        cparams.n_threads_batch = th;
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

        // ViT / Vision Projector (mmproj) initialization based on settings or auto-discovery
        let mut mtmd_ctx = None;
        let selected_vit = crate::settings::get_selected_vit_model();

        if !selected_vit.eq_ignore_ascii_case("disabled") {
            if let (Some(init_mtmd), Some(def_params)) =
                (lib.mtmd_init_from_file, lib.mtmd_context_params_default)
            {
                let mut mmproj_path = None;

                if !selected_vit.eq_ignore_ascii_case("auto") {
                    // Explicit filename or path selected
                    let explicit_p = PathBuf::from(&selected_vit);
                    if explicit_p.exists() {
                        mmproj_path = Some(explicit_p);
                    } else {
                        // Check inside models_dir and model_dir
                        let in_models = crate::manager::models_dir().join(&selected_vit);
                        let in_local = crate::manager::local_hercules_dir().join(&selected_vit);
                        let in_model_dir = model_path
                            .parent()
                            .unwrap_or(&model_path)
                            .join(&selected_vit);
                        if in_models.exists() {
                            mmproj_path = Some(in_models);
                        } else if in_local.exists() {
                            mmproj_path = Some(in_local);
                        } else if in_model_dir.exists() {
                            mmproj_path = Some(in_model_dir);
                        }
                    }
                }

                // Fallback to auto-detection if "auto" or if explicit path wasn't found
                if mmproj_path.is_none() && selected_vit.eq_ignore_ascii_case("auto") {
                    let model_dir = model_path.parent().unwrap_or(&model_path);
                    let search_dirs = vec![
                        model_dir.to_path_buf(),
                        crate::manager::models_dir(),
                        crate::manager::local_hercules_dir(),
                    ];
                    for dir in search_dirs {
                        if let Ok(entries) = std::fs::read_dir(dir) {
                            for entry in entries.flatten() {
                                let p = entry.path();
                                let name = p
                                    .file_name()
                                    .unwrap_or_default()
                                    .to_string_lossy()
                                    .to_lowercase();
                                if (name.contains("mmproj")
                                    || name.contains("vit")
                                    || name.contains("clip"))
                                    && name.ends_with(".gguf")
                                {
                                    mmproj_path = Some(p);
                                    break;
                                }
                            }
                        }
                        if mmproj_path.is_some() {
                            break;
                        }
                    }
                }

                if let Some(m_path) = mmproj_path {
                    if let Ok(m_cstr) = CString::new(m_path.to_string_lossy().as_bytes()) {
                        let mut m_params = unsafe { def_params() };
                        m_params.use_gpu = power_mode.n_gpu_layers() > 0;
                        m_params.n_threads = th;
                        let m_ctx = unsafe { init_mtmd(m_cstr.as_ptr(), model, m_params) };
                        if !m_ctx.is_null() {
                            mtmd_ctx = Some(m_ctx);
                        }
                    }
                }
            }
        }

        Ok(Self {
            model_path,
            model,
            ctx,
            vocab,
            mtmd_ctx,
            n_ctx,
            n_batch: 512,
            sys_snapshot: Mutex::new(None),
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
            (lib.token_to_piece)(
                self.vocab,
                token,
                buf.as_mut_ptr() as *mut std::ffi::c_char,
                64,
                0,
                false,
            )
        };
        if n <= 0 {
            return String::new();
        }
        buf.truncate(n as usize);
        String::from_utf8_lossy(&buf).into_owned()
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

        let system = crate::agent::AgentEngine::system_prompt_compact_for_cwd();
        let sys_prefix = format!("<|im_start|>system\n{}<|im_end|>\n", system);

        // Parse conversational history into structured ChatML turns (<|im_start|>role...<|im_end|>)
        // so the model sees genuine separate turns instead of flattening prior assistant turns
        // into training text within a single user block.
        let chat_json = crate::llama::http::HttpInferenceClient::chat_messages("", user_prompt);
        let mut formatted_suffix = String::new();
        for msg in chat_json {
            let role = msg.get("role").and_then(|r| r.as_str()).unwrap_or("user");
            let content = msg.get("content").and_then(|c| c.as_str()).unwrap_or("");
            if role == "system" {
                continue;
            }
            formatted_suffix.push_str(&format!("<|im_start|>{role}\n{content}<|im_end|>\n"));
        }

        // Prime the generation turn with assistant role without forcing <think> tag
        formatted_suffix.push_str("<|im_start|>assistant\n");
        let user_suffix = formatted_suffix;

        let add_bos = unsafe { (lib.vocab_get_add_bos)(self.vocab) };
        let max_ctx = self.n_ctx as usize;
        let n_batch = self.n_batch.max(1) as usize;
        let n_predict = crate::settings::get_settings().power_mode.max_tokens() as usize;
        let _stderr_guard = StderrSilence::enter();

        // -------------------------------------------------------------------
        // PREFILL: try to restore system-prompt snapshot, fall back to full prefill
        // -------------------------------------------------------------------
        let gen_start_time = std::time::Instant::now();
        let mut prefill_start_time = std::time::Instant::now();
        let mut total_prompt_tokens = 0usize;

        let mut snap_guard = self.sys_snapshot.lock().unwrap();

        // -------------------------------------------------------------------
        // 1. Multimodal Evaluation Path: If images are attached and mtmd_ctx is active,
        // evaluate the image through the Vision Transformer first.
        // -------------------------------------------------------------------
        let mut attached_images = Vec::new();
        if let Some(m_ctx) = self.mtmd_ctx {
            if let (
                Some(bitmap_from_file),
                Some(eval_chunks),
                Some(chunks_init),
                Some(chunks_free),
                Some(bitmap_free),
                Some(tokenize_mtmd),
            ) = (
                lib.mtmd_helper_bitmap_init_from_file,
                lib.mtmd_helper_eval_chunks,
                lib.mtmd_input_chunks_init,
                lib.mtmd_input_chunks_free,
                lib.mtmd_bitmap_free,
                lib.mtmd_tokenize,
            ) {
                // Extract any attached image file paths from <attachment ... path="..."> tags
                let mut search_idx = 0;
                while let Some(att_pos) = user_prompt[search_idx..].find("path=\"") {
                    let real_start = search_idx + att_pos + 6;
                    if let Some(quote_end) = user_prompt[real_start..].find('"') {
                        let path_str = &user_prompt[real_start..real_start + quote_end];
                        let p = PathBuf::from(path_str);
                        if p.exists() {
                            attached_images.push(p);
                        }
                        search_idx = real_start + quote_end + 1;
                    } else {
                        break;
                    }
                }

                if !attached_images.is_empty() {
                    let mut bitmaps: Vec<*const crate::llama::ffi::MtmdBitmap> = Vec::new();
                    let mut wrappers = Vec::new();
                    for img_p in &attached_images {
                        if let Ok(c_path) = CString::new(img_p.to_string_lossy().as_bytes()) {
                            let wrapper =
                                unsafe { bitmap_from_file(m_ctx, c_path.as_ptr(), false) };
                            if !wrapper.bitmap.is_null() {
                                bitmaps.push(wrapper.bitmap);
                                wrappers.push(wrapper);
                            }
                        }
                    }

                    if !bitmaps.is_empty() {
                        // Clear KV memory before evaluating multimodal chunks
                        if let (Some(get_mem), Some(clear)) = (lib.get_memory, lib.memory_clear) {
                            unsafe {
                                let mem = get_mem(self.ctx);
                                if !mem.is_null() {
                                    clear(mem, false);
                                }
                            }
                        }

                        let default_marker = unsafe {
                            lib.mtmd_default_marker
                                .map(|m| {
                                    let c_str = m();
                                    std::ffi::CStr::from_ptr(c_str)
                                        .to_string_lossy()
                                        .into_owned()
                                })
                                .unwrap_or_else(|| "<__media__>".to_string())
                        };

                        let mut full_multimodal_text =
                            format!("{}\n<|im_start|>user\n", sys_prefix);
                        for _ in 0..bitmaps.len() {
                            full_multimodal_text.push_str(&default_marker);
                            full_multimodal_text.push('\n');
                        }
                        full_multimodal_text.push_str(&user_suffix);

                        if let Ok(c_text) = CString::new(full_multimodal_text) {
                            let input_text = crate::llama::ffi::MtmdInputText {
                                text: c_text.as_ptr(),
                                text_len: c_text.as_bytes().len(),
                                add_special: true,
                                parse_special: true,
                            };
                            let chunks = unsafe { chunks_init() };
                            let tok_res = unsafe {
                                tokenize_mtmd(
                                    m_ctx,
                                    chunks,
                                    &input_text,
                                    bitmaps.as_ptr(),
                                    bitmaps.len(),
                                )
                            };
                            if tok_res == 0 {
                                let mut new_n_past: i32 = 0;
                                let eval_res = unsafe {
                                    eval_chunks(
                                        m_ctx,
                                        self.ctx,
                                        chunks,
                                        0,
                                        0,
                                        n_batch as i32,
                                        true,
                                        &mut new_n_past,
                                    )
                                };
                                unsafe { chunks_free(chunks) };
                                for w in wrappers {
                                    unsafe { bitmap_free(w.bitmap) };
                                }

                                if eval_res == 0 {
                                    total_prompt_tokens = new_n_past.max(1) as usize;
                                    let prefill_dur =
                                        prefill_start_time.elapsed().as_secs_f64().max(0.0001);
                                    let prefill_toks_sec = total_prompt_tokens as f64 / prefill_dur;
                                    update_inference_telemetry(|t| {
                                        t.prompt_tokens = total_prompt_tokens;
                                        t.prefill_duration_secs = prefill_dur;
                                        t.prefill_tok_per_sec = prefill_toks_sec;
                                        t.session_total_prompt_tokens += total_prompt_tokens;
                                    });

                                    let chain_params =
                                        unsafe { (lib.sampler_chain_default_params)() };
                                    let chain = unsafe { (lib.sampler_chain_init)(chain_params) };
                                    if !chain.is_null() {
                                        let mtp_mode = crate::settings::get_mtp_mode();
                                        unsafe {
                                            if mtp_mode.ngram_size() >= 2 {
                                                (lib.sampler_chain_add)(
                                                    chain,
                                                    (lib.sampler_init_greedy)(),
                                                );
                                            } else {
                                                (lib.sampler_chain_add)(
                                                    chain,
                                                    (lib.sampler_init_top_p)(0.9, 1),
                                                );
                                                (lib.sampler_chain_add)(
                                                    chain,
                                                    (lib.sampler_init_temp)(0.7),
                                                );
                                                (lib.sampler_chain_add)(
                                                    chain,
                                                    (lib.sampler_init_dist)(0),
                                                );
                                            }
                                        }
                                        let result = self.sample_loop(
                                            chain,
                                            &lib,
                                            &stream_target,
                                            &is_generating,
                                            n_predict,
                                            Vec::new(),
                                            gen_start_time,
                                        );
                                        unsafe { (lib.sampler_free)(chain) };
                                        return result;
                                    }
                                }
                            } else {
                                unsafe { chunks_free(chunks) };
                                for w in wrappers {
                                    unsafe { bitmap_free(w.bitmap) };
                                }
                            }
                        }
                    }
                }
            }
        }

        let can_restore = snap_guard
            .as_ref()
            .map(|s| s.system_text == system)
            .unwrap_or(false);

        if can_restore {
            // Restore the serialised KV state — skips system-prompt re-encoding.
            let snap = snap_guard.as_ref().unwrap();
            if let (Some(set_data), Some(get_size)) = (lib.state_set_data, lib.state_get_size) {
                let needed = unsafe { get_size(self.ctx) };
                if snap.data.len() == needed {
                    let restored =
                        unsafe { set_data(self.ctx, snap.data.as_ptr(), snap.data.len()) };
                    if restored == snap.data.len() {
                        // State restored — now tokenize only the user-turn suffix and prefill it.
                        let mut user_tokens = self.tokenize(&user_suffix, false)?;
                        // Trim if combined length would overflow context
                        let sys_tokens = snap.n_sys_tokens;
                        let available = max_ctx.saturating_sub(sys_tokens + 32);
                        if user_tokens.len() > available {
                            user_tokens.truncate(available);
                        }
                        total_prompt_tokens = sys_tokens + user_tokens.len();
                        // Re-encode user tokens starting from sys position with explicit absolute positions
                        let mut user_prefill_ok = true;
                        let mut i = 0;
                        let total_user_tokens = user_tokens.len();
                        while i < total_user_tokens {
                            if let Ok(is_gen) = is_generating.lock() {
                                if !*is_gen {
                                    return Ok(String::new());
                                }
                            }
                            let end = (i + n_batch).min(total_user_tokens);
                            let is_last_chunk = end == total_user_tokens;
                            let chunk = &user_tokens[i..end];
                            let chunk_len = chunk.len();

                            let mut batch = unsafe { (lib.batch_init)(chunk_len as i32, 0, 1) };
                            batch.n_tokens = chunk_len as i32;
                            unsafe {
                                for (c_idx, &tok) in chunk.iter().enumerate() {
                                    *batch.token.add(c_idx) = tok;
                                    *batch.pos.add(c_idx) = (sys_tokens + i + c_idx) as i32;
                                    *batch.n_seq_id.add(c_idx) = 1;
                                    *(*batch.seq_id.add(c_idx)).add(0) = 0;
                                    *batch.logits.add(c_idx) =
                                        if is_last_chunk && c_idx == chunk_len - 1 {
                                            1
                                        } else {
                                            0
                                        };
                                }
                            }

                            let ret = unsafe { (lib.decode)(self.ctx, batch) };
                            unsafe { (lib.batch_free)(batch) };
                            if ret != 0 {
                                user_prefill_ok = false;
                                break;
                            }
                            i = end;
                        }
                        if user_prefill_ok {
                            let prefill_dur =
                                prefill_start_time.elapsed().as_secs_f64().max(0.0001);
                            let prefill_toks_sec = total_prompt_tokens as f64 / prefill_dur;
                            update_inference_telemetry(|t| {
                                t.prompt_tokens = total_prompt_tokens;
                                t.prefill_duration_secs = prefill_dur;
                                t.prefill_tok_per_sec = prefill_toks_sec;
                                t.session_total_prompt_tokens += total_prompt_tokens;
                            });

                            // If user prefill succeeded, jump straight to sample loop with full token history
                            let chain_params = unsafe { (lib.sampler_chain_default_params)() };
                            let chain = unsafe { (lib.sampler_chain_init)(chain_params) };
                            if !chain.is_null() {
                                let mtp_mode = crate::settings::get_mtp_mode();
                                unsafe {
                                    if mtp_mode.ngram_size() >= 2 {
                                        (lib.sampler_chain_add)(chain, (lib.sampler_init_greedy)());
                                    } else {
                                        (lib.sampler_chain_add)(
                                            chain,
                                            (lib.sampler_init_top_p)(0.9, 1),
                                        );
                                        (lib.sampler_chain_add)(
                                            chain,
                                            (lib.sampler_init_temp)(0.7),
                                        );
                                        (lib.sampler_chain_add)(chain, (lib.sampler_init_dist)(0));
                                    }
                                }
                                let mut all_tokens = snap.system_tokens.clone();
                                all_tokens.extend_from_slice(&user_tokens);
                                let result = self.sample_loop(
                                    chain,
                                    &lib,
                                    &stream_target,
                                    &is_generating,
                                    n_predict,
                                    all_tokens,
                                    gen_start_time,
                                );
                                unsafe { (lib.sampler_free)(chain) };
                                return result;
                            }
                        }
                    }
                }
            }
            // Snapshot size mismatch, decode failure, or set_data unavailable — invalidate and fall through
            *snap_guard = None;
        }

        // Full prefill path (first call, or snapshot invalid/unavailable).
        prefill_start_time = std::time::Instant::now();
        // Clear KV first.
        if let (Some(get_mem), Some(clear)) = (lib.get_memory, lib.memory_clear) {
            unsafe {
                let mem = get_mem(self.ctx);
                if !mem.is_null() {
                    clear(mem, false);
                }
            }
        }

        // Tokenize system-prompt prefix and user suffix separately so that the token count
        // and boundary match the snapshot exactly with zero tokenizer-merge ambiguity.
        let sys_tokens = self.tokenize(&sys_prefix, add_bos)?;
        let n_sys = sys_tokens.len();
        if n_sys > max_ctx.saturating_sub(64) {
            return Err(format!(
                "[llama.cpp lib] system prompt too large: {} tokens for context {}",
                n_sys, max_ctx
            ));
        }
        let mut user_tokens = self.tokenize(&user_suffix, false)?;

        // Ensure system tokens + user tokens stay safely within context limits by trimming
        // user tokens from the front/tail rather than corrupting the system prefix and n_sys.
        let max_user_tokens = max_ctx.saturating_sub(64).saturating_sub(n_sys);
        if user_tokens.len() > max_user_tokens {
            user_tokens.truncate(max_user_tokens);
        }

        let mut tokens = sys_tokens.clone();
        tokens.extend_from_slice(&user_tokens);
        {
            let mut offset = 0usize;
            let total_tokens = tokens.len();
            while offset < total_tokens {
                if let Ok(is_gen) = is_generating.lock() {
                    if !*is_gen {
                        return Ok(String::new());
                    }
                }
                // If we have not yet reached n_sys, cap the chunk end strictly at n_sys
                // so the KV snapshot contains ONLY the system tokens and no trailing user tokens.
                let max_end = if offset < n_sys {
                    (offset + n_batch).min(n_sys)
                } else {
                    (offset + n_batch).min(total_tokens)
                };
                let end = max_end;
                let is_last_chunk = end == total_tokens;
                let chunk = &tokens[offset..end];
                let chunk_len = chunk.len();

                let mut batch = unsafe { (lib.batch_init)(chunk_len as i32, 0, 1) };
                batch.n_tokens = chunk_len as i32;
                unsafe {
                    for (c_idx, &tok) in chunk.iter().enumerate() {
                        *batch.token.add(c_idx) = tok;
                        *batch.pos.add(c_idx) = (offset + c_idx) as i32;
                        *batch.n_seq_id.add(c_idx) = 1;
                        *(*batch.seq_id.add(c_idx)).add(0) = 0;
                        *batch.logits.add(c_idx) = if is_last_chunk && c_idx == chunk_len - 1 {
                            1
                        } else {
                            0
                        };
                    }
                }

                let ret = unsafe { (lib.decode)(self.ctx, batch) };
                unsafe { (lib.batch_free)(batch) };
                if ret != 0 {
                    return Err(format!(
                        "[llama.cpp lib] decode (prefill {}..{}) failed: {}",
                        offset, end, ret
                    ));
                }
                // Exactly after finishing system-prefix tokens, take the KV snapshot.
                if offset + chunk.len() == n_sys && snap_guard.is_none() {
                    if let (Some(get_size), Some(get_data)) =
                        (lib.state_get_size, lib.state_get_data)
                    {
                        let sz = unsafe { get_size(self.ctx) };
                        if sz > 0 && sz < 512 * 1024 * 1024 {
                            // Cap at 512 MB — guard against absurd values on old builds
                            let mut buf = vec![0u8; sz];
                            let written = unsafe { get_data(self.ctx, buf.as_mut_ptr(), sz) };
                            if written == sz {
                                *snap_guard = Some(SyspromptSnapshot {
                                    system_text: system.clone(),
                                    n_sys_tokens: n_sys,
                                    system_tokens: sys_tokens.clone(),
                                    data: buf,
                                });
                            }
                        }
                    }
                }
                offset = end;
            }
        }
        let prefill_dur = prefill_start_time.elapsed().as_secs_f64().max(0.0001);
        let total_prompt_tokens = tokens.len();
        let prefill_toks_sec = total_prompt_tokens as f64 / prefill_dur;
        update_inference_telemetry(|t| {
            t.prompt_tokens = total_prompt_tokens;
            t.prefill_duration_secs = prefill_dur;
            t.prefill_tok_per_sec = prefill_toks_sec;
            t.session_total_prompt_tokens += total_prompt_tokens;
        });

        drop(snap_guard);

        // Build sampler chain and run the token generation loop.
        let sampler_params = unsafe { (lib.sampler_chain_default_params)() };
        let chain = unsafe { (lib.sampler_chain_init)(sampler_params) };
        if chain.is_null() {
            return Err("[llama.cpp lib] sampler_chain_init returned null".into());
        }
        let mtp_mode = crate::settings::get_mtp_mode();
        unsafe {
            if mtp_mode.ngram_size() >= 2 {
                (lib.sampler_chain_add)(chain, (lib.sampler_init_greedy)());
            } else {
                (lib.sampler_chain_add)(chain, (lib.sampler_init_top_p)(0.9, 1));
                (lib.sampler_chain_add)(chain, (lib.sampler_init_temp)(0.7));
                (lib.sampler_chain_add)(chain, (lib.sampler_init_dist)(0));
            }
        }
        let result = self.sample_loop(
            chain,
            &lib,
            &stream_target,
            &is_generating,
            n_predict,
            tokens,
            gen_start_time,
        );
        unsafe { (lib.sampler_free)(chain) };
        result
    }

    /// Token sampling loop — called after prefill is done.
    /// Supports Prompt Lookup Speculative Decoding when ngram_size >= 2.
    /// `chain` must already be initialised; caller is responsible for freeing it.
    fn sample_loop(
        &self,
        chain: *mut crate::llama::ffi::LlamaSampler,
        lib: &crate::llama::ffi::LlamaLib,
        stream_target: &Arc<Mutex<String>>,
        is_generating: &Arc<Mutex<bool>>,
        n_predict: usize,
        mut all_tokens: Vec<i32>,
        gen_start_time: std::time::Instant,
    ) -> Result<String, String> {
        // String-based stop sequences — same set as the HTTP backend.
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

        let mut full_text = String::new();
        if let Ok(mut t) = stream_target.lock() {
            *t = full_text.clone();
        }
        let mut n_generated = 0usize;
        let ngram_size = crate::settings::get_mtp_mode().ngram_size();
        let mem = lib.get_memory.map(|f| unsafe { f(self.ctx) });
        let decode_start_time = std::time::Instant::now();
        let mut first_token_time: Option<std::time::Instant> = None;

        // Sample the first token from prefill logits
        let mut token = unsafe { (lib.sampler_sample)(chain, self.ctx, -1) };

        loop {
            if let Ok(is_gen) = is_generating.lock() {
                if !*is_gen {
                    break;
                }
            }

            let is_eog = unsafe { (lib.token_is_eog)(self.vocab, token) };
            if is_eog || n_generated >= n_predict {
                break;
            }

            if first_token_time.is_none() {
                let now = std::time::Instant::now();
                first_token_time = Some(now);
                let ttft = (now - gen_start_time).as_secs_f64();
                update_inference_telemetry(|t| {
                    t.ttft_secs = ttft;
                });
            }

            let piece = self.token_to_piece(token);
            if !piece.is_empty() {
                full_text.push_str(&piece);

                let tail_start = full_text
                    .char_indices()
                    .rev()
                    .nth(31)
                    .map(|(i, _)| i)
                    .unwrap_or(0);
                let tail = &full_text[tail_start..];
                if let Some(stop) = STOP_SEQS.iter().find(|&&s| tail.contains(s)) {
                    if let Some(idx) = full_text.rfind(stop) {
                        full_text.truncate(idx);
                    }
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
            all_tokens.push(token);

            // -----------------------------------------------------------------
            // Prompt Lookup Speculative Decoding:
            // Draft candidate tokens from earlier token history matching trailing n-gram.
            // -----------------------------------------------------------------
            let mut drafts: Vec<i32> = Vec::new();
            if ngram_size >= 2 && all_tokens.len() > ngram_size + 1 && mem.is_some() {
                let cur_len = all_tokens.len();
                let pattern = &all_tokens[cur_len - ngram_size..];

                // Search backwards in history (excluding the current match at the tail)
                let max_draft_len = 4.min(n_predict.saturating_sub(n_generated));
                let search_end = cur_len.saturating_sub(ngram_size + 1);
                for j in (0..search_end).rev() {
                    if &all_tokens[j..j + ngram_size] == pattern {
                        let draft_start = j + ngram_size;
                        let available = cur_len.saturating_sub(draft_start);
                        let draft_len = max_draft_len.min(available);
                        if draft_len > 0 {
                            drafts = all_tokens[draft_start..draft_start + draft_len].to_vec();
                        }
                        break;
                    }
                }
            }

            if !drafts.is_empty() {
                let n_draft = drafts.len();
                let total_eval = 1 + n_draft;

                let pos_max = if let (Some(m), Some(pos_max_fn)) = (mem, lib.memory_seq_pos_max) {
                    unsafe { pos_max_fn(m, 0) }
                } else {
                    (all_tokens.len() as i32).saturating_sub(1)
                };

                // Allocate a batch and explicitly populate all metadata fields
                let mut batch = unsafe { (lib.batch_init)(total_eval as i32, 0, 1) };
                batch.n_tokens = total_eval as i32;

                unsafe {
                    *batch.token.add(0) = token;
                    *batch.pos.add(0) = pos_max + 1;
                    *batch.n_seq_id.add(0) = 1;
                    *(*batch.seq_id.add(0)).add(0) = 0;
                    *batch.logits.add(0) = 1;

                    for (d_i, &d_tok) in drafts.iter().enumerate() {
                        let idx = d_i + 1;
                        *batch.token.add(idx) = d_tok;
                        *batch.pos.add(idx) = pos_max + 1 + idx as i32;
                        *batch.n_seq_id.add(idx) = 1;
                        *(*batch.seq_id.add(idx)).add(0) = 0;
                        *batch.logits.add(idx) = 1;
                    }
                }

                let ret = unsafe { (lib.decode)(self.ctx, batch) };
                unsafe { (lib.batch_free)(batch) };
                if ret != 0 {
                    break;
                }

                // Verify draft predictions sequentially:
                // logits at index 0 predict the token following `token` (i.e. draft[0])
                // logits at index k predict the token following `draft[k-1]` (i.e. draft[k])
                let mut accepted_count = 0usize;
                let mut next_token = unsafe { (lib.sampler_sample)(chain, self.ctx, 0) };
                let mut should_stop_generation = false;

                for (i, &draft_tok) in drafts.iter().enumerate() {
                    let is_draft_eog = unsafe { (lib.token_is_eog)(self.vocab, draft_tok) };
                    if is_draft_eog {
                        // Do not accept EOG as speculative token; terminate generation and rollback
                        should_stop_generation = true;
                        break;
                    }

                    if next_token == draft_tok {
                        // Draft accepted!
                        let p = self.token_to_piece(draft_tok);
                        if !p.is_empty() {
                            full_text.push_str(&p);

                            let tail_start = full_text
                                .char_indices()
                                .rev()
                                .nth(31)
                                .map(|(i, _)| i)
                                .unwrap_or(0);
                            let tail = &full_text[tail_start..];
                            if let Some(stop) = STOP_SEQS.iter().find(|&&s| tail.contains(s)) {
                                if let Some(idx) = full_text.rfind(stop) {
                                    full_text.truncate(idx);
                                }
                                if let Ok(mut t) = stream_target.lock() {
                                    *t = full_text.clone();
                                }
                                n_generated += 1;
                                all_tokens.push(draft_tok);
                                accepted_count += 1;
                                should_stop_generation = true;
                                break;
                            }

                            if let Ok(mut t) = stream_target.lock() {
                                t.push_str(&p);
                            }
                        }
                        n_generated += 1;
                        all_tokens.push(draft_tok);
                        accepted_count += 1;

                        if n_generated >= n_predict {
                            should_stop_generation = true;
                            break;
                        }

                        // Sample prediction after this accepted draft token
                        next_token =
                            unsafe { (lib.sampler_sample)(chain, self.ctx, (i + 1) as i32) };
                    } else {
                        // Diverged: `next_token` is the true sampled replacement from the model
                        break;
                    }
                }

                // Rollback unaccepted draft positions in KV memory
                if accepted_count < n_draft {
                    if let (Some(m), Some(rm_fn), Some(pos_max_fn)) =
                        (mem, lib.memory_seq_rm, lib.memory_seq_pos_max)
                    {
                        let cur_pos_max = unsafe { pos_max_fn(m, 0) };
                        let excess = (n_draft - accepted_count) as i32;
                        let keep_pos = cur_pos_max - excess + 1;
                        unsafe {
                            rm_fn(m, 0, keep_pos, -1);
                        }
                    }
                }

                if should_stop_generation {
                    break;
                }

                token = next_token;
            } else {
                // Standard 1-token decode step with explicit continuous KV position
                let pos = if let (Some(m), Some(pos_max_fn)) = (mem, lib.memory_seq_pos_max) {
                    unsafe { pos_max_fn(m, 0) + 1 }
                } else {
                    all_tokens.len() as i32
                };

                let mut batch = unsafe { (lib.batch_init)(1, 0, 1) };
                batch.n_tokens = 1;
                unsafe {
                    *batch.token.add(0) = token;
                    *batch.pos.add(0) = pos;
                    *batch.n_seq_id.add(0) = 1;
                    *(*batch.seq_id.add(0)).add(0) = 0;
                    *batch.logits.add(0) = 1;
                }

                let ret = unsafe { (lib.decode)(self.ctx, batch) };
                unsafe { (lib.batch_free)(batch) };
                if ret != 0 {
                    break;
                }

                token = unsafe { (lib.sampler_sample)(chain, self.ctx, -1) };
            }
        }

        let cancelled = is_generating.lock().map(|g| !*g).unwrap_or(false);
        let decode_dur = decode_start_time.elapsed().as_secs_f64().max(0.0001);
        let decode_toks_sec = if decode_dur > 0.0 {
            n_generated as f64 / decode_dur
        } else {
            0.0
        };
        let total_gen_duration = gen_start_time.elapsed().as_secs_f64();
        update_inference_telemetry(|t| {
            t.generated_tokens = n_generated;
            t.decode_duration_secs = decode_dur;
            t.decode_tok_per_sec = decode_toks_sec;
            t.session_total_gen_tokens += n_generated;
            t.session_total_inference_secs += total_gen_duration;
        });

        if full_text.is_empty() {
            if cancelled {
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
                if let Some(m_ctx) = self.mtmd_ctx.take() {
                    if let Some(mtmd_free) = lib.mtmd_free {
                        mtmd_free(m_ctx);
                    }
                }
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

            // Cancellation contract (two layers): the App wraps this future in
            // `select!` against the run's child CancellationToken, so a
            // cancelled run drops this await immediately; the blocking decode
            // thread below ALSO polls `is_generating` per token and exits, so
            // no detached inference keeps burning CPU after cancellation.
            // Every cancel path sets the flag AND the token — keep both.
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
