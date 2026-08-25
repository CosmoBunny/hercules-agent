//! Raw unsafe C FFI bindings to llama.cpp.
//!
//! Two modes, selected at compile time:
//!   - `llama-cpp-static` feature ON  → symbols resolved at link time from
//!     the static archives built by build.rs. Zero runtime overhead, no shared
//!     library search, self-contained binary.
//!   - `llama-cpp-static` feature OFF → `libloading` dlopen path (existing
//!     behaviour) — useful for distro packages or CI without a C++ compiler.
//!
//! All opaque types are `c_void`. Access via [`LlamaLib`] global singleton.

use std::ffi::c_void;
use std::os::raw::{c_char, c_float, c_int};
#[cfg(not(feature = "llama-cpp-static"))]
use std::path::PathBuf;
use std::sync::{Arc, OnceLock};

// ---------------------------------------------------------------------------
// C types
// ---------------------------------------------------------------------------

pub type LlamaToken = i32;
pub type LlamaPos = i32;
pub type LlamaSeqId = i32;

pub type LlamaModel = c_void;
pub type LlamaContext = c_void;
pub type LlamaVocab = c_void;
pub type LlamaSampler = c_void;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct LlamaBatch {
    pub n_tokens: i32,
    pub token: *mut i32,
    pub embd: *mut f32,
    pub pos: *mut i32,
    pub n_seq_id: *mut i32,
    pub seq_id: *mut *mut i32,
    pub logits: *mut i8,
}

/// Matches current llama.cpp `llama_model_params` (devices-first layout).
/// Only mutate fields you need after `llama_model_default_params()`.
#[repr(C)]
pub struct LlamaModelParams {
    /// NULL-terminated device list (NULL = all).
    pub devices: *mut c_void,
    pub tensor_buft_overrides: *const c_void,
    pub n_gpu_layers: i32,
    pub split_mode: i32,
    /// mmap/mlock etc. (`enum llama_load_mode`).
    pub load_mode: i32,
    pub main_gpu: i32,
    pub tensor_split: *const c_float,
    pub progress_callback: *const c_void,
    pub progress_callback_user_data: *const c_void,
    pub kv_overrides: *const c_void,
    // booleans packed at the end (C ABI)
    pub vocab_only: bool,
    pub check_tensors: bool,
    pub use_extra_bufts: bool,
    pub no_host: bool,
    pub no_alloc: bool,
    pub load_mtp: bool,
}

/// Matches current llama.cpp `llama_context_params`.
/// Wrong layout here causes heap/assert crashes (`n_outputs_max`, free invalid, …).
#[repr(C)]
pub struct LlamaContextParams {
    pub n_ctx: u32,
    pub n_batch: u32,
    pub n_ubatch: u32,
    pub n_seq_max: u32,
    /// Present on recent llama.cpp (0 = default). Keep for layout; do not invent values.
    pub n_rs_seq: u32,
    /// 0 = let llama.cpp pick. Must not be shifted by extra phantom fields.
    pub n_outputs_max: u32,
    pub n_threads: i32,
    pub n_threads_batch: i32,

    pub ctx_type: i32,
    pub rope_scaling_type: i32,
    pub pooling_type: i32,
    pub attention_type: i32,
    /// `enum llama_flash_attn_type` (not a bool).
    pub flash_attn_type: i32,

    pub rope_freq_base: f32,
    pub rope_freq_scale: f32,
    pub yarn_ext_factor: f32,
    pub yarn_attn_factor: f32,
    pub yarn_beta_fast: f32,
    pub yarn_beta_slow: f32,
    pub yarn_orig_ctx: u32,
    pub defrag_thold: f32,

    pub cb_eval: *const c_void,
    pub cb_eval_user_data: *const c_void,
    pub type_k: i32,
    pub type_v: i32,
    pub abort_callback: *const c_void,
    pub abort_callback_data: *const c_void,

    pub embeddings: bool,
    pub offload_kqv: bool,
    pub no_perf: bool,
    pub op_offload: bool,
    pub swa_full: bool,
    pub kv_unified: bool,

    // Trailing experimental fields (may grow). Sized so sret buffer is not too small.
    pub samplers: *mut c_void,
    pub n_samplers: usize,
    pub ctx_other: *mut c_void,
    /// Padding so C can write a newer/larger struct without clobbering our stack.
    pub _tail_pad: [u8; 64],
}

#[repr(C)]
pub struct LlamaSamplerChainParams {
    pub no_perf: bool,
}

// ---------------------------------------------------------------------------
// Function pointer types
// ---------------------------------------------------------------------------

type FnBackendInit = unsafe extern "C" fn();
type FnBackendFree = unsafe extern "C" fn();
type FnModelDefaultParams = unsafe extern "C" fn() -> LlamaModelParams;
type FnModelLoadFromFile = unsafe extern "C" fn(*const c_char, LlamaModelParams) -> *mut LlamaModel;
type FnModelFree = unsafe extern "C" fn(*mut LlamaModel);
type FnModelGetVocab = unsafe extern "C" fn(*const LlamaModel) -> *const LlamaVocab;
type FnContextDefaultParams = unsafe extern "C" fn() -> LlamaContextParams;
type FnInitFromModel = unsafe extern "C" fn(*mut LlamaModel, LlamaContextParams) -> *mut LlamaContext;
type FnContextFree = unsafe extern "C" fn(*mut LlamaContext);
type FnNCtx = unsafe extern "C" fn(*const LlamaContext) -> u32;
type FnNCtxTrain = unsafe extern "C" fn(*const LlamaModel) -> i32;
type FnTokenize = unsafe extern "C" fn(*const LlamaVocab, *const c_char, c_int, *mut LlamaToken, c_int, bool, bool) -> c_int;
type FnTokenToPiece = unsafe extern "C" fn(*const LlamaVocab, LlamaToken, *mut c_char, c_int, c_int, bool) -> c_int;
type FnVocabBos = unsafe extern "C" fn(*const LlamaVocab) -> LlamaToken;
type FnVocabEos = unsafe extern "C" fn(*const LlamaVocab) -> LlamaToken;
type FnVocabGetAddBos = unsafe extern "C" fn(*const LlamaVocab) -> bool;
type FnTokenIsEog = unsafe extern "C" fn(*const LlamaVocab, LlamaToken) -> bool;
type FnBatchGetOne = unsafe extern "C" fn(*mut LlamaToken, c_int) -> LlamaBatch;
type FnBatchFree = unsafe extern "C" fn(LlamaBatch);
type FnDecode = unsafe extern "C" fn(*mut LlamaContext, LlamaBatch) -> c_int;
type FnGetLogitsIth = unsafe extern "C" fn(*mut LlamaContext, c_int) -> *mut f32;
type FnSamplerChainDefaultParams = unsafe extern "C" fn() -> LlamaSamplerChainParams;
type FnSamplerChainInit = unsafe extern "C" fn(LlamaSamplerChainParams) -> *mut LlamaSampler;
type FnSamplerChainAdd = unsafe extern "C" fn(*mut LlamaSampler, *mut LlamaSampler);
type FnSamplerInitTemp = unsafe extern "C" fn(f32) -> *mut LlamaSampler;
type FnSamplerInitTopP = unsafe extern "C" fn(f32, usize) -> *mut LlamaSampler;
type FnSamplerInitDist = unsafe extern "C" fn(u32) -> *mut LlamaSampler;
type FnSamplerSample = unsafe extern "C" fn(*mut LlamaSampler, *mut LlamaContext, c_int) -> LlamaToken;
type FnSamplerFree = unsafe extern "C" fn(*mut LlamaSampler);
/// Callback type for llama_log_set — (level: i32, text: *const c_char, user_data: *mut c_void)
type FnLogSet = unsafe extern "C" fn(
    callback: Option<unsafe extern "C" fn(i32, *const c_char, *mut c_void)>,
    user_data: *mut c_void,
);
type FnGetMemory = unsafe extern "C" fn(*const LlamaContext) -> *mut c_void;
type FnMemoryClear = unsafe extern "C" fn(*mut c_void, bool);
/// llama_state_get_size(ctx) → number of bytes needed to serialise the full KV state.
type FnStateGetSize = unsafe extern "C" fn(*const LlamaContext) -> usize;
/// llama_state_get_data(ctx, dst, size) → bytes written.
type FnStateGetData = unsafe extern "C" fn(*mut LlamaContext, *mut u8, usize) -> usize;
/// llama_state_set_data(ctx, src, size) → bytes read.
type FnStateSetData = unsafe extern "C" fn(*mut LlamaContext, *const u8, usize) -> usize;

// ---------------------------------------------------------------------------
// LlamaLib
// ---------------------------------------------------------------------------

pub struct LlamaLib {
    /// Transitive deps (libggml-*.so) kept alive for process lifetime.
    /// Only present in the dynamic-load path; not needed when statically linked.
    #[cfg(not(feature = "llama-cpp-static"))]
    _deps: Vec<libloading::Library>,
    #[cfg(not(feature = "llama-cpp-static"))]
    _lib: libloading::Library,
    pub backend_init: FnBackendInit,
    pub backend_free: FnBackendFree,
    pub model_default_params: FnModelDefaultParams,
    pub model_load_from_file: FnModelLoadFromFile,
    pub model_free: FnModelFree,
    pub model_get_vocab: FnModelGetVocab,
    pub context_default_params: FnContextDefaultParams,
    pub init_from_model: FnInitFromModel,
    pub context_free: FnContextFree,
    pub n_ctx: FnNCtx,
    pub n_ctx_train: FnNCtxTrain,
    pub tokenize: FnTokenize,
    pub token_to_piece: FnTokenToPiece,
    pub vocab_bos: FnVocabBos,
    pub vocab_eos: FnVocabEos,
    pub vocab_get_add_bos: FnVocabGetAddBos,
    pub token_is_eog: FnTokenIsEog,
    pub batch_get_one: FnBatchGetOne,
    pub batch_free: FnBatchFree,
    pub decode: FnDecode,
    pub get_logits_ith: FnGetLogitsIth,
    pub sampler_chain_default_params: FnSamplerChainDefaultParams,
    pub sampler_chain_init: FnSamplerChainInit,
    pub sampler_chain_add: FnSamplerChainAdd,
    pub sampler_init_temp: FnSamplerInitTemp,
    pub sampler_init_top_p: FnSamplerInitTopP,
    pub sampler_init_dist: FnSamplerInitDist,
    pub sampler_sample: FnSamplerSample,
    pub sampler_free: FnSamplerFree,
    pub log_set: FnLogSet,
    /// Optional: clear KV between turns (newer llama.cpp).
    pub get_memory: Option<FnGetMemory>,
    pub memory_clear: Option<FnMemoryClear>,
    /// Optional: serialise / restore the full KV-cache state (llama.cpp ≥ b3000).
    pub state_get_size: Option<FnStateGetSize>,
    pub state_get_data: Option<FnStateGetData>,
    pub state_set_data: Option<FnStateSetData>,
}

unsafe impl Send for LlamaLib {}
unsafe impl Sync for LlamaLib {}

// ===========================================================================
// Static-link path: symbols resolved at compile time from libllama.a
// ===========================================================================

#[cfg(feature = "llama-cpp-static")]
unsafe extern "C" {
    fn llama_backend_init();
    fn llama_backend_free();
    fn llama_model_default_params() -> LlamaModelParams;
    fn llama_model_load_from_file(path: *const c_char, params: LlamaModelParams) -> *mut LlamaModel;
    fn llama_model_free(model: *mut LlamaModel);
    fn llama_model_get_vocab(model: *const LlamaModel) -> *const LlamaVocab;
    fn llama_context_default_params() -> LlamaContextParams;
    fn llama_init_from_model(model: *mut LlamaModel, params: LlamaContextParams) -> *mut LlamaContext;
    fn llama_free(ctx: *mut LlamaContext);
    fn llama_n_ctx(ctx: *const LlamaContext) -> u32;
    fn llama_model_n_ctx_train(model: *const LlamaModel) -> i32;
    fn llama_tokenize(
        vocab: *const LlamaVocab, text: *const c_char, text_len: c_int,
        tokens: *mut LlamaToken, n_tokens_max: c_int, add_special: bool, parse_special: bool,
    ) -> c_int;
    fn llama_token_to_piece(
        vocab: *const LlamaVocab, token: LlamaToken, buf: *mut c_char,
        length: c_int, lstrip: c_int, special: bool,
    ) -> c_int;
    fn llama_vocab_bos(vocab: *const LlamaVocab) -> LlamaToken;
    fn llama_vocab_eos(vocab: *const LlamaVocab) -> LlamaToken;
    fn llama_vocab_get_add_bos(vocab: *const LlamaVocab) -> bool;
    fn llama_vocab_is_eog(vocab: *const LlamaVocab, token: LlamaToken) -> bool;
    fn llama_batch_get_one(tokens: *mut LlamaToken, n_tokens: c_int) -> LlamaBatch;
    fn llama_batch_free(batch: LlamaBatch);
    fn llama_decode(ctx: *mut LlamaContext, batch: LlamaBatch) -> c_int;
    fn llama_get_logits_ith(ctx: *mut LlamaContext, i: c_int) -> *mut f32;
    fn llama_sampler_chain_default_params() -> LlamaSamplerChainParams;
    fn llama_sampler_chain_init(params: LlamaSamplerChainParams) -> *mut LlamaSampler;
    fn llama_sampler_chain_add(chain: *mut LlamaSampler, sampler: *mut LlamaSampler);
    fn llama_sampler_init_temp(t: f32) -> *mut LlamaSampler;
    fn llama_sampler_init_top_p(p: f32, min_keep: usize) -> *mut LlamaSampler;
    fn llama_sampler_init_dist(seed: u32) -> *mut LlamaSampler;
    fn llama_sampler_sample(sampler: *mut LlamaSampler, ctx: *mut LlamaContext, idx: c_int) -> LlamaToken;
    fn llama_sampler_free(sampler: *mut LlamaSampler);
    fn llama_log_set(
        callback: Option<unsafe extern "C" fn(i32, *const c_char, *mut c_void)>,
        user_data: *mut c_void,
    );
    fn llama_get_memory(ctx: *const LlamaContext) -> *mut c_void;
    fn llama_memory_clear(mem: *mut c_void, data: bool);
    // KV state serialisation — present in llama.cpp ≥ b3000
    fn llama_state_get_size(ctx: *const LlamaContext) -> usize;
    fn llama_state_get_data(ctx: *mut LlamaContext, dst: *mut u8, size: usize) -> usize;
    fn llama_state_set_data(ctx: *mut LlamaContext, src: *const u8, size: usize) -> usize;
}

#[cfg(feature = "llama-cpp-static")]
impl LlamaLib {
    /// Construct a `LlamaLib` that calls statically-linked symbols directly.
    pub fn load_static() -> Self {
        Self {
            backend_init:                  llama_backend_init,
            backend_free:                  llama_backend_free,
            model_default_params:          llama_model_default_params,
            model_load_from_file:          llama_model_load_from_file,
            model_free:                    llama_model_free,
            model_get_vocab:               llama_model_get_vocab,
            context_default_params:        llama_context_default_params,
            init_from_model:               llama_init_from_model,
            context_free:                  llama_free,
            n_ctx:                         llama_n_ctx,
            n_ctx_train:                   llama_model_n_ctx_train,
            tokenize:                      llama_tokenize,
            token_to_piece:                llama_token_to_piece,
            vocab_bos:                     llama_vocab_bos,
            vocab_eos:                     llama_vocab_eos,
            vocab_get_add_bos:             llama_vocab_get_add_bos,
            token_is_eog:                  llama_vocab_is_eog,
            batch_get_one:                 llama_batch_get_one,
            batch_free:                    llama_batch_free,
            decode:                        llama_decode,
            get_logits_ith:                llama_get_logits_ith,
            sampler_chain_default_params:  llama_sampler_chain_default_params,
            sampler_chain_init:            llama_sampler_chain_init,
            sampler_chain_add:             llama_sampler_chain_add,
            sampler_init_temp:             llama_sampler_init_temp,
            sampler_init_top_p:            llama_sampler_init_top_p,
            sampler_init_dist:             llama_sampler_init_dist,
            sampler_sample:                llama_sampler_sample,
            sampler_free:                  llama_sampler_free,
            log_set:                       llama_log_set,
            get_memory:                    Some(llama_get_memory),
            memory_clear:                  Some(llama_memory_clear),
            state_get_size:                Some(llama_state_get_size),
            state_get_data:                Some(llama_state_get_data),
            state_set_data:                Some(llama_state_set_data),
        }
    }
}

// Dynamic-load impl only compiled when the static feature is OFF.
#[cfg(not(feature = "llama-cpp-static"))]
impl LlamaLib {
    /// Open a shared library with RTLD_GLOBAL on Unix so later loads can resolve
    /// NEEDED deps from absolute paths we already loaded (no LD_LIBRARY_PATH).
    ///
    /// Note: setting LD_LIBRARY_PATH *after* process start is unreliable on Linux
    /// (the dynamic linker typically does not re-scan it for dlopen deps).
    unsafe fn open_global(path: &std::path::Path) -> Result<libloading::Library, String> {
        #[cfg(unix)]
        {
            use libloading::os::unix::{Library as UnixLibrary, RTLD_GLOBAL, RTLD_NOW};
            let flags = RTLD_NOW | RTLD_GLOBAL;
            // SAFETY: caller of open_global is already unsafe; path is a valid CString path.
            let unix_lib = unsafe { UnixLibrary::open(Some(path), flags) }.map_err(|e| {
                format!("Failed to dlopen {:?}: {}", path, e)
            })?;
            Ok(unix_lib.into())
        }
        #[cfg(windows)]
        {
            // LoadLibraryW uses the directory of the DLL for its dependencies when
            // the path is absolute (with LOAD_WITH_ALTERED_SEARCH_PATH behavior
            // via full path). Preloading from the same dir is still best.
            libloading::Library::new(path).map_err(|e| {
                format!("Failed to LoadLibrary {:?}: {}", path, e)
            })
        }
        #[cfg(not(any(unix, windows)))]
        {
            libloading::Library::new(path).map_err(|e| {
                format!("Failed to load {:?}: {}", path, e)
            })
        }
    }

    /// Preload ggml stack from the same directory as libllama, in dependency order.
    /// Returns handles that must stay alive for the process lifetime.
    fn preload_deps(lib_dir: &std::path::Path) -> Vec<libloading::Library> {
        // Order matters: base → backends → ggml meta → llama
        // Versioned sonames first (what libllama NEEDED entries use), then unversioned.
        #[cfg(target_os = "windows")]
        let candidates: &[&[&str]] = &[
            &["ggml-base.dll"],
            &["ggml-cpu.dll"],
            &["ggml.dll"],
        ];
        #[cfg(target_os = "macos")]
        let candidates: &[&[&str]] = &[
            &["libggml-base.0.dylib", "libggml-base.dylib"],
            &["libggml-cpu.0.dylib", "libggml-cpu.dylib"],
            &["libggml.0.dylib", "libggml.dylib"],
        ];
        #[cfg(not(any(target_os = "windows", target_os = "macos")))]
        let candidates: &[&[&str]] = &[
            &["libggml-base.so.0", "libggml-base.so"],
            &["libggml-cpu.so.0", "libggml-cpu.so"],
            // optional CUDA/Vulkan backends if present next to the install
            &["libggml-cuda.so.0", "libggml-cuda.so"],
            &["libggml-vulkan.so.0", "libggml-vulkan.so"],
            &["libggml-hip.so.0", "libggml-hip.so"],
            &["libggml-metal.so.0", "libggml-metal.so"],
            &["libggml.so.0", "libggml.so"],
        ];

        let mut loaded = Vec::new();
        for names in candidates {
            for name in *names {
                let p = lib_dir.join(name);
                if !p.exists() {
                    continue;
                }
                match unsafe { Self::open_global(&p) } {
                    Ok(lib) => {
                        loaded.push(lib);
                        break; // one name per logical library is enough
                    }
                    Err(_) => {
                        // try next name variant
                    }
                }
            }
        }
        loaded
    }

    pub fn load() -> Result<Self, String> {
        let path = Self::resolve_path();

        // Auto-resolve transitive deps from libllama's directory so the user never
        // needs LD_LIBRARY_PATH. Preload with RTLD_GLOBAL + absolute paths.
        let mut deps = Vec::new();
        if let Some(lib_dir) = path.parent() {
            deps = Self::preload_deps(lib_dir);

            // Best-effort env prepend (helps some platforms / nested LoadLibrary).
            // Not relied upon on Linux — preload above is the real fix.
            let dir = lib_dir.to_string_lossy();
            Self::prepend_search_path_best_effort(&dir);
        }

        let lib = unsafe {
            Self::open_global(&path).map_err(|e| format!(
                "{}\nFix: install libllama{} next to its ggml deps (e.g. ~/.local/lib), \
                 or set LIBLLAMA_PATH=/path/to/libllama{}.",
                e, Self::lib_ext(), Self::lib_ext()
            ))?
        };

        macro_rules! sym {
            ($name:expr) => {{
                let s: libloading::Symbol<_> = unsafe {
                    lib.get($name).map_err(|e| {
                        format!("libllama: missing symbol '{}': {}",
                            std::str::from_utf8($name).unwrap_or("?"), e)
                    })?
                };
                *s
            }};
        }

        // Prefer modern symbol names; fall back to deprecated aliases.
        let token_is_eog: FnTokenIsEog = unsafe {
            lib.get(b"llama_vocab_is_eog")
                .or_else(|_| lib.get(b"llama_token_is_eog"))
                .map(|s: libloading::Symbol<FnTokenIsEog>| *s)
                .map_err(|e| format!("libllama: missing llama_vocab_is_eog / llama_token_is_eog: {e}"))?
        };
        let n_ctx_train: FnNCtxTrain = unsafe {
            lib.get(b"llama_model_n_ctx_train")
                .or_else(|_| lib.get(b"llama_n_ctx_train"))
                .map(|s: libloading::Symbol<FnNCtxTrain>| *s)
                .map_err(|e| format!("libllama: missing llama_model_n_ctx_train / llama_n_ctx_train: {e}"))?
        };

        Ok(Self {
            backend_init: sym!(b"llama_backend_init"),
            backend_free: sym!(b"llama_backend_free"),
            model_default_params: sym!(b"llama_model_default_params"),
            model_load_from_file: sym!(b"llama_model_load_from_file"),
            model_free: sym!(b"llama_model_free"),
            model_get_vocab: sym!(b"llama_model_get_vocab"),
            context_default_params: sym!(b"llama_context_default_params"),
            init_from_model: sym!(b"llama_init_from_model"),
            context_free: sym!(b"llama_free"),
            n_ctx: sym!(b"llama_n_ctx"),
            n_ctx_train,
            tokenize: sym!(b"llama_tokenize"),
            token_to_piece: sym!(b"llama_token_to_piece"),
            vocab_bos: sym!(b"llama_vocab_bos"),
            vocab_eos: sym!(b"llama_vocab_eos"),
            vocab_get_add_bos: sym!(b"llama_vocab_get_add_bos"),
            token_is_eog,
            batch_get_one: sym!(b"llama_batch_get_one"),
            batch_free: sym!(b"llama_batch_free"),
            decode: sym!(b"llama_decode"),
            get_logits_ith: sym!(b"llama_get_logits_ith"),
            sampler_chain_default_params: sym!(b"llama_sampler_chain_default_params"),
            sampler_chain_init: sym!(b"llama_sampler_chain_init"),
            sampler_chain_add: sym!(b"llama_sampler_chain_add"),
            sampler_init_temp: sym!(b"llama_sampler_init_temp"),
            sampler_init_top_p: sym!(b"llama_sampler_init_top_p"),
            sampler_init_dist: sym!(b"llama_sampler_init_dist"),
            sampler_sample: sym!(b"llama_sampler_sample"),
            sampler_free: sym!(b"llama_sampler_free"),
            log_set: sym!(b"llama_log_set"),
            get_memory: unsafe {
                lib.get(b"llama_get_memory")
                    .ok()
                    .map(|s: libloading::Symbol<FnGetMemory>| *s)
            },
            memory_clear: unsafe {
                lib.get(b"llama_memory_clear")
                    .ok()
                    .map(|s: libloading::Symbol<FnMemoryClear>| *s)
            },
            state_get_size: unsafe {
                lib.get(b"llama_state_get_size")
                    .ok()
                    .map(|s: libloading::Symbol<FnStateGetSize>| *s)
            },
            state_get_data: unsafe {
                lib.get(b"llama_state_get_data")
                    .ok()
                    .map(|s: libloading::Symbol<FnStateGetData>| *s)
            },
            state_set_data: unsafe {
                lib.get(b"llama_state_set_data")
                    .ok()
                    .map(|s: libloading::Symbol<FnStateSetData>| *s)
            },
            _deps: deps,
            _lib: lib,
        })
    }

    /// Best-effort PATH-style env prepend. Not reliable alone on Linux for dlopen.
    fn prepend_search_path_best_effort(dir: &str) {
        let apply = |var: &str| {
            let sep = if cfg!(windows) { ";" } else { ":" };
            let current = std::env::var(var).unwrap_or_default();
            if current.split(sep).any(|p| p == dir) {
                return;
            }
            let updated = if current.is_empty() {
                dir.to_string()
            } else {
                format!("{}{}{}", dir, sep, current)
            };
            // SAFETY: called once during LlamaLib::load via OnceLock.
            unsafe { std::env::set_var(var, updated) };
        };
        #[cfg(target_os = "linux")]
        apply("LD_LIBRARY_PATH");
        #[cfg(target_os = "macos")]
        {
            apply("DYLD_LIBRARY_PATH");
            apply("DYLD_FALLBACK_LIBRARY_PATH");
        }
        #[cfg(target_os = "windows")]
        apply("PATH");
    }

    /// Platform-specific shared library file extension.
    fn lib_ext() -> &'static str {
        #[cfg(target_os = "windows")]  { ".dll" }
        #[cfg(target_os = "macos")]    { ".dylib" }
        #[cfg(not(any(target_os = "windows", target_os = "macos")))] { ".so" }
    }

    /// Platform-specific library filename (no directory).
    fn lib_name() -> &'static str {
        #[cfg(target_os = "windows")]  { "llama.dll" }
        #[cfg(target_os = "macos")]    { "libllama.dylib" }
        #[cfg(not(any(target_os = "windows", target_os = "macos")))] { "libllama.so" }
    }

    fn resolve_path() -> PathBuf {
        // 1. Explicit env override (highest priority, works on all platforms)
        if let Ok(p) = std::env::var("LIBLLAMA_PATH") {
            return PathBuf::from(p);
        }

        let name = Self::lib_name();

        // 2. Build platform-specific candidate list
        let mut candidates: Vec<PathBuf> = Vec::new();

        // --- Linux ---
        #[cfg(target_os = "linux")]
        {
            // $HOME/.local/lib  (user install via cmake --install)
            if let Ok(home) = std::env::var("HOME") {
                candidates.push(PathBuf::from(&home).join(".local/lib").join(name));
            }
            candidates.push(PathBuf::from("/usr/local/lib").join(name));
            candidates.push(PathBuf::from("/usr/lib").join(name));
            candidates.push(PathBuf::from("/usr/lib/x86_64-linux-gnu").join(name));
            candidates.push(PathBuf::from("/usr/lib/aarch64-linux-gnu").join(name));
        }

        // --- macOS ---
        #[cfg(target_os = "macos")]
        {
            // Homebrew Apple Silicon
            candidates.push(PathBuf::from("/opt/homebrew/lib").join(name));
            // Homebrew Intel
            candidates.push(PathBuf::from("/usr/local/lib").join(name));
            // MacPorts
            candidates.push(PathBuf::from("/opt/local/lib").join(name));
            // User-local
            if let Ok(home) = std::env::var("HOME") {
                candidates.push(PathBuf::from(&home).join(".local/lib").join(name));
            }
        }

        // --- Windows ---
        #[cfg(target_os = "windows")]
        {
            // llama.cpp Windows releases unzip to: C:\llama.cpp\  or Program Files
            candidates.push(PathBuf::from(r"C:\llama.cpp").join(name));
            candidates.push(PathBuf::from(r"C:\Program Files\llama.cpp").join(name));
            // Alongside the executable (most common for bundled apps)
            if let Ok(exe) = std::env::current_exe() {
                if let Some(dir) = exe.parent() {
                    candidates.push(dir.join(name));
                }
            }
            // %LOCALAPPDATA%\llama.cpp
            if let Ok(local) = std::env::var("LOCALAPPDATA") {
                candidates.push(PathBuf::from(&local).join("llama.cpp").join(name));
            }
        }

        // Check each candidate
        for p in &candidates {
            if p.exists() {
                return p.clone();
            }
        }

        // Fallback: let the OS dynamic linker find it by bare name
        // (works if the lib directory is in PATH/LD_LIBRARY_PATH/DYLD_LIBRARY_PATH)
        PathBuf::from(name)
    }
}

// ---------------------------------------------------------------------------
// Global singleton
// ---------------------------------------------------------------------------

// ===========================================================================
// Global singleton — works for both static and dynamic paths
// ===========================================================================

static LLAMA_LIB: OnceLock<Arc<LlamaLib>> = OnceLock::new();

pub fn get_lib() -> Result<Arc<LlamaLib>, String> {
    if let Some(lib) = LLAMA_LIB.get() {
        return Ok(lib.clone());
    }

    #[cfg(feature = "llama-cpp-static")]
    let lib = Arc::new(LlamaLib::load_static());

    #[cfg(not(feature = "llama-cpp-static"))]
    let lib = Arc::new(LlamaLib::load()?);

    let _ = LLAMA_LIB.set(lib.clone());
    Ok(lib)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn load_libllama_auto_deps() {
        let lib = get_lib().expect("auto-load libllama without user LD_LIBRARY_PATH");
        let cparams = unsafe { (lib.context_default_params)() };
        assert!(cparams.n_ctx > 0);
        assert!(cparams.n_threads > 0);
        let mparams = unsafe { (lib.model_default_params)() };
        // n_gpu_layers default is usually high/all; just ensure field is readable
        let _ = mparams.n_gpu_layers;
    }
}
