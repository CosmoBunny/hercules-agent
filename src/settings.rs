//! Runtime settings: power mode + repeat detector.

use std::sync::Mutex;
use sysinfo::System;

/// How hard to push CPU/GPU for local LLM backends (llama-server).
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum PowerMode {
    /// Lower threads / GPU layers when system is already hot.
    PowerSaver,
    /// Default — use available cores and full GPU offload request.
    Normal,
    /// Max threads + max GPU layers; no throttling.
    Extreme,
}

impl PowerMode {
    pub fn label(self) -> &'static str {
        match self {
            Self::PowerSaver => "Power Saver",
            Self::Normal => "Normal (auto)",
            Self::Extreme => "Extreme",
        }
    }

    pub fn description(self) -> &'static str {
        match self {
            Self::PowerSaver => "Ease off when CPU is busy; fewer threads / GPU layers",
            Self::Normal => "Use available cores; GPU offload as available (default)",
            Self::Extreme => "Max threads + max GPU layers; no mercy on low speed",
        }
    }

    /// Threads for llama-server `-t`.
    pub fn threads(self) -> usize {
        let n = std::thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(4)
            .max(1);
        match self {
            Self::PowerSaver => {
                let load = cpu_load_pct();
                if load > 85.0 {
                    1
                } else if load > 65.0 {
                    (n / 2).max(1)
                } else {
                    (n * 3 / 4).max(1)
                }
            }
            // Leave one core free so the TUI + OS stay responsive (less thermal thrash).
            Self::Normal => (n.saturating_sub(1)).max(1).min((n * 3 / 4).max(1)),
            Self::Extreme => n,
        }
    }

    /// GPU layers for `-ngl` (0 = CPU only).
    /// Prefer `HERCULES_N_GPU_LAYERS=0` on no-VRAM / low-RAM machines.
    pub fn n_gpu_layers(self) -> i32 {
        if let Ok(v) = std::env::var("HERCULES_N_GPU_LAYERS") {
            if let Ok(n) = v.parse::<i32>() {
                return n;
            }
        }
        match self {
            // Default CPU. iGPU Vulkan offload often hangs/slows more than it helps;
            // set HERCULES_N_GPU_LAYERS or Extreme when you have real VRAM.
            Self::PowerSaver => 0,
            Self::Normal => 0,
            Self::Extreme => 99,
        }
    }

    /// Chat max tokens for HTTP / llama-server completions.
    pub fn max_tokens(self) -> u32 {
        match self {
            Self::PowerSaver => 512,
            Self::Normal => 4096, // files need room
            Self::Extreme => 8192,
        }
    }

    /// Max new tokens for pure-Rust llama.rs (CPU decode is slower — keep tighter).
    pub fn pure_rust_n_predict(self) -> usize {
        match self {
            Self::PowerSaver => 64,
            Self::Normal => 160,
            Self::Extreme => 320,
        }
    }
}

fn cpu_load_pct() -> f32 {
    let mut sys = System::new();
    sys.refresh_cpu_usage();
    // First refresh often 0; second sample is better but we avoid sleep in hot path
    let cpus = sys.cpus();
    if cpus.is_empty() {
        return 0.0;
    }
    cpus.iter().map(|c| c.cpu_usage()).sum::<f32>() / cpus.len() as f32
}

/// Default context window (tokens). Override with `HERCULES_CTX`.
pub const DEFAULT_CONTEXT_TOKEN_LIMIT: usize = 256_000;
/// Hard ceiling for context (tokens). Menu + env clamp here.
pub const MAX_CONTEXT_TOKEN_LIMIT: usize = 1_048_576; // 1M
/// When estimated context usage reaches this fraction, compact → memory.
pub const CONTEXT_COMPACT_RATIO: f32 = 0.80;

/// Default sampling temperature (low = more deterministic / better tool following).
pub const DEFAULT_TEMPERATURE: f32 = 0.2;

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum WebSearchProvider {
    DuckDuckGo,
    Google,
    Brave,
    Tavily,
    Searxng,
    Arxiv,
}

impl WebSearchProvider {
    pub fn label(self) -> &'static str {
        match self {
            Self::DuckDuckGo => "DuckDuckGo (Default, Zero Config)",
            Self::Google => "Google Search",
            Self::Brave => "Brave Search",
            Self::Tavily => "Tavily AI",
            Self::Searxng => "SearXNG",
            Self::Arxiv => "ArXiv Papers",
        }
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct RuntimeSettings {
    #[serde(default = "default_power_mode")]
    pub power_mode: PowerMode,
    #[serde(default = "default_web_search_provider")]
    pub web_search_provider: WebSearchProvider,
    #[serde(default = "default_max_subagents")]
    pub max_subagents: usize,
    #[serde(default = "default_max_subagent_depth")]
    pub max_subagent_depth: usize,
    #[serde(default = "default_stall_timeout_secs")]
    pub stall_timeout_secs: u64,
    /// Consecutive identical / pattern hits before we intervene.
    #[serde(default = "default_repeat_threshold")]
    pub repeat_threshold: usize,
    /// Also scan `<think>` bodies for looping phrases.
    #[serde(default = "default_true")]
    pub repeat_detect_thinking: bool,
    /// Max context tokens (prompt budget). Default 256K.
    #[serde(default = "context_limit_from_env")]
    pub context_token_limit: usize,
    /// Fraction of limit that triggers auto-compact (default 0.80).
    #[serde(default = "default_compact_ratio")]
    pub compact_ratio: f32,
    /// Sampling temperature for llama.cpp / HTTP / (hint for llama.rs).
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    /// Send recent sub-agent response back to main agent on finish.
    #[serde(default = "default_true")]
    pub subagent_quick_response: bool,
    /// OCR model / engine (auto, llava, qwen2-vl, tesseract).
    #[serde(default = "default_none")]
    pub ocr_model: String,
    /// Image generative model / engine (auto, sd-webui, ollama, diffusers).
    #[serde(default = "default_none")]
    pub image_gen_model: String,
    /// Video generative model / engine (auto, animatediff, cogvideox).
    #[serde(default = "default_none")]
    pub video_gen_model: String,
    /// HuggingFace API token for authenticated searches & downloads.
    #[serde(default)]
    pub hf_token: Option<String>,
}

fn default_power_mode() -> PowerMode { PowerMode::Normal }
fn default_web_search_provider() -> WebSearchProvider { WebSearchProvider::DuckDuckGo }
fn default_max_subagents() -> usize { 4 }
fn default_max_subagent_depth() -> usize { 3 }
fn default_stall_timeout_secs() -> u64 { 300 }
fn default_repeat_threshold() -> usize { 10 }
fn default_true() -> bool { true }
fn default_compact_ratio() -> f32 { CONTEXT_COMPACT_RATIO }
fn default_temperature() -> f32 { DEFAULT_TEMPERATURE }
fn default_none() -> String { "none".to_string() }

impl Default for RuntimeSettings {
    fn default() -> Self {
        let env_tok = std::env::var("HF_TOKEN").ok().filter(|s| !s.trim().is_empty());
        Self {
            power_mode: PowerMode::Normal,
            web_search_provider: WebSearchProvider::DuckDuckGo,
            max_subagents: 4,
            max_subagent_depth: 3,
            stall_timeout_secs: 300, // 5m default
            repeat_threshold: 10,
            repeat_detect_thinking: true,
            context_token_limit: context_limit_from_env(),
            compact_ratio: CONTEXT_COMPACT_RATIO,
            temperature: DEFAULT_TEMPERATURE,
            subagent_quick_response: true,
            ocr_model: "none".to_string(),
            image_gen_model: "none".to_string(),
            video_gen_model: "none".to_string(),
            hf_token: env_tok,
        }
    }
}

pub fn format_stall_timeout(secs: u64) -> String {
    if secs == 0 {
        "Unlimited (No Time Limit)".to_string()
    } else if secs < 60 {
        format!("{}s", secs)
    } else {
        format!("{}m", secs / 60)
    }
}

pub fn cycle_stall_timeout() -> u64 {
    let cur = get_settings().stall_timeout_secs;
    let next = match cur {
        300 => 600,
        600 => 1200,
        1200 => 0, // 0 = Unlimited
        _ => 300,
    };
    if let Ok(mut g) = SETTINGS.lock() {
        let s = g.get_or_insert_with(RuntimeSettings::default);
        s.stall_timeout_secs = next;
    }
    next
}

pub fn nudge_stall_timeout(dir: i32) -> u64 {
    let cur = get_settings().stall_timeout_secs;
    let presets = [300, 600, 1200, 0];
    let idx = presets.iter().position(|&x| x == cur).unwrap_or(0);
    let new_idx = if dir > 0 {
        (idx + 1) % presets.len()
    } else if idx == 0 {
        presets.len() - 1
    } else {
        idx - 1
    };
    let next = presets[new_idx];
    if let Ok(mut g) = SETTINGS.lock() {
        let s = g.get_or_insert_with(RuntimeSettings::default);
        s.stall_timeout_secs = next;
    }
    next
}

pub fn cycle_web_search_provider() -> WebSearchProvider {
    let next = match get_settings().web_search_provider {
        WebSearchProvider::DuckDuckGo => WebSearchProvider::Google,
        WebSearchProvider::Google => WebSearchProvider::Brave,
        WebSearchProvider::Brave => WebSearchProvider::Tavily,
        WebSearchProvider::Tavily => WebSearchProvider::Searxng,
        WebSearchProvider::Searxng => WebSearchProvider::Arxiv,
        WebSearchProvider::Arxiv => WebSearchProvider::DuckDuckGo,
    };
    if let Ok(mut g) = SETTINGS.lock() {
        let s = g.get_or_insert_with(RuntimeSettings::default);
        s.web_search_provider = next;
    }
    next
}

pub fn toggle_subagent_quick_response() -> bool {
    if let Ok(mut g) = SETTINGS.lock() {
        let s = g.get_or_insert_with(RuntimeSettings::default);
        s.subagent_quick_response = !s.subagent_quick_response;
        return s.subagent_quick_response;
    }
    true
}

fn context_limit_from_env() -> usize {
    if let Ok(v) = std::env::var("HERCULES_CTX") {
        if let Ok(n) = v.parse::<usize>() {
            return n.clamp(2048, MAX_CONTEXT_TOKEN_LIMIT);
        }
    }
    // Default always 256K (menu +/− can change 4K…1M).
    DEFAULT_CONTEXT_TOKEN_LIMIT
}

/// Rough token estimate (chars/4). Good enough for budget gating.
pub fn estimate_tokens(text: &str) -> usize {
    (text.chars().count() + 3) / 4
}

pub fn settings_toml_path() -> std::path::PathBuf {
    let home = std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| std::path::PathBuf::from("."));
    home.join(".local").join("hercules").join("settings.toml")
}

pub fn load_settings_from_disk() -> RuntimeSettings {
    let path = settings_toml_path();
    if let Ok(text) = std::fs::read_to_string(&path) {
        if let Ok(s) = toml::from_str::<RuntimeSettings>(&text) {
            return s;
        }
    }
    RuntimeSettings::default()
}

pub fn save_settings_to_disk(settings: &RuntimeSettings) {
    let path = settings_toml_path();
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    if let Ok(text) = toml::to_string_pretty(settings) {
        let _ = std::fs::write(&path, text);
    }
}

static SETTINGS: Mutex<Option<RuntimeSettings>> = Mutex::new(None);

fn ensure_settings() -> RuntimeSettings {
    let mut g = SETTINGS.lock().unwrap_or_else(|e| e.into_inner());
    if g.is_none() {
        *g = Some(load_settings_from_disk());
    }
    g.clone().unwrap_or_default()
}

pub fn get_settings() -> RuntimeSettings {
    ensure_settings()
}

pub fn get_hf_token() -> Option<String> {
    get_settings().hf_token
}

pub fn set_hf_token(tok: String) {
    if let Ok(mut g) = SETTINGS.lock() {
        let s = g.get_or_insert_with(RuntimeSettings::default);
        let trimmed = tok.trim().to_string();
        if trimmed.is_empty() {
            s.hf_token = None;
        } else {
            s.hf_token = Some(trimmed);
        }
        save_settings_to_disk(s);
    }
}

pub fn clear_hf_token() {
    if let Ok(mut g) = SETTINGS.lock() {
        let s = g.get_or_insert_with(RuntimeSettings::default);
        s.hf_token = None;
        save_settings_to_disk(s);
    }
}

pub fn context_token_limit() -> usize {
    // Menu value wins; HERCULES_CTX only seeds default at first init
    get_settings()
        .context_token_limit
        .clamp(2048, MAX_CONTEXT_TOKEN_LIMIT)
}

pub fn set_context_token_limit(n: usize) {
    if let Ok(mut g) = SETTINGS.lock() {
        let s = g.get_or_insert_with(RuntimeSettings::default);
        s.context_token_limit = n.clamp(2048, MAX_CONTEXT_TOKEN_LIMIT);
    }
}

pub fn get_ocr_model() -> String {
    get_settings().ocr_model
}

pub fn set_ocr_model(m: String) {
    if let Ok(mut g) = SETTINGS.lock() {
        let s = g.get_or_insert_with(RuntimeSettings::default);
        s.ocr_model = m;
    }
}

pub fn get_image_gen_model() -> String {
    get_settings().image_gen_model
}

pub fn set_image_gen_model(m: String) {
    if let Ok(mut g) = SETTINGS.lock() {
        let s = g.get_or_insert_with(RuntimeSettings::default);
        s.image_gen_model = m;
    }
}

pub fn get_video_gen_model() -> String {
    get_settings().video_gen_model
}

pub fn set_video_gen_model(m: String) {
    if let Ok(mut g) = SETTINGS.lock() {
        let s = g.get_or_insert_with(RuntimeSettings::default);
        s.video_gen_model = m;
    }
}

fn is_command_available(cmd: &str) -> bool {
    std::process::Command::new("which")
        .arg(cmd)
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

fn detect_ollama_vision_models() -> Vec<String> {
    let mut models = Vec::new();
    if let Ok(output) = std::process::Command::new("ollama").arg("list").output() {
        if output.status.success() {
            let stdout = String::from_utf8_lossy(&output.stdout);
            for line in stdout.lines().skip(1) {
                let name = line.split_whitespace().next().unwrap_or("");
                let lower = name.to_lowercase();
                if lower.contains("llava") || lower.contains("vision") || lower.contains("qwen2-vl") || lower.contains("bakllava") {
                    models.push(name.to_string());
                }
            }
        }
    }
    models
}

pub fn get_available_ocr_engines() -> Vec<String> {
    let mut engines = vec!["none".to_string(), "auto".to_string(), "integrated".to_string()];
    if is_command_available("tesseract") {
        engines.push("tesseract".to_string());
    }
    for m in detect_ollama_vision_models() {
        if !engines.contains(&m) {
            engines.push(m);
        }
    }
    engines
}

pub fn get_available_image_gen_engines() -> Vec<String> {
    let mut engines = vec!["none".to_string(), "auto".to_string(), "integrated".to_string()];
    if is_command_available("python3") {
        engines.push("python-pil".to_string());
    }
    engines
}

pub fn get_available_video_gen_engines() -> Vec<String> {
    let mut engines = vec!["none".to_string(), "auto".to_string(), "integrated".to_string()];
    if is_command_available("ffmpeg") || is_command_available("python3") {
        engines.push("ffmpeg".to_string());
    }
    engines
}

pub fn cycle_ocr_model() -> String {
    let available = get_available_ocr_engines();
    let cur = get_ocr_model();
    let idx = available.iter().position(|r| r == &cur).unwrap_or(0);
    let next = available[(idx + 1) % available.len()].clone();
    set_ocr_model(next.clone());
    next
}

pub fn cycle_image_gen_model() -> String {
    let available = get_available_image_gen_engines();
    let cur = get_image_gen_model();
    let idx = available.iter().position(|r| r == &cur).unwrap_or(0);
    let next = available[(idx + 1) % available.len()].clone();
    set_image_gen_model(next.clone());
    next
}

pub fn cycle_video_gen_model() -> String {
    let available = get_available_video_gen_engines();
    let cur = get_video_gen_model();
    let idx = available.iter().position(|r| r == &cur).unwrap_or(0);
    let next = available[(idx + 1) % available.len()].clone();
    set_video_gen_model(next.clone());
    next
}

/// Menu presets for context window (tokens). Enter / +/− step these.
pub const CONTEXT_PRESETS: &[usize] = &[
    4_096, 8_192, 16_384, 32_768, 65_536, 131_072, 262_144,   // 256K default
    524_288,   // 512K
    1_048_576, // 1M max
];

/// Cycle context limit: 4K → 8K → 16K → 32K → 64K → 128K → 256K → 4K.
pub fn cycle_context_token_limit() -> usize {
    nudge_context_token_limit(1)
}

/// Step context preset: `dir > 0` next larger, `dir < 0` next smaller (clamped).
pub fn nudge_context_token_limit(dir: i32) -> usize {
    let cur = context_token_limit();
    // Snap to nearest preset index
    let mut idx = 0usize;
    let mut best = usize::MAX;
    for (i, &p) in CONTEXT_PRESETS.iter().enumerate() {
        let d = cur.abs_diff(p);
        if d < best {
            best = d;
            idx = i;
        }
    }
    let next_idx = if dir > 0 {
        (idx + 1).min(CONTEXT_PRESETS.len() - 1)
    } else if dir < 0 {
        idx.saturating_sub(1)
    } else {
        idx
    };
    let next = CONTEXT_PRESETS[next_idx];
    set_context_token_limit(next);
    next
}

/// Step repeat threshold by `delta` (clamped 2..=100).
pub fn nudge_repeat_threshold(delta: i32) -> usize {
    let cur = get_settings().repeat_threshold as i32;
    let next = (cur + delta).clamp(2, 100) as usize;
    set_repeat_threshold(next);
    next
}

pub fn temperature() -> f32 {
    get_settings().temperature.clamp(0.0, 2.0)
}

pub fn set_temperature(t: f32) {
    if let Ok(mut g) = SETTINGS.lock() {
        g.get_or_insert_with(RuntimeSettings::default).temperature = t.clamp(0.0, 2.0);
    }
}

/// Step temperature by 0.05 (`dir` +1 / -1). Clamped 0.0..=2.0.
pub fn nudge_temperature(dir: i32) -> f32 {
    let cur = temperature();
    let step = 0.05_f32 * dir as f32;
    let next = ((cur + step) * 100.0).round() / 100.0;
    let next = next.clamp(0.0, 2.0);
    set_temperature(next);
    next
}

/// Human label for context size (e.g. "32K", "256K", "1M").
pub fn format_context_tokens(n: usize) -> String {
    if n >= 1_048_576 {
        let m = n as f64 / 1_048_576.0;
        if (m - m.round()).abs() < 0.05 {
            format!("{}M", m.round() as u32)
        } else {
            format!("{m:.1}M")
        }
    } else if n >= 1024 {
        format!("{}K", n / 1024)
    } else {
        format!("{n}")
    }
}

/// How long to wait for llama-server health with this context size.
pub fn server_health_timeout_secs(n_ctx: usize) -> u64 {
    // First mmap + compute graph on CPU can take several minutes even at 4K–8K.
    match n_ctx {
        0..=8_192 => 240,
        8_193..=32_768 => 300,
        32_769..=65_536 => 420,
        65_537..=131_072 => 540,
        131_073..=262_144 => 720,
        262_145..=524_288 => 900,
        _ => 1200, // 1M
    }
}

pub fn set_power_mode(mode: PowerMode) {
    if let Ok(mut g) = SETTINGS.lock() {
        g.get_or_insert_with(RuntimeSettings::default).power_mode = mode;
    }
}

pub fn set_repeat_threshold(n: usize) {
    if let Ok(mut g) = SETTINGS.lock() {
        g.get_or_insert_with(RuntimeSettings::default)
            .repeat_threshold = n.clamp(2, 100);
    }
}

pub fn cycle_repeat_threshold() {
    if let Ok(mut g) = SETTINGS.lock() {
        let s = g.get_or_insert_with(RuntimeSettings::default);
        let next = match s.repeat_threshold {
            2..=5 => 10,
            6..=10 => 15,
            11..=15 => 20,
            16..=20 => 30,
            _ => 5,
        };
        s.repeat_threshold = next;
    }
}

pub fn toggle_repeat_thinking() {
    if let Ok(mut g) = SETTINGS.lock() {
        let s = g.get_or_insert_with(RuntimeSettings::default);
        s.repeat_detect_thinking = !s.repeat_detect_thinking;
    }
}

// ---------------------------------------------------------------------------
// Repeat detector
// ---------------------------------------------------------------------------

/// Normalize agent output for comparison (tool tags + whitespace).
pub fn normalize_for_repeat(s: &str) -> String {
    let mut t = s.trim().to_string();
    // Collapse whitespace
    while t.contains("  ") {
        t = t.replace("  ", " ");
    }
    t = t.replace('\n', " ");
    // Prefer first tool tag if present
    if let Some(start) = t.find('<') {
        if let Some(end_rel) = t[start..].find('>') {
            let tag = &t[start..start + end_rel + 1];
            // full self-closing or open tag
            return tag.to_lowercase();
        }
    }
    t.to_lowercase()
}

/// Extract think body for pattern scan.
fn think_bodies(s: &str) -> String {
    let mut out = String::new();
    let mut rest = s;
    while let Some(i) = rest.find("<think>") {
        let after = &rest[i + 7..];
        if let Some(j) = after.find("</think>") {
            out.push_str(&after[..j]);
            out.push(' ');
            rest = &after[j + 8..];
        } else {
            out.push_str(after);
            break;
        }
    }
    out
}

/// Simple phrase chunks for looping language ("let me think", "read the file", …).
fn phrase_signature(s: &str) -> String {
    let lower = s.to_lowercase();
    let cleaned: String = lower
        .chars()
        .map(|c| if c.is_alphanumeric() { c } else { ' ' })
        .collect();
    cleaned
        .split_whitespace()
        .take(8)
        .collect::<Vec<_>>()
        .join(" ")
}

/// Returns a human-readable reason if history shows a repeat loop at/above threshold.
pub fn detect_repeat_loop(history: &[String], settings: &RuntimeSettings) -> Option<String> {
    let thr = settings.repeat_threshold.max(2);
    if history.len() < thr {
        return None;
    }

    let norms: Vec<String> = history
        .iter()
        .map(|s| normalize_for_repeat(s))
        .filter(|s| !s.is_empty())
        .collect();
    if norms.len() < thr {
        return None;
    }

    // 1) Same signature thr times in a row
    let last = norms.last().unwrap();
    let mut same = 0usize;
    for n in norms.iter().rev() {
        if n == last {
            same += 1;
        } else {
            break;
        }
    }
    if same >= thr {
        return Some(format!(
            "identical output ×{}: '{}'",
            same,
            truncate(last, 48)
        ));
    }

    // 2) Alternating A,B,A,B… (period 2)
    if norms.len() >= thr {
        let a = &norms[norms.len() - 2];
        let b = &norms[norms.len() - 1];
        if a != b {
            let mut ok = true;
            let mut count = 0usize;
            for (i, n) in norms.iter().rev().enumerate() {
                let expect = if i % 2 == 0 { b } else { a };
                if n != expect {
                    ok = false;
                    break;
                }
                count += 1;
                if count >= thr {
                    break;
                }
            }
            if ok && count >= thr {
                return Some(format!(
                    "alternating pattern ×{}: '{}' ↔ '{}'",
                    count,
                    truncate(a, 24),
                    truncate(b, 24)
                ));
            }
        }
    }

    // 3) Period-3 cycles (A,B,C,A,B,C…)
    if norms.len() >= thr && thr >= 6 {
        let p = 3;
        let slice = &norms[norms.len().saturating_sub(thr)..];
        if slice.len() >= p * 2 {
            let pattern: Vec<&String> = slice[..p].iter().collect();
            let mut ok = true;
            for (i, n) in slice.iter().enumerate() {
                if n != pattern[i % p] {
                    ok = false;
                    break;
                }
            }
            if ok {
                return Some(format!(
                    "repeating cycle (period {}): '{}'",
                    p,
                    truncate(
                        &pattern
                            .iter()
                            .map(|s| s.as_str())
                            .collect::<Vec<_>>()
                            .join(" | "),
                        60
                    )
                ));
            }
        }
    }

    // 4) Thinking-loop language (optional)
    if settings.repeat_detect_thinking {
        let mut think_sigs: Vec<String> = Vec::new();
        for h in history.iter().rev().take(thr.saturating_mul(2)) {
            let body = think_bodies(h);
            if body.trim().len() < 8 {
                continue;
            }
            // Split into rough sentences / clauses
            for part in body.split(|c| c == '.' || c == '\n' || c == '!') {
                let sig = phrase_signature(part);
                if sig.split_whitespace().count() >= 3 {
                    think_sigs.push(sig);
                }
            }
        }
        // Count most common phrase signature in recent thinks
        if think_sigs.len() >= thr {
            use std::collections::HashMap;
            let mut counts: HashMap<&str, usize> = HashMap::new();
            for s in &think_sigs {
                *counts.entry(s.as_str()).or_insert(0) += 1;
            }
            if let Some((sig, c)) = counts.into_iter().max_by_key(|(_, c)| *c) {
                if c >= thr {
                    return Some(format!("thinking loop ×{}: '{}'", c, truncate(sig, 48)));
                }
            }
        }
    }

    None
}

fn truncate(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        let t: String = s.chars().take(max).collect();
        format!("{}…", t)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_identical_tool_spam() {
        let mut hist = Vec::new();
        for _ in 0..10 {
            hist.push(r#"`<ls path="$CURRENT">`"#.to_string());
        }
        let s = RuntimeSettings {
            repeat_threshold: 10,
            repeat_detect_thinking: true,
            ..Default::default()
        };
        assert!(detect_repeat_loop(&hist, &s).is_some());
    }

    #[test]
    fn detects_alternating_pattern() {
        let mut hist = Vec::new();
        for _ in 0..5 {
            hist.push("Let me think about this.".into());
            hist.push(r#"<read src="$CURRENT/a.rs">"#.into());
        }
        let s = RuntimeSettings {
            repeat_threshold: 8,
            repeat_detect_thinking: false,
            ..Default::default()
        };
        let r = detect_repeat_loop(&hist, &s);
        assert!(r.is_some(), "{:?}", r);
        assert!(r.unwrap().contains("alternating"));
    }

    #[test]
    fn below_threshold_ok() {
        let hist = vec![
            r#"<ls path="$CURRENT">"#.into(),
            r#"<ls path="$CURRENT">"#.into(),
        ];
        let s = RuntimeSettings {
            repeat_threshold: 10,
            ..Default::default()
        };
        assert!(detect_repeat_loop(&hist, &s).is_none());
    }
}
