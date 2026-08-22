//! Model download sessions and permanent install registry.
//!
//! # Layout
//! - Staging: `/tmp/hercules/`
//!   - `download.lock` — active session metadata (name, times, status)
//!   - `{slug}/` — partial file being written
//! - Install: `~/.local/hercules/`
//!   - `model/` — completed weight files
//!   - `models.toml` — installed model index (name → path)

use futures_util::StreamExt;
use ollama_rs::models::LocalModel;
use ollama_rs::Ollama;
use serde::{Deserialize, Serialize};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// No progress for this long → abandon lock so another model can install.
const STALE_DOWNLOAD_SECS: u64 = 90;

// ---------------------------------------------------------------------------
// Paths
// ---------------------------------------------------------------------------

pub fn tmp_hercules_dir() -> PathBuf {
    std::env::temp_dir().join("hercules")
}

pub fn download_lock_path() -> PathBuf {
    tmp_hercules_dir().join("download.lock")
}

pub fn local_hercules_dir() -> PathBuf {
    dirs_home()
        .map(|h| h.join(".local").join("hercules"))
        .unwrap_or_else(|| PathBuf::from(".local/hercules"))
}

pub fn models_dir() -> PathBuf {
    local_hercules_dir().join("model")
}

pub fn models_toml_path() -> PathBuf {
    local_hercules_dir().join("models.toml")
}

fn dirs_home() -> Option<PathBuf> {
    std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .map(PathBuf::from)
}

fn ensure_dir(path: &Path) -> Result<(), String> {
    std::fs::create_dir_all(path).map_err(|e| format!("mkdir {}: {}", path.display(), e))
}

fn now_unix() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn format_unix(ts: u64) -> String {
    // Compact UTC-ish breakdown without extra time crates
    let days = ts / 86_400;
    let rem = ts % 86_400;
    let hours = rem / 3_600;
    let mins = (rem % 3_600) / 60;
    let secs = rem % 60;
    // Days since 1970-01-01 → rough YYYY-MM-DD via civil conversion
    let (y, m, d) = civil_from_days(days as i64);
    format!("{:04}-{:02}-{:02} {:02}:{:02}:{:02} UTC", y, m, d, hours, mins, secs)
}

/// Howard Hinnant civil-from-days (UTC calendar date from days since epoch).
fn civil_from_days(z: i64) -> (i32, u32, u32) {
    let z = z + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365;
    let y = (yoe as i64) + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    (y as i32, m as u32, d as u32)
}

fn slugify_model_name(name: &str) -> String {
    let mut s: String = name
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' || c == '.' {
                c
            } else {
                '_'
            }
        })
        .collect();
    while s.contains("__") {
        s = s.replace("__", "_");
    }
    s.trim_matches('_').to_string()
}

/// Human-readable byte size (registry / download labels).
pub fn format_model_size(bytes: u64) -> String {
    format_byte_size(bytes)
}

/// Minimal URL encoding for HF search queries (space → +).
fn urlencoding_loose(s: &str) -> String {
    s.chars()
        .map(|c| match c {
            ' ' => "+".to_string(),
            '/' | '?' | '&' | '=' | '#' => format!("%{:02X}", c as u8),
            c if c.is_ascii_alphanumeric() || c == '-' || c == '_' || c == '.' => c.to_string(),
            c => format!("%{:02X}", c as u32),
        })
        .collect()
}

/// Human-readable byte size (binary-ish SI for download labels).
fn format_byte_size(bytes: u64) -> String {
    if bytes >= 1_000_000_000 {
        format!("{:.1} GB", bytes as f64 / 1_000_000_000.0)
    } else if bytes >= 1_000_000 {
        format!("{:.0} MB", bytes as f64 / 1_000_000.0)
    } else if bytes >= 1_000 {
        format!("{:.0} KB", bytes as f64 / 1_000.0)
    } else if bytes > 0 {
        format!("{bytes} B")
    } else {
        "?".into()
    }
}

/// Rough on-disk size of a **Q4_K_M** GGUF from parameter tags in the model id.
/// Used when HuggingFace API omits LFS sizes (common for large files).
fn estimate_q4_size_label(model_id: &str) -> String {
    let lower = model_id.to_lowercase();
    // Order matters: match longer tags first (1.5b before 1b, 70b before 7b)
    let (params_b, mb) = if lower.contains("405b") {
        (405.0, 220_000.0)
    } else if lower.contains("236b") || lower.contains("235b") {
        (235.0, 130_000.0)
    } else if lower.contains("70b") || lower.contains("72b") {
        (70.0, 40_000.0)
    } else if lower.contains("34b") || lower.contains("33b") || lower.contains("32b") {
        (32.0, 19_000.0)
    } else if lower.contains("27b") {
        (27.0, 16_000.0)
    } else if lower.contains("22b") {
        (22.0, 13_000.0)
    } else if lower.contains("14b") || lower.contains("13b") {
        (14.0, 8_500.0)
    } else if lower.contains("12b") {
        (12.0, 7_200.0)
    } else if lower.contains("9b") {
        (9.0, 5_400.0)
    } else if lower.contains("8b") {
        (8.0, 4_700.0)
    } else if lower.contains("7b") || lower.contains("6.7b") {
        (7.0, 4_200.0)
    } else if lower.contains("4b") || lower.contains("3.8b") {
        (4.0, 2_400.0)
    } else if lower.contains("3b") {
        (3.0, 1_900.0)
    } else if lower.contains("2b") {
        (2.0, 1_300.0)
    } else if lower.contains("1.5b") || lower.contains("1.7b") {
        (1.5, 1_000.0)
    } else if lower.contains("1b") {
        (1.0, 700.0)
    } else if lower.contains("0.5b") || lower.contains("500m") {
        (0.5, 400.0)
    } else {
        // Unknown — do not invent a fake precise GB
        return "size unknown".into();
    };
    let _ = params_b;
    if mb >= 1000.0 {
        format!("{:.1} GB", mb / 1000.0)
    } else {
        format!("{:.0} MB", mb)
    }
}

fn is_multipart_gguf(name: &str) -> bool {
    let lower = name.to_lowercase();
    lower.contains("-of-") || lower.contains(".part") || lower.contains("-00001-of-")
}

/// Rank GGUF filenames: prefer **single-file** mid-size quants (never multi-part if avoidable).
fn pick_best_gguf(files: &[String]) -> String {
    // Prefer single-file only when any exist
    let singles: Vec<&String> = files.iter().filter(|f| !is_multipart_gguf(f)).collect();
    let pool: Vec<&String> = if singles.is_empty() {
        files.iter().collect()
    } else {
        singles
    };

    let mut ranked: Vec<(i32, &String)> = pool
        .into_iter()
        .map(|f| {
            let lower = f.to_lowercase();
            let mut score = if is_multipart_gguf(f) { 500 } else { 0 };
            // Prefer common local-friendly quants
            if lower.contains("q4_k_m") {
                score += 0;
            } else if lower.contains("q4_k_s") {
                score += 1;
            } else if lower.contains("q4_0") || lower.contains("q4_1") {
                score += 2;
            } else if lower.contains("q5_k_m") || lower.contains("q5_0") {
                score += 3;
            } else if lower.contains("q3_k") {
                score += 4;
            } else if lower.contains("q6_k") || lower.contains("q8_0") {
                score += 5;
            } else if lower.contains("f16") || lower.contains("fp16") {
                score += 8;
            } else if lower.contains("f32") {
                score += 10;
            } else {
                score += 6;
            }
            if lower.contains("instruct") || lower.contains("chat") {
                score -= 1;
            }
            // Prefer smaller param counts when name encodes size (1.3b over 16b)
            if lower.contains("0.5b") || lower.contains("1b") || lower.contains("1.3b") {
                score -= 3;
            } else if lower.contains("1.5b") || lower.contains("2b") || lower.contains("3b") {
                score -= 2;
            } else if lower.contains("7b") || lower.contains("8b") {
                score += 2;
            } else if lower.contains("13b")
                || lower.contains("14b")
                || lower.contains("16b")
                || lower.contains("32b")
                || lower.contains("70b")
            {
                score += 20;
            }
            (score, f)
        })
        .collect();
    ranked.sort_by_key(|(s, _)| *s);
    ranked
        .into_iter()
        .next()
        .map(|(_, f)| f.clone())
        .unwrap_or_else(|| files[0].clone())
}

/// Given a multi-part GGUF shard name (e.g. `Foo-Q4_K_M-00001-of-00013.gguf`)
/// and the full file listing, return all sibling parts in order.
/// For single-file GGUFs, returns `vec![name.to_string()]`.
fn find_multipart_siblings(name: &str, all_files: &[String]) -> Vec<String> {
    if !is_multipart_gguf(name) {
        return vec![name.to_string()];
    }
    let lower = name.to_lowercase();
    if let Some(idx) = lower.rfind("-of-") {
        let before_of = &lower[..idx];
        if let Some(dash_idx) = before_of.rfind('-') {
            let stem = &name[..dash_idx];
            let stem_lower = stem.to_lowercase();
            let mut siblings: Vec<String> = all_files
                .iter()
                .filter(|f| {
                    let fl = f.to_lowercase();
                    fl.starts_with(&stem_lower)
                        && fl.ends_with(".gguf")
                        && fl.contains("-of-")
                })
                .cloned()
                .collect();
            siblings.sort();
            if !siblings.is_empty() {
                return siblings;
            }
        }
    }
    vec![name.to_string()]
}

// ---------------------------------------------------------------------------
// Download lock / session
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum DownloadStatus {
    InProgress,
    Incomplete,
    Complete,
}

impl Default for DownloadStatus {
    fn default() -> Self {
        Self::InProgress
    }
}

/// Written to `/tmp/hercules/download.lock` for the active (or last failed) session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DownloadLock {
    pub model_name: String,
    pub source: String,
    pub filename: String,
    pub staging_dir: String,
    pub staging_file: String,
    /// Unix seconds when download started.
    pub time_started: u64,
    /// Unix seconds of last successful progress write (heartbeat).
    pub time_updated: u64,
    /// Set only when download finishes successfully. Never set on failure.
    pub time_finished: Option<u64>,
    pub status: DownloadStatus,
    pub bytes_downloaded: u64,
    pub bytes_total: Option<u64>,
    pub error: Option<String>,
}

impl DownloadLock {
    pub fn new(model_name: &str, source: &str, filename: &str, staging_dir: &Path) -> Self {
        let now = now_unix();
        let staging_file = staging_dir.join(filename);
        Self {
            model_name: model_name.to_string(),
            source: source.to_string(),
            filename: filename.to_string(),
            staging_dir: staging_dir.display().to_string(),
            staging_file: staging_file.display().to_string(),
            time_started: now,
            time_updated: now,
            time_finished: None,
            status: DownloadStatus::InProgress,
            bytes_downloaded: 0,
            bytes_total: None,
            error: None,
        }
    }

    pub fn is_stale(&self) -> bool {
        if self.status == DownloadStatus::Complete {
            return false;
        }
        let age = now_unix().saturating_sub(self.time_updated);
        age >= STALE_DOWNLOAD_SECS
    }

    pub fn save(&self) -> Result<(), String> {
        let path = download_lock_path();
        if let Some(parent) = path.parent() {
            ensure_dir(parent)?;
        }
        let text = toml::to_string_pretty(self).map_err(|e| e.to_string())?;
        std::fs::write(&path, text).map_err(|e| format!("write lock: {}", e))
    }

    pub fn load() -> Option<Self> {
        let path = download_lock_path();
        let text = std::fs::read_to_string(path).ok()?;
        toml::from_str(&text).ok()
    }

    pub fn remove_file() {
        let _ = std::fs::remove_file(download_lock_path());
    }

    /// Force-cancel active lock + optional staging dir cleanup.
    pub fn force_cancel() -> Option<String> {
        let existing = Self::load()?;
        let name = existing.model_name.clone();
        let staging = PathBuf::from(&existing.staging_dir);
        // Keep partials only if complete; otherwise remove staging so next install is clean
        if existing.status != DownloadStatus::Complete {
            let _ = std::fs::remove_dir_all(&staging);
        }
        Self::remove_file();
        Some(name)
    }

    pub fn status_summary() -> String {
        match Self::load() {
            None => "No active download lock.".into(),
            Some(l) => format!(
                "Lock: {} | {:?} | {:.1} MB | updated {} | stale={}",
                l.model_name,
                l.status,
                l.bytes_downloaded as f64 / 1_000_000.0,
                format_unix(l.time_updated),
                l.is_stale()
            ),
        }
    }

    /// Mark incomplete on network/IO failure — do **not** touch `time_finished`.
    pub fn mark_incomplete(&mut self, err: impl Into<String>) {
        self.status = DownloadStatus::Incomplete;
        self.error = Some(err.into());
        // Intentionally leave time_finished = None and do not refresh time_updated
        // beyond last good chunk so stale cleanup still works from last progress.
        let _ = self.save();
    }

    pub fn touch_progress(&mut self, bytes: u64, total: Option<u64>) {
        self.bytes_downloaded = bytes;
        if total.is_some() {
            self.bytes_total = total;
        }
        self.time_updated = now_unix();
        self.status = DownloadStatus::InProgress;
        let _ = self.save();
    }

    pub fn mark_complete(&mut self) {
        let now = now_unix();
        self.status = DownloadStatus::Complete;
        self.time_updated = now;
        self.time_finished = Some(now);
        self.error = None;
        let _ = self.save();
    }
}

// ---------------------------------------------------------------------------
// Installed models registry (models.toml)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InstalledModel {
    pub name: String,
    pub path: String,
    pub source: String,
    pub filename: String,
    pub installed_at: u64,
    pub size_bytes: u64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ModelsRegistry {
    #[serde(default)]
    pub models: Vec<InstalledModel>,
    #[serde(default)]
    pub active_model_path: Option<String>,
}

impl ModelsRegistry {
    pub fn load() -> Self {
        let path = models_toml_path();
        match std::fs::read_to_string(&path) {
            Ok(text) => toml::from_str(&text).unwrap_or_default(),
            Err(_) => Self::default(),
        }
    }

    pub fn save(&self) -> Result<(), String> {
        let path = models_toml_path();
        if let Some(parent) = path.parent() {
            ensure_dir(parent)?;
        }
        let text = toml::to_string_pretty(self).map_err(|e| e.to_string())?;
        std::fs::write(&path, text).map_err(|e| format!("write models.toml: {}", e))
    }

    pub fn upsert(&mut self, entry: InstalledModel) {
        if let Some(existing) = self.models.iter_mut().find(|m| m.name == entry.name) {
            *existing = entry;
        } else {
            self.models.push(entry);
        }
    }

    pub fn get_active_model_path(&self) -> Option<String> {
        self.active_model_path.clone()
    }

    pub fn set_active_model_path(&mut self, path: String) {
        self.active_model_path = Some(path);
    }

    pub fn remove_by_name(&mut self, name: &str) -> Option<InstalledModel> {
        if let Some(pos) = self.models.iter().position(|m| m.name == name) {
            Some(self.models.remove(pos))
        } else {
            None
        }
    }

    pub fn list_display(&self) -> Vec<String> {
        self.models
            .iter()
            .map(|m| {
                let gb = m.size_bytes as f64 / 1_000_000_000.0;
                if gb >= 0.1 {
                    format!("Local GGUF: {} ({:.1} GB) [{}]", m.name, gb, m.path)
                } else {
                    format!(
                        "Local GGUF: {} ({:.0} MB) [{}]",
                        m.name,
                        m.size_bytes as f64 / 1_000_000.0,
                        m.path
                    )
                }
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Manager
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct ModelManager {
    ollama: Ollama,
}

impl ModelManager {
    pub fn new() -> Self {
        // Clean abandoned sessions at startup
        Self::cleanup_stale_downloads();
        let _ = ensure_dir(&tmp_hercules_dir());
        let _ = ensure_dir(&models_dir());
        Self {
            ollama: Ollama::default(),
        }
    }

    pub async fn list_ollama_models(&self) -> Result<Vec<LocalModel>, String> {
        self.ollama
            .list_local_models()
            .await
            .map_err(|e| e.to_string())
    }

    /// Installed Hercules models from `~/.local/hercules/models.toml`.
    pub fn list_installed_local(&self) -> Vec<String> {
        ModelsRegistry::load().list_display()
    }

    pub fn list_installed_entries(&self) -> Vec<InstalledModel> {
        ModelsRegistry::load().models
    }

    pub async fn search_all_models(&self, search: &str) -> Vec<String> {
        let mut results = Vec::new();
        let query_lower = search.trim().to_lowercase();

        // Local Hercules installs first
        for m in self.list_installed_local() {
            if query_lower.is_empty() || m.to_lowercase().contains(&query_lower) {
                results.push(m);
            }
        }

        if let Ok(local_models) = self.ollama.list_local_models().await {
            for m in local_models {
                if query_lower.is_empty() || m.name.to_lowercase().contains(&query_lower) {
                    let size_label = if m.size > 0 {
                        format_byte_size(m.size)
                    } else {
                        estimate_q4_size_label(&m.name)
                    };
                    results.push(format!("Ollama Local: {} ({size_label})", m.name));
                }
            }
        }

        if let Ok(ollama_remote_models) = self.fetch_ollama_models(search).await {
            for m in ollama_remote_models {
                let entry = format!("Ollama: {}", m);
                if !results.contains(&entry) {
                    results.push(entry);
                }
            }
        }

        if let Ok(hf_models) = self.fetch_hf_models(search).await {
            for m in hf_models {
                let entry = format!("HuggingFace: {}", m);
                if !results.contains(&entry) {
                    results.push(entry);
                }
            }
        }

        results
    }

    pub async fn fetch_hf_models(&self, search: &str) -> Result<Vec<String>, String> {
        let trimmed = search.trim();
        let url = if trimmed.contains('/') {
            let parts: Vec<&str> = trimmed.splitn(2, '/').collect();
            let p0 = parts[0].to_lowercase();
            let author = match p0.as_str() {
                "deepseek" => "deepseek-ai",
                "meta" | "llama" => "meta-llama",
                "google" | "gemma" => "google",
                "mistral" => "mistralai",
                "qwen" => "Qwen",
                "microsoft" | "phi" => "microsoft",
                _other => parts[0],
            };
            let sub_query = parts.get(1).unwrap_or(&"");
            if sub_query.is_empty() {
                format!(
                    "https://huggingface.co/api/models?author={}&full=true&limit=25",
                    author
                )
            } else {
                format!(
                    "https://huggingface.co/api/models?author={}&search={}&full=true&limit=25",
                    author, sub_query
                )
            }
        } else {
            if trimmed.is_empty() {
                "https://huggingface.co/api/models?tags=gguf&sort=downloads&direction=-1&limit=25".to_string()
            } else {
                format!(
                    "https://huggingface.co/api/models?search={}&full=true&limit=25",
                    trimmed
                )
            }
        };

        let client = reqwest::Client::new();
        let res = client
            .get(&url)
            .header("User-Agent", "Hercules-CLI/1.0")
            .send()
            .await
            .map_err(|e| e.to_string())?;

        let text = res.text().await.map_err(|e| e.to_string())?;
        let json: serde_json::Value = serde_json::from_str(&text).map_err(|e| e.to_string())?;
        let mut models = Vec::new();
        if let Some(arr) = json.as_array() {
            for item in arr {
                if let Some(id) = item.get("id").and_then(|v| v.as_str()) {
                    // Only count GGUF siblings — summing every repo file (safetensors+tokenizer+…)
                    // was the main cause of wildly wrong "size" labels in the registry.
                    let mut gguf_sizes: Vec<(String, u64)> = Vec::new();
                    if let Some(siblings) = item.get("siblings").and_then(|s| s.as_array()) {
                        for f in siblings {
                            let rfilename = f
                                .get("rfilename")
                                .or_else(|| f.get("filename"))
                                .and_then(|v| v.as_str())
                                .unwrap_or("");
                            let lower = rfilename.to_lowercase();
                            if !lower.ends_with(".gguf") {
                                continue;
                            }
                            if let Some(sz) = f.get("size").and_then(|s| s.as_u64()) {
                                if sz > 0 {
                                    gguf_sizes.push((rfilename.to_string(), sz));
                                }
                            }
                        }
                    }

                    let size_tag = if !gguf_sizes.is_empty() {
                        // Prefer Q4_K_M (or best ranked) single-file size for the label
                        let names: Vec<String> =
                            gguf_sizes.iter().map(|(n, _)| n.clone()).collect();
                        let best = pick_best_gguf(&names);
                        let best_sz = gguf_sizes
                            .iter()
                            .find(|(n, _)| n == &best)
                            .map(|(_, s)| *s)
                            .or_else(|| gguf_sizes.iter().map(|(_, s)| *s).min())
                            .unwrap_or(0);
                        let min_sz = gguf_sizes.iter().map(|(_, s)| *s).min().unwrap_or(0);
                        let max_sz = gguf_sizes.iter().map(|(_, s)| *s).max().unwrap_or(0);
                        if gguf_sizes.len() == 1 || min_sz == max_sz {
                            format!("[{} GGUF]", format_byte_size(best_sz))
                        } else {
                            // Show preferred quant + range of available quants
                            format!(
                                "[{} typ · {}–{}]",
                                format_byte_size(best_sz),
                                format_byte_size(min_sz),
                                format_byte_size(max_sz)
                            )
                        }
                    } else {
                        // No sibling sizes from API — estimate Q4_K_M from param count in id
                        format!("[~{} Q4 est.]", estimate_q4_size_label(id))
                    };

                    models.push(format!("{} {}", id, size_tag));
                }
            }
        }
        Ok(models)
    }

    pub async fn fetch_ollama_models(&self, search: &str) -> Result<Vec<String>, String> {
        let client = reqwest::Client::new();
        let url = if search.trim().is_empty() {
            "https://ollama.com/search".to_string()
        } else {
            format!("https://ollama.com/search?q={}", search.trim())
        };
        let res = client
            .get(&url)
            .header("User-Agent", "Hercules-CLI/1.0")
            .send()
            .await
            .map_err(|e| e.to_string())?;

        let text = res.text().await.map_err(|e| e.to_string())?;
        
        
        let mut models = Vec::new();
        let parts = text.split("href=\"/library/");
        let mut first = true;
        for part in parts {
            if first {
                first = false;
                continue;
            }
            if let Some(idx) = part.find('"') {
                let model = &part[..idx];
                
                // Extract sizes/tags
                let mut tags = Vec::new();
                let mut rest = part;
                while let Some(start) = rest.find("text-blue-600") {
                    rest = &rest[start..];
                    if let Some(close) = rest.find('>') {
                        rest = &rest[close + 1..];
                        if let Some(end) = rest.find("</span>") {
                            let tag_text = rest[..end].trim();
                            if !tag_text.is_empty() && !tag_text.contains('<') {
                                tags.push(tag_text.to_string());
                            }
                            rest = &rest[end..];
                        }
                    } else {
                        break;
                    }
                }
                
                if tags.is_empty() {
                    let entry = model.to_string();
                    if !models.contains(&entry) {
                        models.push(entry);
                    }
                } else {
                    for tag in tags {
                        let entry = format!("{}:{}", model, tag);
                        if !models.contains(&entry) {
                            models.push(entry);
                        }
                    }
                }
            }
        }
        Ok(models)

    }

    /// Abandon incomplete downloads whose last update is older than 10 minutes.
    pub fn cleanup_stale_downloads() {
        let root = tmp_hercules_dir();
        if !root.exists() {
            return;
        }

        // Global lock file
        if let Some(lock) = DownloadLock::load() {
            if lock.is_stale() {
                let staging = PathBuf::from(&lock.staging_dir);
                let _ = std::fs::remove_dir_all(&staging);
                DownloadLock::remove_file();
            }
        }

        // Orphan staging dirs without a live lock
        if let Ok(entries) = std::fs::read_dir(&root) {
            let now = SystemTime::now();
            for entry in entries.flatten() {
                let path = entry.path();
                if path.file_name().and_then(|n| n.to_str()) == Some("download.lock") {
                    continue;
                }
                if !path.is_dir() {
                    continue;
                }
                // Prefer lock freshness; otherwise use dir mtime
                let stale = entry
                    .metadata()
                    .ok()
                    .and_then(|m| m.modified().ok())
                    .and_then(|modified| now.duration_since(modified).ok())
                    .map(|elapsed| elapsed >= Duration::from_secs(STALE_DOWNLOAD_SECS))
                    .unwrap_or(false);
                if stale {
                    // Don't remove if this is the active non-stale lock's staging dir
                    if let Some(lock) = DownloadLock::load() {
                        if !lock.is_stale() && PathBuf::from(&lock.staging_dir) == path {
                            continue;
                        }
                    }
                    let _ = std::fs::remove_dir_all(&path);
                }
            }
        }
    }

    /// Resolve a **single GGUF** file suitable for local llama.rs / llama.cpp.
    ///
    /// HuggingFace base repos (e.g. `Qwen/Qwen2.5-1.5B`) usually ship `.safetensors`
    /// only. Local engines need GGUF, so we:
    /// 1. Look for `.gguf` in the requested repo
    /// 2. Try common GGUF mirrors (`{repo}-GGUF`, bartowski/…, etc.)
    /// 3. HF search for `*GGUF` repos matching the model name
    /// 4. Prefer Q4_K_M / Q4_0 single-file quantizations
    ///
    /// Returns `(download_repo_id, filename)`. Never returns safetensors.
    pub async fn resolve_gguf_file(&self, repo_id: &str) -> Result<(String, String, Vec<String>), String> {
        let clean_repo = repo_id
            .split('[')
            .next()
            .unwrap_or(repo_id)
            .trim()
            .trim_start_matches("HuggingFace: ")
            .trim()
            .to_string();
        // Strip trailing size labels like " [~1.0 GB Q4 est.]"
        let clean_repo = clean_repo
            .split_whitespace()
            .next()
            .unwrap_or(&clean_repo)
            .to_string();
        let client = reqwest::Client::builder()
            .user_agent("Hercules-CLI/1.0")
            .build()
            .map_err(|e| e.to_string())?;

        let short = clean_repo
            .rsplit('/')
            .next()
            .unwrap_or(&clean_repo)
            .to_string();
        let mut candidates: Vec<String> = vec![clean_repo.clone()];
        // Common community GGUF naming patterns
        if !clean_repo.to_lowercase().contains("gguf") {
            candidates.push(format!("{}-GGUF", clean_repo));
            candidates.push(format!("{}-gguf", clean_repo));
            if !clean_repo.contains("Instruct") {
                candidates.push(format!("{}-Instruct-GGUF", clean_repo));
                candidates.push(format!("{}-Instruct-gguf", clean_repo));
            }
            // Popular quant hosts (bartowski, unsloth, lmstudio-community, …)
            for org in [
                "bartowski",
                "unsloth",
                "lmstudio-community",
                "QuantFactory",
                "TheBloke",
            ] {
                candidates.push(format!("{org}/{short}-GGUF"));
                candidates.push(format!("{org}/{short}-gguf"));
                candidates.push(format!("{org}/{short}"));
            }
        }

        let mut tried = Vec::new();
        for repo in &candidates {
            if tried.contains(repo) {
                continue;
            }
            tried.push(repo.clone());
            match self.list_gguf_files(&client, repo).await {
                Ok(files) if !files.is_empty() => {
                    let best = pick_best_gguf(&files);
                    let siblings = find_multipart_siblings(&best, &files);
                    return Ok((repo.clone(), best, siblings));
                }
                Ok(_) => {}
                Err(_) => {}
            }
        }

        // HF search: find any repo with GGUF for this model name
        if let Ok(found) = self.search_gguf_repos(&client, &short).await {
            for repo in found {
                if tried.contains(&repo) {
                    continue;
                }
                tried.push(repo.clone());
                if let Ok(files) = self.list_gguf_files(&client, &repo).await {
                    if !files.is_empty() {
                        let best = pick_best_gguf(&files);
                        let siblings = find_multipart_siblings(&best, &files);
                        return Ok((repo, best, siblings));
                    }
                }
            }
        }

        Err(format!(
            "No GGUF weights for '{}'. That repo is almost certainly safetensors/PyTorch only — \
             Hercules needs a **.gguf** file. Search HF for '{} GGUF' (e.g. \
             bartowski/*-GGUF) and install that id. Tried {} candidates.",
            clean_repo,
            short,
            tried.len()
        ))
    }

    /// Search HuggingFace for GGUF-hosting repos matching `query`.
    async fn search_gguf_repos(
        &self,
        client: &reqwest::Client,
        query: &str,
    ) -> Result<Vec<String>, String> {
        let q = format!("{query} GGUF");
        let url = format!(
            "https://huggingface.co/api/models?search={}&limit=15&full=true&sort=downloads&direction=-1",
            urlencoding_loose(&q)
        );
        let res = client
            .get(&url)
            .send()
            .await
            .map_err(|e| e.to_string())?;
        if !res.status().is_success() {
            return Ok(Vec::new());
        }
        let text = res.text().await.map_err(|e| e.to_string())?;
        let json: serde_json::Value = serde_json::from_str(&text).map_err(|e| e.to_string())?;
        let mut out = Vec::new();
        if let Some(arr) = json.as_array() {
            for item in arr {
                let id = item.get("id").and_then(|v| v.as_str()).unwrap_or("");
                if id.is_empty() {
                    continue;
                }
                let id_l = id.to_lowercase();
                // Prefer repos that look like GGUF packs
                let has_gguf_sib = item
                    .get("siblings")
                    .and_then(|s| s.as_array())
                    .map(|a| {
                        a.iter().any(|f| {
                            f.get("rfilename")
                                .and_then(|x| x.as_str())
                                .map(|n| n.to_lowercase().ends_with(".gguf"))
                                .unwrap_or(false)
                        })
                    })
                    .unwrap_or(false);
                if has_gguf_sib || id_l.contains("gguf") {
                    out.push(id.to_string());
                }
            }
        }
        Ok(out)
    }

    /// Back-compat wrapper: returns filename only, or a diagnostic placeholder on failure.
    pub async fn get_model_weight_filename(&self, repo_id: &str) -> String {
        match self.resolve_gguf_file(repo_id).await {
            Ok((_repo, file, _siblings)) => file,
            Err(e) => format!("ERROR: {}", e),
        }
    }

    async fn list_gguf_files(
        &self,
        client: &reqwest::Client,
        repo: &str,
    ) -> Result<Vec<String>, String> {
        let mut ggufs = Vec::new();

        let url = format!("https://huggingface.co/api/models/{}", repo);
        if let Ok(res) = client.get(&url).send().await {
            if res.status().is_success() {
                if let Ok(text) = res.text().await {
                    if let Ok(json) = serde_json::from_str::<serde_json::Value>(&text) {
                        if let Some(siblings) = json.get("siblings").and_then(|s| s.as_array()) {
                            for file in siblings {
                                if let Some(rfile) = file.get("rfilename").and_then(|f| f.as_str()) {
                                    if rfile.to_lowercase().ends_with(".gguf") {
                                        ggufs.push(rfile.to_string());
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        if ggufs.is_empty() {
            let tree_url = format!("https://huggingface.co/api/models/{}/tree/main", repo);
            if let Ok(res) = client.get(&tree_url).send().await {
                if res.status().is_success() {
                    if let Ok(text) = res.text().await {
                        if let Ok(json) = serde_json::from_str::<serde_json::Value>(&text) {
                            if let Some(arr) = json.as_array() {
                                for item in arr {
                                    if let Some(path) = item.get("path").and_then(|p| p.as_str()) {
                                        if path.to_lowercase().ends_with(".gguf") {
                                            ggufs.push(path.to_string());
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        Ok(ggufs)
    }

    pub fn set_active_gguf_path(&self, path: impl Into<String>) {
        let mut reg = ModelsRegistry::load();
        reg.active_model_path = Some(path.into());
        let _ = reg.save();
    }

    /// Most recently installed local GGUF path, if any.
    pub fn latest_gguf_path(&self) -> Option<PathBuf> {
        let reg = ModelsRegistry::load();
        if let Some(ref path) = reg.active_model_path {
            let p = PathBuf::from(path);
            if p.exists() {
                return Some(p);
            }
        }
        
        self.list_installed_entries()
            .into_iter()
            .rev()
            .find(|e| e.path.ends_with(".gguf") && Path::new(&e.path).exists())
            .map(|e| PathBuf::from(e.path))
    }

    /// Cancel any in-progress / stuck download lock.
    pub fn cancel_download(&self) -> String {
        Self::cleanup_stale_downloads();
        match DownloadLock::force_cancel() {
            Some(name) => format!("Cancelled download session for '{name}'. You can install another model now."),
            None => "No download lock to cancel.".into(),
        }
    }

    pub fn download_status(&self) -> String {
        DownloadLock::status_summary()
    }

    /// Download HF **GGUF** into `/tmp/hercules`, then promote to `~/.local/hercules/model`
    /// and register in `models.toml` on success.
    pub async fn download_hf_model(
        &self,
        repo_id: &str,
        filename: &str,
        shard_files: &[String],
        progress: Arc<Mutex<Option<f64>>>,
        logs: Arc<Mutex<Vec<String>>>,
    ) -> Result<PathBuf, String> {
        Self::cleanup_stale_downloads();

        if filename.starts_with("ERROR:") {
            return Err(filename.trim_start_matches("ERROR:").trim().to_string());
        }
        if !filename.to_lowercase().ends_with(".gguf") {
            return Err(format!(
                "Refusing to download non-GGUF file '{}'. \
                 llama.rs / llama.cpp need a .gguf model. \
                 Use a *-GGUF HuggingFace repo or Ollama pull instead.",
                filename
            ));
        }

        let clean_repo = repo_id
            .split('[')
            .next()
            .unwrap_or(repo_id)
            .trim()
            .trim_start_matches("HuggingFace: ")
            .trim();
        let base_name = filename
            .rsplit('/')
            .next()
            .unwrap_or(filename)
            .to_string();
        let model_name = format!("{}/{}", clean_repo, base_name);

        // Check if model is ALREADY installed on disk
        let installed_dest = models_dir().join(&base_name);
        if installed_dest.exists() {
            let meta_len = std::fs::metadata(&installed_dest).map(|m| m.len()).unwrap_or(0);
            if meta_len > 1_000_000 {
                let mut reg = ModelsRegistry::load();
                reg.upsert(InstalledModel {
                    name: model_name.clone(),
                    path: installed_dest.display().to_string(),
                    source: "huggingface".to_string(),
                    filename: base_name.clone(),
                    installed_at: now_unix(),
                    size_bytes: meta_len,
                });
                let _ = reg.save();

                if let Ok(mut l) = logs.lock() {
                    l.push(format!(
                        "[ALREADY INSTALLED] Model '{}' already exists at {} ({:.1} MB) — skipping download",
                        model_name,
                        installed_dest.display(),
                        meta_len as f64 / 1_000_000.0
                    ));
                }
                *progress.lock().unwrap() = Some(1.0);
                return Ok(installed_dest);
            }
        }

        // Log multi-part info
        if shard_files.len() > 1 {
            if let Ok(mut l) = logs.lock() {
                l.push(format!(
                    "[MULTI-PART] Downloading {} shard files for '{}'",
                    shard_files.len(),
                    base_name
                ));
            }
        }

        // Lock handling: stale / incomplete / different model → cancel previous and continue
        if let Some(existing) = DownloadLock::load() {
            let same = existing.model_name == model_name
                || existing.filename == base_name
                || existing.model_name.ends_with(&base_name);
            if existing.status == DownloadStatus::InProgress && !existing.is_stale() && same {
                // Same file still active — resume (do not error)
                if let Ok(mut l) = logs.lock() {
                    l.push(format!(
                        "[RESUME] Active session for '{}' (started {})",
                        existing.model_name,
                        format_unix(existing.time_started)
                    ));
                }
            } else if existing.status == DownloadStatus::InProgress
                && !existing.is_stale()
                && !same
            {
                // Different model requested while another runs → replace after short grace
                // (user started a new install on purpose)
                if let Ok(mut l) = logs.lock() {
                    l.push(format!(
                        "[CANCEL] Replacing prior download '{}' so '{}' can start",
                        existing.model_name, model_name
                    ));
                }
                let _ = std::fs::remove_dir_all(&existing.staging_dir);
                DownloadLock::remove_file();
            } else {
                // Stale / incomplete / finished lock left behind
                if existing.is_stale()
                    || existing.status == DownloadStatus::Incomplete
                    || existing.status == DownloadStatus::Complete
                {
                    if let Ok(mut l) = logs.lock() {
                        l.push(format!(
                            "[CLEANUP] Clearing old lock '{}' (status={:?}, stale={})",
                            existing.model_name,
                            existing.status,
                            existing.is_stale()
                        ));
                    }
                    if existing.status != DownloadStatus::Complete {
                        let _ = std::fs::remove_dir_all(&existing.staging_dir);
                    }
                    DownloadLock::remove_file();
                }
            }
        }
        let slug = slugify_model_name(&format!("{}_{}", clean_repo, base_name));

        let staging_root = tmp_hercules_dir();
        ensure_dir(&staging_root)?;
        let staging_dir = staging_root.join(&slug);
        ensure_dir(&staging_dir)?;

        let staging_file = staging_dir.join(&base_name);
        // Resume: keep existing partial bytes if same model was mid-download
        let existing_bytes = std::fs::metadata(&staging_file)
            .map(|m| m.len())
            .unwrap_or(0);

        let mut lock = DownloadLock::new(&model_name, "huggingface", &base_name, &staging_dir);
        if existing_bytes > 0 {
            lock.bytes_downloaded = existing_bytes;
            if let Ok(mut l) = logs.lock() {
                l.push(format!(
                    "[RESUME] Continuing from {:.1} MB already staged",
                    existing_bytes as f64 / 1_000_000.0
                ));
            }
        }
        lock.save()?;

        if let Ok(mut l) = logs.lock() {
            l.push(format!(
                "[SESSION] Download lock for '{}' | started {}",
                model_name,
                format_unix(lock.time_started)
            ));
            l.push(format!(
                "[SESSION] Staging: {}",
                staging_dir.display()
            ));
        }

        let actual_shards = if shard_files.is_empty() {
            vec![filename.to_string()]
        } else {
            shard_files.to_vec()
        };

        let client = reqwest::Client::builder()
            .user_agent("Hercules-CLI/1.0")
            .connect_timeout(Duration::from_secs(30))
            .timeout(Duration::from_secs(6 * 3600))
            .pool_idle_timeout(Duration::from_secs(90))
            .tcp_keepalive(Duration::from_secs(30))
            .redirect(reqwest::redirect::Policy::limited(10))
            .build()
            .map_err(|e| {
                lock.mark_incomplete(e.to_string());
                e.to_string()
            })?;

        const MAX_ATTEMPTS: u32 = 8;
        let mut total_downloaded_all = 0u64;

        for (shard_idx, shard_name) in actual_shards.iter().enumerate() {
            let shard_base = shard_name.rsplit('/').next().unwrap_or(shard_name).to_string();
            let shard_url = format!(
                "https://huggingface.co/{}/resolve/main/{}",
                clean_repo, shard_name
            );
            let shard_staging = staging_dir.join(&shard_base);

            if let Ok(mut l) = logs.lock() {
                if actual_shards.len() > 1 {
                    l.push(format!("[MULTI-PART] Starting shard {}/{} -> {}", shard_idx + 1, actual_shards.len(), shard_base));
                }
                l.push(format!("[HTTP] Connecting to {}", shard_url));
            }

            let mut downloaded = std::fs::metadata(&shard_staging)
                .map(|m| m.len())
                .unwrap_or(0);

            let mut total_size: Option<u64> = None;
            let mut last_logged_pct = if downloaded > 0 { -1.0f64 } else { -1.0f64 };
            let mut last_lock_write = std::time::Instant::now();

            for attempt in 1..=MAX_ATTEMPTS {
                let mut file = match std::fs::OpenOptions::new()
                    .create(true)
                    .write(true)
                    .append(true)
                    .open(&shard_staging)
                {
                    Ok(f) => f,
                    Err(e) => {
                        let msg = format!("Cannot open staging file {}: {}", shard_base, e);
                        lock.mark_incomplete(&msg);
                        return Err(msg);
                    }
                };

                downloaded = std::fs::metadata(&shard_staging)
                    .map(|m| m.len())
                    .unwrap_or(downloaded);

                let mut req = client.get(&shard_url);
                if downloaded > 0 {
                    req = req.header("Range", format!("bytes={}-", downloaded));
                    if let Ok(mut l) = logs.lock() {
                        l.push(format!(
                            "[HTTP] Attempt {}/{} — Range resume from byte {}",
                            attempt, MAX_ATTEMPTS, downloaded
                        ));
                    }
                } else if attempt > 1 {
                    if let Ok(mut l) = logs.lock() {
                        l.push(format!(
                            "[HTTP] Attempt {}/{} — restarting stream",
                            attempt, MAX_ATTEMPTS
                        ));
                    }
                }

                let res = match req.send().await {
                    Ok(r) => r,
                    Err(e) => {
                        let msg = format!("Network failure (attempt {}): {}", attempt, e);
                        if let Ok(mut l) = logs.lock() {
                            l.push(format!("[WARN] {}", msg));
                        }
                        lock.touch_progress(total_downloaded_all + downloaded, lock.bytes_total);
                        if attempt == MAX_ATTEMPTS {
                            lock.mark_incomplete(&msg);
                            *progress.lock().unwrap() = None;
                            return Err(msg);
                        }
                        tokio::time::sleep(Duration::from_secs(2u64.pow(attempt.min(4)))).await;
                        continue;
                    }
                };

                let status = res.status();
                if !(status.is_success() || status.as_u16() == 206) {
                    let err_msg = format!(
                        "[HTTP ERROR {}] Download failed for '{}'",
                        status, shard_url
                    );
                    if let Ok(mut l) = logs.lock() {
                        l.push(err_msg.clone());
                    }
                    if attempt == MAX_ATTEMPTS {
                        lock.mark_incomplete(&err_msg);
                        *progress.lock().unwrap() = None;
                        return Err(err_msg);
                    }
                    tokio::time::sleep(Duration::from_secs(2)).await;
                    continue;
                }

                if let Some(cr) = res.headers().get("content-range").and_then(|v| v.to_str().ok()) {
                    if let Some(total_s) = cr.split('/').nth(1) {
                        if let Ok(t) = total_s.parse::<u64>() {
                            total_size = Some(t);
                        }
                    }
                } else if let Some(cl) = res.content_length() {
                    total_size = Some(if status.as_u16() == 206 {
                        downloaded + cl
                    } else {
                        cl
                    });
                }

                if attempt == 1 || total_size.is_some() {
                    if let Ok(mut l) = logs.lock() {
                        let mb = total_size.unwrap_or(0) as f64 / 1_000_000.0;
                        l.push(format!(
                            "[HTTP] {} | total ~{:.2} MB | have {:.1} MB",
                            status,
                            mb,
                            downloaded as f64 / 1_000_000.0
                        ));
                    }
                }

                lock.touch_progress(total_downloaded_all + downloaded, lock.bytes_total);
                lock.status = DownloadStatus::InProgress;
                let _ = lock.save();

                let mut stream = res.bytes_stream();
                let mut stream_ok = true;
                let mut window_bytes = 0u64;
                let mut window_start = std::time::Instant::now();

                while let Some(item) = stream.next().await {
                    if last_lock_write.elapsed() >= Duration::from_secs(5) {
                        lock.touch_progress(total_downloaded_all + downloaded, lock.bytes_total);
                        last_lock_write = std::time::Instant::now();
                    }

                    let chunk = match item {
                        Ok(c) => c,
                        Err(e) => {
                            if let Ok(mut l) = logs.lock() {
                                l.push(format!(
                                    "[WARN] Stream interrupted at {:.1} MB: {} — will resume",
                                    downloaded as f64 / 1_000_000.0,
                                    e
                                ));
                            }
                            let _ = file.flush();
                            lock.touch_progress(total_downloaded_all + downloaded, lock.bytes_total);
                            lock.status = DownloadStatus::Incomplete;
                            lock.error = Some(format!("stream: {}", e));
                            let _ = lock.save();
                            stream_ok = false;
                            break;
                        }
                    };

                    if let Err(e) = file.write_all(&chunk) {
                        let msg = format!("Write error: {}", e);
                        lock.mark_incomplete(&msg);
                        *progress.lock().unwrap() = None;
                        return Err(msg);
                    }

                    downloaded += chunk.len() as u64;
                    window_bytes += chunk.len() as u64;

                    let total_f = total_size.unwrap_or(downloaded.max(1)) as f64;
                    let ratio = (downloaded as f64 / total_f).min(1.0);
                    let pct = ratio * 100.0;

                    let overall_ratio = if actual_shards.len() > 1 {
                        ((shard_idx as f64) + ratio) / (actual_shards.len() as f64)
                    } else {
                        ratio
                    };

                    if pct - last_logged_pct >= 1.0 {
                        last_logged_pct = pct;
                        lock.touch_progress(total_downloaded_all + downloaded, lock.bytes_total);
                        last_lock_write = std::time::Instant::now();
                        let win_secs = window_start.elapsed().as_secs_f64().max(0.05);
                        let inst_mbps = (window_bytes as f64 / 1_000_000.0) / win_secs;
                        window_bytes = 0;
                        window_start = std::time::Instant::now();
                        if let Ok(mut l) = logs.lock() {
                            let prefix = if actual_shards.len() > 1 {
                                format!("[SHARD {}/{}] ", shard_idx + 1, actual_shards.len())
                            } else {
                                "[STREAM] ".to_string()
                            };
                            l.push(format!(
                                "{}{:.1} MB/s (live) | {:.1}MB / {:.1}MB ({:.1}%)",
                                prefix,
                                inst_mbps,
                                downloaded as f64 / 1_000_000.0,
                                total_f / 1_000_000.0,
                                pct
                            ));
                        }
                    }
                    *progress.lock().unwrap() = Some(overall_ratio);
                }

                let _ = file.flush();
                drop(file);

                if !stream_ok {
                    if attempt == MAX_ATTEMPTS {
                        let msg = format!(
                            "Download failed after {} attempts (unstable network). Partial: {:.1} MB saved — run install again to resume.",
                            MAX_ATTEMPTS,
                            downloaded as f64 / 1_000_000.0
                        );
                        if let Ok(mut l) = logs.lock() {
                            l.push(format!("[ERROR] {}", msg));
                            l.push("[SESSION] Marked incomplete — finish time not set".to_string());
                        }
                        *progress.lock().unwrap() = None;
                        return Err(msg);
                    }
                    tokio::time::sleep(Duration::from_secs(2u64.pow(attempt.min(4)))).await;
                    continue;
                }

                if let Some(total) = total_size {
                    if downloaded + 1024 < total {
                        if let Ok(mut l) = logs.lock() {
                            l.push(format!(
                                "[WARN] Short read {:.1}/{:.1} MB — resume attempt",
                                downloaded as f64 / 1_000_000.0,
                                total as f64 / 1_000_000.0
                            ));
                        }
                        lock.touch_progress(total_downloaded_all + downloaded, lock.bytes_total);
                        if attempt == MAX_ATTEMPTS {
                            let msg = format!("Incomplete download: got {} of {} bytes", downloaded, total);
                            lock.status = DownloadStatus::Incomplete;
                            lock.error = Some(msg.clone());
                            lock.bytes_downloaded = total_downloaded_all + downloaded;
                            let _ = lock.save();
                            *progress.lock().unwrap() = None;
                            return Err(msg);
                        }
                        tokio::time::sleep(Duration::from_secs(2)).await;
                        continue;
                    }
                }

                break;
            }

            total_downloaded_all += downloaded;
        }

        let downloaded = total_downloaded_all;
        // --- Success: mark complete, then promote to ~/.local/hercules/model ---
        lock.bytes_downloaded = downloaded;
        lock.mark_complete();

        if let Ok(mut l) = logs.lock() {
            l.push(format!(
                "[SESSION] Download complete | started {} | finished {}",
                format_unix(lock.time_started),
                format_unix(lock.time_finished.unwrap_or(now_unix()))
            ));
        }

        let mut first_installed_path = None;

        for (shard_idx, shard_name) in actual_shards.iter().enumerate() {
            let shard_base = shard_name.rsplit('/').next().unwrap_or(shard_name).to_string();
            let shard_staging = staging_dir.join(&shard_base);
            let size = std::fs::metadata(&shard_staging).map(|m| m.len()).unwrap_or(0);

            let installed_path = match self.promote_to_local(
                &shard_staging,
                &model_name,
                "huggingface",
                &shard_base,
                size,
                &logs,
            ) {
                Ok(p) => p,
                Err(e) => {
                    if let Ok(mut l) = logs.lock() {
                        l.push(format!("[ERROR] Promote failed for {}: {}", shard_base, e));
                    }
                    *progress.lock().unwrap() = None;
                    return Err(e);
                }
            };

            if shard_idx == 0 {
                first_installed_path = Some(installed_path);
            }
        }

        let installed_path = first_installed_path.unwrap();

        // Cleanup staging + lock after successful install
        let _ = std::fs::remove_dir_all(&staging_dir);
        DownloadLock::remove_file();

        if filename.ends_with(".gguf") {
            let model_alias = base_name
                .trim_end_matches(".gguf")
                .to_lowercase()
                .replace('.', "-");
            let _ = self
                .create_ollama_model_from_gguf(&model_alias, &installed_path, logs.clone())
                .await;
        }

        *progress.lock().unwrap() = Some(1.0);
        if let Ok(mut l) = logs.lock() {
            l.push(format!(
                "[SUCCESS] Installed to {} | registered in {}",
                installed_path.display(),
                models_toml_path().display()
            ));
        }

        Ok(installed_path)
    }

    /// Move completed staging file into `~/.local/hercules/model` and update `models.toml`.
    fn promote_to_local(
        &self,
        staging_file: &Path,
        model_name: &str,
        source: &str,
        filename: &str,
        size_bytes: u64,
        logs: &Arc<Mutex<Vec<String>>>,
    ) -> Result<PathBuf, String> {
        ensure_dir(&models_dir())?;
        let dest = models_dir().join(filename);

        // Prefer rename (same FS); fall back to copy+remove
        let install_result = std::fs::rename(staging_file, &dest).or_else(|_| {
            std::fs::copy(staging_file, &dest).map(|_| {
                let _ = std::fs::remove_file(staging_file);
            })
        });

        install_result.map_err(|e| format!("Failed to move model to install dir: {}", e))?;

        let entry = InstalledModel {
            name: model_name.to_string(),
            path: dest.display().to_string(),
            source: source.to_string(),
            filename: filename.to_string(),
            installed_at: now_unix(),
            size_bytes,
        };

        let mut reg = ModelsRegistry::load();
        reg.upsert(entry);
        reg.save()?;

        if let Ok(mut l) = logs.lock() {
            l.push(format!(
                "[INSTALL] Moved to {} | models.toml updated",
                dest.display()
            ));
        }

        Ok(dest)
    }

    pub async fn create_ollama_model_from_gguf(
        &self,
        model_alias: &str,
        gguf_path: &Path,
        logs: Arc<Mutex<Vec<String>>>,
    ) -> Result<(), String> {
        if let Ok(mut l) = logs.lock() {
            l.push(format!(
                "[OLLAMA SERVE] Registering GGUF model into Ollama daemon: {}",
                model_alias
            ));
        }

        let request = ollama_rs::models::create::CreateModelRequest::new(model_alias.to_string())
            .from_model(gguf_path.to_string_lossy().to_string());

        match self.ollama.create_model(request).await {
            Ok(_) => {
                if let Ok(mut l) = logs.lock() {
                    l.push(format!(
                        "[OLLAMA SERVE SUCCESS] Model '{}' created and serving in Ollama!",
                        model_alias
                    ));
                }
                Ok(())
            }
            Err(e) => Err(e.to_string()),
        }
    }

    pub async fn download_ollama_model(
        &self,
        model_name: &str,
        progress: Arc<Mutex<Option<f64>>>,
        logs: Arc<Mutex<Vec<String>>>,
    ) -> Result<(), String> {
        // Track session for Ollama pulls as well (no file staging, but same lock semantics)
        Self::cleanup_stale_downloads();
        let staging_dir = tmp_hercules_dir().join(slugify_model_name(model_name));
        ensure_dir(&tmp_hercules_dir())?;
        let mut lock = DownloadLock::new(model_name, "ollama", model_name, &staging_dir);
        lock.save()?;

        if let Ok(mut l) = logs.lock() {
            l.push(format!(
                "[SESSION] Ollama pull lock for '{}' | started {}",
                model_name,
                format_unix(lock.time_started)
            ));
            l.push(format!("[OLLAMA] Pulling model from registry: {}", model_name));
        }
        *progress.lock().unwrap() = Some(0.1);

        let mut stream = match self
            .ollama
            .pull_model_stream(model_name.to_string(), false)
            .await
        {
            Ok(s) => s,
            Err(e) => {
                let msg = e.to_string();
                if let Ok(mut l) = logs.lock() {
                    l.push(format!("[OLLAMA ERROR] {}", msg));
                    l.push("[SESSION] Marked incomplete — finish time not set".to_string());
                }
                lock.mark_incomplete(&msg);
                return Err(msg);
            }
        };

        while let Some(res) = stream.next().await {
            match res {
                Ok(status) => {
                    if let Some(total) = status.total {
                        if let Some(completed) = status.completed {
                            let ratio = (completed as f64) / (total.max(1) as f64);
                            *progress.lock().unwrap() = Some(ratio.min(1.0));
                            lock.touch_progress(completed, Some(total));
                        }
                    } else {
                        lock.touch_progress(lock.bytes_downloaded, None);
                    }
                }
                Err(e) => {
                    let msg = e.to_string();
                    if let Ok(mut l) = logs.lock() {
                        l.push(format!("[OLLAMA ERROR] {}", msg));
                        l.push("[SESSION] Marked incomplete — finish time not set".to_string());
                    }
                    lock.status = DownloadStatus::Incomplete;
                    lock.error = Some(msg.clone());
                    let _ = lock.save();
                    *progress.lock().unwrap() = None;
                    return Err(msg);
                }
            }
        }

        lock.mark_complete();
        // Ollama stores its own blobs; still record in models.toml as ollama source
        let mut reg = ModelsRegistry::load();
        reg.upsert(InstalledModel {
            name: model_name.to_string(),
            path: format!("ollama://{}", model_name),
            source: "ollama".to_string(),
            filename: model_name.to_string(),
            installed_at: now_unix(),
            size_bytes: lock.bytes_downloaded,
        });
        let _ = reg.save();
        DownloadLock::remove_file();
        let _ = std::fs::remove_dir_all(&staging_dir);

        *progress.lock().unwrap() = Some(1.0);
        if let Ok(mut l) = logs.lock() {
            l.push(format!(
                "[OLLAMA SUCCESS] Model {} installed | finished {}",
                model_name,
                format_unix(lock.time_finished.unwrap_or(now_unix()))
            ));
            l.push(format!(
                "[INSTALL] Registered in {}",
                models_toml_path().display()
            ));
        }
        Ok(())
    }

    /// Delete a local Hercules model by display name or registry name.
    pub fn delete_local_model(&self, name_or_display: &str) -> Result<(), String> {
        let mut reg = ModelsRegistry::load();
        let key = name_or_display
            .replace("Local GGUF:", "")
            .split('(')
            .next()
            .unwrap_or(name_or_display)
            .trim()
            .to_string();

        let entry = reg
            .models
            .iter()
            .find(|m| m.name == key || m.name.contains(&key) || name_or_display.contains(&m.path))
            .cloned();

        if let Some(entry) = entry {
            if entry.source != "ollama" && !entry.path.starts_with("ollama://") {
                let p = PathBuf::from(&entry.path);
                if p.exists() {
                    let _ = std::fs::remove_file(&p);
                }
            }
            reg.remove_by_name(&entry.name);
            reg.save()?;
            Ok(())
        } else {
            Err(format!("Model not found in registry: {}", name_or_display))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_manager_creation() {
        let _manager = ModelManager::new();
        assert!(true);
    }

    #[test]
    fn test_slugify() {
        assert_eq!(slugify_model_name("org/model:q4"), "org_model_q4");
    }

    #[test]
    fn test_lock_incomplete_does_not_set_finished() {
        let dir = std::env::temp_dir().join("hercules_test_lock");
        let _ = std::fs::create_dir_all(&dir);
        let mut lock = DownloadLock::new("test-model", "test", "f.gguf", &dir);
        assert!(lock.time_finished.is_none());
        lock.mark_incomplete("network down");
        assert_eq!(lock.status, DownloadStatus::Incomplete);
        assert!(lock.time_finished.is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_stale_detection() {
        let dir = std::env::temp_dir().join("hercules_test_stale");
        let mut lock = DownloadLock::new("m", "t", "f.gguf", &dir);
        lock.time_updated = now_unix().saturating_sub(STALE_DOWNLOAD_SECS + 5);
        assert!(lock.is_stale());
        lock.time_updated = now_unix();
        lock.status = DownloadStatus::InProgress;
        assert!(!lock.is_stale());
    }

    #[test]
    fn test_registry_upsert() {
        let mut reg = ModelsRegistry::default();
        reg.upsert(InstalledModel {
            name: "a".into(),
            path: "/p/a".into(),
            source: "hf".into(),
            filename: "a.gguf".into(),
            installed_at: 1,
            size_bytes: 10,
        });
        reg.upsert(InstalledModel {
            name: "a".into(),
            path: "/p/a2".into(),
            source: "hf".into(),
            filename: "a.gguf".into(),
            installed_at: 2,
            size_bytes: 20,
        });
        assert_eq!(reg.models.len(), 1);
        assert_eq!(reg.models[0].path, "/p/a2");
    }
}
