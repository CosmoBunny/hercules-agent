//! llama.cpp backend via `llama-completion` / `llama-cli` / `llama-server`.
//!
//! Important: modern `llama-cli` prints a full interactive TUI to stdout.
//! We prefer **`llama-completion`** for one-shot generation and aggressively
//! strip any residual chrome so Hercules chat stays clean.

use crate::llama::http::HttpInferenceClient;
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

#[derive(Debug, Clone)]
pub enum LlamaCppMode {
    Cli { model_path: PathBuf },
    Server {
        endpoint: String,
        model_name: String,
    },
}

#[derive(Clone)]
pub struct LlamaCppRuntime {
    pub mode: LlamaCppMode,
    pub extra_args: Vec<String>,
    pub n_predict: usize,
    pub temperature: f32,
}

impl LlamaCppRuntime {
    pub fn cli(model_path: impl Into<PathBuf>) -> Self {
        Self {
            mode: LlamaCppMode::Cli {
                model_path: model_path.into(),
            },
            extra_args: Vec::new(),
            n_predict: 128,
            temperature: 0.7,
        }
    }

    pub fn server(endpoint: impl Into<String>, model_name: impl Into<String>) -> Self {
        Self {
            mode: LlamaCppMode::Server {
                endpoint: endpoint.into(),
                model_name: model_name.into(),
            },
            extra_args: Vec::new(),
            n_predict: 256,
            temperature: 0.7,
        }
    }

    pub fn model_path(&self) -> Option<PathBuf> {
        match &self.mode {
            LlamaCppMode::Cli { model_path } => Some(model_path.clone()),
            _ => None,
        }
    }

    pub fn name(&self) -> String {
        match &self.mode {
            LlamaCppMode::Cli { model_path } => {
                format!(
                    "llama.cpp ({})",
                    model_path
                        .file_name()
                        .map(|s| s.to_string_lossy().to_string())
                        .unwrap_or_else(|| model_path.display().to_string())
                )
            }
            LlamaCppMode::Server { endpoint, .. } => {
                format!("llama.cpp server ({})", endpoint)
            }
        }
    }

    pub async fn generate(&self, prompt: &str) -> Result<String, String> {
        let target = Arc::new(Mutex::new(String::new()));
        let flag = Arc::new(Mutex::new(true));
        self.generate_stream(prompt, target, flag).await
    }

    pub async fn generate_stream(
        &self,
        prompt: &str,
        stream_target: Arc<Mutex<String>>,
        is_generating: Arc<Mutex<bool>>,
    ) -> Result<String, String> {
        match &self.mode {
            LlamaCppMode::Server {
                endpoint,
                model_name,
            } => {
                let client = HttpInferenceClient::new(endpoint.clone(), model_name.clone());
                client
                    .generate_stream(prompt, stream_target, is_generating)
                    .await
            }
            LlamaCppMode::Cli { model_path } => {
                // Warm llama-server: load once, then every prompt is a fast HTTP chat turn
                // with the full Hercules system instruction.
                if let Ok(mut t) = stream_target.lock() {
                    if crate::llama::server::managed_server_info()
                        .map(|(p, _, _)| p != *model_path)
                        .unwrap_or(true)
                    {
                        t.push_str(
                            "Starting llama-server (loads GGUF once; GPU layers via -ngl if available)…\n",
                        );
                    }
                }

                let (base_url, model_name) =
                    crate::llama::server::ensure_server_for_model(model_path).await?;

                if let Ok(mut t) = stream_target.lock() {
                    // Clear loading banner before tokens arrive
                    if t.contains("Starting llama-server") || t.contains("Loading model") {
                        t.clear();
                    }
                }

                let client = HttpInferenceClient::new(base_url, model_name);
                // `prompt` may be full You:/Agent: history — HTTP client expands system + turns
                client
                    .generate_stream(prompt, stream_target, is_generating)
                    .await
            }
        }
    }
}

/// Pull the last `You: …` line (or whole string if none).
fn extract_user_utterance(prompt: &str) -> String {
    for line in prompt.lines().rev() {
        let t = line.trim();
        if let Some(rest) = t.strip_prefix("You: ") {
            return rest.to_string();
        }
    }
    // Drop System: status noise if joined history was passed
    let cleaned: Vec<&str> = prompt
        .lines()
        .filter(|l| {
            let t = l.trim();
            !t.starts_with("System:")
                && !t.starts_with("Agent:")
                && !t.is_empty()
        })
        .collect();
    if cleaned.is_empty() {
        prompt.trim().to_string()
    } else {
        cleaned.join("\n")
    }
}

/// Prefer completion binary (clean one-shot), then llama-cli.
pub fn find_llama_binary() -> Option<PathBuf> {
    find_named_binary(&[
        "llama-completion",
        "llama-cli",
        "llama",
        "main",
    ])
}

fn find_named_binary(names: &[&str]) -> Option<PathBuf> {
    for name in names {
        if let Ok(output) = Command::new("which").arg(name).output() {
            if output.status.success() {
                let p = String::from_utf8_lossy(&output.stdout).trim().to_string();
                if !p.is_empty() && Path::new(&p).is_file() {
                    return Some(PathBuf::from(p));
                }
            }
        }
    }
    for name in names {
        for prefix in ["/opt/llama.cpp", "/usr/local/bin", "/usr/bin"] {
            let path = PathBuf::from(prefix).join(name);
            if path.is_file() {
                return Some(path);
            }
        }
        if let Some(home) = std::env::var_os("HOME").or_else(|| std::env::var_os("USERPROFILE")) {
            for rel in [format!(".local/bin/{}", name), format!(".cargo/bin/{}", name)] {
                let path = PathBuf::from(&home).join(rel);
                if path.is_file() {
                    return Some(path);
                }
            }
        }
    }
    None
}

fn run_llama_completion(
    model_path: &Path,
    user_prompt: &str,
    n_predict: usize,
    temperature: f32,
    extra_args: &[String],
    stream_target: Option<Arc<Mutex<String>>>,
    is_generating: Option<Arc<Mutex<bool>>>,
) -> Result<String, String> {
    // Prefer llama-completion; fall back to llama-cli with strict flags
    let bin = find_named_binary(&["llama-completion", "llama-cli"]).ok_or_else(|| {
        "[llama.cpp] No `llama-completion` or `llama-cli` found on PATH / $HOME/.local/bin / /opt/llama.cpp."
            .to_string()
    })?;

    if !model_path.exists() {
        return Err(format!(
            "[llama.cpp] Model file not found: {}",
            model_path.display()
        ));
    }

    let is_completion_bin = bin
        .file_name()
        .and_then(|s| s.to_str())
        .map(|s| s.contains("completion"))
        .unwrap_or(false);

    let system = "You are Hercules, a local coding agent. Be concise. Use tool tags when needed.";

    let mut cmd = Command::new(&bin);
    cmd.arg("-m")
        .arg(model_path)
        .arg("-n")
        .arg(n_predict.to_string())
        .arg("--temp")
        .arg(temperature.to_string())
        .arg("-c")
        .arg("2048")
        .arg("-t")
        .arg(
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(4)
                .to_string(),
        )
        .arg("-sys")
        .arg(system)
        .arg("-p")
        .arg(user_prompt)
        // NOTE: do not use --log-disable — on some builds it also blanks stdout.
        .stdout(Stdio::piped())
        .stderr(Stdio::null()) // keep load spam out of Hercules chat
        .stdin(Stdio::null()); // force EOF so interactive mode exits

    if !is_completion_bin {
        // llama-cli needs these or it stays in chat TUI forever
        cmd.arg("--no-conversation")
            .arg("--single-turn")
            .arg("--no-display-prompt")
            .arg("--simple-io");
    }

    for a in extra_args {
        cmd.arg(a);
    }

    let start = Instant::now();
    let mut child = cmd
        .spawn()
        .map_err(|e| format!("[llama.cpp] Failed to spawn {}: {}", bin.display(), e))?;

    let stdout = child.stdout.take().ok_or("[llama.cpp] No stdout")?;
    let mut reader = BufReader::new(stdout);
    let mut raw = String::new();
    let mut buf = [0u8; 4096];

    // Byte-stream so we don't block on newlines forever during spinner
    loop {
        if let Some(ref flag) = is_generating {
            if let Ok(g) = flag.lock() {
                if !*g {
                    let _ = child.kill();
                    return Err("[Generation Cancelled by User (CTRL+C)]".into());
                }
            }
        }
        match reader.get_mut().read(&mut buf) {
            Ok(0) => break,
            Ok(n) => {
                let chunk = String::from_utf8_lossy(&buf[..n]);
                raw.push_str(&chunk);
            }
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(e) => {
                let _ = child.kill();
                return Err(format!("[llama.cpp] Read error: {}", e));
            }
        }
        if start.elapsed() > Duration::from_secs(600) {
            let _ = child.kill();
            return Err("[llama.cpp] Timed out after 10 minutes (model load can take ~30s on CPU)".into());
        }
    }

    let _ = child.wait();

    let reply = extract_assistant_text(&raw);
    if reply.is_empty() {
        // Last resort: try again with minimal llama-cli flags via prompt file
        return Err(format!(
            "[llama.cpp] No assistant text after {} ms.\n\
             Raw (trimmed): {}\n\
             Tip: first load is slow (~25s). Prefer llama-server for keep-alive.",
            start.elapsed().as_millis(),
            raw.chars().take(200).collect::<String>().replace('\n', " ")
        ));
    }

    if let Some(ref target) = stream_target {
        if let Ok(mut t) = target.lock() {
            *t = reply.clone();
        }
    }
    Ok(reply)
}

/// Strip llama-cli / llama-completion chrome; return only model answer.
fn extract_assistant_text(raw: &str) -> String {
    let text = strip_ansi(raw);

    // 1) Drop chrome / role labels / echoed `> prompt` lines (keep content after)
    let mut body_lines: Vec<String> = Vec::new();
    let mut seen_assistant_role = false;
    let mut after_cli_prompt = false;

    for line in text.lines() {
        let t = line.trim();
        if is_chrome_line(t) {
            continue;
        }
        // completion tool role markers
        if t == "user" || t == "system" {
            seen_assistant_role = false;
            continue;
        }
        if t == "assistant" {
            seen_assistant_role = true;
            body_lines.clear(); // only keep assistant section
            continue;
        }
        // llama-cli echoes `> <user text>` then answer
        if t.starts_with("> ") {
            after_cli_prompt = true;
            body_lines.clear();
            continue;
        }
        if seen_assistant_role || after_cli_prompt {
            body_lines.push(line.to_string());
        }
    }

    // 2) If we never saw role/prompt markers, fall back to last non-chrome paragraph
    if body_lines.is_empty() {
        for line in text.lines() {
            let t = line.trim();
            if t.is_empty() || is_chrome_line(t) || t.starts_with("> ") {
                continue;
            }
            if t == "user" || t == "assistant" || t == "system" {
                continue;
            }
            body_lines.push(line.to_string());
        }
    }

    while body_lines
        .last()
        .map(|l| {
            let t = l.trim();
            t.is_empty() || t.contains("EOF by user") || t.contains("Exiting")
        })
        .unwrap_or(false)
    {
        body_lines.pop();
    }

    body_lines.join("\n").trim().to_string()
}

fn is_chrome_line(t: &str) -> bool {
    if t.is_empty() {
        return false;
    }
    t.starts_with("Loading model")
        || t.starts_with("build")
        || t.starts_with("model")
        || t.starts_with("ftype")
        || t.starts_with("modalities")
        || t.starts_with("using custom")
        || t.starts_with("available commands")
        || t.starts_with("/exit")
        || t.starts_with("/regen")
        || t.starts_with("/clear")
        || t.starts_with("/read")
        || t.starts_with("/glob")
        || t.starts_with("Exiting")
        || t.contains("t/s |")
        || t.contains("EOF by user")
        || t.starts_with("== Running")
        || t.starts_with("- Press")
        || t.starts_with("- To return")
        || t.starts_with("- If you want")
        || t.starts_with("- Not using")
        || t.starts_with("-----")
        || t.starts_with("common_perf")
        || t.starts_with("llama_")
        || t.starts_with("ggml_")
        || t.chars().all(|c| {
            matches!(
                c,
                '█' | '▄' | '▀' | ' ' | '│' | '─' | '┌' | '┐' | '└' | '┘' | '▒' | '░' | '|'
                    | '/' | '-' | '\\'
            )
        })
}

fn strip_ansi(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\u{1b}' {
            // ESC [
            if chars.peek() == Some(&'[') {
                chars.next();
                for x in chars.by_ref() {
                    if x.is_ascii_alphabetic() {
                        break;
                    }
                }
            }
            continue;
        }
        // strip spinner backspaces/carriage returns partially
        if c == '\r' {
            continue;
        }
        out.push(c);
    }
    out
}

pub fn install_hint() -> &'static str {
    "llama.cpp: install from https://github.com/ggml-org/llama.cpp — \
     put `llama-completion` or `llama-cli` on PATH."
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_user_from_history() {
        let p = "System: Welcome\n\nYou: hello there\n\nAgent: hi";
        assert_eq!(extract_user_utterance(p), "hello there");
    }

    #[test]
    fn extract_assistant_from_completion_dump() {
        let raw = r#"
user
Hello, my name is
assistant
Hello! How can I assist you today?

> EOF by user
"#;
        let a = extract_assistant_text(raw);
        assert!(a.contains("How can I assist"), "got: {:?}", a);
        assert!(!a.contains("EOF"));
        assert!(!a.contains("user"));
    }

    #[test]
    fn extract_strips_cli_banner() {
        let raw = r#"
Loading model...
build      : b10107
model      : /tmp/x.gguf
ftype      : Q4_K - Medium
available commands:
  /exit or Ctrl+C     stop or exit

> hello
Hello! How can I assist you today?

[ Prompt: 12.5 t/s | Generation: 4.2 t/s ]
Exiting...
"#;
        let a = extract_assistant_text(raw);
        assert!(a.contains("assist"), "got: {:?}", a);
        assert!(!a.contains("Loading"));
        assert!(!a.contains("/exit"));
        assert!(!a.contains("t/s"));
    }
}
