//! Generic Transformers/PyTorch backend (Phase 4).
//!
//! Executes local SafeTensors/PyTorch HF models — especially architectures
//! llama.cpp rejects — through an isolated Python worker process. The Rust
//! side owns spawn/handshake/streaming/cancel/lifecycle; Python owns
//! transformers/torch loading and generation. Boundary: structured argv +
//! versioned JSONL (stdout), diagnostics on stderr. Never a shell, never
//! `python -c`, never embedded Python.
//!
//! Cancellation is two-layer, reusing the established architecture:
//! cooperative `{"type":"cancel"}` plus process-group kill. The child is
//! spawned with `kill_on_drop(true)`, so the App-level `select!` against
//! the run token dropping this future terminates the OS process even if
//! the worker ignores cooperation.

use super::error::ModelError;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

pub const PROTOCOL_VERSION: u32 = 1;
pub const DEFAULT_DEVICE: &str = "auto";
const HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(120);
const LINE_TIMEOUT: Duration = Duration::from_secs(300);

/// Outbound worker request.
#[derive(Debug, serde::Serialize)]
#[serde(tag = "type", rename_all = "lowercase")]
enum WorkerRequest<'a> {
    Generate {
        request_id: &'a str,
        prompt: &'a str,
        max_new_tokens: u32,
        temperature: Option<f32>,
    },
    Cancel {
        request_id: &'a str,
    },
    Shutdown,
}

/// Inbound worker message (stdout JSONL only).
#[derive(Debug, serde::Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
enum WorkerMessage {
    Hello {
        protocol_version: u32,
    },
    Ready {
        #[allow(dead_code)]
        protocol_version: u32,
        architecture: Option<String>,
        model_type: Option<String>,
        device: Option<String>,
        #[allow(dead_code)]
        device_name: Option<String>,
    },
    Deps {
        transformers: Option<String>,
        torch: Option<String>,
        cuda_available: Option<bool>,
    },
    Token {
        request_id: String,
        text: String,
    },
    Done {
        request_id: String,
    },
    Cancelled {
        request_id: String,
    },
    Error {
        request_id: Option<String>,
        code: Option<String>,
        message: Option<String>,
    },
    Bye {
        #[allow(dead_code)]
        protocol_version: Option<u32>,
    },
    #[serde(other)]
    Unknown,
}

/// Parsed `ready` handshake.
#[derive(Debug, Clone)]
pub struct WorkerReady {
    pub architecture: Option<String>,
    pub model_type: Option<String>,
    pub device: String,
}

/// Dependency/device report from `--check-deps`.
#[derive(Debug, Clone)]
pub struct DependencyReport {
    pub python: String,
    pub transformers_version: Option<String>,
    pub torch_version: Option<String>,
    pub cuda_available: bool,
}

/// Locate the shipped worker script deterministically (never CWD):
/// explicit override → executable dir → dev manifest dir.
pub fn resolve_worker_script(explicit: Option<&str>) -> Option<PathBuf> {
    if let Some(p) = explicit {
        let pb = PathBuf::from(p);
        if pb.is_file() {
            return Some(pb);
        }
        return None;
    }
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            for cand in [
                dir.join("resources/transformers_worker.py"),
                dir.join("transformers_worker.py"),
            ] {
                if cand.is_file() {
                    return Some(cand);
                }
            }
        }
    }
    let dev = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("resources/transformers_worker.py");
    if dev.is_file() {
        return Some(dev);
    }
    None
}

/// Cached dependency probe per python path: availability() must stay cheap.
static PROBE_CACHE: std::sync::OnceLock<
    std::sync::Mutex<std::collections::HashMap<String, Result<DependencyReport, String>>>,
> = std::sync::OnceLock::new();

pub(crate) fn probe_cached(python: &str) -> Result<DependencyReport, ModelError> {
    let cache = PROBE_CACHE.get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()));
    if let Ok(guard) = cache.lock() {
        if let Some(cached) = guard.get(python) {
            return cached.clone().map_err(|e| ModelError::DependencyMissing {
                backend: "Transformers".to_string(),
                dependency: e,
            });
        }
    }
    let result = probe_dependencies(python);
    if let Ok(mut guard) = cache.lock() {
        guard.insert(python.to_string(), result.clone().map_err(|e| e.message()));
    }
    result
}

/// Run `worker.py --check-deps` with structured argv (no shell).
/// Useful errors, never bare BackendUnavailable.
pub fn probe_dependencies(python: &str) -> Result<DependencyReport, ModelError> {
    let worker = resolve_worker_script(None).ok_or_else(|| ModelError::DependencyMissing {
        backend: "Transformers".to_string(),
        dependency: "shipped worker script (resources/transformers_worker.py)".to_string(),
    })?;
    let out = std::process::Command::new(python)
        .arg(&worker)
        .arg("--check-deps")
        .output()
        .map_err(|e| ModelError::DependencyMissing {
            backend: "Transformers".to_string(),
            dependency: format!(
                "Python executable {python:?} was not found ({e}). Install Python."
            ),
        })?;
    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        return Err(ModelError::DependencyMissing {
            backend: "Transformers".to_string(),
            dependency: format!(
                "worker --check-deps failed: {}",
                stderr.lines().next().unwrap_or("unknown error")
            ),
        });
    }
    let stdout = String::from_utf8_lossy(&out.stdout);
    let msg: WorkerMessage = stdout
        .lines()
        .filter_map(|l| serde_json::from_str(l).ok())
        .find(|m| matches!(m, WorkerMessage::Deps { .. }))
        .ok_or_else(|| ModelError::DependencyMissing {
            backend: "Transformers".to_string(),
            dependency: "worker returned no dependency report".to_string(),
        })?;
    match msg {
        WorkerMessage::Deps {
            transformers,
            torch,
            cuda_available,
        } => {
            if transformers.is_none() {
                return Err(ModelError::DependencyMissing {
                    backend: "Transformers".to_string(),
                    dependency: "Python is available, but the 'transformers' package is missing."
                        .to_string(),
                });
            }
            if torch.is_none() {
                return Err(ModelError::DependencyMissing {
                    backend: "Transformers".to_string(),
                    dependency: "PyTorch is not installed in the selected Python environment."
                        .to_string(),
                });
            }
            Ok(DependencyReport {
                python: python.to_string(),
                transformers_version: transformers,
                torch_version: torch,
                cuda_available: cuda_available.unwrap_or(false),
            })
        }
        _ => Err(ModelError::DependencyMissing {
            backend: "Transformers".to_string(),
            dependency: "unexpected worker reply".to_string(),
        }),
    }
}

fn map_worker_error(code: Option<String>, message: Option<String>, stderr_tail: &str) -> String {
    let code = code.unwrap_or_else(|| "worker_error".to_string());
    let message = message.unwrap_or_default();
    match code.as_str() {
        "out_of_memory" => format!("Transformers generation failed: CUDA out of memory. {message}"),
        "cancelled" => "generation cancelled".to_string(),
        _ => {
            if message.contains("out of memory") || stderr_tail.contains("out of memory") {
                format!("Transformers generation failed: CUDA out of memory. {message}")
            } else if message.is_empty() && !stderr_tail.is_empty() {
                format!("Transformers worker failed [{code}]: {stderr_tail}")
            } else {
                format!("Transformers worker failed [{code}]: {message}")
            }
        }
    }
}

/// The backend: configuration only. Execution state lives in the worker
/// process; nothing model-shaped is held here.
#[derive(Clone)]
pub struct TransformersBackend {
    pub model_dir: PathBuf,
    pub expected_arch: Option<String>,
    pub python: String,
    pub device: String,
    pub worker_script: Option<PathBuf>,
    pub max_new_tokens: u32,
    /// Sampling temperature. None = greedy (do_sample off). Initialized
    /// from the shared Hercules temperature so backends behave alike.
    pub temperature: Option<f32>,
    /// Skip the dependency preflight (test seam for stub workers).
    /// Production always probes: a missing torch must fail before spawn.
    pub skip_probe: bool,
    /// Run-scoped cancellation token, set by the App trigger arm before
    /// spawning generation. Drives cooperative cancel + group kill below.
    /// (Kept out of the App select! wrapper: this IS the inner path.)
    pub run_token: Option<tokio_util::sync::CancellationToken>,
}

/// A spawned worker: raw OS handle (for group kill + reaping) plus async
/// stdio halves. The raw handle is always reaped exactly once.
struct WorkerProc {
    child: Option<std::process::Child>,
    stdin: tokio::process::ChildStdin,
    lines: tokio::io::Lines<tokio::io::BufReader<tokio::process::ChildStdout>>,
    stderr_tail: Arc<Mutex<String>>,
}

impl TransformersBackend {
    pub fn new(model_dir: PathBuf) -> Self {
        Self {
            model_dir,
            expected_arch: None,
            python: crate::settings::get_transformers_python()
                .filter(|s| !s.trim().is_empty())
                .unwrap_or_else(default_python),
            device: crate::settings::get_transformers_device(),
            worker_script: None,
            max_new_tokens: 256,
            skip_probe: false,
            run_token: None,
            temperature: Some(crate::settings::temperature()),
        }
    }

    /// Attach the run-scoped cancellation token (called by the App trigger
    /// arm on its backend clone before spawning generation).
    pub fn set_run_token(&mut self, token: tokio_util::sync::CancellationToken) {
        self.run_token = Some(token);
    }

    pub fn name(&self) -> String {
        format!(
            "Transformers ({})",
            self.model_dir
                .file_name()
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_else(|| self.model_dir.display().to_string())
        )
    }

    /// Spawn the worker with structured argv (never a shell) in its own
    /// process group (setsid), so the shared task_manager group-kill path
    /// terminates the worker AND any descendants. Stdio is converted to
    /// async halves; the raw handle stays for kill/reap.
    fn spawn_worker(&self) -> Result<WorkerProc, ModelError> {
        use tokio::io::AsyncBufReadExt;
        if !self.model_dir.is_dir() {
            return Err(ModelError::FileUnavailable {
                repo: self.model_dir.display().to_string(),
                file: "model directory with config.json + weights".to_string(),
            });
        }
        let worker = resolve_worker_script(self.worker_script.as_deref().and_then(|p| p.to_str()))
            .ok_or_else(|| ModelError::DependencyMissing {
                backend: "Transformers".to_string(),
                dependency: "shipped worker script (resources/transformers_worker.py)".to_string(),
            })?;
        let mut cmd = std::process::Command::new(&self.python);
        cmd.arg(&worker)
            .arg("--model-path")
            .arg(&self.model_dir)
            .arg("--device")
            .arg(&self.device)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        #[cfg(unix)]
        {
            use std::os::unix::process::CommandExt;
            unsafe {
                cmd.pre_exec(|| {
                    libc::setsid();
                    Ok(())
                });
            }
        }
        let mut child = cmd.spawn().map_err(|e| ModelError::DependencyMissing {
            backend: "Transformers".to_string(),
            dependency: format!("failed to launch {:?}: {e}", self.python),
        })?;
        let stdin = child.stdin.take().ok_or_else(|| ModelError::LoadFailed {
            backend: "Transformers".to_string(),
            detail: "worker stdin unavailable".to_string(),
        })?;
        let stdout = child.stdout.take().ok_or_else(|| ModelError::LoadFailed {
            backend: "Transformers".to_string(),
            detail: "worker stdout unavailable".to_string(),
        })?;
        let stderr = child.stderr.take().ok_or_else(|| ModelError::LoadFailed {
            backend: "Transformers".to_string(),
            detail: "worker stderr unavailable".to_string(),
        })?;
        let stdin =
            tokio::process::ChildStdin::from_std(stdin).map_err(|e| ModelError::LoadFailed {
                backend: "Transformers".to_string(),
                detail: format!("failed to wrap worker stdin: {e}"),
            })?;
        let stdout =
            tokio::process::ChildStdout::from_std(stdout).map_err(|e| ModelError::LoadFailed {
                backend: "Transformers".to_string(),
                detail: format!("failed to wrap worker stdout: {e}"),
            })?;
        let stderr =
            tokio::process::ChildStderr::from_std(stderr).map_err(|e| ModelError::LoadFailed {
                backend: "Transformers".to_string(),
                detail: format!("failed to wrap worker stderr: {e}"),
            })?;
        // Dedicated stderr drain from birth: a noisy worker (PyTorch
        // warnings, CUDA logs) can never fill the pipe and stall itself.
        // Bounded tail (last 32 KB) stays for crash diagnostics.
        let stderr_tail: Arc<Mutex<String>> = Arc::new(Mutex::new(String::new()));
        {
            use tokio::io::AsyncReadExt;
            let tail = stderr_tail.clone();
            let mut stderr = stderr;
            tokio::spawn(async move {
                let mut buf = vec![0u8; 4096];
                loop {
                    match stderr.read(&mut buf).await {
                        Ok(0) | Err(_) => break,
                        Ok(n) => {
                            let s = String::from_utf8_lossy(&buf[..n]).to_string();
                            if let Ok(mut t) = tail.lock() {
                                t.push_str(&s);
                                const KEEP: usize = 32 * 1024;
                                if t.len() > KEEP {
                                    let cut = t.len() - KEEP;
                                    t.drain(..cut);
                                }
                            }
                        }
                    }
                }
            });
        }
        Ok(WorkerProc {
            child: Some(child),
            stdin,
            lines: tokio::io::BufReader::new(stdout).lines(),
            stderr_tail,
        })
    }

    /// Reap exactly once via the shared group-kill path (TERM group+pid,
    /// grace, KILL group+pid, wait). Runs off-thread: kill_child_tree
    /// sleeps briefly and must never block the async runtime.
    async fn terminate(proc: WorkerProc) -> String {
        let tail = proc
            .stderr_tail
            .lock()
            .map(|t| {
                t.lines()
                    .rev()
                    .take(5)
                    .collect::<Vec<_>>()
                    .into_iter()
                    .rev()
                    .collect::<Vec<_>>()
                    .join("\n")
            })
            .unwrap_or_default();
        if let WorkerProc { mut child, .. } = proc {
            if let Some(mut raw) = child.take() {
                let _ = tokio::task::spawn_blocking(move || {
                    crate::task_manager::kill_child_tree(&mut raw);
                })
                .await;
            }
        }
        tail
    }

    async fn read_message(
        lines: &mut tokio::io::Lines<tokio::io::BufReader<tokio::process::ChildStdout>>,
        timeout: Duration,
    ) -> Result<Option<WorkerMessage>, ModelError> {
        match tokio::time::timeout(timeout, lines.next_line()).await {
            Err(_) => Err(ModelError::GenerationFailed {
                backend: "Transformers".to_string(),
                detail: "worker timed out waiting for protocol message".to_string(),
            }),
            Ok(Err(e)) => Err(ModelError::GenerationFailed {
                backend: "Transformers".to_string(),
                detail: format!("worker stdout read failed: {e}"),
            }),
            Ok(Ok(None)) => Ok(None), // EOF
            Ok(Ok(Some(line))) => {
                if line.trim().is_empty() {
                    return Ok(Some(WorkerMessage::Unknown));
                }
                match serde_json::from_str(&line) {
                    Ok(m) => Ok(Some(m)),
                    Err(_) => Err(ModelError::GenerationFailed {
                        backend: "Transformers".to_string(),
                        detail: "protocol error: malformed JSON from worker".to_string(),
                    }),
                }
            }
        }
    }

    /// Reject messages for another request id. Today the worker serves one
    /// request at a time; strictness keeps the protocol honest for later
    /// concurrency and catches worker bugs early.
    fn check_request_id(msg_id: &str, expected: &str) -> Result<(), ModelError> {
        if msg_id != expected {
            return Err(ModelError::GenerationFailed {
                backend: "Transformers".to_string(),
                detail: format!(
                    "protocol error: message for request {msg_id:?}, expected {expected:?}"
                ),
            });
        }
        Ok(())
    }

    /// Handshake: hello(version) + ready(arch/device). Validates protocol
    /// version and expected architecture before any generation (Rust-side
    /// preflight; the worker validated its own load independently).
    async fn handshake(
        &self,
        lines: &mut tokio::io::Lines<tokio::io::BufReader<tokio::process::ChildStdout>>,
    ) -> Result<WorkerReady, ModelError> {
        let mut ready: Option<WorkerReady> = None;
        let start = Instant::now();
        while start.elapsed() < HANDSHAKE_TIMEOUT {
            match Self::read_message(lines, Duration::from_secs(10)).await? {
                Some(WorkerMessage::Hello { protocol_version }) => {
                    if protocol_version != PROTOCOL_VERSION {
                        return Err(ModelError::LoadFailed {
                            backend: "Transformers".to_string(),
                            detail: format!(
                                "worker protocol {protocol_version} != host {PROTOCOL_VERSION}"
                            ),
                        });
                    }
                }
                Some(WorkerMessage::Ready {
                    architecture,
                    model_type,
                    device,
                    ..
                }) => {
                    ready = Some(WorkerReady {
                        architecture,
                        model_type,
                        device: device.unwrap_or_else(|| "unknown".to_string()),
                    });
                    break;
                }
                Some(WorkerMessage::Error { code, message, .. }) => {
                    return Err(ModelError::LoadFailed {
                        backend: "Transformers".to_string(),
                        detail: map_worker_error(code, message, ""),
                    });
                }
                Some(_) => continue,
                None => break,
            }
        }
        let ready = ready.ok_or_else(|| ModelError::LoadFailed {
            backend: "Transformers".to_string(),
            detail: "worker never became ready (EOF before handshake)".to_string(),
        })?;
        if let Some(expected) = &self.expected_arch {
            // Normalization lives in exactly one place (Rust): the worker
            // reports both raw strings, we compare families of either.
            let exp = super::compatibility::arch_family(expected);
            let matches = [&ready.architecture, &ready.model_type]
                .into_iter()
                .flatten()
                .any(|actual| super::compatibility::arch_family(actual) == exp);
            if !matches {
                return Err(ModelError::ArchitectureUnsupported {
                    architecture: format!(
                        "expected {expected}, worker loaded {}/{}",
                        ready.architecture.as_deref().unwrap_or("?"),
                        ready.model_type.as_deref().unwrap_or("?")
                    ),
                    backend: "Transformers".to_string(),
                });
            }
        }
        Ok(ready)
    }

    /// Stream generation with the run-scoped token driving cancellation:
    /// cooperative cancel message first, shared group-kill on token fire.
    /// App-level select! dropping this future is the final backstop (the
    /// proc is reaped via terminate() on every path).
    pub async fn generate_stream(
        &self,
        prompt: &str,
        stream_target: Arc<Mutex<String>>,
        is_generating: Arc<Mutex<bool>>,
    ) -> Result<String, String> {
        self.generate_stream_inner(prompt, stream_target, is_generating)
            .await
            .map_err(|e| e.message())
    }

    async fn generate_stream_inner(
        &self,
        prompt: &str,
        stream_target: Arc<Mutex<String>>,
        is_generating: Arc<Mutex<bool>>,
    ) -> Result<String, ModelError> {
        // Local config errors first (cheap), environment probe second:
        // a missing directory must not masquerade as a missing torch.
        if !self.model_dir.is_dir() {
            return Err(ModelError::FileUnavailable {
                repo: self.model_dir.display().to_string(),
                file: "model directory with config.json + weights".to_string(),
            });
        }
        // Dependency preflight with useful errors (cached). Skipped only
        // by tests driving stub workers without torch installed.
        if !self.skip_probe {
            probe_cached(&self.python)?;
        }
        let mut proc = self.spawn_worker()?;
        let request_id = format!("r{}", RequestId::next());
        let outcome = self
            .drive_generation(
                &mut proc,
                &request_id,
                prompt,
                &stream_target,
                &is_generating,
            )
            .await;
        // Always reap via the shared group-kill path; stderr tail (drained
        // since spawn) powers crash diagnostics.
        let stderr_tail = Self::terminate(proc).await;
        match outcome {
            Ok(text) => Ok(text),
            Err(mut e) => {
                // Enrich bare worker errors with stderr (OOM etc.).
                if let ModelError::GenerationFailed { detail, .. } = &mut e {
                    if detail.contains("worker failed") && !stderr_tail.is_empty() {
                        detail.push_str(&format!("\nstderr: {stderr_tail}"));
                    }
                }
                Err(e)
            }
        }
    }

    async fn drive_generation(
        &self,
        proc: &mut WorkerProc,
        request_id: &str,
        prompt: &str,
        stream_target: &Arc<Mutex<String>>,
        is_generating: &Arc<Mutex<bool>>,
    ) -> Result<String, ModelError> {
        use tokio::io::AsyncWriteExt;
        let ready = self.handshake(&mut proc.lines).await?;
        let _ = ready;
        let req = WorkerRequest::Generate {
            request_id,
            prompt,
            max_new_tokens: self.max_new_tokens,
            temperature: self.temperature,
        };
        let mut payload =
            serde_json::to_string(&req).map_err(|e| ModelError::GenerationFailed {
                backend: "Transformers".to_string(),
                detail: format!("failed to encode request: {e}"),
            })?;
        payload.push('\n');
        proc.stdin
            .write_all(payload.as_bytes())
            .await
            .map_err(|e| ModelError::GenerationFailed {
                backend: "Transformers".to_string(),
                detail: format!("failed to send request: {e}"),
            })?;
        // Run-token branch: cooperative cancel message, then Cancelled.
        // The shared group-kill in terminate() is the hard fallback, and
        // App-level select! dropping this whole future reaps regardless.
        let token_fired = async {
            match &self.run_token {
                Some(tok) => tok.cancelled().await,
                None => std::future::pending().await,
            }
        };
        let message_loop = async {
            let mut full_text = String::new();
            let mut completed = false;
            let mut cancelled = false;
            loop {
                // Flag path races the token (stall/reject clear it without
                // touching run_token): same cooperative cancel, same exit.
                // (Lock scope ends before any await: std MutexGuard is !Send.)
                let flag_cancelled = is_generating.lock().map(|g| !*g).unwrap_or(false);
                if (flag_cancelled || cancelled) && !completed {
                    if !cancelled {
                        cancelled = true;
                        let cancel = WorkerRequest::Cancel { request_id };
                        if let Ok(mut c) = serde_json::to_string(&cancel) {
                            c.push('\n');
                            let _ = proc.stdin.write_all(c.as_bytes()).await;
                        }
                    }
                    if flag_cancelled {
                        break;
                    }
                }
                match Self::read_message(&mut proc.lines, LINE_TIMEOUT).await? {
                    Some(WorkerMessage::Token {
                        request_id: rid,
                        text,
                    }) => {
                        Self::check_request_id(&rid, request_id)?;
                        full_text.push_str(&text);
                        if let Ok(mut t) = stream_target.lock() {
                            t.push_str(&text);
                        }
                    }
                    Some(WorkerMessage::Done { request_id: rid }) => {
                        Self::check_request_id(&rid, request_id)?;
                        completed = true;
                        break;
                    }
                    Some(WorkerMessage::Cancelled { request_id: rid }) => {
                        Self::check_request_id(&rid, request_id)?;
                        cancelled = true;
                        break;
                    }
                    Some(WorkerMessage::Error {
                        request_id: rid,
                        code,
                        message,
                    }) => {
                        if let Some(rid) = rid {
                            Self::check_request_id(&rid, request_id)?;
                        }
                        let tail = String::new();
                        return Err(ModelError::GenerationFailed {
                            backend: "Transformers".to_string(),
                            detail: map_worker_error(code, message, &tail),
                        });
                    }
                    Some(_) => continue,
                    None => {
                        // EOF is NEVER success: without an explicit done or
                        // cancelled, the worker crashed or was terminated.
                        break;
                    }
                }
            }
            if cancelled {
                return Err(ModelError::Cancelled);
            }
            if !completed {
                return Err(ModelError::GenerationFailed {
                    backend: "Transformers".to_string(),
                    detail: "worker exited without done/cancelled: terminated unexpectedly"
                        .to_string(),
                });
            }
            if full_text.is_empty() {
                return Err(ModelError::GenerationFailed {
                    backend: "Transformers".to_string(),
                    detail: "worker completed without producing output".to_string(),
                });
            }
            Ok(full_text)
        };
        tokio::select! {
            r = message_loop => r,
            _ = token_fired => {
                Err(ModelError::Cancelled)
            }
        }
    }

    pub async fn generate(&self, prompt: &str) -> Result<String, String> {
        let target = Arc::new(Mutex::new(String::new()));
        let flag = Arc::new(Mutex::new(true));
        self.generate_stream(prompt, target, flag).await
    }
}

fn default_python() -> String {
    std::env::var("HERCULES_TRANSFORMERS_PYTHON").unwrap_or_else(|_| "python3".to_string())
}

struct RequestId;
static NEXT_REQUEST: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

impl RequestId {
    fn next() -> u64 {
        NEXT_REQUEST.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_protocol_request_serializes() {
        let r = WorkerRequest::Generate {
            request_id: "1",
            prompt: "hi",
            max_new_tokens: 8,
            temperature: Some(0.7),
        };
        let s = serde_json::to_string(&r).unwrap();
        assert!(s.contains("\"type\":\"generate\""));
        assert!(s.contains("\"request_id\":\"1\""));
        assert!(s.contains("\"temperature\":0.7"), "{s}");
    }

    #[test]
    fn test_protocol_messages_parse() {
        let m: WorkerMessage =
            serde_json::from_str(r#"{"type":"token","request_id":"1","text":"hi"}"#).unwrap();
        assert!(matches!(m, WorkerMessage::Token { .. }));
        let m: WorkerMessage = serde_json::from_str(r#"{"type":"done","request_id":"1"}"#).unwrap();
        assert!(matches!(m, WorkerMessage::Done { .. }));
        let m: WorkerMessage =
            serde_json::from_str(r#"{"type":"error","code":"out_of_memory","message":"boom"}"#)
                .unwrap();
        assert!(matches!(m, WorkerMessage::Error { .. }));
        // Unknown future types never break the parser.
        let m: WorkerMessage = serde_json::from_str(r#"{"type":"teleport"}"#).unwrap();
        assert!(matches!(m, WorkerMessage::Unknown));
        // Malformed JSON is an error, never a panic.
        assert!(serde_json::from_str::<WorkerMessage>("{oops").is_err());
    }

    #[test]
    fn test_worker_script_resolves_in_repo() {
        // Dev tree: manifest-dir fallback finds resources/.
        assert!(resolve_worker_script(None).is_some());
        assert!(resolve_worker_script(Some("/nonexistent/worker.py")).is_none());
    }

    #[test]
    fn test_missing_python_gives_useful_error() {
        let err = probe_dependencies("/nonexistent-python-xyz").unwrap_err();
        let msg = err.message();
        assert!(msg.contains("Python"), "{msg}");
        assert!(!msg.contains("BackendUnavailable"), "{msg}");
    }

    #[test]
    fn test_map_worker_error_oom_and_cancel() {
        let s = map_worker_error(
            Some("out_of_memory".to_string()),
            Some("CUDA out of memory.".to_string()),
            "",
        );
        assert!(s.contains("out of memory"), "{s}");
        assert!(!s.contains("status 137"), "{s}");
        let s = map_worker_error(Some("cancelled".to_string()), None, "");
        assert_eq!(s, "generation cancelled");
    }

    /// Write a stub worker script (speaks the JSONL protocol) into a temp
    /// dir. Lets lifecycle tests run without torch/transformers/models.
    fn stub_worker(body: &str) -> (PathBuf, PathBuf) {
        let dir = std::env::temp_dir().join(format!(
            "herc-tf-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.subsec_nanos())
                .unwrap_or(0)
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let script = dir.join("stub_worker.py");
        std::fs::write(&script, body).unwrap();
        let model_dir = dir.join("model");
        std::fs::create_dir_all(&model_dir).unwrap();
        (script, model_dir)
    }

    const STUB_SERVE: &str = r#"import sys, json, time
def emit(o):
    sys.stdout.write(json.dumps(o) + "\n"); sys.stdout.flush()
emit({"type": "hello", "protocol_version": 1})
emit({"type": "ready", "protocol_version": 1,
      "architecture": "LlamaForCausalLM", "device": "cpu", "device_name": "cpu"})
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        msg = json.loads(line)
    except Exception:
        continue
    if msg.get("type") == "generate":
        rid = msg.get("request_id", "")
        for tok in ["Hello", " world"]:
            emit({"type": "token", "request_id": rid, "text": tok})
            time.sleep(0.05)
        emit({"type": "done", "request_id": rid})
    elif msg.get("type") == "cancel":
        emit({"type": "cancelled", "request_id": msg.get("request_id", "")})
    elif msg.get("type") == "shutdown":
        emit({"type": "bye", "protocol_version": 1})
        break
"#;

    fn test_backend(script: &Path, model_dir: &Path) -> TransformersBackend {
        TransformersBackend {
            model_dir: model_dir.to_path_buf(),
            expected_arch: Some("LlamaForCausalLM".to_string()),
            python: "python3".to_string(),
            device: "cpu".to_string(),
            worker_script: Some(script.to_path_buf()),
            max_new_tokens: 16,
            skip_probe: true,
            run_token: None,
            temperature: Some(0.7),
        }
    }

    #[tokio::test]
    async fn test_worker_handshake_streams_tokens() {
        let (script, model) = stub_worker(STUB_SERVE);
        let backend = test_backend(&script, &model);
        let target = Arc::new(Mutex::new(String::new()));
        let flag = Arc::new(Mutex::new(true));
        let out = backend
            .generate_stream("Say hi", target.clone(), flag)
            .await
            .expect("stub serve must succeed");
        assert_eq!(out, "Hello world");
        assert_eq!(target.lock().unwrap().as_str(), "Hello world");
    }

    #[tokio::test]
    async fn test_worker_arch_mismatch_rejected() {
        let (script, model) = stub_worker(STUB_SERVE);
        let mut backend = test_backend(&script, &model);
        backend.expected_arch = Some("Qwen2ForCausalLM".to_string());
        let err = backend.generate("hi").await.unwrap_err();
        assert!(err.contains("Qwen2") || err.contains("Llama"), "{err}");
    }

    #[tokio::test]
    async fn test_worker_crash_is_typed_not_hang() {
        // Worker exits immediately with stderr: must become a typed error
        // fast, never a hang, Hercules stays alive.
        let (script, model) =
            stub_worker("import sys; print('boom', file=sys.stderr); sys.exit(3)\n");
        let backend = test_backend(&script, &model);
        let start = Instant::now();
        let err = backend.generate("hi").await.unwrap_err();
        assert!(
            start.elapsed() < Duration::from_secs(60),
            "crash handling hung"
        );
        assert!(
            err.contains("never became ready") || err.contains("boom") || err.contains("failed"),
            "{err}"
        );
    }

    #[tokio::test]
    async fn test_worker_malformed_json_is_protocol_error() {
        let (script, model) = stub_worker("print('this is not json', flush=True)\n");
        let backend = test_backend(&script, &model);
        let err = backend.generate("hi").await.unwrap_err();
        assert!(
            err.contains("protocol") || err.contains("ready") || err.contains("failed"),
            "{err}"
        );
    }

    #[tokio::test]
    async fn test_cancel_flag_stops_generation() {
        // Flag cleared before any output: cooperative cancel path.
        let (script, model) = stub_worker(STUB_SERVE);
        let backend = test_backend(&script, &model);
        let target = Arc::new(Mutex::new(String::new()));
        let flag = Arc::new(Mutex::new(false));
        let err = backend
            .generate_stream("hi", target, flag)
            .await
            .unwrap_err();
        assert!(err.to_lowercase().contains("cancel"), "{err}");
    }
    #[test]
    fn test_missing_model_dir_fails_before_spawn() {
        let backend = TransformersBackend::new(PathBuf::from("/nonexistent-model-dir-xyz"));
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let err = rt.block_on(backend.generate("hi")).unwrap_err();
        assert!(err.contains("model directory"), "{err}");
    }

    #[tokio::test]
    async fn test_run_token_cancel_stops_worker() {
        // Pre-cancelled run token: drive must return Cancelled and reap
        // the worker without hanging — proves the token path end to end
        // against a real (stub) OS process.
        let (script, model) = stub_worker(STUB_SERVE);
        let mut backend = test_backend(&script, &model);
        let tok = tokio_util::sync::CancellationToken::new();
        tok.cancel();
        backend.set_run_token(tok);
        let start = Instant::now();
        let err = backend
            .generate_stream(
                "hi",
                Arc::new(Mutex::new(String::new())),
                Arc::new(Mutex::new(true)),
            )
            .await
            .unwrap_err();
        assert!(err.to_lowercase().contains("cancel"), "{err}");
        assert!(start.elapsed() < Duration::from_secs(60), "cancel hung");
    }

    #[tokio::test]
    async fn test_wrong_request_id_is_protocol_error() {
        // Stub answers with a FOREIGN request id: strict validation must
        // fire instead of accepting another request's tokens.
        let (script, model) = stub_worker(
            "import sys, json\nprint(json.dumps({\"type\": \"hello\", \"protocol_version\": 1}), flush=True)\nprint(json.dumps({\"type\": \"ready\", \"protocol_version\": 1, \"architecture\": \"LlamaForCausalLM\", \"device\": \"cpu\"}), flush=True)\nfor line in sys.stdin:\n    m = json.loads(line)\n    if m.get(\"type\") == \"generate\":\n        print(json.dumps({\"type\": \"token\", \"request_id\": \"WRONG\", \"text\": \"x\"}), flush=True)\n        print(json.dumps({\"type\": \"done\", \"request_id\": \"WRONG\"}), flush=True)\n",
        );
        let backend = test_backend(&script, &model);
        let err = backend
            .generate_stream(
                "hi",
                Arc::new(Mutex::new(String::new())),
                Arc::new(Mutex::new(true)),
            )
            .await
            .unwrap_err();
        assert!(
            err.contains("protocol error") && err.contains("WRONG"),
            "{err}"
        );
    }

    #[tokio::test]
    async fn test_eof_after_tokens_is_crash_not_success() {
        // Worker emits a valid token then dies WITHOUT done: partial text
        // must NOT be returned as success.
        let (script, model) = stub_worker(
            "import sys, json\nprint(json.dumps({\"type\": \"hello\", \"protocol_version\": 1}), flush=True)\nprint(json.dumps({\"type\": \"ready\", \"protocol_version\": 1, \"architecture\": \"LlamaForCausalLM\", \"device\": \"cpu\"}), flush=True)\nfor line in sys.stdin:\n    m = json.loads(line)\n    if m.get(\"type\") == \"generate\":\n        rid = m[\"request_id\"]\n        print(json.dumps({\"type\": \"token\", \"request_id\": rid, \"text\": \"partial\"}), flush=True)\n        sys.stdout.flush()\n        break\n",
        );
        let backend = test_backend(&script, &model);
        let target = Arc::new(Mutex::new(String::new()));
        let err = backend
            .generate_stream("hi", target.clone(), Arc::new(Mutex::new(true)))
            .await
            .unwrap_err();
        assert!(err.contains("terminated unexpectedly"), "{err}");
        // Partial text reached the stream target but was NOT returned as Ok.
        assert_eq!(target.lock().unwrap().as_str(), "partial");
    }

    /// Stubborn worker: emits a token, then ignores cancel forever and
    /// sleeps. Proves the hard path — token → Cancelled fast AND the OS
    /// process actually dies (verified via /proc, not flags).
    const STUB_STUBBORN: &str = r#"import sys, json, time
def emit(o):
    sys.stdout.write(json.dumps(o) + "\n"); sys.stdout.flush()
emit({"type": "hello", "protocol_version": 1})
emit({"type": "ready", "protocol_version": 1,
      "architecture": "LlamaForCausalLM", "device": "cpu"})
for line in sys.stdin:
    try:
        msg = json.loads(line)
    except Exception:
        continue
    if msg.get("type") == "generate":
        rid = msg.get("request_id", "")
        emit({"type": "token", "request_id": rid, "text": "x"})
        while True:
            time.sleep(60)
"#;

    fn proc_with_cmdline(fragment: &str) -> bool {
        let Ok(entries) = std::fs::read_dir("/proc") else {
            return false;
        };
        for entry in entries.flatten() {
            let name = entry.file_name();
            let Some(pid) = name.to_str() else { continue };
            if !pid.bytes().all(|b| b.is_ascii_digit()) {
                continue;
            }
            let cmdline = std::fs::read(format!("/proc/{pid}/cmdline")).unwrap_or_default();
            let cmd = String::from_utf8_lossy(&cmdline).replace('\0', " ");
            if cmd.contains(fragment) {
                return true;
            }
        }
        false
    }

    #[tokio::test]
    async fn test_stubborn_worker_is_killed_not_just_cancelled() {
        let (script, model) = stub_worker(STUB_STUBBORN);
        let script_tag = script.to_string_lossy().to_string();
        let mut backend = test_backend(&script, &model);
        let tok = tokio_util::sync::CancellationToken::new();
        backend.set_run_token(tok.clone());
        let target = Arc::new(Mutex::new(String::new()));
        let flag = Arc::new(Mutex::new(true));
        let backend_clone = backend.clone();
        let target_clone = target.clone();
        let handle = tokio::spawn(async move {
            backend_clone
                .generate_stream("hi", target_clone, flag)
                .await
        });
        // Wait until the worker is provably alive and streaming.
        let start = Instant::now();
        while target.lock().unwrap().is_empty() {
            tokio::task::yield_now().await;
            assert!(
                start.elapsed() < Duration::from_secs(30),
                "stubborn worker never produced a token"
            );
        }
        assert!(proc_with_cmdline(&script_tag), "worker process missing");
        // Cancel mid-generation: must return Cancelled quickly…
        tok.cancel();
        let err = handle
            .await
            .expect("generation task must finish")
            .unwrap_err();
        assert!(err.to_lowercase().contains("cancel"), "{err}");
        assert!(start.elapsed() < Duration::from_secs(60), "hard kill hung");
        // …AND the OS process must actually be gone (reaped, not lingering).
        let gone_start = Instant::now();
        while proc_with_cmdline(&script_tag) {
            tokio::time::sleep(Duration::from_millis(100)).await;
            assert!(
                gone_start.elapsed() < Duration::from_secs(15),
                "worker process survived cancellation"
            );
        }
    }

    #[tokio::test]
    async fn test_sequential_generations_do_not_cross_cancel() {
        // Review §8: a stale token on a reused backend must not kill the
        // next generation. Each generation gets a fresh token via
        // set_run_token (as the App trigger arm does).
        let (script, model) = stub_worker(STUB_SERVE);
        let mut backend = test_backend(&script, &model);
        let tok1 = tokio_util::sync::CancellationToken::new();
        backend.set_run_token(tok1.clone());
        tok1.cancel(); // stale: generation A was cancelled
        let tok2 = tokio_util::sync::CancellationToken::new();
        backend.set_run_token(tok2);
        let out = backend
            .generate_stream(
                "hi",
                Arc::new(Mutex::new(String::new())),
                Arc::new(Mutex::new(true)),
            )
            .await
            .expect("fresh token must not inherit cancellation");
        assert_eq!(out, "Hello world");
    }

    /// Opt-in end-to-end: real Python + transformers + tiny local model.
    /// Ignored by default; run with:
    /// HERCULES_TRANSFORMERS_TEST_MODEL=/path/to/tiny-model cargo test -- --ignored
    #[tokio::test]
    #[ignore]
    async fn test_e2e_tiny_model_if_provided() {
        let model_dir = match std::env::var("HERCULES_TRANSFORMERS_TEST_MODEL") {
            Ok(m) if !m.trim().is_empty() => m,
            _ => {
                eprintln!("skipped: set HERCULES_TRANSFORMERS_TEST_MODEL");
                return;
            }
        };
        let mut backend = TransformersBackend::new(PathBuf::from(&model_dir));
        backend.max_new_tokens = 8;
        let target = Arc::new(Mutex::new(String::new()));
        let flag = Arc::new(Mutex::new(true));
        let out = backend
            .generate_stream("Hello", target.clone(), flag)
            .await
            .expect("tiny-model e2e must succeed");
        assert!(!out.trim().is_empty(), "empty generation");
        assert_eq!(target.lock().unwrap().as_str(), out.as_str());
    }
}
