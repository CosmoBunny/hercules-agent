//! Long-lived `llama-server` process so the GGUF is loaded **once**.
//!
//! Used only by the **llama.cpp** track. Pure llama.rs never starts this.

use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitStatus, Stdio};
use std::time::{Duration, Instant};

#[cfg(unix)]
use std::os::unix::process::{CommandExt, ExitStatusExt};

fn find_llama_server() -> Option<PathBuf> {
    const NAMES: &[&str] = &["llama-server"];
    for name in NAMES {
        if let Ok(output) = Command::new("which").arg(name).output() {
            if output.status.success() {
                let p = String::from_utf8_lossy(&output.stdout).trim().to_string();
                if !p.is_empty() && Path::new(&p).is_file() {
                    return Some(PathBuf::from(p));
                }
            }
        }
    }
    for prefix in ["/opt/llama.cpp", "/usr/local/bin", "/usr/bin"] {
        let p = PathBuf::from(prefix).join("llama-server");
        if p.is_file() {
            return Some(p);
        }
    }
    if let Some(home) = std::env::var_os("HOME").or_else(|| std::env::var_os("USERPROFILE")) {
        for rel in [".local/bin/llama-server", ".cargo/bin/llama-server"] {
            let p = PathBuf::from(&home).join(rel);
            if p.is_file() {
                return Some(p);
            }
        }
    }
    None
}

fn free_port() -> Result<u16, String> {
    for port in 18100u16..18200 {
        if TcpListener::bind(("127.0.0.1", port)).is_ok() {
            return Ok(port);
        }
    }
    let listener = TcpListener::bind("127.0.0.1:0").map_err(|e| e.to_string())?;
    listener
        .local_addr()
        .map(|a| a.port())
        .map_err(|e| e.to_string())
}

fn pid_file_path() -> PathBuf {
    let base = std::env::var_os("TMPDIR")
        .or_else(|| std::env::var_os("TMP"))
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/tmp"));
    base.join("hercules").join("llama-server.pid")
}

fn write_pid_file(pid: u32, port: u16, model: &Path) {
    let path = pid_file_path();
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let _ = std::fs::write(
        &path,
        format!(
            "pid={}\nport={}\nmodel={}\n",
            pid,
            port,
            model.display()
        ),
    );
}

fn clear_pid_file() {
    let _ = std::fs::remove_file(pid_file_path());
}

struct ManagedServer {
    model_path: PathBuf,
    port: u16,
    child: Child,
    ngl: i32,
    n_ctx: usize,
    power_mode: crate::settings::PowerMode,
}

impl Drop for ManagedServer {
    fn drop(&mut self) {
        kill_child_hard(&mut self.child);
    }
}

fn kill_child_hard(child: &mut Child) {
    let pid = child.id();
    #[cfg(unix)]
    {
        let p = pid as i32;
        unsafe {
            // Process group (setsid leader) + direct pid
            libc::kill(-p, libc::SIGTERM);
            libc::kill(p, libc::SIGTERM);
        }
        std::thread::sleep(Duration::from_millis(80));
        unsafe {
            libc::kill(-p, libc::SIGKILL);
            libc::kill(p, libc::SIGKILL);
        }
    }
    let _ = child.kill();
    let _ = child.wait();
}

static MANAGED: std::sync::Mutex<Option<ManagedServer>> = std::sync::Mutex::new(None);

/// Classify child exit for human-readable errors (do NOT call everything OOM).
fn describe_exit(status: &ExitStatus) -> String {
    #[cfg(unix)]
    {
        if let Some(sig) = status.signal() {
            return match sig {
                4 => "signal 4 (SIGILL) — illegal instruction. \
                      Your llama-server/libggml was built for a newer CPU (e.g. AVX-512) \
                      than this machine has. Rebuild llama.cpp with AVX2-only / GGML_NATIVE=OFF, \
                      or install a binary matched to this CPU. This is NOT a bad GGUF and NOT OOM."
                    .into(),
                9 => "signal 9 (SIGKILL) — usually the OOM killer (not enough free RAM for weights+KV)."
                    .into(),
                6 => "signal 6 (SIGABRT) — process aborted (assert/crash inside llama-server).".into(),
                11 => "signal 11 (SIGSEGV) — crash inside llama-server (bad build or corrupt GGUF)."
                    .into(),
                n => format!("signal {n}"),
            };
        }
    }
    if let Some(code) = status.code() {
        format!("exit code {code}")
    } else {
        format!("{status}")
    }
}

fn looks_like_oom(status: &ExitStatus) -> bool {
    #[cfg(unix)]
    {
        if let Some(sig) = status.signal() {
            return sig == 9; // SIGKILL only
        }
    }
    false
}

fn looks_like_sigill(status: &ExitStatus) -> bool {
    #[cfg(unix)]
    {
        return status.signal() == Some(4);
    }
    #[cfg(not(unix))]
    {
        let _ = status;
        false
    }
}

/// MemAvailable from /proc (Linux), else a conservative guess.
fn mem_available_bytes() -> u64 {
    if let Ok(text) = std::fs::read_to_string("/proc/meminfo") {
        for line in text.lines() {
            if let Some(rest) = line.strip_prefix("MemAvailable:") {
                let kb: u64 = rest
                    .split_whitespace()
                    .next()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(0);
                return kb.saturating_mul(1024);
            }
        }
    }
    2 * 1024 * 1024 * 1024 // 2 GiB guess
}

fn model_bytes(model_path: &Path) -> u64 {
    std::fs::metadata(model_path).map(|m| m.len()).unwrap_or(0)
}

/// True when loading this GGUF will leave little headroom (thrash / thermal risk).
fn memory_tight(model_path: &Path) -> bool {
    let weights = model_bytes(model_path);
    let avail = mem_available_bytes();
    // Need weights + ~1.2 GiB headroom for OS/TUI/KV; otherwise thrash.
    let need = weights.saturating_add(1_200 * 1024 * 1024);
    avail < need || avail.saturating_sub(weights) < 900 * 1024 * 1024
}

/// Rough max `-c` that can fit after loading weights (CPU path).
///
/// Large weights + high `-c` on low free RAM → OOM (kernel SIGKILL).
fn ram_safe_ctx(model_path: &Path) -> usize {
    let weights = model_bytes(model_path);
    let avail = mem_available_bytes();
    // OS + Hercules TUI + llama-server overhead + fragmentation
    let reserve: u64 = 1_800 * 1024 * 1024;
    let free = avail.saturating_sub(weights).saturating_sub(reserve);
    let free_mb = free / (1024 * 1024);
    // Bigger weights → more KV per token (scale to ~2 GB reference).
    let model_gb = weights as f64 / (1024.0 * 1024.0 * 1024.0);
    let weight = (model_gb / 2.0).clamp(0.5, 3.0);
    let adj = free_mb as f64 / weight;
    if adj < 300.0 {
        4_096
    } else if adj < 600.0 {
        8_192
    } else if adj < 1_200.0 {
        16_384
    } else if adj < 2_400.0 {
        32_768
    } else if adj < 4_800.0 {
        65_536
    } else {
        131_072
    }
}

/// At most two ctx tries — full GGUF reloads burn CPU/RAM/thermals.
/// Start at min(requested, RAM-safe); one lower preset only if first dies.
fn ctx_fallback_ladder(requested: usize, model_path: &Path) -> Vec<usize> {
    let safe = ram_safe_ctx(model_path);
    let start = requested.min(safe).clamp(2_048, crate::settings::MAX_CONTEXT_TOKEN_LIMIT);
    let mut out = vec![start];
    // Single step-down only (avoid 128K→64K→…→4K reload storm).
    let lower = crate::settings::CONTEXT_PRESETS
        .iter()
        .copied()
        .filter(|&p| p < start)
        .max()
        .unwrap_or(4_096)
        .max(4_096);
    if lower < start {
        out.push(lower);
    }
    out
}

/// Threads for spawn: cool down under memory pressure / Power Saver.
fn spawn_threads(power: crate::settings::PowerMode, model_path: &Path) -> usize {
    let base = power.threads();
    if memory_tight(model_path) {
        // Dual-core laptops cook at 100% × all cores while swapping.
        base.min(2).max(1)
    } else {
        base
    }
}

/// GPU layers for spawn.
///
/// Default **0 (CPU only)**. On this class of machine OpenVINO/Vulkan offload often
/// loads OK then dies mid-decode with "Compute error" / empty streams. Only honor
/// explicit `HERCULES_N_GPU_LAYERS` or Extreme when not memory-tight.
fn spawn_ngl(power: crate::settings::PowerMode, model_path: &Path) -> i32 {
    if memory_tight(model_path) {
        return 0;
    }
    // Explicit env always wins (including 0).
    if std::env::var("HERCULES_N_GPU_LAYERS").is_ok() {
        return power.n_gpu_layers();
    }
    // Extreme may request offload; everything else stays CPU-only (stable).
    match power {
        crate::settings::PowerMode::Extreme => power.n_gpu_layers(),
        _ => 0,
    }
}

/// After a dead child, give the kernel a moment to reclaim RSS before reloading GGUF.
async fn reclaim_pause() {
    tokio::time::sleep(Duration::from_millis(800)).await;
}

/// Base URL of the warm server for this GGUF (starts process if needed).
pub async fn ensure_server_for_model(model_path: &Path) -> Result<(String, String), String> {
    let model_path = model_path
        .canonicalize()
        .unwrap_or_else(|_| model_path.to_path_buf());
    if !model_path.is_file() {
        return Err(format!(
            "[llama-server] Model not found: {}",
            model_path.display()
        ));
    }

    let current_power = crate::settings::get_settings().power_mode;
    let desired_ctx = crate::settings::context_token_limit().clamp(
        2048,
        crate::settings::MAX_CONTEXT_TOKEN_LIMIT,
    );
    let desired_ngl = spawn_ngl(current_power, &model_path);

    // Reuse warm server only when model / power / ctx / ngl still match.
    // Never hold the Mutex across `.await` (tokio::spawn requires Send).
    let reuse_url = {
        let mut guard = MANAGED.lock().map_err(|e| e.to_string())?;
        let reuse = guard.as_ref().and_then(|s| {
            if s.model_path == model_path
                && s.power_mode == current_power
                && s.n_ctx == desired_ctx
                && s.ngl == desired_ngl
            {
                Some(format!("http://127.0.0.1:{}", s.port))
            } else {
                None
            }
        });
        if reuse.is_none() {
            *guard = None; // Drop kills mismatched child
        }
        reuse
    };
    if let Some(url) = reuse_url {
        if health_ok(&url).await {
            let name = model_path
                .file_name()
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_else(|| "local-gguf".into());
            return Ok((url, name));
        }
        let mut guard = MANAGED.lock().map_err(|e| e.to_string())?;
        *guard = None;
    }

    let bin = find_llama_server().ok_or_else(|| {
        "[llama-server] binary not found. Install llama.cpp and put `llama-server` on PATH \
         (or under /opt/llama.cpp)."
            .to_string()
    })?;

    // Kill any orphan from a previous thrash before we load again.
    shutdown_orphans_only();

    let ngl = desired_ngl;
    let threads = spawn_threads(current_power, &model_path);
    let start_ctx = desired_ctx.min(ram_safe_ctx(&model_path));

    // Prefer a single load. If offload (ngl>0) dies/times out, one pure-CPU retry only.
    match start_with_ctx_ladder(
        &bin,
        &model_path,
        threads,
        ngl,
        start_ctx,
        current_power,
    )
    .await
    {
        Ok(v) => Ok(v),
        Err(e) if ngl != 0 => {
            reclaim_pause().await;
            start_with_ctx_ladder(
                &bin,
                &model_path,
                threads,
                0,
                start_ctx.min(8_192),
                current_power,
            )
            .await
            .map_err(|e2| format!("{e} → CPU retry: {e2}"))
        }
        Err(e) => Err(e),
    }
}

/// Kill pid-file / port-band orphans without dropping a live managed handle mid-use.
fn shutdown_orphans_only() {
    let pf = pid_file_path();
    if let Ok(text) = std::fs::read_to_string(&pf) {
        for line in text.lines() {
            if let Some(pid_s) = line.strip_prefix("pid=") {
                if let Ok(pid) = pid_s.trim().parse::<i32>() {
                    // Don't kill our currently managed child
                    let managed_pid = MANAGED
                        .lock()
                        .ok()
                        .and_then(|g| g.as_ref().map(|s| s.child.id() as i32));
                    if managed_pid == Some(pid) {
                        continue;
                    }
                    #[cfg(unix)]
                    unsafe {
                        libc::kill(pid, libc::SIGTERM);
                        libc::kill(-pid, libc::SIGTERM);
                    }
                    std::thread::sleep(Duration::from_millis(80));
                    #[cfg(unix)]
                    unsafe {
                        libc::kill(pid, libc::SIGKILL);
                        libc::kill(-pid, libc::SIGKILL);
                    }
                }
            }
        }
    }
}

fn server_log_path() -> PathBuf {
    let base = std::env::var_os("TMPDIR")
        .or_else(|| std::env::var_os("TMP"))
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/tmp"));
    base.join("hercules").join("llama-server.last.log")
}

fn tail_log(path: &Path, max_chars: usize) -> String {
    let Ok(text) = std::fs::read_to_string(path) else {
        return String::new();
    };
    let t = text.trim();
    if t.is_empty() {
        return String::new();
    }
    if t.len() <= max_chars {
        return t.to_string();
    }
    let start = t
        .char_indices()
        .rev()
        .nth(max_chars - 1)
        .map(|(i, _)| i)
        .unwrap_or(0);
    t[start..].to_string()
}

/// Spawn once at a fixed ctx/ngl; success installs ManagedServer.
async fn try_start_server(
    bin: &Path,
    model_path: &Path,
    threads: usize,
    ngl: i32,
    n_ctx: usize,
    power_mode: crate::settings::PowerMode,
) -> Result<(String, String), String> {
    let port = free_port()?;
    let wait_secs = crate::settings::server_health_timeout_secs(n_ctx);
    let log_path = server_log_path();
    if let Some(parent) = log_path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let log_file = std::fs::File::create(&log_path)
        .map_err(|e| format!("[llama-server] cannot write log {}: {e}", log_path.display()))?;

    let mut cmd = Command::new(bin);
    cmd.arg("-m")
        .arg(model_path)
        .arg("--host")
        .arg("127.0.0.1")
        .arg("--port")
        .arg(port.to_string())
        .arg("-c")
        .arg(n_ctx.to_string())
        .arg("-t")
        .arg(threads.to_string())
        .arg("-ngl")
        .arg(ngl.to_string())
        // Avoid auto Flash-Attn / device mishmash that can pick broken backends.
        .arg("-fa")
        .arg("off")
        .arg("--alias")
        .arg(
            model_path
                .file_name()
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_else(|| "local-gguf".into()),
        );
    // Always pin device when ngl=0 so OpenVINO is not auto-selected for ops.
    // (OpenVINO on this host can load the model then SIGILL / Compute error on decode.)
    if ngl == 0 {
        cmd.arg("--device").arg("none");
        cmd.arg("--no-op-offload");
    }
    cmd.stdout(Stdio::from(
        log_file
            .try_clone()
            .map_err(|e| format!("[llama-server] log clone: {e}"))?,
    ))
    .stderr(Stdio::from(log_file));
    #[cfg(unix)]
    unsafe {
        cmd.pre_exec(|| {
            libc::setsid();
            Ok(())
        });
    }
    let mut child = cmd
        .spawn()
        .map_err(|e| format!("[llama-server] spawn failed: {}", e))?;

    write_pid_file(child.id(), port, model_path);

    let url = format!("http://127.0.0.1:{}", port);
    let deadline = Instant::now() + Duration::from_secs(wait_secs);
    let mut last_err = String::from("timeout");

    while Instant::now() < deadline {
        if let Ok(Some(status)) = child.try_wait() {
            clear_pid_file();
            let why = describe_exit(&status);
            let hint = if looks_like_sigill(&status) {
                ""
            } else if looks_like_oom(&status) {
                " Free RAM or lower Context."
            } else {
                ""
            };
            let log_tail = tail_log(&log_path, 400);
            let log_bit = if log_tail.is_empty() {
                String::new()
            } else {
                format!(" Log: {log_tail}")
            };
            return Err(format!(
                "[llama-server] exited during startup (ctx={n_ctx}, ngl={ngl}): {why}.{hint}{log_bit}"
            ));
        }
        if health_ok(&url).await {
            let name = model_path
                .file_name()
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_else(|| "local-gguf".into());
            let mut guard = MANAGED.lock().map_err(|e| e.to_string())?;
            write_pid_file(child.id(), port, model_path);
            *guard = Some(ManagedServer {
                model_path: model_path.to_path_buf(),
                port,
                child,
                ngl,
                n_ctx,
                power_mode,
            });
            return Ok((url, name));
        }
        last_err = format!("waiting for {}/health (ctx={n_ctx})", url);
        tokio::time::sleep(Duration::from_millis(500)).await;
    }

    kill_child_hard(&mut child);
    clear_pid_file();
    let log_tail = tail_log(&log_path, 400);
    Err(format!(
        "[llama-server] not healthy within {wait_secs}s ({last_err}; ctx={n_ctx}, ngl={ngl}). \
         Still loading or backend stuck.{}",
        if log_tail.is_empty() {
            String::new()
        } else {
            format!(" Log: {log_tail}")
        }
    ))
}

/// At most two full loads: RAM-safe ctx, then one step down if OOM/early exit.
async fn start_with_ctx_ladder(
    bin: &Path,
    model_path: &Path,
    threads: usize,
    ngl: i32,
    requested_ctx: usize,
    power_mode: crate::settings::PowerMode,
) -> Result<(String, String), String> {
    let ladder = ctx_fallback_ladder(requested_ctx, model_path);
    let safe = ram_safe_ctx(model_path);
    let mut last_err = String::new();
    let mut tried: Vec<usize> = Vec::new();

    for (i, &n_ctx) in ladder.iter().enumerate() {
        if i > 0 {
            // Previous child already killed inside try_start; wait for RSS reclaim.
            reclaim_pause().await;
            if mem_available_bytes() < model_bytes(model_path).saturating_add(512 * 1024 * 1024)
            {
                last_err = format!(
                    "MemAvailable too low after failed load (need headroom for weights). {last_err}"
                );
                break;
            }
        }
        tried.push(n_ctx);
        match try_start_server(bin, model_path, threads, ngl, n_ctx, power_mode).await {
            Ok(ok) => {
                if n_ctx != crate::settings::context_token_limit() {
                    crate::settings::set_context_token_limit(n_ctx);
                }
                return Ok(ok);
            }
            Err(e) => {
                last_err = e;
                // SIGILL is a bad binary for this CPU — lower ctx will crash the same way.
                if last_err.contains("SIGILL") {
                    break;
                }
                // Only retry lower ctx on OOM/early crash, not on health timeout.
                if !last_err.contains("SIGKILL")
                    && !last_err.contains("OOM")
                    && !last_err.contains("exited during startup")
                {
                    break;
                }
            }
        }
    }

    let model_mb = model_bytes(model_path) / (1024 * 1024);
    let avail_mb = mem_available_bytes() / (1024 * 1024);
    Err(format!(
        "[llama-server] failed (tried {}; RAM-safe≈{}; ngl={ngl}; threads={threads}). \
         weights≈{model_mb}MB, MemAvailable≈{avail_mb}MB. {last_err}. \
         Free RAM, Runtime → Power Saver + lower Context, or a smaller GGUF.",
        tried
            .iter()
            .map(|c| crate::settings::format_context_tokens(*c))
            .collect::<Vec<_>>()
            .join(" → "),
        crate::settings::format_context_tokens(safe),
    ))
}

async fn health_ok(base: &str) -> bool {
    let url = format!("{}/health", base.trim_end_matches('/'));
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(2))
        .build()
        .unwrap_or_else(|_| reqwest::Client::new());
    match client.get(&url).send().await {
        Ok(r) => r.status().is_success(),
        Err(_) => false,
    }
}

/// Stop the managed server + any orphan Hercules llama-server (call on app exit).
pub fn shutdown_managed_server() {
    // 1) Drop managed child (SIGTERM/KILL process group)
    if let Ok(mut g) = MANAGED.lock() {
        *g = None;
    }

    // 2) PID file from last start
    let pf = pid_file_path();
    if let Ok(text) = std::fs::read_to_string(&pf) {
        for line in text.lines() {
            if let Some(pid_s) = line.strip_prefix("pid=") {
                if let Ok(pid) = pid_s.trim().parse::<i32>() {
                    #[cfg(unix)]
                    unsafe {
                        libc::kill(pid, libc::SIGTERM);
                        libc::kill(-pid, libc::SIGTERM);
                    }
                    std::thread::sleep(Duration::from_millis(100));
                    #[cfg(unix)]
                    unsafe {
                        libc::kill(pid, libc::SIGKILL);
                        libc::kill(-pid, libc::SIGKILL);
                    }
                }
            }
        }
    }
    clear_pid_file();

    // 3) Best-effort: kill servers serving Hercules model dir or our port band
    let patterns = [
        "llama-server.*\\.local/hercules/model",
        "llama-server.*hercules/model",
        "llama-server.*--port 181",
        "/opt/llama.cpp/llama-server.*hercules",
    ];
    for pat in patterns {
        let _ = Command::new("pkill")
            .args(["-9", "-f", pat])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status();
    }
}

pub fn managed_server_info() -> Option<(PathBuf, u16, i32)> {
    MANAGED.lock().ok().and_then(|g| {
        g.as_ref()
            .map(|s| (s.model_path.clone(), s.port, s.ngl))
    })
}
