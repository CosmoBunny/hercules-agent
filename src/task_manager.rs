//! Background task manager for long-running shell commands.
//!
//! Commands that finish within [`QUICK_SECS`] return normally.
//! If still running after that, they are parked as task #N; the agent is notified
//! and continues. When the process exits, the full output is delivered to the UI/LLM.
//! Ctrl+C sets kill flags so process groups are terminated.

use std::io::Read;
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// After this many seconds of still-running cmd → push to task manager.
pub const QUICK_SECS: u64 = 10;

static NEXT_ID: AtomicU32 = AtomicU32::new(1);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaskStatus {
    Running,
    Done,
    Failed,
    Killed,
}

#[derive(Debug, Clone)]
pub struct ManagedTask {
    pub id: u32,
    pub cmd: String,
    pub status: TaskStatus,
    pub started: Instant,
    pub output: String,
    /// True once we told the UI/agent this was parked as a long task.
    pub parked_notified: bool,
    pub spawned_by: u32,
}

#[derive(Debug, Clone)]
pub enum TaskEvent {
    /// Cmd exceeded QUICK_SECS — still running as task #id
    Parked { id: u32, cmd: String, spawned_by: u32 },
    /// Process finished (or killed)
    Done {
        id: u32,
        cmd: String,
        output: String,
        killed: bool,
        spawned_by: u32,
    },
}

/// Shared registry of background jobs.
#[derive(Clone, Default)]
pub struct TaskManager {
    inner: Arc<Mutex<TaskManagerInner>>,
}

#[derive(Default)]
struct TaskManagerInner {
    tasks: Vec<ManagedTask>,
    events: Vec<TaskEvent>,
    /// Kill switches per task id
    kills: Vec<(u32, Arc<AtomicBool>)>,
}

static GLOBAL_TASK_MGR: Mutex<Option<TaskManager>> = Mutex::new(None);

pub fn set_global_task_manager(mgr: TaskManager) {
    if let Ok(mut g) = GLOBAL_TASK_MGR.lock() {
        *g = Some(mgr);
    }
}

pub fn get_global_task_manager() -> Option<TaskManager> {
    GLOBAL_TASK_MGR.lock().ok()?.clone()
}

impl TaskManager {
    pub fn new() -> Self {
        let mgr = Self {
            inner: Arc::new(Mutex::new(TaskManagerInner::default())),
        };
        set_global_task_manager(mgr.clone());
        mgr
    }

    pub fn list(&self) -> Vec<ManagedTask> {
        self.inner
            .lock()
            .map(|g| g.tasks.clone())
            .unwrap_or_default()
    }

    pub fn running_count(&self) -> usize {
        self.inner
            .lock()
            .map(|g| {
                g.tasks
                    .iter()
                    .filter(|t| t.status == TaskStatus::Running)
                    .count()
            })
            .unwrap_or(0)
    }

    pub fn take_events(&self) -> Vec<TaskEvent> {
        self.inner
            .lock()
            .map(|mut g| std::mem::take(&mut g.events))
            .unwrap_or_default()
    }

    /// Kill all running tasks (Ctrl+C).
    pub fn kill_all(&self) {
        if let Ok(mut g) = self.inner.lock() {
            for (_, flag) in &g.kills {
                flag.store(true, Ordering::SeqCst);
            }
            for t in &mut g.tasks {
                if t.status == TaskStatus::Running {
                    t.status = TaskStatus::Killed;
                    if t.output.is_empty() {
                        t.output = "[killed by user (Ctrl+C)]".into();
                    }
                }
            }
        }
    }

    pub fn get_task_output(&self, id: u32) -> Option<(String, TaskStatus)> {
        let g = self.inner.lock().ok()?;
        let task = g.tasks.iter().find(|t| t.id == id)?;
        Some((task.output.clone(), task.status))
    }

    pub fn kill_task(&self, id: u32) -> bool {
        if let Ok(mut g) = self.inner.lock() {
            let mut found = false;
            for (tid, flag) in &g.kills {
                if *tid == id {
                    flag.store(true, Ordering::SeqCst);
                    found = true;
                }
            }
            if let Some(t) = g.tasks.iter_mut().find(|t| t.id == id) {
                if t.status == TaskStatus::Running {
                    t.status = TaskStatus::Killed;
                    found = true;
                }
            }
            found
        } else {
            false
        }
    }

    /// Spawn `cmd` in the background. Returns task id immediately.
    /// Completion / park events are queued for the main loop.
    pub fn spawn_cmd(&self, cmd: String, spawned_by: u32) -> u32 {
        let id = NEXT_ID.fetch_add(1, Ordering::SeqCst);
        let kill = Arc::new(AtomicBool::new(false));
        {
            if let Ok(mut g) = self.inner.lock() {
                g.tasks.push(ManagedTask {
                    id,
                    cmd: cmd.clone(),
                    status: TaskStatus::Running,
                    started: Instant::now(),
                    output: String::new(),
                    parked_notified: false,
                    spawned_by,
                });
                g.kills.push((id, kill.clone()));
            }
        }

        let mgr = self.clone();
        let cmd_for_thread = cmd.clone();
        std::thread::Builder::new()
            .name(format!("hercules-task-{id}"))
            .spawn(move || {
                let result = run_cmd_managed(&cmd_for_thread, id, kill, mgr.clone());
                if let Ok(mut g) = mgr.inner.lock() {
                    let mut sby = 0;
                    if let Some(t) = g.tasks.iter_mut().find(|t| t.id == id) {
                        sby = t.spawned_by;
                        match &result {
                            Ok((out, killed)) => {
                                t.output = out.clone();
                                t.status = if *killed {
                                    TaskStatus::Killed
                                } else {
                                    TaskStatus::Done
                                };
                            }
                            Err(e) => {
                                t.output = e.clone();
                                t.status = TaskStatus::Failed;
                            }
                        }
                    }
                    let (output, killed) = match result {
                        Ok((o, k)) => (o, k),
                        Err(e) => (e, false),
                    };
                    g.events.push(TaskEvent::Done {
                        id,
                        cmd: cmd_for_thread,
                        output,
                        killed,
                        spawned_by: sby,
                    });
                    g.kills.retain(|(i, _)| *i != id);
                }
            })
            .unwrap();

        id
    }

    pub fn spawn_agent(&self, backend: crate::backend::AgentBackend, role: String, to: String, model: String, instruction: String, spawned_by: u32) -> u32 {
        let id = NEXT_ID.fetch_add(1, Ordering::SeqCst);
        let kill = Arc::new(AtomicBool::new(false));
        let mut cmd = format!("agent role=\"{role}\"");
        if !model.is_empty() {
            cmd.push_str(&format!(" model=\"{model}\""));
        }
        if !to.is_empty() {
            cmd.push_str(&format!(" to=\"{to}\""));
        }
        {
            if let Ok(mut g) = self.inner.lock() {
                g.tasks.push(ManagedTask {
                    id,
                    cmd: cmd.clone(),
                    status: TaskStatus::Running,
                    started: Instant::now(),
                    output: String::new(),
                    parked_notified: false,
                    spawned_by,
                });
                g.kills.push((id, kill.clone()));
            }
        }

        let mgr = self.clone();
        std::thread::Builder::new()
            .name(format!("hercules-agent-{id}"))
            .spawn(move || {
                let model_tag = if model.is_empty() { String::new() } else { format!(" [Model: {model}]") };
                let prompt = format!("You are an AI sub-agent{model_tag}. Role: {role}\nInstruction: {instruction}");
                let rt = tokio::runtime::Runtime::new().unwrap();
                let output = match rt.block_on(backend.generate(&prompt)) {
                    Ok(res) => format!("<agent action=\"reply\" to=\"{spawned_by}\">\n{}\n</agent>", res),
                    Err(e) => format!("<agent action=\"error\">\n{}\n</agent>", e),
                };

                if let Ok(mut g) = mgr.inner.lock() {
                    let mut sby = 0;
                    if let Some(t) = g.tasks.iter_mut().find(|t| t.id == id) {
                        t.status = TaskStatus::Done;
                        t.output = output.clone();
                        sby = t.spawned_by;
                    }
                    g.events.push(TaskEvent::Done {
                        id,
                        output,
                        killed: false,
                        cmd,
                        spawned_by: sby,
                    });
                }
            })
            .unwrap();

        id
    }

    fn mark_parked(&self, id: u32, cmd: &str) {
        if let Ok(mut g) = self.inner.lock() {
            let mut sby = 0;
            if let Some(t) = g.tasks.iter_mut().find(|t| t.id == id) {
                if t.parked_notified {
                    return;
                }
                t.parked_notified = true;
                sby = t.spawned_by;
            }
            g.events.push(TaskEvent::Parked {
                id,
                cmd: cmd.to_string(),
                spawned_by: sby,
            });
        }
    }

    /// Append live output preview to task (optional, for TERM panel).
    pub fn append_output_preview(&self, id: u32, chunk: &str) {
        if let Ok(mut g) = self.inner.lock() {
            if let Some(t) = g.tasks.iter_mut().find(|t| t.id == id) {
                if t.output.len() < 32_000 {
                    t.output.push_str(chunk);
                }
            }
        }
    }

    pub fn get(&self, id: u32) -> Option<ManagedTask> {
        self.inner
            .lock()
            .ok()
            .and_then(|g| g.tasks.iter().find(|t| t.id == id).cloned())
    }
}

fn run_cmd_managed(
    cmd: &str,
    id: u32,
    kill: Arc<AtomicBool>,
    mgr: TaskManager,
) -> Result<(String, bool), String> {
    let mut child = spawn_shell(cmd).map_err(|e| format!("spawn failed: {e}"))?;
    let start = Instant::now();
    let mut parked = false;

    // Drain stdout/stderr on side threads so pipes don't block
    let stdout = child.stdout.take();
    let stderr = child.stderr.take();
    let out_buf = Arc::new(Mutex::new(String::new()));
    let err_buf = Arc::new(Mutex::new(String::new()));
    let out_c = out_buf.clone();
    let err_c = err_buf.clone();
    let mgr_out = mgr.clone();
    if let Some(mut so) = stdout {
        std::thread::spawn(move || {
            let mut buf = [0u8; 4096];
            loop {
                match so.read(&mut buf) {
                    Ok(0) => break,
                    Ok(n) => {
                        let s = String::from_utf8_lossy(&buf[..n]);
                        if let Ok(mut g) = out_c.lock() {
                            g.push_str(&s);
                        }
                        mgr_out.append_output_preview(id, &s);
                    }
                    Err(_) => break,
                }
            }
        });
    }
    if let Some(mut se) = stderr {
        std::thread::spawn(move || {
            let mut buf = [0u8; 4096];
            loop {
                match se.read(&mut buf) {
                    Ok(0) => break,
                    Ok(n) => {
                        let s = String::from_utf8_lossy(&buf[..n]);
                        if let Ok(mut g) = err_c.lock() {
                            g.push_str(&s);
                        }
                    }
                    Err(_) => break,
                }
            }
        });
    }

    loop {
        if kill.load(Ordering::SeqCst) {
            kill_child_tree(&mut child);
            let out = combine_out(&out_buf, &err_buf);
            let msg = if out.trim().is_empty() {
                format!("[Task #{id} killed by user (Ctrl+C)]")
            } else {
                format!("{out}\n[Task #{id} killed by user (Ctrl+C)]")
            };
            return Ok((msg, true));
        }

        match child.try_wait() {
            Ok(Some(status)) => {
                // Give pipe threads a moment
                std::thread::sleep(Duration::from_millis(30));
                let mut out = combine_out(&out_buf, &err_buf);
                if out.trim().is_empty() {
                    out = format!("(no output, exit {status})");
                } else if !status.success() {
                    out.push_str(&format!("\n[exit {status}]"));
                }
                return Ok((out, false));
            }
            Ok(None) => {
                if !parked && start.elapsed() >= Duration::from_secs(QUICK_SECS) {
                    parked = true;
                    mgr.mark_parked(id, cmd);
                }
                std::thread::sleep(Duration::from_millis(80));
            }
            Err(e) => {
                kill_child_tree(&mut child);
                return Err(format!("wait error: {e}"));
            }
        }
    }
}

fn combine_out(out_buf: &Arc<Mutex<String>>, err_buf: &Arc<Mutex<String>>) -> String {
    let stdout = out_buf.lock().map(|g| g.clone()).unwrap_or_default();
    let stderr = err_buf.lock().map(|g| g.clone()).unwrap_or_default();
    if stderr.trim().is_empty() {
        stdout
    } else if stdout.trim().is_empty() {
        format!("[stderr]\n{stderr}")
    } else {
        format!("{stdout}\n[stderr]\n{stderr}")
    }
}

fn spawn_shell(cmd: &str) -> std::io::Result<Child> {
    let mut c = Command::new("sh");
    c.arg("-c")
        .arg(cmd)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        unsafe {
            c.pre_exec(|| {
                libc::setsid();
                Ok(())
            });
        }
    }
    c.spawn()
}

fn kill_child_tree(child: &mut Child) {
    let pid = child.id();
    #[cfg(unix)]
    {
        let p = pid as i32;
        unsafe {
            libc::kill(-p, libc::SIGTERM);
            libc::kill(p, libc::SIGTERM);
        }
        std::thread::sleep(Duration::from_millis(100));
        unsafe {
            libc::kill(-p, libc::SIGKILL);
            libc::kill(p, libc::SIGKILL);
        }
    }
    let _ = child.kill();
    let _ = child.wait();
}
