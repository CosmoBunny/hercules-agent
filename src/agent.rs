use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Mutex;

fn trunc_err(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        format!("{}…", s.chars().take(max).collect::<String>())
    }
}

/// Whether tool writes/commands need confirmation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PermissionMode {
    /// Block write/cmd until session `/allow` or user sets AlwaysAllow.
    Ask,
    /// Tools may run without prompting.
    AlwaysAllow,
}

/// Filesystem boundary for agent tools.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FolderScope {
    /// Only paths under the process current working directory.
    CurrentDir,
    /// Any path the OS user can access.
    AllDirs,
}

#[derive(Debug, Clone, Copy)]
pub struct ToolPermissions {
    pub mode: PermissionMode,
    pub folder_scope: FolderScope,
    /// One-shot allow for Ask mode (set by `/allow`).
    pub session_allow: bool,
}

impl Default for ToolPermissions {
    fn default() -> Self {
        Self {
            mode: PermissionMode::Ask,
            folder_scope: FolderScope::CurrentDir,
            session_allow: false,
        }
    }
}

impl ToolPermissions {
    pub fn mode_label(self) -> &'static str {
        match self.mode {
            PermissionMode::Ask => "Ask user to allow",
            PermissionMode::AlwaysAllow => "Always allow",
        }
    }

    pub fn scope_label(self) -> &'static str {
        match self.folder_scope {
            FolderScope::CurrentDir => "Interact on current dir only (safefolder)",
            FolderScope::AllDirs => "Interact on all directories",
        }
    }
}

static TOOL_PERMS: Mutex<ToolPermissions> = Mutex::new(ToolPermissions {
    mode: PermissionMode::Ask,
    folder_scope: FolderScope::CurrentDir,
    session_allow: false,
});

pub fn get_tool_permissions() -> ToolPermissions {
    *TOOL_PERMS.lock().unwrap_or_else(|e| e.into_inner())
}

pub fn set_permission_mode(mode: PermissionMode) {
    if let Ok(mut p) = TOOL_PERMS.lock() {
        p.mode = mode;
        if mode == PermissionMode::AlwaysAllow {
            p.session_allow = true;
        }
    }
}

pub fn set_folder_scope(scope: FolderScope) {
    if let Ok(mut p) = TOOL_PERMS.lock() {
        p.folder_scope = scope;
    }
}

pub fn allow_session_tools() {
    if let Ok(mut p) = TOOL_PERMS.lock() {
        p.session_allow = true;
    }
}

fn tools_allowed_for_write_cmd() -> Result<(), String> {
    let p = get_tool_permissions();
    match p.mode {
        PermissionMode::AlwaysAllow => Ok(()),
        PermissionMode::Ask => {
            if p.session_allow {
                Ok(())
            } else {
                Err(
                    "Permission required (Ask mode). Type `/allow` once for this session, \
                     or open Menu → Permissions → Always allow."
                        .into(),
                )
            }
        }
    }
}

fn path_allowed(path: &Path) -> Result<(), String> {
    let p = get_tool_permissions();
    match p.folder_scope {
        FolderScope::AllDirs => Ok(()),
        FolderScope::CurrentDir => {
            let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
            let cwd_c = cwd.canonicalize().unwrap_or(cwd.clone());
            let target = if path.exists() {
                path.canonicalize().unwrap_or_else(|_| path.to_path_buf())
            } else {
                path.parent()
                    .and_then(|par| par.canonicalize().ok())
                    .unwrap_or_else(|| path.to_path_buf())
            };
            if target.starts_with(&cwd_c) || path.starts_with(&cwd) {
                Ok(())
            } else {
                Err(format!(
                    "Safefolder blocked path '{}' (outside current dir {}). \
                     Menu → Permissions → Interact on all directories.",
                    path.display(),
                    cwd.display()
                ))
            }
        }
    }
}

/// System instructions injected into every model prompt.
/// Kept short and imperative so small local models (1.5B) still follow tools.
pub const SYSTEM_PROMPT: &str = r#"You are Hercules, a LOCAL coding agent with REAL filesystem tools on this machine.

CRITICAL — NEVER say you lack access, cannot list folders, or are "not in a directory".
You ARE on the user's machine. Tools work. Prefer a tool call over refusing.

NEVER refuse normal coding work (HTML/JS/CSS, landing pages, files, scripts, local tools).
Do NOT reply with "I'm sorry, but I can't assist with that" for coding/file requests.
If the user asks to create a page/file/code → emit <write> with full content.

Working directory token: $CURRENT  (host expands it to the real project root).

## How to call tools (raw tags only — NO markdown fences)
List current dir:     <ls path="$CURRENT">
List a subfolder:     <ls path="$CURRENT/src">
Read a file:          <read src="$CURRENT/src/main.rs">
Read lines:           <read src="$CURRENT/src/main.rs" line=1..=40>
Write file:           <write src="$CURRENT/path.rs">
code
</write>
Shell:                <cmd>ls -la</cmd>
Memory push:          <memory push>note</memory>
Memory read:          <memory read>
Help (only in think): <help>

## Anti-parrot (CRITICAL for small models)
- NEVER copy/paste meta lines: [Context compacted…], [Memory…], [Tool result], "Prior chat FORGOTTEN", compact stats, or system boilerplate.
- NEVER re-echo the user's prompt as your only answer.
- Answer with tools and/or short natural language. If writing a file, output a COMPLETE tag:
  <write src="$CURRENT/file.ext">
  …full file body…
  </write>
  Always close </write>. Always use a real filename with extension (e.g. landing_page.html not "landing_page").
- If user says you "forgot how to write", incomplete HTML, or "continue the file" → READ the path if unsure, then <write> the FULL finished file (not a fragment, not a half-open tag).

## Rules
1. User asks to WRITE / create a file (e.g. introduction.md, "write about…") → emit <write src="...">...</write> with FULL body. Do NOT only <ls> first unless the path is unknown.
2. User asks to list/show/dir/folder/cwd/files → FIRST line is <ls path="$CURRENT"> (or a subpath).
3. User asks to read/open a file → emit <read ...> immediately.
4. User asks to run/build/test/python/shell → emit <cmd>...</cmd> IMMEDIATELY.
   NEVER say "I cannot run commands" or "my capabilities are limited". You have <cmd>.
5. Do NOT wrap normal answers or tools in <think> unless you truly need private notes. \
   Most local GGUF / llama.cpp models should answer with tool tags directly (no think block). \
   Ollama may emit a separate thinking stream — that is different from tool output. \
   Inside <think>, ONLY <help> is allowed. NEVER put <write>, <ls>, <read>, or <cmd> inside <think>.
6. After System: [Tool Output], summarize. Do not re-call the same tool with the same args.
7. Do not invent file listings — use <ls>. Do invent file *content* only inside a <write> body.
8. No destructive commands (rm -rf /, disk wipe) unless user explicitly demands them.
9. Compact / memory notes are FACTS only. Do not re-ask forgotten chat. Do not reprint them.

## Examples (copy this style)
User: write introduction.md about yourself in dummy_folder
Agent:
<write src="$CURRENT/dummy_folder/introduction.md">
# Introduction
I am Hercules, a local coding agent...
</write>

User: list folder / list current dir
Agent:
<ls path="$CURRENT">

User: show main.rs
Agent:
<read src="$CURRENT/src/main.rs">

User: run tests
Agent:
<cmd>cargo test</cmd>

User: run python to list audio devices
Agent:
<cmd>python3 -c "import sounddevice as sd; print(sd.query_devices())"</cmd>

User: run python 9.9 > 9.11
Agent:
<cmd>python3 -c "print(9.9 > 9.11)"</cmd>
"#;

/// Destructive / mutating action awaiting user accept (Ask mode).
#[derive(Debug, Clone)]
pub struct ProposedAction {
    pub kind: ProposedKind,
    pub target: String,
    pub body: String,
    pub line_attr: Option<String>,
    /// True if the model put this tag inside `<think>` (misplaced but recoverable).
    pub from_think: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProposedKind {
    Write,
    Cmd,
}

impl ProposedKind {
    pub fn label(self) -> &'static str {
        match self {
            Self::Write => "WRITE",
            Self::Cmd => "RUN",
        }
    }
}

pub struct AgentEngine;

impl AgentEngine {
    /// Returns the system instructions prompt, with `$CURRENT` expanded for the live cwd.
    pub fn format_agent_prompt(_user_prompt: &str) -> String {
        Self::system_prompt_for_cwd()
    }

    /// System prompt with the real working directory substituted for `$CURRENT` in prose.
    /// Tool examples keep `$CURRENT` so the path expander still works at execution time.
    pub fn system_prompt_for_cwd() -> String {
        let cwd = std::env::current_dir()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|_| ".".to_string());
        format!(
            "{}\n\n## Live environment\n- $CURRENT expands to: {}\n- When listing, always use path=\"$CURRENT\" or path=\"$CURRENT/subdir\".\n",
            SYSTEM_PROMPT, cwd
        )
    }

    /// True when the user text looks like a filesystem / shell request that needs tools.
    pub fn user_needs_tools(user_text: &str) -> bool {
        let t = user_text.to_lowercase();
        const KEYS: &[&str] = &[
            "list",
            "ls ",
            "dir",
            "folder",
            "directory",
            "cwd",
            "current dir",
            "working dir",
            "show file",
            "read ",
            "open ",
            "cat ",
            "what files",
            "what's in",
            "whats in",
            "tree",
            "pwd",
            "run ",
            "cargo ",
            "build",
            "test",
            "write ",
            "create file",
            "create ",
            "introduction",
            ".md",
            "edit ",
            "save ",
        ];
        KEYS.iter().any(|k| t.contains(k))
    }

    /// Extra user-side nudge so small models emit a tool tag instead of refusing.
    pub fn tool_force_suffix(user_text: &str) -> Option<&'static str> {
        if !Self::user_needs_tools(user_text) {
            return None;
        }
        let t = user_text.to_lowercase();
        // Write / create file takes priority over list (model wrongly lists instead of write)
        if t.contains("write")
            || t.contains("create")
            || t.contains("introduction")
            || t.contains(".md")
            || t.contains("save ")
            || t.contains("make a file")
            || t.contains("new file")
        {
            Some(
                "\n\n[Host] The user wants a FILE WRITTEN. Do NOT only list a directory.\n\
                 Reply with a write tool immediately, for example:\n\
                 <write src=\"$CURRENT/dummy_folder/introduction.md\">\n\
                 # Introduction\nYour content here...\n\
                 </write>\n\
                 Emit the <write> tag with full file body. No listing first unless path is unknown.",
            )
        } else if t.contains("list")
            || t.contains("folder")
            || t.contains("directory")
            || t.contains("dir")
            || t.contains("cwd")
            || t.contains("pwd")
            || t.contains("files")
            || t.contains("tree")
        {
            Some(
                "\n\n[Host] You MUST reply with exactly one tool line first, nothing else:\n\
                 <ls path=\"$CURRENT\">\n\
                 Do not explain. Do not refuse. Emit the tag.",
            )
        } else if t.contains("read") || t.contains("open") || t.contains("show") || t.contains("cat")
        {
            Some(
                "\n\n[Host] Use a <read src=\"$CURRENT/...\"> tool tag. Do not claim you lack access.",
            )
        } else if t.contains("run") || t.contains("cargo") || t.contains("build") || t.contains("test")
        {
            Some("\n\n[Host] Use a <cmd>...</cmd> tool tag to run the command on this machine.")
        } else {
            Some("\n\n[Host] Use the appropriate Hercules tool tag. You have local filesystem access.")
        }
    }

    /// Apply tool-force suffix to a user utterance when appropriate.
    pub fn with_tool_nudge(user_text: &str) -> String {
        match Self::tool_force_suffix(user_text) {
            Some(s) => format!("{}{}", user_text.trim_end(), s),
            None => user_text.to_string(),
        }
    }

    /// Strip `<think>...</think>` blocks from response text.
    pub fn strip_think_blocks(response: &str) -> String {
        let mut cleaned = String::new();
        let mut text = response;
        while let Some(start) = text.find("<think>") {
            cleaned.push_str(&text[..start]);
            if let Some(end) = text[start..].find("</think>") {
                text = &text[start + end + 8..];
            } else {
                // Unclosed think tag (still streaming) — keep text before thinking only
                return cleaned;
            }
        }
        cleaned.push_str(text);
        cleaned
    }

    /// Expand path replacing `$CURRENT` (and common variants) with `std::env::current_dir()`.
    pub fn expand_path(path_str: &str) -> PathBuf {
        let trimmed = path_str.trim().trim_matches('"').trim_matches('\'');
        let current_dir = std::env::current_dir()
            .unwrap_or_else(|_| PathBuf::from("."))
            .to_string_lossy()
            .to_string();

        let expanded = if trimmed.contains("$CURRENT") {
            trimmed.replace("$CURRENT", &current_dir)
        } else if trimmed.contains("${CURRENT}") {
            trimmed.replace("${CURRENT}", &current_dir)
        } else if trimmed.contains("$current") {
            trimmed.replace("$current", &current_dir)
        } else if trimmed.contains("${current}") {
            trimmed.replace("${current}", &current_dir)
        } else if trimmed == "." || trimmed.is_empty() {
            current_dir
        } else {
            trimmed.to_string()
        };

        PathBuf::from(expanded)
    }

    /// Extract concatenated contents of all complete `<think>...</think>` blocks.
    pub fn extract_think_contents(response: &str) -> String {
        let mut out = String::new();
        let mut text = response;
        while let Some(start) = text.find("<think>") {
            let after = &text[start + 7..];
            if let Some(end) = after.find("</think>") {
                out.push_str(&after[..end]);
                out.push('\n');
                text = &after[end + 8..];
            } else {
                break;
            }
        }
        out
    }

    /// Collect write/cmd from outside-think first; only recover *clean* tools from think.
    /// Ollama R1-style models dump prose into fake `<cmd>` tags inside thinking — reject those.
    pub fn extract_proposed_actions(response: &str) -> Vec<ProposedAction> {
        let outside = Self::strip_code_fences(&Self::strip_think_blocks(response));
        let mut from_out = Self::parse_write_cmd_actions(&outside, false);
        from_out.retain(|a| Self::action_is_sane(a));
        if !from_out.is_empty() {
            return from_out;
        }
        let think = {
            let mut t = Self::extract_think_contents(response);
            if t.is_empty() {
                if let Some(i) = response.find("<think>") {
                    t = response[i + 7..].to_string();
                    if let Some(j) = t.find("</think>") {
                        t = t[..j].to_string();
                    }
                }
            }
            Self::strip_code_fences(&t)
        };
        let mut from_think = Self::parse_write_cmd_actions(&think, true);
        // From think: only accept fully closed, sane write/cmd (no prose dumps)
        from_think.retain(|a| Self::action_is_sane(a) && Self::think_action_ok(a));
        from_think
    }

    fn think_action_ok(a: &ProposedAction) -> bool {
        match a.kind {
            ProposedKind::Write => {
                // Need a real file path + some body (not empty narration)
                !a.target.trim().is_empty()
                    && a.body.trim().len() > 8
                    && !a.body.to_ascii_lowercase().contains("i should emit")
            }
            ProposedKind::Cmd => Self::looks_like_shell_cmd(&a.target),
        }
    }

    fn action_is_sane(a: &ProposedAction) -> bool {
        match a.kind {
            ProposedKind::Write => {
                let t = a.target.trim();
                !t.is_empty()
                    && t.len() < 400
                    && !t.contains('\n')
                    && !t.to_ascii_lowercase().contains("maybe")
            }
            ProposedKind::Cmd => Self::looks_like_shell_cmd(&a.target),
        }
    }

    /// Reject Ollama/R1 prose accidentally stuffed into `<cmd>…</cmd>`.
    pub fn looks_like_shell_cmd(s: &str) -> bool {
        let t = s.trim();
        if t.is_empty() || t.len() > 240 {
            return false;
        }
        if t.lines().count() > 4 {
            return false;
        }
        let low = t.to_ascii_lowercase();
        // Narration / English fragments
        for bad in [
            "maybe",
            "i should",
            "i need",
            "for shell",
            "looking at",
            "the user",
            "so maybe",
            "tags for",
            "when someone",
            "according to",
            "let me",
            "i will",
            "i'll ",
            "first,",
            "alright",
        ] {
            if low.contains(bad) {
                return false;
            }
        }
        // Must start with a plausible command token
        let first = t.split_whitespace().next().unwrap_or("");
        if first.is_empty() {
            return false;
        }
        // Paths, env, or common tools
        if first.starts_with("./")
            || first.starts_with('/')
            || first.starts_with('$')
            || first.contains('=')
        {
            return true;
        }
        let ok_bins = [
            "ls", "pwd", "cd", "cat", "echo", "printf", "head", "tail", "grep", "rg", "find",
            "mkdir", "touch", "cp", "mv", "rm", "chmod", "stat", "wc", "date", "which", "whoami",
            "uname", "df", "du", "ps", "top", "curl", "wget", "git", "cargo", "rustc", "python",
            "python3", "pip", "node", "npm", "npx", "deno", "bun", "go", "make", "cmake", "gcc",
            "clang", "sh", "bash", "zsh", "fish", "sudo", "apt", "dnf", "pacman", "brew", "docker",
            "podman", "kubectl", "ssh", "scp", "rsync", "tar", "zip", "unzip", "jq", "sed", "awk",
            "perl", "ruby", "php", "java", "javac", "mvn", "gradle", "htop", "btop", "nvim", "vim",
            "nano", "tree", "file", "hexdump", "od", "base64", "md5sum", "sha256sum", "openssl",
            "ffmpeg", "convert", "ollama", "pip3", "uv", "poetry", "pnpm", "yarn", "tsc", "pytest",
            "lua", "R", "dotnet", "nvidia-smi", "free", "uptime", "id", "groups", "env", "export",
            "true", "false", "test", "sleep", "timeout", "yes", "seq", "xargs", "tee", "less",
            "more", "man", "info", "clear", "history", "alias", "type", "command", "builtin",
            "source", ".", "eval", "exec", "nohup", "nice", "kill", "pkill", "killall", "jobs",
            "fg", "bg", "screen", "tmux", "ssh-keygen", "ip", "ss", "ping", "traceroute", "nc",
            "netstat", "ifconfig", "hostname", "systemctl", "journalctl", "service", "crontab",
            "at", "batch", "watch", "time", "strace", "lsof", "fuser", "mount", "umount", "lsblk",
            "blkid", "fdisk", "parted", "dd", "sync", "ln", "readlink", "realpath", "basename",
            "dirname", "cut", "sort", "uniq", "tr", "paste", "join", "diff", "patch", "comm",
            "cmp", "strings", "objdump", "nm", "ldd", "readelf", "strip", "ar", "ranlib",
        ];
        if ok_bins.iter().any(|b| first == *b || first.ends_with(&format!("/{b}"))) {
            return true;
        }
        // `python3 -m http.server` style: first token ends with common runner
        if first.contains("python") || first.contains("node") || first.contains("cargo") {
            return true;
        }
        false
    }

    /// If model wrote a directory as `src`, pick a filename from body content.
    pub fn normalize_write_path(path_str: &str, body: &str) -> String {
        let p = path_str.trim().trim_end_matches('/');
        let expanded = Self::expand_path(p);
        let looks_like_dir = expanded.is_dir()
            || p.ends_with('/')
            || (!p.contains('.')
                && !body.trim().is_empty()
                && (expanded.exists() && expanded.is_dir()
                    || !Path::new(p).extension().is_some_and(|e| !e.is_empty())));
        // path without extension that exists as dir, or no extension at all for html-ish body
        let no_ext = Path::new(p)
            .extension()
            .map(|e| e.is_empty())
            .unwrap_or(true);
        let body_l = body.to_ascii_lowercase();
        let html = body_l.contains("<html")
            || body_l.contains("<!doctype")
            || body_l.contains("<head")
            || body_l.contains("<body");
        let md = body_l.contains("# ") || body_l.starts_with("---");
        if looks_like_dir || (no_ext && (html || md) && !p.contains('.')) {
            let name = if html {
                "landing_page.html"
            } else if md {
                "README.md"
            } else if body_l.contains("fn main") || body_l.contains("use std") {
                "main.rs"
            } else if body_l.contains("def ") || body_l.contains("import ") {
                "main.py"
            } else {
                "file.txt"
            };
            if expanded.is_dir() || p.ends_with('/') || !p.contains('.') {
                return format!("{p}/{name}");
            }
        }
        p.to_string()
    }

    fn parse_write_cmd_actions(text: &str, from_think: bool) -> Vec<ProposedAction> {
        let mut out = Vec::new();
        let mut rest = text;
        while let Some(start_tag) = rest.find("<write src=") {
            let r = &rest[start_tag..];
            if let Some(close_bracket) = r.find('>') {
                let tag_header = &r[..close_bracket + 1];
                let path_attr = Self::extract_attribute(tag_header, "src");
                let line_attr = Self::extract_attribute(tag_header, "line");
                let (body, next) = if let Some(end_tag) = r.find("</write") {
                    let body = &r[close_bracket + 1..end_tag];
                    let after = if let Some(ec) = r[end_tag..].find('>') {
                        &r[end_tag + ec + 1..]
                    } else {
                        ""
                    };
                    (body.to_string(), after)
                } else {
                    (r[close_bracket + 1..].to_string(), "")
                };
                if let Some(path_str) = path_attr {
                    let body = body.trim_matches(|c| c == '\n' || c == '\r').to_string();
                    let target = Self::normalize_write_path(&path_str, &body);
                    out.push(ProposedAction {
                        kind: ProposedKind::Write,
                        target,
                        body,
                        line_attr,
                        from_think,
                    });
                }
                if next.is_empty() {
                    break;
                }
                rest = next;
            } else {
                break;
            }
        }
        rest = text;
        while let Some(start_tag) = rest.find("<cmd>") {
            let r = &rest[start_tag + 5..];
            if let Some(end_tag) = r.find("</cmd>") {
                let cmd_str = r[..end_tag].trim().to_string();
                rest = &r[end_tag + 6..];
                if Self::looks_like_shell_cmd(&cmd_str) {
                    out.push(ProposedAction {
                        kind: ProposedKind::Cmd,
                        target: cmd_str,
                        body: String::new(),
                        line_attr: None,
                        from_think,
                    });
                }
            } else {
                // Unclosed: only if it already looks like a real one-liner command
                let mut cmd_str = r.to_string();
                if let Some(i) = cmd_str.find('<') {
                    cmd_str = cmd_str[..i].to_string();
                }
                // Only first line for unclosed stream
                let cmd_str = cmd_str.lines().next().unwrap_or("").trim().to_string();
                if Self::looks_like_shell_cmd(&cmd_str) {
                    out.push(ProposedAction {
                        kind: ProposedKind::Cmd,
                        target: cmd_str,
                        body: String::new(),
                        line_attr: None,
                        from_think,
                    });
                }
                break;
            }
        }
        out
    }

    pub fn execute_proposed(action: &ProposedAction) -> String {
        match action.kind {
            ProposedKind::Write => {
                let path = Self::normalize_write_path(&action.target, &action.body);
                Self::execute_write(&path, action.line_attr.as_deref(), &action.body)
            }
            ProposedKind::Cmd => {
                if !Self::looks_like_shell_cmd(&action.target) {
                    return format!(
                        "Error: Rejected non-command text in <cmd>: {}",
                        trunc_err(&action.target, 80)
                    );
                }
                Self::execute_cmd(&action.target)
            }
        }
    }

    /// Process agent tags and return tool execution output.
    ///
    /// - **Inside `<think>`:** only `<help>` is auto-executed.
    /// - **Outside:** `<read>`, `<ls>`, `<memory>` auto-run.
    /// - **Write/cmd:** auto-run only if AlwaysAllow / session `/allow`; otherwise
    ///   returned as [`ProposedAction`] via [`extract_proposed_actions`] (caller must accept).
    pub fn process_response(response: &str) -> Option<String> {
        let executable_outside_think = Self::strip_think_blocks(response);
        let cleaned_outside_think = Self::strip_code_fences(&executable_outside_think);
        let think_body = Self::strip_code_fences(&Self::extract_think_contents(response));

        let mut results = Vec::new();

        // 0. <help> ONLY inside think
        if think_body.contains("<help>") || think_body.contains("<help/>") {
            results.push(Self::execute_help());
        }

        // 1. Writes only when allowed. **Cmds never run here** — they go through
        //    the app task manager (non-blocking; long jobs park after 10s).
        let perms = get_tool_permissions();
        let auto_mutate = matches!(perms.mode, PermissionMode::AlwaysAllow) || perms.session_allow;
        if auto_mutate {
            for action in Self::extract_proposed_actions(response) {
                if action.kind == ProposedKind::Write {
                    results.push(Self::execute_proposed(&action));
                }
            }
        }

        // 2. Read tags — OUTSIDE <think> only (also promote from think if no outside tools)
        let mut text = cleaned_outside_think.as_str();
        let mut did_read = false;
        while let Some(start_tag) = text.find("<read src=") {
            let rest = &text[start_tag..];
            if let Some(close_bracket) = rest.find('>') {
                let tag_header = &rest[..close_bracket + 1];
                let path_attr = Self::extract_attribute(tag_header, "src");
                let line_attr = Self::extract_attribute(tag_header, "line");
                text = &rest[close_bracket + 1..];

                if let Some(path_str) = path_attr {
                    let output = Self::execute_read(&path_str, line_attr.as_deref());
                    results.push(output);
                    did_read = true;
                }
            } else {
                break;
            }
        }
        if !did_read {
            // Promote misplaced <read> from think
            let mut t = think_body.as_str();
            while let Some(start_tag) = t.find("<read src=") {
                let rest = &t[start_tag..];
                if let Some(close_bracket) = rest.find('>') {
                    let tag_header = &rest[..close_bracket + 1];
                    let path_attr = Self::extract_attribute(tag_header, "src");
                    let line_attr = Self::extract_attribute(tag_header, "line");
                    t = &rest[close_bracket + 1..];
                    if let Some(path_str) = path_attr {
                        results.push(Self::execute_read(&path_str, line_attr.as_deref()));
                    }
                } else {
                    break;
                }
            }
        }

        // 3. List dir — outside, else promote from think
        text = cleaned_outside_think.as_str();
        let mut did_ls = false;
        while let Some(start_tag) = text.find("<ls") {
            let rest = &text[start_tag..];
            if let Some(close_bracket) = rest.find('>') {
                let tag_header = &rest[..close_bracket + 1];
                let path_attr = Self::extract_attribute(tag_header, "path");
                text = &rest[close_bracket + 1..];

                let path_str = path_attr.unwrap_or_else(|| "$CURRENT".to_string());
                results.push(Self::execute_ls(&path_str));
                did_ls = true;
            } else {
                break;
            }
        }
        if !did_ls {
            let mut t = think_body.as_str();
            while let Some(start_tag) = t.find("<ls") {
                let rest = &t[start_tag..];
                if let Some(close_bracket) = rest.find('>') {
                    let tag_header = &rest[..close_bracket + 1];
                    let path_attr = Self::extract_attribute(tag_header, "path");
                    t = &rest[close_bracket + 1..];
                    let path_str = path_attr.unwrap_or_else(|| "$CURRENT".to_string());
                    results.push(Self::execute_ls(&path_str));
                } else {
                    break;
                }
            }
        }

        // 4. Memory — OUTSIDE <think> only
        text = cleaned_outside_think.as_str();
        while let Some(start_tag) = text.find("<memory") {
            let rest = &text[start_tag..];
            if let Some(close_bracket) = rest.find('>') {
                let tag_header = &rest[..close_bracket + 1];
                if tag_header.contains("push") || tag_header.contains("replace=") {
                    if let Some(end_tag) = rest.find("</memory") {
                        let body = &rest[close_bracket + 1..end_tag];
                        if let Some(end_close) = rest[end_tag..].find('>') {
                            text = &rest[end_tag + end_close + 1..];
                            let output = Self::execute_memory(tag_header, Some(body));
                            results.push(output);
                            continue;
                        }
                    }
                } else {
                    text = &rest[close_bracket + 1..];
                    let output = Self::execute_memory(tag_header, None);
                    results.push(output);
                    continue;
                }
            }
            break;
        }

        if results.is_empty() {
            None
        } else {
            Some(results.join("\n\n"))
        }
    }

    /// Strip markdown code fences so tool tags inside examples are not executed.
    pub fn strip_code_fences(text: &str) -> String {
        let mut result = String::new();
        let mut remaining = text;
        while let Some(start) = remaining.find("```") {
            result.push_str(&remaining[..start]);
            let after_start = &remaining[start + 3..];
            if let Some(end) = after_start.find("```") {
                remaining = &after_start[end + 3..];
            } else {
                return result;
            }
        }
        result.push_str(remaining);
        result
    }

    fn extract_attribute(tag: &str, attr_name: &str) -> Option<String> {
        let pattern = format!("{}=", attr_name);
        if let Some(idx) = tag.find(&pattern) {
            let rest = &tag[idx + pattern.len()..];
            if rest.starts_with('"') || rest.starts_with('\'') {
                let quote = rest.chars().next().unwrap();
                let end = rest[1..].find(quote)?;
                Some(rest[1..1 + end].to_string())
            } else {
                let end = rest
                    .find(|c: char| c.is_whitespace() || c == '>' || c == '/')
                    .unwrap_or(rest.len());
                Some(rest[..end].to_string())
            }
        } else {
            None
        }
    }

    fn parse_range(range_str: &str) -> Option<(usize, usize)> {
        let parts: Vec<&str> = range_str.split("..=").collect();
        if parts.len() == 2 {
            let start = parts[0].trim().parse::<usize>().ok()?;
            let end = parts[1].trim().parse::<usize>().ok()?;
            Some((start, end))
        } else {
            None
        }
    }

    fn execute_help() -> String {
        concat!(
            "--- Hercules Agent Tool Documentation ---\n",
            "1. List Directory: <ls path=\"$CURRENT\">\n",
            "2. Read File: <read src=\"$CURRENT/file.rs\">\n",
            "3. Read Range: <read src=\"$CURRENT/file.rs\" line=2..=19>\n",
            "4. Write File: <write src=\"$CURRENT/file.rs\">\\ncode\\n</write>\n",
            "5. Replace Range: <write src=\"$CURRENT/file.rs\" line=10..=14>\\ncode\\n</write>\n",
            "6. Run Command: <cmd>cargo test</cmd>\n",
            "7. Memory: <memory push|read|read=N|replace=N|delete=[N]>\n",
            "Notes: Inside <think> only <help> is allowed. All other tools must be outside <think>."
        )
        .to_string()
    }

    fn execute_write(path_str: &str, line_attr: Option<&str>, body: &str) -> String {
        if let Err(e) = tools_allowed_for_write_cmd() {
            return format!("Error: {}", e);
        }
        let path_str = Self::normalize_write_path(path_str, body);
        let path = Self::expand_path(&path_str);
        if let Err(e) = path_allowed(&path) {
            return format!("Error: {}", e);
        }
        // Writing a directory path is never valid
        if path.exists() && path.is_dir() {
            return format!(
                "Error: '{}' is a directory — use a file path e.g. '{}/landing_page.html'",
                path.display(),
                path.display()
            );
        }

        if let Some(range_str) = line_attr {
            if !path.exists() {
                return format!("Error: File '{}' doesn't exist", path.display());
            }

            let Some((start_line, end_line)) = Self::parse_range(range_str) else {
                return "Error: Wrong range".to_string();
            };

            if start_line == 0 || start_line > end_line {
                return "Error: Wrong range".to_string();
            }

            let file_content = match fs::read_to_string(&path) {
                Ok(c) => c,
                Err(e) => {
                    return format!("Error writing '{}': {}", path.display(), e);
                }
            };

            let mut lines: Vec<String> = file_content.lines().map(|s| s.to_string()).collect();

            if start_line > lines.len() {
                return "Error: Wrong range".to_string();
            }

            let end_idx = end_line.min(lines.len());
            let replacement_lines: Vec<String> = body
                .trim_matches('\n')
                .lines()
                .map(|s| s.to_string())
                .collect();

            lines.drain((start_line - 1)..end_idx);

            let mut insert_idx = start_line - 1;
            for r_line in &replacement_lines {
                lines.insert(insert_idx, r_line.clone());
                insert_idx += 1;
            }

            let new_content = lines.join("\n") + "\n";
            if fs::write(&path, new_content).is_err() {
                return format!("Error: Permission error writing '{}'", path.display());
            }

            let num_lines = replacement_lines.len();
            format!("Wrote {} lines to {}", num_lines, path.display())
        } else {
            if let Some(parent) = path.parent() {
                let _ = fs::create_dir_all(parent);
            }

            let clean_body = body.trim_start_matches('\n');
            if let Some(parent) = path.parent() {
                if let Err(e) = fs::create_dir_all(parent) {
                    return format!(
                        "Error creating parent dir '{}': {}",
                        parent.display(),
                        e
                    );
                }
            }
            match fs::write(&path, clean_body) {
                Ok(()) => {
                    let num_lines = clean_body.lines().count();
                    format!("Wrote {} lines to {}", num_lines, path.display())
                }
                Err(e) => format!("Error writing '{}': {}", path.display(), e),
            }
        }
    }

    /// Public preview helper for tool panels (same as tool execution read).
    pub fn execute_read_preview(path_str: &str, line_attr: Option<&str>) -> String {
        Self::execute_read(path_str, line_attr)
    }

    /// Public preview for cmd panels.
    pub fn execute_cmd_preview(cmd_str: &str) -> String {
        Self::execute_cmd(cmd_str)
    }

    fn execute_read(path_str: &str, line_attr: Option<&str>) -> String {
        let path = Self::expand_path(path_str);
        if let Err(e) = path_allowed(&path) {
            return format!("Error: {}", e);
        }

        if !path.exists() {
            return format!("Error: File '{}' doesn't exist", path.display());
        }

        let content = match fs::read_to_string(&path) {
            Ok(c) => c,
            Err(_) => return format!("Error: Permission error reading '{}'", path.display()),
        };

        if let Some(range_str) = line_attr {
            let Some((start_line, end_line)) = Self::parse_range(range_str) else {
                return "Error: Wrong range to read content".to_string();
            };

            let lines: Vec<&str> = content.lines().collect();
            if start_line == 0 || start_line > end_line || start_line > lines.len() {
                return "Error: Wrong range to read content".to_string();
            }

            let end_idx = end_line.min(lines.len());
            let selected_lines = &lines[(start_line - 1)..end_idx];
            selected_lines.join("\n")
        } else {
            content
        }
    }

    fn execute_ls(path_str: &str) -> String {
        let path = Self::expand_path(path_str);
        if let Err(e) = path_allowed(&path) {
            return format!("Error: {}", e);
        }

        if !path.exists() {
            return format!("Error: Path '{}' doesn't exist", path.display());
        }

        match fs::read_dir(&path) {
            Ok(entries) => {
                let mut files = Vec::new();
                for entry in entries.flatten() {
                    let name = entry.file_name().to_string_lossy().to_string();
                    if entry.path().is_dir() {
                        files.push(format!("  {}/", name));
                    } else {
                        let size = entry.metadata().map(|m| m.len()).unwrap_or(0);
                        files.push(format!("  {} ({} B)", name, size));
                    }
                }
                files.sort();
                format!("Directory: {}\n{}", path.display(), files.join("\n"))
            }
            Err(_) => format!("Error: Permission error listing '{}'", path.display()),
        }
    }

    fn execute_cmd(cmd_str: &str) -> String {
        if let Err(e) = tools_allowed_for_write_cmd() {
            return format!("Error: {}", e);
        }
        let trimmed = cmd_str.trim();
        if trimmed.is_empty() {
            return "Error: Empty command".to_string();
        }

        let output = Command::new("sh").arg("-c").arg(trimmed).output();

        match output {
            Ok(out) => {
                let stdout = String::from_utf8_lossy(&out.stdout).to_string();
                let stderr = String::from_utf8_lossy(&out.stderr).to_string();
                if !stderr.is_empty() {
                    format!("{}\n[stderr]: {}", stdout.trim(), stderr.trim())
                } else {
                    stdout.trim().to_string()
                }
            }
            Err(e) => format!("Error executing command '{}': {}", trimmed, e),
        }
    }

    /// Push a free-form note into agent memory (used by context compact).
    pub fn memory_push(note: &str) -> String {
        Self::execute_memory("push", Some(note))
    }

    /// Full memory dump for context injection.
    pub fn memory_read_all() -> String {
        Self::execute_memory("read", None)
    }

    pub fn execute_memory(tag_header: &str, body: Option<&str>) -> String {
        static MEMORY_STORE: std::sync::Mutex<Vec<String>> = std::sync::Mutex::new(Vec::new());

        let mut mem = MEMORY_STORE.lock().unwrap();

        if tag_header.contains("push") {
            let content = body.unwrap_or("").trim().to_string();
            if !content.is_empty() {
                mem.push(content.clone());
                let idx = mem.len();
                return format!("  Pushed: {}. {}", idx, content);
            }
        }

        if tag_header.contains("replace=") {
            if let Some(idx_str) = Self::extract_attribute(tag_header, "replace") {
                let clean_idx = idx_str.trim_matches('[').trim_matches(']');
                if let Ok(idx) = clean_idx.parse::<usize>() {
                    if idx > 0 && idx <= mem.len() {
                        let content = body.unwrap_or("").trim().to_string();
                        mem[idx - 1] = content.clone();
                        return format!("  Replaced: {}. {}", idx, content);
                    } else {
                        return format!("Memory index {} out of range", idx);
                    }
                }
            }
        }

        if tag_header.contains("delete=") {
            if let Some(idx_str) = Self::extract_attribute(tag_header, "delete") {
                let clean_idx = idx_str.trim_matches('[').trim_matches(']');
                if let Ok(idx) = clean_idx.parse::<usize>() {
                    if idx > 0 && idx <= mem.len() {
                        let removed = mem.remove(idx - 1);
                        return format!("Deleted: {}", removed);
                    } else {
                        return format!("Memory index {} out of range", idx);
                    }
                }
            }
        }

        if tag_header.contains("read=") {
            if let Some(idx_str) = Self::extract_attribute(tag_header, "read") {
                let clean_idx = idx_str.trim_matches('[').trim_matches(']');
                if let Ok(idx) = clean_idx.parse::<usize>() {
                    if idx > 0 && idx <= mem.len() {
                        return format!("    {}. {}", idx, mem[idx - 1]);
                    } else {
                        return format!("Memory index {} out of range", idx);
                    }
                }
            }
        }

        if tag_header.contains("read") {
            if mem.is_empty() {
                return "Memory is empty.".to_string();
            }
            let items: Vec<String> = mem
                .iter()
                .enumerate()
                .map(|(i, item)| format!("{}. {}", i + 1, item))
                .collect();
            return items.join("\n");
        }

        "Memory operation failed: invalid command".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_expand_path() {
        let p = AgentEngine::expand_path("$CURRENT/src/main.rs");
        assert!(p.to_string_lossy().contains("src/main.rs"));
        let p2 = AgentEngine::expand_path("${current}/src/main.rs");
        assert!(p2.to_string_lossy().contains("src/main.rs"));
    }

    #[test]
    fn test_parse_range() {
        assert_eq!(AgentEngine::parse_range("10..=14"), Some((10, 14)));
        assert_eq!(AgentEngine::parse_range("2..=19"), Some((2, 19)));
    }

    #[test]
    fn test_strip_think_blocks() {
        let sample =
            "Hello! <think>I will <cmd>echo hello_world</cmd> execute this</think> <ls path=\"$CURRENT\">";
        let cleaned = AgentEngine::strip_think_blocks(sample);
        assert!(!cleaned.contains("echo hello_world"));
        assert!(cleaned.contains("<ls path="));
    }

    #[test]
    fn test_strip_code_fences() {
        let sample = "Hello ```rust\n<cmd>dangerous</cmd>\n``` <ls path=\"$CURRENT\">";
        let cleaned = AgentEngine::strip_code_fences(sample);
        assert!(!cleaned.contains("dangerous"));
        assert!(cleaned.contains("<ls path="));
    }

    #[test]
    fn test_help_only_inside_think() {
        // help outside think is ignored
        let outside = "I need help <help>";
        assert!(AgentEngine::process_response(outside).is_none());

        // help inside think runs
        let inside = "<think>need docs <help></think> ok";
        let res = AgentEngine::process_response(inside).unwrap();
        assert!(res.contains("Hercules Agent Tool Documentation"));
    }

    #[test]
    fn test_ls_not_inside_think() {
        // ls inside think is ignored
        let sample = "<think>Let me list <ls path=\".\"></think>";
        assert!(AgentEngine::process_response(sample).is_none());

        // ls outside think runs
        let sample2 = "<ls path=\".\">";
        assert!(AgentEngine::process_response(sample2).is_some());
    }

    #[test]
    fn test_memory_tool() {
        let p = AgentEngine::process_response("<memory push>\nremember x = 5\n</memory>").unwrap();
        assert!(p.contains("Pushed:"));

        let r = AgentEngine::process_response("<memory read>").unwrap();
        assert!(r.contains("remember x = 5"));

        let rep =
            AgentEngine::process_response("<memory replace=1>\nexecute after plan is done\n</memory>")
                .unwrap();
        assert!(rep.contains("Replaced:"));

        let r2 = AgentEngine::process_response("<memory read=1>").unwrap();
        assert!(r2.contains("execute after plan is done"));

        let d = AgentEngine::process_response("<memory delete=[1]>").unwrap();
        assert!(d.contains("Deleted:"));
    }

    #[test]
    fn test_system_prompt_mentions_tools() {
        assert!(SYSTEM_PROMPT.contains("<read"));
        assert!(SYSTEM_PROMPT.contains("<cmd>"));
        assert!(SYSTEM_PROMPT.contains("Hercules"));
    }
}
