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
/// Kept short and imperative so local models follow tools and user intent.
pub const SYSTEM_PROMPT: &str = r#"You are Hercules, an autonomous agent with real filesystem, shell, and sub-agent tools on this machine.

CRITICAL — NEVER say you lack access or cannot perform actions. You are on the user's system and your tools are functional.

You can assist with any task: coding, researching, writing, analyzing, building, managing files, running commands, and orchestrating sub-agents.
When performing file or shell operations, emit the appropriate tool tags directly.

Working directory token: $CURRENT (host expands it to the real project root).

## How to call tools (raw tags only — NO markdown fences)
List dir:             <ls path="$CURRENT">
List a subfolder:     <ls path="$CURRENT/subdir">
Read a file:          <read src="$CURRENT/path/to/file">
Read lines:           <read src="$CURRENT/path/to/file" line=1..=40>
Write file:           <write src="$CURRENT/path/to/file">
content
</write>
Shell:                <cmd>shell command</cmd>
Spawn sub-agent:      <agent action="spawn" role="ROLE" model="MODEL">task</agent>
Memory push:          <memory push>note</memory>
Memory read:          <memory read>
Help (only in think): <help>

## Rules
1. If the user asks to create or write content to a file → emit <write src="...">...</write> with full body.
2. If the user asks to read/inspect files or directories → emit <read ...> or <ls ...> immediately.
3. If the user asks to run commands, scripts, or programs → emit <cmd>...</cmd> immediately.
4. If the user asks a general question, chat, or request not requiring tools → answer directly in natural language.
5. Do NOT wrap normal answers or tools in <think> unless performing private reasoning.
6. After tool results are provided, summarize or answer the user directly. Do not re-call the same tool repeatedly.
7. No destructive commands (e.g. rm -rf /, disk wipe) unless user explicitly requests them.
"#;

/// Compact system prompt for small GGUFs / llama-server chat.
pub const SYSTEM_PROMPT_COMPACT: &str = r#"You are Hercules, an autonomous agent on the user's machine with real tools.

cwd token: $CURRENT (host expands it).

Tools (raw tags only, no markdown fences):
<ls path="$CURRENT">
<ls path="$CURRENT/subdir">
<read src="$CURRENT/path/to/file">
<read src="$CURRENT/path/to/file" line=1..=40>
<write src="$CURRENT/path/to/file">
content
</write>
<cmd>shell command</cmd>
<agent action="spawn" role="role" model="model">task</agent>
<memory push>note</memory>
<memory read>

Rules:
- Questions, conversation, reasoning → answer directly in natural language.
- Reading or inspecting files → use <read> or <ls>.
- Creating or editing files → use <write src="...">content</write>.
- Running shell commands → use <cmd>command</cmd>.
- Never say you lack file access or tools.
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
    pub chip_id: Option<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProposedKind {
    Write,
    Cmd,
    Mcp,
    Skill,
    WebSearch,
    Agent,
}

impl ProposedKind {
    pub fn label(self) -> &'static str {
        match self {
            Self::Write => "WRITE",
            Self::Cmd => "RUN",
            Self::Mcp => "MCP",
            Self::Skill => "SKILL",
            Self::WebSearch => "WEBSEARCH",
            Self::Agent => "AGENT",
        }
    }
}

pub struct AgentEngine;

impl AgentEngine {
    pub fn format_markdown_tables(text: &str, max_width: usize, scroll_x: usize) -> String {
        let mut out = String::new();
        let mut table_lines = Vec::new();
        
        fn render_table(lines: &[&str], out: &mut String, max_width: usize, scroll_x: usize) {
            if lines.is_empty() { return; }
            let mut rows: Vec<Vec<String>> = Vec::new();
            for line in lines {
                let mut parts: Vec<&str> = line.split('|').collect();
                if let Some(first) = parts.first() {
                    if first.trim().is_empty() {
                        parts.remove(0);
                    }
                }
                if let Some(last) = parts.last() {
                    if last.trim().is_empty() {
                        parts.pop();
                    }
                }
                let mut row = Vec::new();
                for p in parts {
                    row.push(p.trim().to_string());
                }
                if !row.is_empty() {
                    rows.push(row);
                }
            }
            if rows.is_empty() { return; }
            let mut has_sep = false;
            if rows.len() > 1 {
                let is_sep = rows[1].iter().all(|c| !c.is_empty() && c.chars().all(|ch| ch == '-' || ch == ':' || ch == ' '));
                if is_sep {
                    has_sep = true;
                    rows.remove(1);
                }
            }
            let cols = rows.iter().map(|r| r.len()).max().unwrap_or(0);
            let mut widths = vec![0; cols];
            for row in &rows {
                for (i, col) in row.iter().enumerate() {
                    widths[i] = widths[i].max(col.chars().count());
                }
            }
            let top = format!("┌{}┐", widths.iter().map(|w| "─".repeat(*w + 2)).collect::<Vec<_>>().join("┬"));
            let mut table_rows = vec![top];
            
            for (r_idx, row) in rows.iter().enumerate() {
                let mut row_str = String::from("│");
                for i in 0..cols {
                    let text = row.get(i).map(|s| s.as_str()).unwrap_or("");
                    let padding = widths[i].saturating_sub(text.chars().count());
                    row_str.push_str(&format!(" {} {}│", text, " ".repeat(padding)));
                }
                table_rows.push(row_str);
                if r_idx == 0 && has_sep {
                    let mid = format!("├{}┤", widths.iter().map(|w| "─".repeat(*w + 2)).collect::<Vec<_>>().join("┼"));
                    table_rows.push(mid);
                }
            }
            let bot = format!("└{}┘", widths.iter().map(|w| "─".repeat(*w + 2)).collect::<Vec<_>>().join("┴"));
            table_rows.push(bot);
            
            let table_width = table_rows[0].chars().count();
            let eff_width = max_width.saturating_sub(2); // leaving margin
            let needs_scroll = table_width > eff_width;
            let actual_scroll = if needs_scroll { scroll_x.min(table_width - eff_width) } else { 0 };
            
            for r in table_rows {
                let chars: Vec<char> = r.chars().collect();
                if needs_scroll {
                    let start = actual_scroll;
                    let end = (start + eff_width).min(chars.len());
                    if start < chars.len() {
                        let slice: String = chars[start..end].iter().collect();
                        out.push_str(&slice);
                    }
                } else {
                    out.push_str(&r);
                }
                out.push('\n');
            }
            
            if needs_scroll {
                // Windows 98 style scrollbar: [◄][████░░░░░░░][►]
                let track_len = eff_width.saturating_sub(6).max(5);
                let thumb_size = (track_len as f64 * (eff_width as f64 / table_width as f64)).max(1.0) as usize;
                let thumb_pos = (track_len as f64 * (actual_scroll as f64 / (table_width - eff_width) as f64)).min((track_len - thumb_size) as f64) as usize;
                
                let mut scrollbar = String::from("[◄][");
                for i in 0..track_len {
                    if i >= thumb_pos && i < thumb_pos + thumb_size {
                        scrollbar.push('█');
                    } else {
                        scrollbar.push('░');
                    }
                }
                scrollbar.push_str("][►]\n");
                out.push_str(&scrollbar);
            }
        }

        let mut in_table = false;
        for line in text.lines() {
            let is_table_row = line.trim().starts_with('|') && line.trim().contains('|');
            if is_table_row {
                table_lines.push(line);
                in_table = true;
            } else {
                if in_table {
                    render_table(&table_lines, &mut out, max_width, scroll_x);
                    table_lines.clear();
                    in_table = false;
                }
                out.push_str(line);
                out.push('\n');
            }
        }
        if in_table {
            render_table(&table_lines, &mut out, max_width, scroll_x);
        }
        if !text.ends_with('\n') && out.ends_with('\n') {
            out.pop();
        }
        out
    }

    /// Returns the system instructions prompt, with `$CURRENT` expanded for the live cwd.
    pub fn format_agent_prompt(_user_prompt: &str) -> String {
        Self::system_prompt_for_cwd()
    }

    /// System prompt with the real working directory substituted for `$CURRENT` in prose.
    /// Tool examples keep `$CURRENT` so the path expander still works at execution time.
    pub fn system_prompt_for_cwd() -> String {
        Self::system_prompt_for_cwd_with(SYSTEM_PROMPT)
    }

    /// Short system prompt for weak local models (avoids instruction-recital).
    pub fn system_prompt_compact_for_cwd() -> String {
        Self::system_prompt_for_cwd_with(SYSTEM_PROMPT_COMPACT)
    }

    fn system_prompt_for_cwd_with(base: &str) -> String {
        let cwd = std::env::current_dir()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|_| ".".to_string());
        format!(
            "{base}\n\nLive: $CURRENT → {cwd}. Prefer path=\"$CURRENT\" or path=\"$CURRENT/subdir\".\n"
        )
    }

    /// Strip chat special tokens models sometimes emit into the reply.
    pub fn sanitize_model_output(text: &str) -> String {
        let mut s = text.to_string();
        for tok in [
            "<|im_end|>",
            "<|im_start|>",
            "<|endoftext|>",
            "<|EOT|>",
            "</s>",
            "<s>",
            "[INST]",
            "[/INST]",
            "<<SYS>>",
            "<</SYS>>",
        ] {
            s = s.replace(tok, "");
        }
        s
    }

    /// True when the model is clearly reciting the system/instructions block.
    pub fn looks_like_system_echo(text: &str) -> bool {
        let t = text.to_ascii_lowercase();
        let hits = [
            "you are hercules",
            "local coding agent with real filesystem",
            "critical — never say you lack access",
            "critical - never say you lack access",
            "working directory token",
            "how to call tools",
            "anti-parrot",
            "never refuse normal coding work",
        ]
        .iter()
        .filter(|k| t.contains(**k))
        .count();
        hits >= 2 || (t.contains("you are hercules") && t.contains("<ls path="))
    }

    /// User asked for a plan / design first — pure chat, no forced tools.
    pub fn wants_plan_first(user_text: &str) -> bool {
        let t = user_text.to_ascii_lowercase();
        let planish = t.contains("plan")
            || t.contains("approach")
            || t.contains("outline")
            || t.contains("steps before");
        let before_code = t.contains("before coding")
            || t.contains("before you code")
            || t.contains("before writing")
            || t.contains("don't code yet")
            || t.contains("do not code")
            || t.contains("without coding")
            || t.contains("tell me your plan")
            || t.contains("what's your plan")
            || t.contains("what is your plan");
        (planish && (before_code || t.contains("before") || t.contains("first")))
            || before_code
            || t.contains("just plan")
            || t.contains("only plan")
    }

    /// User wants implementation / file creation (write), not directory listing.
    pub fn wants_implement(user_text: &str) -> bool {
        let t = user_text.to_ascii_lowercase();
        if Self::wants_plan_first(user_text) {
            return false;
        }
        t.contains("start coding")
            || t.contains("start writing")
            || t.contains("implement")
            || t.contains("write the code")
            || t.contains("write code")
            || t.contains("begin coding")
            || t.contains("go ahead and code")
            || t.contains("now code")
            || (t.contains("create")
                && (t.contains("page")
                    || t.contains("app")
                    || t.contains("site")
                    || t.contains("store")
                    || t.contains("component")
                    || t.contains("project")
                    || t.contains("html")
                    || t.contains("svelte")
                    || t.contains("react")
                    || t.contains("file")))
            || (t.contains("build") && (t.contains("page") || t.contains("app") || t.contains("ui")))
    }

    /// True when the user text looks like a filesystem / shell request that needs tools.
    /// Intentionally does NOT match bare "create a store page" — that is implement/write,
    /// and forcing tools made the model spam `<ls>` forever.
    pub fn user_needs_tools(user_text: &str) -> bool {
        if Self::wants_plan_first(user_text) {
            return false;
        }
        let t = user_text.to_lowercase();
        const KEYS: &[&str] = &[
            "list files",
            "list dir",
            "list the",
            "ls ",
            "ls\n",
            "dir ",
            "folder",
            "directory",
            "cwd",
            "current dir",
            "working dir",
            "show file",
            "read ",
            "read\n",
            "open ",
            "cat ",
            "what files",
            "what's in",
            "whats in",
            "contents of",
            "tree",
            "pwd",
            "run ",
            "cargo ",
            "create file",
            "write file",
            "write a file",
            ".toml",
            "edit ",
            "save ",
        ];
        KEYS.iter().any(|k| t.contains(k)) || Self::wants_implement(user_text)
    }

    /// Model replied in natural language claiming no file/shell access (ignore tools).
    pub fn looks_like_capability_refusal(text: &str) -> bool {
        let low = text.to_ascii_lowercase();
        const PHRASES: &[&str] = &[
            "don't have the ability",
            "do not have the ability",
            "don't have access",
            "do not have access",
            "cannot read file",
            "can't read file",
            "cannot read files",
            "can't read files",
            "cannot access",
            "can't access",
            "no access to your",
            "no access to the file",
            "unable to read",
            "unable to access",
            "i cannot open",
            "i can't open",
            "as an ai",
            "as a language model",
            "i don't have the ability to directly",
            "i do not have the ability to directly",
        ];
        PHRASES.iter().any(|p| low.contains(p))
    }

    /// True if the assistant text already contains an executable tool tag.
    pub fn response_has_tool_tags(text: &str) -> bool {
        let t = text;
        t.contains("<read src=")
            || t.contains("<ls path=")
            || t.contains("<ls>")
            || t.contains("<write src=")
            || t.contains("<cmd>")
            || t.contains("<memory ")
            || t.contains("<memory>")
    }

    /// True when the user wants a file tool but never named a path/filename.
    /// (e.g. "can you read file?", "read a file") — must NOT invent names.
    pub fn wants_read_without_path(user_text: &str) -> bool {
        let low = user_text.to_ascii_lowercase();
        let asks_read = low.contains("read")
            || low.contains("open ")
            || low.contains("show ")
            || low.contains("cat ");
        if !asks_read {
            return false;
        }
        // Has an explicit path-like token?
        Self::extract_path_candidate(user_text).is_none()
    }

    /// First path-like token from user text, if any (no invented defaults).
    pub fn extract_path_candidate(user_text: &str) -> Option<String> {
        let t = user_text.trim();
        let low = t.to_ascii_lowercase();

        let mut rest: Option<&str> = None;
        for v in ["read ", "open ", "show ", "cat ", "print "] {
            if let Some(i) = low.find(v) {
                rest = Some(t[i + v.len()..].trim());
                break;
            }
        }
        if rest.is_none() {
            for needle in [
                "can you read ",
                "please read ",
                "could you read ",
                "would you read ",
                "can you open ",
                "please open ",
            ] {
                if let Some(i) = low.find(needle) {
                    rest = Some(t[i + needle.len()..].trim());
                    break;
                }
            }
        }

        // Bare filename as the whole message: "Cargo.toml", "src/main.rs"
        let search = rest.unwrap_or(t);
        let candidate = search
            .split(|c: char| c.is_whitespace() || c == '?' || c == '"' || c == '\'')
            .map(str::trim)
            .find(|s| {
                if s.is_empty() {
                    return false;
                }
                let low_c = s.to_ascii_lowercase();
                if matches!(
                    low_c.as_str(),
                    "the"
                        | "a"
                        | "an"
                        | "file"
                        | "files"
                        | "it"
                        | "this"
                        | "that"
                        | "please"
                        | "me"
                        | "some"
                        | "any"
                        | "my"
                        | "your"
                ) {
                    return false;
                }
                // Require path shape: extension, slash, or $CURRENT — not bare words
                s.contains('.') || s.contains('/') || s.starts_with('$')
            })?;
        Some(candidate.to_string())
    }

    /// Best-effort tool tag from user text. Never invents a filename.
    /// Pathless "read a file" → `<ls>` so the model can pick a real name next.
    pub fn synthesize_tool_from_user(user_text: &str) -> Option<String> {
        let t = user_text.trim();
        let low = t.to_ascii_lowercase();

        // list / ls / dir / cwd / pwd / tree
        if low == "ls"
            || low == "dir"
            || low == "pwd"
            || low == "tree"
            || low.contains("list files")
            || low.contains("list dir")
            || low.contains("list the")
            || low.contains("current dir")
            || low.contains("working dir")
            || low.contains("what's in")
            || low.contains("whats in")
            || low.contains("what files")
        {
            return Some(r#"<ls path="$CURRENT">"#.into());
        }

        // "can you read file?" / "read a file" with no real path → list, don't invent
        if Self::wants_read_without_path(user_text) {
            return Some(r#"<ls path="$CURRENT">"#.into());
        }

        let candidate = Self::extract_path_candidate(user_text)?;
        let path = if candidate.starts_with("$CURRENT") || candidate.starts_with('/') {
            candidate
        } else {
            format!("$CURRENT/{candidate}")
        };
        Some(format!(r#"<read src="{path}">"#))
    }

    /// Short host force line for tool-needed turns (kept tiny for 1.5B models).
    /// Paths come only from the user message when recoverable — never a fixed demo file.
    pub fn tool_force_suffix(user_text: &str) -> Option<String> {
        // Plan-first: never force tools (that caused endless `<ls>`).
        if Self::wants_plan_first(user_text) {
            return Some(
                "[Host] User asked for a PLAN first. Reply in natural language only. \
                 No tool tags (<ls>, <read>, <write>, <cmd>) until they say to code."
                    .into(),
            );
        }
        // Implement / create page → write, never list-only.
        if Self::wants_implement(user_text) {
            let path = Self::suggested_path_from_user_text(user_text);
            return Some(format!(
                "[Host] Implement now. Emit ONE <write src=\"{path}\">…full content…</write> \
                 (or a few real project files). Do NOT only <ls>. Do NOT re-list the directory."
            ));
        }
        if !Self::user_needs_tools(user_text) {
            return None;
        }
        if Self::wants_read_without_path(user_text) {
            return Some(
                "[Host] No filename was given. Emit ONLY:\n\
                 <ls path=\"$CURRENT\">\n\
                 Do NOT invent filenames."
                    .into(),
            );
        }
        if let Some(tag) = Self::synthesize_tool_from_user(user_text) {
            // Never force-feed bare <ls> for non-list asks
            if tag.contains("<ls") && !user_text.to_ascii_lowercase().contains("list") {
                return None;
            }
            return Some(format!(
                "[Host] Emit ONLY this tool tag (path taken from the user message):\n{tag}"
            ));
        }
        // Concrete read/list/run only — do NOT default to ls
        None
    }

    /// Append a short tool-force line when the user clearly needs filesystem/shell tools.
    /// Without this, 1–3B models often answer “I can’t read files” despite the system prompt.
    pub fn with_tool_nudge(user_text: &str) -> String {
        match Self::tool_force_suffix(user_text) {
            Some(sfx) => format!("{user_text}\n\n{sfx}"),
            None => user_text.to_string(),
        }
    }

    /// If the model refused tools, build a synthetic response that is pure tool tags.
    /// Returns `None` when no recovery is possible.
    pub fn recover_tools_from_refusal(user_text: &str, assistant_text: &str) -> Option<String> {
        if Self::response_has_tool_tags(assistant_text) {
            return None;
        }
        if Self::wants_plan_first(user_text) || Self::wants_implement(user_text) {
            // Don't inject <ls> for plan/create — that is the spam loop.
            return None;
        }
        if !Self::user_needs_tools(user_text) {
            return None;
        }
        // Recover only for explicit read/list with a recoverable tag.
        let should = Self::looks_like_capability_refusal(assistant_text)
            || Self::synthesize_tool_from_user(user_text).is_some();
        if !should {
            return None;
        }
        let tag = Self::synthesize_tool_from_user(user_text)?;
        if tag.contains("<ls") && !user_text.to_ascii_lowercase().contains("list") {
            return None;
        }
        Some(tag)
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
        from_out.extend(Self::parse_mcp_skill_actions(&outside, false));
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
        from_think.extend(Self::parse_mcp_skill_actions(&think, true));
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
            ProposedKind::Mcp | ProposedKind::Skill | ProposedKind::WebSearch | ProposedKind::Agent => true,
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
            ProposedKind::Mcp | ProposedKind::Skill | ProposedKind::WebSearch | ProposedKind::Agent => true,
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
            "ls",
            "pwd",
            "cd",
            "cat",
            "echo",
            "printf",
            "head",
            "tail",
            "grep",
            "rg",
            "find",
            "mkdir",
            "touch",
            "cp",
            "mv",
            "rm",
            "chmod",
            "stat",
            "wc",
            "date",
            "which",
            "whoami",
            "uname",
            "df",
            "du",
            "ps",
            "top",
            "curl",
            "wget",
            "git",
            "cargo",
            "rustc",
            "python",
            "python3",
            "pip",
            "node",
            "npm",
            "npx",
            "deno",
            "bun",
            "go",
            "make",
            "cmake",
            "gcc",
            "clang",
            "sh",
            "bash",
            "zsh",
            "fish",
            "sudo",
            "apt",
            "dnf",
            "pacman",
            "brew",
            "docker",
            "podman",
            "kubectl",
            "ssh",
            "scp",
            "rsync",
            "tar",
            "zip",
            "unzip",
            "jq",
            "sed",
            "awk",
            "perl",
            "ruby",
            "php",
            "java",
            "javac",
            "mvn",
            "gradle",
            "htop",
            "btop",
            "nvim",
            "vim",
            "nano",
            "tree",
            "file",
            "hexdump",
            "od",
            "base64",
            "md5sum",
            "sha256sum",
            "openssl",
            "ffmpeg",
            "convert",
            "ollama",
            "pip3",
            "uv",
            "poetry",
            "pnpm",
            "yarn",
            "tsc",
            "pytest",
            "lua",
            "R",
            "dotnet",
            "nvidia-smi",
            "free",
            "uptime",
            "id",
            "groups",
            "env",
            "export",
            "true",
            "false",
            "test",
            "sleep",
            "timeout",
            "yes",
            "seq",
            "xargs",
            "tee",
            "less",
            "more",
            "man",
            "info",
            "clear",
            "history",
            "alias",
            "type",
            "command",
            "builtin",
            "source",
            ".",
            "eval",
            "exec",
            "nohup",
            "nice",
            "kill",
            "pkill",
            "killall",
            "jobs",
            "fg",
            "bg",
            "screen",
            "tmux",
            "ssh-keygen",
            "ip",
            "ss",
            "ping",
            "traceroute",
            "nc",
            "netstat",
            "ifconfig",
            "hostname",
            "systemctl",
            "journalctl",
            "service",
            "crontab",
            "at",
            "batch",
            "watch",
            "time",
            "strace",
            "lsof",
            "fuser",
            "mount",
            "umount",
            "lsblk",
            "blkid",
            "fdisk",
            "parted",
            "dd",
            "sync",
            "ln",
            "readlink",
            "realpath",
            "basename",
            "dirname",
            "cut",
            "sort",
            "uniq",
            "tr",
            "paste",
            "join",
            "diff",
            "patch",
            "comm",
            "cmp",
            "strings",
            "objdump",
            "nm",
            "ldd",
            "readelf",
            "strip",
            "ar",
            "ranlib",
        ];
        if ok_bins
            .iter()
            .any(|b| first == *b || first.ends_with(&format!("/{b}")))
        {
            return true;
        }
        // `python3 -m http.server` style: first token ends with common runner
        if first.contains("python") || first.contains("node") || first.contains("cargo") {
            return true;
        }
        false
    }

    /// Suggest a write path from the latest user wording (no hardcoded landing page).
    pub fn suggested_path_from_user_text(user_text: &str) -> String {
        let t = user_text.to_ascii_lowercase();
        let name = if (t.contains("rotat") || t.contains("spin")) && t.contains("cube") {
            "rotating_cube.html"
        } else if t.contains("cube")
            && (t.contains("html") || t.contains("js") || t.contains("webgl"))
        {
            "cube.html"
        } else if t.contains("three.js") || t.contains("threejs") || t.contains("webgl") {
            "webgl_demo.html"
        } else if t.contains("canvas") {
            "canvas_demo.html"
        } else if t.contains("landing") {
            "landing_page.html"
        } else if t.contains("introduction") || t.contains("intro.md") {
            "introduction.md"
        } else if t.contains("readme") {
            "README.md"
        } else if t.contains(".md") || t.contains("markdown") {
            "notes.md"
        } else if t.contains("python") || t.contains(".py") {
            "main.py"
        } else if t.contains("rust") || t.contains(".rs") {
            "main.rs"
        } else if t.contains("html") || t.contains("javascript") || t.contains(" js") {
            "index.html"
        } else {
            "output.txt"
        };
        // Prefer dummy_folder for demo HTML/md when user didn't specify a path
        if name.ends_with(".html") || name.ends_with(".md") {
            format!("$CURRENT/dummy_folder/{name}")
        } else {
            format!("$CURRENT/{name}")
        }
    }

    /// Filename slug from free text (title / topic).
    fn slugify_filename(raw: &str, max_len: usize) -> String {
        let mut out = String::new();
        for c in raw.chars() {
            if c.is_ascii_alphanumeric() {
                out.push(c.to_ascii_lowercase());
            } else if c.is_whitespace() || c == '-' || c == '_' {
                if !out.ends_with('_') && !out.is_empty() {
                    out.push('_');
                }
            }
            if out.len() >= max_len {
                break;
            }
        }
        let out = out.trim_matches('_').to_string();
        if out.is_empty() { "file".into() } else { out }
    }

    /// Infer a sensible filename from file body (never force landing_page.html).
    pub fn infer_filename_from_body(body: &str) -> String {
        let body_l = body.to_ascii_lowercase();
        // <title>…</title>
        if let Some(i) = body_l.find("<title") {
            if let Some(gt) = body[i..].find('>') {
                let rest = &body[i + gt + 1..];
                if let Some(end) = rest.to_ascii_lowercase().find("</title") {
                    let title = rest[..end].trim();
                    if !title.is_empty() && title.len() < 80 {
                        let slug = Self::slugify_filename(title, 40);
                        return format!("{slug}.html");
                    }
                }
            }
        }
        if (body_l.contains("rotat") || body_l.contains("requestanimationframe"))
            && (body_l.contains("cube") || body_l.contains("webgl") || body_l.contains("three"))
        {
            return "rotating_cube.html".into();
        }
        if body_l.contains("webgl") || body_l.contains("three.js") || body_l.contains("three.min") {
            return "webgl_demo.html".into();
        }
        if body_l.contains("<canvas") {
            return "canvas_demo.html".into();
        }
        let html = body_l.contains("<html")
            || body_l.contains("<!doctype")
            || body_l.contains("<head")
            || body_l.contains("<body");
        if html {
            return "index.html".into();
        }
        if body_l.contains("# ") || body_l.starts_with("---") {
            return "README.md".into();
        }
        if body_l.contains("fn main") || body_l.contains("use std") {
            return "main.rs".into();
        }
        if body_l.contains("def ") || body_l.contains("import ") {
            return "main.py".into();
        }
        // Avoid "file.txt" mid-stream — prefer index.html for HTML-ish / empty drafts
        if body.trim().is_empty() || body.len() < 40 {
            return "index.html".into();
        }
        "notes.txt".into()
    }

    /// Collapse multi-write spam for a single-page user request (landing/html page).
    /// Keeps the best HTML write (largest body with doctype/html), drops junk file.txt.
    pub fn collapse_write_actions_for_user(
        user_text: &str,
        actions: Vec<ProposedAction>,
    ) -> Vec<ProposedAction> {
        let writes: Vec<_> = actions
            .iter()
            .filter(|a| a.kind == ProposedKind::Write)
            .cloned()
            .collect();
        let others: Vec<_> = actions
            .into_iter()
            .filter(|a| a.kind != ProposedKind::Write)
            .collect();
        if writes.len() <= 1 {
            let mut out = writes;
            out.extend(others);
            return out;
        }
        let low = user_text.to_ascii_lowercase();
        let single_page = user_text.is_empty()
            || low.contains("landing")
            || low.contains("html")
            || low.contains("web page")
            || low.contains("webpage")
            || low.contains("about you")
            || (low.contains("create") && (low.contains("page") || low.contains("site")));
        // Without user text: only collapse when junk .txt sits next to real HTML
        if user_text.is_empty() {
            let has_html = writes.iter().any(|w| {
                let b = w.body.to_ascii_lowercase();
                b.contains("<html") || b.contains("<!doctype")
            });
            let has_junk = writes.iter().any(|w| {
                let n = w.target.to_ascii_lowercase();
                (n.ends_with(".txt") || n.ends_with("file.txt")) && w.body.len() < 200
            });
            if !(has_html && has_junk) {
                let mut out = writes;
                out.extend(others);
                return out;
            }
        } else if !single_page {
            let mut out = writes;
            out.extend(others);
            return out;
        }
        // Score: prefer closed HTML with real content
        let best = writes.into_iter().max_by_key(|w| {
            let b = w.body.to_ascii_lowercase();
            let mut score = w.body.len() as i64;
            if b.contains("<!doctype") || b.contains("<html") {
                score += 50_000;
            }
            if b.contains("<style") || b.contains("<body") {
                score += 10_000;
            }
            let name = w.target.to_ascii_lowercase();
            if name.ends_with("index.html") || name.ends_with("landing_page.html") {
                score += 5_000;
            }
            if name.ends_with("file.txt") || name.ends_with("notes.txt") || name.ends_with(".txt") {
                score -= 20_000;
            }
            score
        });
        let mut out = Vec::new();
        if let Some(w) = best {
            // Force a clean path for landing pages
            let mut w = w;
            if !w.target.ends_with(".html") {
                w.target = Self::suggested_path_from_user_text(user_text);
            }
            out.push(w);
        }
        out.extend(others);
        out
    }

    /// If model wrote a directory as `src` (or a generic/wrong name), pick a better file path.
    pub fn normalize_write_path(path_str: &str, body: &str) -> String {
        Self::normalize_write_path_with_hint(path_str, body, None)
    }

    /// Same as [`normalize_write_path`], optional user-utterance hint for naming.
    pub fn normalize_write_path_with_hint(
        path_str: &str,
        body: &str,
        user_hint: Option<&str>,
    ) -> String {
        let p = path_str.trim().trim_end_matches('/');
        let expanded = Self::expand_path(p);
        let base_name = Path::new(p)
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("")
            .to_ascii_lowercase();

        // Known bad/generic defaults the model (or old host text) loves to emit.
        let generic_html = matches!(
            base_name.as_str(),
            "landing_page.html"
                | "landing.html"
                | "page.html"
                | "file.html"
                | "output.html"
                | "index.htm"
        );

        let looks_like_dir = expanded.is_dir()
            || p.ends_with('/')
            || (!p.contains('.')
                && (expanded.exists() && expanded.is_dir()
                    || !Path::new(p).extension().is_some_and(|e| !e.is_empty())));
        let no_ext = Path::new(p)
            .extension()
            .map(|e| e.is_empty())
            .unwrap_or(true);

        let body_l = body.to_ascii_lowercase();
        let html = body_l.contains("<html")
            || body_l.contains("<!doctype")
            || body_l.contains("<head")
            || body_l.contains("<body")
            || body_l.contains("<canvas");
        let md = body_l.contains("# ") || body_l.starts_with("---");

        // Replace generic landing_page.html when body/user clearly mean something else.
        if generic_html {
            let better = user_hint
                .map(Self::suggested_path_from_user_text)
                .filter(|s| !s.contains("landing_page") && !s.contains("output.txt"))
                .unwrap_or_else(|| {
                    let name = if body.trim().is_empty() {
                        user_hint
                            .map(|u| {
                                Self::suggested_path_from_user_text(u)
                                    .rsplit('/')
                                    .next()
                                    .unwrap_or("index.html")
                                    .to_string()
                            })
                            .unwrap_or_else(|| "index.html".into())
                    } else {
                        Self::infer_filename_from_body(body)
                    };
                    if p.contains("dummy_folder") || p.contains("$CURRENT") {
                        // keep directory prefix
                        if let Some(parent) = Path::new(p).parent() {
                            let parent = parent.to_string_lossy();
                            if parent.is_empty() || parent == "." {
                                format!("$CURRENT/dummy_folder/{name}")
                            } else {
                                format!("{parent}/{name}")
                            }
                        } else {
                            format!("$CURRENT/dummy_folder/{name}")
                        }
                    } else {
                        format!("$CURRENT/dummy_folder/{name}")
                    }
                });
            // If suggested_path is full $CURRENT/... use it; else join
            if better.starts_with("$CURRENT") || better.contains('/') {
                return better;
            }
        }

        if looks_like_dir || (no_ext && (html || md || !body.trim().is_empty()) && !p.contains('.'))
        {
            let name = if !body.trim().is_empty() {
                Self::infer_filename_from_body(body)
            } else if let Some(u) = user_hint {
                Self::suggested_path_from_user_text(u)
                    .rsplit('/')
                    .next()
                    .unwrap_or("file.txt")
                    .to_string()
            } else if html {
                "index.html".into()
            } else if md {
                "README.md".into()
            } else {
                "file.txt".into()
            };
            if expanded.is_dir() || p.ends_with('/') || !p.contains('.') {
                return format!("{p}/{name}");
            }
        }
        p.to_string()
    }

    /// Merge multiple write actions targeting the same resolved path into one,
    /// concatenating their bodies. This prevents a second `<write>` for the same
    /// file (e.g. model writes HTML then CSS into same `index.html`) from
    /// clobbering the first write.

    fn parse_mcp_skill_actions(text: &str, from_think: bool) -> Vec<ProposedAction> {
        let mut out = Vec::new();
        let mut rest = text;
        while let Some(start_tag) = rest.find("<mcp ") {
            let r = &rest[start_tag..];
            if let Some(close_bracket) = r.find('>') {
                let header = &r[..close_bracket + 1];
                let action = Self::extract_attribute(header, "action").unwrap_or_else(|| "search".to_string());
                let after = &r[close_bracket + 1..];
                if let Some(end_tag) = after.find("</mcp>") {
                    let body = after[..end_tag].trim().to_string();
                    out.push(ProposedAction {
                        kind: ProposedKind::Mcp,
                        target: action,
                        body,
                        line_attr: None,
                        from_think,
                        chip_id: None,
                    });
                    rest = &after[end_tag + 6..];
                    continue;
                }
            }
            break;
        }
        rest = text;
        while let Some(start_tag) = rest.find("<skill ") {
            let r = &rest[start_tag..];
            if let Some(close_bracket) = r.find('>') {
                let header = &r[..close_bracket + 1];
                let action = Self::extract_attribute(header, "action").unwrap_or_else(|| "search".to_string());
                let after = &r[close_bracket + 1..];
                if let Some(end_tag) = after.find("</skill>") {
                    let body = after[..end_tag].trim().to_string();
                    out.push(ProposedAction {
                        kind: ProposedKind::Skill,
                        target: action,
                        body,
                        line_attr: None,
                        from_think,
                        chip_id: None,
                    });
                    rest = &after[end_tag + 8..];
                    continue;
                }
            }
            break;
        }
        out
    }

    fn merge_duplicate_writes(actions: Vec<ProposedAction>) -> Vec<ProposedAction> {
        let mut merged: Vec<ProposedAction> = Vec::new();
        for action in actions {
            if action.kind != ProposedKind::Write || action.line_attr.is_some() {
                // Don't merge line-range writes — they target specific regions
                merged.push(action);
                continue;
            }
            // Check if we already have a full-file write to the same target
            if let Some(existing) = merged
                .iter_mut()
                .find(|a| a.kind == ProposedKind::Write && a.line_attr.is_none() && a.target == action.target)
            {
                // Append the new body — the model split one file across two write tags
                if !existing.body.ends_with('\n') {
                    existing.body.push('\n');
                }
                existing.body.push_str(&action.body);
            } else {
                merged.push(action);
            }
        }
        merged
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
                        chip_id: None,
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
        // cmd parsing follows — merge writes at the very end
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
                        chip_id: None,
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
                        chip_id: None,
                    });
                }
                break;
            }
        }
        Self::merge_duplicate_writes(out)
    }

    pub fn execute_proposed(action: &ProposedAction) -> String {
        match action.kind {
            ProposedKind::Write => {
                let path = Self::normalize_write_path(&action.target, &action.body);
                Self::execute_write(&path, action.line_attr.as_deref(), &action.body)
            }
 
            ProposedKind::WebSearch => {
                tokio::task::block_in_place(|| {
                    tokio::runtime::Handle::current().block_on(async {
                        let provider = websearch::providers::duckduckgo::DuckDuckGoProvider::new();
                        let options = websearch::SearchOptions {
                            query: action.target.clone(),
                            max_results: Some(3),
                            provider: Box::new(provider),
                            ..Default::default()
                        };
                        match websearch::web_search(options).await {
                            Ok(res) => {
                                let mut out = String::new();
                                for r in res {
                                    out.push_str(&format!("Title: {}\nURL: {}\nSnippet: {}\n\n", r.title, r.url, r.snippet.unwrap_or_default()));
                                }
                                if out.is_empty() {
                                    "No results found.".to_string()
                                } else {
                                    out.trim().to_string()
                                }
                            }
                            Err(e) => format!("Error searching web: {}", e),
                        }
                    })
                })
            }
            ProposedKind::Mcp | ProposedKind::Skill | ProposedKind::Agent => String::new(), // handled elsewhere
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
            let actions = Self::collapse_write_actions_for_user(
                "", // no user hint — still drops tiny .txt junk next to a real HTML body
                Self::extract_proposed_actions(response),
            );
            for action in actions {
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

    pub fn extract_attribute(tag: &str, attr_name: &str) -> Option<String> {
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

    
    fn compute_diff(old: &str, new: &str) -> String {
        let old_lines: Vec<&str> = old.lines().collect();
        let new_lines: Vec<&str> = new.lines().collect();
        
        if old_lines.is_empty() {
            let mut out = Vec::new();
            for (idx, line) in new_lines.iter().enumerate() {
                out.push(format!("+ {:4} | {}", idx + 1, line));
            }
            return out.join("\n");
        }

        if old_lines.len() > 1000 || new_lines.len() > 1000 {
            let mut out = Vec::new();
            for (idx, line) in new_lines.iter().enumerate() {
                out.push(format!("+ {:4} | {}", idx + 1, line));
            }
            return out.join("\n");
        }
        
        let n = old_lines.len();
        let m = new_lines.len();
        let mut dp = vec![vec![0; m + 1]; n + 1];
        for i in 1..=n {
            for j in 1..=m {
                if old_lines[i-1] == new_lines[j-1] {
                    dp[i][j] = dp[i-1][j-1] + 1;
                } else {
                    dp[i][j] = dp[i-1][j].max(dp[i][j-1]);
                }
            }
        }
        
        let mut i = n;
        let mut j = m;
        let mut diff = Vec::new();
        while i > 0 || j > 0 {
            if i > 0 && j > 0 && old_lines[i-1] == new_lines[j-1] {
                diff.push(format!("  {:4} | {}", j, new_lines[j-1]));
                i -= 1;
                j -= 1;
            } else if j > 0 && (i == 0 || dp[i][j-1] >= dp[i-1][j]) {
                diff.push(format!("+ {:4} | {}", j, new_lines[j-1]));
                j -= 1;
            } else if i > 0 && (j == 0 || dp[i][j-1] < dp[i-1][j]) {
                diff.push(format!("- {:4} | {}", i, old_lines[i-1]));
                i -= 1;
            }
        }
        diff.reverse();
        diff.join("\n")
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
                "Error: '{}' is a directory — use a file path e.g. '{}/index.html' or a task-specific name",
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

            let old_removed = lines[(start_line - 1)..end_idx].to_vec();
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

            let mut diff = String::new();
            for (idx, old) in old_removed.iter().enumerate() {
                diff.push_str(&format!("- {:4} | {}\n", start_line + idx, old));
            }
            for (idx, new) in replacement_lines.iter().enumerate() {
                diff.push_str(&format!("+ {:4} | {}\n", start_line + idx, new));
            }
            if diff.is_empty() {
                diff = "No changes.\n".to_string();
            }
            diff.trim_end().to_string()
        } else {
            if let Some(parent) = path.parent() {
                let _ = fs::create_dir_all(parent);
            }

            let clean_body = body.trim_start_matches('\n');
            if let Some(parent) = path.parent() {
                if let Err(e) = fs::create_dir_all(parent) {
                    return format!("Error creating parent dir '{}': {}", parent.display(), e);
                }
            }
            let old_content = fs::read_to_string(&path).unwrap_or_default();
            match fs::write(&path, clean_body) {
                Ok(()) => {
                    let diff = Self::compute_diff(&old_content, clean_body);
                    if diff.trim().is_empty() {
                        "No changes.".to_string()
                    } else {
                        diff
                    }
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
        let sample = "Hello! <think>I will <cmd>echo hello_world</cmd> execute this</think> <ls path=\"$CURRENT\">";
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
    fn test_ls_promoted_from_think_when_no_outside() {
        // Misplaced <ls> inside think is promoted (small models put tools in think).
        let sample = "<think>Let me list <ls path=\".\"></think>";
        assert!(AgentEngine::process_response(sample).is_some());

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

        let rep = AgentEngine::process_response(
            "<memory replace=1>\nexecute after plan is done\n</memory>",
        )
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
        // Compact prompt must stay path-generic (no project-specific demo files).
        assert!(!SYSTEM_PROMPT_COMPACT.contains("Cargo.toml"));
        assert!(!SYSTEM_PROMPT_COMPACT.contains("requirements.txt"));
        assert!(SYSTEM_PROMPT_COMPACT.contains(r#"$CURRENT/path/to/file"#));
    }

    #[test]
    fn test_synthesize_read_uses_user_path_not_hardcoded() {
        let tag = AgentEngine::synthesize_tool_from_user("can you read Cargo.toml").unwrap();
        assert_eq!(tag, r#"<read src="$CURRENT/Cargo.toml">"#);

        let tag2 = AgentEngine::synthesize_tool_from_user("open src/main.rs").unwrap();
        assert_eq!(tag2, r#"<read src="$CURRENT/src/main.rs">"#);

        let tag3 = AgentEngine::synthesize_tool_from_user("please read notes.txt").unwrap();
        assert_eq!(tag3, r#"<read src="$CURRENT/notes.txt">"#);

        let ls = AgentEngine::synthesize_tool_from_user("list current dir").unwrap();
        assert_eq!(ls, r#"<ls path="$CURRENT">"#);

        // Pathless read must NOT invent requirements.txt / any filename — list instead
        let vague = AgentEngine::synthesize_tool_from_user("can you read file?").unwrap();
        assert_eq!(vague, r#"<ls path="$CURRENT">"#);
        assert!(!vague.contains("requirements"));
        assert!(AgentEngine::wants_read_without_path("can you read file?"));
        assert!(!AgentEngine::wants_read_without_path("read Cargo.toml"));
    }

    #[test]
    fn test_recover_tools_from_refusal() {
        let refusal = "I don't have the ability to directly read files on your system.";
        let tag =
            AgentEngine::recover_tools_from_refusal("can you read package.json", refusal).unwrap();
        assert_eq!(tag, r#"<read src="$CURRENT/package.json">"#);

        // Already has a tool — no recovery
        assert!(
            AgentEngine::recover_tools_from_refusal("read x.rs", r#"<read src="$CURRENT/x.rs">"#)
                .is_none()
        );
    }

    #[test]
    fn test_with_tool_nudge_appends_for_tool_requests() {
        let nudged = AgentEngine::with_tool_nudge("read foo.bar");
        assert!(nudged.contains("foo.bar"));
        assert!(nudged.contains("[Host]"));
        // greetings stay plain
        assert_eq!(AgentEngine::with_tool_nudge("hello"), "hello");
    }

    #[test]
    fn test_plan_first_and_implement_no_ls_spam() {
        let plan = "create a store with svelte, tell me your plan before coding";
        assert!(AgentEngine::wants_plan_first(plan));
        assert!(!AgentEngine::wants_implement(plan));
        let sfx = AgentEngine::tool_force_suffix(plan).unwrap();
        assert!(sfx.to_ascii_lowercase().contains("plan"));
        assert!(!sfx.contains("<ls path")); // must not force an ls tool call

        let go = "start coding";
        assert!(AgentEngine::wants_implement(go));
        let sfx2 = AgentEngine::tool_force_suffix(go).unwrap();
        assert!(sfx2.contains("<write"));
        assert!(!sfx2.contains("<ls path"));

        // Must not recover ls for create/plan
        assert!(AgentEngine::recover_tools_from_refusal(plan, "sure").is_none());
        assert!(AgentEngine::recover_tools_from_refusal(go, "ok").is_none());
    }
}
