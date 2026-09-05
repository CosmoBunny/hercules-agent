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
List directory:       <ls path="$CURRENT/path">
Read file:            <read src="$CURRENT/path/to/file">
Read line range:      <read src="$CURRENT/path/to/file" line=1..=50>
Write / create file:  <write src="$CURRENT/path/to/file">
[file content to be written]
</write>
Replace line range:   <write src="$CURRENT/path/to/file" line=10..=15>
[replacement content for lines 10 through 15]
</write>
Run tool / command:   <cmd> tool run here </cmd>
MCP tool call:        <mcp server="SERVER_NAME" tool="TOOL_NAME">
{
  "arg_name": "arg_value"
}
</mcp>
Web search:           <websearch query="search terms"> or <websearch>search terms</websearch>
Spawn sub-agent:      <agent action="spawn" role="ROLE" model="MODEL">task description</agent>
Memory push:          <memory push>text</memory>
Memory read:          <memory read>
Help (inside think):  <help>

## Operational Rules
1. To create, write, or create folders, use `<write src="...">` containing the COMPLETE file content from start to finish in a single tag. Parent folders are created automatically. NEVER chunk or write a single file across multiple partial `<write>` tags line by line. Multi-write is ONLY for writing distinct separate files (different src).
2. To read or inspect files/directories, use `<read ...>` or `<ls ...>`.
3. To replace a specific range of lines in an existing file, use `<write src="..." line=START..=END>` with the replacement block.
4. For web search or online documentation, use `<websearch query="..."/>` or `<websearch>query</websearch>`.
5. For running shell utilities, git (clone, status, commit), web fetching (curl, wget), build tools, process management, or OS operations, use `<cmd> tool run here </cmd>`.
6. For external tools provided via Model Context Protocol, use `<mcp server="..." tool="...">{"arg": "val"}</mcp>`.
7. When using CLI tools, only consult help/manual flags (`--help`, `man`, `/?`) if you are unsure of the tool's exact usage syntax and require clarification. Do not run help commands in the first place without actual need.
8. For pure conversation, planning, or questions, reply directly in natural language.
9. Reasoning inside `<think>...</think>` is optional. If you use it, close with `</think>` and emit your tool calls or response outside `<think>`. NEVER put tool tags inside `<think>` — they will be ignored.
10. Never state you lack access to the local machine or tools.
11. Only real emitted tool tags count as actions. Do NOT narrate, quote, or repeat tool tags in prose or inside `<write>` file bodies — the host executes exactly what you emit, once. Do NOT invent tool results; wait for the host to return them.

## Ask Mode
Use Ask Mode ONLY when a choice materially changes the task, required info is genuinely missing, or confirmation is needed before proceeding. Do NOT use it for ordinary factual questions, simple calculations, or when a reasonable default is obvious.
When Ask Mode is required, emit ONLY one complete block and stop — wait for the user:
<askmode ques="Question">
<radio>Option A</radio>
<radio>Option B</radio>
<check>Optional feature</check>
<input>Additional information</input>
</askmode>
Do not describe the syntax in prose. Do not invent a user response. After the host returns an Ask Mode response, continue the task using it.
"#;

/// Compact system prompt for small GGUFs / llama-server chat.
pub const SYSTEM_PROMPT_COMPACT: &str = r#"You are Hercules, an autonomous agent on the user's machine with real tools.

cwd token: $CURRENT (host expands it).

Tools (raw tags only, no markdown fences):
<ls path="$CURRENT/path">
<read src="$CURRENT/path/to/file">
<read src="$CURRENT/path/to/file" line=1..=50>
<write src="$CURRENT/path/to/file">
[file content to be written]
</write>
<write src="$CURRENT/path/to/file" line=10..=15>
[replacement lines]
</write>
<mcp server="server" tool="tool">{"param": "value"}</mcp>
<websearch query="search terms">
<cmd> tool run here </cmd>
<agent action="spawn" role="role" model="model">task</agent>
<memory push>text</memory>
<memory read>

Rules:
- Questions and conversation: respond directly in natural language.
- Inspecting or reading: use `<ls>` or `<read>`.
- Searching web / documentation: use `<websearch query="...">`.
- Calling MCP tools: use `<mcp server="..." tool="...">{"param": "value"}</mcp>`.
- Creating or editing files (and dirs): use `<write src="...">` with the COMPLETE file content in ONE single block. NEVER split or write one file line-by-line across multiple write tags. Multiple writes are only for separate files.
- Line replacements: use `<write src="..." line=START..=END>`.
- Git, web fetch (curl/wget), packages, file moves/deletes, or OS utilities: use `<cmd> tool run here </cmd>`.
- Only check tool help (`--help`, `man`, `/?`) when syntax clarification is genuinely required, not by default.
- Reasoning inside `<think>...</think>` is optional. If used, close with `</think>` and emit tool calls or response outside. NEVER put tool tags inside `<think>`.
- Only real emitted tool tags count as actions: never narrate, quote, or repeat tool tags in prose or inside `<write>` bodies. Never invent tool results; wait for the host.
- Never claim you lack file access or tools.
- Ask Mode: use ONLY when a choice changes the task or info is truly missing. Emit ONLY one <askmode ques="..."> block with radio/check/input children, then stop and wait. Never describe the syntax or invent a response.
"#;

/// Canonical internal representation of ONE executable tool invocation.
///
/// Lifecycle (single authoritative pipeline):
/// model/backend output (plain `String`; see backend-protocol note below)
/// → [`AgentEngine::parse_tool_calls`] (sole parser) → dispatch
/// (exactly-once via [`ToolDispatchRegistry`]) → execution event
/// → Action UI chip (by `chip_id`) → executor
/// ([`AgentEngine::execute_proposed`] / task-manager spawn) → [`ToolResult`]
/// → `tool_result_context` → next generation.
///
/// BACKEND-PROTOCOL DECISION: none of the backends (`LlamaCppLib`,
/// `Ollama`, Burn) expose structured `tool_calls` / `tool_call_id` /
/// `reasoning_content` — `AgentBackend::generate[_stream]` returns plain
/// text only (Ollama's native `thinking` field is wrapped into `<think>`
/// text by the backend adapter). Hercules therefore intentionally uses its
/// custom XML-tag protocol, and the `StreamingParser`-based
/// [`AgentEngine::parse_tool_calls`] is the SINGLE canonical parser. No
/// parallel semantic parser may decide executability.
#[derive(Debug, Clone)]
pub struct ProposedAction {
    pub kind: ProposedKind,
    pub target: String,
    pub body: String,
    pub line_attr: Option<String>,
    /// Always false: extractors only see outside-think content, so thinking
    /// tags are never promoted to actions. Retained for struct compatibility.
    pub from_think: bool,
    pub chip_id: Option<u64>,
    /// Stable identity minted at parse time; preserved onto the
    /// [`ToolResult`] so results bind to the exact call, never to
    /// "latest chip of this kind".
    pub call_id: u64,
    /// Where this call originated; recovery flows through the same pipeline.
    pub source: ToolCallSource,
}

/// Origin of a [`ProposedAction`]. Model and host-recovery calls share the
/// type and the pipeline; the source only records provenance.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolCallSource {
    /// Parsed from the live streaming text.
    ModelStream,
    /// Parsed from the completed response.
    ModelCompletion,
    /// Host-synthesized after a model refusal
    /// ([`AgentEngine::recover_tools_from_refusal`]).
    HostRecovery,
    /// Re-confirmed by the user (accept flow).
    UserAccept,
}

/// Monotonic mint for [`ProposedAction::call_id`].
static NEXT_TOOL_CALL_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

fn next_tool_call_id() -> u64 {
    NEXT_TOOL_CALL_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
}

impl ProposedAction {
    /// Canonical constructor: every parsed call gets a fresh `call_id`.
    pub fn new(
        kind: ProposedKind,
        target: String,
        body: String,
        line_attr: Option<String>,
        source: ToolCallSource,
    ) -> Self {
        Self {
            kind,
            target,
            body,
            line_attr,
            from_think: false,
            chip_id: None,
            call_id: next_tool_call_id(),
            source,
        }
    }

    /// Dedupe key: same kind + target + body = same logical call, even
    /// across re-parses (streaming re-parses the full text every poll, so
    /// `call_id` alone cannot dedupe).
    pub fn fingerprint(&self) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut h = DefaultHasher::new();
        self.kind.hash(&mut h);
        self.target.hash(&mut h);
        self.body.hash(&mut h);
        self.line_attr.hash(&mut h);
        format!("{:?}:{:x}", self.kind, h.finish())
    }
}

/// Exactly-once gate for the canonical dispatch pipeline.
/// A fingerprint that was already claimed (executed or spawned) is rejected,
/// so one model tool call cannot run twice through two paths (e.g. live
/// streaming execution + completion re-dispatch).
#[derive(Debug, Default)]
pub struct ToolDispatchRegistry {
    claimed: std::collections::HashSet<String>,
}

impl ToolDispatchRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Claim a call for execution. Returns `true` exactly once per unique
    /// fingerprint; `false` means "already dispatched — skip".
    pub fn try_claim(&mut self, action: &ProposedAction) -> bool {
        self.claimed.insert(action.fingerprint())
    }

    pub fn is_claimed(&self, action: &ProposedAction) -> bool {
        self.claimed.contains(&action.fingerprint())
    }

    /// Reset at the start of a new agent turn (new user message).
    pub fn reset_turn(&mut self) {
        self.claimed.clear();
    }
}

/// Output of one dispatched [`ProposedAction`], bound to it by `call_id`.
#[derive(Debug, Clone)]
pub struct ToolResult {
    pub call_id: u64,
    pub fingerprint: String,
    pub output: String,
}

impl ToolResult {
    pub fn new(action: &ProposedAction, output: String) -> Self {
        Self {
            call_id: action.call_id,
            fingerprint: action.fingerprint(),
            output,
        }
    }

    /// Context insertion format: carries the call id so the next generation
    /// can attribute the result to the exact call.
    pub fn context_entry(&self, label: &str) -> String {
        format!("[{} result #{}]\n{}", label, self.call_id, self.output)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ProposedKind {
    Write,
    Cmd,
    Read,
    Ls,
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
            Self::Read => "READ",
            Self::Ls => "LIST",
            Self::Mcp => "MCP",
            Self::Skill => "SKILL",
            Self::WebSearch => "WEBSEARCH",
            Self::Agent => "AGENT",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParseState {
    Normal,
    Thinking,
}

impl Default for ParseState {
    fn default() -> Self {
        ParseState::Normal
    }
}

pub struct StreamingParser {
    state: ParseState,
    buffer: String,
    think_depth: usize,
    source: ToolCallSource,
}

impl StreamingParser {
    pub fn new() -> Self {
        Self::with_source(ToolCallSource::ModelCompletion)
    }

    pub fn with_source(source: ToolCallSource) -> Self {
        Self {
            state: ParseState::Normal,
            buffer: String::new(),
            think_depth: 0,
            source,
        }
    }

    pub fn feed(&mut self, chunk: &str) -> Vec<ProposedAction> {
        self.buffer.push_str(chunk);
        let mut actions = Vec::new();
        let mut iterations = 0;
        const MAX_ITERATIONS: usize = 1000;

        while let Some(pos) = self.find_next_boundary() {
            iterations += 1;
            if iterations > MAX_ITERATIONS {
                self.buffer.clear();
                break;
            }
            match self.state {
                ParseState::Normal => {
                    let segment = &self.buffer[..pos];
                    actions.extend(Self::extract_from_segment(segment, self.source));
                    self.buffer.drain(..pos);
                    self.consume_think_start();
                }
                ParseState::Thinking => {
                    let think_end_start = pos - 8;
                    self.buffer.drain(..think_end_start);
                    self.consume_think_end();
                }
            }
        }

        if self.state == ParseState::Normal && !self.buffer.is_empty() {
            actions.extend(Self::extract_from_segment(&self.buffer, self.source));
            self.buffer.clear();
        }

        actions
    }

    fn find_next_boundary(&self) -> Option<usize> {
        let think_start = self.buffer.find("<think>");
        let think_end = self.buffer.find("</think>");

        match (self.state, think_start, think_end) {
            (ParseState::Normal, Some(s), _) => Some(s),
            (ParseState::Thinking, _, Some(e)) => {
                let end_pos = e + 8;
                if end_pos <= self.buffer.len() {
                    Some(end_pos)
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    fn consume_think_start(&mut self) {
        if self.buffer.len() >= 7 && self.buffer.starts_with("<think>") {
            self.buffer.drain(..7);
            self.state = ParseState::Thinking;
            self.think_depth += 1;
        }
    }

    fn consume_think_end(&mut self) {
        if self.buffer.len() >= 8 && self.buffer.starts_with("</think>") {
            self.buffer.drain(..8);
            self.state = ParseState::Normal;
            self.think_depth = self.think_depth.saturating_sub(1);
        }
    }

    fn extract_from_segment(segment: &str, source: ToolCallSource) -> Vec<ProposedAction> {
        let outside = AgentEngine::strip_code_fences(&AgentEngine::strip_think_blocks(segment));
        let mut from_out = AgentEngine::parse_write_cmd_actions(&outside, source);
        from_out.extend(AgentEngine::parse_mcp_skill_actions(&outside, source));
        from_out.extend(AgentEngine::parse_read_ls_actions(&outside, source));
        from_out.retain(|a| AgentEngine::action_is_sane(a));
        from_out
    }

    pub fn flush(&mut self) -> Vec<ProposedAction> {
        if self.state == ParseState::Normal && !self.buffer.is_empty() {
            let actions = Self::extract_from_segment(&self.buffer, self.source);
            self.buffer.clear();
            actions
        } else {
            self.buffer.clear();
            Vec::new()
        }
    }

    pub fn in_thinking(&self) -> bool {
        self.state == ParseState::Thinking
    }

    pub fn reset(&mut self) {
        self.state = ParseState::Normal;
        self.buffer.clear();
        self.think_depth = 0;
    }
}

impl Default for StreamingParser {
    fn default() -> Self {
        Self::new()
    }
}

/// First-class classification of raw model output.
///
/// The host must treat model text as untrusted: this single entry point
/// decides whether a completed response is an Ask Mode control-flow event,
/// an executable tool call, or plain text — before anything with side
/// effects runs. Tool detection is evaluated on outside-think content only.
pub enum ModelOutput {
    AskMode(crate::ask_mode::AskMode),
    ToolCall,
    Text,
}

pub struct AgentEngine;

impl AgentEngine {
    pub fn format_markdown_tables(text: &str, max_width: usize, scroll_x: usize) -> String {
        let mut out = String::new();
        let mut table_lines = Vec::new();

        fn render_table(lines: &[&str], out: &mut String, max_width: usize, scroll_x: usize) {
            if lines.is_empty() {
                return;
            }
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
            if rows.is_empty() {
                return;
            }
            let mut has_sep = false;
            if rows.len() > 1 {
                let is_sep = rows[1].iter().all(|c| {
                    !c.is_empty() && c.chars().all(|ch| ch == '-' || ch == ':' || ch == ' ')
                });
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
            let top = format!(
                "┌{}┐",
                widths
                    .iter()
                    .map(|w| "─".repeat(*w + 2))
                    .collect::<Vec<_>>()
                    .join("┬")
            );
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
                    let mid = format!(
                        "├{}┤",
                        widths
                            .iter()
                            .map(|w| "─".repeat(*w + 2))
                            .collect::<Vec<_>>()
                            .join("┼")
                    );
                    table_rows.push(mid);
                }
            }
            let bot = format!(
                "└{}┘",
                widths
                    .iter()
                    .map(|w| "─".repeat(*w + 2))
                    .collect::<Vec<_>>()
                    .join("┴")
            );
            table_rows.push(bot);

            let table_width = table_rows[0].chars().count();
            let eff_width = max_width.saturating_sub(2); // leaving margin
            let needs_scroll = table_width > eff_width;
            let actual_scroll = if needs_scroll {
                scroll_x.min(table_width - eff_width)
            } else {
                0
            };

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
                let thumb_size =
                    (track_len as f64 * (eff_width as f64 / table_width as f64)).max(1.0) as usize;
                let thumb_pos = (track_len as f64
                    * (actual_scroll as f64 / (table_width - eff_width) as f64))
                    .min((track_len - thumb_size) as f64) as usize;

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

        let os_commands = if cfg!(target_os = "windows") {
            "Host Environment: Windows (cmd / PowerShell)
Tool Run Format: <cmd> tool run here </cmd>
Common Tasks:
- Web fetch / URL:  curl.exe -sL \"url\" or powershell -Command \"Invoke-WebRequest -Uri 'url' -OutFile 'dest'\"
- Git / Repository: git clone <url>, git status, git diff, git commit -m \"...\"
- Move / Rename:    move \"src\" \"dst\" or ren \"old\" \"new\"
- Copy:             copy \"src\" \"dst\" or xcopy \"src\" \"dst\" /E /I
- Remove / Delete:  del \"file\" or rmdir /S /Q \"dir\"
- Find in files:    findstr /S /I /N \"pattern\" *.*
- Processes:        tasklist
- Syntax Help:      Only use `/?` or `--help` on a tool when you genuinely need syntax clarification, not in the first place."
        } else if cfg!(target_os = "macos") {
            "Host Environment: macOS (zsh / bash / BSD utils)
Tool Run Format: <cmd> tool run here </cmd>
Common Tasks:
- Web fetch / URL:  curl -sL \"url\" or wget \"url\"
- Git / Repository: git clone <url>, git status, git diff, git commit -m \"...\"
- Move / Rename:    mv \"src\" \"dst\"
- Copy:             cp \"src\" \"dst\" or cp -R \"src\" \"dst\"
- Remove / Delete:  rm \"file\" or rm -rf \"dir\"
- Find in files:    grep -rn \"pattern\" . or find . -name \"pattern\"
- Processes:        ps aux
- Syntax Help:      Only use `man <tool>` or `<tool> --help` when you genuinely need syntax clarification, not in the first place."
        } else {
            "Host Environment: Linux (bash / sh / GNU coreutils)
Tool Run Format: <cmd> tool run here </cmd>
Common Tasks:
- Web fetch / URL:  curl -sL \"url\" or wget \"url\"
- Git / Repository: git clone <url>, git status, git diff, git commit -m \"...\"
- Move / Rename:    mv \"src\" \"dst\"
- Copy:             cp \"src\" \"dst\" or cp -r \"src\" \"dst\"
- Remove / Delete:  rm \"file\" or rm -rf \"dir\"
- Find in files:    grep -rn \"pattern\" . or find . -name \"pattern\"
- Processes:        ps aux
- Syntax Help:      Only use `man <tool>` or `<tool> --help` when you genuinely need syntax clarification, not in the first place."
        };

        let mcp_tools_section = tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(async {
                let mcp_mgr_lock = crate::mcp::McpManager::instance().await;
                let mut mcp_guard = mcp_mgr_lock.lock().await;
                if let Some(ref mut mgr) = *mcp_guard {
                    mgr.sync_with_settings().await;
                    mgr.generate_prompt_tools_section().await
                } else {
                    String::new()
                }
            })
        });

        format!(
            "{base}\n\nLive Environment:\n- Working Directory: $CURRENT → {cwd}\n- {os_commands}\n{mcp_tools_section}"
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
            || (t.contains("build")
                && (t.contains("page") || t.contains("app") || t.contains("ui")))
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
    /// Thinking zones are reasoning text, not executable calls: only tags
    /// outside `<think>` count.
    pub fn response_has_tool_tags(text: &str) -> bool {
        let stripped = Self::strip_think_blocks(text);
        let t = stripped.as_str();
        t.contains("<read src=")
            || t.contains("<ls path=")
            || t.contains("<ls>")
            || t.contains("<write src=")
            || t.contains("<cmd>")
            || t.contains("<websearch")
            || t.contains("<mcp")
            || t.contains("<skill")
            || t.contains("<agent")
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

    /// When a small model answers with a fenced code block (```lang ... ```)
    /// instead of emitting a `<write>` tool tag, recover the block as a pending
    /// write action so the user isn't left with dead text they must copy by hand.
    pub fn recover_write_from_fenced_code(
        user_text: &str,
        assistant_text: &str,
    ) -> Option<ProposedAction> {
        if Self::response_has_tool_tags(assistant_text) {
            return None;
        }
        if !assistant_text.contains("```") && !assistant_text.contains("~~~") {
            return None;
        }

        let fence = "```";
        let Some(start) = assistant_text.find(fence) else {
            return None;
        };
        let after_start = &assistant_text[start + 3..];
        let lang = after_start
            .lines()
            .next()
            .unwrap_or("")
            .trim()
            .trim_end_matches('~')
            .to_ascii_lowercase();
        let body_start_rel = after_start.find('\n').map(|i| i + 1).unwrap_or(0);
        let body_source = &after_start[body_start_rel..];
        let Some(end_rel) = body_source.find(fence) else {
            return None;
        };
        let body = body_source[..end_rel].trim().to_string();
        if body.is_empty() || body.len() > 200_000 {
            return None;
        }

        let path = Self::filename_from_fence_lang(&lang)
            .map(|f| format!("$CURRENT/{f}"))
            .or_else(|| {
                let from_body = Self::infer_filename_from_body(&body);
                if from_body != "output.txt" {
                    Some(format!("$CURRENT/{from_body}"))
                } else {
                    let from_user = Self::suggested_path_from_user_text(user_text);
                    (from_user != "$CURRENT/output.txt").then_some(from_user)
                }
            })
            .unwrap_or_else(|| "$CURRENT/output.txt".to_string());

        Some(ProposedAction {
            kind: ProposedKind::Write,
            target: path,
            body,
            line_attr: None,
            from_think: false,
            chip_id: None,
            call_id: next_tool_call_id(),
            source: ToolCallSource::ModelCompletion,
        })
    }

    fn filename_from_fence_lang(lang: &str) -> Option<String> {
        if lang.is_empty() {
            return None;
        }
        let l = lang.trim();
        Some(match l {
            "html" | "htm" | "xml" | "xhtml" => "index.html".to_string(),
            "css" => "style.css".to_string(),
            "javascript" | "js" | "jsx" => "script.js".to_string(),
            "typescript" | "ts" | "tsx" => "script.ts".to_string(),
            "python" | "py" => "main.py".to_string(),
            "rust" | "rs" => "main.rs".to_string(),
            "c" => "main.c".to_string(),
            "cpp" | "c++" => "main.cpp".to_string(),
            "go" => "main.go".to_string(),
            "java" => "Main.java".to_string(),
            "sh" | "bash" | "shell" | "zsh" => "script.sh".to_string(),
            "json" => "data.json".to_string(),
            "yaml" | "yml" => "config.yaml".to_string(),
            "toml" => "config.toml".to_string(),
            "md" | "markdown" => "README.md".to_string(),
            _ => return None,
        })
    }

    /// Strip ` thinking... response` blocks from response text.
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

    /// Collect write/cmd from outside-think only; NEVER execute tools from thinking.
    /// Models dump prose into fake `<cmd>` tags inside thinking — never execute those.
    /// THE canonical tool-call parser — the single authority that turns
    /// model/host text into executable [`ProposedAction`]s. All pipelines
    /// (streaming preview, completion dispatch, host recovery) must go
    /// through here; nothing else may decide raw text is executable.
    /// Thinking zones and code fences are stripped before parsing, so
    /// `<think>` content and quoted examples can never become calls.
    pub fn parse_tool_calls(text: &str, source: ToolCallSource) -> Vec<ProposedAction> {
        let mut parser = StreamingParser::with_source(source);
        let actions = parser.feed(text);
        actions.into_iter().chain(parser.flush()).collect()
    }

    /// Legacy alias for [`Self::parse_tool_calls`] with
    /// [`ToolCallSource::ModelCompletion`]. Kept for call-site compatibility.
    pub fn extract_proposed_actions(response: &str) -> Vec<ProposedAction> {
        Self::parse_tool_calls(response, ToolCallSource::ModelCompletion)
    }

    /// Classify a completed model response into its control-flow meaning.
    /// Ask Mode wins over tool calls: when the model asks the user something,
    /// the host must pause generation and wait instead of executing anything.
    pub fn classify_output(response: &str) -> ModelOutput {
        if let Some(ask) = crate::ask_mode::AskModeParser::parse(response)
            .ok()
            .flatten()
        {
            return ModelOutput::AskMode(ask);
        }
        if Self::response_has_tool_tags(&Self::strip_think_blocks(response)) {
            return ModelOutput::ToolCall;
        }
        ModelOutput::Text
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
            ProposedKind::Read => {
                let t = a.target.trim();
                !t.is_empty() && t.len() < 400 && !t.contains('\n')
            }
            ProposedKind::Ls => true,
            ProposedKind::Mcp
            | ProposedKind::Skill
            | ProposedKind::WebSearch
            | ProposedKind::Agent => true,
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

    /// Suggest a default fallback filename when no path is provided.
    pub fn suggested_path_from_user_text(user_text: &str) -> String {
        let name = if let Some(candidate) = Self::extract_path_candidate(user_text) {
            candidate
        } else {
            let slug = Self::slugify_filename(user_text, 32);
            if slug.is_empty() || slug == "file" {
                "output.txt".to_string()
            } else {
                format!("{slug}.txt")
            }
        };
        if name.starts_with("$CURRENT") || name.starts_with('/') {
            name
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

    /// Infer a sensible filename from file body content.
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
        if body.trim().is_empty() {
            return "output.txt".into();
        }
        "output.txt".into()
    }

    /// Deduplicate identical write targets if generated multiple times.
    pub fn collapse_write_actions_for_user(
        _user_text: &str,
        actions: Vec<ProposedAction>,
    ) -> Vec<ProposedAction> {
        let mut seen = std::collections::HashSet::new();
        let mut out = Vec::new();
        for a in actions {
            if a.kind == ProposedKind::Write {
                if seen.insert(a.target.clone()) {
                    out.push(a);
                }
            } else {
                out.push(a);
            }
        }
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

        let looks_like_dir = expanded.is_dir()
            || p.ends_with('/')
            || (!p.contains('.')
                && (expanded.exists() && expanded.is_dir()
                    || !Path::new(p).extension().is_some_and(|e| !e.is_empty())));

        if looks_like_dir {
            let name = if !body.trim().is_empty() {
                Self::infer_filename_from_body(body)
            } else if let Some(u) = user_hint {
                Self::suggested_path_from_user_text(u)
                    .rsplit('/')
                    .next()
                    .unwrap_or("output.txt")
                    .to_string()
            } else {
                "output.txt".into()
            };
            return format!("{p}/{name}");
        }
        p.to_string()
    }

    /// Merge multiple write actions targeting the same resolved path into one,
    /// concatenating their bodies. This prevents a second `<write>` for the same
    /// file (e.g. model writes HTML then CSS into same `index.html`) from
    /// clobbering the first write.

    fn parse_mcp_skill_actions(text: &str, source: ToolCallSource) -> Vec<ProposedAction> {
        // Never match tags quoted inside a <write> file body.
        let scrubbed = Self::blank_closed_write_blocks(text);
        let mut out = Vec::new();
        let mut rest: &str = &scrubbed;
        while let Some(start_tag) = rest.find("<mcp") {
            let r = &rest[start_tag..];
            if let Some(close_bracket) = r.find('>') {
                let header = &r[..close_bracket + 1];
                let server = Self::extract_attribute(header, "server");
                let tool = Self::extract_attribute(header, "tool")
                    .or_else(|| Self::extract_attribute(header, "action"))
                    .unwrap_or_default();
                let target = if let Some(srv) = server {
                    format!("{srv}:{tool}")
                } else {
                    tool
                };
                let after = &r[close_bracket + 1..];
                if let Some(end_tag) = after.find("</mcp>") {
                    let body = after[..end_tag].trim().to_string();
                    out.push(ProposedAction::new(
                        ProposedKind::Mcp,
                        target,
                        body,
                        None,
                        source,
                    ));
                    rest = &after[end_tag + 6..];
                    continue;
                }
            }
            break;
        }
        rest = &scrubbed;
        while let Some(start_tag) = rest.find("<skill ") {
            let r = &rest[start_tag..];
            if let Some(close_bracket) = r.find('>') {
                let header = &r[..close_bracket + 1];
                let action = Self::extract_attribute(header, "action")
                    .unwrap_or_else(|| "search".to_string());
                let after = &r[close_bracket + 1..];
                if let Some(end_tag) = after.find("</skill>") {
                    let body = after[..end_tag].trim().to_string();
                    out.push(ProposedAction::new(
                        ProposedKind::Skill,
                        action,
                        body,
                        None,
                        source,
                    ));
                    rest = &after[end_tag + 8..];
                    continue;
                }
            }
            break;
        }
        rest = &scrubbed;
        while let Some(start_tag) = rest.find("<websearch") {
            let r = &rest[start_tag..];
            if let Some(close_bracket) = r.find('>') {
                let header = &r[..close_bracket + 1];
                let mut query_attr = Self::extract_attribute(header, "query");
                let after = &r[close_bracket + 1..];

                let mut end_pos = after.find("</websearch>");
                let is_explicit_closed = end_pos.is_some();

                if end_pos.is_none() {
                    if query_attr.is_some() {
                        end_pos = Some(0);
                    } else {
                        let next_tool_pos = [
                            "<write",
                            "<cmd",
                            "<read",
                            "<ls",
                            "<agent",
                            "<mcp",
                            "<skill",
                            "<websearch",
                        ]
                        .iter()
                        .filter_map(|tag| after.find(tag))
                        .min();
                        end_pos = next_tool_pos.or_else(|| after.find('\n'));
                    }
                }

                let end = end_pos.unwrap_or(after.len());
                let body = after[..end].trim().to_string();
                let advance = if is_explicit_closed { end + 12 } else { end };

                for stop in [
                    "<|im_end|>",
                    "<|im_start|>",
                    "<|eot_id|>",
                    "<|endoftext|>",
                    "</s>",
                ] {
                    if let Some(ref mut q) = query_attr {
                        *q = q.replace(stop, "").trim().to_string();
                    }
                }

                let target = query_attr.unwrap_or_else(|| {
                    if !body.is_empty() {
                        body.clone()
                    } else {
                        "search".to_string()
                    }
                });

                if !target.is_empty() && target != "search" {
                    out.push(ProposedAction::new(
                        ProposedKind::WebSearch,
                        target,
                        body,
                        None,
                        source,
                    ));
                }
                rest = &after[advance.min(after.len())..];
                continue;
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
            if let Some(existing) = merged.iter_mut().find(|a| {
                a.kind == ProposedKind::Write && a.line_attr.is_none() && a.target == action.target
            }) {
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

    /// Parse `<read>` / `<ls>` into canonical calls. Input must already be
    /// outside-think (and fence-stripped); write bodies are quarantined here
    /// so quoted example markup never becomes a call.
    fn parse_read_ls_actions(text: &str, source: ToolCallSource) -> Vec<ProposedAction> {
        let scrubbed = Self::blank_closed_write_blocks(text);
        let mut out = Vec::new();
        let mut rest: &str = &scrubbed;
        while let Some(start_tag) = rest.find("<read src=") {
            let r = &rest[start_tag..];
            if let Some(close_bracket) = r.find('>') {
                let tag_header = &r[..close_bracket + 1];
                let path_attr = Self::extract_attribute(tag_header, "src");
                let line_attr = Self::extract_attribute(tag_header, "line");
                rest = &r[close_bracket + 1..];
                if let Some(path_str) = path_attr {
                    out.push(ProposedAction::new(
                        ProposedKind::Read,
                        path_str,
                        String::new(),
                        line_attr,
                        source,
                    ));
                }
            } else {
                break;
            }
        }
        rest = &scrubbed;
        while let Some(start_tag) = rest.find("<ls") {
            let r = &rest[start_tag..];
            if let Some(close_bracket) = r.find('>') {
                let tag_header = &r[..close_bracket + 1];
                let path_attr = Self::extract_attribute(tag_header, "path");
                rest = &r[close_bracket + 1..];
                let path_str = path_attr.unwrap_or_else(|| "$CURRENT".to_string());
                out.push(ProposedAction::new(
                    ProposedKind::Ls,
                    path_str,
                    String::new(),
                    None,
                    source,
                ));
            } else {
                break;
            }
        }
        out
    }

    fn parse_write_cmd_actions(text: &str, source: ToolCallSource) -> Vec<ProposedAction> {
        let mut out = Vec::new();
        let mut rest = text;
        while let Some(start_tag) = rest.find("<write") {
            let r = &rest[start_tag..];
            if let Some(close_bracket) = r.find('>') {
                let tag_header = &r[..close_bracket + 1];
                let path_attr = Self::extract_attribute(tag_header, "src");
                let line_attr = Self::extract_attribute(tag_header, "line");
                let content_after_header = &r[close_bracket + 1..];

                let (body, next) = if let Some(end_tag) = content_after_header.find("</write") {
                    let body = &content_after_header[..end_tag];
                    let after = if let Some(ec) = content_after_header[end_tag..].find('>') {
                        &content_after_header[end_tag + ec + 1..]
                    } else {
                        ""
                    };
                    (body.to_string(), after)
                } else {
                    (content_after_header.to_string(), "")
                };
                if let Some(path_str) = path_attr {
                    let body = body.trim_matches(|c| c == '\n' || c == '\r').to_string();
                    let target = Self::normalize_write_path(&path_str, &body);
                    out.push(ProposedAction::new(
                        ProposedKind::Write,
                        target,
                        body,
                        line_attr,
                        source,
                    ));
                }
                if next.is_empty() {
                    break;
                }
                rest = next;
            } else {
                break;
            }
        }
        // cmd parsing follows — <cmd> tags quoted inside a <write> file
        // body are example text, never commands. Merge writes at the end.
        let cmd_scan = Self::blank_closed_write_blocks(text);
        rest = &cmd_scan;
        while let Some(start_tag) = rest.find("<cmd>") {
            let r = &rest[start_tag + 5..];
            if let Some(end_tag) = r.find("</cmd>") {
                let cmd_str = r[..end_tag].trim().to_string();
                rest = &r[end_tag + 6..];
                if Self::looks_like_shell_cmd(&cmd_str) {
                    out.push(ProposedAction::new(
                        ProposedKind::Cmd,
                        cmd_str,
                        String::new(),
                        None,
                        source,
                    ));
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
                    out.push(ProposedAction::new(
                        ProposedKind::Cmd,
                        cmd_str,
                        String::new(),
                        None,
                        source,
                    ));
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

            ProposedKind::Read => {
                let output = Self::execute_read(&action.target, action.line_attr.as_deref());
                let expanded = Self::expand_path(&action.target);
                crate::smart_system::get_smart_system().register_read(
                    crate::smart_system::AgentId::H0,
                    &expanded,
                    &output,
                );
                output
            }

            ProposedKind::Ls => Self::execute_ls(&action.target),

            ProposedKind::WebSearch => {
                tokio::task::block_in_place(|| {
                    tokio::runtime::Handle::current().block_on(async {
                        let settings = crate::settings::get_settings();
                        let provider_setting = settings.web_search_provider;
                        let provider_box: Box<dyn websearch::SearchProvider> = match provider_setting {
                            crate::settings::WebSearchProvider::DuckDuckGo => {
                                Box::new(websearch::providers::duckduckgo::DuckDuckGoProvider::new())
                            }
                            crate::settings::WebSearchProvider::Google => {
                                let key = settings.google_api_key.or_else(|| std::env::var("GOOGLE_API_KEY").ok()).unwrap_or_default();
                                let cx = settings.google_cx.or_else(|| std::env::var("GOOGLE_CX").ok()).unwrap_or_default();
                                match websearch::providers::google::GoogleProvider::new(&key, &cx) {
                                    Ok(p) => Box::new(p),
                                    Err(_) => Box::new(websearch::providers::duckduckgo::DuckDuckGoProvider::new()),
                                }
                            }
                            crate::settings::WebSearchProvider::Brave => {
                                let key = settings.brave_api_key.or_else(|| std::env::var("BRAVE_API_KEY").ok()).unwrap_or_default();
                                match websearch::providers::brave::BraveProvider::new(&key) {
                                    Ok(p) => Box::new(p),
                                    Err(_) => Box::new(websearch::providers::duckduckgo::DuckDuckGoProvider::new()),
                                }
                            }
                            crate::settings::WebSearchProvider::Tavily => {
                                let key = settings.tavily_api_key.or_else(|| std::env::var("TAVILY_API_KEY").ok()).unwrap_or_default();
                                match websearch::providers::tavily::TavilyProvider::new(&key) {
                                    Ok(p) => Box::new(p),
                                    Err(_) => Box::new(websearch::providers::duckduckgo::DuckDuckGoProvider::new()),
                                }
                            }
                            crate::settings::WebSearchProvider::Searxng => {
                                let url = settings.searxng_url.or_else(|| std::env::var("SEARXNG_URL").ok()).unwrap_or_else(|| "http://localhost:8080".to_string());
                                match websearch::providers::searxng::SearxNGProvider::new(&url) {
                                    Ok(p) => Box::new(p),
                                    Err(_) => Box::new(websearch::providers::duckduckgo::DuckDuckGoProvider::new()),
                                }
                            }
                            crate::settings::WebSearchProvider::Arxiv => {
                                Box::new(websearch::providers::arxiv::ArxivProvider::new())
                            }
                        };
                        let options = websearch::SearchOptions {
                            query: action.target.clone(),
                            max_results: Some(3),
                            provider: provider_box,
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
            ProposedKind::Mcp => {
                let (server_opt, tool_name) = if let Some((s, t)) = action.target.split_once(':') {
                    (Some(s), t)
                } else {
                    (None, action.target.as_str())
                };
                tokio::task::block_in_place(|| {
                    tokio::runtime::Handle::current().block_on(async {
                        let mcp_mgr_lock = crate::mcp::McpManager::instance().await;
                        let mut mcp_guard = mcp_mgr_lock.lock().await;
                        if let Some(ref mut mgr) = *mcp_guard {
                            mgr.sync_with_settings().await;
                            match mgr.execute_tool(server_opt, tool_name, &action.body).await {
                                Ok(res) => res.to_plain_text(),
                                Err(e) => e,
                            }
                        } else {
                            "Error: MCP Manager is not initialized.".to_string()
                        }
                    })
                })
            }
            ProposedKind::Skill | ProposedKind::Agent => String::new(), // handled elsewhere
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

    /// Validates syntax of tool tags in the response and returns System error messages if malformed.
    pub fn validate_tool_tags(response: &str) -> Vec<String> {
        let outside = Self::strip_think_blocks(response);
        let mut errors = Vec::new();

        // Check <write> tags
        let mut text = outside.as_str();
        while let Some(start) = text.find("<write") {
            let rest = &text[start..];
            if let Some(close_bracket) = rest.find('>') {
                let header = &rest[..close_bracket + 1];
                let src_attr = Self::extract_attribute(header, "src");
                let action_attr = Self::extract_attribute(header, "action");
                let line_attr = Self::extract_attribute(header, "line");

                if src_attr.is_none() {
                    errors.push("System Error: `<write>` tag is missing the required `src=\"...\"` attribute. Example: `<write src=\"index.html\">...content...</write>`.".to_string());
                }

                if let Some(action) = action_attr {
                    if action == "replace" && line_attr.is_none() {
                        errors.push("System Error: `<write action=\"replace\">` is missing the required `line=START..=END` attribute. Example: `<write src=\"path\" line=10..=15>...replacement...</write>`.".to_string());
                    }
                }

                if let Some(line) = line_attr {
                    if Self::parse_range(&line).is_none() {
                        errors.push(format!("System Error: Invalid line range format in `<write line={line}>`. Use format `line=START..=END` (e.g., `line=5..=20`)."));
                    }
                }
                text = &rest[close_bracket + 1..];
            } else {
                break;
            }
        }

        // Check <read> tags
        text = outside.as_str();
        while let Some(start) = text.find("<read") {
            let rest = &text[start..];
            if let Some(close_bracket) = rest.find('>') {
                let header = &rest[..close_bracket + 1];
                if Self::extract_attribute(header, "src").is_none() {
                    errors.push("System Error: `<read>` tag is missing the required `src=\"...\"` attribute. Example: `<read src=\"path/to/file\">`.".to_string());
                }
                text = &rest[close_bracket + 1..];
            } else {
                break;
            }
        }

        errors
    }

    /// Process tool tags in an agent response.
    ///
    /// Canonical split of responsibilities:
    /// - `<read>` / `<ls>` are parsed into canonical [`ProposedAction`]s by
    ///   [`Self::parse_tool_calls`] and executed ONLY through the App-level
    ///   canonical dispatcher (claim → [`Self::execute_proposed`] → chip →
    ///   result → context). They NEVER run here.
    /// - This function handles `<help>` (think-only), `<write>` (AlwaysAllow
    ///   only, minus already-streamed targets) and `<memory>`.
    /// - **Cmds never run here** — they go through the app task manager.
    pub fn process_response(response: &str) -> Option<String> {
        Self::process_response_with(response, &[])
    }

    /// Same as [`Self::process_response`], but skips `<write>` actions whose
    /// target already executed during streaming (`skip_write_targets`).
    /// Without this, AlwaysAllow mode applies every streamed write twice:
    /// once live from `sync` polling and once here at completion.
    pub fn process_response_with(response: &str, skip_write_targets: &[String]) -> Option<String> {
        let tag_errors = Self::validate_tool_tags(response);
        if !tag_errors.is_empty() {
            return Some(tag_errors.join("\n"));
        }
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
                    if skip_write_targets.iter().any(|t| t == &action.target) {
                        continue;
                    }
                    results.push(Self::execute_proposed(&action));
                }
            }
        }

        // 2. <read> / <ls> NEVER run here — canonical dispatcher owns them
        // (see parse_read_ls_actions + App::claim_tool_call). Kept out so one
        // model tag cannot execute through two paths.
        // 3. Memory — OUTSIDE <think> and <write> bodies only
        let scan_text = Self::blank_closed_write_blocks(&cleaned_outside_think);
        let mut text = scan_text.as_str();
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

    /// Remove closed `<write …>…</write>` blocks from scan input so searches
    /// for OTHER tool tags never match example markup inside a file body
    /// being written (e.g. an HTML sample containing `<read src=…>` text).
    /// Only fully closed blocks are removed; a trailing unclosed write keeps
    /// current semantics (streaming live-detect relies on it).
    fn blank_closed_write_blocks(text: &str) -> String {
        let mut out = String::with_capacity(text.len());
        let mut rest = text;
        while let Some(start) = rest.find("<write") {
            let r = &rest[start..];
            let Some(hdr_end) = r.find('>') else {
                break; // partial header — keep remainder as-is
            };
            let after_hdr = &r[hdr_end + 1..];
            let Some(close_start) = after_hdr.find("</write") else {
                break; // unclosed trailing write — keep remainder as-is
            };
            let close_abs = hdr_end + 1 + close_start;
            let tail = &after_hdr[close_start..];
            let cut = if let Some(gt) = tail.find('>') {
                close_abs + gt + 1
            } else {
                r.len()
            };
            out.push_str(&rest[..start]);
            rest = &r[cut.min(r.len())..];
        }
        out.push_str(rest);
        out
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
                if old_lines[i - 1] == new_lines[j - 1] {
                    dp[i][j] = dp[i - 1][j - 1] + 1;
                } else {
                    dp[i][j] = dp[i - 1][j].max(dp[i][j - 1]);
                }
            }
        }

        let mut i = n;
        let mut j = m;
        let mut diff = Vec::new();
        while i > 0 || j > 0 {
            if i > 0 && j > 0 && old_lines[i - 1] == new_lines[j - 1] {
                diff.push(format!("  {:4} | {}", j, new_lines[j - 1]));
                i -= 1;
                j -= 1;
            } else if j > 0 && (i == 0 || dp[i][j - 1] >= dp[i - 1][j]) {
                diff.push(format!("+ {:4} | {}", j, new_lines[j - 1]));
                j -= 1;
            } else if i > 0 && (j == 0 || dp[i][j - 1] < dp[i - 1][j]) {
                diff.push(format!("- {:4} | {}", i, old_lines[i - 1]));
                i -= 1;
            }
        }
        diff.reverse();
        diff.join("\n")
    }

    pub fn execute_write(path_str: &str, line_attr: Option<&str>, body: &str) -> String {
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
    fn test_think_tools_never_execute() {
        // Tools inside <think> are reasoning text, NEVER executable actions.
        // They must not even parse into canonical calls...
        for sample in [
            "<think>Let me list <ls path=\".\"></think>",
            "<think>Reading <read src=\"Cargo.toml\"></think>",
            "<think>Running <cmd>ls</cmd></think>",
        ] {
            assert!(
                AgentEngine::parse_tool_calls(sample, ToolCallSource::ModelCompletion).is_empty()
            );
            assert!(AgentEngine::process_response(sample).is_none());
        }

        // ...while the same tags outside think parse to exactly one call and
        // execute through the canonical executor.
        let read_calls = AgentEngine::parse_tool_calls(
            "<read src=\"Cargo.toml\">",
            ToolCallSource::ModelCompletion,
        );
        assert_eq!(read_calls.len(), 1);
        assert_eq!(read_calls[0].kind, ProposedKind::Read);
        assert!(AgentEngine::execute_proposed(&read_calls[0]).contains("version"));

        let ls_calls =
            AgentEngine::parse_tool_calls("<ls path=\".\">", ToolCallSource::ModelCompletion);
        assert_eq!(ls_calls.len(), 1);
        assert!(AgentEngine::execute_proposed(&ls_calls[0]).contains("Cargo.toml"));
    }

    #[test]
    fn test_write_body_tags_are_not_actions() {
        // Tags quoted inside a <write> file body are content, never actions.
        let sample = "<write src=\"$CURRENT/probe.html\">\n<html>\n<read src=\"$CURRENT/other\">\n<cmd>rm -rf /tmp/probe</cmd>\n<mcp server=\"s\" tool=\"t\">{}</mcp>\n<websearch>docs</websearch>\n</write>";
        let actions = AgentEngine::extract_proposed_actions(sample);
        assert_eq!(actions.len(), 1);
        assert_eq!(actions[0].kind, ProposedKind::Write);
        // process_response must not execute the quoted read either
        // (outer write is inert under default Ask permissions).
        assert!(AgentEngine::process_response(sample).is_none());
    }

    #[test]
    fn test_response_has_tool_tags_ignores_thinking() {
        assert!(!AgentEngine::response_has_tool_tags(
            "<think><cmd>ls</cmd></think>"
        ));
        // Unclosed think (still streaming): everything after is thinking.
        assert!(!AgentEngine::response_has_tool_tags(
            "hmm <think><cmd>ls</cmd>"
        ));
        assert!(AgentEngine::response_has_tool_tags(
            "<think>hmm</think><cmd>ls</cmd>"
        ));
        assert!(matches!(
            AgentEngine::classify_output("<think><cmd>ls</cmd></think>"),
            ModelOutput::Text
        ));
    }

    // ── Canonical ToolCall lifecycle tests ──────────────────────────────
    use ToolCallSource::*;

    #[test]
    fn test_lifecycle_think_tools_yield_no_calls() {
        for sample in [
            "<think>I should use read</think>",
            "<think><cmd>ls</cmd></think>",
            "<think><websearch>foo</websearch></think>",
            "<think><read src=\"a\"><write src=\"b\">x</write></think>",
        ] {
            assert!(
                AgentEngine::parse_tool_calls(sample, ModelCompletion).is_empty(),
                "think must never become ToolCall: {sample}"
            );
        }
    }

    #[test]
    fn test_lifecycle_prose_yields_no_calls() {
        for sample in [
            "I should use read.",
            "I will search the web.",
            "Action: SEARCH.",
            "Tool: read.",
            "I ran cargo check and it passed.",
        ] {
            assert!(
                AgentEngine::parse_tool_calls(sample, ModelCompletion).is_empty(),
                "prose must never become ToolCall: {sample}"
            );
        }
    }

    #[test]
    fn test_lifecycle_real_tags_yield_one_call_each() {
        let cases = [
            ("<read src=\"$CURRENT/Cargo.toml\">", ProposedKind::Read),
            ("<ls path=\"$CURRENT\">", ProposedKind::Ls),
            ("<cmd>cargo check</cmd>", ProposedKind::Cmd),
            (
                "<write src=\"$CURRENT/f.txt\">\nhello\n</write>",
                ProposedKind::Write,
            ),
            (
                "<websearch query=\"rust borrow checker\">",
                ProposedKind::WebSearch,
            ),
            (
                "<mcp server=\"s\" tool=\"t\">{\"a\": 1}</mcp>",
                ProposedKind::Mcp,
            ),
            ("<skill action=\"search\">q</skill>", ProposedKind::Skill),
        ];
        for (sample, kind) in cases {
            let calls = AgentEngine::parse_tool_calls(sample, ModelCompletion);
            assert_eq!(calls.len(), 1, "wrong call count: {sample}");
            assert_eq!(calls[0].kind, kind, "wrong kind: {sample}");
        }
        // And classification agrees each is a ToolCall.
        assert!(matches!(
            AgentEngine::classify_output("<read src=\"$CURRENT/Cargo.toml\">"),
            ModelOutput::ToolCall
        ));
    }

    #[test]
    fn test_lifecycle_multiple_calls_count() {
        let sample = "<cmd>cargo check</cmd>\n<websearch query=\"x\">";
        let calls = AgentEngine::parse_tool_calls(sample, ModelCompletion);
        assert_eq!(calls.len(), 2);
        assert_ne!(calls[0].call_id, calls[1].call_id);
    }

    #[test]
    fn test_lifecycle_incomplete_stream_is_preview_only() {
        // Incomplete tool: preview view exists but is NOT closed → the
        // dispatcher must never see it as executable.
        let views = crate::tool_panel::detect_all_stream_tools("<cmd>cargo check");
        assert!(!views.iter().any(|v| v.tag_closed));
        let views_closed = crate::tool_panel::detect_all_stream_tools("<cmd>cargo check</cmd>");
        assert!(views_closed.iter().any(|v| v.tag_closed));
    }

    #[test]
    fn test_lifecycle_completed_call_executes_exactly_once() {
        let calls = AgentEngine::parse_tool_calls("<cmd>cargo check</cmd>", ModelStream);
        assert_eq!(calls.len(), 1);
        let mut reg = ToolDispatchRegistry::new();
        assert!(reg.try_claim(&calls[0])); // streaming dispatch
        assert!(!reg.try_claim(&calls[0])); // completion re-dispatch refused
        // Re-parse (as completion does on the full text) → same fingerprint.
        let again = AgentEngine::parse_tool_calls("<cmd>cargo check</cmd>", ModelCompletion);
        assert_eq!(again.len(), 1);
        assert!(!reg.try_claim(&again[0]));
    }

    #[test]
    fn test_lifecycle_recovery_is_host_recovery_call() {
        let tag = AgentEngine::recover_tools_from_refusal(
            "please read Cargo.toml",
            "I cannot read files.",
        );
        let tag = tag.expect("read request must recover");
        let calls = AgentEngine::parse_tool_calls(&tag, HostRecovery);
        assert!(!calls.is_empty());
        assert!(calls.iter().all(|c| c.source == HostRecovery));
    }

    #[test]
    fn test_lifecycle_result_binds_to_call_and_context() {
        let calls = AgentEngine::parse_tool_calls("<cmd>cargo check</cmd>", ModelCompletion);
        let res = ToolResult::new(&calls[0], "ok".to_string());
        assert_eq!(res.call_id, calls[0].call_id);
        assert_eq!(res.fingerprint, calls[0].fingerprint());
        let entry = res.context_entry("RUN");
        assert!(entry.contains(&calls[0].call_id.to_string()));
        assert!(entry.contains("ok"));
    }

    #[test]
    fn test_lifecycle_no_duplicate_execution_paths() {
        // Same write seen by streaming + completion: registry suppresses 2nd.
        let stream_calls = AgentEngine::parse_tool_calls(
            "<write src=\"$CURRENT/d.txt\">\nhi\n</write>",
            ModelStream,
        );
        let mut reg = ToolDispatchRegistry::new();
        assert!(reg.try_claim(&stream_calls[0]));
        let completion_calls = AgentEngine::parse_tool_calls(
            "<write src=\"$CURRENT/d.txt\">\nhi\n</write>",
            ModelCompletion,
        );
        assert!(!reg.try_claim(&completion_calls[0]));
        // And the target skip-list also suppresses re-execution.
        let out = AgentEngine::process_response_with(
            "<write src=\"$CURRENT/d.txt\">\nhi\n</write>",
            &["$CURRENT/d.txt".to_string()],
        );
        assert!(out.is_none());
    }

    #[test]
    fn test_lifecycle_call_ids_unique_no_backend_ids() {
        // Backends return plain text (no tool_call_id); the parser mints
        // unique ids while fingerprints stay stable across re-parses.
        let a = AgentEngine::parse_tool_calls("<cmd>cargo check</cmd>", ModelStream);
        let b = AgentEngine::parse_tool_calls("<cmd>cargo check</cmd>", ModelCompletion);
        assert_ne!(a[0].call_id, b[0].call_id);
        assert_eq!(a[0].fingerprint(), b[0].fingerprint());
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

    #[test]
    fn test_streaming_parser_blocks_tools_in_thinking() {
        use crate::agent::StreamingParser;

        // Tools outside thinking should be extracted immediately
        let mut parser = StreamingParser::new();
        let actions = parser.feed(r#"<write src="$CURRENT/test.txt">hello</write>"#);
        assert_eq!(actions.len(), 1);
        assert_eq!(actions[0].kind, ProposedKind::Write);
        let actions = parser.flush();
        assert_eq!(actions.len(), 0);

        // Tool inside thinking should be blocked
        let mut parser = StreamingParser::new();
        let actions =
            parser.feed(r#"<think> <write src="$CURRENT/test.txt">hello</write> </think>"#);
        assert_eq!(actions.len(), 0);

        // Tool after thinking should work
        let mut parser = StreamingParser::new();
        let actions = parser
            .feed(r#"<think> thinking </think> <write src="$CURRENT/test.txt">hello</write>"#);
        assert_eq!(actions.len(), 1);
        assert_eq!(actions[0].kind, ProposedKind::Write);

        // Streaming chunks: thinking starts in one chunk, ends in another
        let mut parser = StreamingParser::new();
        let actions1 = parser.feed(r#"<think> <write src="$CURRENT/inside.txt">"#);
        assert_eq!(actions1.len(), 0);

        let actions2 =
            parser.feed(r#"blocked</write> </think> <write src="$CURRENT/outside.txt">ok</write>"#);
        assert_eq!(actions2.len(), 1);
        assert_eq!(actions2[0].kind, ProposedKind::Write);
        assert_eq!(actions2[0].target, "$CURRENT/outside.txt");

        let actions3 = parser.flush();
        assert_eq!(actions3.len(), 0);
    }

    #[test]
    fn test_streaming_parser_nested_thinking() {
        use crate::agent::StreamingParser;

        // Nested query in thinking should be treated as content, not as exit
        let mut parser = StreamingParser::new();
        let actions = parser.feed(r#"<think> outer <think> nested query <write src="$CURRENT/test.txt">hello</write> </think>"#);
        assert_eq!(actions.len(), 0);

        // Proper exit
        let mut parser = StreamingParser::new();
        let actions = parser
            .feed(r#"<think> thinking </think> <write src="$CURRENT/test.txt">hello</write>"#);
        assert_eq!(actions.len(), 1);
    }

    #[test]
    fn test_streaming_parser_in_thinking_state() {
        use crate::agent::StreamingParser;

        let mut parser = StreamingParser::new();
        assert!(!parser.in_thinking());

        parser.feed("<think>");
        assert!(parser.in_thinking());

        parser.feed("</think>");
        assert!(!parser.in_thinking());
    }

    #[test]
    fn test_classify_output_ask_mode_wins_over_tools() {
        use crate::agent::{ModelOutput, ProposedKind};

        // Plain question text → Text.
        assert!(matches!(
            AgentEngine::classify_output("What is 2 + 2?"),
            ModelOutput::Text
        ));

        // Tool tags outside think → ToolCall.
        assert!(matches!(
            AgentEngine::classify_output(r#"<read src="$CURRENT/Cargo.toml">"#),
            ModelOutput::ToolCall
        ));

        // Tool-like text inside think only → Text (never executable).
        assert!(matches!(
            AgentEngine::classify_output("<think>Maybe <cmd>ls</cmd>?</think>"),
            ModelOutput::Text
        ));

        // Ask Mode wins even when tool-like text is also present.
        let ask = "<askmode ques=\"Which database?\"><radio>SQLite</radio><radio>Postgres</radio></askmode>";
        let mixed = format!("{ask}\n<read src=\"$CURRENT/Cargo.toml\">");
        match AgentEngine::classify_output(&mixed) {
            ModelOutput::AskMode(mode) => {
                assert_eq!(mode.question, "Which database?");
                assert_eq!(mode.radio_count(), 2);
            }
            _ => panic!("Ask Mode must classify before tool execution"),
        }

        // extract_proposed_actions agrees: think-only tools yield nothing.
        let actions = AgentEngine::extract_proposed_actions("<think><cmd>rm -rf /</cmd></think>");
        assert!(actions.is_empty());
        let actions = AgentEngine::extract_proposed_actions("<think>hmm</think><cmd>ls</cmd>");
        assert_eq!(actions.len(), 1);
        assert!(matches!(actions[0].kind, ProposedKind::Cmd));
    }
}
