//! Tool chips (size-to-fit, under agent) + KramaFrame-driven fly panel.
//!
//! Animation uses KramaFrame progress 0→1 (open) / reverse (close), like the
//! official TUI example: update_progress each frame, get_progress_f32 for t.
//! Geometry: lerp(chip_rect, dock_rect, ease(t)).

use ratatui::{
    layout::Rect,
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, BorderType, Clear, Paragraph, Wrap},
    Frame,
};
use std::time::Instant;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolPanelKind {
    Write,
    Cmd,
    Read,
    Mcp,
    Skill,
    WebSearch,
    Agent,
}

impl ToolPanelKind {
    pub fn title_prefix(self) -> &'static str {
        match self {
            Self::Write => "WRITE",
            Self::Cmd => "TERM",
            Self::Read => "READ",
            Self::Mcp => "MCP",
            Self::Skill => "SKILL",
            Self::WebSearch => "SEARCH",
            Self::Agent => "AGENT",
        }
    }

    pub fn accent(self) -> Color {
        match self {
            Self::Write => Color::Rgb(80, 220, 140),
            Self::Cmd => Color::Rgb(255, 200, 80),
            Self::Read => Color::Rgb(100, 180, 255),
            Self::Mcp => Color::Rgb(200, 100, 255),
            Self::Skill => Color::Rgb(255, 180, 50),
            Self::WebSearch => Color::Rgb(100, 255, 180),
            Self::Agent => Color::Rgb(255, 100, 200),
        }
    }

    pub fn final_fg(self) -> Color {
        match self {
            Self::Write => Color::Rgb(210, 255, 225),
            Self::Cmd => Color::Rgb(180, 255, 180), // terminal green
            Self::Read => Color::Rgb(200, 230, 255),
            Self::Mcp => Color::Rgb(220, 150, 255),
            Self::Skill => Color::Rgb(255, 200, 100),
            Self::WebSearch => Color::Rgb(150, 255, 200),
            Self::Agent => Color::Rgb(255, 150, 200),
        }
    }
}

const CHAR_MS: f32 = 12.0;
const FADE_MS: f32 = 100.0;

/// Size-to-fit bordered button under the agent turn that called the tool.
#[derive(Debug, Clone)]
pub struct ToolChip {
    pub id: u64,
    pub kind: ToolPanelKind,
    pub target: String,
    pub body: String,
    pub tag_closed: bool,
    pub pending: bool,
    pub spawned: bool,
    pub rect: Option<Rect>,
    /// `messages` index of the Agent turn that emitted this tool (scrolls with chat).
    pub anchor_msg: Option<usize>,
}

/// Terminal rows reserved under an agent turn for one chip (bordered 3-row button).
pub const CHIP_ROW_HEIGHT: u16 = 3;

impl ToolChip {
    pub fn label_text(&self) -> String {
        match self.kind {
            ToolPanelKind::Write => {
                let short = self.target.rsplit('/').next().unwrap_or(&self.target);
                let lines = line_count(&self.body);
                if self.pending {
                    // tag closed but waiting for user to press Y — NOT written yet
                    format!(" [PENDING] {short} | {lines} lines > ")
                } else if self.tag_closed {
                    format!(" [WROTE] {short} | {lines} lines > ")
                } else {
                    format!(" [WRITE] {short} | {lines} lines... > ")
                }
            }
            ToolPanelKind::Cmd => {
                let cmd = clean_cmd(&self.target);
                let cmd = trunc(&cmd, 42);
                if self.pending {
                    format!(" [RUN] `{cmd}` > ")
                } else if !self.body.is_empty() {
                    format!(" [RAN] `{cmd}` > ")
                } else {
                    format!(" [RUN] `{cmd}` > ")
                }
            }
            ToolPanelKind::Read => {
                let short = self.target.rsplit('/').next().unwrap_or(&self.target);
                let short = trunc(short, 36);
                let lines = line_count(&self.body);
                if self.tag_closed && !self.body.is_empty() {
                    format!(" [READ] {short} | {lines} lines > ")
                } else if self.tag_closed {
                    format!(" [READ] {short} > ")
                } else {
                    format!(" [READ] {short}... > ")
                }
            }
            ToolPanelKind::Mcp | ToolPanelKind::Skill | ToolPanelKind::WebSearch | ToolPanelKind::Agent => format!(" [{}] ", self.target),
        }
    }

    pub fn fit_width(&self) -> u16 {
        (self.label_text().chars().count() as u16 + 2).clamp(14, 70)
    }

    pub fn draw_at(&mut self, frame: &mut Frame, x: u16, y: u16, max_w: u16) {
        let w = self.fit_width().min(max_w);
        let h = 3u16;
        let area = Rect {
            x,
            y,
            width: w,
            height: h,
        };
        let accent = self.kind.accent();
        frame.render_widget(Clear, area);
        let block = Block::default()
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .border_style(Style::default().fg(accent));
        let inner = block.inner(area);
        frame.render_widget(block, area);
        frame.render_widget(
            Paragraph::new(Line::from(Span::styled(
                self.label_text(),
                Style::default().fg(accent).add_modifier(Modifier::BOLD),
            ))),
            inner,
        );
        self.rect = Some(area);
    }
}

/// Flying detail panel. Progress `t` comes from KramaFrame (0..=1).
#[derive(Debug, Clone)]
pub struct ToolPanel {
    pub chip_id: u64,
    pub kind: ToolPanelKind,
    pub target: String,
    pub body: String,
    pub minimized: bool,
    pub tag_closed: bool,
    pub scroll: u16,
    pub revealed_chars: usize,
    last_reveal: Instant,
    char_born: Vec<Instant>,
    pub chip_rect: Option<Rect>,
    pub dock_rect: Option<Rect>,
    pub drawn_rect: Option<Rect>,
    /// Live stream: keep reveal glued to body end
    pub live_stream: bool,
    /// Hit zones for title chrome (updated every draw)
    pub min_hit: Option<Rect>,
    pub close_hit: Option<Rect>,
    /// Max scroll for body (updated on draw from line count / height)
    pub max_scroll: u16,
    /// Interactive TERM: user clicked panel body (keys go to term input)
    pub interactive: bool,
    /// Keep view pinned to bottom while writing / streaming (cleared on manual scroll-up)
    pub follow_end: bool,
}

impl ToolPanel {
    pub fn from_chip(chip: &ToolChip) -> Self {
        Self {
            chip_id: chip.id,
            kind: chip.kind,
            target: clean_cmd(&chip.target),
            body: chip.body.clone(),
            minimized: false,
            tag_closed: chip.tag_closed,
            scroll: 0,
            revealed_chars: 0,
            last_reveal: Instant::now(),
            char_born: Vec::new(),
            chip_rect: chip.rect,
            dock_rect: None,
            drawn_rect: None,
            live_stream: !chip.tag_closed,
            min_hit: None,
            close_hit: None,
            max_scroll: 0,
            interactive: false,
            follow_end: true,
        }
    }

    pub fn scroll_by(&mut self, delta: i32) {
        if delta < 0 {
            self.scroll = self.scroll.saturating_sub((-delta) as u16);
            // User scrolled up — stop auto-follow until they hit bottom again
            self.follow_end = false;
        } else {
            self.scroll = (self.scroll.saturating_add(delta as u16)).min(self.max_scroll);
            if self.scroll >= self.max_scroll {
                self.follow_end = true;
            }
        }
    }

    pub fn scroll_to_end(&mut self) {
        self.scroll = self.max_scroll;
        self.follow_end = true;
    }

    pub fn set_body_streaming(&mut self, body: String, tag_closed: bool) {
        let grew = body.len() > self.body.len();
        self.body = body;
        self.tag_closed = tag_closed;
        if !tag_closed {
            self.live_stream = true;
        }
        // Stream text into open container: reveal all received chars immediately
        if self.live_stream {
            self.sync_reveal_to_body();
        }
        // Follow writing cursor to bottom while streaming / growing
        if self.live_stream || grew {
            self.follow_end = true;
        }
        if tag_closed {
            self.live_stream = false;
            // One last snap to end so user sees the finish
            self.follow_end = true;
        }
    }

    fn sync_reveal_to_body(&mut self) {
        let n = self.body.chars().count();
        let now = Instant::now();
        while self.char_born.len() < n {
            self.char_born.push(now);
        }
        self.revealed_chars = n;
    }

    pub fn reveal_all(&mut self) {
        self.sync_reveal_to_body();
        self.live_stream = false;
    }

    pub fn tick_reveal(&mut self) {
        if self.live_stream {
            self.sync_reveal_to_body();
            return;
        }
        let total = self.body.chars().count();
        if self.revealed_chars >= total {
            return;
        }
        if self.last_reveal.elapsed().as_secs_f32() * 1000.0 < CHAR_MS {
            return;
        }
        self.revealed_chars += 1;
        self.last_reveal = Instant::now();
        self.char_born.push(Instant::now());
    }

    pub fn visible_body(&self) -> String {
        self.body.chars().take(self.revealed_chars).collect()
    }

    fn char_color(&self, idx: usize, now: Instant) -> Color {
        let final_c = self.kind.final_fg();
        let accent = self.kind.accent();
        let Some(&born) = self.char_born.get(idx) else {
            return final_c;
        };
        let ms = now.duration_since(born).as_secs_f32() * 1000.0;
        if ms >= FADE_MS {
            return final_c;
        }
        let u = ease_out_cubic((ms / FADE_MS).clamp(0.0, 1.0));
        lerp_color(accent, final_c, u)
    }
}

fn line_count(body: &str) -> usize {
    if body.is_empty() {
        0
    } else {
        body.lines().filter(|l| !l.trim().is_empty()).count().max(1)
    }
}

fn clean_cmd(s: &str) -> String {
    s.trim()
        .trim_end_matches('<')
        .trim_end_matches('/')
        .trim_end_matches('>')
        .trim()
        .to_string()
}

/// Collapse whitespace so "cargo  check" matches "cargo check".
pub fn normalize_target(kind: ToolPanelKind, target: &str) -> String {
    let t = clean_cmd(target);
    match kind {
        ToolPanelKind::Cmd => t.split_whitespace().collect::<Vec<_>>().join(" "),
        ToolPanelKind::Write | ToolPanelKind::Read | ToolPanelKind::Mcp | ToolPanelKind::Skill | ToolPanelKind::WebSearch | ToolPanelKind::Agent => t,
        }
}

/// Same tool event? Used to upsert chips *within one stream/turn* only.
/// Callers must also match `anchor_msg` so past chips stay clickable.
pub fn same_tool_target(kind: ToolPanelKind, a: &str, b: &str) -> bool {
    let na = normalize_target(kind, a);
    let nb = normalize_target(kind, b);
    if na == nb {
        return true;
    }
    // path suffix / basename match for file tools
    if matches!(kind, ToolPanelKind::Write | ToolPanelKind::Read) {
        let ba = na.rsplit('/').next().unwrap_or(&na);
        let bb = nb.rsplit('/').next().unwrap_or(&nb);
        if ba == bb && !ba.is_empty() {
            return true;
        }
        return na.ends_with(&nb) || nb.ends_with(&na);
    }
    // cmd: allow trailing garbage from partial stream
    na.starts_with(&nb) || nb.starts_with(&na)
}

fn trunc(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        format!("{}...", s.chars().take(max.saturating_sub(1)).collect::<String>())
    }
}

fn ease_out_cubic(t: f32) -> f32 {
    let u = 1.0 - t;
    1.0 - u * u * u
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

fn lerp_color(a: Color, b: Color, t: f32) -> Color {
    let (ar, ag, ab) = match a {
        Color::Rgb(r, g, b) => (r as f32, g as f32, b as f32),
        _ => (180.0, 180.0, 180.0),
    };
    let (br, bg, bb) = match b {
        Color::Rgb(r, g, b) => (r as f32, g as f32, b as f32),
        _ => (220.0, 220.0, 220.0),
    };
    Color::Rgb(
        (ar + (br - ar) * t) as u8,
        (ag + (bg - ag) * t) as u8,
        (ab + (bb - ab) * t) as u8,
    )
}

// ---------------------------------------------------------------------------
// Stream detect
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct StreamToolView {
    pub kind: ToolPanelKind,
    pub target: String,
    pub body: String,
    pub tag_closed: bool,
}


fn detect_mcp(text: &str) -> Vec<StreamToolView> {
    let mut out = Vec::new();
    let mut rest = text;
    while let Some(start) = rest.find("<mcp ") {
        let r = &rest[start..];
        if let Some(close_bracket) = r.find('>') {
            let header = &r[..close_bracket + 1];
            let action = crate::agent::AgentEngine::extract_attribute(header, "action").unwrap_or_else(|| "search".to_string());
            let end = r.find("</mcp>").unwrap_or(r.len());
            let body = r[close_bracket + 1..end].trim().to_string();
            out.push(StreamToolView {
                kind: ToolPanelKind::Mcp,
                target: action,
                body,
                tag_closed: r.find("</mcp>").is_some(),
            });
            rest = if r.find("</mcp>").is_some() { &r[end + 6..] } else { "" };
        } else {
            break;
        }
    }
    out
}

fn detect_skill(text: &str) -> Vec<StreamToolView> {
    let mut out = Vec::new();
    let mut rest = text;
    while let Some(start) = rest.find("<skill ") {
        let r = &rest[start..];
        if let Some(close_bracket) = r.find('>') {
            let header = &r[..close_bracket + 1];
            let action = crate::agent::AgentEngine::extract_attribute(header, "action").unwrap_or_else(|| "search".to_string());
            let end = r.find("</skill>").unwrap_or(r.len());
            let body = r[close_bracket + 1..end].trim().to_string();
            out.push(StreamToolView {
                kind: ToolPanelKind::Skill,
                target: action,
                body,
                tag_closed: r.find("</skill>").is_some(),
            });
            rest = if r.find("</skill>").is_some() { &r[end + 8..] } else { "" };
        } else {
            break;
        }
    }
    out
}



fn detect_websearch(text: &str) -> Vec<StreamToolView> {
    let mut out = Vec::new();
    let mut rest = text;
    while let Some(start) = rest.find("<websearch ") {
        let r = &rest[start..];
        if let Some(close_bracket) = r.find('>') {
            let header = &r[..close_bracket + 1];
            let action = crate::agent::AgentEngine::extract_attribute(header, "query").unwrap_or_else(|| "search".to_string());
            let end = r.find("</websearch>").unwrap_or(r.len());
            let body = r[close_bracket + 1..end].trim().to_string();
            out.push(StreamToolView {
                kind: ToolPanelKind::WebSearch,
                target: action,
                body,
                tag_closed: r.find("</websearch>").is_some(),
            });
            rest = if r.find("</websearch>").is_some() { &r[end + 12..] } else { "" };
        } else {
            break;
        }
    }
    out
}



fn detect_agent(text: &str) -> Vec<StreamToolView> {
    let mut out = Vec::new();
    let mut rest = text;
    while let Some(start) = rest.find("<agent ") {
        let r = &rest[start..];
        if let Some(close_bracket) = r.find('>') {
            let header = &r[..close_bracket + 1];
            let action = crate::agent::AgentEngine::extract_attribute(header, "action").unwrap_or_else(|| "spawn".to_string());
            let role = crate::agent::AgentEngine::extract_attribute(header, "role").unwrap_or_default();
            let to = crate::agent::AgentEngine::extract_attribute(header, "to").unwrap_or_default();
            
            let mut target_label = action.clone();
            if !role.is_empty() {
                target_label.push_str(&format!(" role={role}"));
            }
            if !to.is_empty() {
                target_label.push_str(&format!(" to={to}"));
            }
            
            let end = r.find("</agent>").unwrap_or(r.len());
            let body = r[close_bracket + 1..end].trim().to_string();
            out.push(StreamToolView {
                kind: ToolPanelKind::Agent,
                target: target_label,
                body,
                tag_closed: r.find("</agent>").is_some(),
            });
            rest = if r.find("</agent>").is_some() { &r[end + 8..] } else { "" };
        } else {
            break;
        }
    }
    out
}


pub fn detect_all_stream_tools(response: &str) -> Vec<StreamToolView> {
    let text = flatten_for_tools(response);
    let mut out = Vec::new();
    // Prefer the *active* write (last open, else last closed) so path renames
    // mid-stream don't spawn a chip per intermediate filename.
    if let Some(w) = detect_primary_write(&text) {
        out.push(w);
    }
    if let Some(c) = detect_cmd(&text) {
        out.push(c);
    }
    // All <read> tags in the stream (not just first)
    out.extend(detect_reads(&text));
    out.extend(detect_ls(&text));
    out.extend(detect_mcp(&text));
    out.extend(detect_skill(&text));
    out.extend(detect_websearch(&text));
    out.extend(detect_agent(&text));
    out
}

fn flatten_for_tools(response: &str) -> String {
    let outside = crate::agent::AgentEngine::strip_code_fences(
        &crate::agent::AgentEngine::strip_think_blocks(response),
    );
    let mut think = crate::agent::AgentEngine::extract_think_contents(response);
    if think.is_empty() {
        if let Some(i) = response.find("<think>") {
            think = response[i + 7..].to_string();
            if let Some(j) = think.find("</think>") {
                think = think[..j].to_string();
            }
        }
    }
    let think = crate::agent::AgentEngine::strip_code_fences(&think);
    if outside.contains("<write")
        || outside.contains("<cmd>")
        || outside.contains("<read src=")
        || outside.contains("<mcp")
        || outside.contains("<skill")
        || outside.contains("<websearch")
        || outside.contains("<agent")
    {
        outside
    } else {
        format!("{outside}\n{think}")
    }
}

fn detect_ls(text: &str) -> Vec<StreamToolView> {
    let outside = crate::agent::AgentEngine::strip_think_blocks(text);
    let search_in = if outside.contains("<ls") {
        outside.as_str()
    } else {
        text
    };
    let mut out = Vec::new();
    let mut rest = search_in;
    while let Some(start) = rest.find("<ls") {
        let r = &rest[start..];
        let Some(gt) = r.find('>') else { break };
        let path = extract_attr(&r[..gt + 1], "path").unwrap_or_else(|| "$CURRENT".into());
        out.push(StreamToolView {
            kind: ToolPanelKind::Read,
            target: expand_path_display(&path),
            body: String::new(),
            tag_closed: true,
        });
        rest = &r[gt + 1..];
    }
    out
}

fn detect_reads(text: &str) -> Vec<StreamToolView> {
    let outside = crate::agent::AgentEngine::strip_think_blocks(text);
    let search_in = if outside.contains("<read src=") {
        outside.as_str()
    } else {
        text
    };
    let mut out = Vec::new();
    let mut rest = search_in;
    while let Some(start) = rest.find("<read src=") {
        let r = &rest[start..];
        let Some(gt) = r.find('>') else { break };
        let path = extract_attr(&r[..gt + 1], "src").unwrap_or_else(|| "unknown".into());
        out.push(StreamToolView {
            kind: ToolPanelKind::Read,
            target: expand_path_display(&path),
            body: String::new(),
            tag_closed: true,
        });
        rest = &r[gt + 1..];
    }
    out
}

/// Pick one write for the live chip: last unclosed write, else the last closed write.
fn detect_primary_write(text: &str) -> Option<StreamToolView> {
    let writes = detect_all_writes(text);
    if writes.is_empty() {
        return None;
    }
    writes
        .iter()
        .rev()
        .find(|w| !w.tag_closed)
        .cloned()
        .or_else(|| writes.last().cloned())
}

/// All `<write>` tags in order (for pending-accept multi-file).
pub fn detect_all_writes(text: &str) -> Vec<StreamToolView> {
    let outside = crate::agent::AgentEngine::strip_think_blocks(text);
    let search_in = if outside.contains("<write") {
        outside.as_str()
    } else {
        text
    };
    let mut out = Vec::new();
    let mut rest = search_in;
    while let Some(start) = rest.find("<write src=") {
        let r = &rest[start..];
        let Some(gt) = r.find('>') else { break };
        let path_raw = extract_attr(&r[..gt + 1], "src").unwrap_or_else(|| "unknown".into());
        let after = &r[gt + 1..];
        if let Some(end) = after.find("</write") {
            let body = after[..end]
                .trim_matches(|c| c == '\n' || c == '\r')
                .to_string();
            // Closed: safe to normalize path from full body once.
            let path = crate::agent::AgentEngine::normalize_write_path(&path_raw, &body);
            out.push(StreamToolView {
                kind: ToolPanelKind::Write,
                target: expand_path_display(&path),
                body,
                tag_closed: true,
            });
            // Advance past this write
            if let Some(close_gt) = after[end..].find('>') {
                rest = &after[end + close_gt + 1..];
            } else {
                break;
            }
        } else {
            // Streaming: keep model path stable — do NOT re-infer from partial body
            // (that produced file.txt → index.html → title_slug.html chips).
            let body = after.to_string();
            let path = if path_raw.contains('.') {
                path_raw
            } else {
                // Directory-only src while streaming — soft default without body sniffing
                format!("{}/index.html", path_raw.trim_end_matches('/'))
            };
            out.push(StreamToolView {
                kind: ToolPanelKind::Write,
                target: expand_path_display(&path),
                body,
                tag_closed: false,
            });
            break; // rest is incomplete tail of this write
        }
    }
    out
}

fn detect_cmd(text: &str) -> Option<StreamToolView> {
    // Prefer tools outside think (Ollama R1 dumps prose into <cmd> inside think)
    let outside = crate::agent::AgentEngine::strip_think_blocks(text);
    let search_in = if outside.contains("<cmd>") {
        outside.as_str()
    } else {
        text
    };
    let start = search_in.find("<cmd>")?;
    let after = &search_in[start + 5..];
    if let Some(end) = after.find("</cmd>") {
        let cmd = clean_cmd(&after[..end]);
        if !crate::agent::AgentEngine::looks_like_shell_cmd(&cmd) {
            return None;
        }
        Some(StreamToolView {
            kind: ToolPanelKind::Cmd,
            target: cmd,
            body: String::new(),
            tag_closed: true,
        })
    } else {
        let mut cmd = after.lines().next().unwrap_or("").to_string();
        if let Some(i) = cmd.find('<') {
            cmd = cmd[..i].to_string();
        }
        let cmd = clean_cmd(&cmd);
        if !crate::agent::AgentEngine::looks_like_shell_cmd(&cmd) {
            return None;
        }
        Some(StreamToolView {
            kind: ToolPanelKind::Cmd,
            target: cmd,
            body: String::new(),
            tag_closed: false,
        })
    }
}

fn expand_path_display(path: &str) -> String {
    crate::agent::AgentEngine::expand_path(path)
        .display()
        .to_string()
}

fn extract_attr(tag: &str, name: &str) -> Option<String> {
    for q in ['"', '\''] {
        let key = format!("{name}={q}");
        if let Some(i) = tag.find(&key) {
            let rest = &tag[i + key.len()..];
            if let Some(j) = rest.find(q) {
                return Some(rest[..j].to_string());
            }
        }
    }
    None
}

pub fn redact_tools_for_chat(content: &str) -> String {
    let mut s = content.to_string();
    while let Some(start) = s.find("<write src=") {
        if let Some(rel_end) = s[start..].find("</write>") {
            s.replace_range(start..start + rel_end + 8, "");
        } else if s[start..].find('>').is_some() {
            s = s[..start].to_string();
            break;
        } else {
            break;
        }
    }
    while let Some(start) = s.find("<cmd>") {
        if let Some(rel_end) = s[start..].find("</cmd>") {
            s.replace_range(start..start + rel_end + 6, "");
        } else {
            s = s[..start].to_string();
            break;
        }
    }
    // Keep short read tags visible in chat (they're the whole answer often)
    s.trim().to_string()
}

/// Classify tool activity in a model reply for UI labels / chips.
pub fn classify_tool_hint(stream: &str) -> &'static str {
    let t = stream;
    if t.contains("<cmd>") {
        "command"
    } else if t.contains("<write src=") {
        "write"
    } else if t.contains("<read src=") {
        "read"
    } else if t.contains("<ls path=") || t.contains("<ls>") {
        "list"
    } else if t.contains("<memory") {
        "memory"
    } else {
        "tool"
    }
}

pub fn format_tool_output_for_chat(raw: &str) -> String {
    let mut s = raw.replace("\r\n", "\n").replace('\r', "\n");
    for needle in [
        "warning:",
        "error:",
        "note:",
        "Finished ",
        "Checking ",
        "Compiling ",
    ] {
        let mut out = String::new();
        let mut rest = s.as_str();
        while let Some(i) = rest.find(needle) {
            let before = &rest[..i];
            out.push_str(before);
            if !before.ends_with('\n') && !before.is_empty() {
                out.push('\n');
            }
            out.push_str(needle);
            rest = &rest[i + needle.len()..];
        }
        out.push_str(rest);
        s = out;
    }
    s
}

// ---------------------------------------------------------------------------
// Draw with KramaFrame progress t (0..=1)
// ---------------------------------------------------------------------------

/// `t` is open amount 0..=1. Caller must pass `get_progress_f32(...).abs()` —
/// Krama reverse stores negative progress; without abs reverse snaps to closed.
pub fn draw_tool_panel(
    frame: &mut Frame,
    panel: &mut ToolPanel,
    t: f32,
    _theme: Color,
) -> Option<Rect> {
    let chip = panel.chip_rect?;
    let dock = panel.dock_rect?;
    let t = ease_out_cubic(t.clamp(0.0, 1.0));

    let x = lerp(chip.x as f32, dock.x as f32, t);
    let y = lerp(chip.y as f32, dock.y as f32, t);
    let w = lerp(chip.width as f32, dock.width as f32, t);
    let h = lerp(chip.height as f32, dock.height as f32, t);

    let max_w = frame.area().width;
    let max_h = frame.area().height;

    let rx = (x.round().max(0.0) as u16).min(max_w);
    let ry = (y.round().max(0.0) as u16).min(max_h);
    let rw = (w.round().max(4.0) as u16).min(max_w.saturating_sub(rx));
    let rh = (h.round().max(3.0) as u16).min(max_h.saturating_sub(ry));

    let rect = Rect {
        x: rx,
        y: ry,
        width: rw,
        height: rh,
    };

    // Clear previous footprint + union so reverse leave no ghost borders
    if let Some(prev) = panel.drawn_rect {
        if prev != rect {
            let ux = prev.x.min(rect.x).min(max_w);
            let uy = prev.y.min(rect.y).min(max_h);
            let ur = (prev.x + prev.width).max(rect.x + rect.width).min(max_w);
            let ub = (prev.y + prev.height).max(rect.y + rect.height).min(max_h);
            frame.render_widget(
                Clear,
                Rect {
                    x: ux,
                    y: uy,
                    width: ur.saturating_sub(ux),
                    height: ub.saturating_sub(uy),
                },
            );
        }
    }
    frame.render_widget(Clear, rect);

    let accent = if panel.interactive && panel.kind == ToolPanelKind::Cmd {
        Color::Rgb(80, 220, 255) // cyan when TERM interactive
    } else {
        panel.kind.accent()
    };
    let is_term = panel.kind == ToolPanelKind::Cmd;

    // Left title + right chrome so [-]/[x] hit-test matches paint
    let mode = if panel.interactive && is_term {
        " LIVE"
    } else {
        ""
    };
    let left_title = format!(
        " {}{} {} ",
        panel.kind.title_prefix(),
        mode,
        trunc(&panel.target, (rect.width as usize).saturating_sub(20))
    );
    let chrome = if panel.minimized { "[+][x]" } else { "[-][x]" };

    if rect.width >= 10 {
        let close_w = 3u16;
        let min_w = 3u16;
        let close_x = rect.x + rect.width.saturating_sub(1 + close_w);
        let min_x = close_x.saturating_sub(1 + min_w);
        panel.close_hit = Some(Rect {
            x: close_x,
            y: rect.y,
            width: close_w + 1,
            height: 1,
        });
        panel.min_hit = Some(Rect {
            x: min_x,
            y: rect.y,
            width: min_w + 1,
            height: 1,
        });
    } else {
        panel.close_hit = None;
        panel.min_hit = None;
    }

    let block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(accent))
        .title(Span::styled(
            left_title,
            Style::default().fg(accent).add_modifier(Modifier::BOLD),
        ))
        .title(
            Line::from(Span::styled(
                format!(" {chrome} "),
                Style::default()
                    .fg(Color::Rgb(220, 220, 220))
                    .add_modifier(Modifier::BOLD),
            ))
            .right_aligned(),
        )
        .style(if is_term {
            Style::default().bg(Color::Rgb(10, 12, 14))
        } else {
            Style::default()
        });

    // Nearly closed / minimized: morph border only
    if t < 0.12 || rect.height <= 3 || panel.minimized {
        frame.render_widget(block, rect);
        panel.drawn_rect = Some(rect);
        return Some(rect);
    }

    let inner = block.inner(rect);
    frame.render_widget(block, rect);

    let now = Instant::now();
    let vis = panel.visible_body();
    let mut lines: Vec<Line> = Vec::new();

    if is_term {
        // tmux-like terminal header
        let head = if panel.interactive {
            format!(" $ {}  [INTERACTIVE — click outside to leave] ", panel.target)
        } else {
            format!(" $ {}  [click to interact] ", panel.target)
        };
        lines.push(Line::from(Span::styled(
            head,
            Style::default()
                .fg(if panel.interactive {
                    Color::Rgb(80, 255, 255)
                } else {
                    Color::Rgb(100, 220, 100)
                })
                .bg(Color::Rgb(20, 24, 28))
                .add_modifier(Modifier::BOLD),
        )));
        lines.push(Line::from(Span::styled(
            "─".repeat(inner.width as usize),
            Style::default().fg(Color::Rgb(40, 50, 40)),
        )));
        for raw in vis.lines() {
            lines.push(Line::from(Span::styled(
                raw.to_string(),
                Style::default().fg(Color::Rgb(180, 255, 180)),
            )));
        }
        if !panel.tag_closed || panel.revealed_chars < panel.body.chars().count() {
            lines.push(Line::from(Span::styled(
                "█",
                Style::default().fg(Color::Rgb(100, 255, 100)),
            )));
        }
    } else {
        lines.push(Line::from(Span::styled(
            format!(
                "> {}  [scroll: wheel/PgUp/PgDn]",
                trunc(&panel.target, (inner.width as usize).saturating_sub(28))
            ),
            Style::default()
                .fg(Color::Rgb(120, 120, 120))
                .add_modifier(Modifier::ITALIC),
        )));
        let mut char_i = 0usize;
        for raw in vis.split_inclusive('\n') {
            let mut spans = Vec::new();
            
            let mut line_fg = None;
            if panel.kind == ToolPanelKind::Write {
                if raw.starts_with('+') {
                    line_fg = Some(Color::Green);
                } else if raw.starts_with('-') {
                    line_fg = Some(Color::Red);
                }
            }

            for ch in raw.chars() {
                if ch == '\n' {
                    char_i += 1;
                    continue;
                }
                
                let mut fg = panel.char_color(char_i, now);
                // Override base color with diff color, but keep fade logic if it's very dim?
                // Actually if line_fg is present, just use it directly, but maybe multiply by fade?
                // Ratatui doesn't easily multiply colors, so just use line_fg if it's revealed, else use char_color (which handles the fade).
                // If char_color is near white/gray (revealed), we override it.
                // Since char_color returns the exact color, we can check if it's the bright end.
                // But it's easier to just use line_fg and ignore the fade for diff lines, or just use it always.
                // Let's just use line_fg directly.
                if let Some(c) = line_fg {
                    fg = c;
                }
                
                spans.push(Span::styled(
                    ch.to_string(),
                    Style::default().fg(fg),
                ));
                char_i += 1;
            }
            lines.push(if spans.is_empty() {
                Line::from("")
            } else {
                Line::from(spans)
            });
        }
        if panel.revealed_chars < panel.body.chars().count() {
            lines.push(Line::from(Span::styled(
                "▍",
                Style::default().fg(accent),
            )));
        }
    }

    // Scroll budget from content height vs viewport
    let content_lines = lines.len() as u16;
    let view_h = inner.height.max(1);
    panel.max_scroll = content_lines.saturating_sub(view_h);
    // Writing cursor autoscroll: pin to bottom while streaming / follow_end
    if panel.follow_end {
        panel.scroll = panel.max_scroll;
    } else if panel.scroll > panel.max_scroll {
        panel.scroll = panel.max_scroll;
    }

    // Scroll hint on right of title when content overflows
    if panel.max_scroll > 0 {
        let pct = if panel.max_scroll == 0 {
            100
        } else {
            (panel.scroll as u32 * 100 / panel.max_scroll as u32).min(100)
        };
        frame.render_widget(
            Paragraph::new(Span::styled(
                format!(" {}/{} ", panel.scroll, panel.max_scroll),
                Style::default().fg(Color::Rgb(140, 140, 160)),
            )),
            Rect {
                x: rect.x.saturating_add(rect.width.saturating_sub(12)),
                y: rect.y.saturating_add(rect.height.saturating_sub(1)),
                width: 10,
                height: 1,
            },
        );
        let _ = pct;
    }

    frame.render_widget(
        Paragraph::new(lines)
            
            .scroll((panel.scroll, 0))
            .style(if is_term {
                Style::default().bg(Color::Rgb(10, 12, 14))
            } else {
                Style::default()
            }),
        inner,
    );
    panel.drawn_rect = Some(rect);
    Some(rect)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PanelChromeHit {
    None,
    Minimize,
    Close,
}

fn point_in(r: Rect, col: u16, row: u16) -> bool {
    col >= r.x && col < r.x + r.width && row >= r.y && row < r.y + r.height
}

/// Prefer live hit rects painted last frame; fall back to right-edge heuristic.
pub fn hit_test_chrome(panel: &ToolPanel, col: u16, row: u16) -> PanelChromeHit {
    if let Some(r) = panel.close_hit {
        if point_in(r, col, row) {
            return PanelChromeHit::Close;
        }
    }
    if let Some(r) = panel.min_hit {
        if point_in(r, col, row) {
            return PanelChromeHit::Minimize;
        }
    }
    // Fallback: top-right of drawn panel (title row)
    let Some(panel_rect) = panel.drawn_rect else {
        return PanelChromeHit::None;
    };
    if row != panel_rect.y || panel_rect.width < 8 {
        return PanelChromeHit::None;
    }
    // Rightmost cells: " [x]" then "[-]"
    let right = panel_rect.x + panel_rect.width;
    if col + 1 >= right.saturating_sub(4) && col < right {
        return PanelChromeHit::Close;
    }
    if col + 1 >= right.saturating_sub(8) && col < right.saturating_sub(4) {
        return PanelChromeHit::Minimize;
    }
    PanelChromeHit::None
}

pub fn hit_test_chip(chips: &[ToolChip], col: u16, row: u16) -> Option<u64> {
    for c in chips.iter().rev() {
        if let Some(r) = c.rect {
            if point_in(r, col, row) {
                return Some(c.id);
            }
        }
    }
    None
}
