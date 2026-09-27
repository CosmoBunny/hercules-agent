//! Tool chips (size-to-fit, under agent) + KramaFrame-driven fly panel.
//!
//! Animation uses KramaFrame progress 0→1 (open) / reverse (close), like the
//! official TUI example: update_progress each frame, get_progress_f32 for t.
//! Geometry: lerp(chip_rect, dock_rect, ease(t)).

use ratatui::{
    Frame,
    layout::Rect,
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, BorderType, Borders, Clear, Paragraph},
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
    pub anchor_msg: Option<usize>,
    pub expanded: bool,
    pub anim_start: Option<Instant>,
    /// Write diff stats once executed (`WROTE file +a -r`).
    pub added: Option<usize>,
    pub removed: Option<usize>,
    /// Read `line="45-55"` scope (`READ file [45,55]`).
    pub line_range: Option<String>,
}

/// Terminal rows reserved under an agent turn for one chip (bordered 3-row button).
pub const CHIP_ROW_HEIGHT: u16 = 3;

impl ToolChip {
    pub fn label_text(&self) -> String {
        self.label_text_with_duration(None)
    }

    /// Kind verb matching the label (`WRITE`/`WROTE`, `RUN`/`RAN`,
    /// `READ`, …) — used by the label and the transcript badge alike so
    /// the two can never disagree.
    pub fn badge_verb(&self) -> &'static str {
        match self.kind {
            ToolPanelKind::Write => {
                if self.pending || !self.tag_closed {
                    "WRITE"
                } else {
                    "WROTE"
                }
            }
            ToolPanelKind::Cmd => {
                if self.pending {
                    "RUN"
                } else if self.tag_closed && !self.body.is_empty() {
                    "RAN"
                } else {
                    "RUN"
                }
            }
            ToolPanelKind::Read => "READ",
            ToolPanelKind::Agent => "AGENT",
            ToolPanelKind::WebSearch => "SEARCH",
            ToolPanelKind::Skill => "SKILL",
            ToolPanelKind::Mcp => "MCP",
        }
    }

    /// Amber badge text above the chip: ` READ `, ` WROTE `,
    /// ` RUN 45s ` — the kind, never the generic "Action".
    pub fn badge_text(&self, dur_secs: Option<u64>) -> String {
        match dur_secs {
            Some(d) => format!(" {} {d}s ", self.badge_verb()),
            None => format!(" {} ", self.badge_verb()),
        }
    }

    /// Plain-text label (tooltips, headers, width). Single format
    /// definition lives in [`Self::label_spans`]; this just concatenates.
    pub fn label_text_with_duration(&self, duration: Option<&str>) -> String {
        self.label_spans(Color::Rgb(0, 0, 0), duration)
            .iter()
            .map(|s| s.content.as_ref())
            .collect::<Vec<_>>()
            .concat()
    }

    /// Compact kind-first label with per-part colors:
    /// `WROTE file +45 -3` (+green/-red), `READ file [45,55]`,
    /// `RAN cmd 34s` / `RUN cmd 45s`. `bg` is painted on every span so
    /// callers can drop the result straight into any row.
    pub fn label_spans(&self, bg: Color, duration: Option<&str>) -> Vec<Span<'static>> {
        const GREEN: Color = Color::Rgb(120, 220, 140);
        const RED: Color = Color::Rgb(255, 120, 120);
        const DIM: Color = Color::Rgb(140, 150, 165);
        let accent = self.kind.accent();
        let verb = |t: String| {
            Span::styled(
                t,
                Style::default()
                    .fg(accent)
                    .bg(bg)
                    .add_modifier(Modifier::BOLD),
            )
        };
        let plain = |t: String| Span::styled(t, Style::default().fg(Color::White).bg(bg));
        let dim = |t: String| Span::styled(t, Style::default().fg(DIM).bg(bg));
        let mut out: Vec<Span<'static>> = Vec::new();
        let mut dur = |out: &mut Vec<Span<'static>>| {
            if let Some(d) = duration {
                out.push(dim(format!(" {d}")));
            }
        };
        match self.kind {
            ToolPanelKind::Write => {
                let short = self
                    .target
                    .rsplit('/')
                    .next()
                    .unwrap_or(&self.target)
                    .to_string();
                if self.pending {
                    out.push(verb(format!("{} ", self.badge_verb())));
                    out.push(plain(short));
                    out.push(dim(" (pending)".into()));
                } else if !self.tag_closed {
                    out.push(verb(format!("{} ", self.badge_verb())));
                    out.push(plain(short));
                    out.push(dim("…".into()));
                } else {
                    out.push(verb(format!("{} ", self.badge_verb())));
                    out.push(plain(short));
                    match (self.added, self.removed) {
                        (Some(a), Some(r)) if a + r > 0 => {
                            out.push(Span::styled(
                                format!(" +{a}"),
                                Style::default()
                                    .fg(GREEN)
                                    .bg(bg)
                                    .add_modifier(Modifier::BOLD),
                            ));
                            out.push(Span::styled(
                                format!(" -{r}"),
                                Style::default().fg(RED).bg(bg).add_modifier(Modifier::BOLD),
                            ));
                        }
                        (Some(_), Some(_)) => {
                            out.push(dim(" (no changes)".into()));
                        }
                        _ => {
                            let lines = line_count(&self.body);
                            out.push(Span::styled(
                                format!(" +{lines}"),
                                Style::default()
                                    .fg(GREEN)
                                    .bg(bg)
                                    .add_modifier(Modifier::BOLD),
                            ));
                        }
                    }
                }
            }
            ToolPanelKind::Cmd => {
                let cmd = trunc(&clean_cmd(&self.target), 42);
                let done = self.tag_closed && !self.body.is_empty();
                if self.pending {
                    out.push(verb(format!("{} ", self.badge_verb())));
                    out.push(plain(cmd));
                    out.push(dim(" (pending)".into()));
                } else if done {
                    out.push(verb(format!("{} ", self.badge_verb())));
                    out.push(plain(cmd));
                    dur(&mut out);
                } else {
                    out.push(verb(format!("{} ", self.badge_verb())));
                    out.push(plain(cmd));
                    if duration.is_some() {
                        dur(&mut out);
                    } else {
                        out.push(dim("…".into()));
                    }
                }
            }
            ToolPanelKind::Read => {
                let short = trunc(self.target.rsplit('/').next().unwrap_or(&self.target), 36);
                out.push(verb(format!("{} ", self.badge_verb())));
                out.push(plain(short));
                if let Some(ref r) = self.line_range {
                    out.push(Span::styled(
                        format!(" [{}]", fmt_line_range(r)),
                        Style::default()
                            .fg(accent)
                            .bg(bg)
                            .add_modifier(Modifier::BOLD),
                    ));
                } else if !self.tag_closed {
                    out.push(dim("…".into()));
                }
                dur(&mut out);
            }
            ToolPanelKind::Agent => {
                out.push(verb(format!("{} ", self.badge_verb())));
                out.push(plain(self.target.clone()));
                dur(&mut out);
            }
            ToolPanelKind::WebSearch => {
                let q = trunc(&self.target, 50);
                out.push(verb(format!("{} ", self.badge_verb())));
                out.push(plain(q));
                if !self.tag_closed {
                    out.push(dim("…".into()));
                }
                dur(&mut out);
            }
            ToolPanelKind::Skill => {
                out.push(verb(format!("{} ", self.badge_verb())));
                out.push(plain(self.target.clone()));
                dur(&mut out);
            }
            ToolPanelKind::Mcp => {
                out.push(verb(format!("{} ", self.badge_verb())));
                out.push(plain(self.target.clone()));
                dur(&mut out);
            }
        }
        out
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
        let bg_color = Color::Rgb(36, 41, 51); // Nordic Dark BG #242933
        frame.render_widget(Clear, area);
        frame.render_widget(Block::default().style(Style::default().bg(bg_color)), area);
        let block = Block::default()
            .style(Style::default().bg(bg_color))
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .border_style(Style::default().fg(accent));
        let block = crate::app_chrome::frame_container(block);
        let inner = block.inner(area);
        frame.render_widget(block, area);
        frame.render_widget(
            Paragraph::new(Line::from(self.label_spans(bg_color, None)))
                .style(Style::default().bg(bg_color)),
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
        body.trim_start_matches(|c| c == '\n' || c == '\r')
            .lines()
            .count()
    }
}

/// Display form of a `<read line="…">` scope: `45-55` → `45,55`,
/// `55` → `55`.
pub fn fmt_line_range(raw: &str) -> String {
    let r = raw.trim();
    if let Some((a, b)) = r.split_once('-') {
        format!("{},{}", a.trim(), b.trim())
    } else {
        r.to_string()
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
        ToolPanelKind::Write
        | ToolPanelKind::Read
        | ToolPanelKind::Mcp
        | ToolPanelKind::Skill
        | ToolPanelKind::WebSearch
        | ToolPanelKind::Agent => t,
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
        format!(
            "{}...",
            s.chars().take(max.saturating_sub(1)).collect::<String>()
        )
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
    /// Raw `<read line="…">` scope, if present.
    pub line_range: Option<String>,
}

fn detect_mcp(text: &str) -> Vec<StreamToolView> {
    let mut out = Vec::new();
    let mut rest = text;
    while let Some(start) = rest.find("<mcp ") {
        let r = &rest[start..];
        if let Some(close_bracket) = r.find('>') {
            let header = &r[..close_bracket + 1];
            let action = crate::agent::AgentEngine::extract_attribute(header, "action")
                .unwrap_or_else(|| "search".to_string());
            let end = r.find("</mcp>").unwrap_or(r.len());
            let body = r[close_bracket + 1..end].trim().to_string();
            out.push(StreamToolView {
                kind: ToolPanelKind::Mcp,
                target: action,
                body,
                tag_closed: r.find("</mcp>").is_some(),
                line_range: None,
            });
            rest = if r.find("</mcp>").is_some() {
                &r[end + 6..]
            } else {
                ""
            };
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
            let action = crate::agent::AgentEngine::extract_attribute(header, "action")
                .unwrap_or_else(|| "search".to_string());
            let end = r.find("</skill>").unwrap_or(r.len());
            let body = r[close_bracket + 1..end].trim().to_string();
            out.push(StreamToolView {
                kind: ToolPanelKind::Skill,
                target: action,
                body,
                tag_closed: r.find("</skill>").is_some(),
                line_range: None,
            });
            rest = if r.find("</skill>").is_some() {
                &r[end + 8..]
            } else {
                ""
            };
        } else {
            break;
        }
    }
    out
}

fn detect_websearch(text: &str) -> Vec<StreamToolView> {
    let mut out = Vec::new();
    let mut rest = text;
    while let Some(start) = crate::agent::AgentEngine::find_tag_open(rest, "<websearch") {
        let r = &rest[start..];
        if let Some(close_bracket) = r.find('>') {
            let header = &r[..close_bracket + 1];
            let mut action = crate::agent::AgentEngine::extract_attribute(header, "query")
                .unwrap_or_else(|| "search".to_string());
            let after = &r[close_bracket + 1..];

            // </websearch> is optional — find explicit end tag or next tool opening tag or line boundary
            let mut end_pos = after.find("</websearch>");
            let is_explicit_closed = end_pos.is_some();

            if end_pos.is_none() {
                // If query="..." attribute was provided, websearch needs no body at all
                if action != "search" {
                    end_pos = Some(0);
                } else {
                    // Check for next tool tag opening (<write, <cmd, <read, <ls, <agent, <mcp, <skill)
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
            let mut body = after[..end].trim().to_string();

            for stop in [
                "<|im_end|>",
                "<|im_start|>",
                "<|eot_id|>",
                "<|endoftext|>",
                "</s>",
            ] {
                action = action.replace(stop, "").trim().to_string();
                body = body.replace(stop, "").trim().to_string();
            }

            if action == "search" && !body.is_empty() {
                action = body.clone();
            }

            let advance = if is_explicit_closed { end + 12 } else { end };

            out.push(StreamToolView {
                kind: ToolPanelKind::WebSearch,
                target: action,
                body,
                tag_closed: true,
                line_range: None, // Optional close: treat self-contained query as closed
            });
            rest = &after[advance.min(after.len())..];
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
            let action = crate::agent::AgentEngine::extract_attribute(header, "action")
                .unwrap_or_else(|| "spawn".to_string());
            let role =
                crate::agent::AgentEngine::extract_attribute(header, "role").unwrap_or_default();
            let to = crate::agent::AgentEngine::extract_attribute(header, "to").unwrap_or_default();
            let model =
                crate::agent::AgentEngine::extract_attribute(header, "model").unwrap_or_default();

            let mut target_label = action.clone();
            if !role.is_empty() {
                target_label.push_str(&format!(" role={role}"));
            }
            if !model.is_empty() {
                target_label.push_str(&format!(" model={model}"));
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
                line_range: None,
            });
            rest = if r.find("</agent>").is_some() {
                &r[end + 8..]
            } else {
                ""
            };
        } else {
            break;
        }
    }
    out
}

/// Preview-only scan of a model stream for UI chips (open/closed tags,
/// path-stability heuristics). This module NEVER executes anything:
/// execution flows exclusively through
/// `AgentEngine::parse_tool_calls` + `App::claim_tool_call` +
/// `AgentEngine::execute_proposed`, which enforces the Ask-mode
/// permission gate and the filesystem sandbox. If this preview and the
/// canonical parser disagree on a malformed/chunked construct, the
/// executor's view wins — a chip is display only, never authority.
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
    // Thinking zones are NEVER executable: chips must only reflect validated
    // tool calls from outside <think>, never model reasoning text.
    crate::agent::AgentEngine::strip_code_fences(&crate::agent::AgentEngine::strip_think_blocks(
        response,
    ))
}

fn detect_ls(text: &str) -> Vec<StreamToolView> {
    // Outside <think> only — a mention inside thinking is not an executed action.
    let outside = crate::agent::AgentEngine::strip_think_blocks(text);
    let mut out = Vec::new();
    let mut rest = outside.as_str();
    while let Some(start) = crate::agent::AgentEngine::find_tag_open(rest, "<ls") {
        let r = &rest[start..];
        let Some(gt) = r.find('>') else { break };
        let path = extract_attr(&r[..gt + 1], "path").unwrap_or_else(|| "$CURRENT".into());
        out.push(StreamToolView {
            kind: ToolPanelKind::Read,
            target: expand_path_display(&path),
            body: String::new(),
            tag_closed: true,
            line_range: None,
        });
        rest = &r[gt + 1..];
    }
    out
}

fn detect_reads(text: &str) -> Vec<StreamToolView> {
    // Outside <think> only — a mention inside thinking is not an executed action.
    let outside = crate::agent::AgentEngine::strip_think_blocks(text);
    let mut out = Vec::new();
    let mut rest = outside.as_str();
    while let Some(start) = rest.find("<read src=") {
        let r = &rest[start..];
        let Some(gt) = r.find('>') else { break };
        let path = extract_attr(&r[..gt + 1], "src").unwrap_or_else(|| "unknown".into());
        let line_range = extract_attr(&r[..gt + 1], "line");
        out.push(StreamToolView {
            kind: ToolPanelKind::Read,
            target: expand_path_display(&path),
            body: String::new(),
            tag_closed: true,
            line_range,
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
    // Outside <think> only — a mention inside thinking is not an executed action.
    let outside = crate::agent::AgentEngine::strip_think_blocks(text);
    let mut out = Vec::new();
    let mut rest = outside.as_str();
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
                line_range: None,
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
                line_range: None,
            });
            break; // rest is incomplete tail of this write
        }
    }
    out
}

fn detect_cmd(text: &str) -> Option<StreamToolView> {
    // Outside <think> only — prose inside thinking is never an executed command.
    let outside = crate::agent::AgentEngine::strip_think_blocks(text);
    let start = outside.find("<cmd>")?;
    let after = &outside[start + 5..];
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
            line_range: None,
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
            line_range: None,
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

/// True when `c` can directly follow a tool-tag name (`<write␣`,
/// `<cmd>`, `<ls/>`, `<read\n` …). Anything else (`<ready>`, `<lsp>`,
/// `<writer>`) is ordinary prose, never a tool.
fn is_tag_delimiter(c: char) -> bool {
    matches!(c, ' ' | '\t' | '\n' | '\r' | '>' | '/')
}

/// Byte length of the tool construct starting at `tag` (which must begin
/// with `<`), or `None` when the text at this position is NOT a
/// structurally valid tool construct. Only allow-listed constructs with
/// well-formed headers — and, for block kinds, a matching closer — are
/// recognized. Everything else (unknown tags, stray closers like
/// `</anything>`, malformed headers, prose prefix-collisions like
/// `<ready>` or `<lsp>`, unterminated trailing fragments) returns `None`
/// and is preserved byte-for-byte by the caller.
fn match_tool_construct_len(tag: &str) -> Option<usize> {
    debug_assert!(tag.starts_with('<'));
    // Stray closing tags are always ordinary text.
    if tag.starts_with("</") {
        return None;
    }
    // Exact single-token constructs.
    if tag.starts_with("<cmd>") {
        let after = &tag["<cmd>".len()..];
        let end = after.find("</cmd>")?;
        return Some("<cmd>".len() + end + "</cmd>".len());
    }
    // `<name` + delimiter openers.
    const BLOCKS: &[(&str, &str)] = &[
        ("<write", "</write"),
        ("<mcp", "</mcp>"),
        ("<skill", "</skill>"),
        ("<websearch", "</websearch>"),
        ("<agent", "</agent>"),
        ("<memory", "</memory>"),
        ("<read", ""),
        ("<ls", ""),
    ];
    for (open, close) in BLOCKS {
        if !tag.starts_with(open) {
            continue;
        }
        let after_open = &tag[open.len()..];
        let mut chars = after_open.chars();
        match chars.next() {
            Some(c) if is_tag_delimiter(c) => {}
            _ => continue, // `<ready>`, `<lsp>`, `<writer>` … prose, keep scanning
        }
        let hdr_end = tag.find('>')?;
        let header = &tag[..hdr_end + 1];
        if *open == "<read" && !header.contains("src=") {
            continue; // bare `<read>` never executes — keep visible
        }
        if *open == "<write" && !header.contains("src=") {
            continue; // malformed write (no target) never executes
        }
        if close.is_empty() {
            return Some(hdr_end + 1); // self-contained `<read …>` / `<ls …>`
        }
        // Block kinds need their matching closer; `<memory>` bare tags
        // (no push/replace) are valid singletons like the parser runs.
        if *open == "<memory" && !header.contains("push") && !header.contains("replace=") {
            return Some(hdr_end + 1);
        }
        let after = &tag[hdr_end + 1..];
        let end = after.find(close)?;
        let tail = &after[end + close.len()..];
        // Closer must terminate (`>` or end); a longer name (`</writex>`)
        // is not our closer.
        match tail.chars().next() {
            Some('>') => return Some(hdr_end + 1 + end + close.len() + 1),
            None => return Some(tag.len()),
            _ => continue,
        }
    }
    None
}

pub fn redact_tools_for_chat(content: &str) -> String {
    // Structural allow-list redaction for transcript DISPLAY (chips carry
    // the executed tool bodies). Only spans that parse as complete, valid
    // tool constructs are removed, plus a trailing UNCLOSED valid write /
    // cmd whose body is already live in its chip (otherwise the partial
    // tag duplicates the chip while streaming). Everything else — prose,
    // unknown tags, stray closers, malformed headers, unterminated
    // fragments of anything else — passes through untouched. In
    // particular there is no strip-to-end fallback for unrecognized text.
    let mut s = String::with_capacity(content.len());
    let mut rest = content;
    while let Some(lt) = rest.find('<') {
        s.push_str(&rest[..lt]);
        let tag = &rest[lt..];
        match match_tool_construct_len(tag) {
            Some(len) => {
                rest = &tag[len.min(tag.len())..];
            }
            None if trailing_live_construct(tag) => {
                // In-progress write/cmd already shown in its chip.
                break;
            }
            None => {
                s.push('<');
                rest = &tag[1..];
            }
        }
    }
    if rest.find('<').is_none() {
        s.push_str(rest);
    }
    s.trim().to_string()
}

/// True when `tag` (starting at `<`) opens a VALID tool construct whose
/// body is already live in a chip but whose closer hasn't arrived: a
/// sourced `<write …>` header, or an exact `<cmd>`. Malformed headers
/// (`<write>` without src) and every other kind stay visible.
fn trailing_live_construct(tag: &str) -> bool {
    if tag.starts_with("<cmd>") {
        return tag["<cmd>".len()..].find("</cmd>").is_none();
    }
    const PREFIXES: &[&str] = &["<write"];
    for open in PREFIXES {
        if !tag.starts_with(open) {
            continue;
        }
        let after_open = &tag[open.len()..];
        match after_open.chars().next() {
            Some(c) if c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '>' || c == '/' => {}
            _ => return false,
        }
        let Some(hdr_end) = tag.find('>') else {
            return false;
        };
        if !tag[..hdr_end + 1].contains("src=") {
            return false;
        }
        if tag[hdr_end + 1..].find("</write").is_none() {
            return true;
        }
    }
    false
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
    let is_term = panel.kind == ToolPanelKind::Cmd;
    let bg_color = if is_term {
        Color::Rgb(20, 24, 30)
    } else {
        Color::Rgb(36, 41, 51) // Nordic Dark BG #242933
    };

    frame.render_widget(Clear, rect);
    frame.render_widget(Block::default().style(Style::default().bg(bg_color)), rect);

    let accent = if panel.interactive && is_term {
        Color::Rgb(80, 220, 255) // cyan when TERM interactive
    } else {
        panel.kind.accent()
    };

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
        .style(Style::default().bg(bg_color))
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(accent))
        .title(Span::styled(
            left_title,
            Style::default()
                .fg(accent)
                .bg(bg_color)
                .add_modifier(Modifier::BOLD),
        ))
        .title(
            Line::from(Span::styled(
                format!(" {chrome} "),
                Style::default()
                    .fg(Color::Rgb(220, 220, 220))
                    .bg(bg_color)
                    .add_modifier(Modifier::BOLD),
            ))
            .right_aligned(),
        );
    let block = crate::app_chrome::frame_container(block);

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
            format!(
                " $ {}  [INTERACTIVE — click outside to leave] ",
                panel.target
            )
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
            Style::default().fg(Color::Rgb(40, 50, 40)).bg(bg_color),
        )));
        for raw in vis.lines() {
            lines.push(Line::from(Span::styled(
                raw.to_string(),
                Style::default().fg(Color::Rgb(180, 255, 180)).bg(bg_color),
            )));
        }
        if !panel.tag_closed || panel.revealed_chars < panel.body.chars().count() {
            lines.push(Line::from(Span::styled(
                "█",
                Style::default().fg(Color::Rgb(100, 255, 100)).bg(bg_color),
            )));
        }
    } else {
        lines.push(Line::from(Span::styled(
            format!(
                "> {}  [scroll: wheel/PgUp/PgDn]",
                trunc(&panel.target, (inner.width as usize).saturating_sub(28))
            ),
            Style::default()
                .fg(Color::Rgb(140, 150, 165))
                .bg(bg_color)
                .add_modifier(Modifier::ITALIC),
        )));
        let mut char_i = 0usize;
        let mut line_num = 1usize;
        for raw in vis.split_inclusive('\n') {
            let mut spans = Vec::new();

            // Render line number gutter for code files
            if matches!(panel.kind, ToolPanelKind::Write | ToolPanelKind::Read) {
                let num_str = format!("{:>3} │ ", line_num);
                spans.push(Span::styled(
                    num_str,
                    Style::default().fg(Color::Rgb(80, 95, 115)).bg(bg_color),
                ));
                line_num += 1;
            }

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
                if let Some(c) = line_fg {
                    fg = c;
                }

                spans.push(Span::styled(
                    ch.to_string(),
                    Style::default().fg(fg).bg(bg_color),
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
                Style::default().fg(accent).bg(bg_color),
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
                Style::default().fg(Color::Rgb(140, 140, 160)).bg(bg_color),
            ))
            .style(Style::default().bg(bg_color)),
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
            .style(Style::default().bg(bg_color)),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prose_creates_no_chips() {
        // Natural-language intent is not an executed action: no chips.
        assert!(detect_all_stream_tools("I should search the web.").is_empty());
        assert!(detect_all_stream_tools("I will read foo.rs next.").is_empty());
        // Bracketed prose lookalikes are not tools either.
        assert!(detect_all_stream_tools("check <lsp> diagnostics").is_empty());
        assert!(detect_all_stream_tools("be <ready> when done").is_empty());
        assert!(detect_all_stream_tools("the <writer> writes").is_empty());
        assert!(detect_all_stream_tools("hello </anything> world").is_empty());
    }

    #[test]
    fn test_think_tools_create_no_chips() {
        // Tool syntax inside thinking is reasoning text, never UI actions.
        assert!(detect_all_stream_tools("<think>Let me <cmd>ls</cmd> first</think>").is_empty());
        assert!(detect_all_stream_tools("<think>Reading <read src=\"a.rs\"></think>").is_empty());
        assert!(
            detect_all_stream_tools("<think>Searching <websearch>rust</websearch></think>")
                .is_empty()
        );
        assert!(
            detect_all_stream_tools(
                "<think>Writing <write src=\"$CURRENT/a.txt\">x</write></think>"
            )
            .is_empty()
        );
    }

    #[test]
    fn test_outside_tools_create_chips() {
        let chips = detect_all_stream_tools("<think>hmm</think><ls path=\"$CURRENT\">");
        assert_eq!(chips.len(), 1);

        let chips =
            detect_all_stream_tools("<think>hmm</think><websearch>rust borrow checker</websearch>");
        assert_eq!(chips.len(), 1);
        assert_eq!(chips[0].kind, ToolPanelKind::WebSearch);
    }

    /// Malformed-tag matrix: the transcript redactor is a structural
    /// allow-list. Only complete, valid tool constructs are removed
    /// (chips carry them); every other byte survives verbatim. In
    /// particular there is no strip-to-end fallback and no
    /// prefix-collision (`<ready>`, `<lsp>`, `<writer>` are prose).
    #[test]
    fn test_redact_preserves_stray_closing_tags() {
        assert_eq!(
            redact_tools_for_chat("hello </anything> world"),
            "hello </anything> world"
        );
        assert_eq!(
            redact_tools_for_chat("hello </write> world"),
            "hello </write> world"
        );
        assert_eq!(
            redact_tools_for_chat("hello </cmd> world"),
            "hello </cmd> world"
        );
    }

    #[test]
    fn test_redact_preserves_unknown_tags() {
        assert_eq!(
            redact_tools_for_chat("hello <unknown> world"),
            "hello <unknown> world"
        );
        assert_eq!(
            redact_tools_for_chat("prose <unknown attr=\"x\"> more prose </unknown>"),
            "prose <unknown attr=\"x\"> more prose </unknown>"
        );
    }

    #[test]
    fn test_redact_preserves_prose_prefix_collisions() {
        assert_eq!(
            redact_tools_for_chat("be <ready> when done"),
            "be <ready> when done"
        );
        assert_eq!(
            redact_tools_for_chat("check <lsp> diagnostics output"),
            "check <lsp> diagnostics output"
        );
        assert_eq!(
            redact_tools_for_chat("the <writer> writes prose"),
            "the <writer> writes prose"
        );
        assert_eq!(
            redact_tools_for_chat("<div><p>literal html</p></div>"),
            "<div><p>literal html</p></div>"
        );
    }

    #[test]
    fn test_redact_preserves_malformed_and_partial_tools() {
        // No src: never executes, stays visible.
        assert_eq!(
            redact_tools_for_chat("hello <write> world"),
            "hello <write> world"
        );
        // Unterminated fragment: stays visible.
        assert_eq!(redact_tools_for_chat("hello <wri"), "hello <wri");
        // Trailing UNCLOSED valid write/cmd: hidden from the message
        // because the body is already live in the tool chip (no
        // duplication while streaming).
        assert_eq!(
            redact_tools_for_chat("prose <write src=\"a\"> body"),
            "prose"
        );
        assert_eq!(redact_tools_for_chat("run <cmd>ls"), "run");
    }

    #[test]
    fn test_redact_removes_only_valid_complete_tools() {
        // Valid block removed, stray closer + prose preserved.
        assert_eq!(
            redact_tools_for_chat("hello </write> middle <write src=\"a\">x</write> end"),
            "hello </write> middle  end"
        );
        assert_eq!(redact_tools_for_chat("run <cmd>ls</cmd> now"), "run  now");
        assert_eq!(
            redact_tools_for_chat("see <read src=\"f\"> and <ls> today"),
            "see  and  today"
        );
    }

    fn mk_chip() -> ToolChip {
        ToolChip {
            id: 1,
            kind: ToolPanelKind::Write,
            target: String::new(),
            body: String::new(),
            tag_closed: true,
            pending: false,
            spawned: false,
            rect: None,
            anchor_msg: None,
            expanded: false,
            anim_start: None,
            added: None,
            removed: None,
            line_range: None,
        }
    }

    #[test]
    fn test_chip_label_wrote_diff_stats() {
        // WROTE file +added -removed, plain-text form.
        let mut c = mk_chip();
        c.target = "/x/src/main.rs".into();
        c.added = Some(45);
        c.removed = Some(3);
        assert_eq!(c.label_text(), "WROTE main.rs +45 -3");
        // No-change write stays honest, never "+0 -0".
        c.added = Some(0);
        c.removed = Some(0);
        assert_eq!(c.label_text(), "WROTE main.rs (no changes)");
        // Stats unknown: fall back to body line count as +N.
        c.added = None;
        c.removed = None;
        c.body = "a\nb\nc".into();
        assert_eq!(c.label_text(), "WROTE main.rs +3");
        // Pending/streaming keep present tense.
        c.pending = true;
        assert_eq!(c.label_text(), "WRITE main.rs (pending)");
        c.pending = false;
        c.tag_closed = false;
        assert_eq!(c.label_text(), "WRITE main.rs…");
    }

    #[test]
    fn test_chip_label_diff_spans_carry_colors() {
        let mut c = mk_chip();
        c.target = "main.rs".into();
        c.added = Some(45);
        c.removed = Some(3);
        let spans = c.label_spans(Color::Rgb(0, 0, 0), None);
        let plus = spans
            .iter()
            .find(|s| s.content == " +45")
            .expect("+45 span");
        assert_eq!(plus.style.fg, Some(Color::Rgb(120, 220, 140)));
        let minus = spans.iter().find(|s| s.content == " -3").expect("-3 span");
        assert_eq!(minus.style.fg, Some(Color::Rgb(255, 120, 120)));
    }

    #[test]
    fn test_chip_label_read_ranges() {
        // [45,55] line range, [55] single line.
        let mut c = mk_chip();
        c.kind = ToolPanelKind::Read;
        c.target = "/x/src/lib.rs".into();
        c.line_range = Some("45-55".into());
        assert_eq!(c.label_text(), "READ lib.rs [45,55]");
        c.line_range = Some("55".into());
        assert_eq!(c.label_text(), "READ lib.rs [55]");
        // Full-file read: no range, no line-count noise.
        c.line_range = None;
        c.body = "a\nb".into();
        assert_eq!(c.label_text(), "READ lib.rs");
        assert_eq!(fmt_line_range("45-55"), "45,55");
        assert_eq!(fmt_line_range("55"), "55");
    }

    #[test]
    fn test_chip_label_cmd_durations() {
        // RAN 34s when done, RUN 45s while running, … with no clock.
        let mut c = mk_chip();
        c.kind = ToolPanelKind::Cmd;
        c.target = "cargo check".into();
        c.body = "ok".into();
        assert_eq!(
            c.label_text_with_duration(Some("34s")),
            "RAN cargo check 34s"
        );
        c.tag_closed = false;
        c.body.clear();
        assert_eq!(
            c.label_text_with_duration(Some("45s")),
            "RUN cargo check 45s"
        );
        assert_eq!(c.label_text(), "RUN cargo check…");
        c.pending = true;
        assert_eq!(c.label_text(), "RUN cargo check (pending)");
    }

    #[test]
    fn test_chip_badge_shows_kind_verb() {
        // Transcript badge carries the kind, never generic "Action".
        let mut c = mk_chip();
        c.kind = ToolPanelKind::Read;
        assert_eq!(c.badge_text(None), " READ ");
        c.kind = ToolPanelKind::Write;
        c.tag_closed = true;
        assert_eq!(c.badge_text(None), " WROTE ");
        c.tag_closed = false;
        assert_eq!(c.badge_text(None), " WRITE ");
        c.kind = ToolPanelKind::Cmd;
        c.tag_closed = true;
        c.body = "ok".into();
        assert_eq!(c.badge_text(Some(34)), " RAN 34s ");
        c.tag_closed = false;
        c.body.clear();
        assert_eq!(c.badge_text(Some(45)), " RUN 45s ");
        // Label and badge share one verb definition.
        assert!(c.label_text_with_duration(Some("45s")).starts_with("RUN "));
        assert_eq!(c.badge_verb(), "RUN");
    }

    #[test]
    fn test_detect_reads_captures_line_attr() {
        let v = detect_all_stream_tools("<read src=\"a.rs\" line=\"45-55\">");
        assert_eq!(v.len(), 1);
        assert_eq!(v[0].kind, ToolPanelKind::Read);
        assert_eq!(v[0].line_range.as_deref(), Some("45-55"));
        // No line attr: full-file read, no range.
        let v = detect_all_stream_tools("<read src=\"b.rs\">");
        assert_eq!(v.len(), 1);
        assert_eq!(v[0].line_range, None);
    }

    fn test_transcript_scenario_stray_tag_between_writes() {
        // Exact failure shape: prose + write + stray + prose + write.
        // Final transcript keeps every prose line and the stray tag;
        // executed writes are carried by chips (removed here).
        let stream = "I will create the project.\n<write src=\"index.html\">H</write>\n</stray>\nNow I will create the next file.\n<write src=\"main.ts\">M</write>\nDone.";
        let shown = redact_tools_for_chat(stream);
        for prose in [
            "I will create the project.",
            "</stray>",
            "Now I will create the next file.",
            "Done.",
        ] {
            assert!(shown.contains(prose), "transcript must keep {prose:?}");
        }
        assert!(
            !shown.contains("<write"),
            "executed writes carried by chips"
        );
    }
}
