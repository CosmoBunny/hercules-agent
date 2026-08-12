use crate::agent::{
    FolderScope, PermissionMode, ProposedAction, allow_session_tools, get_tool_permissions,
    set_folder_scope, set_permission_mode,
};
use crate::backend::{AgentBackend, LlamaCppBackend, LlamaCppLibBackend, OllamaBackend};
use crate::manager::ModelManager;
use crate::task_manager::{QUICK_SECS, TaskEvent, TaskManager};
use crate::tool_panel::{self, PanelChromeHit, ToolChip, ToolPanel, ToolPanelKind};
use crossterm::event::{self, Event, KeyCode, KeyModifiers, MouseButton, MouseEventKind};
use kramaframe::prelude::{KeyFrameFunction, KeyList};
use kramaframe::{BTclasslist, BTframelist, KramaFrame, keylist::TRES16Bits};
use ratatui::{
    Frame,
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, BorderType, Borders, Clear, List, ListItem, ListState, Paragraph},
};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use std::time::Instant;
use sysinfo::System;

pub struct App {
    pub should_quit: bool,
    pub status_message: String,

    // Chat state
    pub input: String,
    pub messages: Vec<String>,
    pub backend: AgentBackend,

    // Registry state
    pub manager: ModelManager,
    pub registry_models: Vec<String>,
    pub registry_state: ListState,

    // System stats
    pub sys: System,

    // Config & Navigation
    pub theme_color: Color,
    pub show_menu: bool,
    pub menu_section: usize, // 0: Registry, 1: Installed Models, 2: Settings
    pub config_state: ListState,

    // Installed Models state
    pub installed_models: Vec<String>,
    pub installed_state: ListState,

    // Focus & Scroll state
    pub input_focused: bool,
    pub input_cursor_position: usize,
    pub scroll_offset: u16,
    pub auto_scroll_enabled: bool,
    pub typewriter_len: usize,
    pub thinking_collapsed: bool,
    pub delete_confirm_model: Option<String>,
    pub esc_hold_start: Option<std::time::Instant>,
    /// Text selection: drag shades rows; Ctrl+C copies; click elsewhere cancels.
    pub selection_start: Option<(u16, u16)>,
    pub selection_end: Option<(u16, u16)>,
    pub is_selecting: bool,
    /// True after a drag produced a range (stays until copy / click-cancel).
    pub has_selection: bool,
    /// True once the pointer moved during this press (distinguishes click vs drag).
    pub selection_dragged: bool,
    /// Click-down while selection active → cancel only if released without drag.
    pub selection_pending_cancel: bool,
    pub selected_text_buffer: String,
    /// Last chat area for selection ↔ screen mapping
    pub last_chat_area: Option<ratatui::layout::Rect>,
    /// Plain text of each logical chat line (same index as draw lines) for copy.
    pub last_chat_plain_lines: Vec<String>,
    /// Estimated context tokens used / limit (for status)
    pub context_tokens_est: usize,
    pub context_compact_count: u32,

    // Dynamic HF models & Search
    pub hf_models: Vec<String>,
    pub registry_search_query: String,
    pub search_results: Arc<Mutex<Option<Vec<String>>>>,

    // Live Activity Logs (Split Pane Console)
    pub activity_logs: Arc<Mutex<Vec<String>>>,
    pub log_pane_collapsed: bool,

    // Animation state using KramaFrame
    pub krama: KramaFrame<BTclasslist, BTframelist<TRES16Bits, i16>>,
    pub anim_tick: u64,
    pub current_log_pane_pct: f64,
    pub last_frame_time: std::time::Instant,

    // Download progress
    pub download_progress: Arc<Mutex<Option<f64>>>,
    pub download_complete: Arc<Mutex<bool>>,

    // Streaming response state
    pub streaming_response: Arc<Mutex<String>>,
    pub is_generating: Arc<Mutex<bool>>,
    pub generation_error: Arc<Mutex<Option<String>>>,

    // Initialization & Auto loop state
    pub initialized: bool,
    pub init_triggered: bool,
    pub auto_tool_turns: usize,

    /// Number of automatic continuations used to finish a truncated tool tag.
    pub incomplete_tool_continuations: usize,

    pub recent_tool_calls: Vec<String>,
    pub repeat_count: usize,
    /// Set on Ctrl+C so the completion path skips tool re-prompt / process races.
    pub user_cancelled_gen: bool,

    // Input undo/redo (snapshots of full prompt text + cursor)
    pub input_undo: Vec<(String, usize)>,
    pub input_redo: Vec<(String, usize)>,

    // F1 keyboard shortcuts overlay (off by default)
    pub show_shortcuts: bool,

    // Permissions tab selection (menu_section == 4)
    pub perms_state: ListState,
    // Runtime tab (power mode + repeat detector) menu_section == 3
    pub runtime_state: ListState,
    /// Animated write/run detail panel (flies from a chip)
    pub tool_panel: Option<ToolPanel>,
    /// Cached rect for chrome hit-testing
    pub tool_panel_rect: Option<ratatui::layout::Rect>,
    /// True while reverse fly is playing (drop panel when t→0)
    pub panel_closing: bool,
    /// Bordered clickable labels (write + run)
    pub tool_chips: Vec<ToolChip>,
    next_chip_id: u64,
    /// Write/cmd waiting for Y accept / N reject (Ask permission mode)
    pub pending_actions: Vec<ProposedAction>,
    /// Full tool stdout/stderr for the *model* context only — not dumped into chat UI.
    pub tool_result_context: Vec<String>,
    /// Long-running shell jobs (http.server, cargo, …)
    pub task_manager: TaskManager,
    /// Stall detection: last time streaming_response grew while generating
    pub gen_last_progress: Option<Instant>,
    pub gen_last_len: usize,
    /// TERM panel interactive input (when panel.interactive)
    pub term_input: String,
    /// Targets already written mid-stream this turn (AlwaysAllow furious mode).
    /// Prevents double-execution when the post-gen path also sees the same tags.
    pub streamed_writes_done: Vec<String>,
}

impl App {
    pub fn new() -> Self {
        let app = Self {
            should_quit: false,
            status_message: "Ready.".to_string(),
            input: String::new(),
            messages: vec!["System: Welcome to Hercules. Ask me anything!".to_string()],
            backend: {
                // Prefer local GGUF with llama.rs (pure Rust); else llama.cpp if path exists
                let mgr = ModelManager::new();
                if let Some(path) = mgr.latest_gguf_path() {
                    AgentBackend::LlamaCppLib(LlamaCppLibBackend::gguf(path))
                } else {
                    AgentBackend::LlamaCpp(LlamaCppBackend::server(
                        "http://localhost:8080".into(),
                        "llama.cpp".into(),
                    ))
                }
            },
            manager: ModelManager::new(),
            registry_models: Vec::new(),
            incomplete_tool_continuations: 0,
            registry_state: ListState::default(),
            sys: System::new_all(),
            theme_color: Color::Rgb(0, 255, 128),
            show_menu: false,
            menu_section: 0,
            config_state: {
                let mut st = ListState::default();
                st.select(Some(0)); // llama.rs first
                st
            },
            installed_models: ModelManager::new().list_installed_local(),
            installed_state: {
                let mut st = ListState::default();
                st.select(Some(0));
                st
            },
            input_focused: true,
            input_cursor_position: 0,
            scroll_offset: 0,
            auto_scroll_enabled: true,
            typewriter_len: 1000,
            thinking_collapsed: false,
            delete_confirm_model: None,
            esc_hold_start: None,
            selection_start: None,
            selection_end: None,
            is_selecting: false,
            has_selection: false,
            selection_dragged: false,
            selection_pending_cancel: false,
            selected_text_buffer: String::new(),
            last_chat_area: None,
            last_chat_plain_lines: Vec::new(),
            context_tokens_est: 0,
            context_compact_count: 0,
            hf_models: Vec::new(),
            registry_search_query: String::new(),
            search_results: Arc::new(Mutex::new(None)),
            activity_logs: Arc::new(Mutex::new(vec![
                "[SYSTEM] Hercules Engine initialized.".to_string(),
                "[HARDWARE] Vulkan/WGPU GPU acceleration active.".to_string(),
            ])),
            log_pane_collapsed: false,
            krama: {
                let mut k = KramaFrame::default();
                k.extend_iter_classlist([
                    (
                        "slide",
                        KeyFrameFunction::new_cubic_bezier_f32(1.0, 0.0, 0.6, 1.0),
                    ),
                    ("focus", KeyFrameFunction::EaseInOut),
                    ("pbar", KeyFrameFunction::Linear),
                    ("menu_fade", KeyFrameFunction::EaseOut),
                    ("list_fade", KeyFrameFunction::EaseOut),
                    ("help_fade", KeyFrameFunction::EaseOut),
                    // Panel fly: cubic ease, 500ms, progress 0→1 open / reverse close
                    (
                        "panel_fly",
                        KeyFrameFunction::new_cubic_bezier_f32(0.22, 1.0, 0.36, 1.0),
                    ),
                ]);
                k.framelist.extend([
                    ("slide", KeyList::new(0, TRES16Bits::from_millis(300))),
                    ("focus", KeyList::new(0, TRES16Bits::from_millis(600))),
                    ("pbar", KeyList::new(0, TRES16Bits::from_millis(1000))),
                    ("menu_fade", KeyList::new(0, TRES16Bits::from_millis(250))),
                    ("list_fade", KeyList::new(0, TRES16Bits::from_millis(350))),
                    ("help_fade", KeyList::new(0, TRES16Bits::from_millis(280))),
                    ("panel_fly", KeyList::new(0, TRES16Bits::from_millis(500))),
                ]);
                k.restart_progress("slide", 0);
                k.restart_progress("focus", 0);
                k.restart_progress("pbar", 0);
                k.restart_progress("menu_fade", 0);
                k.restart_progress("list_fade", 0);
                // panel_* started when a tool panel opens
                k
            },
            anim_tick: 0,
            current_log_pane_pct: 32.0,
            last_frame_time: std::time::Instant::now(),
            download_progress: Arc::new(Mutex::new(None)),
            download_complete: Arc::new(Mutex::new(false)),
            streaming_response: Arc::new(Mutex::new(String::new())),
            is_generating: Arc::new(Mutex::new(false)),
            generation_error: Arc::new(Mutex::new(None)),
            initialized: true,
            init_triggered: true,
            auto_tool_turns: 0,
            recent_tool_calls: Vec::new(),
            repeat_count: 0,
            user_cancelled_gen: false,
            input_undo: Vec::new(),
            input_redo: Vec::new(),
            show_shortcuts: false,
            perms_state: {
                let mut st = ListState::default();
                st.select(Some(0));
                st
            },
            runtime_state: {
                let mut st = ListState::default();
                st.select(Some(1)); // Normal power by default
                st
            },
            tool_panel: None,
            tool_panel_rect: None,
            panel_closing: false,
            tool_chips: Vec::new(),
            next_chip_id: 1,
            pending_actions: Vec::new(),
            tool_result_context: Vec::new(),
            task_manager: TaskManager::new(),
            gen_last_progress: None,
            gen_last_len: 0,
            term_input: String::new(),
            streamed_writes_done: Vec::new(),
        };

        let manager_clone = app.manager.clone();
        let search_results_clone = app.search_results.clone();
        tokio::spawn(async move {
            let mut combined = Vec::new();
            if let Ok(ollama_models) = manager_clone.list_ollama_models().await {
                for m in ollama_models {
                    let sz = if m.size > 0 {
                        crate::manager::format_model_size(m.size)
                    } else {
                        "?".into()
                    };
                    combined.push(format!("Ollama Local: {} ({sz})", m.name));
                }
            }
            let hf = manager_clone.search_all_models("deepseek").await;
            combined.extend(hf);
            *search_results_clone.lock().unwrap() = Some(combined);
        });

        app
    }
    fn record_write_result(&mut self, action: &crate::agent::ProposedAction, result: &str) {
        let pretty = tool_panel::format_tool_output_for_chat(result);

        // Result is for the LLM/context, NOT the source-code chip body.
        self.tool_result_context.push(pretty.clone());

        if self.tool_result_context.len() > 8 {
            let n = self.tool_result_context.len() - 8;
            self.tool_result_context.drain(0..n);
        }

        // Update ONLY the chip that produced this result.
        if let Some(chip_id) = action.chip_id {
            if let Some(chip) = self.tool_chips.iter_mut().find(|c| c.id == chip_id) {
                chip.pending = false;
                chip.tag_closed = true;
            }

            self.force_open_panel_from_chip(chip_id);
        }

        let lines = pretty.lines().count();

        self.messages
            .push(format!("System: [OK] write finished ({lines} lines)"));
    }

    /// Upsert bordered chips from stream (write + run). Auto-opens on new/update.
    /// Index of the latest Agent: message (chips stick under that turn).
    fn latest_agent_msg_idx(&self) -> Option<usize> {
        self.messages
            .iter()
            .enumerate()
            .rev()
            .find(|(_, m)| m.starts_with("Agent:"))
            .map(|(i, _)| i)
    }

    fn sync_tool_chips(&mut self, stream: &str) {
        let mut auto_open: Option<u64> = None;
        let anchor = self.latest_agent_msg_idx();
        for view in tool_panel::detect_all_stream_tools(stream) {
            let target = tool_panel::normalize_target(view.kind, &view.target);
            // Only upsert within the *current* agent turn — never steal older chips.
            // For WRITE: also coalesce path renames on the same turn (one streaming write
            // must not become file.txt + index.html + title_slug.html chips).
            let existing_idx = self
                .tool_chips
                .iter()
                .enumerate()
                .rev()
                .find(|(_, c)| {
                    if c.kind != view.kind || c.anchor_msg != anchor {
                        return false;
                    }
                    if tool_panel::same_tool_target(view.kind, &c.target, &target) {
                        return true;
                    }
                    // Same open write stream, path still settling
                    view.kind == ToolPanelKind::Write && (!c.tag_closed || !view.tag_closed)
                })
                .map(|(i, _)| i);

            if let Some(i) = existing_idx {
                let chip = &mut self.tool_chips[i];
                let was_open = !chip.tag_closed;
                // Prefer longer / final path once closed; while open keep first stable path
                // unless model path clearly has an extension and differs only by rename.
                if view.tag_closed || chip.target.is_empty() || !chip.target.contains('.') {
                    chip.target = target.clone();
                } else if view.tag_closed {
                    chip.target = target.clone();
                }
                if view.kind == ToolPanelKind::Write || !view.body.is_empty() {
                    chip.body = view.body;
                }
                chip.tag_closed = view.tag_closed;
                if was_open
                    || matches!(view.kind, ToolPanelKind::Write | ToolPanelKind::Read)
                    || view.tag_closed
                {
                    auto_open = Some(chip.id);
                }
            } else {
                let id = self.next_chip_id;
                self.next_chip_id += 1;
                self.tool_chips.push(ToolChip {
                    id,
                    kind: view.kind,
                    target,
                    body: view.body,
                    tag_closed: view.tag_closed,
                    pending: false,
                    rect: None,
                    anchor_msg: anchor,
                });
                auto_open = Some(id);
            }
        }
        self.dedupe_tool_chips();
        if let Some(id) = auto_open {
            if self.panel_closing {
                self.panel_closing = false;
            }
            self.force_open_panel_from_chip(id);
        }
    }

    /// Collapse chips that refer to the same tool **within the same agent turn**.
    /// Different turns keep separate chips so click opens that turn's panel, not the latest.
    fn dedupe_tool_chips(&mut self) {
        let mut kept: Vec<ToolChip> = Vec::new();
        for chip in self.tool_chips.drain(..) {
            if let Some(existing) = kept.iter_mut().find(|c| {
                c.kind == chip.kind
                    && c.anchor_msg == chip.anchor_msg
                    && tool_panel::same_tool_target(chip.kind, &c.target, &chip.target)
            }) {
                if chip.body.len() >= existing.body.len() {
                    existing.body = chip.body;
                }
                existing.tag_closed = existing.tag_closed || chip.tag_closed;
                existing.pending = existing.pending || chip.pending;
                existing.target = tool_panel::normalize_target(chip.kind, &chip.target);
            } else {
                kept.push(chip);
            }
        }
        // Keep more history so older RUN/READ/WRITE chips stay clickable
        if kept.len() > 24 {
            let n = kept.len() - 24;
            kept.drain(0..n);
        }
        self.tool_chips = kept;
    }

    /// Mouse-click chip → fly panel (toggle same chip closes).
    fn open_panel_from_chip(&mut self, chip_id: u64) {
        if let Some(ref p) = self.tool_panel {
            if !self.panel_closing && p.chip_id == chip_id {
                self.close_tool_panel();
                return;
            }
        }
        self.force_open_panel_from_chip(chip_id);
    }

    /// Always open / switch to this chip (no toggle-close). Restarts fly open.
    fn force_open_panel_from_chip(&mut self, chip_id: u64) {
        let Some(chip) = self.tool_chips.iter().find(|c| c.id == chip_id).cloned() else {
            return;
        };
        // Already showing this chip and open — just refresh body, don't restart anim
        if let Some(ref mut p) = self.tool_panel {
            if !self.panel_closing && p.chip_id == chip_id {
                p.set_body_streaming(chip.body.clone(), chip.tag_closed);
                if chip.tag_closed || !chip.body.is_empty() {
                    p.reveal_all();
                }
                return;
            }
        }
        let mut panel = ToolPanel::from_chip(&chip);
        if chip.tag_closed || !chip.body.is_empty() {
            panel.reveal_all();
        }
        if chip.kind == ToolPanelKind::Write && !chip.tag_closed {
            panel.live_stream = true;
        }
        self.tool_panel = Some(panel);
        self.panel_closing = false;
        self.krama.restart_progress("panel_fly", 0);
        self.status_message = format!(
            "{} — [-] minimize  [x] close  (or click chip)",
            chip.kind.title_prefix()
        );
    }

    fn close_tool_panel(&mut self) {
        if self.tool_panel.is_some() && !self.panel_closing {
            // If fly not fully open yet, still reverse from current progress
            let t = self.krama.get_progress_f32("panel_fly", 0).abs();
            if t < 0.05 {
                // Already near closed — drop immediately
                self.tool_panel = None;
                self.tool_panel_rect = None;
                self.panel_closing = false;
                return;
            }
            // If finished at +1, reverse_animate → negative progress (1→0 abs)
            // If mid-open, reverse from current
            if !self.krama.is_reversed("panel_fly", 0) {
                self.krama.reverse_animate("panel_fly", 0);
            }
            self.panel_closing = true;
            self.status_message = "Panel closing…".to_string();
        } else if self.tool_panel.is_none() {
            self.tool_panel_rect = None;
            self.panel_closing = false;
        }
    }

    fn toggle_minimize_tool_panel(&mut self) {
        if let Some(ref mut p) = self.tool_panel {
            if self.panel_closing {
                return;
            }
            p.minimized = !p.minimized;
            if p.minimized {
                p.interactive = false;
                self.term_input.clear();
            }
            self.status_message = if p.minimized {
                "Panel minimized — click [+] to restore".into()
            } else {
                "Panel restored".into()
            };
        }
    }

    fn enter_term_interactive(&mut self) {
        if let Some(ref mut p) = self.tool_panel {
            if p.kind != ToolPanelKind::Cmd {
                return;
            }
            p.interactive = true;
            p.minimized = false;
            self.input_focused = false;
            self.status_message =
                "TERM interactive — type + Enter to run; click outside to leave".into();
        }
    }

    fn exit_term_interactive(&mut self) {
        if let Some(ref mut p) = self.tool_panel {
            if p.interactive {
                p.interactive = false;
                self.status_message = "Left TERM interactive".into();
            }
        }
        self.term_input.clear();
    }

    fn term_is_interactive(&self) -> bool {
        self.tool_panel
            .as_ref()
            .map(|p| p.interactive && p.kind == ToolPanelKind::Cmd)
            .unwrap_or(false)
    }

    /// Run a line typed in interactive TERM via task manager.
    fn term_run_line(&mut self, line: &str) {
        let line = line.trim();
        if line.is_empty() {
            return;
        }
        let cmd = line.to_string();
        // Append to panel body as echo
        if let Some(ref mut p) = self.tool_panel {
            if !p.body.ends_with('\n') && !p.body.is_empty() {
                p.body.push('\n');
            }
            p.body.push_str(&format!("$ {cmd}\n"));
            p.scroll_to_end();
        }
        let id = self.task_manager.spawn_cmd(cmd.clone());
        self.messages
            .push(format!("System: [TERM] Task #{id} `{cmd}` (interactive)"));
        self.status_message = format!("TERM ran Task #{id}");
        if let Ok(mut l) = self.activity_logs.lock() {
            l.push(format!("[TERM interactive] Task #{id}: {cmd}"));
        }
    }

    fn clear_selection(&mut self) {
        self.selection_start = None;
        self.selection_end = None;
        self.is_selecting = false;
        self.has_selection = false;
        self.selection_dragged = false;
        self.selection_pending_cancel = false;
        self.selected_text_buffer.clear();
    }

    fn copy_selection_to_clipboard(&mut self) -> bool {
        if self.selected_text_buffer.trim().is_empty() {
            self.rebuild_selected_text();
        }
        if self.selected_text_buffer.trim().is_empty() {
            return false;
        }
        let text = self.selected_text_buffer.clone();
        // Silent path only — arboard/xclip stderr was corrupting the TUI header
        let ok = crate::clipboard::copy_text_silent(&text);
        let n = text.chars().count();
        self.status_message = if ok {
            format!(
                "Copied {n} chars → clipboard + {}",
                crate::clipboard::clipboard_file_path()
            )
        } else {
            format!(
                "Saved {n} chars to {} (no system clipboard tool)",
                crate::clipboard::clipboard_file_path()
            )
        };
        if let Ok(mut l) = self.activity_logs.lock() {
            l.push(format!("[CLIPBOARD] {n} chars (silent)"));
        }
        self.clear_selection();
        true
    }

    fn rebuild_selected_text(&mut self) {
        let (Some(start), Some(end)) = (self.selection_start, self.selection_end) else {
            self.selected_text_buffer.clear();
            return;
        };
        let min_y = start.1.min(end.1) as i32;
        let max_y = start.1.max(end.1) as i32;
        let chat_y = self
            .last_chat_area
            .map(|a| a.y.saturating_add(1))
            .unwrap_or(1) as i32;
        // Prefer plain lines from last draw (matches shaded rows)
        let mut extracted: Vec<String> = Vec::new();
        if !self.last_chat_plain_lines.is_empty() {
            for (i, line) in self.last_chat_plain_lines.iter().enumerate() {
                let screen_y = chat_y + i as i32 - self.scroll_offset as i32;
                if screen_y >= min_y && screen_y <= max_y {
                    extracted.push(line.clone());
                }
            }
        } else {
            let full = self.messages.join("\n");
            for (l_idx, line) in full.lines().enumerate() {
                let screen_y = chat_y + l_idx as i32 - self.scroll_offset as i32;
                if screen_y >= min_y && screen_y <= max_y {
                    extracted.push(line.to_string());
                }
            }
        }
        self.selected_text_buffer = extracted.join("\n");
    }

    fn selection_active(&self) -> bool {
        self.is_selecting || self.has_selection
    }

    /// Runtime menu: +/- on selected row (repeat threshold or context window).
    fn runtime_nudge_selected(&mut self, dir: i32) {
        use crate::settings::{
            format_context_tokens, get_settings, nudge_context_token_limit, nudge_repeat_threshold,
        };
        let Some(i) = self.runtime_state.selected() else {
            return;
        };
        match i {
            // llama.cpp sub-backend
            3 => {
                use crate::settings::cycle_llama_rs_sub_backend;
                let b = cycle_llama_rs_sub_backend();
                self.status_message = format!(
                    "llama.cpp sub-backend: {} (HERCULES_COMPUTE_BACKEND={})",
                    b.label(),
                    b.env_val()
                );
            }
            // Stall timeout
            4 => {
                use crate::settings::{format_stall_timeout, nudge_stall_timeout};
                let t = nudge_stall_timeout(dir);
                self.status_message = format!(
                    "Stall timeout: {}  (+/− to adjust)",
                    format_stall_timeout(t)
                );
            }
            // Repeat threshold
            5 => {
                let n = nudge_repeat_threshold(dir);
                self.status_message = format!("Repeat threshold: {n}  (+/− to adjust)");
            }
            // Context window
            7 => {
                let n = nudge_context_token_limit(dir);
                crate::llama::server::shutdown_managed_server();
                self.status_message = format!(
                    "Context: {} tokens  (+/− step; llama-server restarts next gen)",
                    format_context_tokens(n)
                );
            }
            _ => {
                self.status_message =
                    "Select Sub-Backend / Stall / Repeat / Context row, then press +/−".into();
                return;
            }
        }
        // Status + activity log only — never push settings into chat/AI context
        let s = get_settings();
        if let Ok(mut l) = self.activity_logs.lock() {
            l.push(format!(
                "[RUNTIME] power={} ctx={} temp={:.2} repeat={} think={}",
                s.power_mode.label(),
                format_context_tokens(crate::settings::context_token_limit()),
                crate::settings::temperature(),
                s.repeat_threshold,
                s.repeat_detect_thinking
            ));
        }
    }

    /// Finish mouse-up selection: keep highlight until Ctrl+C or click-cancel.
    fn finalize_selection(&mut self) {
        self.is_selecting = false;
        if self.selection_pending_cancel && !self.selection_dragged {
            self.clear_selection();
            self.status_message = "Selection cancelled".into();
            return;
        }
        self.selection_pending_cancel = false;
        // Keep selection even if rebuild yields empty (coords still shaded)
        let (Some(s), Some(e)) = (self.selection_start, self.selection_end) else {
            self.clear_selection();
            return;
        };
        let moved = s != e || self.selection_dragged;
        if !moved {
            // Pure click with no prior selection intent
            self.clear_selection();
            return;
        }
        self.has_selection = true;
        self.rebuild_selected_text();
        let n = self.selected_text_buffer.chars().count();
        self.status_message = if n > 0 {
            format!("Selected {n} chars — Ctrl+C copy, click cancel")
        } else {
            "Selected rows — Ctrl+C copy, click cancel".into()
        };
    }

    /// Insert a character (or newline) at the input cursor.
    fn input_insert_char(&mut self, c: char) {
        self.push_input_undo();
        let pos = self.input_cursor_position.min(self.input.chars().count());
        let mut chars: Vec<char> = self.input.chars().collect();
        chars.insert(pos, c);
        self.input = chars.into_iter().collect();
        self.input_cursor_position = pos + 1;
    }

    /// Logical lines of the prompt (split on `\n`).
    fn input_line_count(&self) -> usize {
        if self.input.is_empty() {
            1
        } else {
            self.input.lines().count().max(1) + if self.input.ends_with('\n') { 1 } else { 0 }
        }
    }

    /// Map char cursor index → (col, row) inside the input box content width.
    fn input_cursor_col_row(&self, content_width: usize) -> (u16, u16) {
        let width = content_width.max(1);
        let pos = self.input_cursor_position.min(self.input.chars().count());
        let prefix: String = self.input.chars().take(pos).collect();
        let mut row: u16 = 0;
        let mut col: u16 = 0;
        // Leading space in display
        col = 1;
        for ch in prefix.chars() {
            if ch == '\n' {
                row = row.saturating_add(1);
                col = 0;
            } else {
                col = col.saturating_add(1);
                if col as usize >= width {
                    row = row.saturating_add(1);
                    col = 0;
                }
            }
        }
        (col, row)
    }

    /// Queue write/cmd for user accept; open preview panel.
    fn propose_actions(&mut self, mut actions: Vec<ProposedAction>) {
        if actions.is_empty() {
            return;
        }
        // Ensure chips exist + mark pending; auto-open latest
        let mut open_id = None;
        for a in &mut actions {
            let target = match a.kind {
                crate::agent::ProposedKind::Write => {
                    crate::agent::AgentEngine::expand_path(&a.target)
                        .display()
                        .to_string()
                }
                crate::agent::ProposedKind::Cmd => a.target.clone(),
            };
            let kind = match a.kind {
                crate::agent::ProposedKind::Write => ToolPanelKind::Write,
                crate::agent::ProposedKind::Cmd => ToolPanelKind::Cmd,
            };
            let target = tool_panel::normalize_target(kind, &target);
            let anchor = self.latest_agent_msg_idx();
            if let Some(chip) = self.tool_chips.iter_mut().find(|c| {
                c.kind == kind
                    && c.anchor_msg == anchor
                    && tool_panel::same_tool_target(kind, &c.target, &target)
            }) {
                chip.target = target.clone();
                chip.body = a.body.clone();
                chip.tag_closed = true;
                chip.pending = true;

                let chip_id = chip.id;

                if chip.anchor_msg.is_none() {
                    chip.anchor_msg = anchor;
                }

                a.chip_id = Some(chip_id);

                open_id = Some(chip_id);
            } else {
                let id = self.next_chip_id;
                self.next_chip_id += 1;

                self.tool_chips.push(ToolChip {
                    id,
                    kind,
                    target,
                    body: a.body.clone(),
                    tag_closed: true,
                    pending: true,
                    rect: None,
                    anchor_msg: anchor,
                });

                a.chip_id = Some(id);

                open_id = Some(id);
            }
        }
        self.dedupe_tool_chips();
        if let Some(id) = open_id {
            self.force_open_panel_from_chip(id);
        }
        let n = actions.len();
        let summary = actions
            .iter()
            .map(|a| {
                format!(
                    "{} -> {}",
                    a.kind.label(),
                    if a.target.len() > 48 {
                        format!("{}...", &a.target[..45])
                    } else {
                        a.target.clone()
                    }
                )
            })
            .collect::<Vec<_>>()
            .join("; ");
        let think_note = if actions.iter().any(|a| a.from_think) {
            " (recovered from <think> — model nested the tool wrongly)"
        } else {
            ""
        };
        // Only show the PENDING chat message when the AI has already finished.
        // Mid-generation we stay silent so the stream is never interrupted.
        let currently_generating = *self.is_generating.lock().unwrap();
        if !currently_generating {
            self.messages.push(format!(
                "System: [PENDING] {n} action(s){think_note}: {summary}\n\
                 Press Y or Enter to ACCEPT | N to reject | A always-allow this session"
            ));
            self.status_message = "Y/Enter=Accept | N=Reject | A=Always allow session".to_string();
        } else {
            // AI still streaming — update status bar only, no chat disruption
            self.status_message = format!("Pending {n} write(s) — Y to accept after AI finishes");
        }
        self.pending_actions = actions;
        self.input_focused = false; // so Y/N aren't typed into the prompt
    }

    fn accept_pending_actions(&mut self) {
        if self.pending_actions.is_empty() {
            return;
        }
        allow_session_tools();
        let actions = std::mem::take(&mut self.pending_actions);
        self.status_message = "Running accepted action…".to_string();

        let mut writes = Vec::new();
        let mut cmds = Vec::new();
        for a in actions {
            match a.kind {
                crate::agent::ProposedKind::Write => writes.push(a),
                crate::agent::ProposedKind::Cmd => cmds.push(a),
            }
        }

        if !writes.is_empty() {
            let mut written = Vec::new();
            for a in &writes {
                let result = crate::agent::AgentEngine::execute_proposed(a);

                // Update the chip to [WROTE] without injecting a tool-result turn.
                // We intentionally do NOT push to tool_result_context or
                // trigger_generation_from_context — the AI continues uninterrupted.
                if let Some(chip_id) = a.chip_id {
                    if let Some(chip) = self.tool_chips.iter_mut().find(|c| c.id == chip_id) {
                        chip.pending = false;
                        chip.tag_closed = true;
                    }
                    self.force_open_panel_from_chip(chip_id);
                }

                let ok = !result.starts_with("Error");
                written.push((a.target.rsplit('/').next().unwrap_or(&a.target).to_string(), ok));
            }
            let summary = written
                .iter()
                .map(|(name, ok)| if *ok { format!("✓ {name}") } else { format!("✗ {name}") })
                .collect::<Vec<_>>()
                .join(", ");
            self.status_message = format!("Written: {summary}");
            // Do NOT re-trigger generation — let AI finish its current response.
        }

        if !cmds.is_empty() {
            self.spawn_cmds_to_task_manager(cmds);
            // Cmds still trigger re-prompt so the AI sees the shell output
        }
    }

    /// Run shell cmds via task manager (non-blocking; park after 10s).
    fn spawn_cmds_to_task_manager(&mut self, cmds: Vec<ProposedAction>) {
        for a in cmds {
            let cmd = a.target.clone();
            let id = self.task_manager.spawn_cmd(cmd.clone());
            // Update chip body with task id
            if let Some(chip) = self.tool_chips.iter_mut().rev().find(|c| {
                c.kind == ToolPanelKind::Cmd
                    && tool_panel::same_tool_target(ToolPanelKind::Cmd, &c.target, &cmd)
            }) {
                chip.pending = false;
                chip.tag_closed = true;
                chip.body = format!("[Task #{id} running]\n$ {cmd}\n…");
            } else {
                let cid = self.next_chip_id;
                self.next_chip_id += 1;
                self.tool_chips.push(ToolChip {
                    id: cid,
                    kind: ToolPanelKind::Cmd,
                    target: cmd.clone(),
                    body: format!("[Task #{id} running]\n$ {cmd}\n…"),
                    tag_closed: true,
                    pending: false,
                    rect: None,
                    anchor_msg: self.latest_agent_msg_idx(),
                });
            }
            if let Some(chip) = self.tool_chips.iter().rev().find(|c| {
                c.kind == ToolPanelKind::Cmd
                    && tool_panel::same_tool_target(ToolPanelKind::Cmd, &c.target, &cmd)
            }) {
                self.force_open_panel_from_chip(chip.id);
            }
            self.messages.push(format!(
                "System: [Task #{id}] started: `{cmd}` (if >{QUICK_SECS}s → task manager; Ctrl+C kills)"
            ));
            if let Ok(mut l) = self.activity_logs.lock() {
                l.push(format!("[TASK #{id}] started: {cmd}"));
            }
        }
        self.status_message = format!(
            "{} background task(s) — Ctrl+C kills all",
            self.task_manager.running_count()
        );
    }

    /// Drain task manager events (parked / done) into chat + optional re-prompt.
    fn poll_task_events(&mut self) {
        let events = self.task_manager.take_events();
        for ev in events {
            match ev {
                TaskEvent::Parked { id, cmd } => {
                    self.messages.push(format!(
                        "System: [Task #{id}] still running after {QUICK_SECS}s — pushed to task manager. \
                         Command: `{cmd}`. Agent may continue; output arrives when finished. \
                         Ctrl+C kills running tasks."
                    ));
                    self.tool_result_context.push(format!(
                        "[Task #{id} PARKED — still running]\ncmd: {cmd}\n\
                         Do not re-run this command. Wait for [Task #{id} DONE] or start other work."
                    ));
                    if let Some(chip) = self.tool_chips.iter_mut().rev().find(|c| {
                        c.kind == ToolPanelKind::Cmd
                            && tool_panel::same_tool_target(ToolPanelKind::Cmd, &c.target, &cmd)
                    }) {
                        chip.body = format!(
                            "[Task #{id} — long running >{QUICK_SECS}s]\n$ {cmd}\n(waiting… Ctrl+C to kill)"
                        );
                    }
                    // Nudge model that work is async
                    if !*self.is_generating.lock().unwrap() {
                        self.auto_tool_turns += 1;
                        if self.auto_tool_turns <= 8 {
                            self.trigger_generation_from_context();
                        }
                    }
                    self.status_message = format!("Task #{id} parked (long-running)");
                    if let Ok(mut l) = self.activity_logs.lock() {
                        l.push(format!("[TASK #{id}] parked after {QUICK_SECS}s: {cmd}"));
                    }
                }
                TaskEvent::Done {
                    id,
                    cmd,
                    output,
                    killed,
                } => {
                    let label = if killed { "KILLED" } else { "DONE" };
                    let pretty = tool_panel::format_tool_output_for_chat(&output);
                    if let Some(chip) = self.tool_chips.iter_mut().rev().find(|c| {
                        c.kind == ToolPanelKind::Cmd
                            && tool_panel::same_tool_target(ToolPanelKind::Cmd, &c.target, &cmd)
                    }) {
                        chip.body = format!("[Task #{id} {label}]\n$ {cmd}\n{pretty}");
                        chip.tag_closed = true;
                        chip.pending = false;
                        let cid = chip.id;
                        self.force_open_panel_from_chip(cid);
                    }
                    self.tool_result_context.push(format!(
                        "[Task #{id} {label}]\ncmd: {cmd}\n\n{pretty}\n\n\
                         Use this output. Do not re-run the same command unless needed."
                    ));
                    if self.tool_result_context.len() > 8 {
                        let n = self.tool_result_context.len() - 8;
                        self.tool_result_context.drain(0..n);
                    }
                    self.messages.push(format!(
                        "System: [Task #{id} {label}] `{cmd}` ({} lines) — result sent to agent",
                        pretty.lines().count()
                    ));
                    if let Ok(mut l) = self.activity_logs.lock() {
                        l.push(format!("[TASK #{id}] {label} ({} bytes)", pretty.len()));
                    }
                    self.status_message = format!("Task #{id} {label}");
                    if !killed && !*self.is_generating.lock().unwrap() {
                        self.auto_tool_turns += 1;
                        if self.auto_tool_turns <= 8 {
                            self.trigger_generation_from_context();
                        }
                    }
                }
            }
        }

        // Live-update TERM panel body from running tasks
        for t in self.task_manager.list() {
            if t.status != crate::task_manager::TaskStatus::Running {
                continue;
            }
            if let Some(chip) = self.tool_chips.iter_mut().rev().find(|c| {
                c.kind == ToolPanelKind::Cmd
                    && tool_panel::same_tool_target(ToolPanelKind::Cmd, &c.target, &t.cmd)
            }) {
                if !t.output.is_empty() {
                    chip.body = format!(
                        "[Task #{} running {:.0}s]\n$ {}\n{}",
                        t.id,
                        t.started.elapsed().as_secs_f32(),
                        t.cmd,
                        t.output
                    );
                }
            }
        }
    }

    /// Returns true when the latest tool tag was opened but never closed.
    fn has_incomplete_tool_tag(stream: &str) -> bool {
        let last_write = stream.rfind("<write src=");
        let last_cmd = stream.rfind("<cmd>");

        match (last_write, last_cmd) {
            (None, None) => false,

            (Some(w), None) => stream[w..].find("</write>").is_none(),

            (None, Some(c)) => stream[c..].find("</cmd>").is_none(),

            (Some(w), Some(c)) => {
                if w > c {
                    stream[w..].find("</write>").is_none()
                } else {
                    stream[c..].find("</cmd>").is_none()
                }
            }
        }
    }

    fn continue_incomplete_tool(&mut self, partial: &str) -> bool {
        const MAX_CONTINUATIONS: usize = 3;

        if !Self::has_incomplete_tool_tag(partial) {
            return false;
        }

        if self.incomplete_tool_continuations >= MAX_CONTINUATIONS {
            self.messages.push(
                "System: [ERROR] Tool generation remained incomplete after \
             3 continuation attempts. Partial output was not executed."
                    .into(),
            );

            self.status_message = "Incomplete tool — not executed.".into();
            return false;
        }

        self.incomplete_tool_continuations += 1;

        let last_write = partial.rfind("<write src=").unwrap_or(0);
        let last_cmd = partial.rfind("<cmd>").unwrap_or(0);

        let kind = if last_write > last_cmd {
            "write"
        } else {
            "cmd"
        };

        self.messages.push(format!(
            "You: Continue the incomplete {kind} tool from exactly where you stopped. \
         Do NOT restart or repeat the previous content. \
         Emit only the remaining content and make sure the closing \
         </{kind}> tag is emitted."
        ));

        self.status_message = format!(
            "Continuing incomplete {kind} ({}/3)…",
            self.incomplete_tool_continuations
        );

        self.trigger_generation_from_context();

        true
    }

    /// Close open write/cmd chips when stream dies mid-tag (error / stall / leave).
    fn finalize_incomplete_tools(&mut self, reason: &str) {
        let mut n = 0usize;
        for c in &mut self.tool_chips {
            if !c.tag_closed {
                c.tag_closed = true;
                n += 1;
                if c.kind == ToolPanelKind::Write {
                    if !c.body.is_empty() && !c.body.contains("[INCOMPLETE") {
                        c.body.push_str(&format!(
                            "\n\n/* [INCOMPLETE write — {reason}. Re-ask to finish or edit. */\n"
                        ));
                    }
                } else if c.kind == ToolPanelKind::Cmd && c.body.is_empty() {
                    c.body = format!("[INCOMPLETE cmd — {reason}]");
                }
            }
        }
        if n > 0 {
            self.messages.push(format!(
                "System: [OK] Recovered {n} incomplete tool chip(s) after {reason}. \
                 Open WRITE/RUN chip to view partial output; re-prompt to finish."
            ));
            if let Some(id) = self
                .tool_chips
                .iter()
                .rev()
                .find(|c| c.kind == ToolPanelKind::Write)
                .map(|c| c.id)
            {
                self.force_open_panel_from_chip(id);
            }
            if let Ok(mut l) = self.activity_logs.lock() {
                l.push(format!("[TOOLS] finalized {n} incomplete after {reason}"));
            }
        }
    }

    /// Stall detect:
    /// - While llama-server is loading (banner / no tokens yet): use long timeout
    ///   matching server health wait (can be minutes on CPU).
    /// - After real tokens started: 20s with no growth → interrupt.
    fn check_generation_stall(&mut self) {
        let is_gen = *self.is_generating.lock().unwrap();
        if !is_gen {
            self.gen_last_progress = None;
            self.gen_last_len = 0;
            return;
        }
        let stream = self
            .streaming_response
            .lock()
            .map(|s| s.clone())
            .unwrap_or_default();
        let len = stream.len();
        let loading = stream.is_empty()
            || stream.contains("Starting llama-server")
            || stream.contains("Loading model")
            || stream.contains("loads GGUF once")
            || stream.contains("[llama.rs]")
            || stream.contains("Prefilling prompt");
        // Real model text (not only the loading banner)
        let has_tokens = !loading && !stream.trim().is_empty() && !stream.starts_with("__HERCULES");

        let now = Instant::now();
        if self.gen_last_progress.is_none() {
            self.gen_last_progress = Some(now);
            self.gen_last_len = len;
            return;
        }
        if len != self.gen_last_len {
            self.gen_last_len = len;
            self.gen_last_progress = Some(now);
            return;
        }

        let limit_secs = if has_tokens {
            20u64 // idle mid-stream
        } else {
            crate::settings::get_settings().stall_timeout_secs
        };

        if limit_secs > 0 {
            let stalled = self
                .gen_last_progress
                .map(|t| t.elapsed().as_secs() >= limit_secs)
                .unwrap_or(false);
            if !stalled {
                return;
            }
        } else {
            // limit_secs == 0 means Unlimited (watchdog disabled)
            return;
        }
        *self.is_generating.lock().unwrap() = false;
        {
            let mut target = self.streaming_response.lock().unwrap();
            if !target.starts_with("__HERCULES") {
                target.push_str(&format!(
                    "\n[Generation stalled {limit_secs}s — interrupted. \
                     If llama-server was still loading, raise patience or lower Context. Ctrl+C also cancels.]"
                ));
            }
        }
        self.gen_last_progress = None;
        // Recover mid-write stuck state
        let partial = self.streaming_response.lock().unwrap().clone();
        if !partial.is_empty() && !partial.starts_with("__HERCULES") {
            self.sync_tool_chips(&partial);
            if let Some(last) = self.messages.last_mut() {
                if last.starts_with("Agent: ") {
                    let shown = tool_panel::redact_tools_for_chat(&partial);
                    *last = format!("Agent: {shown}\n[STALL {limit_secs}s]");
                }
            }
        }
        self.finalize_incomplete_tools(&format!("stall {limit_secs}s"));
        self.status_message =
            format!("Generation stalled {limit_secs}s — interrupted (Ctrl+C cancels)");
        self.messages.push(format!(
            "System: [STALL] No progress for {limit_secs}s — generation interrupted. \
             Partial WRITE/RUN recovered if any. Server load can take minutes on CPU."
        ));
        if let Ok(mut l) = self.activity_logs.lock() {
            l.push(format!(
                "[STALL] idle {limit_secs}s (has_tokens={has_tokens}) → interrupt"
            ));
        }
    }

    fn reject_pending_actions(&mut self) {
        if self.pending_actions.is_empty() {
            return;
        }
        let n = self.pending_actions.len();
        self.pending_actions.clear();
        // Reject = stop: interrupt any active generation so the model doesn't
        // keep streaming after the user said no.
        let was_gen = *self.is_generating.lock().unwrap();
        if was_gen {
            self.user_cancelled_gen = true;
            self.auto_tool_turns = 0;
            *self.is_generating.lock().unwrap() = false;
            let partial = self.streaming_response.lock().unwrap().clone();
            if !partial.is_empty() && !partial.starts_with("__HERCULES") {
                self.finalize_incomplete_tools("rejected");
            }
        }
        self.messages.push(format!(
            "System: [REJECTED] {n} pending action(s). File not written."
        ));
        self.status_message = "Rejected — generation stopped.".to_string();
        self.close_tool_panel();
    }

    /// UI-only system lines (settings, theme, downloads) — never sent to the model.
    fn is_ui_only_message(m: &str) -> bool {
        let t = m.trim_start();
        if !t.starts_with("System:") {
            return false;
        }
        t.contains("Runtime →")
            || t.contains("Runtime ->")
            || t.contains("Theme changed")
            || t.contains("Session saved")
            || t.contains("Session loaded")
            || t.contains("Unknown command")
            || t.contains("Pulling Ollama")
            || t.contains("Resolving model weights")
            || t.contains("Switched active engine")
            || t.contains("Backend switched")
            || t.contains("Permissions →")
            || t.contains("Power mode:")
            || t.contains("Context limit:")
            || t.contains("Temperature:")
            || t.contains("Repeat threshold:")
            || t.contains("Repeat detect")
            || t.contains("Active Engine:")
            || t.contains("GGUF ready at")
            || t.contains("Ollama model")
            || t.contains("Download flagged")
            || t.contains("Download finished")
            || t.contains("Menu Modal")
            || t.starts_with("System: Welcome to Hercules")
    }

    /// Chat context for the model: user/agent turns + tool outputs + memory.
    /// Settings / UI system lines are excluded.
    pub fn build_context_prompt(&self) -> String {
        let mut parts = Vec::new();

        // Durable memory (survives compact). Label as Notes (not "You:") so 3B models
        // don't parrot the dump as their reply.
        let mem = crate::agent::AgentEngine::memory_read_all();
        if mem != "Memory is empty." && !mem.trim().is_empty() {
            let clean_mem: String = mem
                .lines()
                .filter(|l| {
                    let l = l.trim();
                    !l.contains("Runtime →")
                        && !l.contains("Runtime ->")
                        && !l.contains("repeat_threshold")
                        && !l.contains("think_detect")
                        && !l.contains("power=")
                        && !l.contains("[Context compacted")
                        && !l.contains("Prior chat FORGOTTEN")
                })
                .collect::<Vec<_>>()
                .join("\n");
            let clean_mem = trunc_chars(clean_mem.trim(), 1_800);
            if !clean_mem.is_empty() {
                parts.push(format!(
                    "Notes (facts only — do NOT reprint this block):\n{clean_mem}"
                ));
            }
        }

        let mut chat: Vec<String> = self
            .messages
            .iter()
            .filter_map(|m| {
                // Never feed UI settings / chrome into the model
                if Self::is_ui_only_message(m) {
                    return None;
                }
                if m.starts_with("You: ") || m.starts_with("Agent: ") {
                    if m == "Agent: " || m == "Agent: ▍" || m.trim() == "Agent:" {
                        return None;
                    }
                    // Skip pure refusal loops (keep latest user request useful)
                    let body = m.strip_prefix("Agent: ").unwrap_or(m);
                    if m.starts_with("Agent: ") {
                        let clean = tool_panel::redact_tools_for_chat(body);
                        let low = clean.to_ascii_lowercase();
                        if clean.trim().is_empty() {
                            return None;
                        }
                        // Drop canned refusals from context so they don't reinforce
                        if low.contains("i'm sorry, but i can't assist")
                            || low.contains("i cannot assist")
                            || low.contains("i can't help with that")
                        {
                            return None;
                        }
                        // Drop agent turns that only parrot compact/meta boilerplate
                        if low.contains("context compacted")
                            || low.contains("prior chat forgotten")
                            || low.contains("facts live in memory")
                        {
                            return None;
                        }
                        // Drop half-open write tags with almost no body (echo noise)
                        if clean.contains("<write")
                            && !clean.contains("</write")
                            && clean.len() < 120
                        {
                            return None;
                        }
                        return Some(format!("Agent: {clean}"));
                    }
                    Some(m.clone())
                } else if m.starts_with("System: [") {
                    // NEVER feed long compact banners — small models copy them verbatim.
                    if m.contains("[Context compacted") {
                        return None;
                    }
                    if m.contains("[Task")
                        || m.contains("[OK]")
                        || m.contains("[STALL]")
                        || m.contains("[Tool")
                    {
                        let short = trunc_chars(m.strip_prefix("System: ").unwrap_or(m), 240);
                        return Some(format!("Result: {short}"));
                    }
                    None
                } else {
                    None
                }
            })
            .collect();
        // Keep tool dumps small — full `ls` + history was > n_batch and crashed libllama.
        for r in self.tool_result_context.iter().rev().take(2).rev() {
            chat.push(format!(
                "Result:\n{}\n(Use this; do not re-run the same tool.)",
                trunc_chars(r.trim(), 800)
            ));
        }
        let start = chat.len().saturating_sub(24);
        parts.push(chat[start..].join("\n\n"));
        let body = parts.join("\n\n");

        let mut out = body;
        if self.auto_tool_turns > 0 {
            out.push_str(
                "\n\nInstruction: Tool results are above (Result: blocks). \
                 Reply in natural language only — tell the user what you found. \
                 Do NOT emit any tool tags (<read>, <ls>, <write>, <cmd>). \
                 Do NOT say you lack file access. Open chips already show full content.",
            );
        }
        out
    }

    /// When the model re-emits the same tool instead of answering, post a short host reply
    /// from the last tool dump so the chat does not "flash then clear".
    fn host_answer_from_prior_tools(&mut self) {
        let Some(raw) = self.tool_result_context.last().cloned() else {
            self.messages
                .push("System: Tool already ran. Open the chip above for full output.".into());
            return;
        };
        let preview = trunc_chars(raw.trim(), 900);
        let lines = raw.lines().count();
        // Replace empty / tool-only agent bubble if present
        if let Some(last) = self.messages.last_mut() {
            if last.starts_with("Agent: ") {
                let body = last.strip_prefix("Agent: ").unwrap_or(last);
                let only_tool = crate::agent::AgentEngine::response_has_tool_tags(body)
                    && tool_panel::redact_tools_for_chat(body).trim().is_empty();
                let emptyish = body.trim().is_empty()
                    || body.contains("[Host recovered")
                    || body.contains("[Interrupted");
                if only_tool || emptyish {
                    *last = format!(
                        "Agent: Done — tool already finished ({lines} lines). \
                         Open the chip for the full file/output. Preview:\n\n{preview}"
                    );
                    return;
                }
            }
        }
        self.messages.push(format!(
            "Agent: Done — tool already finished ({lines} lines). \
             Open the chip for the full file/output. Preview:\n\n{preview}"
        ));
    }

    /// Full conversation size for meter/compact — excludes UI-only system lines.
    fn estimate_full_session_tokens(&self) -> usize {
        use crate::settings::estimate_tokens;
        let mut n = estimate_tokens(crate::agent::SYSTEM_PROMPT);
        for m in &self.messages {
            if Self::is_ui_only_message(m) {
                continue;
            }
            n = n.saturating_add(estimate_tokens(m));
        }
        for r in self.tool_result_context.iter() {
            n = n.saturating_add(estimate_tokens(r));
        }
        let mem = crate::agent::AgentEngine::memory_read_all();
        if mem != "Memory is empty." {
            n = n.saturating_add(estimate_tokens(&mem));
        }
        n
    }

    /// Auto-compact when full session ≥ 80% of context limit.
    fn maybe_compact_context(&mut self) {
        let settings = crate::settings::get_settings();
        let limit = settings.context_token_limit.max(2048);
        let est = self.estimate_full_session_tokens();
        self.context_tokens_est = est;
        let threshold = ((limit as f32) * settings.compact_ratio) as usize;
        // Also compact if message count is huge (hallucination / loop risk)
        let msg_pressure = self.messages.len() >= 40
            || self
                .tool_result_context
                .iter()
                .map(|s| s.len())
                .sum::<usize>()
                > 80_000;
        if est < threshold && !msg_pressure {
            return;
        }
        self.compact_context_to_memory(false);
    }

    /// Compress chat → memory and forget prior turns.
    /// `manual` = user typed `/compact` (more aggressive keep-tail).
    fn compact_context_to_memory(&mut self, manual: bool) {
        let before = self.estimate_full_session_tokens();
        self.context_tokens_est = before;

        // Gather material to compress — skip UI settings so they never enter memory/AI
        let mut archive = String::new();
        for m in &self.messages {
            if Self::is_ui_only_message(m) {
                continue;
            }
            if m.starts_with("You: ") || m.starts_with("Agent: ") || m.starts_with("System:") {
                if archive.len() > 200_000 {
                    break;
                }
                archive.push_str(m);
                archive.push('\n');
            }
        }
        for r in &self.tool_result_context {
            if archive.len() > 200_000 {
                break;
            }
            archive.push_str("[tool]\n");
            archive.push_str(r);
            archive.push('\n');
        }

        if archive.trim().is_empty() && !manual {
            return;
        }

        // Deterministic compress — never ask the model to summarize (hallucinates)
        let summary = compress_transcript(&archive);
        let note = format!(
            "[compact #{} | {}] {}",
            self.context_compact_count + 1,
            if manual { "manual /compact" } else { "auto" },
            summary
        );
        let _ = crate::agent::AgentEngine::memory_push(&note);

        // Manual: keep only last user+agent pair; auto: last 4 turns
        let keep_n = if manual { 2usize } else { 4usize };
        let recent: Vec<String> = self
            .messages
            .iter()
            .rev()
            .filter(|m| m.starts_with("You: ") || m.starts_with("Agent: "))
            .take(keep_n)
            .cloned()
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect();

        self.messages.clear();
        self.messages.push(format!(
            "System: [Context compacted — {}] was ~{} tokens (limit {} · {}% auto). \
             Prior chat FORGOTTEN. Facts live in memory (compact #{}). \
             Do NOT invent old conversation details; use [Memory] only. \
             Prefer tools over guessing.",
            if manual { "manual /compact" } else { "auto" },
            before,
            crate::settings::context_token_limit(),
            (crate::settings::get_settings().compact_ratio * 100.0) as u32,
            self.context_compact_count + 1
        ));
        self.messages.extend(recent);
        // Drop tool chips tied to forgotten turns (reduce UI noise / stale anchors)
        if manual {
            self.tool_chips.clear();
            self.tool_panel = None;
            self.tool_panel_rect = None;
            self.panel_closing = false;
        } else {
            let agent_idxs: Vec<usize> = self
                .messages
                .iter()
                .enumerate()
                .filter(|(_, m)| m.starts_with("Agent:"))
                .map(|(i, _)| i)
                .collect();
            for chip in &mut self.tool_chips {
                chip.anchor_msg = agent_idxs.last().copied();
            }
        }
        self.tool_result_context.clear();
        self.auto_tool_turns = 0;
        self.recent_tool_calls.clear();
        self.repeat_count = 0;
        self.context_compact_count += 1;
        self.context_tokens_est = self.estimate_full_session_tokens();
        self.status_message = format!(
            "{} compact #{} · ~{} → ~{} tokens (memory updated)",
            if manual { "Manual" } else { "Auto" },
            self.context_compact_count,
            before,
            self.context_tokens_est
        );
        if let Ok(mut l) = self.activity_logs.lock() {
            l.push(format!(
                "[CONTEXT] {} compact #{} — ~{} tok → memory, history cleared",
                if manual { "Manual" } else { "Auto" },
                self.context_compact_count,
                before
            ));
        }
    }

    /// Store tool stdout for LLM + chip terminal; one short line in chat only.
    /// Auto-opens the matching panel for that tool (specific chip, not a generic last one).
    fn record_tool_result_ui(&mut self, kind_hint: &str, full: &str) {
        let pretty = tool_panel::format_tool_output_for_chat(full);
        self.tool_result_context.push(pretty.clone());
        if self.tool_result_context.len() > 8 {
            let n = self.tool_result_context.len() - 8;
            self.tool_result_context.drain(0..n);
        }

        let want_kind = if kind_hint.contains("read") || kind_hint.contains("list") {
            ToolPanelKind::Read
        } else if kind_hint.contains("write") {
            ToolPanelKind::Write
        } else if kind_hint.contains("command") || kind_hint.contains("cmd") {
            ToolPanelKind::Cmd
        } else {
            // Prefer most recent non-write chip, else write
            ToolPanelKind::Cmd
        };

        let anchor = self.latest_agent_msg_idx();
        let mut open_id = None;

        // Prefer chip on the current agent turn with matching kind; else any matching kind.
        let chip_idx = self
            .tool_chips
            .iter()
            .enumerate()
            .rev()
            .find(|(_, c)| c.kind == want_kind && c.anchor_msg == anchor)
            .map(|(i, _)| i)
            .or_else(|| {
                self.tool_chips
                    .iter()
                    .enumerate()
                    .rev()
                    .find(|(_, c)| c.kind == want_kind)
                    .map(|(i, _)| i)
            });

        if let Some(i) = chip_idx {
            let chip = &mut self.tool_chips[i];
            chip.tag_closed = true;
            chip.pending = false;

            open_id = Some(chip.id);
        } else {
            // Result without a prior chip — create one under last agent
            let id = self.next_chip_id;
            self.next_chip_id += 1;
            let target = match want_kind {
                ToolPanelKind::Cmd => "command".into(),
                ToolPanelKind::Read => "file".into(),
                ToolPanelKind::Write => "file".into(),
            };
            self.tool_chips.push(ToolChip {
                id,
                kind: want_kind,
                target,
                body: pretty.clone(),
                tag_closed: true,
                pending: false,
                rect: None,
                anchor_msg: anchor,
            });
            open_id = Some(id);
        }

        self.dedupe_tool_chips();
        if let Some(id) = open_id {
            self.force_open_panel_from_chip(id);
        }
        let lines = pretty.lines().count();
        self.messages
            .push(format!("System: [OK] {kind_hint} finished ({lines} lines)"));
    }

    /// Latest user message only (best for one-shot llama.cpp).
    pub fn last_user_message(&self) -> Option<String> {
        self.messages.iter().rev().find_map(|m| {
            m.strip_prefix("You: ")
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
        })
    }

    fn push_input_undo(&mut self) {
        const MAX: usize = 80;
        self.input_undo
            .push((self.input.clone(), self.input_cursor_position));
        if self.input_undo.len() > MAX {
            self.input_undo.remove(0);
        }
        self.input_redo.clear();
    }

    fn input_undo_apply(&mut self) {
        if let Some((text, cur)) = self.input_undo.pop() {
            self.input_redo
                .push((self.input.clone(), self.input_cursor_position));
            self.input = text;
            self.input_cursor_position = cur.min(self.input.chars().count());
        }
    }

    fn input_redo_apply(&mut self) {
        if let Some((text, cur)) = self.input_redo.pop() {
            self.input_undo
                .push((self.input.clone(), self.input_cursor_position));
            self.input = text;
            self.input_cursor_position = cur.min(self.input.chars().count());
        }
    }

    /// Move cursor left by one word (char-index).
    fn cursor_word_left(&self) -> usize {
        let chars: Vec<char> = self.input.chars().collect();
        let mut i = self.input_cursor_position.min(chars.len());
        if i == 0 {
            return 0;
        }
        i -= 1;
        while i > 0 && chars[i].is_whitespace() {
            i -= 1;
        }
        while i > 0 && !chars[i - 1].is_whitespace() {
            i -= 1;
        }
        i
    }

    fn cursor_word_right(&self) -> usize {
        let chars: Vec<char> = self.input.chars().collect();
        let n = chars.len();
        let mut i = self.input_cursor_position.min(n);
        while i < n && !chars[i].is_whitespace() {
            i += 1;
        }
        while i < n && chars[i].is_whitespace() {
            i += 1;
        }
        i
    }

    fn delete_word_backward(&mut self) {
        let start = self.cursor_word_left();
        let end = self.input_cursor_position.min(self.input.chars().count());
        if start >= end {
            return;
        }
        self.push_input_undo();
        let chars: Vec<char> = self.input.chars().collect();
        let new: String = chars[..start].iter().chain(chars[end..].iter()).collect();
        self.input = new;
        self.input_cursor_position = start;
    }

    pub fn trigger_generation_from_context(&mut self) {
        // 80% of context budget → compress to memory, forget old turns
        self.maybe_compact_context();
        let context_prompt = self.build_context_prompt();
        // Status shows full-session estimate (auto-compact uses this, not trimmed prompt)
        self.context_tokens_est = self.estimate_full_session_tokens();
        let stream_target = self.streaming_response.clone();
        let is_gen = self.is_generating.clone();
        let gen_err = self.generation_error.clone();

        *stream_target.lock().unwrap() = String::new();
        *gen_err.lock().unwrap() = None;
        *is_gen.lock().unwrap() = true;
        self.gen_last_progress = Some(Instant::now());
        self.gen_last_len = 0;

        self.messages.push("Agent: \u{258d}".to_string());
        self.typewriter_len = 0;
        let limit = crate::settings::context_token_limit();
        let pct = if limit > 0 {
            (self.context_tokens_est * 100 / limit).min(100)
        } else {
            0
        };
        self.status_message = format!(
            "Generating via {}… ctx ~{}tok ({}% of {})",
            self.backend.name(),
            self.context_tokens_est,
            pct,
            limit
        );

        match &self.backend {
            AgentBackend::LlamaCppLib(backend) => {
                let backend_clone = backend.clone();
                let is_gen_task = is_gen.clone();
                // After tools run, must pass full context (incl. tool results). Using only
                // last_user_message re-asks "read X" → same tool forever.
                let prompt = if self.auto_tool_turns > 0
                    || !self.tool_result_context.is_empty()
                    || context_prompt.lines().count() > 1
                {
                    context_prompt.clone()
                } else {
                    self.last_user_message()
                        .filter(|m| !m.trim().is_empty())
                        .unwrap_or_else(|| context_prompt.clone())
                };
                let stream_target2 = stream_target.clone();
                tokio::spawn(async move {
                    match backend_clone
                        .generate_stream(&prompt, stream_target2.clone(), is_gen_task)
                        .await
                    {
                        Ok(_) => {}
                        Err(e) => {
                            *gen_err.lock().unwrap() = Some(e);
                        }
                    }
                    *is_gen.lock().unwrap() = false;
                });
            }
            AgentBackend::LlamaCpp(backend) => {
                let backend_clone = backend.clone();
                let is_gen_task = is_gen.clone();
                // Warm server + chat API: pass You:/Agent: history (system injected by HTTP client)
                let prompt = if context_prompt.trim().is_empty() {
                    self.last_user_message().unwrap_or_default()
                } else {
                    context_prompt.clone()
                };
                tokio::spawn(async move {
                    match backend_clone
                        .generate_stream(&prompt, stream_target, is_gen_task)
                        .await
                    {
                        Ok(_) => {}
                        Err(e) => {
                            *gen_err.lock().unwrap() = Some(e);
                        }
                    }
                    *is_gen.lock().unwrap() = false;
                });
            }
            AgentBackend::Ollama(ollama_backend) => {
                let backend_clone = ollama_backend.clone();
                let is_gen_task = is_gen.clone();
                tokio::spawn(async move {
                    match backend_clone
                        .generate_stream(&context_prompt, stream_target, is_gen_task)
                        .await
                    {
                        Ok(_) => {}
                        Err(e) => {
                            *gen_err.lock().unwrap() = Some(e);
                        }
                    }
                    *is_gen.lock().unwrap() = false;
                });
            }
            #[cfg(feature = "gpu")]
            AgentBackend::BurnWgpu(_) => {
                let backend_clone = self.backend.clone();
                let stream_target_clone = stream_target.clone();
                let is_gen_clone = is_gen.clone();
                tokio::spawn(async move {
                    if let Ok(resp) = backend_clone.generate(&context_prompt).await {
                        if let Ok(mut target) = stream_target_clone.lock() {
                            *target = resp;
                        }
                    }
                    *is_gen_clone.lock().unwrap() = false;
                });
            }
        }
    }

    fn input_insert_text(&mut self, text: &str) {
        if text.is_empty() {
            return;
        }

        self.push_input_undo();

        let normalized = text.replace("\r\n", "\n").replace('\r', "\n");

        let mut chars: Vec<char> = self.input.chars().collect();
        let pos = self.input_cursor_position.min(chars.len());

        let inserted_len = normalized.chars().count();

        chars.splice(pos..pos, normalized.chars());

        self.input = chars.into_iter().collect();
        self.input_cursor_position = pos + inserted_len;
    }

    pub async fn handle_events(&mut self) -> Result<(), std::io::Error> {
        self.sys.refresh_cpu_usage();
        self.sys.refresh_memory();

        let now = std::time::Instant::now();
        let delta = now.duration_since(self.last_frame_time);
        self.last_frame_time = now;
        let delta_ms = (delta.as_secs_f64() * 1000.0) as u16;

        self.anim_tick = self.anim_tick.wrapping_add(1);
        self.typewriter_len = self.typewriter_len.saturating_add(3);
        self.krama
            .update_progress(TRES16Bits::from_millis(delta_ms.max(1))); // 60 FPS display synced clock

        if let Some(ref mut panel) = self.tool_panel {
            if let Some(chip) = self.tool_chips.iter().find(|c| {
                c.kind == panel.kind
                    && (c.target == panel.target
                        || panel
                            .target
                            .ends_with(c.target.rsplit('/').next().unwrap_or("")))
            }) {
                if let Some(r) = chip.rect {
                    panel.chip_rect = Some(r);
                }
                // Live-update body while write still streaming
                if panel.kind == ToolPanelKind::Write && !chip.tag_closed {
                    panel.set_body_streaming(chip.body.clone(), chip.tag_closed);
                }
                if panel.kind == ToolPanelKind::Cmd && !chip.body.is_empty() {
                    panel.set_body_streaming(chip.body.clone(), true);
                }
            }
            panel.tick_reveal();
            // Drop panel after reverse fly completes.
            // Krama reverse: progress is negative; abs goes 1→0. At end progress=0
            // and is_reversed becomes false — so track panel_closing ourselves.
            if self.panel_closing {
                let t = self.krama.get_progress_f32("panel_fly", 0).abs();
                if t <= 0.03 {
                    self.tool_panel = None;
                    self.tool_panel_rect = None;
                    self.panel_closing = false;
                }
            }
        }

        // Task manager (long cmds) + generation stall (20s idle)
        self.poll_task_events();
        self.check_generation_stall();

        // Legacy marker (writes / rare paths)
        {
            let mut slot = self.streaming_response.lock().unwrap();
            if slot.starts_with("__HERCULES_TOOL_DONE__\n") {
                let joined = slot["__HERCULES_TOOL_DONE__\n".len()..].to_string();
                slot.clear();
                drop(slot);
                self.record_tool_result_ui("command", &joined);
                self.status_message = "Command finished — terminal open".to_string();
                self.auto_tool_turns += 1;
                if self.auto_tool_turns <= 8 {
                    self.trigger_generation_from_context();
                }
            }
        }

        // Check for streaming response updates
        {
            let current_stream = self.streaming_response.lock().unwrap().clone();
            let is_gen = *self.is_generating.lock().unwrap();
            if is_gen && !current_stream.is_empty() && !current_stream.starts_with("__HERCULES") {
                // Update last message — redact <write>/<cmd> bodies to labels only
                if let Some(last) = self.messages.last_mut() {
                    if last.starts_with("Agent: ") || last.starts_with("Agent: ▍") {
                        let shown = tool_panel::redact_tools_for_chat(&current_stream);
                        *last = format!("Agent: {}", shown);
                    }
                }
                // Chips only while streaming (panel opens from chip click)
                self.sync_tool_chips(&current_stream);

                // ── Furious AlwaysAllow: execute complete writes the instant their
                //    closing tag arrives, without interrupting the AI stream. ──────
                let perms = crate::agent::get_tool_permissions();
                let auto_ok = matches!(perms.mode, PermissionMode::AlwaysAllow) || perms.session_allow;
                if auto_ok {
                    let raw = crate::agent::AgentEngine::extract_proposed_actions(&current_stream);
                    for action in raw {
                        if action.kind != crate::agent::ProposedKind::Write {
                            continue;
                        }
                        // Only act on targets we haven't already written this turn
                        if self.streamed_writes_done.contains(&action.target) {
                            continue;
                        }
                        let result = crate::agent::AgentEngine::execute_proposed(&action);
                        self.streamed_writes_done.push(action.target.clone());
                        // Update chip to [WROTE] without touching tool_result_context
                        // or calling trigger_generation_from_context — AI keeps streaming.
                        if let Some(chip) = self.tool_chips.iter_mut().rev().find(|c| {
                            c.kind == crate::tool_panel::ToolPanelKind::Write
                                && crate::tool_panel::same_tool_target(
                                    crate::tool_panel::ToolPanelKind::Write,
                                    &c.target,
                                    &action.target,
                                )
                        }) {
                            chip.pending = false;
                            chip.tag_closed = true;
                        }
                        if let Ok(mut l) = self.activity_logs.lock() {
                            let ok = !result.starts_with("Error");
                            l.push(format!(
                                "[FURIOUS] mid-stream write {} — {}",
                                action.target,
                                if ok { "OK" } else { &result }
                            ));
                        }
                    }
                }
            }
            if !is_gen {
                // Generation finished
                let gen_err = self.generation_error.lock().unwrap().take();
                let cancelled = self.user_cancelled_gen;
                if cancelled {
                    self.user_cancelled_gen = false;
                    self.auto_tool_turns = 0;
                    *self.streaming_response.lock().unwrap() = String::new();
                    self.gen_last_progress = None;
                    if self.status_message.starts_with("Generating")
                        || self.status_message.contains("via llama")
                    {
                        self.status_message = "Interrupted (CTRL+C).".into();
                    }
                } else if let Some(err) = gen_err {
                    // Recover mid-write / mid-cmd chips so UI is not stuck half-open
                    let partial = self.streaming_response.lock().unwrap().clone();
                    if !partial.is_empty() && !partial.starts_with("__HERCULES") {
                        self.sync_tool_chips(&partial);
                        self.finalize_incomplete_tools("server/generation error");
                        if let Some(last) = self.messages.last_mut() {
                            if last.starts_with("Agent: ") {
                                let shown = tool_panel::redact_tools_for_chat(&partial);
                                *last = format!("Agent: {shown}\n[Interrupted — {err}]");
                            }
                        }
                    } else if let Some(last) = self.messages.last_mut() {
                        if last.starts_with("Agent: ") {
                            *last = format!("Error: {}", err);
                        }
                    }
                    self.status_message = "Generation failed — partial tools recovered.".into();
                    *self.streaming_response.lock().unwrap() = String::new();
                    self.gen_last_progress = None;
                } else if !current_stream.is_empty() && !current_stream.starts_with("__HERCULES") {
                    if let Some(last) = self.messages.last_mut() {
                        if last.starts_with("Agent: ") {
                            let shown = tool_panel::redact_tools_for_chat(&current_stream);
                            *last = format!("Agent: {}", shown);
                            self.typewriter_len = 1000;
                        }
                    }

                    self.sync_tool_chips(&current_stream);

                    let incomplete = Self::has_incomplete_tool_tag(&current_stream);

                    if incomplete {
                        if self.continue_incomplete_tool(&current_stream) {
                            // Do not process the incomplete action in this turn.
                            return Ok(());
                        }

                        self.finalize_incomplete_tools("continuation limit reached");
                    } else {
                        self.incomplete_tool_continuations = 0;
                    }

                    let proposed_raw =
                        crate::agent::AgentEngine::extract_proposed_actions(&current_stream);

                    // Landing page / single HTML ask → one write, not file.txt + 3 html names
                    let proposed = if let Some(user) = self.last_user_message() {
                        crate::agent::AgentEngine::collapse_write_actions_for_user(
                            &user,
                            proposed_raw,
                        )
                    } else {
                        proposed_raw
                    };
                    let perms = get_tool_permissions();
                    let auto_ok =
                        matches!(perms.mode, PermissionMode::AlwaysAllow) || perms.session_allow;
                    let need_accept = !auto_ok && !proposed.is_empty();

                    // read/ls/memory/writes only — cmds never block here
                    let mut effective_stream = current_stream.clone();
                    let mut tool_output_opt =
                        crate::agent::AgentEngine::process_response(&effective_stream);

                    // Recover tools only on the *first* attempt. After we already have
                    // tool results, recovery re-fires the same <read> and wipes the answer.
                    let already_have_tools =
                        !self.tool_result_context.is_empty() || self.auto_tool_turns > 0;
                    if tool_output_opt.is_none() && !already_have_tools {
                        if let Some(user) = self.last_user_message() {
                            if let Some(tag) = crate::agent::AgentEngine::recover_tools_from_refusal(
                                &user,
                                &effective_stream,
                            ) {
                                if let Ok(mut l) = self.activity_logs.lock() {
                                    l.push(format!(
                                        "[HERCULES] tool recovery after model refusal → {tag}"
                                    ));
                                }
                                if let Some(last) = self.messages.last_mut() {
                                    if last.starts_with("Agent: ") {
                                        *last = format!(
                                            "Agent: {tag}\n[Host recovered tool after model refused filesystem access]"
                                        );
                                    }
                                }
                                effective_stream = tag;
                                tool_output_opt =
                                    crate::agent::AgentEngine::process_response(&effective_stream);
                            }
                        }
                    }

                    *self.streaming_response.lock().unwrap() = String::new();
                    self.gen_last_progress = None;

                    // Identical tool tag twice in a row → stop (don't re-read forever)
                    let same_as_prev = self
                        .recent_tool_calls
                        .last()
                        .map(|prev| {
                            crate::settings::normalize_for_repeat(prev)
                                == crate::settings::normalize_for_repeat(&effective_stream)
                        })
                        .unwrap_or(false);

                    let prose = tool_panel::redact_tools_for_chat(&effective_stream);
                    let only_repeated_tool = same_as_prev
                        && crate::agent::AgentEngine::response_has_tool_tags(&effective_stream);
                    // After tools, model answered with empty / tool-only noise → host summary
                    let needs_host_summary = already_have_tools
                        && (only_repeated_tool
                            || prose.trim().is_empty()
                            || crate::agent::AgentEngine::looks_like_capability_refusal(&prose));

                    if !only_repeated_tool {
                        self.recent_tool_calls.push(effective_stream.clone());
                    }
                    let max_hist = crate::settings::get_settings()
                        .repeat_threshold
                        .saturating_mul(3)
                        .max(30);
                    while self.recent_tool_calls.len() > max_hist {
                        self.recent_tool_calls.remove(0);
                    }

                    let settings = crate::settings::get_settings();
                    let loop_hit =
                        crate::settings::detect_repeat_loop(&self.recent_tool_calls, &settings);

                    let cmds: Vec<_> = proposed
                        .iter()
                        .filter(|a| a.kind == crate::agent::ProposedKind::Cmd)
                        .cloned()
                        .collect();
                    let writes_pending: Vec<_> = proposed
                        .iter()
                        .filter(|a| a.kind == crate::agent::ProposedKind::Write)
                        .cloned()
                        .collect();

                    if needs_host_summary {
                        // Do not re-run tools or leave an empty agent bubble.
                        self.host_answer_from_prior_tools();
                        self.auto_tool_turns = 0;
                        self.status_message = "Ready.".to_string();
                        if let Ok(mut l) = self.activity_logs.lock() {
                            l.push("[HERCULES] host summary from prior tool (no re-run)".into());
                        }
                    } else if only_repeated_tool {
                        self.host_answer_from_prior_tools();
                        self.auto_tool_turns = 0;
                        self.status_message = "Ready.".to_string();
                        if let Ok(mut l) = self.activity_logs.lock() {
                            l.push("[REPEAT] identical tool tag — host summary instead".into());
                        }
                    } else if let Some(reason) = loop_hit {
                        self.repeat_count = settings.repeat_threshold;
                        self.messages.push(format!(
                            "System: Repeat detector (threshold {}): {}. \
                             Stop looping — answer the user directly without re-running the same tool.",
                            settings.repeat_threshold, reason
                        ));
                        self.auto_tool_turns = 999;
                        self.status_message = "Repeat loop blocked.".to_string();
                        if let Ok(mut l) = self.activity_logs.lock() {
                            l.push(format!("[REPEAT DETECTOR] {}", reason));
                        }
                        self.recent_tool_calls.clear();
                        self.repeat_count = 0;
                    } else if need_accept {
                        // Ask mode: queue the actions but do NOT interrupt the AI.
                        // The PENDING system message is suppressed while generating;
                        // propose_actions will show it once the stream is done.
                        self.propose_actions(proposed);
                        // Do not auto re-prompt until user accepts/rejects
                    } else {
                        // AlwaysAllow / session: writes already executed mid-stream;
                        // skip any target already in streamed_writes_done to avoid
                        // double-writing, then clear the set for the next turn.
                        let had_cmds = !cmds.is_empty();
                        if had_cmds {
                            self.spawn_cmds_to_task_manager(cmds);
                        }
                        // Only process read/ls/memory output (writes already handled mid-stream)
                        let mut had_tool_out = false;
                        if let Some(tool_output) = tool_output_opt {
                            // Suppress write-only results that were already applied mid-stream
                            let is_write_only = tool_panel::classify_tool_hint(&effective_stream) == "write";
                            let all_done = writes_pending
                                .iter()
                                .all(|w| self.streamed_writes_done.contains(&w.target));
                            if !(is_write_only && all_done) {
                                had_tool_out = true;
                                let hint = tool_panel::classify_tool_hint(&effective_stream);
                                self.record_tool_result_ui(hint, &tool_output);
                                if let Ok(mut l) = self.activity_logs.lock() {
                                    l.push(format!(
                                        "[HERCULES] {hint} done ({} bytes) → chip/terminal",
                                        tool_output.len()
                                    ));
                                }
                            }
                        }
                        // Clear mid-stream dedup set for the next turn
                        self.streamed_writes_done.clear();
                        // Do NOT re-trigger generation for writes — they were already
                        // applied silently. Only re-prompt when reads/cmds produce context.
                        if !had_cmds && self.task_manager.running_count() == 0 {
                            if had_tool_out {
                                self.auto_tool_turns += 1;
                                if self.auto_tool_turns <= 8 {
                                    self.trigger_generation_from_context();
                                } else {
                                    self.messages.push(
                                        "System: [Autonomous loop threshold reached (8 tool turns)]"
                                            .to_string(),
                                    );
                                    self.auto_tool_turns = 0;
                                    self.status_message = "Ready.".to_string();
                                }
                            } else {
                                self.auto_tool_turns = 0;
                                self.status_message = "Ready.".to_string();
                            }
                        } else if had_cmds {
                            self.status_message =
                                "Command(s) in task manager — waiting / Ctrl+C to kill".into();
                        }
                    }
                }
            }
        }

        // Esc hold ≥1s → exit; released early → cancelled in key handler
        if let Some(start) = self.esc_hold_start {
            let pct = (start.elapsed().as_secs_f64() / 1.0).min(1.0);
            if pct >= 1.0 {
                if let Ok(mut g) = self.is_generating.lock() {
                    *g = false;
                }
                crate::llama::server::shutdown_managed_server();
                crate::llama::libinfer::shutdown_warm_lib_engine();
                self.should_quit = true;
            }
        }

        let target_log_pct = if self.log_pane_collapsed { 0.0 } else { 32.0 };
        let factor = ((delta.as_secs_f64() * 18.0) as f64).min(1.0);
        self.current_log_pane_pct += (target_log_pct - self.current_log_pane_pct) * factor;

        let new_results = self.search_results.lock().unwrap().take();
        if let Some(models) = new_results {
            if !models.is_empty() {
                let mut installed = self.manager.list_installed_local();
                let mut search_items = Vec::new();
                for m in models {
                    if m.starts_with("Ollama Local:") || m.starts_with("Local GGUF:") {
                        if !installed.contains(&m) {
                            installed.push(m.clone());
                        }
                    } else {
                        search_items.push(m);
                    }
                }
                self.installed_models = installed;
                self.hf_models = search_items;
                self.krama.restart_progress("list_fade", 0);
            }
        }

        let is_complete = *self.download_complete.lock().unwrap();
        if is_complete {
            *self.download_complete.lock().unwrap() = false;
            *self.download_progress.lock().unwrap() = None;

            // Refresh installed list from ~/.local/hercules/models.toml
            let mut installed = self.manager.list_installed_local();
            for m in self.installed_models.iter() {
                if m.starts_with("Ollama") && !installed.contains(m) {
                    installed.push(m.clone());
                }
            }
            self.installed_models = installed;

            // Prefer llama.cpp for installed GGUF (fast + low RAM vs pure-rust dequant)
            if let Some(path) = self.manager.latest_gguf_path() {
                match self.backend {
                    AgentBackend::LlamaCpp(_) => {
                        self.backend = AgentBackend::LlamaCpp(LlamaCppBackend::cli(path.clone()));
                        self.status_message =
                            format!("Active Engine: llama.cpp ({})", path.display());
                    }
                    _ => {
                        self.backend =
                            AgentBackend::LlamaCppLib(LlamaCppLibBackend::gguf(path.clone()));
                        self.status_message =
                            format!("Active Engine: llama.cpp lib ({})", path.display());
                    }
                }
                self.messages.push(format!(
                    "System: GGUF ready at {}. Activated llama.cpp via warm llama-server \
                     (first message loads weights once; later prompts stay hot). \
                     GPU layers: env HERCULES_N_GPU_LAYERS (default 99, set 0 for CPU).",
                    path.display()
                ));
                if let Ok(mut l) = self.activity_logs.lock() {
                    l.push(format!(
                        "[SYSTEM] Activated llama.cpp (warm server) → {}",
                        path.display()
                    ));
                }
            } else if let Some(entry) = self
                .manager
                .list_installed_entries()
                .into_iter()
                .rev()
                .next()
            {
                if entry.path.starts_with("ollama://") {
                    let name = entry.path.trim_start_matches("ollama://").to_string();
                    self.backend = AgentBackend::Ollama(OllamaBackend::new(name.clone()));
                    self.status_message = format!("Active Engine: Ollama ({})", name);
                    self.messages.push(format!(
                        "System: Ollama model '{}' ready and registered.",
                        name
                    ));
                } else {
                    self.messages.push(format!(
                        "System: Download finished ({}), but it is not a GGUF. \
                         Install a *-GGUF model for llama.rs, or use Ollama.",
                        entry.path
                    ));
                }
            } else {
                self.messages.push(
                    "System: Download flagged complete, but no GGUF found in models.toml. \
                     Check activity logs — base safetensors repos are rejected."
                        .to_string(),
                );
            }
        }

        if event::poll(Duration::from_millis(16))? {
            match event::read()? {
                Event::Mouse(mouse) => match mouse.kind {
                    MouseEventKind::ScrollUp => {
                        let over_panel = self
                            .tool_panel
                            .as_ref()
                            .and_then(|p| p.drawn_rect)
                            .map(|r| {
                                mouse.column >= r.x
                                    && mouse.column < r.x + r.width
                                    && mouse.row >= r.y
                                    && mouse.row < r.y + r.height
                            })
                            .unwrap_or(false);
                        if over_panel {
                            if let Some(ref mut p) = self.tool_panel {
                                p.scroll_by(-3);
                            }
                        } else {
                            self.auto_scroll_enabled = false;
                            self.scroll_offset = self.scroll_offset.saturating_sub(2);
                            self.input_focused = false;
                        }
                    }
                    MouseEventKind::ScrollDown => {
                        let over_panel = self
                            .tool_panel
                            .as_ref()
                            .and_then(|p| p.drawn_rect)
                            .map(|r| {
                                mouse.column >= r.x
                                    && mouse.column < r.x + r.width
                                    && mouse.row >= r.y
                                    && mouse.row < r.y + r.height
                            })
                            .unwrap_or(false);
                        if over_panel {
                            if let Some(ref mut p) = self.tool_panel {
                                p.scroll_by(3);
                            }
                        } else {
                            self.scroll_offset = self.scroll_offset.saturating_add(2);
                            self.input_focused = false;
                        }
                    }
                    MouseEventKind::Down(MouseButton::Left) => {
                        // Chip / chrome first (always clear selection)
                        if let Some(id) =
                            tool_panel::hit_test_chip(&self.tool_chips, mouse.column, mouse.row)
                        {
                            self.clear_selection();
                            self.exit_term_interactive();
                            self.open_panel_from_chip(id);
                        } else if self.tool_panel.is_some() {
                            let chrome = self
                                .tool_panel
                                .as_ref()
                                .map(|p| tool_panel::hit_test_chrome(p, mouse.column, mouse.row))
                                .unwrap_or(PanelChromeHit::None);
                            let (panel_kind, in_body) = self
                                .tool_panel
                                .as_ref()
                                .map(|p| {
                                    let in_b = p.drawn_rect.map(|r| {
                                        mouse.column >= r.x
                                            && mouse.column < r.x + r.width
                                            && mouse.row >= r.y
                                            && mouse.row < r.y + r.height
                                    });
                                    (p.kind, in_b.unwrap_or(false))
                                })
                                .unwrap_or((ToolPanelKind::Cmd, false));
                            match chrome {
                                PanelChromeHit::Close => {
                                    self.clear_selection();
                                    self.exit_term_interactive();
                                    self.close_tool_panel();
                                }
                                PanelChromeHit::Minimize => {
                                    self.clear_selection();
                                    self.toggle_minimize_tool_panel();
                                }
                                PanelChromeHit::None => {
                                    if in_body {
                                        self.clear_selection();
                                        self.input_focused = false;
                                        if panel_kind == ToolPanelKind::Cmd {
                                            self.enter_term_interactive();
                                        } else {
                                            self.exit_term_interactive();
                                            self.status_message =
                                                "WRITE — scroll with mouse wheel / PgUp/PgDn"
                                                    .into();
                                        }
                                    } else {
                                        self.exit_term_interactive();
                                        self.selection_pending_cancel = self.has_selection;
                                        self.selection_start = Some((mouse.column, mouse.row));
                                        self.selection_end = Some((mouse.column, mouse.row));
                                        self.is_selecting = true;
                                        self.selection_dragged = false;
                                        if !self.has_selection {
                                            self.selected_text_buffer.clear();
                                        }
                                        self.input_focused = false;
                                    }
                                }
                            }
                        } else {
                            // Click outside any panel — leave TERM interactive
                            self.exit_term_interactive();
                            self.selection_pending_cancel = self.has_selection;
                            self.selection_start = Some((mouse.column, mouse.row));
                            self.selection_end = Some((mouse.column, mouse.row));
                            self.is_selecting = true;
                            self.selection_dragged = false;
                            if !self.has_selection {
                                self.selected_text_buffer.clear();
                            }
                            self.input_focused = false;
                        }
                    }
                    MouseEventKind::Drag(MouseButton::Left) => {
                        if self.is_selecting {
                            self.selection_end = Some((mouse.column, mouse.row));
                            if let (Some(s), Some(e)) = (self.selection_start, self.selection_end) {
                                if s != e {
                                    self.selection_dragged = true;
                                    self.selection_pending_cancel = false;
                                    // New drag replaces old selection
                                    self.has_selection = false;
                                }
                            }
                            if mouse.row <= 4 {
                                self.scroll_offset = self.scroll_offset.saturating_sub(1);
                                self.auto_scroll_enabled = false;
                            } else if let Some(area) = self.last_chat_area {
                                if mouse.row + 2 >= area.y + area.height {
                                    self.scroll_offset = self.scroll_offset.saturating_add(1);
                                    self.auto_scroll_enabled = false;
                                }
                            }
                        }
                    }
                    MouseEventKind::Up(MouseButton::Left) => {
                        if self.is_selecting {
                            self.selection_end = Some((mouse.column, mouse.row));
                            self.finalize_selection();
                        }
                    }
                    _ => {}
                },

                Event::Paste(text) => {
                    if self.input_focused && !self.show_menu {
                        self.input_insert_text(&text);
                    }
                }

                Event::Key(key) => {
                    use crossterm::event::KeyEventKind;

                    // Esc released early → cancel exit glow (progress back to 0)
                    let skip_key = if key.code == KeyCode::Esc && key.kind == KeyEventKind::Release
                    {
                        if self.esc_hold_start.is_some() {
                            self.esc_hold_start = None;
                            self.status_message = "Exit cancelled.".to_string();
                        }
                        true
                    } else if key.kind == KeyEventKind::Release {
                        // Ignore other key releases
                        true
                    } else {
                        false
                    };

                    if !skip_key {
                        // TERM interactive: keys go to term input (not main prompt)
                        if self.term_is_interactive()
                            && !key.modifiers.contains(KeyModifiers::CONTROL)
                            && self.pending_actions.is_empty()
                            && !self.show_menu
                        {
                            match key.code {
                                KeyCode::Esc => {
                                    self.exit_term_interactive();
                                }
                                KeyCode::Enter => {
                                    let line = std::mem::take(&mut self.term_input);
                                    self.term_run_line(&line);
                                }
                                KeyCode::Backspace => {
                                    self.term_input.pop();
                                }
                                KeyCode::PageUp => {
                                    if let Some(ref mut p) = self.tool_panel {
                                        p.scroll_by(-8);
                                    }
                                }
                                KeyCode::PageDown => {
                                    if let Some(ref mut p) = self.tool_panel {
                                        p.scroll_by(8);
                                    }
                                }
                                KeyCode::Up => {
                                    if let Some(ref mut p) = self.tool_panel {
                                        p.scroll_by(-1);
                                    }
                                }
                                KeyCode::Down => {
                                    if let Some(ref mut p) = self.tool_panel {
                                        p.scroll_by(1);
                                    }
                                }
                                KeyCode::Char(c) => {
                                    self.term_input.push(c);
                                }
                                _ => {}
                            }
                            // skip rest of key handling (TERM owns keyboard)
                        } else if self.tool_panel.is_some()
                            && !self.input_focused
                            && !self.show_menu
                            && matches!(
                                key.code,
                                KeyCode::PageUp | KeyCode::PageDown | KeyCode::Up | KeyCode::Down
                            )
                            && !key.modifiers.contains(KeyModifiers::CONTROL)
                        {
                            // Scroll WRITE/TERM panel without entering interactive
                            if let Some(ref mut p) = self.tool_panel {
                                match key.code {
                                    KeyCode::PageUp => p.scroll_by(-8),
                                    KeyCode::PageDown => p.scroll_by(8),
                                    KeyCode::Up => p.scroll_by(-1),
                                    KeyCode::Down => p.scroll_by(1),
                                    _ => {}
                                }
                            }
                        } else if !self.pending_actions.is_empty()
                            && !key.modifiers.contains(KeyModifiers::CONTROL)
                        {
                            match key.code {
                                KeyCode::Char('y') | KeyCode::Char('Y') | KeyCode::Enter => {
                                    self.accept_pending_actions();
                                }
                                KeyCode::Char('n') | KeyCode::Char('N') => {
                                    self.reject_pending_actions();
                                }
                                KeyCode::Char('a') | KeyCode::Char('A') => {
                                    set_permission_mode(PermissionMode::AlwaysAllow);
                                    allow_session_tools();
                                    self.accept_pending_actions();
                                    self.status_message =
                                    "Always allow + accepted. Writes/cmds auto-run this session."
                                        .to_string();
                                }
                                _ => {}
                            }
                            // Don't also type Y into input / other handlers for these keys
                            if matches!(
                                key.code,
                                KeyCode::Char('y')
                                    | KeyCode::Char('Y')
                                    | KeyCode::Char('n')
                                    | KeyCode::Char('N')
                                    | KeyCode::Char('a')
                                    | KeyCode::Char('A')
                                    | KeyCode::Enter
                            ) {
                                // fall through only for non-accept keys below — skip rest
                            } else {
                                // other keys while pending still work
                            }
                        }

                        let pending_consumed = !self.pending_actions.is_empty()
                            && matches!(
                                key.code,
                                KeyCode::Char('y')
                                    | KeyCode::Char('Y')
                                    | KeyCode::Char('n')
                                    | KeyCode::Char('N')
                                    | KeyCode::Char('a')
                                    | KeyCode::Char('A')
                                    | KeyCode::Enter
                            )
                            && !key.modifiers.contains(KeyModifiers::CONTROL);

                        // Pressing anything else while not Esc cancels hold
                        if key.code != KeyCode::Esc {
                            self.esc_hold_start = None;
                        }

                        if pending_consumed {
                            // already handled accept/reject
                        } else {
                            match key.code {
                                KeyCode::Esc => {
                                    // Ctrl+Esc → exit immediately (no hold)
                                    if key.modifiers.contains(KeyModifiers::CONTROL) {
                                        self.esc_hold_start = None;
                                        if let Ok(mut g) = self.is_generating.lock() {
                                            *g = false;
                                        }
                                        crate::llama::server::shutdown_managed_server();
                                        crate::llama::libinfer::shutdown_warm_lib_engine();
                                        self.should_quit = true;
                                        self.status_message = "Exiting…".to_string();
                                    } else if self.delete_confirm_model.is_some() {
                                        self.delete_confirm_model = None;
                                        self.esc_hold_start = None;
                                    } else if self.show_menu {
                                        self.show_menu = false;
                                        self.esc_hold_start = None;
                                    } else {
                                        // Start / continue hold-to-exit (1s)
                                        if self.esc_hold_start.is_none() {
                                            self.esc_hold_start = Some(std::time::Instant::now());
                                            self.status_message =
                                        "Hold Esc to exit… (release to cancel) | Ctrl+Esc = quit now"
                                            .to_string();
                                        }
                                    }
                                }
                                KeyCode::Char('l') | KeyCode::Char('L')
                                    if key.modifiers.contains(KeyModifiers::CONTROL) =>
                                {
                                    self.log_pane_collapsed = !self.log_pane_collapsed;
                                }
                                KeyCode::F(3) => {
                                    self.log_pane_collapsed = !self.log_pane_collapsed;
                                }
                                KeyCode::Char('f') | KeyCode::Char('F')
                                    if key.modifiers.contains(KeyModifiers::CONTROL) =>
                                {
                                    self.input_focused = !self.input_focused;
                                }
                                KeyCode::Char('m') | KeyCode::Char('M')
                                    if key.modifiers.contains(KeyModifiers::CONTROL) =>
                                {
                                    self.show_menu = !self.show_menu;
                                    if self.show_menu {
                                        self.krama.restart_progress("menu_fade", 0);
                                        self.krama.restart_progress("list_fade", 0);
                                        let manager_clone = self.manager.clone();
                                        let search_results_clone = self.search_results.clone();
                                        tokio::spawn(async move {
                                            let mut combined = Vec::new();
                                            if let Ok(ollama_models) =
                                                manager_clone.list_ollama_models().await
                                            {
                                                for m in ollama_models {
                                                    let sz = if m.size > 0 {
                                                        crate::manager::format_model_size(m.size)
                                                    } else {
                                                        "?".into()
                                                    };
                                                    combined.push(format!(
                                                        "Ollama Local: {} ({sz})",
                                                        m.name
                                                    ));
                                                }
                                            }
                                            let hf =
                                                manager_clone.search_all_models("deepseek").await;
                                            combined.extend(hf);
                                            *search_results_clone.lock().unwrap() = Some(combined);
                                        });
                                    }
                                }
                                KeyCode::F(2) => {
                                    self.show_menu = !self.show_menu;
                                    if self.show_menu {
                                        self.krama.restart_progress("menu_fade", 0);
                                        self.krama.restart_progress("list_fade", 0);
                                        let manager_clone = self.manager.clone();
                                        let search_results_clone = self.search_results.clone();
                                        tokio::spawn(async move {
                                            let mut combined = Vec::new();
                                            if let Ok(ollama_models) =
                                                manager_clone.list_ollama_models().await
                                            {
                                                for m in ollama_models {
                                                    let sz = if m.size > 0 {
                                                        crate::manager::format_model_size(m.size)
                                                    } else {
                                                        "?".into()
                                                    };
                                                    combined.push(format!(
                                                        "Ollama Local: {} ({sz})",
                                                        m.name
                                                    ));
                                                }
                                            }
                                            let hf =
                                                manager_clone.search_all_models("deepseek").await;
                                            combined.extend(hf);
                                            *search_results_clone.lock().unwrap() = Some(combined);
                                        });
                                    }
                                }
                                KeyCode::F(1) => {
                                    self.show_shortcuts = !self.show_shortcuts;
                                    if self.show_shortcuts {
                                        self.krama.restart_progress("help_fade", 0);
                                        self.status_message =
                                            "Key shortcuts visible (F1 to hide)".to_string();
                                    } else {
                                        self.status_message = "Key shortcuts hidden".to_string();
                                    }
                                }
                                KeyCode::Left => {
                                    if self.show_menu {
                                        self.menu_section = if self.menu_section == 0 {
                                            4
                                        } else {
                                            self.menu_section - 1
                                        };
                                    } else if self.input_focused {
                                        if key.modifiers.contains(KeyModifiers::ALT) {
                                            self.input_cursor_position = self.cursor_word_left();
                                        } else {
                                            self.input_cursor_position =
                                                self.input_cursor_position.saturating_sub(1);
                                        }
                                    }
                                }
                                KeyCode::Right => {
                                    if self.show_menu {
                                        self.menu_section = (self.menu_section + 1) % 5;
                                    } else if self.input_focused {
                                        if key.modifiers.contains(KeyModifiers::ALT) {
                                            self.input_cursor_position = self.cursor_word_right();
                                        } else {
                                            self.input_cursor_position =
                                                (self.input_cursor_position + 1)
                                                    .min(self.input.chars().count());
                                        }
                                    }
                                }
                                KeyCode::Home => {
                                    if self.input_focused && !self.show_menu {
                                        self.input_cursor_position = 0;
                                    }
                                }
                                KeyCode::End => {
                                    if self.input_focused && !self.show_menu {
                                        self.input_cursor_position = self.input.chars().count();
                                    }
                                }
                                KeyCode::Up => {
                                    if self.show_menu {
                                        if self.menu_section == 0 {
                                            let total =
                                                self.registry_models.len() + self.hf_models.len();
                                            let i = match self.registry_state.selected() {
                                                Some(i) => {
                                                    if i == 0 {
                                                        total.saturating_sub(1)
                                                    } else {
                                                        i - 1
                                                    }
                                                }
                                                None => 0,
                                            };
                                            self.registry_state.select(Some(i));
                                        } else if self.menu_section == 1 {
                                            let i = match self.installed_state.selected() {
                                                Some(i) => {
                                                    if i == 0 {
                                                        self.installed_models
                                                            .len()
                                                            .saturating_sub(1)
                                                    } else {
                                                        i - 1
                                                    }
                                                }
                                                None => 0,
                                            };
                                            self.installed_state.select(Some(i));
                                        } else if self.menu_section == 2 {
                                            const CONFIG_LEN: usize = 3; // llama.rs | llama.cpp | Ollama
                                            let i = match self.config_state.selected() {
                                                Some(i) => {
                                                    if i == 0 {
                                                        CONFIG_LEN - 1
                                                    } else {
                                                        i - 1
                                                    }
                                                }
                                                None => 0,
                                            };
                                            self.config_state.select(Some(i));
                                        } else if self.menu_section == 3 {
                                            const RT_LEN: usize = 8; // power×3 + sub_backend + repeat + think + ctx + temp
                                            let i = match self.runtime_state.selected() {
                                                Some(i) => {
                                                    if i == 0 {
                                                        RT_LEN - 1
                                                    } else {
                                                        i - 1
                                                    }
                                                }
                                                None => 0,
                                            };
                                            self.runtime_state.select(Some(i));
                                        } else {
                                            // Permissions tab
                                            const PERMS_LEN: usize = 4;
                                            let i = match self.perms_state.selected() {
                                                Some(i) => {
                                                    if i == 0 {
                                                        PERMS_LEN - 1
                                                    } else {
                                                        i - 1
                                                    }
                                                }
                                                None => 0,
                                            };
                                            self.perms_state.select(Some(i));
                                        }
                                    } else if !self.input_focused {
                                        self.scroll_offset = self.scroll_offset.saturating_sub(1);
                                        self.auto_scroll_enabled = false;
                                    }
                                }
                                KeyCode::Down => {
                                    if self.show_menu {
                                        if self.menu_section == 0 {
                                            let total =
                                                self.registry_models.len() + self.hf_models.len();
                                            let i = match self.registry_state.selected() {
                                                Some(i) => {
                                                    if i >= total.saturating_sub(1) {
                                                        0
                                                    } else {
                                                        i + 1
                                                    }
                                                }
                                                None => 0,
                                            };
                                            self.registry_state.select(Some(i));
                                        } else if self.menu_section == 1 {
                                            let i = match self.installed_state.selected() {
                                                Some(i) => {
                                                    if i >= self
                                                        .installed_models
                                                        .len()
                                                        .saturating_sub(1)
                                                    {
                                                        0
                                                    } else {
                                                        i + 1
                                                    }
                                                }
                                                None => 0,
                                            };
                                            self.installed_state.select(Some(i));
                                        } else if self.menu_section == 2 {
                                            const CONFIG_LEN: usize = 3; // llama.rs | llama.cpp | Ollama
                                            let i = match self.config_state.selected() {
                                                Some(i) => {
                                                    if i >= CONFIG_LEN - 1 {
                                                        0
                                                    } else {
                                                        i + 1
                                                    }
                                                }
                                                None => 0,
                                            };
                                            self.config_state.select(Some(i));
                                        } else if self.menu_section == 3 {
                                            const RT_LEN: usize = 8;
                                            let i = match self.runtime_state.selected() {
                                                Some(i) => {
                                                    if i >= RT_LEN - 1 {
                                                        0
                                                    } else {
                                                        i + 1
                                                    }
                                                }
                                                None => 0,
                                            };
                                            self.runtime_state.select(Some(i));
                                        } else {
                                            const PERMS_LEN: usize = 4;
                                            let i = match self.perms_state.selected() {
                                                Some(i) => {
                                                    if i >= PERMS_LEN - 1 {
                                                        0
                                                    } else {
                                                        i + 1
                                                    }
                                                }
                                                None => 0,
                                            };
                                            self.perms_state.select(Some(i));
                                        }
                                    } else if !self.input_focused {
                                        self.scroll_offset = self.scroll_offset.saturating_add(1);
                                    }
                                }
                                KeyCode::PageUp if !self.input_focused => {
                                    self.scroll_offset = self.scroll_offset.saturating_sub(5);
                                }
                                KeyCode::PageDown if !self.input_focused => {
                                    self.scroll_offset = self.scroll_offset.saturating_add(5);
                                }
                                KeyCode::Char('c')
                                    if key.modifiers.contains(KeyModifiers::CONTROL) =>
                                {
                                    // Selection wins: Ctrl+C copies shaded text
                                    if self.selection_active() {
                                        if !self.copy_selection_to_clipboard() {
                                            self.status_message = "Nothing to copy".into();
                                        }
                                    } else {
                                        // Interrupt generation + kill background tasks
                                        let was_gen = *self.is_generating.lock().unwrap();
                                        let n_tasks = self.task_manager.running_count();
                                        if was_gen {
                                            // Signal cancel first; do not race process_response / re-prompt.
                                            self.user_cancelled_gen = true;
                                            self.auto_tool_turns = 0;
                                            *self.is_generating.lock().unwrap() = false;
                                            let partial = {
                                                let mut target =
                                                    self.streaming_response.lock().unwrap();
                                                if !target.starts_with("__HERCULES") {
                                                    target.push_str(
                                                "\n[Generation Interrupted by User (CTRL+C)]",
                                            );
                                                }
                                                target.clone()
                                            };
                                            self.gen_last_progress = None;
                                            if !partial.is_empty()
                                                && !partial.starts_with("__HERCULES")
                                            {
                                                self.sync_tool_chips(&partial);
                                                self.finalize_incomplete_tools("Ctrl+C");
                                            }
                                        }
                                        if n_tasks > 0 {
                                            self.task_manager.kill_all();
                                            self.messages.push(format!(
                                        "System: [CTRL+C] killed {n_tasks} background task(s)"
                                    ));
                                        }
                                        self.auto_tool_turns = 0;
                                        if was_gen || n_tasks > 0 {
                                            self.status_message = format!(
                                                "Interrupted (CTRL+C) — gen={} tasks_killed={}",
                                                was_gen, n_tasks
                                            );
                                            if let Ok(mut l) = self.activity_logs.lock() {
                                                l.push(format!(
                                                    "[CANCEL] CTRL+C gen={was_gen} tasks={n_tasks}"
                                                ));
                                            }
                                        } else {
                                            self.input.clear();
                                            self.input_cursor_position = 0;
                                            self.status_message = "Input cleared.".to_string();
                                        }
                                    }
                                }
                                KeyCode::Char('t') | KeyCode::Char('T')
                                    if key.modifiers.contains(KeyModifiers::CONTROL) =>
                                {
                                    self.thinking_collapsed = !self.thinking_collapsed;
                                    self.status_message = format!(
                                        "Thinking block: {}",
                                        if self.thinking_collapsed {
                                            "Collapsed"
                                        } else {
                                            "Expanded"
                                        }
                                    );
                                    if let Ok(mut l) = self.activity_logs.lock() {
                                        l.push(format!(
                                            "[UI] Thinking block toggled to {}",
                                            if self.thinking_collapsed {
                                                "Collapsed"
                                            } else {
                                                "Expanded"
                                            }
                                        ));
                                    }
                                }
                                // Panel chrome only when input unfocused (mouse is primary open)
                                KeyCode::Char('x') | KeyCode::Char('X')
                                    if self.tool_panel.is_some()
                                        && !self.input_focused
                                        && !self.show_menu
                                        && !key.modifiers.contains(KeyModifiers::CONTROL) =>
                                {
                                    self.close_tool_panel();
                                }
                                // Runtime menu: +/- adjust ctx / repeat threshold
                                KeyCode::Char('+')
                                | KeyCode::Char('=')
                                | KeyCode::Char('-')
                                | KeyCode::Char('_')
                                    if self.show_menu
                                        && self.menu_section == 3
                                        && !key.modifiers.contains(KeyModifiers::CONTROL) =>
                                {
                                    let inc =
                                        matches!(key.code, KeyCode::Char('+') | KeyCode::Char('='));
                                    self.runtime_nudge_selected(if inc { 1 } else { -1 });
                                }
                                KeyCode::Char('-') | KeyCode::Char('_')
                                    if self.tool_panel.is_some()
                                        && !self.input_focused
                                        && !self.show_menu
                                        && !key.modifiers.contains(KeyModifiers::CONTROL) =>
                                {
                                    self.toggle_minimize_tool_panel();
                                }

                                KeyCode::Backspace | KeyCode::Char('h')
                                    if key.modifiers.contains(KeyModifiers::CONTROL)
                                        || key.modifiers.contains(KeyModifiers::ALT) =>
                                {
                                    // Ctrl+Backspace or Alt+Backspace: delete previous word
                                    if self.input_focused && !self.show_menu {
                                        self.delete_word_backward();
                                    }
                                }
                                KeyCode::Char('z') | KeyCode::Char('Z')
                                    if key.modifiers.contains(KeyModifiers::CONTROL)
                                        && !key.modifiers.contains(KeyModifiers::SHIFT) =>
                                {
                                    if self.input_focused && !self.show_menu {
                                        self.input_undo_apply();
                                    }
                                }
                                KeyCode::Char('y') | KeyCode::Char('Y')
                                    if key.modifiers.contains(KeyModifiers::CONTROL)
                                        && self.delete_confirm_model.is_none() =>
                                {
                                    if self.input_focused && !self.show_menu {
                                        self.input_redo_apply();
                                    }
                                }
                                KeyCode::Char('z') | KeyCode::Char('Z')
                                    if key.modifiers.contains(KeyModifiers::CONTROL)
                                        && key.modifiers.contains(KeyModifiers::SHIFT) =>
                                {
                                    if self.input_focused && !self.show_menu {
                                        self.input_redo_apply();
                                    }
                                }
                                KeyCode::Char('y') | KeyCode::Char('Y')
                                    if self.delete_confirm_model.is_some() =>
                                {
                                    if let Some(target) = self.delete_confirm_model.take() {
                                        if target.contains("Local GGUF:")
                                            || target.contains(".gguf")
                                        {
                                            if let Err(e) = self.manager.delete_local_model(&target)
                                            {
                                                if let Ok(mut l) = self.activity_logs.lock() {
                                                    l.push(format!("[DELETE ERROR] {}", e));
                                                }
                                            } else if let Ok(mut l) = self.activity_logs.lock() {
                                                l.push(format!(
                                            "[DELETE] Removed from models.toml and disk: {}",
                                            target
                                        ));
                                            }
                                        }
                                        self.installed_models.retain(|m| m != &target);
                                        self.status_message =
                                            format!("Model '{}' deleted successfully.", target);
                                        if let Ok(mut l) = self.activity_logs.lock() {
                                            l.push(format!(
                                                "[DELETE] User confirmed deletion of model '{}'",
                                                target
                                            ));
                                        }
                                    }
                                }
                                KeyCode::Char('n') | KeyCode::Char('N')
                                    if self.delete_confirm_model.is_some() =>
                                {
                                    self.delete_confirm_model = None;
                                }
                                // Ctrl+J = newline (classic terminal multiline)
                                KeyCode::Char('j') | KeyCode::Char('J')
                                    if key.modifiers.contains(KeyModifiers::CONTROL)
                                        && self.input_focused
                                        && !self.show_menu =>
                                {
                                    self.input_insert_char('\n');
                                }
                                KeyCode::Char(c) => {
                                    if !key.modifiers.contains(KeyModifiers::CONTROL)
                                        && !key.modifiers.contains(KeyModifiers::ALT)
                                    {
                                        if self.show_menu && self.menu_section == 0 {
                                            self.registry_search_query.push(c);
                                            let query = self.registry_search_query.clone();
                                            let manager = self.manager.clone();
                                            let results = self.search_results.clone();
                                            tokio::spawn(async move {
                                                let matches =
                                                    manager.search_all_models(&query).await;
                                                *results.lock().unwrap() = Some(matches);
                                            });
                                        } else if self.input_focused && !self.show_menu {
                                            // Paste / typed text may include newlines
                                            if c == '\n' || c == '\r' {
                                                self.input_insert_char('\n');
                                            } else {
                                                self.input_insert_char(c);
                                            }
                                        }
                                    }
                                }
                                KeyCode::Backspace => {
                                    if key.modifiers.contains(KeyModifiers::ALT)
                                        || key.modifiers.contains(KeyModifiers::CONTROL)
                                    {
                                        // handled above for word-delete
                                    } else if self.show_menu && self.menu_section == 0 {
                                        self.registry_search_query.pop();
                                        let query = self.registry_search_query.clone();
                                        let manager = self.manager.clone();
                                        let results = self.search_results.clone();
                                        tokio::spawn(async move {
                                            let matches = manager.search_all_models(&query).await;
                                            *results.lock().unwrap() = Some(matches);
                                        });
                                    } else if self.input_focused && !self.show_menu {
                                        if self.input_cursor_position > 0 && !self.input.is_empty()
                                        {
                                            self.push_input_undo();
                                            let pos = self
                                                .input_cursor_position
                                                .min(self.input.chars().count());
                                            let mut chars: Vec<char> = self.input.chars().collect();
                                            chars.remove(pos - 1);
                                            self.input = chars.into_iter().collect();
                                            self.input_cursor_position -= 1;
                                        }
                                    }
                                }
                                KeyCode::Delete => {
                                    if self.show_menu && self.menu_section == 1 {
                                        if let Some(idx) = self.installed_state.selected() {
                                            if idx < self.installed_models.len() {
                                                self.delete_confirm_model =
                                                    Some(self.installed_models[idx].clone());
                                            }
                                        }
                                    } else if self.input_focused && !self.show_menu {
                                        let len = self.input.chars().count();
                                        if self.input_cursor_position < len {
                                            self.push_input_undo();
                                            let mut chars: Vec<char> = self.input.chars().collect();
                                            chars.remove(self.input_cursor_position);
                                            self.input = chars.into_iter().collect();
                                        }
                                    }
                                }
                                KeyCode::Enter => {
                                    // Multiline: Shift+Enter / Alt+Enter → newline
                                    // Plain Enter → send · Ctrl+Enter → force send / interrupt
                                    // Ctrl+J also inserts newline (handled separately)
                                    let want_newline = self.input_focused
                                        && !self.show_menu
                                        && (key.modifiers.contains(KeyModifiers::SHIFT)
                                            || key.modifiers.contains(KeyModifiers::ALT));

                                    if want_newline {
                                        self.input_insert_char('\n');
                                    } else if self.show_menu {
                                        if self.menu_section == 0 {
                                            let all_items: Vec<String> = self
                                                .registry_models
                                                .iter()
                                                .cloned()
                                                .chain(self.hf_models.iter().cloned())
                                                .collect();
                                            if let Some(i) = self.registry_state.selected() {
                                                if i < all_items.len() {
                                                    let item_str = all_items[i].clone();
                                                    if item_str.starts_with("Ollama:")
                                                        || item_str.starts_with("Ollama Local:")
                                                    {
                                                        let ollama_name = item_str
                                                            .replace("Ollama:", "")
                                                            .replace("Ollama Local:", "")
                                                            .split('(')
                                                            .next()
                                                            .unwrap_or(&item_str)
                                                            .trim()
                                                            .to_string();

                                                        self.status_message = format!(
                                                            "Pulling Ollama Model {}",
                                                            ollama_name
                                                        );
                                                        self.messages.push(format!(
                                                            "System: Pulling Ollama Model: {}",
                                                            ollama_name
                                                        ));
                                                        if let Ok(mut l) = self.activity_logs.lock()
                                                        {
                                                            l.push(format!("[OLLAMA] Initiated pull stream for model: {}", ollama_name));
                                                        }

                                                        let progress_clone =
                                                            self.download_progress.clone();
                                                        let complete_clone =
                                                            self.download_complete.clone();
                                                        let logs_clone = self.activity_logs.clone();
                                                        let manager_clone = self.manager.clone();

                                                        tokio::spawn(async move {
                                                            let res = manager_clone
                                                                .download_ollama_model(
                                                                    &ollama_name,
                                                                    progress_clone,
                                                                    logs_clone,
                                                                )
                                                                .await;
                                                            if res.is_ok() {
                                                                *complete_clone.lock().unwrap() =
                                                                    true;
                                                            } else {
                                                                *complete_clone.lock().unwrap() =
                                                                    false;
                                                            }
                                                        });
                                                    } else {
                                                        // Strip "HuggingFace:" and size tags like "[~1.9 GB Q4 est.]"
                                                        let repo_id = {
                                                            let s = item_str
                                                                .replace("HuggingFace:", "")
                                                                .trim()
                                                                .to_string();
                                                            let s = s
                                                                .split('[')
                                                                .next()
                                                                .unwrap_or(&s)
                                                                .trim();
                                                            s.split_whitespace()
                                                                .next()
                                                                .unwrap_or(s)
                                                                .to_string()
                                                        };

                                                        self.status_message = format!(
                                                            "Resolving weights for {}",
                                                            repo_id
                                                        );
                                                        self.messages.push(format!("System: Resolving model weights and initiating download for: {}", repo_id));
                                                        if let Ok(mut l) = self.activity_logs.lock()
                                                        {
                                                            l.push(format!("[USER] Initiated download for HuggingFace model: {}", repo_id));
                                                        }

                                                        let progress_clone =
                                                            self.download_progress.clone();
                                                        let complete_clone =
                                                            self.download_complete.clone();
                                                        let logs_clone = self.activity_logs.clone();
                                                        let manager_clone = self.manager.clone();

                                                        tokio::spawn(async move {
                                                            // Resolve GGUF only (may remap repo to *-GGUF mirror)
                                                            let resolved = manager_clone
                                                                .resolve_gguf_file(&repo_id)
                                                                .await;
                                                            match resolved {
                                                                Ok((
                                                                    dl_repo,
                                                                    weight_filename,
                                                                    shard_files,
                                                                )) => {
                                                                    if let Ok(mut l) =
                                                                        logs_clone.lock()
                                                                    {
                                                                        if dl_repo
                                                                            != repo_id
                                                                                .split('[')
                                                                                .next()
                                                                                .unwrap_or(&repo_id)
                                                                                .trim()
                                                                        {
                                                                            l.push(format!(
                                                                        "[RESOLVE] No GGUF in '{}'; using mirror repo '{}'",
                                                                        repo_id, dl_repo
                                                                    ));
                                                                        }
                                                                        l.push(format!(
                                                                    "[RESOLVE] Selected GGUF: {}/{}",
                                                                    dl_repo, weight_filename
                                                                ));
                                                                    }
                                                                    let progress_for_dl =
                                                                        progress_clone.clone();
                                                                    let res = manager_clone
                                                                        .download_hf_model(
                                                                            &dl_repo,
                                                                            &weight_filename,
                                                                            &shard_files,
                                                                            progress_for_dl,
                                                                            logs_clone.clone(),
                                                                        )
                                                                        .await;
                                                                    match res {
                                                                        Ok(path) => {
                                                                            if let Ok(mut l) =
                                                                                logs_clone.lock()
                                                                            {
                                                                                l.push(format!(
                                                                            "[SUCCESS] Installed GGUF at {}",
                                                                            path.display()
                                                                        ));
                                                                            }
                                                                            *complete_clone
                                                                                .lock()
                                                                                .unwrap() = true;
                                                                        }
                                                                        Err(e) => {
                                                                            if let Ok(mut l) =
                                                                                logs_clone.lock()
                                                                            {
                                                                                l.push(format!(
                                                                                    "[ERROR] {}",
                                                                                    e
                                                                                ));
                                                                            }
                                                                            *complete_clone
                                                                                .lock()
                                                                                .unwrap() = false;
                                                                            *progress_clone
                                                                                .lock()
                                                                                .unwrap() = None;
                                                                        }
                                                                    }
                                                                }
                                                                Err(e) => {
                                                                    if let Ok(mut l) =
                                                                        logs_clone.lock()
                                                                    {
                                                                        l.push(format!(
                                                                            "[RESOLVE ERROR] {}",
                                                                            e
                                                                        ));
                                                                    }
                                                                    *complete_clone
                                                                        .lock()
                                                                        .unwrap() = false;
                                                                    *progress_clone
                                                                        .lock()
                                                                        .unwrap() = None;
                                                                }
                                                            }
                                                        });
                                                    }
                                                    self.show_menu = false;
                                                }
                                            }
                                        } else if self.menu_section == 1 {
                                            if let Some(i) = self.installed_state.selected() {
                                                if i < self.installed_models.len() {
                                                    let selected_model =
                                                        self.installed_models[i].clone();
                                                    if selected_model.contains("Ollama") {
                                                        let model_name = selected_model
                                                            .replace("Local Installed:", "")
                                                            .replace("Ollama:", "")
                                                            .replace("Ollama Local:", "")
                                                            .split('(')
                                                            .next()
                                                            .unwrap_or("llama3.2")
                                                            .trim()
                                                            .to_string();
                                                        self.backend = AgentBackend::Ollama(
                                                            OllamaBackend::new(model_name.clone()),
                                                        );
                                                        self.status_message = format!(
                                                            "Active Engine: Ollama ({})",
                                                            model_name
                                                        );
                                                    } else if selected_model.contains("Local GGUF:")
                                                        || selected_model
                                                            .to_lowercase()
                                                            .contains(".gguf")
                                                        || selected_model.contains("GGUF")
                                                    {
                                                        // Path is recorded in display as [...path] or in models.toml
                                                        let path =
                                                            selected_model
                                                                .rfind('[')
                                                                .and_then(|i| {
                                                                    let rest =
                                                                        &selected_model[i + 1..];
                                                                    rest.strip_suffix(']')
                                                                        .map(|s| s.to_string())
                                                                })
                                                                .map(std::path::PathBuf::from)
                                                                .filter(|p| p.exists())
                                                                .or_else(|| {
                                                                    self.manager
                                                            .list_installed_entries()
                                                            .into_iter()
                                                            .find(|e| {
                                                                selected_model.contains(&e.name)
                                                                    || selected_model
                                                                        .contains(&e.path)
                                                            })
                                                            .map(|e| {
                                                                std::path::PathBuf::from(e.path)
                                                            })
                                                            .filter(|p| p.exists())
                                                                });

                                                        if let Some(path) = path {
                                                            match self.backend {
                                                                AgentBackend::LlamaCpp(_) => {
                                                                    self.backend =
                                                                        AgentBackend::LlamaCpp(
                                                                            LlamaCppBackend::cli(
                                                                                path.clone(),
                                                                            ),
                                                                        );
                                                                    self.status_message = format!(
                                                                        "Active Engine: llama.cpp ({})",
                                                                        path.display()
                                                                    );
                                                                }
                                                                _ => {
                                                                    self.backend = AgentBackend::LlamaCppLib(
                                                                LlamaCppLibBackend::gguf(path.clone()),
                                                            );
                                                                    self.status_message = format!(
                                                                        "Active Engine: llama.cpp lib ({})",
                                                                        path.display()
                                                                    );
                                                                }
                                                            }
                                                        } else {
                                                            self.backend = AgentBackend::LlamaCpp(
                                                                LlamaCppBackend::server(
                                                                    "http://localhost:8080"
                                                                        .to_string(),
                                                                    selected_model.clone(),
                                                                ),
                                                            );
                                                            self.status_message = format!(
                                                                "Active Engine: llama.cpp server for '{}' (file not found on disk)",
                                                                selected_model
                                                            );
                                                        }
                                                    } else if let Some(path) =
                                                        self.manager.latest_gguf_path()
                                                    {
                                                        // Default activate as llama.rs pure-Rust
                                                        self.backend = AgentBackend::LlamaCppLib(
                                                            LlamaCppLibBackend::gguf(path.clone()),
                                                        );
                                                        self.status_message = format!(
                                                            "Active Engine: llama.cpp lib ({})",
                                                            path.display()
                                                        );
                                                    } else {
                                                        self.status_message = format!(
                                                            "No local GGUF for '{}'",
                                                            selected_model
                                                        );
                                                    }
                                                    self.messages.push(format!("System: Switched active engine model to '{}'", selected_model));
                                                    self.initialized = false;
                                                    self.init_triggered = false;
                                                    self.show_menu = false;
                                                }
                                            }
                                        } else if self.menu_section == 2 {
                                            if let Some(i) = self.config_state.selected() {
                                                let current_path = self
                                                    .backend
                                                    .current_model_path()
                                                    .or_else(|| self.manager.latest_gguf_path());
                                                match i {
                                                    0 => {
                                                        // llama.rs pure Rust
                                                        if let Some(path) = current_path {
                                                            self.backend =
                                                                AgentBackend::LlamaCppLib(
                                                                    LlamaCppLibBackend::gguf(
                                                                        path.clone(),
                                                                    ),
                                                                );
                                                            self.status_message = format!(
                                                                "Active Engine: llama.cpp lib ({})",
                                                                path.display()
                                                            );
                                                        } else {
                                                            self.backend =
                                                                AgentBackend::LlamaCppLib(
                                                                    LlamaCppLibBackend::http(
                                                                        "http://localhost:8080"
                                                                            .to_string(),
                                                                        "llama.rs".to_string(),
                                                                    ),
                                                                );
                                                            self.status_message = "Active Engine: llama.cpp lib HTTP :8080 (download a GGUF first)".to_string();
                                                        }
                                                    }
                                                    1 => {
                                                        // llama.cpp
                                                        if let Some(path) = current_path {
                                                            self.backend = AgentBackend::LlamaCpp(
                                                                LlamaCppBackend::cli(path.clone()),
                                                            );
                                                            self.status_message = format!(
                                                                "Active Engine: llama.cpp ({})",
                                                                path.display()
                                                            );
                                                        } else {
                                                            self.backend = AgentBackend::LlamaCpp(
                                                                LlamaCppBackend::server(
                                                                    "http://localhost:8080"
                                                                        .to_string(),
                                                                    "llama.cpp".to_string(),
                                                                ),
                                                            );
                                                            self.status_message = "Active Engine: llama.cpp server :8080 (no local GGUF)".to_string();
                                                        }
                                                    }
                                                    _ => {
                                                        self.backend = AgentBackend::Ollama(
                                                            OllamaBackend::new(
                                                                "llama3.2:latest".to_string(),
                                                            ),
                                                        );
                                                        self.status_message =
                                                    "Active Engine: Ollama (http://localhost:11434)"
                                                        .to_string();
                                                    }
                                                }
                                                self.messages.push(format!(
                                                    "System: Backend switched to {}",
                                                    self.backend.name()
                                                ));
                                                self.initialized = false;
                                                self.init_triggered = false;
                                                self.show_menu = false;
                                            }
                                        } else if self.menu_section == 3 {
                                            if let Some(i) = self.runtime_state.selected() {
                                                use crate::settings::{
                                                    PowerMode, cycle_context_token_limit,
                                                    cycle_repeat_threshold, format_context_tokens,
                                                    get_settings, set_power_mode,
                                                    toggle_repeat_thinking,
                                                };
                                                match i {
                                                    0 => {
                                                        set_power_mode(PowerMode::PowerSaver);
                                                        // llama.cpp managed server restarts on next use
                                                        crate::llama::server::shutdown_managed_server();
                                                        self.status_message =
                                                    "Power mode: Power Saver (llama.cpp restarts; llama.rs uses fewer tokens)"
                                                        .to_string();
                                                    }
                                                    1 => {
                                                        set_power_mode(PowerMode::Normal);
                                                        crate::llama::server::shutdown_managed_server();
                                                        self.status_message =
                                                            "Power mode: Normal (default)"
                                                                .to_string();
                                                    }
                                                    2 => {
                                                        set_power_mode(PowerMode::Extreme);
                                                        crate::llama::server::shutdown_managed_server();
                                                        self.status_message =
                                                    "Power mode: Extreme (llama.cpp max ngl; llama.rs more tokens)"
                                                        .to_string();
                                                    }
                                                    3 => {
                                                        use crate::settings::cycle_llama_rs_sub_backend;
                                                        let b = cycle_llama_rs_sub_backend();
                                                        self.status_message = format!(
                                                            "llama.cpp sub-backend: {} (HERCULES_COMPUTE_BACKEND={})",
                                                            b.label(),
                                                            b.env_val()
                                                        );
                                                    }
                                                    4 => {
                                                        use crate::settings::{
                                                            cycle_stall_timeout,
                                                            format_stall_timeout,
                                                        };
                                                        let t = cycle_stall_timeout();
                                                        self.status_message = format!(
                                                            "Stall Watchdog Timeout: {}",
                                                            format_stall_timeout(t)
                                                        );
                                                    }
                                                    5 => {
                                                        cycle_repeat_threshold();
                                                        let s = get_settings();
                                                        self.status_message = format!(
                                                            "Repeat threshold: {} consecutive hits",
                                                            s.repeat_threshold
                                                        );
                                                    }
                                                    6 => {
                                                        toggle_repeat_thinking();
                                                        let s = get_settings();
                                                        self.status_message = format!(
                                                            "Repeat detect on thinking: {}",
                                                            if s.repeat_detect_thinking {
                                                                "ON"
                                                            } else {
                                                                "OFF"
                                                            }
                                                        );
                                                    }
                                                    7 => {
                                                        let n = cycle_context_token_limit();
                                                        crate::llama::server::shutdown_managed_server();
                                                        self.status_message = format!(
                                                            "Context limit: {} tokens (llama-server -c restarts on next gen)",
                                                            format_context_tokens(n)
                                                        );
                                                    }
                                                    _ => {}
                                                }
                                                // Status bar + activity log only (not chat / not model ctx)
                                                let s = get_settings();
                                                if let Ok(mut l) = self.activity_logs.lock() {
                                                    l.push(format!(
                                                "[RUNTIME] power={} ctx={} temp={:.2} repeat={} think={}",
                                                s.power_mode.label(),
                                                format_context_tokens(
                                                    crate::settings::context_token_limit()
                                                ),
                                                crate::settings::temperature(),
                                                s.repeat_threshold,
                                                s.repeat_detect_thinking
                                            ));
                                                }
                                            }
                                        } else if self.menu_section == 4 {
                                            if let Some(i) = self.perms_state.selected() {
                                                match i {
                                                    0 => {
                                                        set_permission_mode(PermissionMode::Ask);
                                                        self.status_message =
                                                    "Permissions: Ask user to allow (writes/cmd blocked until /allow)"
                                                        .to_string();
                                                    }
                                                    1 => {
                                                        set_permission_mode(
                                                            PermissionMode::AlwaysAllow,
                                                        );
                                                        self.status_message =
                                                    "Permissions: Always allow tool writes & commands"
                                                        .to_string();
                                                    }
                                                    2 => {
                                                        set_folder_scope(FolderScope::CurrentDir);
                                                        self.status_message =
                                                            "Safefolder: current directory only"
                                                                .to_string();
                                                    }
                                                    _ => {
                                                        set_folder_scope(FolderScope::AllDirs);
                                                        self.status_message =
                                                            "Safefolder: all directories allowed"
                                                                .to_string();
                                                    }
                                                }
                                                let p = get_tool_permissions();
                                                self.messages.push(format!(
                                                    "System: Permissions → {} | {}",
                                                    p.mode_label(),
                                                    p.scope_label()
                                                ));
                                                if let Ok(mut l) = self.activity_logs.lock() {
                                                    l.push(format!(
                                                        "[PERMS] mode={} scope={}",
                                                        p.mode_label(),
                                                        p.scope_label()
                                                    ));
                                                }
                                            }
                                        }
                                    } else if self.input_focused && !self.input.trim().is_empty() {
                                        let is_ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
                                        if *self.is_generating.lock().unwrap() {
                                            if is_ctrl {
                                                // CTRL + Enter forces prompt submission and interrupts ongoing generation
                                                *self.is_generating.lock().unwrap() = false;
                                                if let Some(last) = self.messages.last_mut() {
                                                    if last.starts_with("Agent: ") {
                                                        last.push_str("\n[Interrupted by User]");
                                                    }
                                                }
                                            } else {
                                                self.status_message = "Generating... Shift+Enter=newline · CTRL+Enter=interrupt & send".to_string();
                                                return Ok(());
                                            }
                                        }

                                        let prompt = self.input.clone();
                                        self.input.clear();
                                        self.input_cursor_position = 0;
                                        self.typewriter_len = 0;
                                        self.auto_scroll_enabled = true;
                                        self.auto_tool_turns = 0;

                                        let backend = self.backend.name();
                                        if let Ok(mut l) = self.activity_logs.lock() {
                                            l.push(format!(
                                                "[PROMPT] Processing: \"{}\" via {}",
                                                prompt, backend
                                            ));
                                        }

                                        if prompt.starts_with('/') {
                                            self.messages.push(format!("You: {}", prompt));
                                            let parts: Vec<&str> =
                                                prompt.split_whitespace().collect();
                                            match parts[0] {
                                                "/help" => {
                                                    self.messages.push(
                                                        "System: Press F1 for shortcut overlay.\n\
/allow   — grant write/cmd for this session (Ask mode)\n\
/compact — compress chat → memory, forget old turns\n\
/cancel-download — clear stuck HF download lock so you can install another model\n\
/download-status — show active download lock\n\
/copy /theme /save /load — utilities"
                                                            .to_string(),
                                                    );
                                                }
                                                "/allow" => {
                                                    allow_session_tools();
                                                    self.messages.push(
                                                "System: Session tool permission granted (write/cmd allowed until restart)."
                                                    .to_string(),
                                            );
                                                }
                                                "/compact" | "/compact!" | "/gc" => {
                                                    let before =
                                                        self.estimate_full_session_tokens();
                                                    let msgs = self.messages.len();
                                                    self.compact_context_to_memory(true);
                                                    self.messages.push(format!(
                                                "System: [OK] /compact done — was ~{} tokens / {} messages → \
                                                 ~{} tokens now. Summary in memory. Type a new question.",
                                                before,
                                                msgs,
                                                self.context_tokens_est
                                            ));
                                                }
                                                "/tasks" => {
                                                    let list = self.task_manager.list();
                                                    if list.is_empty() {
                                                        self.messages.push(
                                                    "System: Task manager empty — no background jobs."
                                                        .into(),
                                                );
                                                    } else {
                                                        let mut lines = vec![
                                                            "System: Task manager:".to_string(),
                                                        ];
                                                        for t in list {
                                                            let age = t.started.elapsed().as_secs();
                                                            lines.push(format!(
                                                                "  #{} [{:?}] {age}s  `{}`",
                                                                t.id, t.status, t.cmd
                                                            ));
                                                        }
                                                        lines.push(
                                                            "  Ctrl+C kills all running tasks."
                                                                .into(),
                                                        );
                                                        self.messages.push(lines.join("\n"));
                                                    }
                                                }
                                                "/cancel-download" | "/cancel_download"
                                                | "/cdl" => {
                                                    let msg = self.manager.cancel_download();
                                                    self.messages.push(format!("System: {msg}"));
                                                    self.status_message = msg;
                                                    if let Ok(mut l) = self.activity_logs.lock() {
                                                        l.push("[DOWNLOAD] cancel-download".into());
                                                    }
                                                    // Clear stuck progress UI
                                                    *self.download_progress.lock().unwrap() = None;
                                                    *self.download_complete.lock().unwrap() = false;
                                                }
                                                "/download-status" | "/dlstatus" => {
                                                    let s = self.manager.download_status();
                                                    self.messages.push(format!("System: {s}"));
                                                    self.status_message = s;
                                                }
                                                "/copy" => {
                                                    let full_chat = self.messages.join("\n\n");
                                                    let ok = crate::clipboard::copy_text_silent(
                                                        &full_chat,
                                                    );
                                                    let _ = std::fs::write(
                                                        "/tmp/hercules_chat_export.txt",
                                                        &full_chat,
                                                    );
                                                    self.messages.push(format!(
                                                "System: Chat exported to /tmp/hercules_chat_export.txt{}",
                                                if ok { " (+ clipboard)" } else { "" }
                                            ));
                                                }
                                                "/theme" => {
                                                    if parts.len() > 1 {
                                                        match parts[1] {
                                                            "blue" => {
                                                                self.theme_color =
                                                                    Color::Rgb(0, 150, 255)
                                                            }
                                                            "red" => {
                                                                self.theme_color =
                                                                    Color::Rgb(255, 50, 50)
                                                            }
                                                            "green" | _ => {
                                                                self.theme_color =
                                                                    Color::Rgb(0, 255, 128)
                                                            }
                                                        }
                                                        self.messages.push(format!(
                                                            "System: Theme changed to {}",
                                                            parts[1]
                                                        ));
                                                    }
                                                }
                                                "/save" => {
                                                    if parts.len() > 1 {
                                                        let _ = std::fs::write(
                                                            parts[1],
                                                            self.messages.join("\n"),
                                                        );
                                                        self.messages.push(format!(
                                                            "System: Session saved to {}",
                                                            parts[1]
                                                        ));
                                                    }
                                                }
                                                "/load" => {
                                                    if parts.len() > 1 {
                                                        if let Ok(data) =
                                                            std::fs::read_to_string(parts[1])
                                                        {
                                                            self.messages = data
                                                                .split('\n')
                                                                .map(|s| s.to_string())
                                                                .collect();
                                                            self.messages.push(format!(
                                                                "System: Session loaded from {}",
                                                                parts[1]
                                                            ));
                                                        } else {
                                                            self.messages.push(
                                                                "System: Failed to load session"
                                                                    .to_string(),
                                                            );
                                                        }
                                                    }
                                                }
                                                _ => self.messages.push(format!(
                                                    "System: Unknown command '{}'",
                                                    parts[0]
                                                )),
                                            }
                                        } else {
                                            self.messages.push(format!("You: {}", prompt));
                                            self.trigger_generation_from_context();
                                        }
                                    }
                                }
                                _ => {}
                            }
                        } // end if !pending_consumed
                    } // end if !skip_key
                }
                _ => {}
            }
        }
        Ok(())
    }

    pub fn draw(&mut self, frame: &mut Frame) {
        let theme_color = self.theme_color;
        let dark_gray = Color::Rgb(100, 100, 100);
        let light_blue = Color::Rgb(150, 180, 255);
        let white = Color::White;

        let area = frame.area();

        // Grow input box with multiline content (3..=10 rows total including borders)
        let input_lines = self.input_line_count();
        let input_inner_h = (input_lines as u16).clamp(1, 8);
        let input_box_h = (input_inner_h + 2).clamp(3, 10); // borders

        // Top bar 3 rows — classic two-arm resource box (CPU%+°C, Mem%+GB).
        //
        //  L0: CTX …              Model: …                 ╭[C: 64% 70C]╮
        //  L1: ╭[Hercules]╮       Press CTRL+M…            ├[M: 53% 4.1G]┤
        //  L2: ╰──────────┴───────…────────────────────────┴─────────────╯
        //
        // Right column paints ALL 3 lines of the stats box (complete, closed).
        // Floor is one continuous string so arms always meet.
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints(
                [
                    Constraint::Length(3),
                    Constraint::Min(1),
                    Constraint::Length(input_box_h),
                    Constraint::Length(1),
                ]
                .as_ref(),
            )
            .split(area);

        const BRAND_W: usize = 12;
        // Exact width of every stats line including floor under the box
        const STATS_W: usize = 16;
        const LEFT_W: usize = 24;

        let ctx_limit = crate::settings::context_token_limit().max(1);
        let ctx_used = self.estimate_full_session_tokens();
        self.context_tokens_est = ctx_used;
        let ctx_pct = ((ctx_used as f64 / ctx_limit as f64) * 100.0).min(999.0);
        let ctx_label = crate::settings::format_context_tokens(ctx_limit);
        let ctx_color = if ctx_pct >= 80.0 {
            Color::Rgb(255, 100, 100)
        } else if ctx_pct >= 50.0 {
            Color::Rgb(255, 200, 80)
        } else {
            theme_color
        };
        let filled = ((ctx_pct / 100.0) * 8.0).round() as usize;
        let mut bar = String::new();
        for i in 0..8 {
            bar.push(if i < filled { '█' } else { '░' });
        }
        let ctx_line = fit_width(&format!("CTX {bar} {ctx_pct:.0}% {ctx_label}"), LEFT_W);

        let top_chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints(
                [
                    Constraint::Length(LEFT_W as u16),
                    Constraint::Min(1),
                    Constraint::Length(STATS_W as u16),
                ]
                .as_ref(),
            )
            .split(chunks[0]);

        let exit_hold_pct = self
            .esc_hold_start
            .map(|start| (start.elapsed().as_secs_f64() / 1.0).clamp(0.0, 1.0));
        let exiting_text = if let Some(pct) = exit_hold_pct {
            format!(" EXIT {:.0}%", pct * 100.0)
        } else {
            "".to_string()
        };
        let exit_glow = if let Some(pct) = exit_hold_pct {
            let pulse = ((self.anim_tick as f64 * 0.35).sin() * 0.5 + 0.5) as f32;
            let r = (180.0 + 75.0 * pct as f32 + 20.0 * pulse) as u8;
            Color::Rgb(
                r.min(255),
                (40.0 * (1.0 - pct as f32)) as u8,
                (40.0 * (1.0 - pct as f32)) as u8,
            )
        } else {
            theme_color
        };

        let brand_top = "╭[Hercules]╮"; // 12
        // Left L0–L1 only; L2 is the shared floor (painted full-width below)
        let logo_text = vec![
            Line::from(Span::styled(
                ctx_line,
                Style::default().fg(ctx_color).add_modifier(Modifier::BOLD),
            )),
            Line::from(vec![
                Span::styled(
                    brand_top.to_string(),
                    Style::default().fg(exit_glow).add_modifier(Modifier::BOLD),
                ),
                Span::styled(
                    fit_width(&exiting_text, LEFT_W.saturating_sub(BRAND_W)),
                    Style::default()
                        .fg(Color::Rgb(255, 80, 80))
                        .bg(if exit_hold_pct.is_some() {
                            Color::Rgb(40, 0, 0)
                        } else {
                            Color::Reset
                        })
                        .add_modifier(Modifier::BOLD),
                ),
            ]),
        ];
        frame.render_widget(Paragraph::new(logo_text), top_chunks[0]);

        let active_model = self.backend.name();
        let mid_w = top_chunks[1].width as usize;
        let model_line = fit_width(&format!("Model: {active_model}"), mid_w);
        let menu_line = fit_width("Press CTRL+M or F2 for Menu Modal", mid_w);
        let hint_text = vec![
            Line::from(Span::styled(
                model_line,
                Style::default().fg(light_blue).add_modifier(Modifier::BOLD),
            )),
            Line::from(Span::styled(
                menu_line,
                Style::default()
                    .fg(theme_color)
                    .add_modifier(Modifier::BOLD),
            )),
        ];
        frame.render_widget(
            Paragraph::new(hint_text).alignment(Alignment::Right),
            top_chunks[1],
        );

        // Two-arm resource box — complete 3-line box in the right column
        let cpu_usage = self.sys.global_cpu_usage().clamp(0.0, 100.0);
        let mem_usage = (self.sys.used_memory() as f32 / self.sys.total_memory().max(1) as f32
            * 100.0)
            .clamp(0.0, 100.0);
        let mem_gb = self.sys.used_memory() as f64 / 1_073_741_824.0;
        let cpu_temp_c = cpu_package_temp_c(&self.sys);
        let (c_line, m_line, s_bot) = fixed_resource_box(cpu_usage, cpu_temp_c, mem_usage, mem_gb);
        debug_assert_eq!(c_line.chars().count(), STATS_W);
        debug_assert_eq!(m_line.chars().count(), STATS_W);
        debug_assert_eq!(s_bot.chars().count(), STATS_W);

        // Complete box: top, middle, bottom (bottom also painted in full floor for join)
        let stats_lines = vec![
            Line::from(Span::styled(
                c_line.clone(),
                Style::default().fg(light_blue).add_modifier(Modifier::BOLD),
            )),
            Line::from(Span::styled(
                m_line.clone(),
                Style::default().fg(light_blue).add_modifier(Modifier::BOLD),
            )),
            Line::from(Span::styled(
                s_bot.clone(),
                Style::default().fg(Color::Rgb(140, 140, 150)),
            )),
        ];
        frame.render_widget(Paragraph::new(stats_lines), top_chunks[2]);

        // Full-width floor on L2: joins brand → mid ─ → stats bottom (same glyphs as s_bot)
        let full_w = chunks[0].width as usize;
        let floor = continuous_top_floor(full_w, LEFT_W, STATS_W);
        frame.render_widget(
            Paragraph::new(Line::from(Span::styled(
                floor,
                Style::default().fg(Color::Rgb(140, 140, 150)),
            ))),
            Rect {
                x: chunks[0].x,
                y: chunks[0].y.saturating_add(2),
                width: chunks[0].width,
                height: 1,
            },
        );

        // --- Dynamic Focus Glowing Border Pulse with KramaFrame ---
        // Exit-hold overrides input border with red glow
        let focus_progress = self.krama.get_progress_f32("focus", 0);
        let border_color = if exit_hold_pct.is_some() {
            exit_glow
        } else if self.input_focused {
            let pulse = (self.anim_tick as f64 * 0.18 + focus_progress as f64).sin() * 0.5 + 0.5;
            let g = (140.0 + 115.0 * pulse) as u8;
            let b = (80.0 + 60.0 * pulse) as u8;
            Color::Rgb(0, g, b)
        } else {
            dark_gray
        };

        // --- Split Terminal Layout (Main Chat + Smooth Sliding Activity Console) ---
        let log_pct = self.current_log_pane_pct.round() as u16;
        let main_split = if log_pct == 0 {
            Layout::default()
                .direction(Direction::Horizontal)
                .constraints([Constraint::Percentage(100)].as_ref())
                .split(chunks[1])
        } else {
            Layout::default()
                .direction(Direction::Horizontal)
                .constraints(
                    [
                        Constraint::Percentage(100 - log_pct), // Left Pane: Chat
                        Constraint::Percentage(log_pct),       // Right Pane: Activity Console
                    ]
                    .as_ref(),
                )
                .split(chunks[1])
        };

        // --- Dynamic Main Body Layout (Chat + Main Body Aligned Gradient Progress Bar) ---
        let progress_val = *self.download_progress.lock().unwrap();
        let left_chunks = if progress_val.is_some() {
            Layout::default()
                .direction(Direction::Vertical)
                .constraints([Constraint::Min(1), Constraint::Length(3)].as_ref())
                .split(main_split[0])
        } else {
            Layout::default()
                .direction(Direction::Vertical)
                .constraints([Constraint::Min(1)].as_ref())
                .split(main_split[0])
        };

        let chat_area = left_chunks[0];

        // Keep chip anchors on valid Agent turns so buttons scroll with that turn
        {
            let last_agent = self.latest_agent_msg_idx();
            let n = self.messages.len();
            for c in &mut self.tool_chips {
                let ok = c
                    .anchor_msg
                    .map(|i| i < n && self.messages[i].starts_with("Agent:"))
                    .unwrap_or(false);
                if !ok {
                    c.anchor_msg = last_agent;
                }
            }
        }

        let is_generating_val = *self.is_generating.lock().unwrap();
        let anim_tick = self.anim_tick;
        let mut chat_lines: Vec<Line> = Vec::new();
        // (chip_id, logical chat_lines index where chip spacer starts)
        let mut chip_line_starts: Vec<(u64, usize)> = Vec::new();

        for (m_idx, m) in self.messages.iter().enumerate() {
            let is_last_message = m_idx == self.messages.len() - 1;

            if m.starts_with("You:") {
                chat_lines.push(Line::from(vec![
                    Span::styled(
                        "You: ",
                        Style::default()
                            .fg(theme_color)
                            .add_modifier(Modifier::BOLD),
                    ),
                    Span::styled(&m[5..], Style::default().fg(white)),
                ]));
                chat_lines.push(Line::from(""));
            } else if m.starts_with("Agent:") || m.starts_with("Error:") {
                let content = if m.starts_with("Agent:") {
                    &m[7..]
                } else {
                    &m[7..]
                };

                // Thinking UI only for real <think>…</think> (Ollama wraps its
                // `thinking` stream field that way). llama.cpp / plain GGUF usually
                // have NO think tags — never treat their whole stream as "Thinking".
                let (think_part, output_part, think_label) =
                    if let Some(start_think) = content.find("<think>") {
                        if let Some(end_think) = content.find("</think>") {
                            let think = &content[start_think + 7..end_think];
                            let rest = &content[end_think + 8..];
                            let before = &content[..start_think];
                            // Content before <think> still counts as agent output
                            let out = if before.trim().is_empty() {
                                rest.to_string()
                            } else {
                                format!("{}{}", before, rest)
                            };
                            (Some(think.to_string()), out, "Model thinking")
                        } else {
                            // Unclosed <think> while streaming (Ollama or explicit tags)
                            let think = &content[start_think + 7..];
                            let before = content[..start_think].to_string();
                            (
                                Some(think.to_string()),
                                before,
                                "Model thinking (streaming)",
                            )
                        }
                    } else {
                        // No think tags → all content is Agent output (llama.cpp default)
                        (None, content.to_string(), "")
                    };

                let total_chars = content.chars().count();
                let reveal_limit = if is_last_message { total_chars } else { 10000 };

                // 1. Render Thinking Process only when real <think> exists
                if let Some(ref think_text) = think_part {
                    if !think_text.trim().is_empty()
                        || (is_generating_val && content.contains("<think>"))
                    {
                        if self.thinking_collapsed {
                            chat_lines.push(Line::from(Span::styled(
                                format!("[{think_label} · Collapsed — CTRL+T]"),
                                Style::default()
                                    .fg(Color::Rgb(180, 130, 255))
                                    .add_modifier(Modifier::BOLD | Modifier::ITALIC),
                            )));
                        } else {
                            chat_lines.push(Line::from(Span::styled(
                                format!("[{think_label} — CTRL+T collapse]"),
                                Style::default()
                                    .fg(Color::Rgb(180, 130, 255))
                                    .add_modifier(Modifier::BOLD | Modifier::ITALIC),
                            )));

                            let visible_think = reveal_limit.min(think_text.chars().count());

                            let mut global_think_ch = 0;
                            if think_text.trim().is_empty() && is_generating_val {
                                let pulse = (anim_tick as f64 * 0.3).sin() * 0.5 + 0.5;
                                let b_val = (160.0 + 95.0 * pulse) as u8;
                                chat_lines.push(Line::from(Span::styled(
                                    "  Reasoning… █",
                                    Style::default()
                                        .fg(Color::Rgb(180, 130, b_val))
                                        .add_modifier(Modifier::ITALIC),
                                )));
                            } else {
                                for raw_line in think_text.lines() {
                                    let mut line_spans =
                                        vec![Span::styled("  │ ", Style::default().fg(dark_gray))];
                                    for ch in raw_line.chars() {
                                        if global_think_ch >= visible_think {
                                            break;
                                        }
                                        let age = visible_think.saturating_sub(global_think_ch);
                                        let progress = if is_generating_val && is_last_message {
                                            (age as f64 / 10.0).clamp(0.1, 1.0)
                                        } else {
                                            1.0
                                        };
                                        let r = (35.0 + (190.0 - 35.0) * progress) as u8;
                                        let g = (30.0 + (150.0 - 30.0) * progress) as u8;
                                        let b = (50.0 + (255.0 - 50.0) * progress) as u8;
                                        line_spans.push(Span::styled(
                                            ch.to_string(),
                                            Style::default().fg(Color::Rgb(r, g, b)),
                                        ));
                                        global_think_ch += 1;
                                    }
                                    global_think_ch += 1;
                                    chat_lines.push(Line::from(line_spans));
                                }
                            }
                        }
                        chat_lines.push(Line::from(""));
                    }
                }

                // 2. Render Agent Response Output (llama.cpp / tools live here — not under Thinking)
                if !output_part.trim().is_empty()
                    || (think_part.is_none() && (is_generating_val || !content.trim().is_empty()))
                {
                    let text_to_render = if output_part.is_empty() && think_part.is_none() {
                        content
                    } else {
                        output_part.as_str()
                    };
                    let think_len = think_part.as_ref().map(|t| t.chars().count()).unwrap_or(0);
                    let available_output = reveal_limit.saturating_sub(think_len);

                    if !text_to_render.trim().is_empty() || is_generating_val {
                        let agent_label = if is_generating_val && is_last_message {
                            "Agent (streaming): "
                        } else {
                            "Agent: "
                        };
                        chat_lines.push(Line::from(Span::styled(
                            agent_label,
                            Style::default().fg(light_blue).add_modifier(Modifier::BOLD),
                        )));

                        let mut in_code_block = false;
                        let total_lines = text_to_render.lines().count();
                        let mut global_out_ch = 0;
                        for (l_idx, raw_line) in text_to_render.lines().enumerate() {
                            let is_last_line = l_idx + 1 == total_lines;
                            let trimmed = raw_line.trim();

                            if trimmed.starts_with("```") {
                                in_code_block = !in_code_block;
                                let tag = if in_code_block {
                                    " --- Code Block ---"
                                } else {
                                    " --- End Code ---"
                                };
                                chat_lines.push(Line::from(Span::styled(
                                    tag,
                                    Style::default()
                                        .fg(theme_color)
                                        .add_modifier(Modifier::BOLD),
                                )));
                                global_out_ch += raw_line.chars().count() + 1;
                                continue;
                            }

                            let mut line_spans = Vec::new();

                            if trimmed.starts_with('|') && trimmed.contains('|') {
                                // Markdown Table Formatting using Box Borders
                                line_spans.push(Span::styled("  ", Style::default()));
                                for cell in trimmed.split('|').filter(|s| !s.trim().is_empty()) {
                                    if cell.chars().all(|c| c == '-' || c == ':' || c == ' ') {
                                        line_spans.push(Span::styled(
                                            "─────┼─────",
                                            Style::default().fg(dark_gray),
                                        ));
                                    } else {
                                        line_spans.push(Span::styled(
                                            format!(" {} │", cell.trim()),
                                            Style::default()
                                                .fg(Color::Rgb(0, 230, 255))
                                                .add_modifier(Modifier::BOLD),
                                        ));
                                    }
                                }
                                global_out_ch += raw_line.chars().count() + 1;
                            } else if trimmed.starts_with('#') {
                                line_spans.push(Span::styled(
                                    format!("  {}", trimmed),
                                    Style::default()
                                        .fg(theme_color)
                                        .add_modifier(Modifier::BOLD),
                                ));
                                global_out_ch += raw_line.chars().count() + 1;
                            } else if trimmed.starts_with('-') || trimmed.starts_with('*') {
                                line_spans
                                    .push(Span::styled("  * ", Style::default().fg(theme_color)));
                                for ch in trimmed[1..].chars() {
                                    if global_out_ch >= available_output {
                                        break;
                                    }
                                    let age = available_output.saturating_sub(global_out_ch);
                                    let progress = if is_generating_val && is_last_message {
                                        (age as f64 / 10.0).clamp(0.1, 1.0)
                                    } else {
                                        1.0
                                    };
                                    let r = (40.0 + (240.0 - 40.0) * progress) as u8;
                                    let g = (55.0 + (245.0 - 55.0) * progress) as u8;
                                    let b = (65.0 + (255.0 - 65.0) * progress) as u8;
                                    line_spans.push(Span::styled(
                                        ch.to_string(),
                                        Style::default().fg(Color::Rgb(r, g, b)),
                                    ));
                                    global_out_ch += 1;
                                }
                                global_out_ch += 1;
                            } else {
                                line_spans.push(Span::styled("  ", Style::default()));
                                for ch in raw_line.chars() {
                                    if global_out_ch >= available_output {
                                        break;
                                    }
                                    let age = available_output.saturating_sub(global_out_ch);
                                    let progress = if is_generating_val && is_last_message {
                                        (age as f64 / 10.0).clamp(0.1, 1.0)
                                    } else {
                                        1.0
                                    };
                                    let r = (40.0 + (240.0 - 40.0) * progress) as u8;
                                    let g = (55.0 + (245.0 - 55.0) * progress) as u8;
                                    let b = (65.0 + (255.0 - 65.0) * progress) as u8;
                                    line_spans.push(Span::styled(
                                        ch.to_string(),
                                        Style::default().fg(Color::Rgb(r, g, b)),
                                    ));
                                    global_out_ch += 1;
                                }
                                global_out_ch += 1;
                            }

                            if is_last_message && is_generating_val && is_last_line {
                                let pulse = (anim_tick as f64 * 0.3).sin() * 0.5 + 0.5;
                                let b_val = (150.0 + 105.0 * pulse) as u8;
                                line_spans.push(Span::styled(
                                    " █",
                                    Style::default()
                                        .fg(Color::Rgb(0, 255, b_val))
                                        .add_modifier(Modifier::BOLD),
                                ));
                            }

                            chat_lines.push(Line::from(line_spans));
                        }
                        chat_lines.push(Line::from(""));
                    }
                }

                // Stick WRITE/RUN chips under THIS agent turn (not chat bottom)
                if m.starts_with("Agent:") {
                    let ids: Vec<u64> = self
                        .tool_chips
                        .iter()
                        .filter(|c| c.anchor_msg == Some(m_idx))
                        .map(|c| c.id)
                        .collect();
                    for id in ids {
                        let start = chat_lines.len();
                        for _ in 0..tool_panel::CHIP_ROW_HEIGHT {
                            chat_lines.push(Line::from(""));
                        }
                        chip_line_starts.push((id, start));
                    }
                }
            } else if m.starts_with("System:") {
                // Multi-line system / tool output (cargo etc.)
                for (i, line) in m.lines().enumerate() {
                    let style = if i == 0 {
                        Style::default().fg(dark_gray)
                    } else {
                        Style::default().fg(Color::Rgb(160, 160, 160))
                    };
                    chat_lines.push(Line::from(Span::styled(line.to_string(), style)));
                }
                chat_lines.push(Line::from(""));
            } else {
                chat_lines.push(Line::from(Span::styled(
                    m.clone(),
                    Style::default().fg(dark_gray),
                )));
            }
        }

        // Orphan chips (no anchor row yet) — park under last agent turn spacers already
        // inserted; if none, reserve at end so they remain clickable
        {
            let placed: std::collections::HashSet<u64> =
                chip_line_starts.iter().map(|(id, _)| *id).collect();
            let orphans: Vec<u64> = self
                .tool_chips
                .iter()
                .filter(|c| !placed.contains(&c.id))
                .map(|c| c.id)
                .collect();
            for id in orphans {
                let start = chat_lines.len();
                for _ in 0..tool_panel::CHIP_ROW_HEIGHT {
                    chat_lines.push(Line::from(""));
                }
                chip_line_starts.push((id, start));
            }
        }

        let available_width = (chat_area.width.saturating_sub(2) as usize).max(1);
        let mut total_visual_lines: u16 = 0;
        for line in &chat_lines {
            let w = line.width();
            if w == 0 {
                total_visual_lines += 1;
            } else {
                let lines = (w + available_width - 1) / available_width;
                total_visual_lines += lines as u16;
            }
        }

        let viewport_height = chat_area.height.saturating_sub(2);
        let max_scroll = total_visual_lines.saturating_sub(viewport_height);

        if self.auto_scroll_enabled {
            self.scroll_offset = max_scroll;
        } else if self.scroll_offset >= max_scroll {
            self.scroll_offset = max_scroll;
            self.auto_scroll_enabled = true;
        }

        // Snapshot plain lines for clipboard (before shade mutates styles)
        self.last_chat_plain_lines = chat_lines
            .iter()
            .map(|line| {
                line.spans
                    .iter()
                    .map(|s| s.content.as_ref())
                    .collect::<String>()
            })
            .collect();

        // Shade selected rows while dragging OR after release (has_selection)
        if self.selection_active() {
            if let (Some(start), Some(end)) = (self.selection_start, self.selection_end) {
                let min_y = start.1.min(end.1) as i32;
                let max_y = start.1.max(end.1) as i32;
                let chat_top = chat_area.y as i32 + 1;
                let sel_bg = Color::Rgb(28, 72, 128);
                for (i, line) in chat_lines.iter_mut().enumerate() {
                    let screen_y = chat_top + i as i32 - self.scroll_offset as i32;
                    if screen_y >= min_y && screen_y <= max_y {
                        let spans: Vec<Span> = line
                            .spans
                            .iter()
                            .map(|s| {
                                Span::styled(
                                    s.content.clone(),
                                    s.style.bg(sel_bg).add_modifier(Modifier::BOLD),
                                )
                            })
                            .collect();
                        *line = if spans.is_empty() {
                            Line::from(Span::styled(" ", Style::default().bg(sel_bg)))
                        } else {
                            Line::from(spans)
                        };
                    }
                }
            }
        }

        // Logical → visual line starts (for chip placement under agent turns)
        let mut visual_at: Vec<u16> = Vec::with_capacity(chat_lines.len() + 1);
        {
            let mut acc = 0u16;
            for line in &chat_lines {
                visual_at.push(acc);
                let w = line.width();
                acc = acc.saturating_add(if w == 0 {
                    1
                } else {
                    ((w + available_width - 1) / available_width) as u16
                });
            }
        }

        let chat_box = Paragraph::new(chat_lines)
            .scroll((self.scroll_offset, 0))
            .wrap(ratatui::widgets::Wrap { trim: false })
            .block(
                Block::default()
                    .borders(if log_pct == 0 {
                        Borders::NONE
                    } else {
                        Borders::RIGHT
                    })
                    .border_style(Style::default().fg(dark_gray))
                    .title(if self.selection_active() {
                        " Main Chat  [select: Ctrl+C copy | click cancel] "
                    } else {
                        " Main Chat "
                    }),
            );
        frame.render_widget(chat_box, chat_area);

        // Draw chips at their agent-turn anchors (visual line + scroll)
        {
            let x = chat_area.x.saturating_add(3);
            let max_w = chat_area.width.saturating_sub(6);
            let chat_top = chat_area.y.saturating_add(1);
            let chat_bot = chat_area
                .y
                .saturating_add(chat_area.height.saturating_sub(1));

            for chip in &mut self.tool_chips {
                chip.rect = None;
            }

            for (chip_id, logical_start) in &chip_line_starts {
                let Some(chip) = self.tool_chips.iter_mut().find(|c| c.id == *chip_id) else {
                    continue;
                };
                let vis = visual_at.get(*logical_start).copied().unwrap_or(0);
                if vis < self.scroll_offset {
                    continue;
                }
                let y_off = vis.saturating_sub(self.scroll_offset);
                let y = chat_top.saturating_add(y_off);
                if y.saturating_add(tool_panel::CHIP_ROW_HEIGHT) > chat_bot {
                    continue;
                }
                chip.draw_at(frame, x, y, max_w);
            }
        }

        self.last_chat_area = Some(chat_area);

        // KramaFrame fly: abs(progress) so reverse animates 1→0 (not clamp-to-0 snap)
        if let Some(ref mut panel) = self.tool_panel {
            let t = self.krama.get_progress_f32("panel_fly", 0).abs();
            // Sync chip origin every frame
            if let Some(chip) = self
                .tool_chips
                .iter()
                .find(|c| c.id == panel.chip_id)
                .cloned()
            {
                if let Some(r) = chip.rect {
                    panel.chip_rect = Some(r);
                }
                if panel.kind == ToolPanelKind::Write {
                    panel.set_body_streaming(chip.body.clone(), chip.tag_closed);
                } else if panel.kind == ToolPanelKind::Cmd && !chip.body.is_empty() {
                    panel.set_body_streaming(chip.body.clone(), true);
                }
            } else if let Some(chip) = self.tool_chips.iter().find(|c| {
                c.kind == panel.kind
                    && tool_panel::same_tool_target(panel.kind, &c.target, &panel.target)
            }) {
                if let Some(r) = chip.rect {
                    panel.chip_rect = Some(r);
                }
            }
            if panel.chip_rect.is_none() {
                panel.chip_rect = Some(Rect {
                    x: chat_area.x + 3,
                    y: chat_area.y + 4,
                    width: 30,
                    height: 3,
                });
            }
            let dock_w = (chat_area.width * 45 / 100).clamp(30, chat_area.width.saturating_sub(10));
            // Minimized dock is title bar only
            let dock_h = if panel.minimized {
                3
            } else {
                chat_area.height.saturating_sub(2)
            };
            panel.dock_rect = Some(Rect {
                x: chat_area.x + chat_area.width.saturating_sub(dock_w),
                y: chat_area.y + 1,
                width: dock_w,
                height: dock_h,
            });
            if let Some(rect) = tool_panel::draw_tool_panel(frame, panel, t, theme_color) {
                self.tool_panel_rect = Some(rect);
            }
        } else {
            self.tool_panel_rect = None;
        }

        // Main Body Aligned Gradient Progress Bar
        if let Some(ratio) = progress_val {
            let total_width = left_chunks[1].width.saturating_sub(22) as usize;
            let mut bar_spans = Vec::new();
            bar_spans.push(Span::styled(
                " [DOWNLOADING] ",
                Style::default()
                    .fg(theme_color)
                    .add_modifier(Modifier::BOLD),
            ));

            let subblocks = [' ', '▏', '▎', '▍', '▌', '▋', '▊', '▉', '█'];
            let total_subblocks = total_width * 8;
            let filled_subblocks = ((ratio.clamp(0.0, 1.0) * total_subblocks as f64).round()
                as usize)
                .min(total_subblocks);
            let full_blocks = filled_subblocks / 8;
            let partial_idx = filled_subblocks % 8;

            for i in 0..full_blocks {
                let norm = i as f64 / total_width.max(1) as f64;
                let r = (255.0 * norm) as u8;
                let g = 255;
                let b = (255.0 * (1.0 - norm)) as u8;
                bar_spans.push(Span::styled("█", Style::default().fg(Color::Rgb(r, g, b))));
            }

            if full_blocks < total_width && partial_idx > 0 {
                let norm = full_blocks as f64 / total_width.max(1) as f64;
                let r = (255.0 * norm) as u8;
                let g = 255;
                let b = (255.0 * (1.0 - norm)) as u8;
                let partial_str = subblocks[partial_idx].to_string();
                bar_spans.push(Span::styled(
                    partial_str,
                    Style::default().fg(Color::Rgb(r, g, b)),
                ));
            }

            let rendered_blocks = full_blocks + if partial_idx > 0 { 1 } else { 0 };
            for _ in rendered_blocks..total_width {
                bar_spans.push(Span::styled("░", Style::default().fg(dark_gray)));
            }
            bar_spans.push(Span::styled(
                format!(" {:.1}% ", ratio * 100.0),
                Style::default().fg(white).add_modifier(Modifier::BOLD),
            ));

            let pbar = Paragraph::new(Line::from(bar_spans)).block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_type(BorderType::Rounded)
                    .border_style(Style::default().fg(theme_color))
                    .title(" Dynamic Model Weights Progress Bar "),
            );
            frame.render_widget(pbar, left_chunks[1]);
        }

        if log_pct > 0 {
            let logs_guard = self.activity_logs.lock().unwrap();
            let logs_text = logs_guard.join("\n");
            let log_lines_count = logs_guard.len() as u16;
            let console_scroll = log_lines_count.saturating_sub(15);

            let console_box = Paragraph::new(logs_text)
                .scroll((console_scroll, 0))
                .wrap(ratatui::widgets::Wrap { trim: true })
                .block(
                    Block::default()
                        .borders(Borders::ALL)
                        .border_type(BorderType::Rounded)
                        .border_style(Style::default().fg(light_blue))
                        .title(" Live Activity Log [CTRL+L/F3: Collapse] "),
                );
            frame.render_widget(console_box, main_split[1]);
        }

        // --- Input Area ---
        let input_layout = Layout::default()
            .direction(Direction::Horizontal)
            .constraints(
                [
                    Constraint::Length(2), // Padding
                    Constraint::Min(1),    // Input
                    Constraint::Length(2), // Padding
                ]
                .as_ref(),
            )
            .split(chunks[2]);

        let title_text = if let Some(pct) = exit_hold_pct {
            format!(
                " [ EXITING {:.0}% — keep holding Esc | release to cancel | Ctrl+Esc = quit now ] ",
                pct * 100.0
            )
        } else if self.term_is_interactive() {
            " [ TERM interactive · type command · Enter=run · Esc/click outside=leave ] "
                .to_string()
        } else if !self.pending_actions.is_empty() {
            format!(
                " [ {} pending - Y/Enter ACCEPT | N REJECT | A always-allow ] ",
                self.pending_actions.len()
            )
        } else if self.input_focused {
            " [ Prompt · Enter=send · Shift+Enter / Ctrl+J = newline · CTRL+F unfocus ] "
                .to_string()
        } else {
            " [ Focus: Main Body | CTRL+F focus input | Y accept pending tools ] ".to_string()
        };

        // Multiline prompt body (or TERM interactive line)
        let input_lines_ui: Vec<Line> = if self.term_is_interactive() {
            vec![Line::from(vec![
                Span::styled(" $ ", Style::default().fg(Color::Rgb(100, 255, 100))),
                Span::styled(
                    if self.term_input.is_empty() {
                        "type shell command…".to_string()
                    } else {
                        self.term_input.clone()
                    },
                    Style::default().fg(if self.term_input.is_empty() {
                        dark_gray
                    } else {
                        Color::Rgb(180, 255, 180)
                    }),
                ),
                Span::styled("█", Style::default().fg(Color::Rgb(100, 255, 100))),
            ])]
        } else if self.input.is_empty() {
            vec![Line::from(Span::styled(
                " Type prompt or /help... (Shift+Enter for new line)",
                Style::default().fg(dark_gray),
            ))]
        } else {
            // Preserve empty trailing line after final \n
            let mut lines: Vec<Line> = self
                .input
                .split('\n')
                .enumerate()
                .map(|(i, row)| {
                    let prefix = if i == 0 { " " } else { " " };
                    Line::from(Span::styled(
                        format!("{prefix}{row}"),
                        Style::default().fg(theme_color),
                    ))
                })
                .collect();
            if self.input.ends_with('\n') {
                lines.push(Line::from(Span::styled(
                    " ",
                    Style::default().fg(theme_color),
                )));
            }
            if lines.is_empty() {
                lines.push(Line::from(""));
            }
            lines
        };

        let input_block = Block::default()
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .border_style(Style::default().fg(border_color))
            .title(Span::styled(
                title_text,
                if exit_hold_pct.is_some() {
                    Style::default()
                        .fg(Color::Rgb(255, 120, 120))
                        .add_modifier(Modifier::BOLD)
                } else {
                    Style::default().fg(border_color)
                },
            ));

        let input_box = Paragraph::new(input_lines_ui)
            .block(input_block)
            .wrap(ratatui::widgets::Wrap { trim: false });
        frame.render_widget(input_box, input_layout[1]);

        // --- Footer Status Bar (status_message truncated so it never collides with input) ---
        let engine_short = {
            let n = self.backend.name();
            if n.chars().count() > 36 {
                format!("{}…", n.chars().take(34).collect::<String>())
            } else {
                n
            }
        };
        let status_short = {
            let s = self.status_message.chars().take(48).collect::<String>();
            if self.status_message.chars().count() > 48 {
                format!("{s}…")
            } else {
                s
            }
        };
        let footer_text = Line::from(vec![
            Span::styled(
                format!(" {engine_short} "),
                Style::default()
                    .bg(if exit_hold_pct.is_some() {
                        Color::Rgb(180, 40, 40)
                    } else {
                        theme_color
                    })
                    .fg(Color::Black)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::raw(" "),
            Span::styled(status_short, Style::default().fg(Color::Rgb(200, 200, 200))),
            Span::raw(" │ "),
            Span::styled("Shift+Enter=↵", Style::default().fg(dark_gray)),
            Span::raw(" │ "),
            Span::styled(
                if self.selection_active() {
                    "Ctrl+C=copy selection"
                } else {
                    "Ctrl+C=interrupt"
                },
                Style::default().fg(if self.selection_active() {
                    Color::Rgb(120, 200, 255)
                } else {
                    dark_gray
                }),
            ),
        ]);
        frame.render_widget(Paragraph::new(footer_text), chunks[3]);

        let is_downloading = self.download_progress.lock().unwrap().is_some();
        if self.input_focused && !self.show_menu && !is_downloading {
            let inner_w = input_layout[1].width.saturating_sub(2).max(1) as usize;
            let (col, row) = self.input_cursor_col_row(inner_w);
            let max_row = input_layout[1].height.saturating_sub(3);
            let row = row.min(max_row);
            let cursor_x = input_layout[1].x.saturating_add(1).saturating_add(col);
            let cursor_y = input_layout[1].y.saturating_add(1).saturating_add(row);
            // Keep cursor inside the box
            let max_x = input_layout[1].x + input_layout[1].width.saturating_sub(2);
            let max_y = input_layout[1].y + input_layout[1].height.saturating_sub(2);
            frame.set_cursor_position((cursor_x.min(max_x), cursor_y.min(max_y)));
        }

        // --- Unified Menu Modal (Popup with KramaFrame Fade) ---
        if self.show_menu {
            let menu_fade_val = self.krama.get_progress_f32("menu_fade", 0);
            let menu_border_color = Color::Rgb(
                0,
                (255.0 * menu_fade_val) as u8,
                (128.0 * menu_fade_val) as u8,
            );

            let popup_layout = Layout::default()
                .direction(Direction::Vertical)
                .constraints(
                    [
                        Constraint::Percentage(15),
                        Constraint::Percentage(70),
                        Constraint::Percentage(15),
                    ]
                    .as_ref(),
                )
                .split(frame.area());

            let center_layout = Layout::default()
                .direction(Direction::Horizontal)
                .constraints(
                    [
                        Constraint::Percentage(15),
                        Constraint::Percentage(70),
                        Constraint::Percentage(15),
                    ]
                    .as_ref(),
                )
                .split(popup_layout[1]);

            let area = center_layout[1];
            frame.render_widget(Clear, area);

            let menu_block = Block::default()
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded)
                .border_style(Style::default().fg(menu_border_color))
                .title(" Menu Modal [ Tab / Left / Right: Switch Section | Esc: Close ] ");

            let inner_menu = menu_block.inner(area);
            frame.render_widget(menu_block, area);

            let menu_chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints([Constraint::Length(3), Constraint::Min(1)].as_ref())
                .split(inner_menu);

            // Tab bar headers inside menu
            let reg_style = if self.menu_section == 0 {
                Style::default()
                    .fg(Color::Black)
                    .bg(theme_color)
                    .add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(white)
            };
            let inst_style = if self.menu_section == 1 {
                Style::default()
                    .fg(Color::Black)
                    .bg(theme_color)
                    .add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(white)
            };
            let cfg_style = if self.menu_section == 2 {
                Style::default()
                    .fg(Color::Black)
                    .bg(theme_color)
                    .add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(white)
            };
            let rt_style = if self.menu_section == 3 {
                Style::default()
                    .fg(Color::Black)
                    .bg(theme_color)
                    .add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(white)
            };
            let perm_style = if self.menu_section == 4 {
                Style::default()
                    .fg(Color::Black)
                    .bg(theme_color)
                    .add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(white)
            };

            let tab_header = Paragraph::new(Line::from(vec![
                Span::raw(" "),
                Span::styled(" [1] Registry ", reg_style),
                Span::styled(" [2] Installed ", inst_style),
                Span::styled(" [3] Engine ", cfg_style),
                Span::styled(" [4] Runtime ", rt_style),
                Span::styled(" [5] Perms ", perm_style),
            ]));
            frame.render_widget(tab_header, menu_chunks[0]);

            if self.menu_section == 0 {
                // Model Registry with Live Search Bar
                let reg_chunks = Layout::default()
                    .direction(Direction::Vertical)
                    .constraints([Constraint::Length(3), Constraint::Min(1)].as_ref())
                    .split(menu_chunks[1]);

                let search_text = if self.registry_search_query.is_empty() {
                    Span::styled(
                        "Type to query HuggingFace / Ollama API live...",
                        Style::default().fg(dark_gray),
                    )
                } else {
                    Span::styled(
                        format!(" {}", self.registry_search_query),
                        Style::default().fg(theme_color),
                    )
                };
                let search_box = Paragraph::new(Line::from(search_text)).block(
                    Block::default()
                        .borders(Borders::ALL)
                        .border_type(BorderType::Rounded)
                        .title(" Live HuggingFace / Ollama Search Bar "),
                );
                frame.render_widget(search_box, reg_chunks[0]);

                let list_fade_val = self.krama.get_progress_f32("list_fade", 0);
                let list_item_color = Color::Rgb(
                    0,
                    (255.0 * list_fade_val) as u8,
                    (180.0 * list_fade_val) as u8,
                );

                let mut items: Vec<ListItem> = self
                    .registry_models
                    .iter()
                    .map(|m| ListItem::new(Span::styled(m, Style::default().fg(list_item_color))))
                    .collect();

                let hf_items = self.hf_models.iter().map(|m| {
                    let is_ollama = m.starts_with("Ollama:");
                    let color = if is_ollama {
                        list_item_color
                    } else {
                        Color::Rgb(
                            150,
                            (200.0 * list_fade_val) as u8,
                            (255.0 * list_fade_val) as u8,
                        )
                    };
                    ListItem::new(Span::styled(m, Style::default().fg(color)))
                });
                items.extend(hf_items);

                let list = List::new(items)
                    .block(Block::default().borders(Borders::TOP).title(
                        " Open Weights Registry [Up/Down: Navigate | Enter: Download & Install] ",
                    ))
                    .highlight_style(Style::default().bg(theme_color).fg(Color::Black))
                    .highlight_symbol(">> ");

                frame.render_stateful_widget(list, reg_chunks[1], &mut self.registry_state);
            } else if self.menu_section == 1 {
                // Installed Models Tab (Real Local Models with KramaFrame Fade)
                let list_fade_val = self.krama.get_progress_f32("list_fade", 0);
                let list_item_color = Color::Rgb(
                    0,
                    (255.0 * list_fade_val) as u8,
                    (180.0 * list_fade_val) as u8,
                );

                let items: Vec<ListItem> = self
                    .installed_models
                    .iter()
                    .map(|m| {
                        ListItem::new(Span::styled(
                            format!("Local Installed: {}", m),
                            Style::default().fg(list_item_color),
                        ))
                    })
                    .collect();

                let list = List::new(items)
                    .block(Block::default().borders(Borders::TOP).title(
                        " Installed Models [Up/Down: Navigate | Enter: Activate & Use Model] ",
                    ))
                    .highlight_style(Style::default().bg(theme_color).fg(Color::Black))
                    .highlight_symbol(">> ");

                frame.render_stateful_widget(list, menu_chunks[1], &mut self.installed_state);
            } else if self.menu_section == 2 {
                // Settings Configuration (no Burn/WGPU demo)
                let active_backend_str = self.backend.name();
                let options = vec![
                    ListItem::new(Span::styled(
                        "llama.cpp (in-process libllama.so — fast, no subprocess)",
                        Style::default().fg(light_blue),
                    )),
                    ListItem::new(Span::styled(
                        "llama.cpp (warm llama-server / CLI + GPU -ngl)",
                        Style::default().fg(Color::Rgb(255, 180, 80)),
                    )),
                    ListItem::new(Span::styled(
                        "Ollama Engine (Local daemon: http://localhost:11434)",
                        Style::default().fg(white),
                    )),
                ];
                let list = List::new(options)
                    .block(Block::default().borders(Borders::TOP).title(format!(
                        " Active Engine: {} [Enter to Select] ",
                        active_backend_str
                    )))
                    .highlight_style(Style::default().bg(theme_color).fg(Color::Black))
                    .highlight_symbol(">> ");

                frame.render_stateful_widget(list, menu_chunks[1], &mut self.config_state);
            } else if self.menu_section == 3 {
                // Runtime: power mode + context + repeat detector
                let s = crate::settings::get_settings();
                let ctx_n = crate::settings::context_token_limit();
                let ctx_label = crate::settings::format_context_tokens(ctx_n);
                let mk = |active: bool, label: &str| {
                    if active {
                        format!("● {}", label)
                    } else {
                        format!("○ {}", label)
                    }
                };
                let options = vec![
                    ListItem::new(Span::styled(
                        mk(
                            s.power_mode == crate::settings::PowerMode::PowerSaver,
                            "Power Saver — ease off when CPU hot",
                        ),
                        Style::default().fg(Color::Rgb(100, 220, 140)),
                    )),
                    ListItem::new(Span::styled(
                        mk(
                            s.power_mode == crate::settings::PowerMode::Normal,
                            "Normal (default) — auto cores + GPU offload",
                        ),
                        Style::default().fg(theme_color),
                    )),
                    ListItem::new(Span::styled(
                        mk(
                            s.power_mode == crate::settings::PowerMode::Extreme,
                            "Extreme — max threads + max GPU layers",
                        ),
                        Style::default().fg(Color::Rgb(255, 100, 100)),
                    )),
                    ListItem::new(Span::styled(
                        format!(
                            "llama.cpp Sub-Backend: {}  (Enter / +/− cycles Auto→SIMD→GPU→Scalar)",
                            s.llama_rs_sub_backend.label()
                        ),
                        Style::default().fg(Color::Rgb(100, 200, 255)),
                    )),
                    ListItem::new(Span::styled(
                        format!(
                            "Stall Watchdog Timeout: {}  (Enter / +/− cycles 5m→10m→20m→Unlimited)",
                            crate::settings::format_stall_timeout(s.stall_timeout_secs)
                        ),
                        Style::default().fg(Color::Rgb(255, 180, 80)),
                    )),
                    ListItem::new(Span::styled(
                        format!(
                            "Repeat threshold: {}  (+/− step · Enter cycle)",
                            s.repeat_threshold
                        ),
                        Style::default().fg(light_blue),
                    )),
                    ListItem::new(Span::styled(
                        format!(
                            "Repeat detector on thinking: {}  (Enter toggles)",
                            if s.repeat_detect_thinking {
                                "ON"
                            } else {
                                "OFF"
                            }
                        ),
                        Style::default().fg(Color::Rgb(255, 200, 80)),
                    )),
                    ListItem::new(Span::styled(
                        format!(
                            "Context window: {} ({ctx_n})  (+/− step · Enter cycle 4K…1M)",
                            ctx_label
                        ),
                        Style::default().fg(Color::Rgb(180, 160, 255)),
                    )),
                ];
                let list =
                    List::new(options)
                        .block(Block::default().borders(Borders::TOP).title(
                            " Runtime [Power | Ctx | Temp | Repeat] Enter=apply · +/−=nudge ",
                        ))
                        .highlight_style(Style::default().bg(theme_color).fg(Color::Black))
                        .highlight_symbol(">> ");
                frame.render_stateful_widget(list, menu_chunks[1], &mut self.runtime_state);
            } else {
                // Permissions tab
                let p = get_tool_permissions();
                let mode_ask = if p.mode == PermissionMode::Ask {
                    "● Ask user to allow (default)"
                } else {
                    "○ Ask user to allow"
                };
                let mode_always = if p.mode == PermissionMode::AlwaysAllow {
                    "● Always allow (tools may write/run)"
                } else {
                    "○ Always allow"
                };
                let scope_cur = if p.folder_scope == FolderScope::CurrentDir {
                    "● Interact on current dir only (safefolder)"
                } else {
                    "○ Interact on current dir only (safefolder)"
                };
                let scope_all = if p.folder_scope == FolderScope::AllDirs {
                    "● Interact on all directories"
                } else {
                    "○ Interact on all directories"
                };
                let options = vec![
                    ListItem::new(Span::styled(mode_ask, Style::default().fg(theme_color))),
                    ListItem::new(Span::styled(mode_always, Style::default().fg(light_blue))),
                    ListItem::new(Span::styled(
                        scope_cur,
                        Style::default().fg(Color::Rgb(255, 200, 80)),
                    )),
                    ListItem::new(Span::styled(scope_all, Style::default().fg(white))),
                ];
                let list = List::new(options)
                    .block(Block::default().borders(Borders::TOP).title(
                        " Tool Permissions [Enter: Apply]  Ask=block write/cmd until /allow ",
                    ))
                    .highlight_style(Style::default().bg(theme_color).fg(Color::Black))
                    .highlight_symbol(">> ");
                frame.render_stateful_widget(list, menu_chunks[1], &mut self.perms_state);
            }
        }

        // F1 keyboard shortcuts overlay (off by default; one-shot fade-in)
        if self.show_shortcuts {
            let fade = self.krama.get_progress_f32("help_fade", 0).clamp(0.0, 1.0);
            let alpha = (200.0 * fade) as u8;
            let fg = Color::Rgb(
                (220.0 * fade) as u8,
                (255.0 * fade) as u8,
                (230.0 * fade) as u8,
            );
            let border = Color::Rgb(0, alpha.saturating_add(40), (180.0 * fade) as u8);

            let popup_layout = Layout::default()
                .direction(Direction::Vertical)
                .constraints(
                    [
                        Constraint::Percentage(12),
                        Constraint::Percentage(76),
                        Constraint::Percentage(12),
                    ]
                    .as_ref(),
                )
                .split(frame.area());
            let center = Layout::default()
                .direction(Direction::Horizontal)
                .constraints(
                    [
                        Constraint::Percentage(15),
                        Constraint::Percentage(70),
                        Constraint::Percentage(15),
                    ]
                    .as_ref(),
                )
                .split(popup_layout[1]);
            let area = center[1];
            frame.render_widget(Clear, area);

            let help = concat!(
                " F1              Toggle this shortcuts panel (default: off)\n",
                " F2 / Ctrl+M     Menu (Registry / Installed / Engine / Permissions)\n",
                " F3 / Ctrl+L     Collapse activity log\n",
                " Ctrl+F          Focus / unfocus input\n",
                " Left / Right    Move cursor by character\n",
                " Alt+Left/Right  Move cursor by word\n",
                " Alt+Backspace   Delete previous word (also Ctrl+Backspace)\n",
                " Ctrl+Z          Undo input\n",
                " Ctrl+Y / Ctrl+Shift+Z   Redo input\n",
                " Home / End      Start / end of prompt\n",
                " Ctrl+C          Interrupt generation or clear input\n",
                " Ctrl+Enter      Force-send / interrupt generation\n",
                " Ctrl+T          Collapse/expand thinking block\n",
                " Esc (hold 1s)   Quit\n",
                "\n",
                " /allow          Grant write/cmd for this session (Ask permission mode)\n",
                " /compact        Compress history → memory, forget old turns (anti-hallucination)\n",
                " /gc             Alias for /compact\n",
                " /tasks          List background task manager jobs\n",
                " Ctrl+C          Interrupt generation + kill long-running tasks\n",
                " Menu→Permissions  Ask vs Always allow · safefolder current vs all dirs\n",
                "\n",
                " Long cmds (>10s) park in task manager; output returns when done.\n",
                " Generation idle 20s → auto-interrupt (stall / ctx overload).\n",
                "\n",
                " Downloads resume after network drops (Range). Re-run install to continue.\n",
            );
            let help_para = Paragraph::new(help).style(Style::default().fg(fg)).block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_type(BorderType::Rounded)
                    .border_style(Style::default().fg(border))
                    .title(" Keyboard Shortcuts [F1 to close] "),
            );
            frame.render_widget(help_para, area);
        }

        // --- Model Deletion Confirmation Modal ---
        if let Some(ref target) = self.delete_confirm_model {
            let popup_layout = Layout::default()
                .direction(Direction::Vertical)
                .constraints(
                    [
                        Constraint::Percentage(30),
                        Constraint::Percentage(40),
                        Constraint::Percentage(30),
                    ]
                    .as_ref(),
                )
                .split(frame.area());

            let center_layout = Layout::default()
                .direction(Direction::Horizontal)
                .constraints(
                    [
                        Constraint::Percentage(20),
                        Constraint::Percentage(60),
                        Constraint::Percentage(20),
                    ]
                    .as_ref(),
                )
                .split(popup_layout[1]);

            let area = center_layout[1];
            frame.render_widget(Clear, area);

            let block = Block::default()
                .borders(Borders::ALL)
                .border_type(BorderType::Double)
                .border_style(Style::default().fg(Color::Rgb(255, 80, 80)))
                .title(" Confirm Model Deletion (Agreement Required) ");

            let confirm_lines = vec![
                Line::from(Span::styled(
                    "Model Deletion Confirmation",
                    Style::default()
                        .fg(Color::Rgb(255, 80, 80))
                        .add_modifier(Modifier::BOLD),
                )),
                Line::from(""),
                Line::from(format!("Target Model: {}", target)),
                Line::from("Are you sure you want to delete this model weight from memory/disk?"),
                Line::from(""),
                Line::from(vec![
                    Span::styled(
                        " [Y] Confirm Delete ",
                        Style::default()
                            .bg(Color::Rgb(255, 50, 50))
                            .fg(Color::White)
                            .add_modifier(Modifier::BOLD),
                    ),
                    Span::raw("   "),
                    Span::styled(
                        " [N / Esc] Cancel ",
                        Style::default().bg(dark_gray).fg(Color::White),
                    ),
                ]),
            ];

            let dialog = Paragraph::new(confirm_lines)
                .block(block)
                .alignment(Alignment::Center);
            frame.render_widget(dialog, area);
        }
    }
}

/// Deterministic transcript compress for context budget (small models can't self-summarize reliably).
fn compress_transcript(archive: &str) -> String {
    let mut user_bits = Vec::new();
    let mut agent_bits = Vec::new();
    let mut tool_bits = Vec::new();
    let mut paths = Vec::new();
    let mut cmds = Vec::new();

    for raw in archive.lines() {
        let line = raw.trim();
        if line.is_empty() {
            continue;
        }
        if let Some(rest) = line.strip_prefix("You: ") {
            let t = rest.trim();
            if t.starts_with("[Tool result]") || t.starts_with("[Memory") {
                continue;
            }
            if user_bits.len() < 12 {
                user_bits.push(trunc_chars(t, 160));
            }
        } else if let Some(rest) = line.strip_prefix("Agent: ") {
            let t = rest.trim();
            if t.contains("[Generation") || t.contains("[Interrupted") {
                continue;
            }
            if agent_bits.len() < 8 {
                agent_bits.push(trunc_chars(t, 160));
            }
            // harvest tool tags
            if let Some(i) = t.find("<cmd>") {
                if let Some(j) = t[i..].find("</cmd>") {
                    cmds.push(trunc_chars(&t[i + 5..i + j], 80));
                }
            }
            if let Some(i) = t.find("src=\"") {
                let rest = &t[i + 5..];
                if let Some(j) = rest.find('"') {
                    paths.push(rest[..j].to_string());
                }
            }
        } else if line.starts_with("[tool]")
            || line.starts_with("error:")
            || line.starts_with("warning:")
        {
            if tool_bits.len() < 6 {
                tool_bits.push(trunc_chars(line, 120));
            }
        }
    }

    let mut out = String::new();
    out.push_str("Compressed session facts. Prior full chat is forgotten.\n");
    if !user_bits.is_empty() {
        out.push_str("User goals: ");
        out.push_str(&user_bits.join(" | "));
        out.push('\n');
    }
    if !agent_bits.is_empty() {
        out.push_str("Agent outcomes: ");
        out.push_str(&agent_bits.join(" | "));
        out.push('\n');
    }
    if !cmds.is_empty() {
        cmds.dedup();
        out.push_str("Commands seen: ");
        out.push_str(&cmds.join(" ; "));
        out.push('\n');
    }
    if !paths.is_empty() {
        paths.dedup();
        out.push_str("Paths: ");
        out.push_str(&paths.join(", "));
        out.push('\n');
    }
    if !tool_bits.is_empty() {
        out.push_str("Tool notes: ");
        out.push_str(&tool_bits.join(" | "));
        out.push('\n');
    }
    if out.len() < 40 {
        out.push_str(&trunc_chars(archive, 400));
    }
    trunc_chars(&out, 2000)
}

fn trunc_chars(s: &str, max: usize) -> String {
    let n = s.chars().count();
    if n <= max {
        s.to_string()
    } else {
        format!(
            "{}…",
            s.chars().take(max.saturating_sub(1)).collect::<String>()
        )
    }
}

/// Pad/truncate to exact display width (char count).
fn fit_width(s: &str, width: usize) -> String {
    if width == 0 {
        return String::new();
    }
    let n = s.chars().count();
    if n == width {
        s.to_string()
    } else if n > width {
        s.chars().take(width).collect()
    } else {
        format!("{s}{}", " ".repeat(width - n))
    }
}

/// Best-effort CPU package temperature (°C) via sysinfo thermal sensors.
fn cpu_package_temp_c(_sys: &sysinfo::System) -> f32 {
    use sysinfo::Components;
    let comps = Components::new_with_refreshed_list();
    let mut best: Option<f32> = None;
    for c in comps.iter() {
        let label = c.label().to_ascii_lowercase();
        let Some(t) = c.temperature() else {
            continue;
        };
        if !t.is_finite() || t <= 0.0 || t > 150.0 {
            continue;
        }
        let prefer = label.contains("package")
            || label.contains("tctl")
            || label.contains("cpu")
            || label.contains("coretemp")
            || label.contains("k10temp")
            || label.contains("acpitz");
        if prefer {
            best = Some(match best {
                Some(b) => b.max(t),
                None => t,
            });
        } else if best.is_none() {
            best = Some(t);
        }
    }
    best.unwrap_or(0.0)
}

/// Classic two-arm resource box — **exactly 16 cells per line**, including floor.
///
/// ```text
/// ╭[C: 64%  70C ]╮
/// ├[M: 53%  4.1G]┤
/// ┴──────────────╯
/// ```
fn fixed_resource_box(
    cpu_pct: f32,
    cpu_c: f32,
    mem_pct: f32,
    mem_gb: f64,
) -> (String, String, String) {
    const W: usize = 16;
    // Space before ] makes C always 16 (raw format is 15 without it).
    let c = format!(
        "╭[C:{:>3.0}% {:>3.0}C ]╮",
        cpu_pct.clamp(0.0, 100.0),
        cpu_c.clamp(0.0, 150.0)
    );
    let m = format!(
        "├[M:{:>3.0}% {:>4.1}G]┤",
        mem_pct.clamp(0.0, 100.0),
        mem_gb.clamp(0.0, 999.9)
    );
    let bot = format!("┴{}╯", "─".repeat(W - 2));
    assert_eq!(c.chars().count(), W, "C width: {c:?}");
    assert_eq!(m.chars().count(), W, "M width: {m:?}");
    assert_eq!(bot.chars().count(), W, "bot width: {bot:?}");
    assert!(c.ends_with('╮'));
    assert!(m.ends_with('┤'));
    assert!(bot.ends_with('╯'));
    (c, m, bot)
}

/// Full-width L2 floor: brand arm + mid ─ + stats bottom (same STATS_W).
///
/// `╰──────────┴────…────┴─────────────╯`
fn continuous_top_floor(full_w: usize, left_w: usize, stats_w: usize) -> String {
    if full_w < 4 {
        return "─".repeat(full_w);
    }
    let brand = 12usize; // ╰──────────┴
    let left_w = left_w.max(brand);
    let stats_w = stats_w.min(full_w.saturating_sub(brand + 1)).max(3);
    let mid_w = full_w.saturating_sub(left_w).saturating_sub(stats_w);

    let mut s = String::with_capacity(full_w);
    s.push_str("╰──────────┴"); // 12 — opens right under Hercules
    if left_w > brand {
        s.push_str(&"─".repeat(left_w - brand));
    }
    s.push_str(&"─".repeat(mid_w));
    // Stats bottom: must match fixed_resource_box bot exactly
    s.push('┴');
    s.push_str(&"─".repeat(stats_w.saturating_sub(2)));
    s.push('╯');
    fit_width(&s, full_w)
}
