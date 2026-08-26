use crate::agent::{
    FolderScope, PermissionMode, ProposedAction, allow_session_tools, get_tool_permissions,
    set_folder_scope, set_permission_mode,
};
use crate::backend::{AgentBackend, LlamaCppLibBackend, OllamaBackend};
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

#[derive(Debug, Clone, Copy)]
pub struct CodeBlockToggleHit {
    pub block_idx: usize,
    pub screen_y: i32,
    pub normal_x: (u16, u16),
    pub preview_x: (u16, u16),
}

#[derive(Debug, Clone)]
pub struct CodeBlockCopyHit {
    pub block_idx: usize,
    pub screen_y: i32,
    pub copy_x: (u16, u16),
    pub code_body: String,
}

#[derive(Debug, Clone, Copy)]
pub struct CodeBlockScrollHit {
    pub block_idx: usize,
    pub screen_y: i32,
    pub left_btn_x: (u16, u16),
    pub track_x: (u16, u16),
    pub right_btn_x: (u16, u16),
    pub max_scroll: usize,
}

pub const NORDIC_BG: Color = Color::Rgb(46, 52, 64);        // #2E3440 Polar Night
pub const NORDIC_DARK_BG: Color = Color::Rgb(36, 41, 51);   // #242933
pub const NORDIC_CARD_BG: Color = Color::Rgb(59, 66, 82);   // #3B4252
pub const NORDIC_TEXT: Color = Color::Rgb(236, 239, 244);   // #ECEFF4 Snow Storm
pub const NORDIC_MUTED: Color = Color::Rgb(129, 161, 193);  // #81A1C1 Frost Blue
pub const NORDIC_ACCENT: Color = Color::Rgb(136, 192, 208); // #88C0D0 Frost Cyan

pub fn interpolate_rgb(c1: (u8, u8, u8), c2: (u8, u8, u8), t: f32) -> (u8, u8, u8) {
    let t = t.clamp(0.0, 1.0);
    let r = (c1.0 as f32 + (c2.0 as f32 - c1.0 as f32) * t).round() as u8;
    let g = (c1.1 as f32 + (c2.1 as f32 - c1.1 as f32) * t).round() as u8;
    let b = (c1.2 as f32 + (c2.2 as f32 - c1.2 as f32) * t).round() as u8;
    (r, g, b)
}

pub fn multi_stop_gradient(stops: &[(f32, (u8, u8, u8))], pos: f32) -> Color {
    let pos = pos.clamp(0.0, 1.0);
    if stops.is_empty() {
        return Color::White;
    }
    if stops.len() == 1 || pos <= stops[0].0 {
        let (r, g, b) = stops[0].1;
        return Color::Rgb(r, g, b);
    }
    for i in 0..stops.len() - 1 {
        let (p1, c1) = stops[i];
        let (p2, c2) = stops[i + 1];
        if pos >= p1 && pos <= p2 {
            let seg_t = if (p2 - p1).abs() < 0.0001 { 0.0 } else { (pos - p1) / (p2 - p1) };
            let (r, g, b) = interpolate_rgb(c1, c2, seg_t);
            return Color::Rgb(r, g, b);
        }
    }
    let (r, g, b) = stops.last().unwrap().1;
    Color::Rgb(r, g, b)
}

pub fn get_status_gradient_stops(
    is_generating: bool,
    is_thinking: bool,
    exit_hold_pct: Option<f64>,
) -> Option<Vec<(f32, (u8, u8, u8))>> {
    if let Some(pct) = exit_hold_pct {
        let p = pct as f32;
        let r_mid = (180.0 + 75.0 * p).min(255.0) as u8;
        return Some(vec![
            (0.0, (140, 20, 30)),
            (0.5, (r_mid, 30, 40)),
            (1.0, (255, 50, 50)),
        ]);
    }
    if is_generating {
        if is_thinking {
            // Thinking: linear-gradient(rgba(0, 8, 117, 1) 1%, rgba(200, 75, 250, 1) 51%, rgba(65, 0, 217, 1) 100%)
            Some(vec![
                (0.01, (0, 8, 117)),
                (0.51, (200, 75, 250)),
                (1.00, (65, 0, 217)),
            ])
        } else {
            // Streaming: linear-gradient(rgba(6, 0, 181, 1) 0%, rgba(64, 255, 220, 1) 50%, rgba(0, 33, 196, 1) 100%)
            Some(vec![
                (0.0, (6, 0, 181)),
                (0.5, (64, 255, 220)),
                (1.0, (0, 33, 196)),
            ])
        }
    } else {
        // Idle: White static color (no continuous animation)
        None
    }
}

pub struct App {
    pub should_quit: bool,
    pub status_message: String,

    // Chat state
    pub input: String,
    pub messages: Vec<String>,
    pub session_id: Option<String>,
    pub backend: AgentBackend,
    pub input_anim_height: f32,
    pub model_badge_hit: Option<(u16, u16, u16)>,
    pub input_scroll_y: u16,

    // Registry state
    pub manager: ModelManager,
    pub registry_models: Vec<String>,
    pub registry_state: ListState,

    // System stats
    pub sys: System,

    // Config & Navigation
    pub theme_color: Color,
    pub show_menu: bool,
    pub menu_section: usize, // 0: Help, 1: Registry, 2: Modal (Installed), 3: Settings
    pub header_dropdown_open: bool,
    pub header_anim_progress: f32, // 0.0 = top header at row 0, 1.0 = top header at row 1 (revealing menu on row 0)
    pub menu_anim_progress: f32,   // 0.0 = closed, 1.0 = fully open (fade up/down)
    pub menu_closing: bool,        // true during closing animation
    pub header_bar_hit: Option<(u16, u16, u16)>, // (row, start_col, end_col)
    pub menu_tab_hits: Vec<(usize, u16, u16)>,   // (section_idx, start_col, end_col) on row 0
    pub container_close_hit: Option<(u16, u16, u16)>, // (row, start_col, end_col) for " x " close button
    pub settings_col: usize,       // 0 = left category tabs, 1 = right values
    pub settings_tab: usize,       // 0: Power, 1: Stall, 2: Repeat, 3: Context, 4: Permissions, 5: HF Token
    pub hf_token_input: String,
    pub hf_token_editing: bool,
    pub registry_tab: usize,       // 0: HuggingFace, 1: Ollama
    pub config_state: ListState,

    // Installed Models state
    pub installed_models: Vec<String>,
    pub installed_state: ListState,

    // Focus & Scroll state
    pub input_focused: bool,
    pub input_cursor_position: usize,
    pub scroll_offset: u16,
    pub auto_scroll_enabled: bool,
    pub table_scroll_x: usize,
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
    /// Visual start offset of each logical line (for accurate mouse selection mapping)
    pub last_chat_visual_at: Vec<u16>,
    /// Active preview mode toggles per code block index
    pub code_block_previews: std::collections::HashSet<usize>,
    /// Interactive Normal/Preview toggle hit zones
    pub code_block_hits: Vec<CodeBlockToggleHit>,
    /// Interactive Copy button hit zones
    pub code_block_copy_hits: Vec<CodeBlockCopyHit>,
    /// Height transition animations per code block (to_preview, start_time)
    pub code_block_anims: std::collections::HashMap<usize, (bool, std::time::Instant)>,
    /// Horizontal scroll offsets per code block preview (code_block_idx -> scroll_x)
    pub code_block_scrolls: std::collections::HashMap<usize, usize>,
    /// Interactive preview horizontal scroll bar hit zones
    pub code_block_scroll_hits: Vec<CodeBlockScrollHit>,
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
    pub last_metrics_time: std::time::Instant,

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
    /// Pending sub-agent messages queued while host/agent was busy streaming
    pub pending_agent_messages: Vec<crate::task_manager::TaskEvent>,
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
        let cwd = std::env::current_dir().unwrap_or_else(|_| std::path::PathBuf::from("."));
        let default_sid = crate::session::new_session_id_for_dir(&cwd);
        let app = Self {
            should_quit: false,
            status_message: "Ready.".to_string(),
            input: String::new(),
            messages: vec!["System: Welcome to Hercules. Ask me anything!".to_string()],
            session_id: Some(default_sid),
            input_anim_height: 3.0,
            model_badge_hit: None,
            input_scroll_y: 0,
            backend: {
                // Prefer local GGUF with llama.rs (pure Rust); else llama.cpp if path exists
                let mgr = ModelManager::new();
                if let Some(path) = mgr.latest_gguf_path() {
                    AgentBackend::LlamaCppLib(LlamaCppLibBackend::gguf(path))
                } else {
                    AgentBackend::LlamaCppLib(LlamaCppLibBackend::http(
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
            header_dropdown_open: false,
            header_anim_progress: 0.0,
            menu_anim_progress: 0.0,
            menu_closing: false,
            header_bar_hit: None,
            menu_tab_hits: Vec::new(),
            container_close_hit: None,
            settings_col: 0,
            settings_tab: 0,
            hf_token_input: String::new(),
            hf_token_editing: false,
            registry_tab: 0,
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
            table_scroll_x: 0,
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
            last_chat_visual_at: Vec::new(),
            code_block_previews: std::collections::HashSet::new(),
            code_block_hits: Vec::new(),
            code_block_copy_hits: Vec::new(),
            code_block_anims: std::collections::HashMap::new(),
            code_block_scrolls: std::collections::HashMap::new(),
            code_block_scroll_hits: Vec::new(),
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
            last_frame_time: std::time::Instant::now(),
            last_metrics_time: std::time::Instant::now(),
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
            pending_agent_messages: Vec::new(),
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

    pub fn with_session(session: crate::session::Session) -> Self {
        let mut app = Self::new();
        let sid = session.session_id.clone();
        if !session.messages.is_empty() {
            app.messages = session.messages;
            // Clean up transient messages (resumed lines, stall warnings)
            app.messages.retain(|m| {
                !m.starts_with("System: Resumed session ")
                    && !m.starts_with("System: [STALL]")
                    && !m.starts_with("System: [CTRL+C]")
            });
            for m in &mut app.messages {
                if m.starts_with("Agent: ") {
                    if let Some(pos) = m.find("\n[Generation stalled") {
                        m.truncate(pos);
                    }
                    if let Some(pos) = m.find("\n[STALL") {
                        m.truncate(pos);
                    }
                    if let Some(pos) = m.find("\n[Generation Interrupted") {
                        m.truncate(pos);
                    }
                }
            }
            // Auto-activate the model used in this session if recorded
            for m in &app.messages {
                if m.starts_with("System: Switched active engine model to ") {
                    if let Some(open) = m.find('[') {
                        if let Some(close) = m.rfind(']') {
                            let path_str = &m[open + 1..close];
                            let path = std::path::PathBuf::from(path_str);
                            if path.exists() {
                                app.backend = AgentBackend::LlamaCppLib(
                                    LlamaCppLibBackend::gguf(path.clone()),
                                );
                                app.manager.set_active_gguf_path(path.display().to_string());
                                break;
                            }
                        }
                    }
                }
            }
            app.messages.push(format!("System: Resumed session {}", sid));
        }
        app.status_message = format!("Resumed session {}", sid);
        app.session_id = Some(sid);
        app
    }

    pub fn save_current_session(&self) {
        if let Some(ref sid) = self.session_id {
            // Filter out transient messages
            let persistent_messages: Vec<String> = self
                .messages
                .iter()
                .filter(|m| !m.starts_with("System: Resumed session "))
                .cloned()
                .collect();

            // Do not store empty sessions that only contain the initial greeting / system messages
            let has_user_or_agent = persistent_messages
                .iter()
                .any(|m| m.starts_with("You: ") || m.starts_with("Agent: "));

            if !has_user_or_agent {
                return;
            }

            let working_dir = std::env::current_dir()
                .unwrap_or_else(|_| std::path::PathBuf::from("."))
                .to_string_lossy()
                .to_string();
            let mut session = crate::session::Session::new(sid.clone(), working_dir);
            session.messages = persistent_messages;
            let _ = crate::session::save_session(&session);
        }
    }

    /// Returns list of available models for subagents, capped to the current backend's repository.
    pub fn available_swarm_models(&self) -> Vec<String> {
        match &self.backend {
            AgentBackend::Ollama(_) => {
                self.installed_models
                    .iter()
                    .filter(|m| m.starts_with("Ollama:"))
                    .map(|m| m.trim_start_matches("Ollama:").trim().to_string())
                    .collect()
            }
            AgentBackend::LlamaCppLib(_) => {
                self.installed_models
                    .iter()
                    .filter(|m| !m.starts_with("Ollama:"))
                    .cloned()
                    .collect()
            }
            #[cfg(feature = "gpu")]
            AgentBackend::BurnWgpu(_) => vec![],
        }
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
                    pending: false, spawned: false,
                    rect: None,
                    anchor_msg: anchor,
                    expanded: false,
                    anim_start: None,
                });
                auto_open = Some(id);
            }
        }
        self.dedupe_tool_chips();

        let mut instant_cmds = Vec::new();
        let mut instant_mcps = Vec::new();
        let perms = crate::agent::get_tool_permissions();
        let can_instant = perms.session_allow || perms.mode == crate::agent::PermissionMode::AlwaysAllow;
        if can_instant {
            for chip in self.tool_chips.iter_mut() {
                if chip.kind == tool_panel::ToolPanelKind::Cmd && chip.tag_closed && !chip.spawned {
                    chip.spawned = true;
                    instant_cmds.push(crate::agent::ProposedAction {
                        kind: crate::agent::ProposedKind::Cmd,
                        target: chip.target.clone(),
                        body: String::new(),
                        line_attr: None,
                        from_think: false,
                        chip_id: Some(chip.id),
                    });
                } else if matches!(chip.kind, tool_panel::ToolPanelKind::Mcp | tool_panel::ToolPanelKind::Skill | tool_panel::ToolPanelKind::WebSearch | tool_panel::ToolPanelKind::Agent) && chip.tag_closed && !chip.spawned {
                    chip.spawned = true;
                    let pkind = if chip.kind == tool_panel::ToolPanelKind::Mcp { crate::agent::ProposedKind::Mcp } else if chip.kind == tool_panel::ToolPanelKind::Skill { crate::agent::ProposedKind::Skill } else if chip.kind == tool_panel::ToolPanelKind::Agent { crate::agent::ProposedKind::Agent } else { crate::agent::ProposedKind::WebSearch };
                    instant_mcps.push(crate::agent::ProposedAction {
                        kind: pkind,
                        target: chip.target.clone(),
                        body: chip.body.clone(),
                        line_attr: None,
                        from_think: false,
                        chip_id: Some(chip.id),
                    });
                }
            }
        }
        if !instant_cmds.is_empty() {
            self.spawn_cmds_to_task_manager(instant_cmds);
        }
        if !instant_mcps.is_empty() {
            for a in instant_mcps {
                if a.kind == crate::agent::ProposedKind::Agent {
                    let role = crate::agent::AgentEngine::extract_attribute(&a.target, "role").unwrap_or_default();
                    let to = crate::agent::AgentEngine::extract_attribute(&a.target, "to").unwrap_or_default();
                    let model = crate::agent::AgentEngine::extract_attribute(&a.target, "model").unwrap_or_default();
                    let sub_backend = self.backend.with_model(&model, &self.manager);
                    let instruction = a.body.clone();
                    let agent_id = self.task_manager.spawn_agent(
                        sub_backend,
                        role.clone(),
                        to,
                        model.clone(),
                        instruction,
                        0 // spawned_by host/orchestrator
                    );
                    let model_label = if model.is_empty() { String::new() } else { format!(" [model={model}]") };
                    if let Some(chip) = self.tool_chips.iter_mut().find(|c| c.id == a.chip_id.unwrap()) {
                        chip.pending = false;
                        chip.body = format!("[Agent Task #{agent_id} ({role}{model_label}) spawning]\n(waiting for reply…)");
                    }
                    continue;
                }
                
                let result = crate::agent::AgentEngine::execute_proposed(&a);
                if let Some(chip) = self.tool_chips.iter_mut().find(|c| c.id == a.chip_id.unwrap()) {
                    chip.pending = false;
                    chip.body = result.clone();
                }
                self.tool_result_context.push(format!("[{} result]\n{}", a.kind.label(), result));
            }
        }

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
        let id = self.task_manager.spawn_cmd(cmd.clone(), 0);
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
            .map(|a| a.y.saturating_add(1) as i32)
            .unwrap_or(1);
        // Prefer plain lines from last draw (matches shaded rows accurately with wrapping)
        let mut extracted: Vec<String> = Vec::new();
        if !self.last_chat_plain_lines.is_empty() {
            for (i, line) in self.last_chat_plain_lines.iter().enumerate() {
                let vis_start = self.last_chat_visual_at.get(i).copied().unwrap_or(i as u16) as i32;
                let vis_count = if i + 1 < self.last_chat_visual_at.len() {
                    (self.last_chat_visual_at[i + 1] - self.last_chat_visual_at[i]) as i32
                } else {
                    1
                };
                let screen_y_start = chat_y + vis_start - self.scroll_offset as i32;
                let screen_y_end = screen_y_start + vis_count - 1;
                if max_y >= screen_y_start && min_y <= screen_y_end {
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
                // Feature removed
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


    /// Map char cursor index → (col, row) inside the input box content width.
    fn input_cursor_col_row(&self, content_width: usize) -> (u16, u16) {
        let width = content_width.max(1);
        let pos = self.input_cursor_position.min(self.input.chars().count());
        let prefix: String = self.input.chars().take(pos).collect();
        let mut row: u16 = 0;
        let mut col: u16 = 0;
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

    fn char_pos_from_col_row(&self, target_col: u16, target_row: u16, content_width: usize) -> usize {
        let width = content_width.max(1);
        let mut cur_row: u16 = 0;
        let mut cur_col: u16 = 0;
        let mut best_pos = 0;
        let total_chars = self.input.chars().count();

        for (pos, ch) in self.input.chars().enumerate() {
            if cur_row == target_row && cur_col == target_col {
                return pos;
            }
            if cur_row == target_row {
                best_pos = pos;
            }
            if ch == '\n' {
                if cur_row == target_row {
                    return pos;
                }
                cur_row = cur_row.saturating_add(1);
                cur_col = 0;
            } else {
                cur_col = cur_col.saturating_add(1);
                if cur_col as usize >= width {
                    if cur_row == target_row {
                        return pos;
                    }
                    cur_row = cur_row.saturating_add(1);
                    cur_col = 0;
                }
            }
        }
        if cur_row == target_row {
            total_chars
        } else if cur_row < target_row {
            total_chars
        } else {
            best_pos
        }
    }

    fn input_cursor_up(&mut self, content_width: usize) {
        let (col, row) = self.input_cursor_col_row(content_width);
        if row > 0 {
            self.input_cursor_position = self.char_pos_from_col_row(col, row - 1, content_width);
        } else {
            self.input_cursor_position = 0;
        }
    }

    fn input_cursor_down(&mut self, content_width: usize) {
        let (col, row) = self.input_cursor_col_row(content_width);
        let total_pos = self.char_pos_from_col_row(col, row + 1, content_width);
        self.input_cursor_position = total_pos;
    }

    /// Queue write/cmd for user accept; open preview panel.
    fn propose_actions(&mut self, mut actions: Vec<ProposedAction>) {
        if actions.is_empty() {
            return;
        }

        actions.retain(|a| {
            let pkind = match a.kind {
                crate::agent::ProposedKind::Cmd => tool_panel::ToolPanelKind::Cmd,
                crate::agent::ProposedKind::Write => tool_panel::ToolPanelKind::Write,
                crate::agent::ProposedKind::Mcp => tool_panel::ToolPanelKind::Mcp,
                crate::agent::ProposedKind::Skill => tool_panel::ToolPanelKind::Skill,
                crate::agent::ProposedKind::WebSearch => tool_panel::ToolPanelKind::WebSearch,
                crate::agent::ProposedKind::Agent => tool_panel::ToolPanelKind::Agent,
            };
            !self.tool_chips.iter().any(|c| {
                c.kind == pkind
                && tool_panel::same_tool_target(c.kind, &c.target, &a.target)
                && c.spawned
            })
        });

        if actions.is_empty() {
            return;
        }

        // Ensure chips exist + mark pending; auto-open latest
        for a in &mut actions {
            let target = match a.kind {
                crate::agent::ProposedKind::Write => {
                    crate::agent::AgentEngine::expand_path(&a.target)
                        .display()
                        .to_string()
                }
                _ => a.target.clone(),
            };
            let kind = match a.kind {
                crate::agent::ProposedKind::Write => ToolPanelKind::Write,
                crate::agent::ProposedKind::Cmd => ToolPanelKind::Cmd,
                crate::agent::ProposedKind::Mcp => ToolPanelKind::Mcp,
                crate::agent::ProposedKind::Skill => ToolPanelKind::Skill,
                crate::agent::ProposedKind::WebSearch => ToolPanelKind::WebSearch,
                crate::agent::ProposedKind::Agent => ToolPanelKind::Agent,
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
            } else {
                let id = self.next_chip_id;
                self.next_chip_id += 1;

                self.tool_chips.push(ToolChip {
                    id,
                    kind,
                    target,
                    body: a.body.clone(),
                    tag_closed: true,
                    pending: true, spawned: false,
                    rect: None,
                    anchor_msg: anchor,
                    expanded: false,
                    anim_start: None,
                });

                a.chip_id = Some(id);
            }
        }
        self.dedupe_tool_chips();
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
        let mut mcps = Vec::new();
        let mut cmds = Vec::new();
        for a in actions {
            match a.kind {
                crate::agent::ProposedKind::Write => writes.push(a),
                crate::agent::ProposedKind::Cmd => cmds.push(a),
                crate::agent::ProposedKind::Mcp => mcps.push(a),
                crate::agent::ProposedKind::Skill => mcps.push(a),
                crate::agent::ProposedKind::WebSearch => mcps.push(a),
                crate::agent::ProposedKind::Agent => mcps.push(a),
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


        if !mcps.is_empty() {
            let mut agent_ids = Vec::new();
            for a in &mcps {
                if a.kind == crate::agent::ProposedKind::Agent {
                    let role = crate::agent::AgentEngine::extract_attribute(&a.target, "role").unwrap_or_default();
                    let to = crate::agent::AgentEngine::extract_attribute(&a.target, "to").unwrap_or_default();
                    let model = crate::agent::AgentEngine::extract_attribute(&a.target, "model").unwrap_or_default();
                    let sub_backend = self.backend.with_model(&model, &self.manager);
                    let instruction = a.body.clone();
                    let agent_id = self.task_manager.spawn_agent(
                        sub_backend,
                        role.clone(),
                        to,
                        model.clone(),
                        instruction,
                        0 // spawned_by host/orchestrator
                    );
                    agent_ids.push(agent_id);
                    let model_label = if model.is_empty() { String::new() } else { format!(" [model={model}]") };
                    if let Some(chip_id) = a.chip_id {
                        if let Some(chip) = self.tool_chips.iter_mut().find(|c| c.id == chip_id) {
                            chip.pending = false;
                            chip.tag_closed = true;
                            chip.body = format!("[Agent Task #{agent_id} ({role}{model_label}) spawning]\n(waiting for reply…)");
                        }
                    }
                    continue;
                }

                let result = crate::agent::AgentEngine::execute_proposed(a);
                if let Some(chip_id) = a.chip_id {
                    if let Some(chip) = self.tool_chips.iter_mut().find(|c| c.id == chip_id) {
                        chip.pending = false;
                        chip.tag_closed = true;
                        chip.body = result.clone();
                    }
                    self.force_open_panel_from_chip(chip_id);
                }
                self.tool_result_context.push(format!("[{} result]\n{}", a.kind.label(), result));
            }
            if !*self.is_generating.lock().unwrap() && agent_ids.is_empty() {
                self.trigger_generation_from_context();
            }
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
            let id = self.task_manager.spawn_cmd(cmd.clone(), 0);
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
                    pending: false, spawned: false,
                    rect: None,
                    anchor_msg: self.latest_agent_msg_idx(),
                    expanded: false,
                    anim_start: None,
                });
            }
            self.messages.push(format!(
                "System: [Task #{id} (Agent 0)] started: `{cmd}` (if >{QUICK_SECS}s → task manager; Ctrl+C kills)"
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
    /// If the agent is currently busy streaming, messages are queued in `pending_agent_messages`
    /// and delivered cleanly once streaming finishes.
    fn poll_task_events(&mut self) {
        let is_busy = *self.is_generating.lock().unwrap();

        // 1. Ingest new events from task manager
        let new_events = self.task_manager.take_events();
        for ev in new_events {
            if is_busy {
                if let TaskEvent::Done { id, cmd, output: _, killed: _, spawned_by } = &ev {
                    if let Some(chip) = self.tool_chips.iter_mut().rev().find(|c| {
                        (c.kind == ToolPanelKind::Cmd || c.kind == ToolPanelKind::Agent)
                            && tool_panel::same_tool_target(c.kind, &c.target, cmd)
                    }) {
                        chip.body = format!("[Task #{id} DONE (Agent {spawned_by}) — queued in inbox]\n(waiting for current response to finish)");
                    }
                }
                self.pending_agent_messages.push(ev);
            } else {
                self.deliver_task_event(ev);
            }
        }

        // 2. If agent is free and queued messages exist, deliver them now
        if !is_busy && !self.pending_agent_messages.is_empty() {
            let queued = std::mem::take(&mut self.pending_agent_messages);
            for ev in queued {
                self.deliver_task_event(ev);
            }
        }

        // 3. Live-update TERM panel body from running tasks
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

    fn deliver_task_event(&mut self, ev: TaskEvent) {
        match ev {
            TaskEvent::Parked { id, cmd, spawned_by } => {
                self.messages.push(format!(
                    "System: [Task #{id} (Agent {spawned_by})] still running after {QUICK_SECS}s — pushed to task manager. \
                     Command: `{cmd}`. Agent may continue; output arrives when finished. \
                     Ctrl+C kills running tasks."
                ));
                self.tool_result_context.push(format!(
                    "[Task #{id} PARKED — still running]\ncmd: {cmd}\n\
                     Do not re-run this command. Wait for [Task #{id} DONE (Agent {spawned_by})] or start other work."
                ));
                if let Some(chip) = self.tool_chips.iter_mut().rev().find(|c| {
                    (c.kind == ToolPanelKind::Cmd || c.kind == ToolPanelKind::Agent)
                        && tool_panel::same_tool_target(c.kind, &c.target, &cmd)
                }) {
                    chip.body = format!(
                        "[Task #{id} — long running >{QUICK_SECS}s]\n$ {cmd}\n(waiting… Ctrl+C to kill)"
                    );
                }
                if !*self.is_generating.lock().unwrap() {
                    self.auto_tool_turns += 1;
                    if self.auto_tool_turns == 20 {
                        self.messages.push(
                            "System: [Agent has taken 20 tool turns — press Ctrl+C to stop]"
                                .to_string(),
                        );
                    }
                    self.trigger_generation_from_context();
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
                spawned_by,
            } => {
                let label = if killed { "KILLED" } else { "DONE" };
                let pretty = tool_panel::format_tool_output_for_chat(&output);
                if let Some(chip) = self.tool_chips.iter_mut().rev().find(|c| {
                    (c.kind == ToolPanelKind::Cmd || c.kind == ToolPanelKind::Agent)
                        && tool_panel::same_tool_target(c.kind, &c.target, &cmd)
                }) {
                    chip.body = format!(
                        "[Task #{id} {label} (Agent {spawned_by})]\n$ {cmd}\n{pretty}"
                    );
                    chip.tag_closed = true;
                    chip.pending = false;
                    let cid = chip.id;
                    self.force_open_panel_from_chip(cid);
                }
                self.tool_result_context.push(format!(
                    "[Task #{id} {label} (Agent {spawned_by})]\ncmd: {cmd}\n\n{pretty}\n\n\
                     Use this output. Do not re-run the same command unless needed."
                ));
                if self.tool_result_context.len() > 8 {
                    let n = self.tool_result_context.len() - 8;
                    self.tool_result_context.drain(0..n);
                }
                self.messages.push(format!(
                    "System: [Task #{id} {label} (Agent {spawned_by})] `{cmd}` ({} lines) — result delivered to agent",
                    pretty.lines().count()
                ));
                if let Ok(mut l) = self.activity_logs.lock() {
                    l.push(format!("[TASK #{id}] {label} ({} bytes)", pretty.len()));
                }
                self.status_message = format!("Task #{id} {label} (Agent {spawned_by})");
                if !killed && !*self.is_generating.lock().unwrap() {
                    self.auto_tool_turns += 1;
                    if self.auto_tool_turns == 20 {
                        self.messages.push(
                            "System: [Agent has taken 20 tool turns — press Ctrl+C to stop]"
                                .to_string(),
                        );
                    }
                    self.trigger_generation_from_context();
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
            crate::settings::get_settings().stall_timeout_secs.clamp(60, 300)
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
            let last_user = self
                .messages
                .iter()
                .rev()
                .find_map(|m| m.strip_prefix("You: ").map(|s| s.to_string()))
                .unwrap_or_default();
            if crate::agent::AgentEngine::wants_plan_first(&last_user) {
                out.push_str(
                    "\n\nInstruction: Give a clear multi-step PLAN in natural language. \
                     Do NOT emit tool tags. Do NOT ls again.",
                );
            } else if crate::agent::AgentEngine::wants_implement(&last_user)
                || last_user.to_ascii_lowercase().contains("start coding")
            {
                out.push_str(
                    "\n\nInstruction: Directory listing (if any) is above. \
                     Now IMPLEMENT: emit <write src=\"...\"> with full file content. \
                     Do NOT emit <ls> again. One primary app file is fine to start.",
                );
            } else {
                out.push_str(
                    "\n\nInstruction: Tool results are above (Result: blocks). \
                     Reply in natural language only — tell the user what you found. \
                     Do NOT emit any tool tags (<read>, <ls>, <write>, <cmd>). \
                     Do NOT say you lack file access. Open chips already show full content.",
                );
            }
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
        } else {
            // Result without a prior chip — create one under last agent
            let id = self.next_chip_id;
            self.next_chip_id += 1;
            let target = match want_kind {
                ToolPanelKind::Cmd => "command".into(),
                ToolPanelKind::Read => "file".into(),
                ToolPanelKind::Write => "file".into(),
                _ => String::new(),
            };
            self.tool_chips.push(ToolChip {
                id,
                kind: want_kind,
                target,
                body: pretty.clone(),
                tag_closed: true,
                pending: false, spawned: false,
                rect: None,
                anchor_msg: anchor,
                expanded: false,
                anim_start: None,
            });
        }

        self.dedupe_tool_chips();
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

        self.messages.push("Agent: ".to_string());
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

    pub fn adjust_setting_value(&mut self, dir: i32) {
        use crate::settings::{
            PowerMode, cycle_context_token_limit, cycle_repeat_threshold,
            cycle_stall_timeout, format_context_tokens, format_stall_timeout,
            get_settings, set_power_mode, toggle_repeat_thinking,
        };
        use crate::app::{
            get_tool_permissions, set_permission_mode, set_folder_scope,
            PermissionMode, FolderScope,
        };
        match self.settings_tab {
            0 => {
                // Power mode
                let s = get_settings();
                let next_mode = if dir > 0 {
                    match s.power_mode {
                        PowerMode::PowerSaver => PowerMode::Normal,
                        PowerMode::Normal => PowerMode::Extreme,
                        PowerMode::Extreme => PowerMode::PowerSaver,
                    }
                } else {
                    match s.power_mode {
                        PowerMode::PowerSaver => PowerMode::Extreme,
                        PowerMode::Normal => PowerMode::PowerSaver,
                        PowerMode::Extreme => PowerMode::Normal,
                    }
                };
                set_power_mode(next_mode);
                crate::llama::server::shutdown_managed_server();
                self.status_message = format!("Power mode: {}", next_mode.label());
            }
            1 => {
                // Stall time
                let t = cycle_stall_timeout();
                self.status_message = format!("Stall Watchdog Timeout: {}", format_stall_timeout(t));
            }
            2 => {
                // Repeat detector
                if dir != 0 {
                    cycle_repeat_threshold();
                } else {
                    toggle_repeat_thinking();
                }
                let s = get_settings();
                self.status_message = format!(
                    "Repeat threshold: {} | Think detect: {}",
                    s.repeat_threshold,
                    if s.repeat_detect_thinking { "ON" } else { "OFF" }
                );
            }
            3 => {
                // Context window
                let n = cycle_context_token_limit();
                crate::llama::server::shutdown_managed_server();
                self.status_message = format!("Context limit: {}", format_context_tokens(n));
            }
            4 => {
                // Permissions
                let p = get_tool_permissions();
                if dir > 0 {
                    let next_mode = match p.mode {
                        PermissionMode::Ask => PermissionMode::AlwaysAllow,
                        PermissionMode::AlwaysAllow => PermissionMode::Ask,
                    };
                    set_permission_mode(next_mode);
                } else {
                    let next_scope = match p.folder_scope {
                        FolderScope::CurrentDir => FolderScope::AllDirs,
                        FolderScope::AllDirs => FolderScope::CurrentDir,
                    };
                    set_folder_scope(next_scope);
                }
                let p2 = get_tool_permissions();
                self.status_message = format!("Permissions: {} | {}", p2.mode_label(), p2.scope_label());
            }
            5 => {
                // HF Token
                if dir > 0 {
                    self.hf_token_input = crate::settings::get_hf_token().unwrap_or_default();
                    self.hf_token_editing = true;
                    self.status_message = "Type or paste HuggingFace token...".to_string();
                } else {
                    crate::settings::clear_hf_token();
                    self.status_message = "HuggingFace token removed.".to_string();
                }
            }
            _ => {}
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

    pub async fn handle_events(&mut self) -> Result<bool, std::io::Error> {
        let now = std::time::Instant::now();
        if now.duration_since(self.last_metrics_time).as_millis() >= 1000 {
            self.sys.refresh_cpu_usage();
            self.sys.refresh_memory();
            self.last_metrics_time = now;
        }

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
                if self.auto_tool_turns == 20 {
                    self.messages.push(
                        "System: [Agent has taken 20 tool turns — press Ctrl+C to stop]"
                            .to_string(),
                    );
                }
                self.trigger_generation_from_context();
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
                            chip.body = format!("{result}\n(Auto-allowed write)");
                        }
                    }
                }
            }
        }

        // Check if generation finished
        {
            let is_gen = *self.is_generating.lock().unwrap();
            let current_stream = self.streaming_response.lock().unwrap().clone();
            let err_opt = self.generation_error.lock().unwrap().take();
            let settings = crate::settings::get_settings();

            if !is_gen {
                self.streamed_writes_done.clear();
                if let Some(err) = err_opt {
                    let recovered = crate::agent::AgentEngine::extract_proposed_actions(&current_stream);
                    if !recovered.is_empty() {
                        let count = recovered.len();
                        self.messages.push(format!(
                            "System: Generation interrupted after {count} tool(s) were produced: {err}"
                        ));
                    } else if let Some(last) = self.messages.last_mut() {
                        if last.starts_with("Agent: ") {
                            *last = format!("Error: {}", err);
                        }
                    }
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
                            return Ok(true);
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
                    let last_user = self.last_user_message().unwrap_or_default();
                    let only_ls = effective_stream.contains("<ls")
                        && !effective_stream.contains("<write")
                        && !effective_stream.contains("<read src=");
                    // After a useless ls loop on a create/plan task, re-prompt to plan/write
                    // instead of "Done — tool already finished" (that felt like spam).
                    let wants_plan = crate::agent::AgentEngine::wants_plan_first(&last_user);
                    let wants_code = crate::agent::AgentEngine::wants_implement(&last_user)
                        || last_user.to_ascii_lowercase().contains("start coding");
                    let ls_spam_on_create = already_have_tools
                        && only_ls
                        && (wants_plan || wants_code || only_repeated_tool);

                    let loop_hit =
                        crate::settings::detect_repeat_loop(&self.recent_tool_calls, &settings);

                    if only_repeated_tool || ls_spam_on_create {
                        self.messages.push(
                            "System: [Host] Finished inspecting files. Ready for next prompt."
                                .to_string(),
                        );
                        self.recent_tool_calls.clear();
                        self.repeat_count = 0;
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
                        // "Ask" permission mode — auto-execute writes immediately for
                        // continuous operation. Ctrl+C is the user's stop signal.
                        let tool_out = crate::agent::AgentEngine::process_response(&effective_stream);
                        for a in &proposed {
                            if a.kind == crate::agent::ProposedKind::Write {
                                let path = crate::agent::AgentEngine::expand_path(&a.target);
                                if let Some(parent) = path.parent() {
                                    let _ = std::fs::create_dir_all(parent);
                                }
                                let _ = std::fs::write(&path, &a.body);
                                let anchor = self.latest_agent_msg_idx();
                                let kind = tool_panel::ToolPanelKind::Write;
                                let target_str = tool_panel::normalize_target(kind, &path.display().to_string());
                                let chip_exists = self.tool_chips.iter().any(|c| {
                                    c.kind == kind && tool_panel::same_tool_target(c.kind, &c.target, &target_str)
                                });
                                if !chip_exists {
                                    let id = self.next_chip_id;
                                    self.next_chip_id += 1;
                                    self.tool_chips.push(tool_panel::ToolChip {
                                        id,
                                        kind,
                                        target: target_str,
                                        body: a.body.clone(),
                                        tag_closed: true,
                                        pending: false, spawned: false,
                                        rect: None,
                                        anchor_msg: anchor,
                                        expanded: false,
                                        anim_start: None,
                                    });
                                }
                            }
                        }
                        if let Some(out) = tool_out {
                            self.record_tool_result_ui("tool", &out);
                            self.auto_tool_turns += 1;
                            self.trigger_generation_from_context();
                        } else {
                            self.auto_tool_turns = 0;
                            self.status_message = "Ready.".to_string();
                        }
                    } else if let Some(tool_output) = tool_output_opt {
                        if !crate::agent::AgentEngine::response_has_tool_tags(&effective_stream) {
                            self.auto_tool_turns = 0;
                            self.status_message = "Ready.".to_string();
                        } else {
                            let tool_name = if effective_stream.contains("<read") {
                                "read"
                            } else if effective_stream.contains("<ls") {
                                "ls"
                            } else if effective_stream.contains("<write") {
                                "write"
                            } else {
                                "tool"
                            };
                            self.record_tool_result_ui(tool_name, &tool_output);
                            self.auto_tool_turns += 1;
                            if self.auto_tool_turns == 20 {
                                self.messages.push(
                                    "System: [Agent has taken 20 tool turns — press Ctrl+C to stop]"
                                        .to_string(),
                                );
                            }
                            self.trigger_generation_from_context();
                        }
                    } else {
                        self.auto_tool_turns = 0;
                        self.status_message = "Ready.".to_string();
                    }

                    if !prose.is_empty() {
                        self.context_tokens_est = self.estimate_full_session_tokens();
                        if let Ok(mut l) = self.activity_logs.lock() {
                            l.push(format!("[SESSION] tokens ≈ {}", self.context_tokens_est));
                        }
                    }
                }
            }
        }

        // Check if an async model search finished
        {
            let mut res = self.search_results.lock().unwrap();
            if let Some(models) = res.take() {
                let mut installed = self.manager.list_installed_local();
                let mut hf_items = Vec::new();
                let mut ollama_items = Vec::new();
                for m in models {
                    if m.starts_with("Ollama Local:") || m.starts_with("Local GGUF:") {
                        if !installed.contains(&m) {
                            installed.push(m.clone());
                        }
                    } else if m.starts_with("Ollama: ") {
                        let stripped = m.trim_start_matches("Ollama: ").to_string();
                        if !ollama_items.contains(&stripped) {
                            ollama_items.push(stripped);
                        }
                    } else if m.starts_with("HuggingFace: ") {
                        let stripped = m.trim_start_matches("HuggingFace: ").to_string();
                        if !hf_items.contains(&stripped) {
                            hf_items.push(stripped);
                        }
                    } else if m.starts_with("Ollama:") {
                        let stripped = m.trim_start_matches("Ollama:").trim().to_string();
                        if !ollama_items.contains(&stripped) {
                            ollama_items.push(stripped);
                        }
                    } else {
                        if !hf_items.contains(&m) {
                            hf_items.push(m);
                        }
                    }
                }
                self.installed_models = installed;
                self.hf_models = hf_items;
                self.registry_models = ollama_items;
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

            if let Some(path) = self.manager.latest_gguf_path() {
                self.backend = AgentBackend::LlamaCppLib(LlamaCppLibBackend::gguf(path.clone()));
                self.manager.set_active_gguf_path(path.display().to_string());
                if let Some(entry) = self.manager.list_installed_entries().into_iter().rev().next() {
                    if !entry.name.is_empty() {
                        self.status_message = format!("Active Engine: {}", entry.name);
                    } else {
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

        // Check hold-to-exit (1s duration)
        let _esc_hold_progress = if let Some(start) = self.esc_hold_start {
            let elapsed = start.elapsed().as_secs_f32();
            let p = (elapsed / 1.0).clamp(0.0, 1.0);
            if p >= 1.0 {
                self.esc_hold_start = None;
                if let Ok(mut g) = self.is_generating.lock() {
                    *g = false;
                }
                crate::llama::server::shutdown_managed_server();
                crate::llama::libinfer::shutdown_warm_lib_engine();
                self.should_quit = true;
                self.status_message = "Exiting…".to_string();
            }
            Some(p)
        } else {
            None
        };

        // Animation state updates for header dropdown and menu modal via KramaFrame
        if self.show_menu {
            if self.menu_closing {
                if !self.krama.is_reversed("menu_fade", 0) {
                    self.krama.reverse_animate("menu_fade", 0);
                }
                let t = self.krama.get_progress_f32("menu_fade", 0).abs();
                self.menu_anim_progress = t;
                if !self.krama.is_animating("menu_fade", 0) || t <= 0.01 {
                    self.show_menu = false;
                    self.menu_closing = false;
                    self.menu_anim_progress = 0.0;
                    self.krama.restart_progress("menu_fade", 0);
                }
            } else {
                if self.krama.is_reversed("menu_fade", 0) {
                    self.krama.reverse_animate("menu_fade", 0);
                }
                let t = self.krama.get_progress_f32("menu_fade", 0).abs();
                self.menu_anim_progress = if self.krama.is_animating("menu_fade", 0) { t } else { 1.0 };
            }
        } else {
            self.menu_anim_progress = 0.0;
        }

        if self.header_dropdown_open {
            if self.krama.is_reversed("slide", 0) {
                self.krama.reverse_animate("slide", 0);
            }
            let t = self.krama.get_progress_f32("slide", 0).abs();
            self.header_anim_progress = if self.krama.is_animating("slide", 0) { t } else { 1.0 };
        } else {
            if !self.krama.is_reversed("slide", 0) && self.header_anim_progress > 0.0 {
                self.krama.reverse_animate("slide", 0);
            }
            let t = self.krama.get_progress_f32("slide", 0).abs();
            self.header_anim_progress = t;
            if !self.krama.is_animating("slide", 0) {
                self.header_anim_progress = 0.0;
            }
        }

        let is_generating_val = *self.is_generating.lock().unwrap();
        let is_downloading = self.download_progress.lock().unwrap().is_some();
        let is_krama_animating = self.krama.is_any_animation_inprogress();

        let is_animating = is_generating_val
            || is_downloading
            || !self.code_block_anims.is_empty()
            || is_krama_animating
            || self.esc_hold_start.is_some()
            || self.menu_closing
            || self.panel_closing;

        let poll_dur = if is_animating {
            Duration::from_millis(8)
        } else {
            Duration::from_millis(100)
        };

        if event::poll(poll_dur)? {
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
                            if self.header_dropdown_open {
                                self.header_dropdown_open = false;
                            }
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
                            if self.header_dropdown_open {
                                self.header_dropdown_open = false;
                            }
                        }
                    }
                    MouseEventKind::Down(MouseButton::Left) => {
                        // Check container close button " x " hit
                        if let Some((close_y, close_x0, close_x1)) = self.container_close_hit {
                            if mouse.row == close_y && mouse.column >= close_x0 && mouse.column <= close_x1 {
                                self.menu_closing = true;
                                self.clear_selection();
                                self.exit_term_interactive();
                                return Ok(true);
                            }
                        }

                        // Check row 0 menu tab hits
                        if self.header_dropdown_open && mouse.row == 0 {
                            for (sec_idx, x0, x1) in &self.menu_tab_hits {
                                if mouse.column >= *x0 && mouse.column <= *x1 {
                                    self.menu_section = *sec_idx;
                                    self.show_menu = true;
                                    self.menu_closing = false;
                                    self.header_dropdown_open = false; // slide header back up
                                    self.krama.restart_progress("menu_fade", 0);
                                    self.clear_selection();
                                    self.exit_term_interactive();
                                    return Ok(true);
                                }
                            }
                        }

                        // Check click on top header bar (Row 0 when closed, Row 1 when open)
                        if mouse.row == 0 || mouse.row == 1 {
                            if let Some((h_row, h_x0, h_x1)) = self.header_bar_hit {
                                if (mouse.row == h_row || (mouse.row <= 1 && !self.header_dropdown_open)) && mouse.column >= h_x0 && mouse.column <= h_x1 {
                                    self.header_dropdown_open = !self.header_dropdown_open;
                                    self.krama.restart_progress("slide", 0);
                                    self.clear_selection();
                                    self.exit_term_interactive();
                                    return Ok(true);
                                }
                            }
                        }

                        // Clicking anywhere outside dropdown closes it
                        if self.header_dropdown_open {
                            self.header_dropdown_open = false;
                        }

                        // Check click on Model Name badge to toggle input prompt focus
                        if let Some((badge_y, badge_x0, badge_x1)) = self.model_badge_hit {
                            if mouse.row == badge_y && mouse.column >= badge_x0 && mouse.column <= badge_x1 {
                                self.input_focused = !self.input_focused;
                                if !self.input_focused {
                                    self.input_scroll_y = 0;
                                }
                                self.clear_selection();
                                self.exit_term_interactive();
                                return Ok(true);
                            }
                            let input_h = (self.input_anim_height.round() as u16).max(1);
                            if mouse.row >= badge_y && mouse.row < badge_y + input_h {
                                self.input_focused = true;
                                self.clear_selection();
                                self.exit_term_interactive();
                                return Ok(true);
                            }
                        }

                        // Check interactive code block Copy button first
                        let mut copy_action: Option<(usize, String)> = None;
                        for hit in &self.code_block_copy_hits {
                            if mouse.row as i32 == hit.screen_y && mouse.column >= hit.copy_x.0 && mouse.column <= hit.copy_x.1 {
                                copy_action = Some((hit.block_idx, hit.code_body.clone()));
                                break;
                            }
                        }
                        if let Some((b_idx, code_body)) = copy_action {
                            let ok = crate::clipboard::copy_text_silent(&code_body);
                            let n = code_body.lines().count().max(1);
                            self.status_message = if ok {
                                format!("Copied code block #{} ({} lines) → clipboard", b_idx + 1, n)
                            } else {
                                format!("Saved code block #{} to {}", b_idx + 1, crate::clipboard::clipboard_file_path())
                            };
                            if let Ok(mut l) = self.activity_logs.lock() {
                                l.push(format!("[CLIPBOARD] Code block #{} ({} lines)", b_idx + 1, n));
                            }
                            self.clear_selection();
                            self.exit_term_interactive();
                            self.input_focused = false;
                            return Ok(true);
                        }

                        // Check interactive preview horizontal scrollbar
                        let mut scroll_action: Option<(usize, usize)> = None;
                        for hit in &self.code_block_scroll_hits {
                            if mouse.row as i32 == hit.screen_y {
                                let cur = self.code_block_scrolls.get(&hit.block_idx).copied().unwrap_or(0);
                                if mouse.column >= hit.left_btn_x.0 && mouse.column <= hit.left_btn_x.1 {
                                    scroll_action = Some((hit.block_idx, cur.saturating_sub(8)));
                                    break;
                                } else if mouse.column >= hit.right_btn_x.0 && mouse.column <= hit.right_btn_x.1 {
                                    scroll_action = Some((hit.block_idx, (cur + 8).min(hit.max_scroll)));
                                    break;
                                } else if mouse.column >= hit.track_x.0 && mouse.column <= hit.track_x.1 {
                                    let track_w = (hit.track_x.1.saturating_sub(hit.track_x.0)).max(1) as f32;
                                    let click_offset = (mouse.column.saturating_sub(hit.track_x.0)) as f32;
                                    let pct = (click_offset / track_w).clamp(0.0, 1.0);
                                    let target_sc = (pct * hit.max_scroll as f32).round() as usize;
                                    scroll_action = Some((hit.block_idx, target_sc));
                                    break;
                                }
                            }
                        }
                        if let Some((b_idx, new_scroll)) = scroll_action {
                            self.code_block_scrolls.insert(b_idx, new_scroll);
                            self.clear_selection();
                            self.exit_term_interactive();
                            self.input_focused = false;
                            return Ok(true);
                        }

                        // Check interactive code block Normal / Preview button toggles
                        let mut toggle_action: Option<(usize, bool)> = None;
                        for hit in &self.code_block_hits {
                            if mouse.row as i32 == hit.screen_y {
                                if mouse.column >= hit.normal_x.0 && mouse.column <= hit.normal_x.1 {
                                    toggle_action = Some((hit.block_idx, false));
                                    break;
                                } else if mouse.column >= hit.preview_x.0 && mouse.column <= hit.preview_x.1 {
                                    toggle_action = Some((hit.block_idx, true));
                                    break;
                                }
                            }
                        }
                        if let Some((b_idx, is_preview)) = toggle_action {
                            if is_preview {
                                self.code_block_previews.insert(b_idx);
                                self.status_message = format!("Code block #{} set to Preview mode", b_idx + 1);
                            } else {
                                self.code_block_previews.remove(&b_idx);
                                self.status_message = format!("Code block #{} set to Normal mode", b_idx + 1);
                            }
                            self.code_block_anims.insert(b_idx, (is_preview, std::time::Instant::now()));
                            self.clear_selection();
                            self.exit_term_interactive();
                            self.input_focused = false;
                        } else if let Some(id) =
                            tool_panel::hit_test_chip(&self.tool_chips, mouse.column, mouse.row)
                        {
                            self.clear_selection();
                            self.exit_term_interactive();
                            self.is_selecting = false;
                            self.selection_start = None;
                            self.selection_end = None;
                            if let Some(chip) = self.tool_chips.iter_mut().find(|c| c.id == id) {
                                chip.expanded = !chip.expanded;
                                chip.anim_start = Some(std::time::Instant::now());
                            }
                            return Ok(true);
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
                                    } else if mouse.modifiers.contains(KeyModifiers::CONTROL) {
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
                                    } else {
                                        self.clear_selection();
                                        self.is_selecting = false;
                                        self.input_focused = false;
                                    }
                                }
                            }
                        } else if mouse.modifiers.contains(KeyModifiers::CONTROL) {
                            // Click with CTRL outside any panel — initiate selection
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
                        } else {
                            // Regular click outside panels: clear any active selection
                            self.exit_term_interactive();
                            self.clear_selection();
                            self.is_selecting = false;
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
                    if self.show_menu && self.menu_section == 3 && self.settings_tab == 5 && self.hf_token_editing {
                        self.hf_token_input.push_str(text.trim());
                    } else if self.input_focused && !self.show_menu {
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
                                        if self.hf_token_editing {
                                            self.hf_token_editing = false;
                                            self.hf_token_input.clear();
                                            self.status_message = "Token editing cancelled.".to_string();
                                        } else if self.settings_col == 1 {
                                            // Exit second column back to tab column
                                            self.settings_col = 0;
                                        } else {
                                            self.menu_closing = true;
                                        }
                                        self.esc_hold_start = None;
                                    } else if self.header_dropdown_open {
                                        self.header_dropdown_open = false;
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
                                KeyCode::F(1) => {
                                    if self.show_menu && self.menu_section == 0 {
                                        self.menu_closing = true;
                                    } else {
                                        self.menu_section = 0; // Help
                                        self.show_menu = true;
                                        self.menu_closing = false;
                                        self.header_dropdown_open = false;
                                        self.krama.restart_progress("menu_fade", 0);
                                    }
                                }
                                KeyCode::F(2) => {
                                    if self.show_menu && self.menu_section == 1 {
                                        self.menu_closing = true;
                                    } else {
                                        self.menu_section = 1; // Registry
                                        self.show_menu = true;
                                        self.menu_closing = false;
                                        self.header_dropdown_open = false;
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
                                KeyCode::F(3) => {
                                    if self.show_menu && self.menu_section == 2 {
                                        self.menu_closing = true;
                                    } else {
                                        self.menu_section = 2; // Modal (Installed models)
                                        self.show_menu = true;
                                        self.menu_closing = false;
                                        self.header_dropdown_open = false;
                                        self.krama.restart_progress("menu_fade", 0);
                                        self.installed_models = self.manager.list_installed_local();
                                    }
                                }
                                KeyCode::F(4) => {
                                    if self.show_menu && self.menu_section == 3 {
                                        self.menu_closing = true;
                                    } else {
                                        self.menu_section = 3; // Settings
                                        self.settings_col = 0;
                                        self.show_menu = true;
                                        self.menu_closing = false;
                                        self.header_dropdown_open = false;
                                        self.krama.restart_progress("menu_fade", 0);
                                    }
                                }
                                KeyCode::Char('f') | KeyCode::Char('F')
                                    if key.modifiers.contains(KeyModifiers::CONTROL) =>
                                {
                                    self.input_focused = !self.input_focused;
                                }
                                KeyCode::Char('m') | KeyCode::Char('M')
                                    if key.modifiers.contains(KeyModifiers::CONTROL) =>
                                {
                                    self.header_dropdown_open = !self.header_dropdown_open;
                                }
                                KeyCode::Left if self.show_menu => {
                                    if self.menu_section == 1 {
                                        // Registry tab: toggle HF / Ollama
                                        self.registry_tab = if self.registry_tab == 0 { 1 } else { 0 };
                                    } else if self.menu_section == 3 {
                                        if !self.hf_token_editing {
                                            if self.settings_col == 1 {
                                                self.adjust_setting_value(-1);
                                            } else {
                                                self.settings_tab = if self.settings_tab == 0 { 5 } else { self.settings_tab - 1 };
                                            }
                                        }
                                    }
                                }
                                KeyCode::Char('a') if self.show_menu && self.menu_section == 3 => {
                                    if !self.hf_token_editing {
                                        if self.settings_col == 1 {
                                            self.adjust_setting_value(-1);
                                        } else {
                                            self.settings_tab = if self.settings_tab == 0 { 5 } else { self.settings_tab - 1 };
                                        }
                                    } else {
                                        self.hf_token_input.push('a');
                                    }
                                }
                                KeyCode::Right if self.show_menu => {
                                    if self.menu_section == 1 {
                                        // Registry tab: toggle HF / Ollama
                                        self.registry_tab = if self.registry_tab == 0 { 1 } else { 0 };
                                    } else if self.menu_section == 3 {
                                        if !self.hf_token_editing {
                                            if self.settings_col == 1 {
                                                self.adjust_setting_value(1);
                                            } else {
                                                self.settings_tab = (self.settings_tab + 1) % 6;
                                            }
                                        }
                                    }
                                }
                                KeyCode::Char('d') if self.show_menu && self.menu_section == 3 => {
                                    if !self.hf_token_editing {
                                        if self.settings_col == 1 {
                                            if self.settings_tab == 5 {
                                                crate::settings::clear_hf_token();
                                                self.status_message = "HuggingFace token removed.".to_string();
                                            } else {
                                                self.adjust_setting_value(1);
                                            }
                                        } else {
                                            self.settings_tab = (self.settings_tab + 1) % 6;
                                        }
                                    } else {
                                        self.hf_token_input.push('d');
                                    }
                                }
                                KeyCode::Left => {
                                    if self.input_focused {
                                        if key.modifiers.contains(KeyModifiers::ALT) {
                                            self.input_cursor_position = self.cursor_word_left();
                                        } else {
                                            self.input_cursor_position =
                                                self.input_cursor_position.saturating_sub(1);
                                        }
                                    } else {
                                        self.table_scroll_x = self.table_scroll_x.saturating_sub(4);
                                    }
                                }
                                KeyCode::Right => {
                                    if self.input_focused {
                                        if key.modifiers.contains(KeyModifiers::ALT) {
                                            self.input_cursor_position = self.cursor_word_right();
                                        } else {
                                            self.input_cursor_position =
                                                (self.input_cursor_position + 1)
                                                    .min(self.input.chars().count());
                                        }
                                    } else {
                                        self.table_scroll_x = self.table_scroll_x.saturating_add(4);
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
                                KeyCode::Up if self.show_menu => {
                                    if self.menu_section == 1 {
                                        let q_lower = self.registry_search_query.trim().to_lowercase();
                                        let total = if self.registry_tab == 0 {
                                            self.hf_models.iter().filter(|m| q_lower.is_empty() || m.to_lowercase().contains(&q_lower)).count()
                                        } else {
                                            self.registry_models.iter().filter(|m| q_lower.is_empty() || m.to_lowercase().contains(&q_lower)).count()
                                        };
                                        let i = match self.registry_state.selected() {
                                            Some(i) => if i == 0 { total.saturating_sub(1) } else { i - 1 },
                                            None => 0,
                                        };
                                        self.registry_state.select(Some(i));
                                    } else if self.menu_section == 2 {
                                        let i = match self.installed_state.selected() {
                                            Some(i) => if i == 0 { self.installed_models.len().saturating_sub(1) } else { i - 1 },
                                            None => 0,
                                        };
                                        self.installed_state.select(Some(i));
                                    } else if self.menu_section == 3 {
                                        if !self.hf_token_editing {
                                            if self.settings_col == 0 {
                                                self.settings_tab = if self.settings_tab == 0 { 5 } else { self.settings_tab - 1 };
                                            } else {
                                                self.adjust_setting_value(-1);
                                            }
                                        }
                                    }
                                }
                                KeyCode::Char('w') if self.show_menu && self.menu_section != 1 => {
                                    if self.menu_section == 2 {
                                        let i = match self.installed_state.selected() {
                                            Some(i) => if i == 0 { self.installed_models.len().saturating_sub(1) } else { i - 1 },
                                            None => 0,
                                        };
                                        self.installed_state.select(Some(i));
                                    } else if self.menu_section == 3 {
                                        if !self.hf_token_editing {
                                            if self.settings_col == 0 {
                                                self.settings_tab = if self.settings_tab == 0 { 5 } else { self.settings_tab - 1 };
                                            } else {
                                                self.adjust_setting_value(-1);
                                            }
                                        } else {
                                            self.hf_token_input.push('w');
                                        }
                                    }
                                }
                                KeyCode::Down if self.show_menu => {
                                    if self.menu_section == 1 {
                                        let q_lower = self.registry_search_query.trim().to_lowercase();
                                        let total = if self.registry_tab == 0 {
                                            self.hf_models.iter().filter(|m| q_lower.is_empty() || m.to_lowercase().contains(&q_lower)).count()
                                        } else {
                                            self.registry_models.iter().filter(|m| q_lower.is_empty() || m.to_lowercase().contains(&q_lower)).count()
                                        };
                                        let i = match self.registry_state.selected() {
                                            Some(i) => if i >= total.saturating_sub(1) { 0 } else { i + 1 },
                                            None => 0,
                                        };
                                        self.registry_state.select(Some(i));
                                    } else if self.menu_section == 2 {
                                        let i = match self.installed_state.selected() {
                                            Some(i) => if i >= self.installed_models.len().saturating_sub(1) { 0 } else { i + 1 },
                                            None => 0,
                                        };
                                        self.installed_state.select(Some(i));
                                    } else if self.menu_section == 3 {
                                        if !self.hf_token_editing {
                                            if self.settings_col == 0 {
                                                self.settings_tab = (self.settings_tab + 1) % 6;
                                            } else {
                                                self.adjust_setting_value(1);
                                            }
                                        }
                                    }
                                }
                                KeyCode::Char('s') if self.show_menu && self.menu_section != 1 => {
                                    if self.menu_section == 2 {
                                        let i = match self.installed_state.selected() {
                                            Some(i) => if i >= self.installed_models.len().saturating_sub(1) { 0 } else { i + 1 },
                                            None => 0,
                                        };
                                        self.installed_state.select(Some(i));
                                    } else if self.menu_section == 3 {
                                        if !self.hf_token_editing {
                                            if self.settings_col == 0 {
                                                self.settings_tab = (self.settings_tab + 1) % 6;
                                            } else {
                                                self.adjust_setting_value(1);
                                            }
                                        } else {
                                            self.hf_token_input.push('s');
                                        }
                                    }
                                }
                                KeyCode::Up => {
                                    if self.input_focused {
                                        self.input_cursor_up(80);
                                    } else {
                                        self.scroll_offset = self.scroll_offset.saturating_sub(1);
                                        self.auto_scroll_enabled = false;
                                    }
                                }
                                KeyCode::Down => {
                                    if self.input_focused {
                                        self.input_cursor_down(80);
                                    } else {
                                        self.scroll_offset = self.scroll_offset.saturating_add(1);
                                        self.auto_scroll_enabled = false;
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
                                KeyCode::Char('v') | KeyCode::Char('V')
                                    if key.modifiers.contains(KeyModifiers::CONTROL) =>
                                {
                                    if let Some(text) = crate::clipboard::read_clipboard_silent() {
                                        if self.show_menu && self.menu_section == 3 && self.settings_tab == 5 && self.hf_token_editing {
                                            self.hf_token_input.push_str(text.trim());
                                        } else if self.input_focused && !self.show_menu {
                                            self.input_insert_text(&text);
                                        }
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
                                        if self.show_menu && self.menu_section == 1 {
                                            self.registry_search_query.push(c);
                                            let query = self.registry_search_query.clone();
                                            let manager = self.manager.clone();
                                            let results = self.search_results.clone();
                                            tokio::spawn(async move {
                                                let matches =
                                                    manager.search_all_models(&query).await;
                                                *results.lock().unwrap() = Some(matches);
                                            });
                                        } else if self.show_menu && self.menu_section == 3 && self.settings_tab == 5 && self.hf_token_editing {
                                            if c != '\n' && c != '\r' {
                                                self.hf_token_input.push(c);
                                            }
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
                                    } else if self.show_menu && self.menu_section == 1 {
                                        self.registry_search_query.pop();
                                        let query = self.registry_search_query.clone();
                                        let manager = self.manager.clone();
                                        let results = self.search_results.clone();
                                        tokio::spawn(async move {
                                            let matches = manager.search_all_models(&query).await;
                                            *results.lock().unwrap() = Some(matches);
                                        });
                                    } else if self.show_menu && self.menu_section == 3 && self.settings_tab == 5 && self.hf_token_editing {
                                        self.hf_token_input.pop();
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
                                    } else if self.show_menu && self.menu_section == 3 && self.settings_tab == 5 {
                                        if self.hf_token_editing {
                                            self.hf_token_input.clear();
                                        } else {
                                            crate::settings::clear_hf_token();
                                            self.status_message = "HuggingFace token removed.".to_string();
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
                                            // Help tab: Enter closes menu
                                            self.menu_closing = true;
                                        } else if self.menu_section == 1 {
                                            // Registry tab: download selected model
                                            let items = if self.registry_tab == 0 {
                                                &self.hf_models
                                            } else {
                                                &self.registry_models
                                            };
                                            if let Some(i) = self.registry_state.selected() {
                                                if i < items.len() {
                                                    let item_str = items[i].clone();
                                                    if item_str.starts_with("Ollama:")
                                                        || item_str.starts_with("Ollama Local:")
                                                    {
                                                        let ollama_name = item_str
                                                            .replace("Ollama:", "")
                                                            .replace("Ollama Local:", "")
                                                            .split('(')
                                                            .next()
                                                            .unwrap_or(&item_str)
                                                            .split('[')
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

                                                        *self.download_progress.lock().unwrap() = Some(0.0);
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

                                                        *self.download_progress.lock().unwrap() = Some(0.0);
                                                        let progress_clone =
                                                            self.download_progress.clone();
                                                        let complete_clone =
                                                            self.download_complete.clone();
                                                        let logs_clone = self.activity_logs.clone();
                                                        let manager_clone = self.manager.clone();

                                                        tokio::spawn(async move {
                                                            let resolved = manager_clone
                                                                .resolve_gguf_file(&repo_id)
                                                                .await;
                                                            match resolved {
                                                                Ok((
                                                                    dl_repo,
                                                                    weight_filename,
                                                                    shard_files,
                                                                )) => {
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
                                                    self.menu_closing = true;
                                                }
                                            }
                                        } else if self.menu_section == 2 {
                                            // Modal (Installed models) tab: activate model
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
                                                            self.backend = AgentBackend::LlamaCppLib(
                                                                LlamaCppLibBackend::gguf(path.clone()),
                                                            );
                                                            self.manager.set_active_gguf_path(path.display().to_string());
                                                            self.status_message = format!(
                                                                "Active Engine: llama.cpp lib ({})",
                                                                path.display()
                                                            );
                                                        }
                                                    } else if let Some(path) =
                                                        self.manager.latest_gguf_path()
                                                    {
                                                        self.backend = AgentBackend::LlamaCppLib(
                                                            LlamaCppLibBackend::gguf(path.clone()),
                                                        );
                                                        self.manager.set_active_gguf_path(path.display().to_string());
                                                        self.status_message = format!(
                                                            "Active Engine: llama.cpp lib ({})",
                                                            path.display()
                                                        );
                                                    }
                                                    self.messages.push(format!("System: Switched active engine model to '{}'", selected_model));
                                                    self.initialized = false;
                                                    self.init_triggered = false;
                                                    self.menu_closing = true;
                                                }
                                            }
                                        } else if self.menu_section == 3 {
                                            // Settings tab (2-column)
                                            if self.settings_tab == 5 {
                                                if self.hf_token_editing {
                                                    // Save token
                                                    let tok = self.hf_token_input.trim().to_string();
                                                    if tok.is_empty() {
                                                        crate::settings::clear_hf_token();
                                                        self.status_message = "HuggingFace token cleared.".to_string();
                                                    } else {
                                                        crate::settings::set_hf_token(tok);
                                                        self.status_message = "HuggingFace token saved successfully.".to_string();
                                                    }
                                                    self.hf_token_editing = false;
                                                    self.hf_token_input.clear();
                                                } else if self.settings_col == 0 {
                                                    self.settings_col = 1;
                                                } else {
                                                    // Enter editing mode
                                                    self.hf_token_input = crate::settings::get_hf_token().unwrap_or_default();
                                                    self.hf_token_editing = true;
                                                }
                                            } else if self.settings_col == 0 {
                                                self.settings_col = 1; // shift focus to Column 2
                                            } else {
                                                self.adjust_setting_value(1); // cycle / modify
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
                                                return Ok(true);
                                            }
                                        }

                                        let prompt = self.input.clone();
                                        self.input.clear();
                                        self.input_cursor_position = 0;
                                        self.typewriter_len = 0;
                                        self.auto_scroll_enabled = true;
                                        self.auto_tool_turns = 0;
                                        self.recent_tool_calls.clear();
                                        self.tool_result_context.clear();

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
/swarm   — initialize swarm orchestrator mode and spawn sub-agents\n\
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
                                                "/swarm" => {
                                                    let instruction = if parts.len() > 1 {
                                                        prompt[7..].trim()
                                                    } else {
                                                        "accomplish this task"
                                                    };
                                                    let models = self.available_swarm_models();
                                                    let models_str = if models.is_empty() {
                                                        "Default active model".to_string()
                                                    } else {
                                                        models.join(", ")
                                                    };
                                                    let backend_type = match &self.backend {
                                                        AgentBackend::Ollama(_) => "Ollama repository",
                                                        AgentBackend::LlamaCppLib(_) => "llama.cpp / local GGUF repository",
                                                        #[cfg(feature = "gpu")]
                                                        AgentBackend::BurnWgpu(_) => "WGPU repository",
                                                    };
                                                    self.status_message = "Swarm Orchestrator active — delegating to sub-agents...".to_string();
                                                    self.messages.push(
                                                        format!("System: [Swarm Orchestrator initialized] Available models: [{models_str}]")
                                                    );
                                                    // Trigger agent generation with the swarm orchestration prompt in memory context
                                                    self.tool_result_context.push(format!(
                                                        "[System Orchestration]\nYou are the Swarm Orchestrator (H0). You can spawn specialized sub-agents with custom models using `<agent action=\"spawn\" role=\"ROLE\" model=\"MODEL\">task description</agent>`.\nAvailable models for your current {backend_type}: [{models_str}]\nTask: {}\nDo not write full code directly. Delegate tasks to specialized sub-agents.",
                                                        instruction
                                                    ));
                                                    self.trigger_generation_from_context();
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
        } else {
            // No event — redraw if any animation or background stream is active
            return Ok(is_animating);
        }
        Ok(true)
    }

    pub fn draw(&mut self, frame: &mut Frame) {
        let theme_color = self.theme_color;
        let dark_gray = Color::Rgb(100, 100, 100);
        let _light_blue = Color::Rgb(150, 180, 255);
        let white = Color::White;

        let area = frame.area();

        // 1. Fill entire screen background with Nordic Gray
        frame.render_widget(Block::default().style(Style::default().bg(NORDIC_BG)), area);

        let is_gen = *self.is_generating.lock().unwrap();
        let is_thinking = if is_gen {
            let s = self.streaming_response.lock().unwrap();
            s.contains("<think>") && !s.contains("</think>")
        } else {
            false
        };
        let exit_hold_pct = self
            .esc_hold_start
            .map(|start| (start.elapsed().as_secs_f64() / 1.0).clamp(0.0, 1.0));
        let stops = get_status_gradient_stops(is_gen, is_thinking, exit_hold_pct);
        let stream_len = if is_gen {
            self.streaming_response.lock().unwrap().len()
        } else {
            0
        };
        let phase = if is_gen {
            self.anim_tick as f32 * 0.02 + stream_len as f32 * 0.008
        } else {
            self.anim_tick as f32 * 0.015
        };

        // Dynamic input height with smooth collapse/expand animation
        let inner_w = area.width.saturating_sub(4).max(1) as usize;
        let mut wrapped_prompt_lines: Vec<String> = Vec::new();
        for row in self.input.split('\n') {
            if row.is_empty() {
                wrapped_prompt_lines.push(String::new());
            } else {
                let mut cur = String::new();
                let mut col = 0;
                for ch in row.chars() {
                    if col >= inner_w {
                        wrapped_prompt_lines.push(cur.clone());
                        cur.clear();
                        col = 0;
                    }
                    cur.push(ch);
                    col += 1;
                }
                if !cur.is_empty() || wrapped_prompt_lines.is_empty() {
                    wrapped_prompt_lines.push(cur);
                }
            }
        }
        if self.input.ends_with('\n') {
            wrapped_prompt_lines.push(String::new());
        }
        if wrapped_prompt_lines.is_empty() {
            wrapped_prompt_lines.push(String::new());
        }

        let content_count = wrapped_prompt_lines.len().max(1);
        let target_input_h = if self.input_focused {
            (1 + content_count).clamp(3, 7) as f32
        } else {
            1.0f32
        };

        let h_diff = target_input_h - self.input_anim_height;
        if h_diff.abs() < 0.05 {
            self.input_anim_height = target_input_h;
        } else {
            self.input_anim_height += h_diff * 0.35;
        }
        let input_box_h = (self.input_anim_height.round() as u16).clamp(1, 7);

        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints(
                [
                    Constraint::Length(1), // 1-line top header bar
                    Constraint::Min(1),
                    Constraint::Length(input_box_h),
                ]
                .as_ref(),
            )
            .split(area);

        let top_area = chunks[0];
        let full_top_w = top_area.width as usize;

        let ctx_limit = crate::settings::context_token_limit().max(1);
        let ctx_used = self.estimate_full_session_tokens();
        self.context_tokens_est = ctx_used;
        let ctx_pct = ((ctx_used as f64 / ctx_limit as f64) * 100.0).min(999.0);
        let _ctx_color = if ctx_pct >= 80.0 {
            Color::Rgb(255, 130, 140)
        } else if ctx_pct >= 50.0 {
            Color::Rgb(255, 220, 140)
        } else {
            Color::Rgb(143, 218, 255)
        };

        // Right side: CTX meter (e.g. " 🭧🭓CTX 1% 250K ")
        let ctx_label = crate::settings::format_context_tokens(ctx_limit);
        let right_ctx_str = format!("CTX {:.0}% {}", ctx_pct, ctx_label);
        let right_trans = "🭧🭓";
        let right_len = 2 + right_ctx_str.chars().count() + 2; // 🭧🭓 + " " + text + " "

        // Left side: " HERCULES " + "🭞🭜"
        let brand_text = " HERCULES ";
        let left_trans = "🭞🭜";
        let left_brand_w = brand_text.chars().count() + 2;

        let mid_bar_w = full_top_w.saturating_sub(left_brand_w + right_len);

        let get_bar_color = |col_idx: usize, bar_width: usize| -> Color {
            if let Some(ref st) = stops {
                let norm = col_idx as f32 / bar_width.max(1) as f32;
                let cycle = ((norm * 0.3) - (phase * 0.04)).rem_euclid(1.0);
                let wave = if cycle < 0.5 {
                    cycle * 2.0
                } else {
                    (1.0 - cycle) * 2.0
                };
                multi_stop_gradient(st, wave)
            } else {
                Color::Rgb(236, 239, 244)
            }
        };

        let get_contrast_text_color = |bg: Color| -> Color {
            if let Color::Rgb(r, g, b) = bg {
                let lum = 0.299 * (r as f32) + 0.587 * (g as f32) + 0.114 * (b as f32);
                if lum < 135.0 {
                    Color::Rgb(245, 248, 255)
                } else {
                    NORDIC_BG
                }
            } else {
                NORDIC_BG
            }
        };

        let mut top_spans: Vec<Span> = Vec::new();
        let mut cur_top_col = 0;
        let progress_val = *self.download_progress.lock().unwrap();
        let esc_hold_p = self.esc_hold_start.map(|s| (s.elapsed().as_secs_f32() / 1.0).clamp(0.0, 1.0));

        // 1. Left brand badge (colors matching the bar at col 0)
        let brand_bg = get_bar_color(0, full_top_w);
        let brand_fg = get_contrast_text_color(brand_bg);
        top_spans.push(Span::styled(
            brand_text,
            Style::default().fg(brand_fg).bg(brand_bg).add_modifier(Modifier::BOLD),
        ));
        top_spans.push(Span::styled(left_trans, Style::default().fg(brand_bg).bg(NORDIC_BG)));
        cur_top_col += left_brand_w;

        // 2. Middle bar: 🬂
        if let Some(hold_ratio) = esc_hold_p {
            // While holding ESC to exit: fill left-to-right red gradient
            let filled_count = ((hold_ratio * mid_bar_w as f32).round() as usize).min(mid_bar_w);
            for i in 0..mid_bar_w {
                if i < filled_count {
                    let norm = i as f32 / mid_bar_w.max(1) as f32;
                    let r = (240.0 * (1.0 - norm) + 200.0 * norm) as u8;
                    let g = (60.0 * (1.0 - norm) + 40.0 * norm) as u8;
                    let b = (60.0 * (1.0 - norm) + 80.0 * norm) as u8;
                    top_spans.push(Span::styled("🬂", Style::default().fg(Color::Rgb(r, g, b)).bg(NORDIC_BG)));
                } else {
                    top_spans.push(Span::styled("🬂", Style::default().fg(Color::Rgb(76, 86, 106)).bg(NORDIC_BG)));
                }
                cur_top_col += 1;
            }
        } else if let Some(dl_ratio) = progress_val {
            // While downloading model: base color gray (76, 86, 106) and fill cyan-to-blue left to right
            let filled_count = ((dl_ratio.clamp(0.0, 1.0) * mid_bar_w as f64).round() as usize).min(mid_bar_w);
            for i in 0..mid_bar_w {
                if i < filled_count {
                    let norm = i as f32 / mid_bar_w.max(1) as f32;
                    let r = (0.0 * (1.0 - norm) + 60.0 * norm) as u8;
                    let g = (220.0 * (1.0 - norm) + 140.0 * norm) as u8;
                    let b = (255.0 * (1.0 - norm) + 255.0 * norm) as u8;
                    top_spans.push(Span::styled("🬂", Style::default().fg(Color::Rgb(r, g, b)).bg(NORDIC_BG)));
                } else {
                    top_spans.push(Span::styled("🬂", Style::default().fg(Color::Rgb(76, 86, 106)).bg(NORDIC_BG)));
                }
                cur_top_col += 1;
            }
        } else {
            // Normal mode: identical smooth gradient to prompt bar
            for _ in 0..mid_bar_w {
                let col_c = get_bar_color(cur_top_col, full_top_w);
                top_spans.push(Span::styled("🬂", Style::default().fg(col_c).bg(NORDIC_BG)));
                cur_top_col += 1;
            }
        }

        // 3. Right CTX meter
        let ctx_bg = get_bar_color(cur_top_col, full_top_w);
        let ctx_fg = get_contrast_text_color(ctx_bg);
        top_spans.push(Span::styled(right_trans, Style::default().fg(ctx_bg).bg(NORDIC_BG)));
        top_spans.push(Span::styled(
            format!(" {} ", right_ctx_str),
            Style::default().fg(ctx_fg).bg(ctx_bg).add_modifier(Modifier::BOLD),
        ));

        // When dropdown is open / sliding:
        // Row 0 shows: " Help  Registry  Modal  Settings "
        // Row 1 shows the header bar
        self.menu_tab_hits.clear();
        let header_y = if self.header_anim_progress > 0.5 {
            // Draw Row 0 menu bar
            let mut menu_spans: Vec<Span> = Vec::new();
            let mut col_ptr = 0u16;

            let tabs = [
                (0, " Help "),
                (1, " Registry "),
                (2, " Modal "),
                (3, " Settings "),
            ];

            menu_spans.push(Span::styled(" ", Style::default().bg(NORDIC_BG)));
            col_ptr += 1;

            for (sec_idx, label) in tabs {
                let label_len = label.chars().count() as u16;
                let is_active = self.show_menu && self.menu_section == sec_idx;
                let (tab_fg, tab_bg) = if is_active {
                    (NORDIC_BG, Color::White)
                } else {
                    (Color::Rgb(220, 230, 242), Color::Rgb(46, 52, 64))
                };
                let x0 = col_ptr;
                let x1 = col_ptr + label_len - 1;
                self.menu_tab_hits.push((sec_idx, x0, x1));

                menu_spans.push(Span::styled(
                    label,
                    Style::default().fg(tab_fg).bg(tab_bg).add_modifier(Modifier::BOLD),
                ));
                col_ptr += label_len;

                menu_spans.push(Span::styled(" ", Style::default().bg(NORDIC_BG)));
                col_ptr += 1;
            }

            frame.render_widget(
                Paragraph::new(Line::from(menu_spans)).style(Style::default().bg(NORDIC_BG)),
                Rect {
                    x: top_area.x,
                    y: 0,
                    width: top_area.width,
                    height: 1,
                },
            );
            1
        } else {
            0
        };

        // Render header bar at header_y
        self.header_bar_hit = Some((header_y, top_area.x, top_area.x + top_area.width.saturating_sub(1)));
        frame.render_widget(
            Paragraph::new(Line::from(top_spans)).style(Style::default().bg(NORDIC_BG)),
            Rect {
                x: top_area.x,
                y: header_y,
                width: top_area.width,
                height: 1,
            },
        );

        // --- Main Chat Body Layout (Full width, no side log panel) ---
        let chat_area = Rect {
            x: chunks[1].x,
            y: if header_y == 1 { chunks[1].y.saturating_add(1) } else { chunks[1].y },
            width: chunks[1].width,
            height: if header_y == 1 { chunks[1].height.saturating_sub(1) } else { chunks[1].height },
        };
        let available_width = (chat_area.width.saturating_sub(2) as usize).max(1);

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
        let mut section_headers: Vec<(usize, String, Color)> = Vec::new();
        let mut all_toggle_buttons: Vec<(usize, usize, u16, u16, u16, u16)> = Vec::new();
        let mut all_copy_buttons: Vec<(usize, usize, u16, u16, String)> = Vec::new();
        let mut all_scroll_buttons: Vec<(usize, usize, u16, u16, u16, u16, u16, u16, usize)> = Vec::new();
        let content_bg = Color::Rgb(34, 39, 48); // Shaded dark background for process output

        macro_rules! push_full_shaded {
            ($lines:expr, $spans:expr, $cur_w:expr, $max_w:expr, $bg:expr) => {{
                let mut spans = $spans;
                let cur_w = $cur_w;
                let max_w = $max_w;
                if cur_w < max_w {
                    spans.push(Span::styled(" ".repeat(max_w.saturating_sub(cur_w)), Style::default().bg($bg)));
                }
                $lines.push(Line::from(spans));
            }};
        }

        for (m_idx, m) in self.messages.iter().enumerate() {
            let is_last_message = m_idx == self.messages.len() - 1;

            if m.starts_with("You:") {
                let user_bg = Color::Rgb(163, 190, 140); // Sage Green
                let user_text = m.strip_prefix("You:").unwrap_or(&m[4..]).trim();
                let title_spans = vec![Span::styled(" You ", Style::default().fg(NORDIC_BG).bg(user_bg).add_modifier(Modifier::BOLD))];
                section_headers.push((chat_lines.len(), "You".to_string(), user_bg));
                push_full_shaded!(&mut chat_lines, title_spans, 5, available_width, content_bg);
                for u_line in user_text.lines() {
                    let inline_spans = crate::markdown::parse_inline(u_line, false, false);
                    let mut raw_spans = Vec::new();
                    for ispan in inline_spans {
                        let mut style = Style::default().fg(white).bg(content_bg);
                        if ispan.bold {
                            style = style.add_modifier(Modifier::BOLD);
                        }
                        if ispan.italic {
                            style = style.add_modifier(Modifier::ITALIC);
                        }
                        if ispan.strikethrough {
                            style = style.add_modifier(Modifier::CROSSED_OUT);
                        }
                        if ispan.code {
                            style = style.fg(Color::Rgb(255, 190, 100)).bg(content_bg).add_modifier(Modifier::BOLD);
                        }
                        if ispan.link_url.is_some() {
                            style = style.fg(Color::Rgb(0, 200, 255)).bg(content_bg).add_modifier(Modifier::UNDERLINED);
                        }
                        raw_spans.push(Span::styled(ispan.text, style));
                    }
                    // Word wrap so every wrapped line gets '▎ ' and full width shading
                    let mut cur_spans = vec![
                        Span::styled("▎", Style::default().fg(user_bg).bg(content_bg)),
                        Span::styled(" ", Style::default().bg(content_bg)),
                    ];
                    let mut cur_w = 2;
                    for sp in raw_spans {
                        let text = sp.content.to_string();
                        let style = sp.style;
                        for word in text.split_inclusive(' ') {
                            let wl = word.chars().count();
                            if cur_w + wl > available_width && cur_w > 2 {
                                push_full_shaded!(&mut chat_lines, cur_spans, cur_w, available_width, content_bg);
                                cur_spans = vec![
                                    Span::styled("▎", Style::default().fg(user_bg).bg(content_bg)),
                                    Span::styled(" ", Style::default().bg(content_bg)),
                                ];
                                cur_w = 2;
                            }
                            cur_spans.push(Span::styled(word.to_string(), style));
                            cur_w += wl;
                        }
                    }
                    if cur_w > 2 {
                        push_full_shaded!(&mut chat_lines, cur_spans, cur_w, available_width, content_bg);
                    }
                }
                chat_lines.push(Line::from(""));
            } else if m.starts_with("Agent:") || m.starts_with("Error:") {
                let content = if m.starts_with("Agent:") {
                    &m[7..]
                } else {
                    &m[7..]
                };

                let (think_part, output_part, think_label) =
                    if let Some(start_think) = content.find("<think>") {
                        if let Some(end_think) = content.find("</think>") {
                            let think = &content[start_think + 7..end_think];
                            let rest = &content[end_think + 8..];
                            let before = &content[..start_think];
                            let out = if before.trim().is_empty() {
                                rest.to_string()
                            } else {
                                format!("{}{}", before, rest)
                            };
                            (Some(think.to_string()), out, "Thinking")
                        } else {
                            let think = &content[start_think + 7..];
                            let before = content[..start_think].to_string();
                            (
                                Some(think.to_string()),
                                before,
                                "Thinking",
                            )
                        }
                    } else {
                        (None, content.to_string(), "")
                    };

                let total_chars = content.chars().count();
                let reveal_limit = if is_last_message { total_chars } else { 10000 };

                if let Some(ref think_text) = think_part {
                    let clean_think = think_text.trim_start_matches(|c| c == '\n' || c == '\r');
                    if !clean_think.trim().is_empty()
                        || (is_generating_val && content.contains("<think>"))
                    {
                        let think_bg = Color::Rgb(180, 100, 240); // Soft Purple
                        let think_tag = if self.thinking_collapsed {
                            format!(" {think_label} (collapsed) ")
                        } else {
                            format!(" {think_label} ")
                        };
                        let think_len = think_tag.chars().count();
                        let title_spans = vec![Span::styled(think_tag, Style::default().fg(Color::Rgb(245, 248, 255)).bg(think_bg).add_modifier(Modifier::BOLD))];
                        section_headers.push((chat_lines.len(), think_label.to_string(), think_bg));
                        push_full_shaded!(&mut chat_lines, title_spans, think_len, available_width, content_bg);

                        if !self.thinking_collapsed {
                            let visible_think = reveal_limit.min(clean_think.chars().count());

                            let mut global_think_ch = 0;
                            if clean_think.trim().is_empty() && is_generating_val {
                                let pulse = (anim_tick as f64 * 0.3).sin() * 0.5 + 0.5;
                                let b_val = (160.0 + 95.0 * pulse) as u8;
                                let spans = vec![
                                    Span::styled("▎", Style::default().fg(think_bg).bg(content_bg)),
                                    Span::styled(
                                        " Reasoning… █",
                                        Style::default()
                                            .fg(Color::Rgb(210, 160, b_val))
                                            .bg(content_bg)
                                            .add_modifier(Modifier::ITALIC),
                                    ),
                                ];
                                push_full_shaded!(&mut chat_lines, spans, 15, available_width, content_bg);
                            } else {
                                for raw_line in clean_think.lines() {
                                    let line_str = raw_line.trim_start();
                                    if line_str.is_empty() {
                                        continue;
                                    }
                                    let mut line_spans = Vec::new();
                                    for ch in line_str.chars() {
                                        if global_think_ch >= visible_think {
                                            break;
                                        }
                                        let age = visible_think.saturating_sub(global_think_ch);
                                        let is_streaming = is_generating_val && is_last_message;
                                        let target_r = 220.0;
                                        let target_g = 160.0;
                                        let target_b = 255.0;

                                        let style_color = if is_streaming && age < 8 {
                                            let (sr, sg, sb) = (0.0, 255.0, 120.0); // Vibrant neon green
                                            let t = (age as f64 / 6.0).clamp(0.0, 1.0);
                                            let r = (sr + (target_r - sr) * t).round() as u8;
                                            let g = (sg + (target_g - sg) * t).round() as u8;
                                            let b = (sb + (target_b - sb) * t).round() as u8;
                                            Color::Rgb(r, g, b)
                                        } else {
                                            Color::Rgb(target_r as u8, target_g as u8, target_b as u8)
                                        };
                                        line_spans.push(Span::styled(
                                            ch.to_string(),
                                            Style::default().fg(style_color).bg(content_bg),
                                        ));
                                        global_think_ch += 1;
                                    }
                                    global_think_ch += 1;

                                    let mut cur_spans = vec![
                                        Span::styled("▎", Style::default().fg(think_bg).bg(content_bg)),
                                        Span::styled(" ", Style::default().bg(content_bg)),
                                    ];
                                    let mut cur_w = 2;
                                    for sp in line_spans {
                                        let text = sp.content.to_string();
                                        let style = sp.style;
                                        for word in text.split_inclusive(' ') {
                                            let wl = word.chars().count();
                                            if cur_w + wl > available_width && cur_w > 2 {
                                                push_full_shaded!(&mut chat_lines, cur_spans, cur_w, available_width, content_bg);
                                                cur_spans = vec![
                                                    Span::styled("▎", Style::default().fg(think_bg).bg(content_bg)),
                                                    Span::styled(" ", Style::default().bg(content_bg)),
                                                ];
                                                cur_w = 2;
                                            }
                                            cur_spans.push(Span::styled(word.to_string(), style));
                                            cur_w += wl;
                                        }
                                    }
                                    if cur_w > 2 {
                                        push_full_shaded!(&mut chat_lines, cur_spans, cur_w, available_width, content_bg);
                                    }
                                }
                            }
                        }
                        chat_lines.push(Line::from(""));
                    }
                }

                if !output_part.trim().is_empty()
                    || (think_part.is_none() && (is_generating_val || !content.trim().is_empty()))
                {
                    let raw_text = if output_part.is_empty() && think_part.is_none() {
                        content
                    } else {
                        output_part.as_str()
                    };
                    let text_to_render = raw_text.trim_start_matches(|c| c == '\n' || c == '\r').to_string();
                    let think_len = think_part.as_ref().map(|t| t.chars().count()).unwrap_or(0);
                    let available_output = reveal_limit.saturating_sub(think_len);

                    let active_write_chip = self.tool_chips.iter().any(|c| c.kind == tool_panel::ToolPanelKind::Write && !c.tag_closed);
                    let should_show_agent = !text_to_render.trim().is_empty() || (is_generating_val && !active_write_chip && think_part.is_none());

                    if should_show_agent {
                        let agent_bg = Color::Rgb(136, 192, 208);
                        let agent_label = if is_generating_val && is_last_message {
                            if active_write_chip {
                                " Agent (writing) "
                            } else {
                                " Agent (streaming) "
                            }
                        } else {
                            " Agent "
                        };
                        let agent_len = agent_label.chars().count();
                        let title_spans = vec![Span::styled(agent_label, Style::default().fg(NORDIC_BG).bg(agent_bg).add_modifier(Modifier::BOLD))];
                        section_headers.push((chat_lines.len(), agent_label.trim().to_string(), agent_bg));
                        push_full_shaded!(&mut chat_lines, title_spans, agent_len, available_width, content_bg);

                        let mut global_out_ch = 0;
                        let start_line_idx = chat_lines.len();
                        let mut local_toggles = Vec::new();
                        let mut local_copies = Vec::new();
                        let mut local_scrolls = Vec::new();
                        let md_lines = crate::markdown::render_markdown_to_lines(
                            &text_to_render,
                            available_output,
                            &mut global_out_ch,
                            is_generating_val,
                            is_last_message,
                            anim_tick,
                            theme_color,
                            dark_gray,
                            &self.code_block_previews,
                            Some(&mut local_toggles),
                            Some(&mut local_copies),
                            Some(&mut local_scrolls),
                            available_width.saturating_sub(4),
                            Some(&self.code_block_anims),
                            Some(&self.code_block_scrolls),
                        );
                        for (local_idx, b_idx, n_s, n_e, p_s, p_e) in local_toggles {
                            all_toggle_buttons.push((start_line_idx + local_idx, b_idx, n_s + 2, n_e + 2, p_s + 2, p_e + 2));
                        }
                        for (local_idx, b_idx, c_s, c_e, body) in local_copies {
                            all_copy_buttons.push((start_line_idx + local_idx, b_idx, c_s + 2, c_e + 2, body));
                        }
                        for (local_idx, b_idx, l_s, l_e, t_s, t_e, r_s, r_e, max_sc) in local_scrolls {
                            all_scroll_buttons.push((start_line_idx + local_idx, b_idx, l_s + 2, l_e + 2, t_s + 2, t_e + 2, r_s + 2, r_e + 2, max_sc));
                        }
                        for md_l in md_lines {
                            let spans_with_bg: Vec<Span> = md_l.spans.into_iter().map(|s| {
                                Span::styled(s.content, s.style.bg(content_bg))
                            }).collect();

                            let is_code_or_table = spans_with_bg.iter().any(|s| {
                                s.content.contains('│') || s.content.contains('┌') || s.content.contains('└') || s.content.contains('─')
                            });

                            if is_code_or_table {
                                let mut cur_w = 2;
                                for s in &spans_with_bg {
                                    cur_w += s.content.chars().count();
                                }
                                let mut spans = vec![
                                    Span::styled("▎", Style::default().fg(agent_bg).bg(content_bg)),
                                    Span::styled(" ", Style::default().bg(content_bg)),
                                ];
                                spans.extend(spans_with_bg);
                                push_full_shaded!(&mut chat_lines, spans, cur_w, available_width, content_bg);
                            } else {
                                let mut cur_spans = vec![
                                    Span::styled("▎", Style::default().fg(agent_bg).bg(content_bg)),
                                    Span::styled(" ", Style::default().bg(content_bg)),
                                ];
                                let mut cur_w = 2;
                                for sp in spans_with_bg {
                                    let text = sp.content.to_string();
                                    let style = sp.style;
                                    for word in text.split_inclusive(' ') {
                                        let wl = word.chars().count();
                                        if cur_w + wl > available_width && cur_w > 2 {
                                            push_full_shaded!(&mut chat_lines, cur_spans, cur_w, available_width, content_bg);
                                            cur_spans = vec![
                                                Span::styled("▎", Style::default().fg(agent_bg).bg(content_bg)),
                                                Span::styled(" ", Style::default().bg(content_bg)),
                                            ];
                                            cur_w = 2;
                                        }
                                        cur_spans.push(Span::styled(word.to_string(), style));
                                        cur_w += wl;
                                    }
                                }
                                if cur_w > 2 {
                                    push_full_shaded!(&mut chat_lines, cur_spans, cur_w, available_width, content_bg);
                                }
                            }
                        }
                        chat_lines.push(Line::from(""));
                    }
                }

                // Render Action process block embedded under agent turn
                let action_bg = Color::Rgb(235, 203, 139); // Amber / Gold
                let matching_chips: Vec<&ToolChip> = self
                    .tool_chips
                    .iter()
                    .filter(|c| c.anchor_msg == Some(m_idx))
                    .collect();

                for chip in matching_chips {
                    let action_tag = " Action ";

                    // Record chip start BEFORE title line so the hit rect covers
                    // both the title row AND the summary row below it.
                    let chip_start = chat_lines.len();
                    chip_line_starts.push((chip.id, chip_start));

                    let title_spans = vec![Span::styled(
                        action_tag,
                        Style::default().fg(NORDIC_BG).bg(action_bg).add_modifier(Modifier::BOLD),
                    )];
                    section_headers.push((chat_lines.len(), format!("Action: {}", chip.label_text()), action_bg));
                    push_full_shaded!(&mut chat_lines, title_spans, 8, available_width, content_bg);

                    let chip_summary = chip.label_text();
                    let cur_spans = vec![
                        Span::styled("▎", Style::default().fg(action_bg).bg(content_bg)),
                        Span::styled(" ", Style::default().bg(content_bg)),
                        Span::styled(
                            chip_summary.clone(),
                            Style::default().fg(chip.kind.accent()).bg(content_bg).add_modifier(Modifier::BOLD),
                        ),
                    ];
                    let cur_w = 2 + chip_summary.chars().count();
                    push_full_shaded!(&mut chat_lines, cur_spans, cur_w, available_width, content_bg);

                    // Compute animated content line count
                    let max_body_lines = chip.body.lines().count().min(25);
                    let target_open = chip.expanded || !chip.tag_closed;
                    let elapsed_ms = chip.anim_start.map(|s| s.elapsed().as_millis()).unwrap_or(300);
                    let anim_progress = (elapsed_ms as f32 / 200.0).clamp(0.0, 1.0);
                    let visible_lines = if target_open {
                        if chip.tag_closed {
                            // Expanding or already expanded
                            ((max_body_lines as f32) * anim_progress).round() as usize
                        } else {
                            // Actively writing / streaming: show full current body
                            max_body_lines
                        }
                    } else {
                        // Collapsing once completed
                        let remaining = 1.0 - anim_progress;
                        ((max_body_lines as f32) * remaining).round() as usize
                    };

                    if visible_lines > 0 && !chip.body.trim().is_empty() {
                        // Top horizontal separator rule
                        let rule_spans = vec![
                            Span::styled("▎", Style::default().fg(action_bg).bg(content_bg)),
                            Span::styled(" ", Style::default().bg(content_bg)),
                            Span::styled("─".repeat(available_width.saturating_sub(4).max(10)), Style::default().fg(Color::Rgb(60, 68, 82)).bg(content_bg)),
                        ];
                        push_full_shaded!(&mut chat_lines, rule_spans, available_width, available_width, content_bg);

                        for (line_idx, b_line) in chip.body.lines().take(visible_lines).enumerate() {
                            let is_del = b_line.starts_with("- ") || b_line.starts_with('-');
                            let is_add = b_line.starts_with("+ ") || b_line.starts_with('+');
                            
                            let (line_fg, sign_color) = if is_del {
                                (Color::Rgb(255, 130, 140), Color::Rgb(255, 90, 100))
                            } else if is_add {
                                (Color::Rgb(163, 190, 140), Color::Rgb(80, 220, 140))
                            } else {
                                (NORDIC_TEXT, Color::Rgb(100, 110, 130))
                            };

                            let line_num_str = format!(" {:2} │ ", line_idx + 1);
                            let cur_spans = vec![
                                Span::styled("▎", Style::default().fg(action_bg).bg(content_bg)),
                                Span::styled(" ", Style::default().bg(content_bg)),
                                Span::styled(line_num_str, Style::default().fg(sign_color).bg(content_bg)),
                                Span::styled(b_line.to_string(), Style::default().fg(line_fg).bg(content_bg)),
                            ];
                            let cur_w = 2 + 7 + b_line.chars().count();
                            push_full_shaded!(&mut chat_lines, cur_spans, cur_w, available_width, content_bg);
                        }
                    }
                    chat_lines.push(Line::from(""));
                }
            } else if m.starts_with("System:") {
                let sys_bg = Color::Rgb(94, 129, 172); // Nordic Slate Blue
                let sys_body = m.strip_prefix("System:").unwrap_or(&m[7..]).trim();
                let title_spans = vec![Span::styled(" System ", Style::default().fg(Color::Rgb(245, 248, 255)).bg(sys_bg).add_modifier(Modifier::BOLD))];
                section_headers.push((chat_lines.len(), "System".to_string(), sys_bg));
                push_full_shaded!(&mut chat_lines, title_spans, 8, available_width, content_bg);
                for line in sys_body.lines() {
                    let mut cur_spans = vec![
                        Span::styled("▎", Style::default().fg(sys_bg).bg(content_bg)),
                        Span::styled(" ", Style::default().bg(content_bg)),
                    ];
                    let mut cur_w = 2;
                    for word in line.split_inclusive(' ') {
                        let wl = word.chars().count();
                        if cur_w + wl > available_width && cur_w > 2 {
                            push_full_shaded!(&mut chat_lines, cur_spans, cur_w, available_width, content_bg);
                            cur_spans = vec![
                                Span::styled("▎", Style::default().fg(sys_bg).bg(content_bg)),
                                Span::styled(" ", Style::default().bg(content_bg)),
                            ];
                            cur_w = 2;
                        }
                        cur_spans.push(Span::styled(word.to_string(), Style::default().fg(NORDIC_TEXT).bg(content_bg)));
                        cur_w += wl;
                    }
                    if cur_w > 2 {
                        push_full_shaded!(&mut chat_lines, cur_spans, cur_w, available_width, content_bg);
                    }
                }
                chat_lines.push(Line::from(""));
            } else {
                chat_lines.push(Line::from(Span::styled(
                    m.clone(),
                    Style::default().fg(NORDIC_MUTED),
                )));
            }
        }

        let available_width = (chat_area.width.saturating_sub(2) as usize).max(1);
        let mut total_visual_lines: u16 = 0;
        let mut visual_at: Vec<u16> = Vec::with_capacity(chat_lines.len() + 1);
        for line in &chat_lines {
            visual_at.push(total_visual_lines);
            let w = line.width();
            if w == 0 {
                total_visual_lines += 1;
            } else {
                let lines = (w + available_width - 1) / available_width;
                total_visual_lines += lines as u16;
            }
        }
        self.last_chat_visual_at = visual_at.clone();

        // Calculate screen coordinates for interactive code block Normal/Preview and Copy buttons
        self.code_block_hits.clear();
        self.code_block_copy_hits.clear();
        let chat_top = chat_area.y as i32 + 1;
        let chat_left = chat_area.x;
        for (l_idx, b_idx, n_s, n_e, p_s, p_e) in all_toggle_buttons {
            if let Some(&vis) = visual_at.get(l_idx) {
                let screen_y = chat_top + vis as i32 - self.scroll_offset as i32;
                self.code_block_hits.push(CodeBlockToggleHit {
                    block_idx: b_idx,
                    screen_y,
                    normal_x: (chat_left.saturating_add(n_s), chat_left.saturating_add(n_e)),
                    preview_x: (chat_left.saturating_add(p_s), chat_left.saturating_add(p_e)),
                });
            }
        }
        for (l_idx, b_idx, c_s, c_e, body) in all_copy_buttons {
            if let Some(&vis) = visual_at.get(l_idx) {
                let screen_y = chat_top + vis as i32 - self.scroll_offset as i32;
                self.code_block_copy_hits.push(CodeBlockCopyHit {
                    block_idx: b_idx,
                    screen_y,
                    copy_x: (chat_left.saturating_add(c_s), chat_left.saturating_add(c_e)),
                    code_body: body,
                });
            }
        }
        self.code_block_scroll_hits.clear();
        for (l_idx, b_idx, l_s, l_e, t_s, t_e, r_s, r_e, max_sc) in all_scroll_buttons {
            if let Some(&vis) = visual_at.get(l_idx) {
                let screen_y = chat_top + vis as i32 - self.scroll_offset as i32;
                self.code_block_scroll_hits.push(CodeBlockScrollHit {
                    block_idx: b_idx,
                    screen_y,
                    left_btn_x: (chat_left.saturating_add(l_s), chat_left.saturating_add(l_e)),
                    track_x: (chat_left.saturating_add(t_s), chat_left.saturating_add(t_e)),
                    right_btn_x: (chat_left.saturating_add(r_s), chat_left.saturating_add(r_e)),
                    max_scroll: max_sc,
                });
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

        // Shade selected rows while dragging OR after release (has_selection) using visual row alignment
        if self.selection_active() {
            if let (Some(start), Some(end)) = (self.selection_start, self.selection_end) {
                let min_y = start.1.min(end.1) as i32;
                let max_y = start.1.max(end.1) as i32;
                let sel_bg = Color::Rgb(28, 72, 128);
                for (i, line) in chat_lines.iter_mut().enumerate() {
                    let vis_start = visual_at.get(i).copied().unwrap_or(i as u16) as i32;
                    let vis_count = if i + 1 < visual_at.len() {
                        (visual_at[i + 1] - visual_at[i]) as i32
                    } else {
                        1
                    };
                    let screen_y_start = chat_top + vis_start - self.scroll_offset as i32;
                    let screen_y_end = screen_y_start + vis_count - 1;

                    if max_y >= screen_y_start && min_y <= screen_y_end {
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

        let chat_box = Paragraph::new(chat_lines)
            .style(Style::default().bg(NORDIC_BG))
            .scroll((self.scroll_offset, 0))
            .wrap(ratatui::widgets::Wrap { trim: false })
            .block(
                Block::default()
                    .borders(Borders::NONE)
            );
        frame.render_widget(chat_box, chat_area);

        // Sticky process title on top of chat area (like code review sticky header)
        if self.scroll_offset > 0 {
            // Find active section at top of viewport (visual line == self.scroll_offset)
            let mut active_header: Option<(&str, Color)> = None;
            for (line_idx, label, color) in &section_headers {
                let vis = visual_at.get(*line_idx).copied().unwrap_or(0);
                if vis <= self.scroll_offset {
                    active_header = Some((label.as_str(), *color));
                } else {
                    break;
                }
            }

            if let Some((label, bg_col)) = active_header {
                let sticky_label = format!(" {label} ");
                let sticky_w = (sticky_label.chars().count() as u16).min(chat_area.width.saturating_sub(4));
                let sticky_area = Rect {
                    x: chat_area.x, // Sticky to left side matching actual section label direction
                    y: chat_area.y,
                    width: sticky_w,
                    height: 1,
                };
                let sticky_span = Span::styled(
                    sticky_label,
                    Style::default().fg(NORDIC_BG).bg(bg_col).add_modifier(Modifier::BOLD),
                );
                frame.render_widget(Paragraph::new(Line::from(sticky_span)), sticky_area);
            }
        }

        // Draw chips at their agent-turn anchors (visual line + scroll)
        {
            let _x = chat_area.x.saturating_add(3);
            let _max_w = chat_area.width.saturating_sub(6);
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
                if y > chat_bot {
                    continue;
                }
                chip.rect = Some(Rect {
                    x: chat_area.x,
                    y,
                    width: chat_area.width,
                    height: 2, // covers title row + summary row
                });
            }
        }

        self.last_chat_area = Some(chat_area);
        self.tool_panel_rect = None;

        // --- Input Area (Full width bar, 1-char padded user text lines) ---
        let input_area = chunks[2];
        let bar_w = input_area.width as usize;
        let raw_model = self.backend.name();
        let exact_model = if let Some(start) = raw_model.find('(') {
            if let Some(end) = raw_model.rfind(')') {
                if end > start {
                    raw_model[start + 1..end].trim().to_string()
                } else {
                    raw_model
                }
            } else {
                raw_model
            }
        } else {
            raw_model
        };
        let model_clean = if exact_model.chars().count() > 34 {
            format!("{}…", exact_model.chars().take(32).collect::<String>())
        } else {
            exact_model
        };
        // Extra spaces on both sides: "  {MODEL NAME}  "
        let badge_text = format!("  {}  ", model_clean);
        let badge_len = badge_text.chars().count();
        let right_trans = "🭆🭂"; // 2 transition characters
        let right_trans_len = 2;

        let main_badge_text = " Main ";
        let main_badge_len = main_badge_text.chars().count();
        let main_trans = "🭍🭑";
        let main_trans_len = 2;
        let left_main_total = main_badge_len + main_trans_len;

        let total_right_badge_w = right_trans_len + badge_len;
        let mid_bar_w = bar_w.saturating_sub(left_main_total + total_right_badge_w).max(1);

        let white_c = Color::Rgb(236, 239, 244); // #ECEFF4 Snow White

        let get_bar_color = |col_idx: usize, bar_width: usize| -> Color {
            if let Some(ref st) = stops {
                let norm = col_idx as f32 / bar_width.max(1) as f32;
                // 0.3 factor gradient smoothly traveling from left to right
                let cycle = ((norm * 0.3) - (phase * 0.04)).rem_euclid(1.0);
                let wave = if cycle < 0.5 {
                    cycle * 2.0
                } else {
                    (1.0 - cycle) * 2.0
                };
                multi_stop_gradient(st, wave)
            } else {
                white_c
            }
        };

        let get_contrast_text_color = |bg: Color| -> Color {
            if let Color::Rgb(r, g, b) = bg {
                let lum = 0.299 * (r as f32) + 0.587 * (g as f32) + 0.114 * (b as f32);
                if lum < 135.0 {
                    Color::Rgb(245, 248, 255) // Light text on dark bg
                } else {
                    NORDIC_BG // Dark text on light bg
                }
            } else {
                NORDIC_BG
            }
        };

        let mut bar_spans: Vec<Span> = Vec::new();
        let mut cur_col_idx = 0;

        // 1. Left " Main " badge
        let main_bg = get_bar_color(0, bar_w);
        let main_fg = get_contrast_text_color(main_bg);
        bar_spans.push(Span::styled(
            main_badge_text,
            Style::default().fg(main_fg).bg(main_bg).add_modifier(Modifier::BOLD),
        ));
        bar_spans.push(Span::styled(main_trans, Style::default().fg(main_bg).bg(NORDIC_BG)));
        cur_col_idx += left_main_total;

        // 2. Middle bar characters: 🬭 (smooth gradient character by character)
        for _ in 0..mid_bar_w {
            let col_c = get_bar_color(cur_col_idx, bar_w);
            bar_spans.push(Span::styled("🬭", Style::default().fg(col_c).bg(NORDIC_BG)));
            cur_col_idx += 1;
        }

        // 3. Right transition: 🭆🭂 and Model Name badge (single uniform bg color)
        let c_badge_bg = get_bar_color(cur_col_idx, bar_w);
        let c_badge_fg = get_contrast_text_color(c_badge_bg);

        bar_spans.push(Span::styled(right_trans, Style::default().fg(c_badge_bg).bg(NORDIC_BG)));
        cur_col_idx += right_trans_len;

        let badge_start_col = input_area.x + cur_col_idx as u16;
        bar_spans.push(Span::styled(
            badge_text,
            Style::default().fg(c_badge_fg).bg(c_badge_bg).add_modifier(Modifier::BOLD),
        ));
        cur_col_idx += badge_len;
        let badge_end_col = input_area.x + cur_col_idx as u16;

        // Record hit zone for clicking on Model Name to toggle focus
        self.model_badge_hit = Some((input_area.y, badge_start_col, badge_end_col));

        let mut input_ui_lines: Vec<Line> = vec![Line::from(bar_spans)];

        // Render content lines only if expanded (input_box_h > 1 && input_focused)
        if input_box_h > 1 && self.input_focused {
            let available_content_rows = input_box_h.saturating_sub(1);
            let (_cursor_col, cursor_row) = self.input_cursor_col_row(inner_w);

            let total_prompt_lines = wrapped_prompt_lines.len().max(1);
            let max_possible_scroll = total_prompt_lines.saturating_sub(available_content_rows as usize) as u16;

            // Auto-scroll input vertically so cursor is always visible
            if cursor_row >= self.input_scroll_y + available_content_rows {
                self.input_scroll_y = (cursor_row + 1).saturating_sub(available_content_rows);
            } else if cursor_row < self.input_scroll_y {
                self.input_scroll_y = cursor_row;
            }
            self.input_scroll_y = self.input_scroll_y.min(max_possible_scroll);

            if self.term_is_interactive() {
                input_ui_lines.push(Line::from(vec![
                    Span::styled(" $ ", Style::default().fg(Color::Rgb(163, 190, 140)).bg(NORDIC_BG)),
                    Span::styled(
                        if self.term_input.is_empty() {
                            "type shell command…".to_string()
                        } else {
                            self.term_input.clone()
                        },
                        Style::default().fg(if self.term_input.is_empty() {
                            NORDIC_MUTED
                        } else {
                            Color::Rgb(163, 190, 140)
                        }).bg(NORDIC_BG),
                    ),
                    Span::styled("█", Style::default().fg(Color::Rgb(163, 190, 140)).bg(NORDIC_BG)),
                ]));
            } else if self.input.is_empty() {
                input_ui_lines.push(Line::from(Span::styled(
                    " Type a prompt...",
                    Style::default().fg(NORDIC_MUTED).bg(NORDIC_BG),
                )));
            } else {
                let total_lines = wrapped_prompt_lines.len();
                let has_scrollbar = total_lines > available_content_rows as usize;
                let text_w = if has_scrollbar {
                    inner_w.saturating_sub(1)
                } else {
                    inner_w
                };
                let max_scroll = total_lines.saturating_sub(available_content_rows as usize);
                let thumb_height = (((available_content_rows as f32 / total_lines as f32) * available_content_rows as f32).round() as usize).max(1);
                let thumb_start = if max_scroll > 0 {
                    ((self.input_scroll_y as f32 / max_scroll as f32) * (available_content_rows as usize - thumb_height) as f32).round() as usize 
                } else {
                    0
                };

                let start_idx = (self.input_scroll_y as usize).min(total_lines);
                let end_idx = (start_idx + available_content_rows as usize).min(total_lines);

                for (r, line_str) in wrapped_prompt_lines[start_idx..end_idx].iter().enumerate() {
                    let mut row_spans = vec![
                        Span::raw(" "), // 1-character left padding for user text input
                        Span::styled(
                            line_str.clone(),
                            Style::default().fg(NORDIC_TEXT).bg(NORDIC_BG),
                        ),
                    ];

                    if has_scrollbar {
                        let line_len = line_str.chars().count();
                        let pad_spaces = text_w.saturating_sub(line_len);
                        if pad_spaces > 0 {
                            row_spans.push(Span::styled(" ".repeat(pad_spaces), Style::default().bg(NORDIC_BG)));
                        }
                        let is_thumb = r >= thumb_start && r < thumb_start + thumb_height;
                        let sb_char = if is_thumb { "┃" } else { "│" };
                        let sb_color = if is_thumb { Color::Rgb(143, 218, 255) } else { Color::Rgb(76, 86, 106) };
                        row_spans.push(Span::styled(sb_char, Style::default().fg(sb_color).bg(NORDIC_BG)));
                    }

                    input_ui_lines.push(Line::from(row_spans));
                }
            }
        }

        let input_box = Paragraph::new(input_ui_lines)
            .style(Style::default().bg(NORDIC_BG));
        frame.render_widget(input_box, input_area);

        let is_downloading = self.download_progress.lock().unwrap().is_some();
        if self.input_focused && !self.show_menu && !is_downloading && input_box_h > 1 {
            let (col, row) = self.input_cursor_col_row(inner_w);
            let available_content_rows = input_box_h.saturating_sub(1);
            let visible_rel_row = row.saturating_sub(self.input_scroll_y);
            if visible_rel_row < available_content_rows {
                let cursor_x = input_area.x.saturating_add(1).saturating_add(col as u16);
                let cursor_y = input_area.y.saturating_add(1).saturating_add(visible_rel_row as u16);
                let max_x = input_area.x + input_area.width.saturating_sub(1);
                let max_y = input_area.y + input_box_h.saturating_sub(1);
                frame.set_cursor_position((cursor_x.min(max_x), cursor_y.min(max_y)));
            }
        }

        // --- Custom Framed Container Modal Widget ---
        if self.show_menu {
            let anim_p = self.menu_anim_progress.clamp(0.0, 1.0);
            if anim_p > 0.01 {
                let full_w = area.width;
                let full_h = area.height;

                // Modal dimensions with smooth width and height slide/fade animation
                let target_w = (full_w.saturating_sub(12)).min(110).max(60);
                let target_h = (full_h.saturating_sub(8)).min(28).max(18);
                let modal_w = ((target_w as f32 * (0.3 + 0.7 * anim_p)).round() as u16).max(36);
                let modal_h = ((target_h as f32 * (0.3 + 0.7 * anim_p)).round() as u16).max(12);

                let modal_x = area.x + (full_w.saturating_sub(modal_w)) / 2;
                let modal_y = area.y + (full_h.saturating_sub(modal_h)) / 2;
                let container_rect = Rect {
                    x: modal_x,
                    y: modal_y,
                    width: modal_w,
                    height: modal_h,
                };

                // Clear container background with exact screen background
                frame.render_widget(Clear, container_rect);
                frame.render_widget(Block::default().style(Style::default().bg(NORDIC_BG)), container_rect);

                let menu_title_str = match self.menu_section {
                    0 => " Help ",
                    1 => " Registry ",
                    2 => " Modal ",
                    _ => " Settings ",
                };

                // Color transition with opacity fade
                let border_alpha = (255.0 * anim_p).round() as u8;
                let border_color = Color::Rgb(border_alpha, border_alpha, border_alpha);
                let close_btn_str = " x ";
                let close_btn_len = close_btn_str.chars().count() as u16;

                // Top Left: 🭈🭆🭂{ menu }🭞🭜
                // Top Right: 🭧🭓 x 🭍🭑🬽
                let tl_badge_w = 3 + menu_title_str.chars().count() as u16 + 2; // "🭈🭆🭂" + title + "🭞🭜"
                let tr_badge_w = 2 + close_btn_len + 3; // "🭧🭓" + " x " + "🭍🭑🬽"
                let top_bar_fill_w = modal_w.saturating_sub(tl_badge_w + tr_badge_w) as usize;

                // Record close button hit
                let close_btn_x0 = modal_x + modal_w.saturating_sub(tr_badge_w) + 2;
                let close_btn_x1 = close_btn_x0 + close_btn_len.saturating_sub(1);
                self.container_close_hit = Some((modal_y, close_btn_x0, close_btn_x1));

                // --- Row 0 (Top line) ---
                let mut row0_spans: Vec<Span> = Vec::new();
                // 🭈🭆🭂
                row0_spans.push(Span::styled("🭈🭆🭂", Style::default().fg(border_color).bg(NORDIC_BG)));
                // { menu } with white background and base text
                row0_spans.push(Span::styled(
                    menu_title_str,
                    Style::default().fg(NORDIC_BG).bg(border_color).add_modifier(Modifier::BOLD),
                ));
                // 🭞🭜
                row0_spans.push(Span::styled("🭞🭜", Style::default().fg(border_color).bg(NORDIC_BG)));
                // Top border fill 🬂
                for _ in 0..top_bar_fill_w {
                    row0_spans.push(Span::styled("🬂", Style::default().fg(border_color).bg(NORDIC_BG)));
                }
                // 🭧🭓
                row0_spans.push(Span::styled("🭧🭓", Style::default().fg(border_color).bg(NORDIC_BG)));
                // " x " close button
                row0_spans.push(Span::styled(
                    close_btn_str,
                    Style::default().fg(NORDIC_BG).bg(border_color).add_modifier(Modifier::BOLD),
                ));
                // 🭍🭑🬽
                row0_spans.push(Span::styled("🭍🭑🬽", Style::default().fg(border_color).bg(NORDIC_BG)));
                frame.render_widget(Paragraph::new(Line::from(row0_spans)).style(Style::default().bg(NORDIC_BG)), Rect {
                    x: modal_x,
                    y: modal_y,
                    width: modal_w,
                    height: 1,
                });

                // --- Row 1 (Top sub-corners) ---
                // Left: 🭝🭜🭘  Right: 🭣🭧🭒
                if modal_h >= 4 {
                    let mut row1_spans: Vec<Span> = Vec::new();
                    row1_spans.push(Span::styled("🭝🭜🭘", Style::default().fg(border_color).bg(NORDIC_BG)));
                    let middle_spaces = modal_w.saturating_sub(6) as usize;
                    if middle_spaces > 0 {
                        row1_spans.push(Span::raw(" ".repeat(middle_spaces)));
                    }
                    row1_spans.push(Span::styled("🭣🭧🭒", Style::default().fg(border_color).bg(NORDIC_BG)));
                    frame.render_widget(Paragraph::new(Line::from(row1_spans)).style(Style::default().bg(NORDIC_BG)), Rect {
                        x: modal_x,
                        y: modal_y + 1,
                        width: modal_w,
                        height: 1,
                    });
                }

                // --- Middle Rows (Left ▌ and Right ▐) ---
                for r in 2..modal_h.saturating_sub(2) {
                    let left_span = Span::styled("▌", Style::default().fg(border_color).bg(NORDIC_BG));
                    frame.render_widget(Paragraph::new(Line::from(left_span)), Rect {
                        x: modal_x,
                        y: modal_y + r,
                        width: 1,
                        height: 1,
                    });
                    let right_span = Span::styled("▐", Style::default().fg(border_color).bg(NORDIC_BG));
                    frame.render_widget(Paragraph::new(Line::from(right_span)), Rect {
                        x: modal_x + modal_w.saturating_sub(1),
                        y: modal_y + r,
                        width: 1,
                        height: 1,
                    });
                }

                // --- Row H-2 (Bottom sub-corners) ---
                // Left: 🭌🭑🬽  Right: 🭈🭆🭁
                if modal_h >= 4 {
                    let mut row_sub_b_spans: Vec<Span> = Vec::new();
                    row_sub_b_spans.push(Span::styled("🭌🭑🬽", Style::default().fg(border_color).bg(NORDIC_BG)));
                    let middle_spaces = modal_w.saturating_sub(6) as usize;
                    if middle_spaces > 0 {
                        row_sub_b_spans.push(Span::raw(" ".repeat(middle_spaces)));
                    }
                    row_sub_b_spans.push(Span::styled("🭈🭆🭁", Style::default().fg(border_color).bg(NORDIC_BG)));
                    frame.render_widget(Paragraph::new(Line::from(row_sub_b_spans)).style(Style::default().bg(NORDIC_BG)), Rect {
                        x: modal_x,
                        y: modal_y + modal_h.saturating_sub(2),
                        width: modal_w,
                        height: 1,
                    });
                }

                // --- Row H-1 (Bottom line) ---
                // Left: 🭣🭧🭓🭍🭑  Bottom line: 🬭  Right: 🭆🭂🭞🭜🭘
                let bl_w = 5u16; // "🭣🭧🭓🭍🭑"
                let br_w = 5u16; // "🭆🭂🭞🭜🭘"
                let bot_fill_w = modal_w.saturating_sub(bl_w + br_w) as usize;
                let mut row_bot_spans: Vec<Span> = Vec::new();
                row_bot_spans.push(Span::styled("🭣🭧🭓🭍🭑", Style::default().fg(border_color).bg(NORDIC_BG)));
                for _ in 0..bot_fill_w {
                    row_bot_spans.push(Span::styled("🬭", Style::default().fg(border_color).bg(NORDIC_BG)));
                }
                row_bot_spans.push(Span::styled("🭆🭂🭞🭜🭘", Style::default().fg(border_color).bg(NORDIC_BG)));
                frame.render_widget(Paragraph::new(Line::from(row_bot_spans)).style(Style::default().bg(NORDIC_BG)), Rect {
                    x: modal_x,
                    y: modal_y + modal_h.saturating_sub(1),
                    width: modal_w,
                    height: 1,
                });

                // --- Inner Container Content Area ---
                let content_inner = Rect {
                    x: modal_x + 3,
                    y: modal_y + 2,
                    width: modal_w.saturating_sub(6),
                    height: modal_h.saturating_sub(4),
                };

                match self.menu_section {
                    0 => {
                        // === Help Section ===
                        let help_lines = vec![
                            Line::from(Span::styled(" Hercules Keyboard Navigation & Quick Reference ", Style::default().fg(Color::White).bg(NORDIC_BG).add_modifier(Modifier::BOLD))),
                            Line::from(Span::styled("", Style::default().bg(NORDIC_BG))),
                            Line::from(vec![
                                Span::styled(" F1 ", Style::default().fg(NORDIC_BG).bg(Color::White).add_modifier(Modifier::BOLD)),
                                Span::styled("          Help & Keybindings guide", Style::default().fg(NORDIC_TEXT).bg(NORDIC_BG)),
                            ]),
                            Line::from(vec![
                                Span::styled(" F2 ", Style::default().fg(NORDIC_BG).bg(Color::White).add_modifier(Modifier::BOLD)),
                                Span::styled("          Model Registry (Download from HuggingFace & Ollama)", Style::default().fg(NORDIC_TEXT).bg(NORDIC_BG)),
                            ]),
                            Line::from(vec![
                                Span::styled(" F3 ", Style::default().fg(NORDIC_BG).bg(Color::White).add_modifier(Modifier::BOLD)),
                                Span::styled("          Modal (Choose & activate installed local models)", Style::default().fg(NORDIC_TEXT).bg(NORDIC_BG)),
                            ]),
                            Line::from(vec![
                                Span::styled(" F4 ", Style::default().fg(NORDIC_BG).bg(Color::White).add_modifier(Modifier::BOLD)),
                                Span::styled("          Settings (Power mode, stall watchdog, permissions)", Style::default().fg(NORDIC_TEXT).bg(NORDIC_BG)),
                            ]),
                            Line::from(Span::styled("", Style::default().bg(NORDIC_BG))),
                            Line::from(vec![
                                Span::styled(" Esc ", Style::default().fg(Color::Rgb(255, 180, 180)).bg(NORDIC_BG).add_modifier(Modifier::BOLD)),
                                Span::styled("         Close menu / (Hold 1s) Quit application", Style::default().fg(NORDIC_TEXT).bg(NORDIC_BG)),
                            ]),
                            Line::from(vec![
                                Span::styled(" Ctrl+Esc ", Style::default().fg(Color::Rgb(255, 120, 120)).bg(NORDIC_BG).add_modifier(Modifier::BOLD)),
                                Span::styled("    Exit immediately", Style::default().fg(NORDIC_TEXT).bg(NORDIC_BG)),
                            ]),
                            Line::from(vec![
                                Span::styled(" Ctrl+F ", Style::default().fg(Color::Rgb(143, 218, 255)).bg(NORDIC_BG).add_modifier(Modifier::BOLD)),
                                Span::styled("      Focus / Unfocus user prompt bar", Style::default().fg(NORDIC_TEXT).bg(NORDIC_BG)),
                            ]),
                            Line::from(vec![
                                Span::styled(" Ctrl+C ", Style::default().fg(Color::Rgb(255, 200, 100)).bg(NORDIC_BG).add_modifier(Modifier::BOLD)),
                                Span::styled("      Interrupt streaming response or tool execution", Style::default().fg(NORDIC_TEXT).bg(NORDIC_BG)),
                            ]),
                            Line::from(vec![
                                Span::styled(" PgUp / PgDn ", Style::default().fg(Color::Rgb(180, 160, 255)).bg(NORDIC_BG).add_modifier(Modifier::BOLD)),
                                Span::styled(" Scroll conversation history", Style::default().fg(NORDIC_TEXT).bg(NORDIC_BG)),
                            ]),
                        ];
                        frame.render_widget(Paragraph::new(help_lines).style(Style::default().bg(NORDIC_BG)), content_inner);
                    }
                    1 => {
                        // === Registry Section (Tab Bar + Search Bar + Filtered Model List) ===
                        let chunks = Layout::default()
                            .direction(Direction::Vertical)
                            .constraints([Constraint::Length(2), Constraint::Length(2), Constraint::Min(1)].as_ref())
                            .split(content_inner);

                        let hf_style = if self.registry_tab == 0 {
                            Style::default().fg(NORDIC_BG).bg(Color::White).add_modifier(Modifier::BOLD)
                        } else {
                            Style::default().fg(Color::Rgb(160, 180, 200)).bg(NORDIC_BG)
                        };
                        let ol_style = if self.registry_tab == 1 {
                            Style::default().fg(NORDIC_BG).bg(Color::White).add_modifier(Modifier::BOLD)
                        } else {
                            Style::default().fg(Color::Rgb(160, 180, 200)).bg(NORDIC_BG)
                        };

                        let tab_bar = Paragraph::new(Line::from(vec![
                            Span::styled(" [ HuggingFace Models ] ", hf_style),
                            Span::styled("  ", Style::default().bg(NORDIC_BG)),
                            Span::styled(" [ Ollama Models ] ", ol_style),
                            Span::styled("   (Left/Right to switch tab | Enter to download)", Style::default().fg(Color::Rgb(120, 140, 160)).bg(NORDIC_BG)),
                        ])).style(Style::default().bg(NORDIC_BG));
                        frame.render_widget(tab_bar, chunks[0]);

                        // Search Bar
                        let search_text = if self.registry_search_query.is_empty() {
                            Span::styled(" Search models (type to filter query)...", Style::default().fg(Color::Rgb(100, 120, 140)).bg(NORDIC_BG))
                        } else {
                            Span::styled(format!(" Search: {}", self.registry_search_query), Style::default().fg(Color::White).bg(NORDIC_BG).add_modifier(Modifier::BOLD))
                        };
                        frame.render_widget(Paragraph::new(Line::from(vec![search_text])).style(Style::default().bg(NORDIC_BG)), chunks[1]);

                        let q_raw = self.registry_search_query.trim();
                        let q_lower = q_raw.to_lowercase();
                        let total_w = chunks[2].width as usize;

                        // Helper closure to build highlighted spans with bold matching chars
                        let build_highlighted_spans = |text: &str, query: &str, base_style: Style, match_style: Style| -> Vec<Span> {
                            if query.is_empty() {
                                return vec![Span::styled(text.to_string(), base_style)];
                            }
                            let text_lower = text.to_lowercase();
                            let mut spans = Vec::new();
                            let mut last_idx = 0;
                            
                            // Check if query is in text as substring or match sub-tokens
                            let search_term = if let Some(slash_idx) = query.rfind('/') {
                                &query[slash_idx + 1..]
                            } else {
                                query
                            };
                            let needle = search_term.to_lowercase();

                            if !needle.is_empty() && text_lower.contains(&needle) {
                                for (match_start, _) in text_lower.match_indices(&needle) {
                                    if match_start > last_idx {
                                        spans.push(Span::styled(text[last_idx..match_start].to_string(), base_style));
                                    }
                                    let match_end = match_start + needle.len();
                                    spans.push(Span::styled(text[match_start..match_end].to_string(), match_style));
                                    last_idx = match_end;
                                }
                                if last_idx < text.len() {
                                    spans.push(Span::styled(text[last_idx..].to_string(), base_style));
                                }
                            } else {
                                spans.push(Span::styled(text.to_string(), base_style));
                            }
                            spans
                        };

                        let items: Vec<ListItem> = if self.registry_tab == 0 {
                            let selected_idx = self.registry_state.selected();
                            self.hf_models.iter()
                                .filter(|m| q_lower.is_empty() || m.to_lowercase().contains(&q_lower) || {
                                    let sub = if let Some(idx) = q_lower.rfind('/') { &q_lower[idx+1..] } else { &q_lower };
                                    !sub.is_empty() && m.to_lowercase().contains(sub)
                                })
                                .enumerate()
                                .map(|(row_idx, m)| {
                                    let is_selected = selected_idx == Some(row_idx);
                                    let row_bg = if is_selected { Color::Rgb(59, 66, 82) } else { NORDIC_BG };

                                    // Split org/repo [size] into columns: Org │ Model │ Size
                                    let (repo_part, size_part) = if let Some(bracket_idx) = m.find('[') {
                                        (m[..bracket_idx].trim(), m[bracket_idx..].trim())
                                    } else {
                                        (m.trim(), "")
                                    };
                                    let (org, model_name) = if let Some(slash_idx) = repo_part.find('/') {
                                        (&repo_part[..slash_idx], &repo_part[slash_idx + 1..])
                                    } else {
                                        ("-", repo_part)
                                    };

                                    let org_col = format!("{:<14}", if org.len() > 14 { &org[..14] } else { org });
                                    let size_clean = size_part.trim_matches(|c| c == '[' || c == ']').replace("Q4 est.", "est").replace("GGUF", "").trim().to_string();
                                    let size_col = format!("{:>12}", if size_clean.len() > 12 { &size_clean[..12] } else { &size_clean });
                                    
                                    let used_w = 14 + 3 + 12 + 3; // org + " │ " + size + " │ "
                                    let model_max_w = total_w.saturating_sub(used_w).max(10);
                                    let model_col = format!("{:<width$}", if model_name.len() > model_max_w { &model_name[..model_max_w] } else { model_name }, width = model_max_w);

                                    let org_base_style = Style::default().fg(if is_selected { Color::Rgb(143, 218, 255) } else { Color::Rgb(136, 192, 208) }).bg(row_bg);
                                    let org_match_style = Style::default().fg(Color::White).bg(Color::Rgb(94, 129, 172)).add_modifier(Modifier::BOLD);
                                    let org_spans = build_highlighted_spans(&org_col, q_raw, org_base_style, org_match_style);

                                    let model_base_style = Style::default().fg(if is_selected { Color::White } else { Color::Rgb(220, 230, 242) }).bg(row_bg).add_modifier(if is_selected { Modifier::BOLD } else { Modifier::empty() });
                                    let model_match_style = Style::default().fg(Color::Rgb(235, 203, 139)).bg(Color::Rgb(67, 76, 94)).add_modifier(Modifier::BOLD);
                                    let model_spans = build_highlighted_spans(&model_col, q_raw, model_base_style, model_match_style);

                                    let mut line_spans = Vec::new();
                                    line_spans.extend(org_spans);
                                    line_spans.push(Span::styled(" │ ", Style::default().fg(Color::Rgb(76, 86, 106)).bg(row_bg)));
                                    line_spans.extend(model_spans);
                                    line_spans.push(Span::styled(" │ ", Style::default().fg(Color::Rgb(76, 86, 106)).bg(row_bg)));
                                    line_spans.push(Span::styled(size_col, Style::default().fg(if is_selected { Color::Rgb(180, 240, 160) } else { Color::Rgb(163, 190, 140) }).bg(row_bg).add_modifier(if is_selected { Modifier::BOLD } else { Modifier::empty() })));

                                    ListItem::new(Line::from(line_spans)).style(Style::default().bg(row_bg))
                                }).collect()
                        } else {
                            let selected_idx = self.registry_state.selected();
                            self.registry_models.iter()
                                .filter(|m| q_lower.is_empty() || m.to_lowercase().contains(&q_lower) || {
                                    let sub = if let Some(idx) = q_lower.rfind('/') { &q_lower[idx+1..] } else { &q_lower };
                                    !sub.is_empty() && m.to_lowercase().contains(sub)
                                })
                                .enumerate()
                                .map(|(row_idx, m)| {
                                    let is_selected = selected_idx == Some(row_idx);
                                    let row_bg = if is_selected { Color::Rgb(59, 66, 82) } else { NORDIC_BG };

                                    let (name_part, size_part) = if let Some(idx) = m.find('(') {
                                        (m[..idx].trim(), m[idx..].trim_matches(|c| c == '(' || c == ')').trim())
                                    } else if let Some(idx) = m.find('[') {
                                        (m[..idx].trim(), m[idx..].trim_matches(|c| c == '[' || c == ']').trim())
                                    } else {
                                        (m.trim(), "")
                                    };

                                    let (org, model_name) = if let Some(slash_idx) = name_part.find('/') {
                                        (&name_part[..slash_idx], &name_part[slash_idx + 1..])
                                    } else {
                                        ("ollama", name_part)
                                    };

                                    let org_col = format!("{:<14}", if org.len() > 14 { &org[..14] } else { org });
                                    let size_col = format!("{:>12}", if size_part.len() > 12 { &size_part[..12] } else { size_part });
                                    let used_w = 14 + 3 + 12 + 3;
                                    let model_max_w = total_w.saturating_sub(used_w).max(10);
                                    let model_col = format!("{:<width$}", if model_name.len() > model_max_w { &model_name[..model_max_w] } else { model_name }, width = model_max_w);

                                    let org_base_style = Style::default().fg(if is_selected { Color::Rgb(143, 218, 255) } else { Color::Rgb(136, 192, 208) }).bg(row_bg);
                                    let org_match_style = Style::default().fg(Color::White).bg(Color::Rgb(94, 129, 172)).add_modifier(Modifier::BOLD);
                                    let org_spans = build_highlighted_spans(&org_col, q_raw, org_base_style, org_match_style);

                                    let model_base_style = Style::default().fg(if is_selected { Color::White } else { Color::Rgb(220, 230, 242) }).bg(row_bg).add_modifier(if is_selected { Modifier::BOLD } else { Modifier::empty() });
                                    let model_match_style = Style::default().fg(Color::Rgb(235, 203, 139)).bg(Color::Rgb(67, 76, 94)).add_modifier(Modifier::BOLD);
                                    let model_spans = build_highlighted_spans(&model_col, q_raw, model_base_style, model_match_style);

                                    let mut line_spans = Vec::new();
                                    line_spans.extend(org_spans);
                                    line_spans.push(Span::styled(" │ ", Style::default().fg(Color::Rgb(76, 86, 106)).bg(row_bg)));
                                    line_spans.extend(model_spans);
                                    line_spans.push(Span::styled(" │ ", Style::default().fg(Color::Rgb(76, 86, 106)).bg(row_bg)));
                                    line_spans.push(Span::styled(size_col, Style::default().fg(if is_selected { Color::Rgb(180, 240, 160) } else { Color::Rgb(163, 190, 140) }).bg(row_bg).add_modifier(if is_selected { Modifier::BOLD } else { Modifier::empty() })));

                                    ListItem::new(Line::from(line_spans)).style(Style::default().bg(row_bg))
                                }).collect()
                        };

                        if items.is_empty() {
                            let empty_msg = if q_raw.is_empty() {
                                "Loading model catalog from Hugging Face & Ollama..."
                            } else {
                                "No models found matching query. Press Backspace or search another author/model."
                            };
                            let p = Paragraph::new(Line::from(vec![
                                Span::styled(format!("  {}", empty_msg), Style::default().fg(Color::Rgb(160, 180, 200)).bg(NORDIC_BG))
                            ])).style(Style::default().bg(NORDIC_BG));
                            frame.render_widget(p, chunks[2]);
                        } else {
                            let list = List::new(items)
                                .style(Style::default().bg(NORDIC_BG));
                            frame.render_stateful_widget(list, chunks[2], &mut self.registry_state);
                        }
                    }
                    2 => {
                        // === Modal (Installed Models) Section ===
                        let chunks = Layout::default()
                            .direction(Direction::Vertical)
                            .constraints([Constraint::Length(2), Constraint::Min(1)].as_ref())
                            .split(content_inner);

                        let info = Paragraph::new(Line::from(vec![
                            Span::styled(" Installed Models ", Style::default().fg(Color::White).bg(NORDIC_BG).add_modifier(Modifier::BOLD)),
                            Span::styled(" (Up/Down or W/S to navigate | Enter to activate model)", Style::default().fg(Color::Rgb(120, 140, 160)).bg(NORDIC_BG)),
                        ])).style(Style::default().bg(NORDIC_BG));
                        frame.render_widget(info, chunks[0]);

                        let selected_idx = self.installed_state.selected();
                        let items: Vec<ListItem> = self.installed_models.iter().enumerate().map(|(row_idx, m)| {
                            let is_selected = selected_idx == Some(row_idx);
                            let row_bg = if is_selected { Color::Rgb(59, 66, 82) } else { NORDIC_BG };

                            let is_active = self.backend.name().contains(m) || m.contains(&self.backend.name());
                            let (badge_txt, badge_fg, badge_bg) = if is_active {
                                (" ACTIVE ", NORDIC_BG, Color::Rgb(163, 190, 140))
                            } else {
                                (" READY  ", NORDIC_BG, Color::Rgb(76, 86, 106))
                            };

                            let clean_name = m.replace("Local GGUF:", "").replace("Ollama Local:", "").trim().to_string();
                            let (repo, model_label) = if let Some(idx) = clean_name.find('/') {
                                (&clean_name[..idx], clean_name[idx+1..].trim())
                            } else {
                                ("local", clean_name.as_str())
                            };

                            let repo_col = format!("{:<14}", if repo.len() > 14 { &repo[..14] } else { repo });
                            let total_w = chunks[1].width as usize;
                            let used_w = 9 + 1 + 14 + 3; // badge + space + repo + " │ "
                            let name_max_w = total_w.saturating_sub(used_w).max(10);
                            let name_col = format!("{:<width$}", if model_label.len() > name_max_w { &model_label[..name_max_w] } else { model_label }, width = name_max_w);

                            ListItem::new(Line::from(vec![
                                Span::styled(badge_txt, Style::default().fg(badge_fg).bg(badge_bg).add_modifier(Modifier::BOLD)),
                                Span::styled(" ", Style::default().bg(row_bg)),
                                Span::styled(repo_col, Style::default().fg(if is_selected { Color::Rgb(143, 218, 255) } else { Color::Rgb(136, 192, 208) }).bg(row_bg)),
                                Span::styled(" │ ", Style::default().fg(Color::Rgb(76, 86, 106)).bg(row_bg)),
                                Span::styled(name_col, Style::default().fg(if is_selected { Color::White } else { Color::Rgb(220, 230, 242) }).bg(row_bg).add_modifier(if is_selected { Modifier::BOLD } else { Modifier::empty() })),
                            ])).style(Style::default().bg(row_bg))
                        }).collect();

                        let list = List::new(items)
                            .style(Style::default().bg(NORDIC_BG));
                        frame.render_stateful_widget(list, chunks[1], &mut self.installed_state);
                    }
                    _ => {
                        // === Settings Section (Two-Column Layout) ===
                        let s = crate::settings::get_settings();
                        let p = get_tool_permissions();
                        let ctx_n = crate::settings::context_token_limit();
                        let ctx_label = crate::settings::format_context_tokens(ctx_n);

                        let cols = Layout::default()
                            .direction(Direction::Horizontal)
                            .constraints([Constraint::Length(22), Constraint::Length(2), Constraint::Min(1)].as_ref())
                            .split(content_inner);

                        // Column 1: Tabs (w/Up to go up, s/Down to go down)
                        let tab_names = [
                            "Power Mode",
                            "Stall Time",
                            "Repeat Detector",
                            "Context Window",
                            "Permissions",
                            "HF Token",
                        ];

                        let mut tab_items: Vec<ListItem> = Vec::new();
                        for (idx, name) in tab_names.iter().enumerate() {
                            let is_selected = self.settings_tab == idx;
                            let is_focused_col = self.settings_col == 0;
                            let (fg, bg) = if is_selected && is_focused_col {
                                (NORDIC_BG, Color::White)
                            } else if is_selected {
                                (Color::White, Color::Rgb(59, 66, 82))
                            } else {
                                (Color::Rgb(160, 175, 195), NORDIC_BG)
                            };

                            let symbol = if is_selected { "● " } else { "  " };
                            tab_items.push(ListItem::new(Line::from(vec![
                                Span::styled(symbol, Style::default().fg(fg).bg(bg).add_modifier(Modifier::BOLD)),
                                Span::styled(*name, Style::default().fg(fg).bg(bg).add_modifier(if is_selected { Modifier::BOLD } else { Modifier::empty() })),
                            ])));
                        }

                        let col1_list = List::new(tab_items).style(Style::default().bg(NORDIC_BG));
                        frame.render_widget(col1_list, cols[0]);

                        // Column separator
                        let sep_lines: Vec<Line> = (0..cols[1].height).map(|_| Line::from(Span::styled("│", Style::default().fg(Color::Rgb(76, 86, 106)).bg(NORDIC_BG)))).collect();
                        frame.render_widget(Paragraph::new(sep_lines), cols[1]);

                        // Column 2: Value options
                        let col2_focus = self.settings_col == 1;
                        let focus_badge = if self.settings_tab == 5 {
                            if self.hf_token_editing {
                                Span::styled(" [EDITING: Type token | Enter=Save | Esc=Cancel] ", Style::default().fg(NORDIC_BG).bg(Color::Rgb(163, 190, 140)).add_modifier(Modifier::BOLD))
                            } else if col2_focus {
                                Span::styled(" [FOCUSED: Enter=Edit/Add | D/Delete=Remove] ", Style::default().fg(NORDIC_BG).bg(Color::Rgb(143, 218, 255)).add_modifier(Modifier::BOLD))
                            } else {
                                Span::styled(" [Press Enter to Configure Token] ", Style::default().fg(Color::Rgb(120, 140, 160)).bg(NORDIC_BG))
                            }
                        } else if col2_focus {
                            Span::styled(" [FOCUSED: A/D or Left/Right to change] ", Style::default().fg(NORDIC_BG).bg(Color::Rgb(143, 218, 255)).add_modifier(Modifier::BOLD))
                        } else {
                            Span::styled(" [Press Enter to Edit Value] ", Style::default().fg(Color::Rgb(120, 140, 160)).bg(NORDIC_BG))
                        };

                        let mut val_lines: Vec<Line> = vec![
                            Line::from(focus_badge),
                            Line::from(Span::styled("", Style::default().bg(NORDIC_BG))),
                        ];

                        match self.settings_tab {
                            0 => {
                                // Power mode options
                                let modes = [
                                    (crate::settings::PowerMode::PowerSaver, "Power Saver (ease off when CPU is warm)"),
                                    (crate::settings::PowerMode::Normal, "Normal (default - auto threads & GPU offload)"),
                                    (crate::settings::PowerMode::Extreme, "Extreme (max GPU layers & full CPU threads)"),
                                ];
                                for (m, desc) in modes {
                                    let active = s.power_mode == m;
                                    let sym = if active { "● " } else { "○ " };
                                    let color = if active { Color::Rgb(163, 190, 140) } else { Color::Rgb(160, 175, 195) };
                                    val_lines.push(Line::from(vec![
                                        Span::styled(sym, Style::default().fg(color).bg(NORDIC_BG).add_modifier(Modifier::BOLD)),
                                        Span::styled(desc, Style::default().fg(if active { Color::White } else { color }).bg(NORDIC_BG).add_modifier(if active { Modifier::BOLD } else { Modifier::empty() })),
                                    ]));
                                }
                            }
                            1 => {
                                // Stall watchdog options
                                val_lines.push(Line::from(vec![
                                    Span::styled("Watchdog Timeout: ", Style::default().fg(Color::White).bg(NORDIC_BG).add_modifier(Modifier::BOLD)),
                                    Span::styled(crate::settings::format_stall_timeout(s.stall_timeout_secs), Style::default().fg(Color::Rgb(235, 203, 139)).bg(NORDIC_BG).add_modifier(Modifier::BOLD)),
                                ]));
                                val_lines.push(Line::from(Span::styled("Cycles: 5 min → 10 min → 20 min → Unlimited", Style::default().fg(Color::Rgb(120, 140, 160)).bg(NORDIC_BG))));
                            }
                            2 => {
                                // Repeat detector
                                val_lines.push(Line::from(vec![
                                    Span::styled("Repeat Threshold: ", Style::default().fg(Color::White).bg(NORDIC_BG).add_modifier(Modifier::BOLD)),
                                    Span::styled(format!("{} consecutive outputs", s.repeat_threshold), Style::default().fg(Color::Rgb(143, 218, 255)).bg(NORDIC_BG).add_modifier(Modifier::BOLD)),
                                ]));
                                val_lines.push(Line::from(vec![
                                    Span::styled("Detect on Thinking: ", Style::default().fg(Color::White).bg(NORDIC_BG).add_modifier(Modifier::BOLD)),
                                    Span::styled(if s.repeat_detect_thinking { "ENABLED" } else { "DISABLED" }, Style::default().fg(if s.repeat_detect_thinking { Color::Rgb(163, 190, 140) } else { Color::Rgb(255, 120, 120) }).bg(NORDIC_BG).add_modifier(Modifier::BOLD)),
                                ]));
                            }
                            3 => {
                                // Context window
                                val_lines.push(Line::from(vec![
                                    Span::styled("Context Window Limit: ", Style::default().fg(Color::White).bg(NORDIC_BG).add_modifier(Modifier::BOLD)),
                                    Span::styled(format!("{} ({} tokens)", ctx_label, ctx_n), Style::default().fg(Color::Rgb(180, 160, 255)).bg(NORDIC_BG).add_modifier(Modifier::BOLD)),
                                ]));
                                val_lines.push(Line::from(Span::styled("Cycles: 4K → 8K → 16K → 32K → 64K → 128K → 250K → 1M", Style::default().fg(Color::Rgb(120, 140, 160)).bg(NORDIC_BG))));
                            }
                            4 => {
                                // Permissions
                                val_lines.push(Line::from(vec![
                                    Span::styled("Action Permission Mode: ", Style::default().fg(Color::White).bg(NORDIC_BG).add_modifier(Modifier::BOLD)),
                                    Span::styled(p.mode_label(), Style::default().fg(Color::Rgb(143, 218, 255)).bg(NORDIC_BG).add_modifier(Modifier::BOLD)),
                                ]));
                                val_lines.push(Line::from(vec![
                                    Span::styled("Directory Access Scope: ", Style::default().fg(Color::White).bg(NORDIC_BG).add_modifier(Modifier::BOLD)),
                                    Span::styled(p.scope_label(), Style::default().fg(Color::Rgb(235, 203, 139)).bg(NORDIC_BG).add_modifier(Modifier::BOLD)),
                                ]));
                            }
                            _ => {
                                // HuggingFace Token
                                let tok_opt = crate::settings::get_hf_token();
                                let has_token = tok_opt.is_some();
                                let masked_token = if let Some(ref t) = tok_opt {
                                    if t.len() > 10 {
                                        format!("{}...{}", &t[..4], &t[t.len() - 4..])
                                    } else {
                                        "********".to_string()
                                    }
                                } else {
                                    "None (Unauthenticated / Anonymous)".to_string()
                                };

                                val_lines.push(Line::from(vec![
                                    Span::styled("Current Token: ", Style::default().fg(Color::White).bg(NORDIC_BG).add_modifier(Modifier::BOLD)),
                                    Span::styled(
                                        masked_token,
                                        Style::default().fg(if has_token { Color::Rgb(163, 190, 140) } else { Color::Rgb(235, 203, 139) }).bg(NORDIC_BG).add_modifier(Modifier::BOLD),
                                    ),
                                ]));
                                val_lines.push(Line::from(Span::styled(
                                    "Used for Hugging Face model registry searches and GGUF downloads.",
                                    Style::default().fg(Color::Rgb(120, 140, 160)).bg(NORDIC_BG),
                                )));
                                val_lines.push(Line::from(Span::styled("", Style::default().bg(NORDIC_BG))));

                                if self.hf_token_editing {
                                    val_lines.push(Line::from(vec![
                                        Span::styled("New Token: ", Style::default().fg(Color::Rgb(143, 218, 255)).bg(NORDIC_BG).add_modifier(Modifier::BOLD)),
                                        Span::styled(&self.hf_token_input, Style::default().fg(Color::White).bg(Color::Rgb(46, 52, 64))),
                                        Span::styled("▍", Style::default().fg(Color::Rgb(143, 218, 255)).bg(Color::Rgb(46, 52, 64))),
                                    ]));
                                    val_lines.push(Line::from(Span::styled(
                                        "Paste or type token (starts with 'hf_...'), then press Enter to save.",
                                        Style::default().fg(Color::Rgb(160, 175, 195)).bg(NORDIC_BG),
                                    )));
                                } else {
                                    val_lines.push(Line::from(vec![
                                        Span::styled("[ Enter ] ", Style::default().fg(Color::Rgb(143, 218, 255)).bg(NORDIC_BG).add_modifier(Modifier::BOLD)),
                                        Span::styled(if has_token { "Change / Overwrite Token" } else { "Add HF Token" }, Style::default().fg(Color::White).bg(NORDIC_BG)),
                                    ]));
                                    if has_token {
                                        val_lines.push(Line::from(vec![
                                            Span::styled("[ D / Del ] ", Style::default().fg(Color::Rgb(255, 120, 120)).bg(NORDIC_BG).add_modifier(Modifier::BOLD)),
                                            Span::styled("Remove / Clear Saved Token", Style::default().fg(Color::White).bg(NORDIC_BG)),
                                        ]));
                                    }
                                }
                            }
                        }

                        frame.render_widget(Paragraph::new(val_lines).style(Style::default().bg(NORDIC_BG)), cols[2]);
                    }
                }
            }
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
            frame.render_widget(Block::default().style(Style::default().bg(NORDIC_DARK_BG)), area);

            let block = Block::default()
                .style(Style::default().bg(NORDIC_DARK_BG))
                .borders(Borders::ALL)
                .border_type(BorderType::Double)
                .border_style(Style::default().fg(Color::Rgb(255, 80, 80)))
                .title(" Confirm Model Deletion (Agreement Required) ");

            let confirm_lines = vec![
                Line::from(Span::styled(
                    "Model Deletion Confirmation",
                    Style::default()
                        .fg(Color::Rgb(255, 80, 80))
                        .bg(NORDIC_DARK_BG)
                        .add_modifier(Modifier::BOLD),
                )),
                Line::from(Span::styled("", Style::default().bg(NORDIC_DARK_BG))),
                Line::from(Span::styled(format!("Target Model: {}", target), Style::default().fg(NORDIC_TEXT).bg(NORDIC_DARK_BG))),
                Line::from(Span::styled("Are you sure you want to delete this model weight from memory/disk?", Style::default().fg(NORDIC_TEXT).bg(NORDIC_DARK_BG))),
                Line::from(Span::styled("", Style::default().bg(NORDIC_DARK_BG))),
                Line::from(vec![
                    Span::styled(
                        " [Y] Confirm Delete ",
                        Style::default()
                            .bg(Color::Rgb(255, 50, 50))
                            .fg(Color::White)
                            .add_modifier(Modifier::BOLD),
                    ),
                    Span::styled("   ", Style::default().bg(NORDIC_DARK_BG)),
                    Span::styled(
                        " [N / Esc] Cancel ",
                        Style::default().bg(NORDIC_BG).fg(NORDIC_TEXT),
                    ),
                ]),
            ];

            let dialog = Paragraph::new(confirm_lines)
                .style(Style::default().bg(NORDIC_DARK_BG))
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
