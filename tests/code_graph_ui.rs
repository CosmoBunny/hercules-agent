//! Code Graph (F6) TUI integration tests:
//! F6 toggle, loading state, empty graph, node styles by kind, selected background
//! highlight (no border), details pane relationship counts, and layout determinism.

use crossterm::event::{KeyCode, KeyModifiers};
use hercules_agent::app::{App, CodeGraphPane};
use hercules_agent::code_graph::{CodeGraphBuilder, CodeGraphConfig, EdgeKind};
use ratatui::backend::TestBackend;
use ratatui::style::Color;
use ratatui::Terminal;

fn key(code: KeyCode) -> crossterm::event::KeyEvent {
    crossterm::event::KeyEvent::new(code, KeyModifiers::NONE)
}

fn render_app(app: &mut App, w: u16, h: u16) -> String {
    let backend = TestBackend::new(w, h);
    let mut terminal = Terminal::new(backend).unwrap();
    terminal.draw(|f| app.draw(f)).unwrap();
    let buffer = terminal.backend().buffer().clone();
    let mut out = String::new();
    for y in 0..buffer.area.height {
        for x in 0..buffer.area.width {
            out.push(buffer[(x, y)].symbol().chars().next().unwrap_or(' '));
        }
        out.push('\n');
    }
    out
}

fn buffer(app: &mut App, w: u16, h: u16) -> ratatui::buffer::Buffer {
    let backend = TestBackend::new(w, h);
    let mut terminal = Terminal::new(backend).unwrap();
    terminal.draw(|f| app.draw(f)).unwrap();
    terminal.backend().buffer().clone()
}

fn demo_graph() -> hercules_agent::code_graph::CodeGraph {
    let config = CodeGraphConfig::default();
    let mut builder = CodeGraphBuilder::new(config).unwrap();
    builder
        .add_file(
            "fn aaa() {\n    bbb();\n}\n\nfn bbb() {}\n",
            std::path::PathBuf::from("demo.rs"),
        )
        .unwrap()
}

/// Fill App code-graph state from a graph (incl. adjacency maps, as the UI does on load)
fn install_graph(app: &mut App, graph: hercules_agent::code_graph::CodeGraph) {
    let mut outgoing = std::collections::HashMap::new();
    let mut incoming = std::collections::HashMap::new();
    for (idx, edge) in graph.edges.iter().enumerate() {
        outgoing
            .entry(edge.from.clone())
            .or_insert_with(Vec::new)
            .push(idx);
        incoming
            .entry(edge.to.clone())
            .or_insert_with(Vec::new)
            .push(idx);
    }
    app.code_graph = Some(graph);
    app.code_graph_outgoing = outgoing;
    app.code_graph_incoming = incoming;
    app.code_graph_loading = false;
    app.code_graph_error = None;
}

#[tokio::test(flavor = "multi_thread")]
async fn f6_toggles_code_graph_panel() {
    let mut app = App::new();
    app.code_graph_loading = true; // avoid workspace rebuild during test
    app.handle_key(key(KeyCode::F(6))).await;
    assert!(app.show_menu, "F6 must open the menu");
    assert_eq!(app.menu_section, 5, "F6 must select the Code Graph section");
    assert_eq!(app.code_graph_pane, CodeGraphPane::Graph);

    // F6 again closes (closing animation state)
    app.handle_key(key(KeyCode::F(6))).await;
    assert!(app.menu_closing, "F6 on open Code Graph must close it");

    // Esc also closes
    let mut app2 = App::new();
    app2.code_graph_loading = true;
    app2.handle_key(key(KeyCode::F(6))).await;
    app2.handle_key(key(KeyCode::Esc)).await;
    assert!(app2.menu_closing, "Esc must close the Code Graph panel");
}

#[tokio::test(flavor = "multi_thread")]
async fn loading_state_is_shown() {
    let mut app = App::new();
    app.show_menu = true;
    app.menu_anim_progress = 1.0;
    app.menu_section = 5;
    app.code_graph_loading = true;
    let out = render_app(&mut app, 120, 40);
    assert!(
        out.contains("Loading project graph"),
        "loading text missing:\n{}",
        out
    );
    assert!(
        out.contains("Tree-sitter"),
        "loading text must mention Tree-sitter"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn empty_graph_state_is_shown() {
    let mut app = App::new();
    app.show_menu = true;
    app.menu_anim_progress = 1.0;
    app.menu_section = 5;
    app.code_graph_loading = false;
    app.code_graph = None;
    let out = render_app(&mut app, 120, 40);
    assert!(
        out.contains("No code graph data available"),
        "empty state missing:\n{}",
        out
    );
    assert!(out.contains("R to rebuild"), "rebuild hint missing");
}

#[test]
fn node_styles_by_kind() {
    use hercules_agent::code_graph::NodeKind;
    // Distinct semantic fg color per kind
    let f = App::code_graph_node_style(NodeKind::Function, false, false);
    let s = App::code_graph_node_style(NodeKind::Struct, false, false);
    let t = App::code_graph_node_style(NodeKind::Trait, false, false);
    let m = App::code_graph_node_style(NodeKind::Method, false, false);
    assert_ne!(f.fg, s.fg, "Function vs Struct colors must differ");
    assert_ne!(f.fg, t.fg, "Function vs Trait colors must differ");
    assert_ne!(m.fg, s.fg, "Method vs Struct colors must differ");

    // Kind prefixes are plain ASCII tags
    assert_eq!(App::node_kind_prefix(NodeKind::Function), "[F]");
    assert_eq!(App::node_kind_prefix(NodeKind::Method), "[M]");
    assert_eq!(App::node_kind_prefix(NodeKind::Struct), "[S]");
    assert_eq!(App::node_kind_prefix(NodeKind::Trait), "[T]");
    assert_eq!(App::node_kind_prefix(NodeKind::Impl), "[I]");

    // Selected: background-only highlight (no border modifier), keeps a bg
    let sel = App::code_graph_node_style(NodeKind::Function, true, false);
    assert!(
        sel.bg.is_some(),
        "selection must be highlighted via background"
    );
    assert!(
        sel.add_modifier.is_empty()
            || !sel
                .add_modifier
                .contains(ratatui::style::Modifier::REVERSED)
    );
    // Dimmed: muted color
    let dim = App::code_graph_node_style(NodeKind::Function, false, true);
    assert_ne!(dim.fg, f.fg, "dimmed style must differ from normal");
}

#[tokio::test(flavor = "multi_thread")]
async fn selected_node_bg_highlight_no_border_in_graph() {
    let mut app = App::new();
    app.show_menu = true;
    app.menu_anim_progress = 1.0;
    app.menu_section = 5;
    install_graph(&mut app, demo_graph());
    app.code_graph_selected = 0;

    let buf = buffer(&mut app, 140, 44);
    let selection_bg = Color::Rgb(129, 161, 193);
    let mut selected_cells = 0;
    for y in 0..buf.area.height {
        for x in 0..buf.area.width {
            if buf[(x, y)].bg == selection_bg {
                selected_cells += 1;
            }
        }
    }
    assert!(
        selected_cells > 0,
        "selected node must be filled with the selection background"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn details_pane_shows_relationship_counts() {
    let mut app = App::new();
    app.show_menu = true;
    app.menu_anim_progress = 1.0;
    app.menu_section = 5;
    let graph = demo_graph();

    // select 'bbb' (callee) in the visible (sorted) node list
    let mut app_tmp: Vec<String> = graph
        .nodes
        .iter()
        .filter(|n| n.kind != hercules_agent::code_graph::NodeKind::Module)
        .map(|n| n.name.clone())
        .collect();
    app_tmp.sort();
    let sel = app_tmp.iter().position(|n| n == "bbb").unwrap();

    install_graph(&mut app, graph);
    app.code_graph_selected = sel;

    let out = render_app(&mut app, 150, 46);
    assert!(out.contains("Details"), "details pane missing");
    assert!(
        out.contains("Called by"),
        "relationship 'Called by' missing:\n{}",
        out
    );
    assert!(
        out.contains("Calls (out)"),
        "relationship 'Calls (out)' missing"
    );
    assert!(
        out.contains("References"),
        "relationship 'References' missing"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn tree_sitter_calls_edge_exists_and_no_defines() {
    // After the Defines audit: Calls edges come from call sites, no Defines duplication
    let graph = demo_graph();
    let aaa = graph.nodes.iter().find(|n| n.name == "aaa").unwrap();
    let bbb = graph.nodes.iter().find(|n| n.name == "bbb").unwrap();
    assert!(
        graph
            .edges
            .iter()
            .any(|e| e.kind == EdgeKind::Calls && e.from == aaa.id && e.to == bbb.id),
        "Tree-sitter must produce aaa -> bbb Calls edge"
    );
    assert!(
        !graph.edges.iter().any(|e| e.kind == EdgeKind::Defines),
        "no Defines edge may duplicate a Calls relationship"
    );
}

#[test]
fn layout_is_deterministic() {
    use ratatui::layout::Rect;
    let area = Rect::new(0, 0, 120, 30);
    let a = App::compute_code_graph_layout(10, area);
    let b = App::compute_code_graph_layout(10, area);
    assert_eq!(a, b, "same input must yield identical layout");
    assert_eq!(a.len(), 10);
    // distinct positions for distinct nodes
    let mut seen = std::collections::HashSet::new();
    for r in &a {
        assert!(seen.insert((r.x, r.y)), "nodes must not overlap positions");
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn page_keys_scroll_details_only_in_details_pane() {
    let mut app = App::new();
    app.show_menu = true;
    app.menu_anim_progress = 1.0;
    app.menu_section = 5;
    install_graph(&mut app, demo_graph());

    // Outside the Details pane, PgDn must not scroll details nor the conversation
    app.handle_key(key(KeyCode::PageDown)).await;
    assert_eq!(
        app.code_graph_detail_scroll, 0,
        "PgDn outside Details pane must not scroll details"
    );
    assert_eq!(
        app.scroll_offset, 0,
        "PgDn with Code Graph open must not scroll conversation"
    );

    // Inside the Details pane it scrolls the details content
    app.code_graph_pane = CodeGraphPane::Details;
    app.handle_key(key(KeyCode::PageDown)).await;
    assert_eq!(
        app.code_graph_detail_scroll, 5,
        "PgDn in Details pane must scroll details"
    );
    app.handle_key(key(KeyCode::PageUp)).await;
    assert_eq!(
        app.code_graph_detail_scroll, 0,
        "PgUp in Details pane must scroll back"
    );
}

#[test]
fn layout_never_overlaps_even_when_overflowing() {
    use ratatui::layout::Rect;
    let area = Rect::new(0, 0, 60, 15); // capacity: 2 cols x 2 rows = 4
    let capacity = App::code_graph_capacity(area);
    assert_eq!(capacity, 4, "capacity math must match layout cell size");

    // Far more nodes than capacity: no wrap-around, no duplicated positions
    let rects = App::compute_code_graph_layout(100, area);
    assert!(
        rects.len() <= capacity,
        "layout must not exceed grid capacity"
    );
    let mut seen = std::collections::HashSet::new();
    for r in &rects {
        assert!(
            seen.insert((r.x, r.y)),
            "node positioned on an occupied cell (overlap)"
        );
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn navigating_code_graph_while_loading_does_not_panic() {
    let mut app = App::new();
    app.show_menu = true;
    app.menu_anim_progress = 1.0;
    app.menu_section = 5;
    app.code_graph_loading = true;
    app.code_graph = None;
    app.code_graph_pane = CodeGraphPane::Nodes;

    // Graph absent while loading — navigation must be a no-op, never a panic
    app.handle_key(key(KeyCode::Up)).await;
    app.handle_key(key(KeyCode::Down)).await;
    app.handle_key(key(KeyCode::Home)).await;
    app.handle_key(key(KeyCode::End)).await;

    assert_eq!(app.code_graph_selected, 0);
}

#[tokio::test(flavor = "multi_thread")]
async fn menu_renders_immediately_on_open() {
    // Regression: menu visibility was gated on animation progress, so the menu
    // was invisible until the fade animation advanced past 0.01.
    let mut app = App::new();
    app.show_menu = true;
    app.menu_section = 5;
    app.menu_anim_progress = 0.0;
    app.code_graph_loading = true;
    let out = render_app(&mut app, 120, 40);
    assert!(
        out.contains("Code Graph"),
        "menu must render on the first frame regardless of animation progress:\n{}",
        out
    );
    assert!(
        out.contains("Loading project graph"),
        "loading state must be visible immediately"
    );
}
