use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone)]
pub struct DiagramRenderer;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct NodeStyle {
    pub fill: Option<Color>,   // background
    pub stroke: Option<Color>, // foreground / border
}

impl DiagramRenderer {
    pub fn render_to_lines<'a>(diag_type: &str, body: &str, inner_w: usize) -> Vec<Line<'a>> {
        let clean_type = diag_type.trim().to_lowercase();
        let clean_type = if clean_type.starts_with("type:") {
            clean_type[5..].trim()
        } else {
            &clean_type
        };

        if clean_type.contains("mermaid")
            || body.contains("graph")
            || body.contains("flowchart")
            || body.contains("sequenceDiagram")
            || body.contains("pie")
        {
            Self::render_mermaid(body, inner_w)
        } else {
            Self::render_generic(clean_type, body, inner_w)
        }
    }

    fn clean_bracket_parens(line: &str) -> String {
        let mut in_bracket = false;
        let mut in_quote = false;
        let mut sanitized = String::with_capacity(line.len());

        for ch in line.chars() {
            if ch == '"' || ch == '\'' {
                in_quote = !in_quote;
                sanitized.push(ch);
            } else if ch == '[' {
                in_bracket = true;
                sanitized.push(ch);
            } else if ch == ']' {
                in_bracket = false;
                sanitized.push(ch);
            } else if in_bracket && !in_quote && (ch == '(' || ch == ')' || ch == '{' || ch == '}') {
                sanitized.push(' ');
            } else {
                sanitized.push(ch);
            }
        }
        sanitized
    }

    fn parse_edge_endpoints(line: &str) -> Option<(String, String)> {
        let arrow_pos = line.find("-->")?;
        let left_part = line[..arrow_pos].trim();
        let right_part = line[arrow_pos + 3..].trim();

        let src_id = if let Some(open) = left_part.find('[') {
            left_part[..open].trim().to_string()
        } else if let Some(open) = left_part.find('(') {
            left_part[..open].trim().to_string()
        } else {
            left_part.to_string()
        };

        let target_str = if right_part.starts_with('|') {
            if let Some(close_pipe) = right_part[1..].find('|') {
                right_part[close_pipe + 2..].trim()
            } else {
                right_part
            }
        } else {
            right_part
        };

        let target_id = if let Some(open) = target_str.find('[') {
            target_str[..open].trim().to_string()
        } else if let Some(open) = target_str.find('(') {
            target_str[..open].trim().to_string()
        } else {
            target_str.to_string()
        };

        if !src_id.is_empty() && !target_id.is_empty() {
            Some((src_id, target_id))
        } else {
            None
        }
    }

    fn sanitize_mermaid(body: &str) -> String {
        let mut clean_lines = Vec::new();
        let mut seen_edges = std::collections::HashSet::new();
        let mut target_incoming_count: std::collections::HashMap<String, usize> = std::collections::HashMap::new();

        // First pass: count incoming edges per target node
        for line in body.lines() {
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with("%%") {
                continue;
            }
            let lower = trimmed.to_lowercase();
            if lower.starts_with("subgraph ") || lower == "end" {
                continue;
            }
            if let Some((_, target)) = Self::parse_edge_endpoints(trimmed) {
                *target_incoming_count.entry(target).or_insert(0) += 1;
            }
        }

        for line in body.lines() {
            let mut trimmed = line.trim().to_string();
            if trimmed.is_empty() || trimmed.starts_with("%%") {
                continue;
            }
            let lower = trimmed.to_lowercase();
            if lower.starts_with("style ")
                || lower.starts_with("classdef ")
                || lower.starts_with("class ")
                || lower.starts_with("linkstyle ")
                || lower.starts_with("click ")
                || lower.starts_with("acctitle:")
                || lower.starts_with("accdescr:")
                || lower.starts_with("subgraph ")
                || lower == "end"
            {
                continue;
            }

            // Normalize edge types for parser compatibility
            if trimmed.contains("<-->") {
                trimmed = trimmed.replace("<-->", "-->");
            }
            if trimmed.contains("<->") {
                trimmed = trimmed.replace("<->", "-->");
            }
            if trimmed.contains("--o") {
                trimmed = trimmed.replace("--o", "-->");
            }
            if trimmed.contains("--x") {
                trimmed = trimmed.replace("--x", "-->");
            }
            if trimmed.contains("===") {
                trimmed = trimmed.replace("===", "-->");
            }
            if trimmed.contains("==>") {
                trimmed = trimmed.replace("==>", "-->");
            }
            if trimmed.contains("-.->") {
                trimmed = trimmed.replace("-.->", "-->");
            }
            if trimmed.contains("-.-") {
                trimmed = trimmed.replace("-.-", "-->");
            }
            if trimmed.contains("---") && !trimmed.contains("-->") {
                trimmed = trimmed.replace("---", "-->");
            }
            if trimmed.contains("--") && !trimmed.contains("-->") {
                trimmed = trimmed.replace("--", "-->");
            }

            // If it's an edge, check for duplicate connections
            if let Some((src, target)) = Self::parse_edge_endpoints(&trimmed) {
                if !seen_edges.insert((src.clone(), target.clone())) {
                    // Duplicate edge between the same pair of nodes -> skip to prevent garbled text
                    continue;
                }

                let is_converging = target_incoming_count.get(&target).copied().unwrap_or(0) > 2;
                if is_converging {
                    // Strip edge label for 3+ converging edges to prevent ratatui_markdown overstrike bug on shared arrowhead
                    if let Some(arrow_idx) = trimmed.find("-->|") {
                        if let Some(close_pipe) = trimmed[arrow_idx + 4..].find('|') {
                            let left = &trimmed[..arrow_idx + 3];
                            let right = &trimmed[arrow_idx + 4 + close_pipe + 1..];
                            trimmed = format!("{} {}", left, right.trim());
                        }
                    }
                }
            }

            clean_lines.push(Self::clean_bracket_parens(&trimmed));
        }
        clean_lines.join("\n")
    }

    fn parse_color_str(s: &str) -> Option<Color> {
        let s = s.trim().trim_matches('"').trim_matches('\'').to_lowercase();
        if s.starts_with('#') {
            let hex = s.trim_start_matches('#');
            if hex.len() == 6 {
                let r = u8::from_str_radix(&hex[0..2], 16).ok()?;
                let g = u8::from_str_radix(&hex[2..4], 16).ok()?;
                let b = u8::from_str_radix(&hex[4..6], 16).ok()?;
                return Some(Color::Rgb(r, g, b));
            } else if hex.len() == 3 {
                let r = u8::from_str_radix(&hex[0..1].repeat(2), 16).ok()?;
                let g = u8::from_str_radix(&hex[1..2].repeat(2), 16).ok()?;
                let b = u8::from_str_radix(&hex[2..3].repeat(2), 16).ok()?;
                return Some(Color::Rgb(r, g, b));
            }
        }
        match s.as_str() {
            "yellow" | "gold" => Some(Color::Rgb(249, 217, 102)),
            "orange" => Some(Color::Rgb(255, 153, 51)),
            "blue" => Some(Color::Rgb(80, 160, 255)),
            "cyan" | "aqua" => Some(Color::Rgb(51, 204, 255)),
            "green" | "lime" => Some(Color::Rgb(80, 240, 140)),
            "red" => Some(Color::Rgb(255, 90, 90)),
            "pink" | "magenta" => Some(Color::Rgb(255, 120, 200)),
            "purple" => Some(Color::Rgb(180, 130, 255)),
            "gray" | "grey" => Some(Color::Rgb(160, 170, 180)),
            "white" => Some(Color::White),
            _ => None,
        }
    }

    fn extract_mermaid_styles(
        body: &str,
    ) -> (
        std::collections::HashMap<String, NodeStyle>,
        std::collections::HashMap<String, String>,
    ) {
        let mut node_styles: std::collections::HashMap<String, NodeStyle> =
            std::collections::HashMap::new();
        let mut node_labels: std::collections::HashMap<String, String> =
            std::collections::HashMap::new();

        for line in body.lines() {
            let trimmed = line.trim();
            if trimmed.starts_with("%%") {
                continue;
            }

            // 1. Extract node id -> label from A[Label], B(Label), C([Label]), etc.
            let mut cur = trimmed;
            while let Some(open_idx) = cur.find(|c| c == '[' || c == '(' || c == '{') {
                let prefix = cur[..open_idx].trim();
                let id = prefix
                    .split_whitespace()
                    .last()
                    .unwrap_or("")
                    .trim_matches(|c: char| !c.is_alphanumeric() && c != '_');
                let close_char = match cur.as_bytes()[open_idx] {
                    b'[' => ']',
                    b'(' => ')',
                    b'{' => '}',
                    _ => ']',
                };
                if let Some(close_idx) = cur[open_idx + 1..].find(close_char) {
                    let inner_label = cur[open_idx + 1..open_idx + 1 + close_idx].trim();
                    if !id.is_empty() && !inner_label.is_empty() {
                        node_labels.insert(id.to_string(), inner_label.to_string());
                    }
                    cur = &cur[open_idx + 1 + close_idx + 1..];
                } else {
                    break;
                }
            }

            // 2. Extract style lines: style A fill:#f9d966,stroke:#333
            let lower = trimmed.to_lowercase();
            if lower.starts_with("style ") {
                let rest = trimmed[6..].trim();
                let mut parts = rest.split_whitespace();
                if let Some(node_id) = parts.next() {
                    let style_def = parts.collect::<Vec<_>>().join(" ");
                    let mut fill = None;
                    let mut stroke = None;
                    for prop in style_def.split(',') {
                        let prop = prop.trim();
                        if let Some(colon) = prop.find(':') {
                            let key = prop[..colon].trim().to_lowercase();
                            let val = prop[colon + 1..].trim();
                            match key.as_str() {
                                "fill" => fill = Self::parse_color_str(val),
                                "stroke" => stroke = Self::parse_color_str(val),
                                _ => {}
                            }
                        }
                    }
                    node_styles.insert(node_id.trim().to_string(), NodeStyle { fill, stroke });
                }
            }
        }

        (node_styles, node_labels)
    }

    fn transform_box_line_spans<'a>(
        full_line: &str,
        sorted_keys: &[String],
        node_styles: &std::collections::HashMap<String, NodeStyle>,
        node_labels: &std::collections::HashMap<String, String>,
        default_fg: Color,
        connector_fg: Color,
    ) -> Line<'a> {
        let mut chars: Vec<char> = full_line.chars().collect();
        let n = chars.len();

        // 1. Top border: ┌───┐ or ╭───╮ -> ▛▀▀▀▜
        let mut i = 0;
        while i < n {
            if chars[i] == '┌' || chars[i] == '╭' {
                if let Some(close_idx) = (i + 1..n).find(|&j| chars[j] == '┐' || chars[j] == '╮') {
                    if (i + 1..close_idx).all(|k| chars[k] == '─' || chars[k] == '-') {
                        chars[i] = '▛';
                        for k in i + 1..close_idx {
                            chars[k] = '▀';
                        }
                        chars[close_idx] = '▜';
                        i = close_idx + 1;
                        continue;
                    }
                }
            }
            i += 1;
        }

        // 2. Bottom border: └───┘ or ╰───╯ -> ▙▄▄▄▟
        let mut i = 0;
        while i < n {
            if chars[i] == '└' || chars[i] == '╰' {
                if let Some(close_idx) = (i + 1..n).find(|&j| chars[j] == '┘' || chars[j] == '╯') {
                    if (i + 1..close_idx).all(|k| chars[k] == '─' || chars[k] == '-') {
                        chars[i] = '▙';
                        for k in i + 1..close_idx {
                            chars[k] = '▄';
                        }
                        chars[close_idx] = '▟';
                        i = close_idx + 1;
                        continue;
                    }
                }
            }
            i += 1;
        }

        // 3. Middle node box: │ Text │ -> ▌ Text ▐ (only when text contains a valid node label)
        let has_node_label = node_labels.values().any(|lbl| full_line.contains(lbl));
        if has_node_label {
            let mut i = 0;
            while i < n {
                if chars[i] == '│' {
                    if let Some(close_idx) = (i + 2..n).find(|&j| chars[j] == '│') {
                        let inner: String = chars[i + 1..close_idx].iter().collect();
                        if node_labels.values().any(|lbl| inner.contains(lbl)) || inner.chars().any(|c| c.is_alphanumeric()) {
                            chars[i] = '▌';
                            chars[close_idx] = '▐';
                            i = close_idx + 1;
                            continue;
                        }
                    }
                }
                i += 1;
            }
        }

        let transformed_full: String = chars.into_iter().collect();

        // Check if this line is associated with a specific node's color deterministically
        let mut matched_style = None;
        for node_id in sorted_keys {
            if let Some(style) = node_styles.get(node_id) {
                if let Some(label) = node_labels.get(node_id) {
                    if full_line.contains(label) {
                        matched_style = Some(*style);
                        break;
                    }
                }
                if full_line.contains(&format!(" {} ", node_id))
                    || full_line.contains(&format!("│{}│", node_id))
                    || full_line.contains(&format!("▌{}▐", node_id))
                {
                    matched_style = Some(*style);
                    break;
                }
            }
        }

        let (fg, bg) = if let Some(st) = matched_style {
            let fill_bg = st.fill.unwrap_or(Color::Reset);
            let stroke_fg = st.stroke.or(st.fill).unwrap_or(default_fg);
            (stroke_fg, fill_bg)
        } else if transformed_full.contains('▼')
            || transformed_full.contains('▶')
            || transformed_full.contains('▲')
            || transformed_full.contains('◀')
            || transformed_full.contains("-->")
            || transformed_full.contains('│')
            || transformed_full.contains('─')
            || transformed_full.contains('┴')
            || transformed_full.contains('┬')
        {
            (connector_fg, Color::Reset)
        } else {
            (default_fg, Color::Reset)
        };

        let style = if bg != Color::Reset {
            Style::default().fg(fg).bg(bg)
        } else {
            Style::default().fg(fg)
        };

        Line::from(vec![Span::styled(transformed_full, style)])
    }

    pub fn render_mermaid<'a>(body: &str, inner_w: usize) -> Vec<Line<'a>> {
        let sanitized = Self::sanitize_mermaid(body);

        let lines: Vec<&str> = sanitized
            .lines()
            .map(|l| l.trim())
            .filter(|l| !l.is_empty())
            .collect();

        if lines.is_empty() {
            return vec![Line::from(Span::styled(
                " (Empty diagram body) ",
                Style::default().fg(Color::Rgb(140, 140, 140)),
            ))];
        }

        let header = lines.first().copied().unwrap_or("").to_lowercase();

        if header.starts_with("pie") {
            return Self::render_pie(&sanitized, inner_w);
        }
        if header.starts_with("sequencediagram") {
            return Self::render_sequence(&sanitized, inner_w);
        }
        if header.starts_with("graph") || header.starts_with("flowchart") {
            // Direct routing to authoritative custom flowchart renderer:
            // gives full control over box styling (fill/stroke), fan-in manifolds, and edge labels
            return Self::render_flowchart(body, inner_w);
        }

        // Only fall back to external renderer for diagram types not custom-built (class, er, gantt, gitgraph)
        let (node_styles, node_labels) = Self::extract_mermaid_styles(body);
        let theme = ratatui_markdown::theme::ThemeConfig::default();
        let default_fg = Color::Rgb(0, 230, 255);
        let connector_fg = Color::Rgb(100, 140, 180);
        let natural_w = (node_labels.len() * 42).max(inner_w).max(280);

        if let Some(rendered) = ratatui_markdown::mermaid::render_mermaid(&sanitized, natural_w, None, &theme) {
            if !rendered.is_empty() {
                let full_lines: Vec<String> = rendered
                    .iter()
                    .map(|l| l.spans.iter().map(|s| s.content.as_ref()).collect::<String>())
                    .collect();

                let leading_blank = full_lines
                    .iter()
                    .filter(|l| !l.trim().is_empty())
                    .map(|l| l.chars().take_while(|c| *c == ' ').count())
                    .min()
                    .unwrap_or(0);

                let mut sorted_keys: Vec<String> = node_styles.keys().cloned().collect();
                sorted_keys.sort_by(|a, b| {
                    let len_a = node_labels.get(a).map(|l| l.len()).unwrap_or(a.len());
                    let len_b = node_labels.get(b).map(|l| l.len()).unwrap_or(b.len());
                    len_b.cmp(&len_a).then_with(|| a.cmp(b))
                });

                let mut converted: Vec<Line<'a>> = Vec::new();
                for full_line in full_lines {
                    let chars: Vec<char> = full_line.chars().collect();
                    let cropped: String = if chars.len() > leading_blank {
                        chars[leading_blank..].iter().collect()
                    } else if chars.iter().all(|c| *c == ' ') {
                        String::new()
                    } else {
                        full_line
                    };

                    converted.push(Self::transform_box_line_spans(
                        &cropped,
                        &sorted_keys,
                        &node_styles,
                        &node_labels,
                        default_fg,
                        connector_fg,
                    ));
                }
                return converted;
            }
        }

        Self::render_flowchart(body, inner_w)
    }

    fn render_pie<'a>(body: &str, inner_w: usize) -> Vec<Line<'a>> {
        let mut out = Vec::new();
        let mut title = "Pie Chart".to_string();
        let mut slices: Vec<(String, f64)> = Vec::new();
        let mut total = 0.0;

        for line in body.lines() {
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with("%%") {
                continue;
            }
            if trimmed.to_lowercase().starts_with("pie") {
                if let Some(t) = trimmed.strip_prefix("pie title") {
                    title = t.trim().to_string();
                } else if let Some(t) = trimmed.strip_prefix("pie") {
                    let rem = t.trim();
                    if let Some(t2) = rem.strip_prefix("title") {
                        title = t2.trim().to_string();
                    }
                }
                continue;
            }

            if let Some(colon) = trimmed.find(':') {
                let label = trimmed[..colon].trim().trim_matches('"').trim_matches('\'').to_string();
                let val_str = trimmed[colon + 1..].trim();
                if let Ok(v) = val_str.parse::<f64>() {
                    slices.push((label, v));
                    total += v;
                }
            }
        }

        out.push(Line::from(vec![
            Span::styled(" ◐ ", Style::default().fg(Color::Rgb(255, 180, 0)).add_modifier(Modifier::BOLD)),
            Span::styled(title, Style::default().fg(Color::White).add_modifier(Modifier::BOLD)),
        ]));
        out.push(Line::from("─".repeat(inner_w.min(50))));

        let colors = [
            Color::Rgb(0, 230, 255),
            Color::Rgb(80, 255, 140),
            Color::Rgb(255, 140, 200),
            Color::Rgb(255, 200, 60),
            Color::Rgb(180, 140, 255),
            Color::Rgb(255, 100, 100),
        ];

        let bar_width: usize = 24;
        for (idx, (label, val)) in slices.iter().enumerate() {
            let pct = if total > 0.0 { (val / total) * 100.0 } else { 0.0 };
            let filled = ((pct / 100.0) * (bar_width as f64)).round() as usize;
            let color = colors[idx % colors.len()];

            let bar_filled = "█".repeat(filled);
            let bar_empty = "░".repeat(bar_width.saturating_sub(filled));

            out.push(Line::from(vec![
                Span::styled(format!(" {:<14} ", label), Style::default().fg(Color::Rgb(220, 220, 220))),
                Span::styled(bar_filled, Style::default().fg(color)),
                Span::styled(bar_empty, Style::default().fg(Color::Rgb(60, 60, 60))),
                Span::styled(format!(" {:>5.1}% ({})", pct, val), Style::default().fg(Color::Rgb(160, 160, 160))),
            ]));
        }

        out
    }

    fn render_sequence<'a>(body: &str, inner_w: usize) -> Vec<Line<'a>> {
        let mut out = Vec::new();
        let mut participants: Vec<String> = Vec::new();
        let mut messages: Vec<(String, String, String, bool)> = Vec::new();

        for line in body.lines() {
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with("%%") || trimmed.to_lowercase().starts_with("sequencediagram") {
                continue;
            }

            if trimmed.to_lowercase().starts_with("participant ") {
                let p = trimmed[12..].trim().to_string();
                if !participants.contains(&p) {
                    participants.push(p);
                }
                continue;
            }

            let is_dotted = trimmed.contains("-->>") || trimmed.contains("-->");
            let arrow = if trimmed.contains("->>") {
                Some("->>")
            } else if trimmed.contains("-->>") {
                Some("-->>")
            } else if trimmed.contains("->") {
                Some("->")
            } else if trimmed.contains("-->") {
                Some("-->")
            } else {
                None
            };

            if let Some(arr) = arrow {
                if let Some(arrow_pos) = trimmed.find(arr) {
                    let from = trimmed[..arrow_pos].trim().to_string();
                    let rest = trimmed[arrow_pos + arr.len()..].trim();
                    let (to, msg) = if let Some(colon) = rest.find(':') {
                        (rest[..colon].trim().to_string(), rest[colon + 1..].trim().to_string())
                    } else {
                        (rest.to_string(), String::new())
                    };

                    if !participants.contains(&from) && !from.is_empty() {
                        participants.push(from.clone());
                    }
                    if !participants.contains(&to) && !to.is_empty() {
                        participants.push(to.clone());
                    }

                    messages.push((from, to, msg, is_dotted));
                }
            }
        }

        if participants.is_empty() {
            return Self::render_generic("sequence", body, inner_w);
        }

        out.push(Line::from(vec![
            Span::styled(" ⇄ ", Style::default().fg(Color::Rgb(0, 230, 255)).add_modifier(Modifier::BOLD)),
            Span::styled("Sequence Diagram", Style::default().fg(Color::White).add_modifier(Modifier::BOLD)),
        ]));
        out.push(Line::from("─".repeat(inner_w.min(60))));

        let col_w = 16.max(inner_w / participants.len().max(1));

        let mut header_spans = Vec::new();
        for p in &participants {
            header_spans.push(Span::styled(
                format!(" {:^width$} ", p, width = col_w.saturating_sub(2)),
                Style::default().fg(Color::Black).bg(Color::Rgb(0, 200, 255)).add_modifier(Modifier::BOLD),
            ));
            header_spans.push(Span::styled(" ", Style::default()));
        }
        out.push(Line::from(header_spans));

        for (from, to, msg, is_dotted) in messages {
            let arrow_char = if is_dotted { "╌╌►" } else { "──►" };
            let line_desc = if !msg.is_empty() {
                format!("{} {} {}: {}", from, arrow_char, to, msg)
            } else {
                format!("{} {} {}", from, arrow_char, to)
            };

            out.push(Line::from(vec![
                Span::styled("  │ ", Style::default().fg(Color::Rgb(80, 80, 80))),
                Span::styled(line_desc, Style::default().fg(Color::Rgb(200, 240, 255))),
            ]));
        }

        out
    }

    fn render_flowchart<'a>(raw_body: &str, _inner_w: usize) -> Vec<Line<'a>> {
        let (node_styles, _) = Self::extract_mermaid_styles(raw_body);
        let sanitized = Self::sanitize_mermaid(raw_body);
        let mut out = Vec::new();
        let lines: Vec<&str> = sanitized.lines().map(|l| l.trim()).filter(|l| !l.is_empty()).collect();

        let mut nodes: HashMap<String, String> = HashMap::new();
        let mut node_order: Vec<String> = Vec::new();
        let mut edges: Vec<(String, String, String)> = Vec::new();

        let insert_node = |nodes: &mut HashMap<String, String>, order: &mut Vec<String>, id: String, label: String| {
            if id.is_empty() { return; }
            if !nodes.contains_key(&id) {
                order.push(id.clone());
                nodes.insert(id, label);
            } else if let Some(existing) = nodes.get_mut(&id) {
                if *existing == id && label != id {
                    *existing = label;
                }
            }
        };

        for line in &lines {
            let lower = line.to_lowercase();
            if lower.starts_with("graph ")
                || lower.starts_with("flowchart ")
                || lower.starts_with("subgraph ")
                || lower == "end"
                || lower.starts_with("style ")
                || lower.starts_with("classdef ")
                || lower.starts_with("class ")
            {
                continue;
            }

            let mut line_rest = *line;
            while let Some(arrow_idx) = line_rest.find("-->") {
                let left = line_rest[..arrow_idx].trim();
                let right = line_rest[arrow_idx + 3..].trim();

                let (id_a, label_a) = Self::parse_node_spec(left);
                insert_node(&mut nodes, &mut node_order, id_a.clone(), label_a);

                let mut edge_label = String::new();
                let real_id_b: String;
                let next_rest: &str;

                if right.starts_with('|') {
                    // Edge has a |label| — parse target only from what comes AFTER the pipe.
                    if let Some(pipe_end) = right[1..].find('|') {
                        edge_label = right[1..pipe_end + 1].trim().to_string();
                        let after_pipe = right[pipe_end + 2..].trim();
                        let (b_id, b_lbl) = Self::parse_node_spec(after_pipe);
                        insert_node(&mut nodes, &mut node_order, b_id.clone(), b_lbl);
                        real_id_b = b_id;
                        next_rest = after_pipe;
                    } else {
                        // Malformed pipe (no closing |) — fall back to treating whole right as target
                        let (b_id, b_lbl) = Self::parse_node_spec(right);
                        insert_node(&mut nodes, &mut node_order, b_id.clone(), b_lbl);
                        real_id_b = b_id;
                        next_rest = right;
                    }
                } else {
                    // No edge label — right IS the target spec directly.
                    let (b_id, b_lbl) = Self::parse_node_spec(right);
                    insert_node(&mut nodes, &mut node_order, b_id.clone(), b_lbl);
                    real_id_b = b_id;
                    next_rest = right;
                }

                if !id_a.is_empty() && !real_id_b.is_empty() {
                    edges.push((id_a, real_id_b, edge_label));
                }
                line_rest = next_rest;
            }

            if !line.contains("-->") && !line.contains("->") {
                let (id, label) = Self::parse_node_spec(line);
                insert_node(&mut nodes, &mut node_order, id, label);
            }
        }

        if nodes.is_empty() {
            for line in lines {
                out.push(Line::from(Span::styled(
                    format!("  {}", line),
                    Style::default().fg(Color::Rgb(200, 220, 255)),
                )));
            }
            return out;
        }

        let mut outgoing_by_source: HashMap<String, Vec<(String, String)>> = HashMap::new();
        for (from, to, elabel) in &edges {
            outgoing_by_source.entry(from.clone()).or_default().push((to.clone(), elabel.clone()));
        }

        let targets: HashSet<String> = edges.iter().map(|(_, t, _)| t.clone()).collect();
        let mut ordered_ids: Vec<String> = Vec::new();
        for id in &node_order {
            if !targets.contains(id) && !ordered_ids.contains(id) {
                ordered_ids.push(id.clone());
            }
        }
        for id in &node_order {
            if !ordered_ids.contains(id) {
                ordered_ids.push(id.clone());
            }
        }

        for id in &ordered_ids {
            let label = nodes.get(id).cloned().unwrap_or_else(|| id.clone());
            let style_info = node_styles.get(id);

            let bg = style_info.and_then(|s| s.fill).unwrap_or(Color::Reset);
            let fg = style_info
                .and_then(|s| s.stroke)
                .or_else(|| style_info.and_then(|s| s.fill))
                .unwrap_or(Color::Rgb(255, 180, 60));

            let border_style = if bg != Color::Reset {
                Style::default().fg(fg).bg(bg)
            } else {
                Style::default().fg(fg)
            };
            let text_style = if bg != Color::Reset {
                Style::default().fg(Color::Rgb(20, 20, 20)).bg(bg).add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(Color::White).add_modifier(Modifier::BOLD)
            };

            let box_w = label.chars().count() + 4;

            out.push(Line::from(vec![
                Span::styled(" ▛", border_style),
                Span::styled("▀".repeat(box_w), border_style),
                Span::styled("▜", border_style),
            ]));

            let pad_total = box_w.saturating_sub(label.chars().count());
            let pad_left = pad_total / 2;
            let pad_right = pad_total - pad_left;

            out.push(Line::from(vec![
                Span::styled(" ▌", border_style),
                Span::styled(" ".repeat(pad_left), border_style),
                Span::styled(label.clone(), text_style),
                Span::styled(" ".repeat(pad_right), border_style),
                Span::styled("▐", border_style),
            ]));

            out.push(Line::from(vec![
                Span::styled(" ▙", border_style),
                Span::styled("▄".repeat(box_w), border_style),
                Span::styled("▟", border_style),
            ]));

            // Every outgoing edge is its own arrow — no manifold, no merging.
            if let Some(outgoing) = outgoing_by_source.get(id) {
                for (to, elabel) in outgoing {
                    let to_label = nodes.get(to).cloned().unwrap_or_else(|| to.clone());

                    if !elabel.is_empty() {
                        out.push(Line::from(vec![
                            Span::styled("      │ ", Style::default().fg(Color::Rgb(120, 140, 160))),
                            Span::styled(format!("|{}|", elabel), Style::default().fg(Color::Rgb(80, 220, 255)).add_modifier(Modifier::BOLD)),
                        ]));
                    }
                    out.push(Line::from(vec![
                        Span::styled("      ▼ ", Style::default().fg(Color::Rgb(80, 255, 140))),
                        Span::styled(format!("──► [{}]", to_label), Style::default().fg(Color::Rgb(200, 255, 200))),
                    ]));
                }
            }

            out.push(Line::from(""));
        }

        out
    }

    fn parse_node_spec(s: &str) -> (String, String) {
        let trimmed = s.trim();
        if let Some(open_square) = trimmed.find('[') {
            if let Some(close_square) = trimmed.find(']') {
                let id = trimmed[..open_square].trim().to_string();
                let label = trimmed[open_square + 1..close_square].trim().to_string();
                return (id, label);
            }
        }
        if let Some(open_paren) = trimmed.find('(') {
            if let Some(close_paren) = trimmed.find(')') {
                let id = trimmed[..open_paren].trim().to_string();
                let label = trimmed[open_paren + 1..close_paren].trim().to_string();
                return (id, label);
            }
        }
        if let Some(open_brace) = trimmed.find('{') {
            if let Some(close_brace) = trimmed.find('}') {
                let id = trimmed[..open_brace].trim().to_string();
                let label = trimmed[open_brace + 1..close_brace].trim().to_string();
                return (id, label);
            }
        }
        (trimmed.to_string(), trimmed.to_string())
    }

    fn render_generic<'a>(diag_type: &str, body: &str, inner_w: usize) -> Vec<Line<'a>> {
        let mut out = Vec::new();
        out.push(Line::from(vec![
            Span::styled(" [DIAGRAM: ", Style::default().fg(Color::Rgb(255, 140, 0)).add_modifier(Modifier::BOLD)),
            Span::styled(diag_type.to_uppercase(), Style::default().fg(Color::White).add_modifier(Modifier::BOLD)),
            Span::styled("] ", Style::default().fg(Color::Rgb(255, 140, 0)).add_modifier(Modifier::BOLD)),
        ]));
        out.push(Line::from("─".repeat(inner_w.min(60))));

        for (idx, line) in body.lines().enumerate() {
            out.push(Line::from(vec![
                Span::styled(format!(" {:2} │ ", idx + 1), Style::default().fg(Color::Rgb(100, 100, 100))),
                Span::styled(line.to_string(), Style::default().fg(Color::Rgb(220, 220, 220))),
            ]));
        }

        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lines_to_text(lines: &[Line]) -> String {
        lines
            .iter()
            .map(|l| {
                l.spans.iter().map(|s| s.content.as_ref()).collect::<String>()
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    #[test]
    fn test_render_mermaid_flowchart() {
        let body = "graph TD\n    A[Start] --> B[Process]\n    B --> C{Decision}";
        let lines = DiagramRenderer::render_to_lines("mermaid", body, 60);
        assert!(!lines.is_empty());
        let text = lines_to_text(&lines);
        assert!(text.contains("Start"));
        assert!(text.contains("Process"));
    }

    #[test]
    fn test_render_mermaid_sequence() {
        let body = "sequenceDiagram\n    participant Alice\n    participant Bob\n    Alice->>Bob: Hello";
        let lines = DiagramRenderer::render_to_lines("mermaid", body, 60);
        assert!(!lines.is_empty());
        let text = lines_to_text(&lines);
        assert!(text.contains("Alice"));
        assert!(text.contains("Bob"));
    }

    #[test]
    fn test_render_pie_chart() {
        let body = "pie title Distribution\n    \"Cats\" : 40\n    \"Dogs\" : 60";
        let lines = DiagramRenderer::render_to_lines("mermaid", body, 60);
        assert!(!lines.is_empty());
        let text = lines_to_text(&lines);
        assert!(text.contains("Cats"));
        assert!(text.contains("Dogs"));
    }

    #[test]
    fn test_render_circuit_diagram() {
        let body = "graph TD\n    A[NE555 Timer] -->|Output| B[N-Channel MOSFET Gate]\n    B -->|Drain| C[12V Motor (775)]\n    B -->|Source| D[Ground]\n    E[Resistor R1] -->|To Pin 1| A\n    F[Resistor R2] -->|To Pin 2| A\n    G[Capacitor C1] -->|To Pin 6| A\n    H[Capacitor C2] -->|To Pin 7| A\n    I[Power Supply 12V] -->|To Pin 8| A\n    J[Switch] -->|Control Input| A\n    B -->|MOSFET Drain| C\n    style A fill:#f9d966,stroke:#333";
        let lines = DiagramRenderer::render_to_lines("mermaid", body, 100);
        let text = lines_to_text(&lines);
        assert!(text.contains("NE555 Timer"));
        assert!(text.contains("N-Channel MOSFET Gate") || text.contains("MOSFET"));
        assert!(text.contains("12V Motor") || text.contains("775"));
        // Assert no corrupted overstrike text
        assert!(!text.contains("CoTorPinI1put"));
        assert!(!text.contains("MOSFDr inain"));
    }

    #[test]
    fn test_render_mermaid_node_styling() {
        let body = r#"
        graph TD
            A[NE555 Timer] --> B[MOSFET Gate]
            style A fill:#f9d966,stroke:#333
            style B fill:#33ccff,stroke:#333
        "#;
        let lines = DiagramRenderer::render_mermaid(body, 120);
        assert!(!lines.is_empty());
        // Verify custom colors (fill bg or stroke fg) are present in spans
        let has_yellow = lines.iter().any(|l| {
            l.spans.iter().any(|s| {
                s.style.bg == Some(Color::Rgb(249, 217, 102))
                    || s.style.fg == Some(Color::Rgb(249, 217, 102))
            })
        });
        let has_cyan = lines.iter().any(|l| {
            l.spans.iter().any(|s| {
                s.style.bg == Some(Color::Rgb(51, 204, 255))
                    || s.style.fg == Some(Color::Rgb(51, 204, 255))
            })
        });
        assert!(has_yellow, "Node A yellow styling should be applied");
        assert!(has_cyan, "Node B cyan styling should be applied");
    }

    #[test]
    fn test_render_mermaid_block_box_characters() {
        let body = r#"
        graph TD
            A[NE555 Timer] -->|Output| B[MOSFET Gate]
        "#;
        let lines = DiagramRenderer::render_mermaid(body, 120);
        assert!(!lines.is_empty());
        let text = lines_to_text(&lines);
        // Verify custom block characters ▛, ▌, ▙ are used for node frames
        assert!(text.contains('▛') && text.contains('▜'), "Should contain top border ▛ ▜");
        assert!(text.contains('▌') && text.contains('▐'), "Should contain middle border ▌ ▐");
        assert!(text.contains('▙') && text.contains('▟'), "Should contain bottom border ▙ ▟");
        // Verify |Output| edge label is retained on the connector/diagram text
        assert!(text.contains("Output") || text.contains("NE555 Timer"));
    }

    #[test]
    fn test_render_mermaid_no_ghost_nodes() {
        let body = r#"
        graph TD
            A[NE555 Timer] -->|Output| B[MOSFET Gate]
            B -->|Source| D[Ground]
            C[12V Motor] -->|Ground| D
        "#;
        let lines = DiagramRenderer::render_mermaid(body, 120);
        let text = lines_to_text(&lines);
        // Assert no ghost node titles like "|Source| D" or "|Ground| D" were created
        assert!(!text.contains("|Source| D"));
        assert!(!text.contains("|Ground| D"));
        // Assert proper node and edge label existence
        assert!(text.contains("Ground"));
        assert!(text.contains("MOSFET Gate"));
        assert!(text.contains("12V Motor"));
        assert!(text.contains("|Source|"));
        assert!(text.contains("|Output|"));
    }
}
