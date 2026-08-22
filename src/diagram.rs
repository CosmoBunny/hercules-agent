use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone)]
pub struct DiagramRenderer;

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

    pub fn render_mermaid<'a>(body: &str, inner_w: usize) -> Vec<Line<'a>> {
        let lines: Vec<&str> = body
            .lines()
            .map(|l| l.trim())
            .filter(|l| !l.is_empty() && !l.starts_with("%%"))
            .collect();

        if lines.is_empty() {
            return vec![Line::from(Span::styled(
                " (Empty diagram body) ",
                Style::default().fg(Color::Rgb(140, 140, 140)),
            ))];
        }

        let header = lines.first().copied().unwrap_or("").to_lowercase();

        if header.starts_with("pie") {
            Self::render_pie(&lines, inner_w)
        } else if header.starts_with("sequencediagram") {
            Self::render_sequence(&lines, inner_w)
        } else {
            Self::render_flowchart(&lines, inner_w)
        }
    }

    fn render_pie<'a>(lines: &[&str], inner_w: usize) -> Vec<Line<'a>> {
        let mut out = Vec::new();
        let mut title = "Pie Chart".to_string();
        let mut entries = Vec::new();

        for line in lines {
            if line.to_lowercase().starts_with("pie title") {
                title = line[9..].trim().trim_matches('"').to_string();
            } else if line.to_lowercase().starts_with("pie") {
                if let Some(pos) = line.to_lowercase().find("title") {
                    title = line[pos + 5..].trim().trim_matches('"').to_string();
                }
            } else if let Some(colon) = line.find(':') {
                let label = line[..colon].trim().trim_matches('"').to_string();
                let val_str = line[colon + 1..].trim();
                if let Ok(val) = val_str.parse::<f64>() {
                    entries.push((label, val));
                }
            }
        }

        out.push(Line::from(vec![
            Span::styled(" [MERMAID PIE CHART] ", Style::default().fg(Color::Rgb(255, 180, 60)).add_modifier(Modifier::BOLD)),
            Span::styled(format!("\"{}\"", title), Style::default().fg(Color::White).add_modifier(Modifier::BOLD)),
        ]));
        out.push(Line::from("─".repeat(inner_w.min(60))));

        let total: f64 = entries.iter().map(|(_, v)| v).sum();
        if total <= 0.0 || entries.is_empty() {
            out.push(Line::from(" (No valid chart entries)"));
            return out;
        }

        let max_label_len = entries.iter().map(|(l, _)| l.len()).max().unwrap_or(10).max(8);
        let bar_max_w = inner_w.saturating_sub(max_label_len + 18).clamp(10, 40);

        let colors = [
            Color::Rgb(80, 220, 255),
            Color::Rgb(80, 255, 140),
            Color::Rgb(255, 200, 80),
            Color::Rgb(255, 120, 220),
            Color::Rgb(180, 140, 255),
            Color::Rgb(255, 100, 100),
        ];

        for (idx, (label, val)) in entries.iter().enumerate() {
            let pct = (val / total) * 100.0;
            let fill_len = ((pct / 100.0) * bar_max_w as f64).round() as usize;
            let empty_len = bar_max_w.saturating_sub(fill_len);

            let bar_color = colors[idx % colors.len()];

            let padded_label = format!("{:width$}", label, width = max_label_len);
            let bar_fill = "█".repeat(fill_len);
            let bar_empty = "░".repeat(empty_len);

            out.push(Line::from(vec![
                Span::styled(format!("  {} ", padded_label), Style::default().fg(Color::White)),
                Span::styled(format!("[{}", bar_fill), Style::default().fg(bar_color)),
                Span::styled(format!("{}]", bar_empty), Style::default().fg(Color::Rgb(80, 80, 80))),
                Span::styled(format!(" {:5.1}% ({})", pct, val), Style::default().fg(Color::Rgb(200, 200, 200))),
            ]));
        }

        out
    }

    fn render_sequence<'a>(lines: &[&str], inner_w: usize) -> Vec<Line<'a>> {
        let mut out = Vec::new();
        let mut participants = Vec::new();
        let mut messages = Vec::new();

        out.push(Line::from(vec![
            Span::styled(" [SEQUENCE DIAGRAM] ", Style::default().fg(Color::Rgb(80, 220, 255)).add_modifier(Modifier::BOLD)),
        ]));
        out.push(Line::from("─".repeat(inner_w.min(60))));

        for line in lines {
            if line.to_lowercase().starts_with("sequencediagram") || line.to_lowercase().starts_with("autonumber") {
                continue;
            }
            if line.to_lowercase().starts_with("participant") || line.to_lowercase().starts_with("actor") {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 2 {
                    let name = parts[1].trim_matches('"');
                    if !participants.contains(&name.to_string()) {
                        participants.push(name.to_string());
                    }
                }
                continue;
            }

            // Arrow parsing: Alice->>Bob: Hello or Alice-->Bob: Hi
            if let Some(colon) = line.find(':') {
                let arrow_part = line[..colon].trim();
                let msg = line[colon + 1..].trim();

                let arrow_op = if arrow_part.contains("->>") {
                    "->>"
                } else if arrow_part.contains("-->>") {
                    "-->>"
                } else if arrow_part.contains("->") {
                    "->"
                } else if arrow_part.contains("-->") {
                    "-->"
                } else {
                    ""
                };

                if !arrow_op.is_empty() {
                    let mut parts = arrow_part.split(arrow_op);
                    if let (Some(from), Some(to)) = (parts.next(), parts.next()) {
                        let f = from.trim().to_string();
                        let t = to.trim().to_string();
                        if !f.is_empty() && !t.is_empty() {
                            if !participants.contains(&f) {
                                participants.push(f.clone());
                            }
                            if !participants.contains(&t) {
                                participants.push(t.clone());
                            }
                            messages.push((f, t, msg.to_string(), arrow_op.contains('-')));
                        }
                    }
                }
            }
        }

        if participants.is_empty() {
            out.push(Line::from(" (No sequence participants found)"));
            return out;
        }

        // Render Lifeline headers
        let mut header_spans = Vec::new();
        header_spans.push(Span::raw("  "));
        for p in &participants {
            header_spans.push(Span::styled(
                format!(" ┌{:─^12}┐ ", p),
                Style::default().fg(Color::Rgb(255, 200, 80)).add_modifier(Modifier::BOLD),
            ));
        }
        out.push(Line::from(header_spans));

        // Lifeline vertical tracks
        for (f, t, msg, is_dotted) in &messages {
            let from_idx = participants.iter().position(|x| x == f).unwrap_or(0);
            let to_idx = participants.iter().position(|x| x == t).unwrap_or(0);

            let mut track_spans = Vec::new();
            track_spans.push(Span::raw("  "));

            let (min_i, max_i) = if from_idx < to_idx { (from_idx, to_idx) } else { (to_idx, from_idx) };
            let is_forward = from_idx < to_idx;

            for i in 0..participants.len() {
                if i < min_i || i > max_i {
                    track_spans.push(Span::styled("      │       ", Style::default().fg(Color::Rgb(100, 100, 100))));
                } else if i == from_idx {
                    if is_forward {
                        track_spans.push(Span::styled("      ├──────►", Style::default().fg(Color::Rgb(80, 255, 140)).add_modifier(Modifier::BOLD)));
                    } else {
                        track_spans.push(Span::styled("◄─────┤       ", Style::default().fg(Color::Rgb(80, 255, 140)).add_modifier(Modifier::BOLD)));
                    }
                } else if i == to_idx {
                    if is_forward {
                        track_spans.push(Span::styled("◄─────┤       ", Style::default().fg(Color::Rgb(80, 255, 140)).add_modifier(Modifier::BOLD)));
                    } else {
                        track_spans.push(Span::styled("      ├──────►", Style::default().fg(Color::Rgb(80, 255, 140)).add_modifier(Modifier::BOLD)));
                    }
                } else {
                    let line_style = if *is_dotted { "─ ─ ─ ─ ─ ─ ─ " } else { "──────────────" };
                    track_spans.push(Span::styled(line_style, Style::default().fg(Color::Rgb(80, 220, 255))));
                }
            }
            out.push(Line::from(track_spans));

            let mut msg_spans = Vec::new();
            msg_spans.push(Span::raw("  "));
            for i in 0..participants.len() {
                if i == min_i {
                    let trunc_msg = if msg.len() > 22 { &msg[..22] } else { msg };
                    msg_spans.push(Span::styled(
                        format!("   {:<22} ", trunc_msg),
                        Style::default().fg(Color::White).add_modifier(Modifier::ITALIC),
                    ));
                } else if i > min_i && i <= max_i {
                    // Filler space
                } else {
                    msg_spans.push(Span::styled("      │       ", Style::default().fg(Color::Rgb(100, 100, 100))));
                }
            }
            out.push(Line::from(msg_spans));
        }

        out
    }

    fn render_flowchart<'a>(lines: &[&str], inner_w: usize) -> Vec<Line<'a>> {
        let mut out = Vec::new();
        let mut nodes: HashMap<String, String> = HashMap::new();
        let mut edges: Vec<(String, String, String)> = Vec::new();

        out.push(Line::from(vec![
            Span::styled(" [FLOWCHART / GRAPH DIAGRAM] ", Style::default().fg(Color::Rgb(255, 140, 0)).add_modifier(Modifier::BOLD)),
        ]));
        out.push(Line::from("─".repeat(inner_w.min(60))));

        let insert_node = |nodes: &mut HashMap<String, String>, id: String, label: String| {
            if id.is_empty() { return; }
            if let Some(existing) = nodes.get(&id) {
                if existing != &id && label == id {
                    return;
                }
            }
            nodes.insert(id, label);
        };

        for line in lines {
            if line.to_lowercase().starts_with("graph") || line.to_lowercase().starts_with("flowchart") {
                continue;
            }

            let mut line_rest = *line;
            while let Some(_arrow_pos) = line_rest.find("-->").or_else(|| line_rest.find("->")) {
                let is_long = line_rest.contains("-->");
                let arrow_str = if is_long { "-->" } else { "->" };
                let parts: Vec<&str> = line_rest.splitn(2, arrow_str).collect();

                if parts.len() == 2 {
                    let left = parts[0].trim();
                    let right = parts[1].trim();

                    let (id_a, label_a) = Self::parse_node_spec(left);
                    let (id_b, label_b) = Self::parse_node_spec(right);

                    insert_node(&mut nodes, id_a.clone(), label_a);
                    insert_node(&mut nodes, id_b.clone(), label_b);

                    let mut edge_label = String::new();
                    let mut real_id_b = id_b.clone();
                    if right.starts_with('|') {
                        if let Some(pipe_end) = right[1..].find('|') {
                            edge_label = right[1..pipe_end + 1].trim().to_string();
                            let after_pipe = right[pipe_end + 2..].trim();
                            let (b_id, b_lbl) = Self::parse_node_spec(after_pipe);
                            if !b_id.is_empty() {
                                real_id_b = b_id.clone();
                                insert_node(&mut nodes, b_id, b_lbl);
                            }
                        }
                    }

                    if !id_a.is_empty() && !real_id_b.is_empty() {
                        edges.push((id_a, real_id_b, edge_label));
                    }
                    line_rest = right;
                } else {
                    break;
                }
            }

            if !line.contains("->") {
                let (id, label) = Self::parse_node_spec(line);
                insert_node(&mut nodes, id, label);
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

        let mut ordered_ids: Vec<String> = Vec::new();
        let targets: HashSet<String> = edges.iter().map(|(_, t, _)| t.clone()).collect();
        for id in nodes.keys() {
            if !targets.contains(id) {
                ordered_ids.push(id.clone());
            }
        }
        for id in nodes.keys() {
            if !ordered_ids.contains(id) {
                ordered_ids.push(id.clone());
            }
        }

        for (_idx, id) in ordered_ids.iter().enumerate() {
            let label = nodes.get(id).cloned().unwrap_or_else(|| id.clone());

            out.push(Line::from(vec![
                Span::styled(" ┌─", Style::default().fg(Color::Rgb(255, 180, 60))),
                Span::styled("─".repeat(label.len() + 4), Style::default().fg(Color::Rgb(255, 180, 60))),
                Span::styled("─┐", Style::default().fg(Color::Rgb(255, 180, 60))),
            ]));

            out.push(Line::from(vec![
                Span::styled(" │  ", Style::default().fg(Color::Rgb(255, 180, 60))),
                Span::styled(label, Style::default().fg(Color::White).add_modifier(Modifier::BOLD)),
                Span::styled("  │", Style::default().fg(Color::Rgb(255, 180, 60))),
            ]));

            out.push(Line::from(vec![
                Span::styled(" └─", Style::default().fg(Color::Rgb(255, 180, 60))),
                Span::styled("─".repeat(nodes.get(id).map(|l| l.len()).unwrap_or(id.len()) + 4), Style::default().fg(Color::Rgb(255, 180, 60))),
                Span::styled("─┘", Style::default().fg(Color::Rgb(255, 180, 60))),
            ]));

            let outgoing: Vec<&(String, String, String)> = edges.iter().filter(|(f, _, _)| f == id).collect();
            for (_out_idx, (_, to, elabel)) in outgoing.iter().enumerate() {
                let to_label = nodes.get(to).cloned().unwrap_or_else(|| to.clone());

                if !elabel.is_empty() {
                    out.push(Line::from(vec![
                        Span::styled("      │ ", Style::default().fg(Color::Rgb(120, 120, 120))),
                        Span::styled(format!("({})", elabel), Style::default().fg(Color::Rgb(80, 220, 255)).add_modifier(Modifier::ITALIC)),
                    ]));
                }
                out.push(Line::from(vec![
                    Span::styled("      ▼ ", Style::default().fg(Color::Rgb(80, 255, 140))),
                    Span::styled(format!("──► [{}]", to_label), Style::default().fg(Color::Rgb(200, 255, 200))),
                ]));
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
            .flat_map(|l| l.spans.iter().map(|s| s.content.as_ref()))
            .collect::<Vec<_>>()
            .join("\n")
    }

    #[test]
    fn test_render_mermaid_flowchart() {
        let body = "graph TD\n    A[Start] --> B[Process]\n    B --> C{Decision}";
        let lines = DiagramRenderer::render_to_lines("mermaid", body, 60);
        assert!(!lines.is_empty());
        let text = lines_to_text(&lines);
        println!("RENDERED TEXT:\n{}", text);
        assert!(text.contains("FLOWCHART"));
        assert!(text.contains("Start"));
        assert!(text.contains("Process"));
    }

    #[test]
    fn test_render_mermaid_sequence() {
        let body = "sequenceDiagram\n    participant Alice\n    participant Bob\n    Alice->>Bob: Hello";
        let lines = DiagramRenderer::render_to_lines("mermaid", body, 60);
        assert!(!lines.is_empty());
        let text = lines_to_text(&lines);
        assert!(text.contains("SEQUENCE DIAGRAM"));
        assert!(text.contains("Alice"));
        assert!(text.contains("Bob"));
    }

    #[test]
    fn test_render_mermaid_pie() {
        let body = "pie title Distribution\n    \"Cats\" : 40\n    \"Dogs\" : 60";
        let lines = DiagramRenderer::render_to_lines("mermaid", body, 60);
        assert!(!lines.is_empty());
        let text = lines_to_text(&lines);
        assert!(text.contains("PIE CHART"));
        assert!(text.contains("Cats"));
        assert!(text.contains("Dogs"));
    }
}
