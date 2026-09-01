use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};

#[derive(Clone, Debug, PartialEq)]
pub struct InlineSpan {
    pub text: String,
    pub bold: bool,
    pub italic: bool,
    pub strikethrough: bool,
    pub code: bool,
    pub link_url: Option<String>,
}

/// Parses inline markdown formatting into a list of `InlineSpan` items.
/// Supports:
/// - Bold: `**text**` and `__text__`
/// - Italic: `*text*` and `_text_`
/// - Bold+Italic: `***text***` and `___text___`
/// - Strikethrough: `~~text~~`
/// - Inline code: `` `text` ``
/// - Links: `[text](url)`
pub fn parse_inline(text: &str, base_bold: bool, base_italic: bool) -> Vec<InlineSpan> {
    let mut spans = Vec::new();
    parse_inline_into(text, base_bold, base_italic, false, false, None, &mut spans);
    spans
}

fn parse_inline_into(
    text: &str,
    bold: bool,
    italic: bool,
    strikethrough: bool,
    code: bool,
    link_url: Option<String>,
    out: &mut Vec<InlineSpan>,
) {
    if text.is_empty() {
        return;
    }

    if code {
        out.push(InlineSpan {
            text: text.to_string(),
            bold,
            italic,
            strikethrough,
            code: true,
            link_url,
        });
        return;
    }

    let mut i = 0;
    let bytes = text.as_bytes();
    let mut plain_start = 0;

    while i < bytes.len() {
        // 1. Inline code: `code`
        if bytes[i] == b'`' {
            if let Some(end_rel) = text[i + 1..].find('`') {
                let end = i + 1 + end_rel;
                if plain_start < i {
                    out.push(InlineSpan {
                        text: text[plain_start..i].to_string(),
                        bold,
                        italic,
                        strikethrough,
                        code: false,
                        link_url: link_url.clone(),
                    });
                }
                let code_content = &text[i + 1..end];
                out.push(InlineSpan {
                    text: code_content.to_string(),
                    bold,
                    italic,
                    strikethrough,
                    code: true,
                    link_url: link_url.clone(),
                });
                i = end + 1;
                plain_start = i;
                continue;
            }
        }

        // 2. Bold italic: *** or ___
        if text[i..].starts_with("***") || text[i..].starts_with("___") {
            let marker = &text[i..i + 3];
            if let Some(end_rel) = text[i + 3..].find(marker) {
                let end = i + 3 + end_rel;
                if plain_start < i {
                    out.push(InlineSpan {
                        text: text[plain_start..i].to_string(),
                        bold,
                        italic,
                        strikethrough,
                        code: false,
                        link_url: link_url.clone(),
                    });
                }
                parse_inline_into(
                    &text[i + 3..end],
                    true,
                    true,
                    strikethrough,
                    false,
                    link_url.clone(),
                    out,
                );
                i = end + 3;
                plain_start = i;
                continue;
            }
        }

        // 3. Bold: ** or __
        if text[i..].starts_with("**") || text[i..].starts_with("__") {
            let marker = &text[i..i + 2];
            if let Some(end_rel) = text[i + 2..].find(marker) {
                let end = i + 2 + end_rel;
                if plain_start < i {
                    out.push(InlineSpan {
                        text: text[plain_start..i].to_string(),
                        bold,
                        italic,
                        strikethrough,
                        code: false,
                        link_url: link_url.clone(),
                    });
                }
                parse_inline_into(
                    &text[i + 2..end],
                    true,
                    italic,
                    strikethrough,
                    false,
                    link_url.clone(),
                    out,
                );
                i = end + 2;
                plain_start = i;
                continue;
            }
        }

        // 4. Strikethrough: ~~
        if text[i..].starts_with("~~") {
            if let Some(end_rel) = text[i + 2..].find("~~") {
                let end = i + 2 + end_rel;
                if plain_start < i {
                    out.push(InlineSpan {
                        text: text[plain_start..i].to_string(),
                        bold,
                        italic,
                        strikethrough,
                        code: false,
                        link_url: link_url.clone(),
                    });
                }
                parse_inline_into(
                    &text[i + 2..end],
                    bold,
                    italic,
                    true,
                    false,
                    link_url.clone(),
                    out,
                );
                i = end + 2;
                plain_start = i;
                continue;
            }
        }

        // 5. Italic: * or _
        if (bytes[i] == b'*' || bytes[i] == b'_') && (i == 0 || bytes[i - 1] != b'\\') {
            let marker = if bytes[i] == b'*' { "*" } else { "_" };
            if !text[i..].starts_with("**") && !text[i..].starts_with("__") {
                if let Some(end_rel) = text[i + 1..].find(marker) {
                    let end = i + 1 + end_rel;
                    if !text[end..].starts_with("**")
                        && !text[end..].starts_with("__")
                        && end > i + 1
                    {
                        if plain_start < i {
                            out.push(InlineSpan {
                                text: text[plain_start..i].to_string(),
                                bold,
                                italic,
                                strikethrough,
                                code: false,
                                link_url: link_url.clone(),
                            });
                        }
                        parse_inline_into(
                            &text[i + 1..end],
                            bold,
                            true,
                            strikethrough,
                            false,
                            link_url.clone(),
                            out,
                        );
                        i = end + 1;
                        plain_start = i;
                        continue;
                    }
                }
            }
        }

        // 6. Links: [label](url)
        if bytes[i] == b'[' {
            if let Some(close_b) = text[i + 1..].find(']') {
                let close_b_idx = i + 1 + close_b;
                if text[close_b_idx..].starts_with("](") {
                    let open_p_idx = close_b_idx + 1;
                    if let Some(close_p) = text[open_p_idx + 1..].find(')') {
                        let close_p_idx = open_p_idx + 1 + close_p;
                        if plain_start < i {
                            out.push(InlineSpan {
                                text: text[plain_start..i].to_string(),
                                bold,
                                italic,
                                strikethrough,
                                code: false,
                                link_url: link_url.clone(),
                            });
                        }
                        let link_label = &text[i + 1..close_b_idx];
                        let link_dest = &text[open_p_idx + 1..close_p_idx];
                        parse_inline_into(
                            link_label,
                            bold,
                            italic,
                            strikethrough,
                            false,
                            Some(link_dest.to_string()),
                            out,
                        );
                        i = close_p_idx + 1;
                        plain_start = i;
                        continue;
                    }
                }
            }
        }

        let ch = text[i..].chars().next().unwrap();
        i += ch.len_utf8();
    }

    if plain_start < text.len() {
        out.push(InlineSpan {
            text: text[plain_start..].to_string(),
            bold,
            italic,
            strikethrough,
            code,
            link_url,
        });
    }
}

/// Interpolates newly revealed streaming characters from bright neon green
/// towards their final markdown target color based on token arrival age.
/// `age` is the distance in characters from the newest streaming character (0 = newest character at cursor).
/// As `age` increases (0 -> 6), the color smoothly transitions from green to `target_color`.
fn stream_token_color(target_color: Color, age: usize, is_streaming: bool) -> Color {
    if !is_streaming || age >= 8 {
        return target_color;
    }
    let (tr, tg, tb) = match target_color {
        Color::Rgb(r, g, b) => (r as f64, g as f64, b as f64),
        Color::Green => (0.0, 255.0, 0.0),
        Color::Blue => (0.0, 150.0, 255.0),
        Color::Yellow => (255.0, 255.0, 0.0),
        Color::Red => (255.0, 50.0, 50.0),
        Color::Cyan => (0.0, 230.0, 255.0),
        _ => (240.0, 245.0, 255.0),
    };

    // Fresh token start color: vibrant neon green
    let (sr, sg, sb) = (0.0, 255.0, 120.0);

    // Smooth transition over 6 characters (matches typing speed)
    let t = (age as f64 / 6.0).clamp(0.0, 1.0);

    let r = (sr + (tr - sr) * t).round() as u8;
    let g = (sg + (tg - sg) * t).round() as u8;
    let b = (sb + (tb - sb) * t).round() as u8;

    Color::Rgb(r, g, b)
}

fn slice_line_horizontal<'a>(line: Line<'a>, start_x: usize, max_len: usize) -> Vec<Span<'a>> {
    let mut out = Vec::new();
    let mut cur_col = 0;
    let end_x = start_x + max_len;

    for span in line.spans {
        let span_len = span.content.chars().count();
        let span_end = cur_col + span_len;

        if span_end > start_x && cur_col < end_x {
            let slice_start = if cur_col < start_x { start_x - cur_col } else { 0 };
            let slice_end = if span_end > end_x { span_len - (span_end - end_x) } else { span_len };

            if slice_end > slice_start {
                let sliced_str: String = span.content.chars().skip(slice_start).take(slice_end - slice_start).collect();
                out.push(Span::styled(sliced_str, span.style));
            }
        }
        cur_col += span_len;
    }
    out
}

/// Renders full GitHub-flavored markdown text into ratatui `Line` elements with
/// streaming reveal, syntax styling, headers, bullets, code blocks, tables, and blockquotes.
pub fn render_markdown_to_lines<'a>(
    text: &str,
    available_output: usize,
    global_out_ch: &mut usize,
    is_generating: bool,
    is_last_message: bool,
    anim_tick: u64,
    theme_color: Color,
    dark_gray: Color,
    preview_blocks: &std::collections::HashSet<usize>,
    mut out_toggle_buttons: Option<&mut Vec<(usize, usize, u16, u16, u16, u16)>>,
    mut out_copy_buttons: Option<&mut Vec<(usize, usize, u16, u16, String)>>,
    mut out_scroll_buttons: Option<&mut Vec<(usize, usize, u16, u16, u16, u16, u16, u16, usize)>>,
    target_width: usize,
    block_anims: Option<&std::collections::HashMap<usize, (bool, std::time::Instant)>>,
    scroll_offsets: Option<&std::collections::HashMap<usize, usize>>,
) -> Vec<Line<'a>> {
    let mut lines_out: Vec<Line<'a>> = Vec::new();
    let total_lines = text.lines().count();
    let mut in_code_block = false;
    let mut code_block_index: usize = 0;
    let mut code_line_num: usize = 1;
    let mut current_code_lang = String::new();
    let mut current_code_body = String::new();
    let mut current_code_lines: Vec<Line<'a>> = Vec::new();
    let mut current_header_line_idx: usize = 0;
    let mut current_copy_start: u16 = 0;
    let mut current_copy_end: u16 = 0;
    let mut current_table_lines: Vec<&str> = Vec::new();
    let code_bg = Color::Rgb(36, 41, 51);
    let gutter_fg = Color::Rgb(94, 129, 172);
    let target_block_width: usize = target_width.saturating_sub(4).max(40);
    let max_code_chars_per_line: usize = target_block_width.saturating_sub(12).max(20);

    fn flush_table<'a>(
        lines_out: &mut Vec<Line<'a>>,
        table_lines: &[&str],
        _target_block_width: usize,
        _theme_color: Color,
        dark_gray: Color,
        available_output: usize,
        global_out_ch: &mut usize,
        is_generating: bool,
        is_last_message: bool,
    ) {
        if table_lines.is_empty() { return; }

        let mut rows: Vec<Vec<String>> = Vec::new();
        let mut sep_row_idx: Option<usize> = None;

        for (_idx, line) in table_lines.iter().enumerate() {
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
            if parts.is_empty() { continue; }

            let is_sep = parts.iter().all(|c| !c.trim().is_empty() && c.chars().all(|ch| ch == '-' || ch == ':' || ch == ' '));
            if is_sep {
                if sep_row_idx.is_none() {
                    sep_row_idx = Some(rows.len());
                }
                continue;
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

        let num_cols = rows.iter().map(|r| r.len()).max().unwrap_or(0);
        if num_cols == 0 { return; }

        let mut col_widths = vec![0usize; num_cols];
        for row in &rows {
            for (i, cell) in row.iter().enumerate() {
                let cell_clean = cell.replace("**", "").replace("`", "").replace("*", "").replace("~~", "");
                col_widths[i] = col_widths[i].max(cell_clean.chars().count());
            }
        }

        // Add padding: minimum 4 chars wide per column
        for w in &mut col_widths {
            *w = (*w).max(4);
        }

        let border_fg = dark_gray;

        // Top Border: ┌─────────┬─────────┐
        let mut top_spans = vec![
            Span::styled("  ", Style::default()),
            Span::styled("┌", Style::default().fg(border_fg)),
        ];
        for (i, w) in col_widths.iter().enumerate() {
            top_spans.push(Span::styled("─".repeat(*w + 2), Style::default().fg(border_fg)));
            if i + 1 < num_cols {
                top_spans.push(Span::styled("┬", Style::default().fg(border_fg)));
            } else {
                top_spans.push(Span::styled("┐", Style::default().fg(border_fg)));
            }
        }
        lines_out.push(Line::from(top_spans));

        // Render Data Rows & Separators
        for (r_idx, row) in rows.iter().enumerate() {
            let is_header = r_idx == 0 && sep_row_idx.is_some();
            let mut row_spans = vec![
                Span::styled("  ", Style::default()),
                Span::styled("│", Style::default().fg(border_fg)),
            ];

            for (c_idx, w) in col_widths.iter().enumerate() {
                let cell_text = row.get(c_idx).map(|s| s.as_str()).unwrap_or("");
                let cell_clean = cell_text.replace("**", "").replace("`", "").replace("*", "").replace("~~", "");
                let text_w = cell_clean.chars().count();
                let pad_spaces = w.saturating_sub(text_w);

                row_spans.push(Span::styled(" ", Style::default()));

                let inline = parse_inline(cell_text, is_header, false);
                for span in inline {
                    for ch in span.text.chars() {
                        if *global_out_ch >= available_output {
                            break;
                        }
                        let age = available_output.saturating_sub(*global_out_ch);
                        let is_streaming = is_generating && is_last_message;
                        let base_c = if span.code {
                            Color::Rgb(140, 220, 255)
                        } else if is_header || span.bold {
                            Color::Rgb(0, 230, 255)
                        } else {
                            Color::Rgb(220, 235, 250)
                        };
                        let mut style = Style::default().fg(stream_token_color(base_c, age, is_streaming));
                        if is_header || span.bold { style = style.add_modifier(Modifier::BOLD); }
                        if span.italic { style = style.add_modifier(Modifier::ITALIC); }
                        row_spans.push(Span::styled(ch.to_string(), style));
                        *global_out_ch += 1;
                    }
                }

                if pad_spaces > 0 {
                    row_spans.push(Span::styled(" ".repeat(pad_spaces), Style::default()));
                }
                row_spans.push(Span::styled(" │", Style::default().fg(border_fg)));
            }

            lines_out.push(Line::from(row_spans));

            // Mid separator: ├─────────┼─────────┤
            if is_header {
                let mut mid_spans = vec![
                    Span::styled("  ", Style::default()),
                    Span::styled("├", Style::default().fg(border_fg)),
                ];
                for (i, w) in col_widths.iter().enumerate() {
                    mid_spans.push(Span::styled("─".repeat(*w + 2), Style::default().fg(border_fg)));
                    if i + 1 < num_cols {
                        mid_spans.push(Span::styled("┼", Style::default().fg(border_fg)));
                    } else {
                        mid_spans.push(Span::styled("┤", Style::default().fg(border_fg)));
                    }
                }
                lines_out.push(Line::from(mid_spans));
            }
        }

        // Bottom Border: └─────────┴─────────┘
        let mut bot_spans = vec![
            Span::styled("  ", Style::default()),
            Span::styled("└", Style::default().fg(border_fg)),
        ];
        for (i, w) in col_widths.iter().enumerate() {
            bot_spans.push(Span::styled("─".repeat(*w + 2), Style::default().fg(border_fg)));
            if i + 1 < num_cols {
                bot_spans.push(Span::styled("┴", Style::default().fg(border_fg)));
            } else {
                bot_spans.push(Span::styled("┘", Style::default().fg(border_fg)));
            }
        }
        lines_out.push(Line::from(bot_spans));
        *global_out_ch += 2;
    }

    for (l_idx, raw_line) in text.lines().enumerate() {
        let is_last_line = l_idx + 1 == total_lines;
        let trimmed = raw_line.trim();

        // 1. Code Block Fence
        if trimmed.starts_with("```") || trimmed.starts_with("~~~") {
            in_code_block = !in_code_block;
            if in_code_block {
                code_line_num = 1;
                let lang = trimmed
                    .trim_start_matches('`')
                    .trim_start_matches('~')
                    .trim()
                    .to_string();
                current_code_lang = lang.to_lowercase();
                current_code_body.clear();
                current_code_lines.clear();
                let tag_prefix = if lang.is_empty() {
                    "  ┌── code ".to_string()
                } else {
                    format!("  ┌── {} ", lang)
                };

                let copy_btn = " Copy ";
                let copy_w = copy_btn.chars().count();
                let prefix_w = tag_prefix.chars().count();
                let fill_count = target_block_width.saturating_sub(prefix_w + copy_w + 2).max(2);
                let copy_start_col = (prefix_w + fill_count) as u16;
                let copy_end_col = copy_start_col + copy_w as u16;

                current_header_line_idx = lines_out.len();
                current_copy_start = copy_start_col;
                current_copy_end = copy_end_col;

                let border_style = Style::default().fg(theme_color).bg(code_bg).add_modifier(Modifier::BOLD);
                let copy_style = Style::default().fg(Color::Rgb(150, 200, 255)).bg(Color::Rgb(28, 44, 68)).add_modifier(Modifier::BOLD);

                let mut fence_spans = Vec::new();
                fence_spans.push(Span::styled(tag_prefix, border_style));
                fence_spans.push(Span::styled("─".repeat(fill_count), border_style));
                fence_spans.push(Span::styled(copy_btn, copy_style));
                fence_spans.push(Span::styled("─┐", border_style));

                lines_out.push(Line::from(fence_spans));
            } else {
                if let Some(ref mut copies) = out_copy_buttons {
                    copies.push((current_header_line_idx, code_block_index, current_copy_start, current_copy_end, current_code_body.clone()));
                }

                let is_preview = preview_blocks.contains(&code_block_index);
                let is_previewable = current_code_lang.contains("mermaid")
                    || current_code_lang == "markdown"
                    || current_code_lang == "md"
                    || current_code_lang == "html";

                let mut preview_lines: Vec<Line<'a>> = Vec::new();
                let mut max_diag_w: usize = 0;
                let visible_preview_w = target_block_width.saturating_sub(6).max(20);
                let scroll_x = scroll_offsets.and_then(|m| m.get(&code_block_index).copied()).unwrap_or(0);

                if is_previewable && !current_code_body.trim().is_empty() {
                    if current_code_lang.contains("mermaid") {
                        let natural_w = target_block_width.max(160);
                        let plines = crate::diagram::DiagramRenderer::render_mermaid(&current_code_body, natural_w);
                        max_diag_w = plines.iter().map(|l| l.width()).max().unwrap_or(0);
                        let max_scroll = max_diag_w.saturating_sub(visible_preview_w);
                        let active_scroll = scroll_x.min(max_scroll);

                        for pline in plines {
                            let sliced = slice_line_horizontal(pline, active_scroll, visible_preview_w);
                            let mut pspans = vec![Span::styled("  │ ", Style::default().fg(theme_color).bg(code_bg))];
                            let mut p_char_count = 4;
                            for s in sliced {
                                p_char_count += s.content.chars().count();
                                pspans.push(Span::styled(s.content, s.style.bg(code_bg)));
                            }
                            if p_char_count < target_block_width - 1 {
                                let pad_spaces = (target_block_width - 1) - p_char_count;
                                pspans.push(Span::styled(" ".repeat(pad_spaces), Style::default().bg(code_bg)));
                            }
                            pspans.push(Span::styled("│", Style::default().fg(theme_color).bg(code_bg)));
                            preview_lines.push(Line::from(pspans));
                        }
                    } else if current_code_lang == "markdown" || current_code_lang == "md" {
                        for md_l in current_code_body.lines() {
                            let inline = parse_inline(md_l, false, false);
                            let mut pspans = vec![Span::styled("  │ ", Style::default().fg(theme_color).bg(code_bg))];
                            let mut p_char_count = 4;
                            for s in inline {
                                p_char_count += s.text.chars().count();
                                let mut st = Style::default().fg(Color::Rgb(230, 240, 255)).bg(code_bg);
                                if s.bold { st = st.add_modifier(Modifier::BOLD); }
                                if s.italic { st = st.add_modifier(Modifier::ITALIC); }
                                pspans.push(Span::styled(s.text, st));
                            }
                            if p_char_count < target_block_width - 1 {
                                let pad_spaces = (target_block_width - 1) - p_char_count;
                                pspans.push(Span::styled(" ".repeat(pad_spaces), Style::default().bg(code_bg)));
                            }
                            pspans.push(Span::styled("│", Style::default().fg(theme_color).bg(code_bg)));
                            preview_lines.push(Line::from(pspans));
                        }
                    }
                }

                // Choose display lines (preview or normal code) with animated height interpolation
                let code_h = current_code_lines.len().max(1);
                let prev_h = preview_lines.len().max(1);

                // Trim trailing blank lines from code block
                while current_code_lines.len() > 1 {
                    let is_blank = current_code_lines.last().map(|l| {
                        let text: String = l.spans.iter().map(|sp| sp.content.as_ref()).collect();
                        let inner = text.trim_matches(|c: char| c == '│' || c == ' ' || c.is_ascii_digit());
                        inner.is_empty()
                    }).unwrap_or(false);
                    if is_blank {
                        current_code_lines.pop();
                    } else {
                        break;
                    }
                }

                let mut active_display_lines = if is_preview {
                    preview_lines
                } else {
                    current_code_lines.clone()
                };

                if let Some(anims) = block_anims {
                    if let Some(&(to_preview, start_time)) = anims.get(&code_block_index) {
                        let elapsed = start_time.elapsed().as_secs_f32();
                        if elapsed < 0.25 {
                            let t = (elapsed / 0.22).clamp(0.0, 1.0);
                            let ease = t * t * (3.0 - 2.0 * t);
                            let (from_h, to_h) = if to_preview {
                                (code_h, prev_h)
                            } else {
                                (prev_h, code_h)
                            };
                            let target_h = ((from_h as f32 * (1.0 - ease) + to_h as f32 * ease).round() as usize).max(1);
                            if active_display_lines.len() > target_h {
                                active_display_lines.truncate(target_h);
                            } else {
                                while active_display_lines.len() < target_h {
                                    let pad_w = target_block_width.saturating_sub(4);
                                    active_display_lines.push(Line::from(vec![
                                        Span::styled("  │", Style::default().fg(theme_color).bg(code_bg)),
                                        Span::styled(" ".repeat(pad_w), Style::default().bg(code_bg)),
                                        Span::styled("│", Style::default().fg(theme_color).bg(code_bg)),
                                    ]));
                                }
                            }
                        }
                    }
                }

                lines_out.extend(active_display_lines);

                // Render footer with Normal / Preview buttons and optional horizontal scroll bar
                if is_previewable {
                    let mut fence_spans = Vec::new();
                    let prefix = "  └──";
                    let normal_str = " Normal ";
                    let preview_str = " Preview ";
                    let normal_w = normal_str.chars().count();
                    let preview_w = preview_str.chars().count();
                    let space_between = 2;

                    let border_style = Style::default().fg(theme_color).bg(code_bg);
                    fence_spans.push(Span::styled(prefix, border_style));
                    let mut cur_col = prefix.chars().count() as u16;

                    let max_scroll = max_diag_w.saturating_sub(visible_preview_w);
                    if is_preview && max_scroll > 0 {
                        let left_btn = " ◄ ";
                        let track_total = 10;
                        let thumb_w = 3;
                        let active_scroll = scroll_x.min(max_scroll);
                        let thumb_pos = (active_scroll * (track_total - thumb_w)) / max_scroll.max(1);
                        let track_str = format!(" [{}{}{}] ", "─".repeat(thumb_pos), "═".repeat(thumb_w), "─".repeat((track_total - thumb_w).saturating_sub(thumb_pos)));
                        let right_btn = " ► ";

                        let left_s = cur_col;
                        let left_e = left_s + left_btn.chars().count() as u16;
                        cur_col = left_e;

                        let track_s = cur_col;
                        let track_e = track_s + track_str.chars().count() as u16;
                        cur_col = track_e;

                        let right_s = cur_col;
                        let right_e = right_s + right_btn.chars().count() as u16;
                        cur_col = right_e;

                        let btn_style = Style::default().fg(Color::Rgb(100, 220, 255)).bg(Color::Rgb(24, 38, 58)).add_modifier(Modifier::BOLD);
                        let track_style = Style::default().fg(Color::Rgb(140, 180, 220)).bg(Color::Rgb(16, 26, 42));

                        fence_spans.push(Span::styled(left_btn, btn_style));
                        fence_spans.push(Span::styled(track_str, track_style));
                        fence_spans.push(Span::styled(right_btn, btn_style));

                        if let Some(ref mut scrolls) = out_scroll_buttons {
                            scrolls.push((lines_out.len(), code_block_index, left_s, left_e, track_s, track_e, right_s, right_e, max_scroll));
                        }
                    }

                    let right_buttons_w = normal_w + space_between + preview_w + 2; // +2 for "─┘"
                    let used_w = cur_col as usize + right_buttons_w;
                    let fill_count = target_block_width.saturating_sub(used_w).max(2);

                    fence_spans.push(Span::styled("─".repeat(fill_count), border_style));
                    cur_col += fill_count as u16;

                    let normal_start_col = cur_col;
                    let normal_end_col = normal_start_col + normal_w as u16;

                    let (normal_style, preview_style) = if is_preview {
                        (
                            Style::default().fg(Color::Rgb(140, 165, 195)).bg(code_bg),
                            Style::default().fg(Color::Black).bg(Color::Rgb(0, 255, 120)).add_modifier(Modifier::BOLD),
                        )
                    } else {
                        (
                            Style::default().fg(Color::Black).bg(Color::Rgb(0, 255, 120)).add_modifier(Modifier::BOLD),
                            Style::default().fg(Color::Rgb(140, 165, 195)).bg(code_bg),
                        )
                    };

                    fence_spans.push(Span::styled(normal_str, normal_style));
                    fence_spans.push(Span::styled("  ", Style::default().bg(code_bg)));

                    let preview_start_col = normal_end_col + space_between as u16;
                    let preview_end_col = preview_start_col + preview_w as u16;

                    fence_spans.push(Span::styled(preview_str, preview_style));
                    fence_spans.push(Span::styled("─┘", border_style));

                    if let Some(ref mut toggles) = out_toggle_buttons {
                        toggles.push((lines_out.len(), code_block_index, normal_start_col, normal_end_col, preview_start_col, preview_end_col));
                    }

                    lines_out.push(Line::from(fence_spans));
                } else {
                    let fill_count = target_block_width.saturating_sub(5).max(5);
                    let fence_tag = format!("  └──{}┘", "─".repeat(fill_count));
                    let mut fence_spans = Vec::new();
                    for ch in fence_tag.chars() {
                        fence_spans.push(Span::styled(ch.to_string(), Style::default().fg(theme_color).bg(code_bg)));
                    }
                    lines_out.push(Line::from(fence_spans));
                }
                code_block_index += 1;
            }
            continue;
        }

        // 2. Inside Code Block
        if in_code_block {
            current_code_body.push_str(raw_line);
            current_code_body.push('\n');

            let code_chars: Vec<char> = raw_line.chars().collect();
            let chunks: Vec<Vec<char>> = if code_chars.is_empty() {
                vec![Vec::new()]
            } else {
                code_chars.chunks(max_code_chars_per_line).map(|c| c.to_vec()).collect()
            };

            let num_digits = (code_line_num.max(1).ilog10() as usize + 1).max(2);
            for (chunk_idx, chunk) in chunks.into_iter().enumerate() {
                let gutter_str = if chunk_idx == 0 {
                    format!(" {:>width$} │ ", code_line_num, width = num_digits)
                } else {
                    format!(" {:>width$} │ ", "", width = num_digits)
                };

                let mut line_spans = vec![
                    Span::styled("  │", Style::default().fg(theme_color).bg(code_bg)),
                    Span::styled(
                        gutter_str.clone(),
                        Style::default().fg(gutter_fg).bg(code_bg),
                    ),
                ];
                let mut chunk_char_count = 0;

                for ch in chunk {
                    if *global_out_ch >= available_output {
                        break;
                    }
                    let age = available_output.saturating_sub(*global_out_ch);
                    let is_streaming = is_generating && is_last_message;
                    let code_color = Color::Rgb(220, 230, 245);
                    let style = Style::default()
                        .fg(stream_token_color(code_color, age, is_streaming))
                        .bg(code_bg);
                    line_spans.push(Span::styled(ch.to_string(), style));
                    *global_out_ch += 1;
                    chunk_char_count += 1;
                }

                if is_last_message && is_generating && is_last_line {
                    let pulse = (anim_tick as f64 * 0.4).sin() * 0.5 + 0.5;
                    let g_val = (210.0 + 45.0 * pulse) as u8;
                    let b_val = (100.0 + 80.0 * pulse) as u8;
                    line_spans.push(Span::styled(
                        " █",
                        Style::default()
                            .fg(Color::Rgb(0, g_val, b_val))
                            .bg(code_bg)
                            .add_modifier(Modifier::BOLD),
                    ));
                    chunk_char_count += 2;
                }

                // Fill background to entire width and add right border '│'
                // '  │' prefix (3 chars) + gutter_str + chunk_char_count + right border '│' (1 char)
                let total_line_chars = 3 + gutter_str.chars().count() + chunk_char_count;
                if total_line_chars < target_block_width - 1 {
                    let pad_spaces = (target_block_width - 1) - total_line_chars;
                    line_spans.push(Span::styled(
                        " ".repeat(pad_spaces),
                        Style::default().bg(code_bg),
                    ));
                }
                line_spans.push(Span::styled(
                    "│",
                    Style::default().fg(theme_color).bg(code_bg),
                ));

                current_code_lines.push(Line::from(line_spans));
            }
            code_line_num += 1;
            *global_out_ch += 1;
            continue;
        }

        // 3. Horizontal Rule
        let is_hr = (trimmed.len() >= 3)
            && (trimmed.chars().all(|c| c == '-')
                || trimmed.chars().all(|c| c == '*')
                || trimmed.chars().all(|c| c == '_'));
        if is_hr {
            let hr_str = "  ────────────────────────────────────────";
            let mut hr_spans = Vec::new();
            for ch in hr_str.chars() {
                if *global_out_ch >= available_output {
                    break;
                }
                let age = available_output.saturating_sub(*global_out_ch);
                let is_streaming = is_generating && is_last_message;
                let style = Style::default().fg(stream_token_color(dark_gray, age, is_streaming));
                hr_spans.push(Span::styled(ch.to_string(), style));
                *global_out_ch += 1;
            }
            *global_out_ch += raw_line.chars().count().saturating_sub(hr_str.chars().count()) + 1;
            lines_out.push(Line::from(hr_spans));
            continue;
        }

        // 4. Markdown Table Formatting
        if trimmed.starts_with('|') && trimmed.contains('|') {
            current_table_lines.push(raw_line);
            continue;
        } else if !current_table_lines.is_empty() {
            flush_table(&mut lines_out, &current_table_lines, target_block_width, theme_color, dark_gray, available_output, global_out_ch, is_generating, is_last_message);
            current_table_lines.clear();
        }

        // 5. Headings: # h1 (green), ## (blue), ### (yellow), ####+ (red)
        let hash_count = trimmed.chars().take_while(|c| *c == '#').count();
        if hash_count > 0 && hash_count <= 6 && trimmed[hash_count..].starts_with(' ') {
            let level = hash_count;
            let h_color = match level {
                1 => Color::Rgb(0, 255, 0),     // green
                2 => Color::Rgb(0, 150, 255),   // blue
                3 => Color::Rgb(255, 255, 0),   // yellow
                _ => Color::Rgb(255, 50, 50),   // red
            };

            let header_text = trimmed[hash_count..].trim();
            let mut line_spans = vec![Span::styled("  ", Style::default())];

            let inline_spans = parse_inline(header_text, true, false);
            for span in inline_spans {
                let span_len = span.text.chars().count();
                if *global_out_ch >= available_output {
                    break;
                }
                let take_len = (available_output - *global_out_ch).min(span_len);
                let age = available_output.saturating_sub(*global_out_ch);
                let is_streaming = is_generating && is_last_message;
                let style = Style::default()
                    .fg(stream_token_color(h_color, age, is_streaming))
                    .add_modifier(Modifier::BOLD);
                let rendered_text: String = if take_len == span_len {
                    span.text
                } else {
                    span.text.chars().take(take_len).collect()
                };
                line_spans.push(Span::styled(rendered_text, style));
                *global_out_ch += take_len;
            }
            *global_out_ch += hash_count + 1; // Account for stripped '# '

            if is_last_message && is_generating && is_last_line {
                let pulse = (anim_tick as f64 * 0.4).sin() * 0.5 + 0.5;
                let g_val = (210.0 + 45.0 * pulse) as u8;
                let b_val = (100.0 + 80.0 * pulse) as u8;
                line_spans.push(Span::styled(
                    " █",
                    Style::default()
                        .fg(Color::Rgb(0, g_val, b_val))
                        .add_modifier(Modifier::BOLD),
                ));
            }

            lines_out.push(Line::from(line_spans));
            continue;
        }

        // 6. Blockquote: > quote
        if trimmed.starts_with('>') {
            let quote_text = trimmed.trim_start_matches('>').trim();
            let mut line_spans = vec![Span::styled(
                "  │ ",
                Style::default().fg(Color::Rgb(0, 200, 230)),
            )];

            let inline_spans = parse_inline(quote_text, false, true);
            for span in inline_spans {
                let span_len = span.text.chars().count();
                if *global_out_ch >= available_output {
                    break;
                }
                let take_len = (available_output - *global_out_ch).min(span_len);
                let age = available_output.saturating_sub(*global_out_ch);
                let is_streaming = is_generating && is_last_message;
                let base_c = if span.code {
                    Color::Rgb(255, 190, 100)
                } else {
                    Color::Rgb(180, 195, 210)
                };
                let mut style = Style::default()
                    .fg(stream_token_color(base_c, age, is_streaming))
                    .add_modifier(Modifier::ITALIC);
                if span.bold {
                    style = style.add_modifier(Modifier::BOLD);
                }
                let rendered_text: String = if take_len == span_len {
                    span.text
                } else {
                    span.text.chars().take(take_len).collect()
                };
                line_spans.push(Span::styled(rendered_text, style));
                *global_out_ch += take_len;
            }
            *global_out_ch += 2; // for '> '

            if is_last_message && is_generating && is_last_line {
                let pulse = (anim_tick as f64 * 0.4).sin() * 0.5 + 0.5;
                let g_val = (210.0 + 45.0 * pulse) as u8;
                let b_val = (100.0 + 80.0 * pulse) as u8;
                line_spans.push(Span::styled(
                    " █",
                    Style::default()
                        .fg(Color::Rgb(0, g_val, b_val))
                        .add_modifier(Modifier::BOLD),
                ));
            }

            lines_out.push(Line::from(line_spans));
            continue;
        }

        // 7. Checkboxes / Task lists: - [ ] todo, - [x] done
        let is_task_unchecked = trimmed.starts_with("- [ ] ")
            || trimmed.starts_with("* [ ] ")
            || trimmed.starts_with("+ [ ] ");
        let is_task_checked = trimmed.starts_with("- [x] ")
            || trimmed.starts_with("- [X] ")
            || trimmed.starts_with("* [x] ")
            || trimmed.starts_with("* [X] ")
            || trimmed.starts_with("+ [x] ")
            || trimmed.starts_with("+ [X] ");

        if is_task_unchecked || is_task_checked {
            let leading_spaces = raw_line.chars().take_while(|c| c.is_whitespace()).count();
            let indent_str = " ".repeat(leading_spaces + 2);
            let mut line_spans = Vec::new();

            line_spans.push(Span::styled(indent_str, Style::default()));

            if is_task_checked {
                line_spans.push(Span::styled(
                    "☑ ",
                    Style::default().fg(Color::Rgb(0, 255, 120)).add_modifier(Modifier::BOLD),
                ));
            } else {
                line_spans.push(Span::styled(
                    "☐ ",
                    Style::default().fg(Color::Rgb(170, 170, 170)),
                ));
            }

            let task_text = &trimmed[6..];
            let inline_spans = parse_inline(task_text, false, false);
            for span in inline_spans {
                let span_len = span.text.chars().count();
                if *global_out_ch >= available_output {
                    break;
                }
                let take_len = (available_output - *global_out_ch).min(span_len);
                let age = available_output.saturating_sub(*global_out_ch);
                let is_streaming = is_generating && is_last_message;
                let base_c = if span.code {
                    Color::Rgb(255, 190, 100)
                } else if span.link_url.is_some() {
                    Color::Rgb(0, 200, 255)
                } else if is_task_checked {
                    Color::Rgb(150, 160, 170)
                } else {
                    Color::Rgb(235, 240, 250)
                };
                let mut style = Style::default().fg(stream_token_color(base_c, age, is_streaming));
                if span.bold {
                    style = style.add_modifier(Modifier::BOLD);
                }
                if span.italic {
                    style = style.add_modifier(Modifier::ITALIC);
                }
                if span.strikethrough || is_task_checked {
                    style = style.add_modifier(Modifier::CROSSED_OUT);
                }
                if span.link_url.is_some() {
                    style = style.add_modifier(Modifier::UNDERLINED);
                }
                let rendered_text: String = if take_len == span_len {
                    span.text
                } else {
                    span.text.chars().take(take_len).collect()
                };
                line_spans.push(Span::styled(rendered_text, style));
                *global_out_ch += take_len;
            }
            *global_out_ch += 6; // for marker

            if is_last_message && is_generating && is_last_line {
                let pulse = (anim_tick as f64 * 0.4).sin() * 0.5 + 0.5;
                let g_val = (210.0 + 45.0 * pulse) as u8;
                let b_val = (100.0 + 80.0 * pulse) as u8;
                line_spans.push(Span::styled(
                    " █",
                    Style::default()
                        .fg(Color::Rgb(0, g_val, b_val))
                        .add_modifier(Modifier::BOLD),
                ));
            }

            lines_out.push(Line::from(line_spans));
            continue;
        }

        // 8. Bullets / Unordered Lists: * bullet -> ● bullet, - bullet, + bullet
        let is_bullet = trimmed.starts_with("- ")
            || trimmed.starts_with("* ")
            || trimmed.starts_with("+ ");

        // 9. Ordered Lists: 1. item, 2. item
        let is_ordered = !is_bullet
            && trimmed
                .chars()
                .next()
                .map(|c| c.is_ascii_digit())
                .unwrap_or(false)
            && trimmed.contains(". ")
            && trimmed
                .split(". ")
                .next()
                .unwrap_or("")
                .chars()
                .all(|c| c.is_ascii_digit());

        let mut line_spans = Vec::new();
        let leading_spaces = raw_line.chars().take_while(|c| c.is_whitespace()).count();

        let text_to_parse = if is_bullet {
            let bullet_char = match leading_spaces {
                0..=1 => "● ",
                2..=3 => "○ ",
                _ => "▪ ",
            };
            let indent_str = format!("  {}{}", " ".repeat(leading_spaces), bullet_char);
            line_spans.push(Span::styled(
                indent_str,
                Style::default().fg(theme_color).add_modifier(Modifier::BOLD),
            ));
            &trimmed[2..]
        } else if is_ordered {
            let dot_pos = trimmed.find(". ").unwrap();
            let num_str = &trimmed[..dot_pos + 2];
            let indent_str = format!("  {}{}", " ".repeat(leading_spaces), num_str);
            line_spans.push(Span::styled(
                indent_str,
                Style::default().fg(theme_color).add_modifier(Modifier::BOLD),
            ));
            &trimmed[dot_pos + 2..]
        } else {
            line_spans.push(Span::styled("  ", Style::default()));
            raw_line
        };

        let inline_spans = parse_inline(text_to_parse, false, false);
        for span in inline_spans {
            let span_len = span.text.chars().count();
            if *global_out_ch >= available_output {
                break;
            }
            let take_len = (available_output - *global_out_ch).min(span_len);
            let age = available_output.saturating_sub(*global_out_ch);
            let is_streaming = is_generating && is_last_message;

            let base_c = if span.code {
                Color::Rgb(255, 190, 100)
            } else if span.link_url.is_some() {
                Color::Rgb(0, 200, 255)
            } else {
                Color::Rgb(240, 245, 255)
            };

            let mut style = Style::default().fg(stream_token_color(base_c, age, is_streaming));
            if span.bold {
                style = style.add_modifier(Modifier::BOLD);
            }
            if span.italic {
                style = style.add_modifier(Modifier::ITALIC);
            }
            if span.strikethrough {
                style = style.add_modifier(Modifier::CROSSED_OUT);
            }
            if span.link_url.is_some() {
                style = style.add_modifier(Modifier::UNDERLINED);
            }

            let rendered_text: String = if take_len == span_len {
                span.text
            } else {
                span.text.chars().take(take_len).collect()
            };

            line_spans.push(Span::styled(rendered_text, style));
            *global_out_ch += take_len;
        }
        *global_out_ch += 1;

        if is_last_message && is_generating && is_last_line {
            let pulse = (anim_tick as f64 * 0.4).sin() * 0.5 + 0.5;
            let g_val = (210.0 + 45.0 * pulse) as u8;
            let b_val = (100.0 + 80.0 * pulse) as u8;
            line_spans.push(Span::styled(
                " █",
                Style::default()
                    .fg(Color::Rgb(0, g_val, b_val))
                    .add_modifier(Modifier::BOLD),
            ));
        }

        lines_out.push(Line::from(line_spans));
    }

    if !current_table_lines.is_empty() {
        flush_table(&mut lines_out, &current_table_lines, target_block_width, theme_color, dark_gray, available_output, global_out_ch, is_generating, is_last_message);
    }

    if in_code_block {
        while current_code_lines.len() > 1 {
            let is_blank = current_code_lines.last().map(|l| {
                let text: String = l.spans.iter().map(|sp| sp.content.as_ref()).collect();
                let inner = text.trim_matches(|c: char| c == '│' || c == ' ' || c.is_ascii_digit());
                inner.is_empty()
            }).unwrap_or(false);
            if is_blank {
                current_code_lines.pop();
            } else {
                break;
            }
        }

        if current_code_lines.is_empty() {
            let mut line_spans = vec![
                Span::styled("  │", Style::default().fg(theme_color).bg(code_bg)),
                Span::styled(format!(" {:2} │ ", 1), Style::default().fg(gutter_fg).bg(code_bg)),
            ];
            let mut cur_w = 4 + 6;
            if is_last_message && is_generating {
                let pulse = (anim_tick as f64 * 0.4).sin() * 0.5 + 0.5;
                let g_val = (210.0 + 45.0 * pulse) as u8;
                let b_val = (100.0 + 80.0 * pulse) as u8;
                line_spans.push(Span::styled(" █", Style::default().fg(Color::Rgb(0, g_val, b_val)).bg(code_bg).add_modifier(Modifier::BOLD)));
                cur_w += 2;
            }
            if cur_w < target_block_width.saturating_sub(1) {
                line_spans.push(Span::styled(" ".repeat((target_block_width.saturating_sub(1)) - cur_w), Style::default().bg(code_bg)));
            }
            line_spans.push(Span::styled("│", Style::default().fg(theme_color).bg(code_bg)));
            lines_out.push(Line::from(line_spans));
        } else {
            lines_out.extend(current_code_lines);
        }

        let fill_count = target_block_width.saturating_sub(5).max(5);
        let fence_tag = format!("  └──{}┘", "─".repeat(fill_count));
        let mut fence_spans = Vec::new();
        for ch in fence_tag.chars() {
            fence_spans.push(Span::styled(ch.to_string(), Style::default().fg(theme_color).bg(code_bg)));
        }
        lines_out.push(Line::from(fence_spans));
    }

    lines_out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_inline_bold() {
        let spans = parse_inline("**hello bold**", false, false);
        assert_eq!(spans.len(), 1);
        assert_eq!(spans[0].text, "hello bold");
        assert!(spans[0].bold);
        assert!(!spans[0].italic);
    }

    #[test]
    fn test_parse_inline_italic() {
        let spans = parse_inline("*hello italic*", false, false);
        assert_eq!(spans.len(), 1);
        assert_eq!(spans[0].text, "hello italic");
        assert!(!spans[0].bold);
        assert!(spans[0].italic);
    }

    #[test]
    fn test_parse_inline_bold_italic() {
        let spans = parse_inline("***bold and italic***", false, false);
        assert_eq!(spans.len(), 1);
        assert_eq!(spans[0].text, "bold and italic");
        assert!(spans[0].bold);
        assert!(spans[0].italic);
    }

    #[test]
    fn test_parse_inline_code() {
        let spans = parse_inline("`let x = 10;`", false, false);
        assert_eq!(spans.len(), 1);
        assert_eq!(spans[0].text, "let x = 10;");
        assert!(spans[0].code);
    }

    #[test]
    fn test_parse_inline_strikethrough() {
        let spans = parse_inline("~~deleted~~", false, false);
        assert_eq!(spans.len(), 1);
        assert_eq!(spans[0].text, "deleted");
        assert!(spans[0].strikethrough);
    }

    #[test]
    fn test_parse_inline_link() {
        let spans = parse_inline("[GitHub](https://github.com)", false, false);
        assert_eq!(spans.len(), 1);
        assert_eq!(spans[0].text, "GitHub");
        assert_eq!(spans[0].link_url.as_deref(), Some("https://github.com"));
    }

    #[test]
    fn test_render_headings_colors() {
        let mut ch_count = 0;
        let set = std::collections::HashSet::new();
        let lines = render_markdown_to_lines(
            "# Heading 1\n## Heading 2\n### Heading 3\n#### Heading 4",
            1000,
            &mut ch_count,
            false,
            false,
            0,
            Color::Rgb(0, 230, 255),
            Color::Rgb(80, 80, 80),
            &set,
            None,
            None,
            None,
            88,
            None,
            None,
        );
        assert_eq!(lines.len(), 4);
    }

    #[test]
    fn test_render_bullets() {
        let mut ch_count = 0;
        let set = std::collections::HashSet::new();
        let lines = render_markdown_to_lines(
            "* bullet item\n- another bullet",
            1000,
            &mut ch_count,
            false,
            false,
            0,
            Color::Rgb(0, 230, 255),
            Color::Rgb(80, 80, 80),
            &set,
            None,
            None,
            None,
            88,
            None,
            None,
        );
        assert_eq!(lines.len(), 2);
    }

    #[test]
    fn test_render_code_block_line_numbers() {
        let mut ch_count = 0;
        let set = std::collections::HashSet::new();
        let lines = render_markdown_to_lines(
            "```rust\nfn main() {\n    println!(\"hello\");\n}\n```",
            1000,
            &mut ch_count,
            false,
            false,
            0,
            Color::Rgb(0, 230, 255),
            Color::Rgb(80, 80, 80),
            &set,
            None,
            None,
            None,
            88,
            None,
            None,
        );
        // Header, 3 body lines, footer = 5 lines total
        assert_eq!(lines.len(), 5);
        assert!(lines[1].spans.iter().any(|s| s.content.contains(" 1 │ ")));
        assert!(lines[2].spans.iter().any(|s| s.content.contains(" 2 │ ")));
        assert!(lines[3].spans.iter().any(|s| s.content.contains(" 3 │ ")));
    }

    #[test]
    fn test_render_mermaid_preview() {
        let mut ch_count = 0;
        let set = std::collections::HashSet::new();
        let lines = render_markdown_to_lines(
            "```mermaid\ngraph TD\nA-->B\n```",
            1000,
            &mut ch_count,
            false,
            false,
            0,
            Color::Rgb(0, 230, 255),
            Color::Rgb(80, 80, 80),
            &set,
            None,
            None,
            None,
            88,
            None,
            None,
        );
        // Header, 2 code lines, footer with Normal / Preview buttons
        assert_eq!(lines.len(), 4);
        let has_normal_btn = lines.iter().any(|l| {
            let s: String = l.spans.iter().map(|sp| sp.content.as_ref()).collect();
            s.contains("Normal") && s.contains("Preview")
        });
        assert!(has_normal_btn);
    }

    #[test]
    fn test_render_code_block_wrapping() {
        let mut ch_count = 0;
        let set = std::collections::HashSet::new();
        let long_line = "a".repeat(150);
        let md = format!("```\n{}\n```", long_line);
        let lines = render_markdown_to_lines(
            &md,
            1000,
            &mut ch_count,
            false,
            false,
            0,
            Color::Rgb(0, 230, 255),
            Color::Rgb(80, 80, 80),
            &set,
            None,
            None,
            None,
            88,
            None,
            None,
        );
        // Header + wrapped lines (at least 2 visual lines for 150 chars) + footer >= 4 lines
        assert!(lines.len() >= 4);
        assert!(lines[1].spans.iter().any(|s| s.content.contains(" 1 │ ")));
        assert!(lines[2].spans.iter().any(|s| s.content.contains("   │ ")));
    }

    #[test]
    fn test_render_mermaid_preview_active() {
        let mut ch_count = 0;
        let mut set = std::collections::HashSet::new();
        set.insert(0); // Block 0 in preview mode
        let mut toggles = Vec::new();
        let lines = render_markdown_to_lines(
            "```mermaid\ngraph TD\nA-->B\n```",
            1000,
            &mut ch_count,
            false,
            false,
            0,
            Color::Rgb(0, 230, 255),
            Color::Rgb(80, 80, 80),
            &set,
            Some(&mut toggles),
            None,
            None,
            88,
            None,
            None,
        );
        println!("TOTAL LINES: {}", lines.len());
        for (i, l) in lines.iter().enumerate() {
            let s: String = l.spans.iter().map(|sp| sp.content.as_ref()).collect();
            println!("LINE {}: {:?}", i, s);
        }
        let has_diagram_text = lines.iter().any(|l| {
            let s: String = l.spans.iter().map(|sp| sp.content.as_ref()).collect();
            s.contains("FLOWCHART") || s.contains("GRAPH") || s.contains("A") || s.contains("┌")
        });
        assert!(has_diagram_text);
    }
}
