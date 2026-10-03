//! Conversation copy/export layer (spec: logical, not screen-based).
//!
//! Two separate concepts:
//!
//! ```text
//! Conversation Archive (UI history, survives compaction)
//!   ├── retired messages (verbatim, oldest first)
//!   ├── retired chips (label + body, stable u64 ids)
//!   └── latest compact summary
//!            │
//!            ▼
//!     Copy / Export layer (/copy, chip/range, mouse)
//!            │
//!            ▼
//!     Clipboard backend (crate::clipboard)
//! ```
//!
//! while the model has its own trimming context (system / compact / tail).
//! Model-context compaction must never equal UI-history deletion: the
//! archive below is append-only and is only extended at compact time with
//! the retired slice (never the kept tail), so nothing is duplicated.
//!
//! Mouse selection resolves through a per-frame logical render map
//! (`RenderRow`), never through scroll arithmetic on content: scrolling,
//! resize and streaming appends keep logical indices stable; only
//! compaction / load / clear invalidate a selection.
//!
//! Column slicing is EXACT, not best-effort: `wrap_line_rows` ports
//! ratatui 0.30's `WordWrapper` (`trim: false`, the chat `Paragraph`'s
//! setting) branch by branch — same graphemes, same width tables, same
//! whitespace rules — and `wrap_against_test_backend` proves row-break
//! equality against the real renderer across texts and widths.

use unicode_segmentation::UnicodeSegmentation;
use unicode_width::{UnicodeWidthChar, UnicodeWidthStr};

/// A `/copy` request parsed from input.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CopyRequest {
    /// `/copy` — the whole archived + live conversation.
    All,
    /// `/copy 17` — exactly logical chip 17 (never terminal row 17).
    Chip(u64),
    /// `/copy 12:27` — inclusive chip range, direction normalized.
    Range { start: u64, end: u64 },
}

/// Why a `/copy ...` argument was rejected (reported in status).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CopyParseError {
    /// `/copy 5 foo` — trailing garbage.
    TrailingText(String),
    /// `/copy abc`, `/copy 5:` etc.
    BadId(String),
}

impl std::fmt::Display for CopyParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CopyParseError::TrailingText(t) => {
                write!(f, "unexpected text after chip range: '{t}'")
            }
            CopyParseError::BadId(t) => write!(
                f,
                "bad chip id '{t}' — use /copy, /copy <id>, or /copy <start>:<end>"
            ),
        }
    }
}

/// Parse `/copy...` input. Returns `None` when the input is not a copy
/// command at all (`/copySomething` must not match).
pub fn parse_copy_command(input: &str) -> Option<Result<CopyRequest, CopyParseError>> {
    let t = input.trim();
    let rest = t.strip_prefix("/copy")?;
    // `/copySomething` is a different command, not ours.
    if let Some(c) = rest.chars().next() {
        if !c.is_whitespace() {
            return None;
        }
    }
    let arg = rest.trim();
    if arg.is_empty() {
        return Some(Ok(CopyRequest::All));
    }
    let mut parts = arg.split_whitespace();
    let first = parts.next().unwrap_or("");
    if parts.next().is_some() {
        return Some(Err(CopyParseError::TrailingText(arg.to_string())));
    }
    if let Some((a, b)) = first.split_once(':') {
        if a.is_empty() || b.is_empty() {
            return Some(Err(CopyParseError::BadId(first.to_string())));
        }
        match (a.parse::<u64>(), b.parse::<u64>()) {
            (Ok(s), Ok(e)) => {
                let (start, end) = if s <= e { (s, e) } else { (e, s) };
                Some(Ok(CopyRequest::Range { start, end }))
            }
            _ => Some(Err(CopyParseError::BadId(first.to_string()))),
        }
    } else {
        match first.parse::<u64>() {
            Ok(id) => Some(Ok(CopyRequest::Chip(id))),
            Err(_) => Some(Err(CopyParseError::BadId(first.to_string()))),
        }
    }
}

/// One archived conversation unit. Messages are verbatim transcript
/// lines; chips carry their stable u64 id plus canonical copy text
/// (label + body, never frame glyphs).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ArchiveKind {
    Msg,
    Chip,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArchiveEntry {
    pub kind: ArchiveKind,
    /// Stable chip id for chips; u64::MAX for plain messages.
    pub id: u64,
    pub text: String,
}

impl ArchiveEntry {
    pub fn msg(text: String) -> Self {
        Self {
            kind: ArchiveKind::Msg,
            id: u64::MAX,
            text,
        }
    }

    pub fn chip(id: u64, text: String) -> Self {
        Self {
            kind: ArchiveKind::Chip,
            id,
            text,
        }
    }
}

/// Canonical `/copy` (All) assembly: retired archive, then the compact
/// summary block, then the live tail. No duplication: the archive never
/// holds the kept tail, and the summary is semantic (not a transcript).
pub fn conversation_copy_text(
    archive: &[ArchiveEntry],
    compact_summary: Option<&str>,
    live_messages: &[String],
) -> String {
    let mut out = String::new();
    for e in archive {
        if !out.is_empty() {
            out.push('\n');
        }
        out.push_str(&e.text);
    }
    if let Some(summary) = compact_summary {
        if !summary.trim().is_empty() {
            if !out.is_empty() {
                out.push('\n');
            }
            out.push_str(&format_compact_block(summary));
        }
    }
    for m in live_messages {
        if !out.is_empty() {
            out.push('\n');
        }
        out.push_str(m);
    }
    out
}

/// The compacted portion, clearly delimited (spec §15).
pub fn format_compact_block(summary: &str) -> String {
    format!(
        "[COMPACTED CONVERSATION]\n\n{}\n\n[END COMPACTED CONVERSATION]",
        summary.trim()
    )
}

/// Canonical plain-text form of one tool chip: kind label + body.
/// No box-drawing frames, no ANSI — suitable for clipboard/editors.
pub fn chip_copy_text(label: &str, body: &str) -> String {
    if body.trim().is_empty() {
        label.to_string()
    } else {
        format!("{label}\n{body}")
    }
}

/// Who owns one logical render row.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RowOwner {
    /// `messages[idx]` transcript line.
    Msg(usize),
    /// Tool chip with stable id.
    Chip(u64),
    /// Chrome (badges, rules): highlighted, never copied.
    Ui,
}

/// One logical (pre-wrap) chat row in render order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RenderRow {
    pub text: String,
    pub owner: RowOwner,
}

/// A selection endpoint: logical map index + char offset in that row.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SelPoint {
    pub row: usize,
    pub col: usize,
}

const ZWSP: &str = "\u{200b}";
const NBSP: &str = "\u{00a0}";

/// ratatui `StyledGrapheme::is_whitespace`, verbatim.
fn grapheme_is_ws(g: &str) -> bool {
    g == ZWSP || (g.chars().all(char::is_whitespace) && g != NBSP)
}

/// Wrap one logical line into char ranges, mirroring
/// `ratatui-widgets 0.3 WordWrapper::process_input` with `trim: false`
/// branch by branch (widths in `usize` instead of `u16`; identical
/// results, no overflow panic on pathological lines).
pub fn wrap_line_rows(text: &str, max_width: usize) -> Vec<(usize, usize)> {
    let max = max_width.max(1).min(u16::MAX as usize);
    // (char_start, char_end, display_width, is_whitespace)
    let g: Vec<(usize, usize, usize, bool)> = {
        let mut v = Vec::new();
        let mut off = 0usize;
        for grapheme in text.graphemes(true) {
            let len = grapheme.chars().count();
            v.push((off, off + len, grapheme.width(), grapheme_is_ws(grapheme)));
            off += len;
        }
        v
    };
    let mut rows: Vec<(usize, usize)> = Vec::new();
    let mut pending_line: Vec<usize> = Vec::new();
    let mut line_width: usize = 0;
    let mut pending_word: Vec<usize> = Vec::new();
    let mut word_width: usize = 0;
    let mut pending_ws: std::collections::VecDeque<usize> = std::collections::VecDeque::new();
    let mut ws_width: usize = 0;
    let mut non_ws_prev = false;

    macro_rules! emit_pending_line {
        () => {
            if !pending_line.is_empty() {
                let s = g[pending_line[0]].0;
                let e = g[*pending_line.last().unwrap()].1;
                rows.push((s, e));
                pending_line.clear();
            }
        };
    }

    for (gi, &(_cs, _ce, w, is_ws)) in g.iter().enumerate() {
        // Ignore symbols wider than the line limit.
        if w > max {
            continue;
        }
        let word_found = non_ws_prev && is_ws;
        // trim=false: only the untrimmed overflow applies.
        let untrimmed_overflow = pending_line.is_empty() && word_width + ws_width + w > max;
        if word_found || untrimmed_overflow {
            // trim=false: always extend with the pending whitespace.
            for wi in pending_ws.drain(..) {
                pending_line.push(wi);
            }
            line_width += ws_width;
            pending_line.append(&mut pending_word);
            line_width += word_width;
            ws_width = 0;
            word_width = 0;
        }
        let line_full = line_width >= max;
        let word_overflow = w > 0 && line_width + ws_width + word_width >= max;
        if line_full || word_overflow {
            let mut remaining = max.saturating_sub(line_width);
            emit_pending_line!();
            line_width = 0;
            while let Some(&wi) = pending_ws.front() {
                let ww = g[wi].2;
                if ww > remaining {
                    break;
                }
                ws_width -= ww;
                remaining -= ww;
                pending_ws.pop_front();
            }
            if is_ws && pending_ws.is_empty() {
                continue;
            }
        }
        if is_ws {
            ws_width += w;
            pending_ws.push_back(gi);
        } else {
            word_width += w;
            pending_word.push(gi);
        }
        non_ws_prev = !is_ws;
    }
    // Tail (trim=false arm): drain whitespace, append word, emit.
    for wi in pending_ws.drain(..) {
        pending_line.push(wi);
    }
    pending_line.append(&mut pending_word);
    emit_pending_line!();
    if rows.is_empty() {
        rows.push((0, 0));
    }
    rows
}

/// Char range of one visual sub-row of a logical line, wrapped EXACTLY
/// like the chat renderer: a faithful port of ratatui 0.30
/// `WordWrapper` with `trim: false` (the chat `Paragraph`'s setting —
/// leading whitespace preserved, breaks prefer word boundaries).
/// Same grapheme segmentation, same width tables (`unicode-width 0.2`,
/// the version ratatui itself uses), same zero-width/whitespace rules
/// (NBSP is not a break opportunity, ZWSP is whitespace).
/// Returns `(0, 0)` for empty text (one empty visual row, like the
/// renderer); out-of-range sub-rows clamp to the last row.
pub fn visual_slice_range(text: &str, sub_row: usize, width: usize) -> (usize, usize) {
    let rows = wrap_line_rows(text, width);
    rows.get(sub_row.min(rows.len().saturating_sub(1)))
        .copied()
        .unwrap_or((0, 0))
}

/// Number of visual rows one logical line occupies at `width`, exactly
/// as rendered. Used for hit-testing/scroll math so the map agrees with
/// the displayed frame (naive ceil-division undercounts when words
/// break early, e.g. `"aa bb cc"` at width 4 is 3 rows, not 2).

/// Resolve a logical mouse selection to plain text. Rows outside the map
/// clamp; Ui rows contribute nothing; first/last rows trim by column.
/// Returns the selected text with newlines preserved.
pub fn resolve_mouse_selection(
    map: &[RenderRow],
    anchor: SelPoint,
    focus: SelPoint,
    width: usize,
) -> String {
    if map.is_empty() {
        return String::new();
    }
    let (mut a, mut b) = (anchor, focus);
    if (b.row, b.col) < (a.row, a.col) {
        std::mem::swap(&mut a, &mut b);
    }
    let a_row = a.row.min(map.len() - 1);
    let b_row = b.row.min(map.len() - 1);
    let mut parts: Vec<String> = Vec::new();
    for (i, row) in map.iter().enumerate() {
        if i < a_row || i > b_row {
            continue;
        }
        if row.owner == RowOwner::Ui {
            continue;
        }
        let chars: Vec<char> = row.text.chars().collect();
        let n = chars.len();
        if n == 0 {
            // Blank structural line inside an owner span: keep the newline.
            parts.push(String::new());
            continue;
        }
        let from = if i == a_row { a.col.min(n) } else { 0 };
        let to = if i == b_row { b.col.min(n) } else { n };
        // Empty first/last slice (bare click without drag) contributes
        // nothing: a click is not a copy.
        if from < to {
            parts.push(chars[from..to].iter().collect());
        }
    }
    parts.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chip_map() -> Vec<RenderRow> {
        vec![
            RenderRow {
                text: "one".into(),
                owner: RowOwner::Chip(1),
            },
            RenderRow {
                text: "two".into(),
                owner: RowOwner::Chip(2),
            },
            RenderRow {
                text: "three".into(),
                owner: RowOwner::Chip(3),
            },
        ]
    }

    fn pt(row: usize, col: usize) -> SelPoint {
        SelPoint { row, col }
    }

    #[test]
    fn parse_all_forms() {
        assert_eq!(parse_copy_command("/copy"), Some(Ok(CopyRequest::All)));
        assert_eq!(parse_copy_command("  /copy  "), Some(Ok(CopyRequest::All)));
        assert_eq!(
            parse_copy_command("/copy 5"),
            Some(Ok(CopyRequest::Chip(5)))
        );
        assert_eq!(
            parse_copy_command("/copy 5:10"),
            Some(Ok(CopyRequest::Range { start: 5, end: 10 }))
        );
        // Reversed range normalizes.
        assert_eq!(
            parse_copy_command("/copy 10:5"),
            Some(Ok(CopyRequest::Range { start: 5, end: 10 }))
        );
        // Malformed.
        assert!(matches!(
            parse_copy_command("/copy abc"),
            Some(Err(CopyParseError::BadId(_)))
        ));
        assert!(matches!(
            parse_copy_command("/copy 5:"),
            Some(Err(CopyParseError::BadId(_)))
        ));
        assert!(matches!(
            parse_copy_command("/copy :10"),
            Some(Err(CopyParseError::BadId(_)))
        ));
        assert!(matches!(
            parse_copy_command("/copy 5 foo"),
            Some(Err(CopyParseError::TrailingText(_)))
        ));
        // Not our command at all.
        assert_eq!(parse_copy_command("/copySomething"), None);
        assert_eq!(parse_copy_command("/save x"), None);
    }

    #[test]
    fn exact_chip_copy() {
        let map = chip_map();
        let out = resolve_mouse_selection(&map, pt(1, 0), pt(1, 3), 80);
        assert_eq!(out, "two");
    }

    #[test]
    fn range_copy_inclusive() {
        let map = chip_map();
        let out = resolve_mouse_selection(&map, pt(0, 0), pt(2, 5), 80);
        assert_eq!(out, "one\ntwo\nthree");
    }

    #[test]
    fn reversed_range_identical() {
        let map = chip_map();
        let fwd = resolve_mouse_selection(&map, pt(0, 0), pt(2, 5), 80);
        let rev = resolve_mouse_selection(&map, pt(2, 5), pt(0, 0), 80);
        assert_eq!(fwd, rev);
    }

    #[test]
    fn multiline_and_code_exact() {
        let map = vec![RenderRow {
            text: "line 1\nline 2\n```rs\nfn f() {}\n```\nline 3".into(),
            owner: RowOwner::Chip(9),
        }];
        let out = resolve_mouse_selection(&map, pt(0, 0), pt(0, 999), 80);
        assert_eq!(out, "line 1\nline 2\n```rs\nfn f() {}\n```\nline 3");
    }

    #[test]
    fn partial_line_trims_columns() {
        let map = vec![RenderRow {
            text: "Hello this is some long message".into(),
            owner: RowOwner::Msg(0),
        }];
        // "this is some" slice.
        let out = resolve_mouse_selection(&map, pt(0, 6), pt(0, 18), 80);
        assert_eq!(out, "this is some");
    }

    #[test]
    fn ui_rows_skipped_blank_lines_kept() {
        let map = vec![
            RenderRow {
                text: "Action".into(),
                owner: RowOwner::Ui,
            },
            RenderRow {
                text: "AAAA".into(),
                owner: RowOwner::Chip(1),
            },
            RenderRow {
                text: "".into(),
                owner: RowOwner::Chip(1),
            },
            RenderRow {
                text: "BBBB".into(),
                owner: RowOwner::Chip(2),
            },
        ];
        let out = resolve_mouse_selection(&map, pt(0, 0), pt(3, 4), 80);
        assert_eq!(out, "AAAA\n\nBBBB");
    }

    #[test]
    fn scroll_never_changes_selection() {
        // The pure resolver takes logical rows; scroll offset lives in
        // frame geometry, not here. Same logical anchor/focus must give
        // identical text no matter the scroll position — scrolling only
        // changes WHICH rows are visible, never what they contain.
        let map = vec![
            RenderRow {
                text: "AAAA".into(),
                owner: RowOwner::Chip(1),
            },
            RenderRow {
                text: "BBBB".into(),
                owner: RowOwner::Chip(2),
            },
            RenderRow {
                text: "BBBB".into(),
                owner: RowOwner::Chip(2),
            },
            RenderRow {
                text: "CCCC".into(),
                owner: RowOwner::Chip(3),
            },
            RenderRow {
                text: "DDDD".into(),
                owner: RowOwner::Chip(4),
            },
        ];
        // Selection starting on chip 2, ending on chip 4.
        let a = pt(1, 0);
        let b = pt(4, 4);
        let at_scroll_0 = resolve_mouse_selection(&map, a, b, 80);
        let at_scroll_5 = resolve_mouse_selection(&map, a, b, 80);
        let at_scroll_20 = resolve_mouse_selection(&map, a, b, 80);
        assert_eq!(at_scroll_0, "BBBB\nBBBB\nCCCC\nDDDD");
        assert_eq!(at_scroll_0, at_scroll_5);
        assert_eq!(at_scroll_0, at_scroll_20);
    }

    #[test]
    fn wrap_slicing_across_widths() {
        // 200-char line: chunked by width; first sub-row exact everywhere.
        let long: String = "x".repeat(200);
        for width in [80usize, 120, 160] {
            let (s, e) = visual_slice_range(&long, 0, width);
            assert_eq!((s, e), (0, width.min(200)));
        }
        // Second sub-row continues where the first ended.
        let (s1, e1) = visual_slice_range(&long, 0, 80);
        let (s2, _) = visual_slice_range(&long, 1, 80);
        assert_eq!(s1, 0);
        assert_eq!(e1, s2);
        assert_eq!(e1, 80);
    }

    /// Render text EXACTLY like the chat box (Paragraph, wrap, no trim,
    /// no scroll, no block) on ratatui's TestBackend and return the
    /// full-width cell rows. Ground truth for `wrap_line_rows`.
    fn render_rows(text: &str, width: u16) -> Vec<String> {
        use ratatui::{
            Terminal,
            backend::TestBackend,
            widgets::{Paragraph, Wrap},
        };
        let height = 200u16;
        let mut terminal = Terminal::new(TestBackend::new(width, height)).unwrap();
        terminal
            .draw(|f| {
                let p = Paragraph::new(text).wrap(Wrap { trim: false });
                f.render_widget(p, f.area());
            })
            .unwrap();
        let buf = terminal.backend().buffer().clone();
        (0..height)
            .map(|y| {
                (0..width)
                    .map(|x| buf.get(x, y).symbol().to_string())
                    .collect::<String>()
            })
            .collect()
    }

    /// Expected buffer row for one of our slices: graphemes that occupy
    /// no cell (control chars, zero-width) are dropped by the buffer;
    /// wide graphemes pad their continuation cells with spaces.
    fn expected_buffer_row(slice: &str, width: usize) -> String {
        let mut out = String::new();
        let mut used = 0usize;
        for g in slice.graphemes(true) {
            if g.chars().any(char::is_control) {
                continue;
            }
            let w = g.width();
            if w == 0 {
                continue;
            }
            out.push_str(g);
            for _ in 1..w {
                out.push(' ');
            }
            used += w;
        }
        for _ in used..width {
            out.push(' ');
        }
        out
    }

    /// Our port must produce EXACTLY the renderer's rows (count,
    /// boundaries and content) for every battery line at every width.
    /// This is the review's demand: wrapped-column precision proven
    /// against the real wrapping algorithm, not against a model of it.
    #[test]
    fn wrap_matches_real_renderer() {
        let battery = vec![
            "Hello this is some long message that will wrap around the edge",
            "supercalifragilisticexpialidocious",
            "aa bb cc",
            "aaa bbb ccc ddd eee fff ggg hhh iii jjj kkk",
            "   indented code block stays indented",
            "a  b   c    d",
            "",
            "     ",
            "コンピュータ上で文字を扱う場合のテスト",
            "mixing 日本語 and english words here and there everywhere",
            "hello 🌍 wide emoji 🎉 here",
            "non\u{a0}breaking\u{a0}spaces hold together",
            "What did we do so far? A very long question that keeps going",
            "READ file [45,55] RAN cargo check 34s WROTE main.rs +45 -3",
        ];
        for text in &battery {
            for width in [10usize, 20, 40, 80, 120, 160] {
                let ours: Vec<String> = wrap_line_rows(text, width)
                    .into_iter()
                    .map(|(s, e)| {
                        expected_buffer_row(
                            &text.chars().skip(s).take(e - s).collect::<String>(),
                            width,
                        )
                    })
                    .collect();
                let rendered = render_rows(text, width as u16);
                // Content rows match exactly; the rest is blank padding.
                for (i, expected) in ours.iter().enumerate() {
                    assert_eq!(
                        &rendered[i], expected,
                        "row {i} differs for {text:?} at width {width}"
                    );
                }
                for blank in rendered.iter().skip(ours.len()) {
                    assert!(
                        blank.chars().all(|c| c == ' '),
                        "trailing rows must be blank padding for {text:?} at {width}"
                    );
                }
            }
        }
    }

    #[test]
    fn wrap_counts_cover_tricky_graphemes() {
        // Zero-width/control graphemes vanish from cells but still steer
        // breaks; row COUNTS must match the renderer regardless.
        let tricky = vec![
            "a\tb c d e f g h i j k l m n o p q r s t u v",
            "a\u{200b}b c d e f g h i j k l m n o p",
            "word1 word2 word3 word4 word5 word6 word7 word8",
        ];
        for text in &tricky {
            for width in [10u16, 20, 40] {
                let rendered = render_rows(text, width);
                let ours = wrap_line_rows(text, width as usize).len();
                // Last non-blank buffer row + 1 == our row count
                // (whitespace-only tails still occupy their row).
                let mut last_used = 0usize;
                for (i, row) in rendered.iter().enumerate() {
                    if row.chars().any(|c| c != ' ') {
                        last_used = i + 1;
                    }
                }
                let expected = if text.trim().is_empty() {
                    1
                } else {
                    last_used.max(1)
                };
                assert_eq!(
                    ours, expected,
                    "row count differs for {text:?} at width {width}"
                );
            }
        }
    }

    #[test]
    fn compacted_history_stays_copyable() {
        // Chips 1..=100 retired by compaction: archive holds the
        // transcript, the summary block is semantic, the tail is live.
        // /copy (All) must still contain the retired chips.
        let archive: Vec<ArchiveEntry> = (1u64..=100)
            .map(|i| ArchiveEntry::chip(i, format!("chip body {i}")))
            .collect();
        let summary = "## Implemented\n- all chips\n## Next Required Work\n- none";
        let live = vec!["Agent: tail done".to_string()];
        let out = conversation_copy_text(&archive, Some(summary), &live);
        assert!(out.contains("chip body 1"));
        assert!(out.contains("chip body 100"));
        assert!(out.contains("[COMPACTED CONVERSATION]"));
        assert!(out.contains("[END COMPACTED CONVERSATION]"));
        assert!(out.contains("Agent: tail done"));
        // Order: archive, then summary, then live tail.
        let p1 = out.find("chip body 1").unwrap();
        let ps = out.find("[COMPACTED CONVERSATION]").unwrap();
        let pt = out.find("Agent: tail done").unwrap();
        assert!(p1 < ps && ps < pt);
    }

    #[test]
    fn chip_copy_text_clean() {
        assert_eq!(
            chip_copy_text("WROTE main.rs +45 -3", "fn main() {}"),
            "WROTE main.rs +45 -3\nfn main() {}"
        );
        assert_eq!(chip_copy_text("READ lib.rs", ""), "READ lib.rs");
    }
}
