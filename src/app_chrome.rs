//! Application-chrome visual style: top bar, menu modal frame, input
//! badges, and other application-owned decoration.
//!
//! Scope boundary (read before extending): this controls APPLICATION
//! CHROME ONLY. AI responses, Markdown, code blocks, terminal output,
//! tool output and agent messages are never touched here.
//!
//! Central contract — every run below has an IDENTICAL cell width in all
//! three styles, so geometry, hitboxes, keyboard navigation and layout
//! math never change with the style. Tests pin the widths. Renderers must
//! take glyphs AND widths from [`AppChrome`], never hard-code literals.
//!
//! Semantics (not mere glyph renames):
//! - Modern:      diagonal/block decoration ON, drawing-box border ON.
//! - BorderLine:  diagonal/block decoration OFF, drawing-box border ON.
//! - None:        both OFF — spacing/text/selection only, spaces keep cells.

/// Application chrome style. Persisted in `RuntimeSettings`; default is
/// [`AppChromeStyle::Modern`] (today's appearance, unchanged).
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum AppChromeStyle {
    /// Fully decorated: diagonal blocks + box-drawing borders.
    Modern,
    /// Box-drawing borders only, no diagonal/block decoration.
    BorderLine,
    /// Clean terminal-safe UI: spacing/text/attributes only.
    None,
}

impl AppChromeStyle {
    pub fn all() -> [AppChromeStyle; 3] {
        use AppChromeStyle::*;
        [Modern, BorderLine, None]
    }

    pub fn label(self) -> &'static str {
        match self {
            AppChromeStyle::Modern => "Modern",
            AppChromeStyle::BorderLine => "Border Line",
            AppChromeStyle::None => "None",
        }
    }

    pub fn description(self) -> &'static str {
        match self {
            AppChromeStyle::Modern => "Diagonal blocks + drawing-box borders",
            AppChromeStyle::BorderLine => "Drawing-box borders only, no diagonal decoration",
            AppChromeStyle::None => "Clean terminal UI, no decoration, no box borders",
        }
    }

    pub fn cycle_next(self) -> AppChromeStyle {
        match self {
            AppChromeStyle::Modern => AppChromeStyle::BorderLine,
            AppChromeStyle::BorderLine => AppChromeStyle::None,
            AppChromeStyle::None => AppChromeStyle::Modern,
        }
    }

    pub fn cycle_prev(self) -> AppChromeStyle {
        match self {
            AppChromeStyle::Modern => AppChromeStyle::None,
            AppChromeStyle::None => AppChromeStyle::BorderLine,
            AppChromeStyle::BorderLine => AppChromeStyle::Modern,
        }
    }

    /// Diagonal/block decoration (spiky transition glyphs)?
    pub fn show_diagonal_blocks(self) -> bool {
        matches!(self, AppChromeStyle::Modern)
    }

    /// Drawing-box borders (corners, edges, rules)?
    pub fn show_box_border(self) -> bool {
        !matches!(self, AppChromeStyle::None)
    }

    /// THE centralized glyph mapping. Every run keeps its documented cell
    /// width in all three styles — see width tests below.
    pub fn chrome(self) -> AppChrome {
        match self {
            AppChromeStyle::Modern => AppChrome {
                diagonal_blocks: true,
                box_border: true,
                // Modal row 0: 3 + title + 2 + fill + 2 + " x " + 3.
                modal_tl: "🭈🭆🭂",
                modal_title_right: "🭞🭜",
                modal_fill: "🬂",
                modal_tr_mid: "🭧🭓",
                modal_tr_end: "🭍🭑🬽",
                // Row 1 sub-corners (3 cells each side; middle is spaces).
                modal_sub_l: "🭝🭜🭘",
                modal_sub_r: "🭣🭧🭒",
                // Middle rows (1 cell each side).
                modal_side_l: "▌",
                modal_side_r: "▐",
                // Row H-2 sub-corners (3 cells each side).
                modal_bot_l: "🭌🭑🬽",
                modal_bot_r: "🭈🭆🭁",
                // Row H-1 bottom line (5 cells each side + fill).
                modal_foot_l: "🭣🭧🭓🭍🭑",
                modal_foot_fill: "🬭",
                modal_foot_r: "🭆🭂🭞🭜🭘",
                // Top bar transitions (2 cells) and mid fill.
                bar_left_trans: "🭞🭜",
                bar_right_trans: "🭧🭓",
                bar_fill: "🬂",
                // Input bar: transitions (2 cells) and mid fill.
                input_trans: "🭆🭂",
                input_main_trans: "🭍🭑",
                input_fill: "🬭",
            },
            AppChromeStyle::BorderLine => AppChrome {
                diagonal_blocks: false,
                box_border: true,
                modal_tl: "┌──",
                // Title sits ON the top edge: the rule continues on both
                // sides of the badge (a corner here would notch the edge).
                modal_title_right: "──",
                modal_fill: "─",
                modal_tr_mid: "──",
                modal_tr_end: "──┐",
                modal_sub_l: "│  ",
                modal_sub_r: "  │",
                modal_side_l: "│",
                modal_side_r: "│",
                modal_bot_l: "│  ",
                modal_bot_r: "  │",
                modal_foot_l: "└────",
                modal_foot_fill: "─",
                modal_foot_r: "────┘",
                // Bar/badge junctions stay a continuous rule in BorderLine:
                // no bracket decoration, same 2-cell widths as Modern.
                bar_left_trans: "──",
                bar_right_trans: "──",
                bar_fill: "─",
                input_trans: "──",
                input_main_trans: "──",
                input_fill: "─",
            },
            AppChromeStyle::None => AppChrome {
                diagonal_blocks: false,
                box_border: false,
                modal_tl: "   ",
                modal_title_right: "  ",
                modal_fill: " ",
                modal_tr_mid: "  ",
                modal_tr_end: "   ",
                modal_sub_l: "   ",
                modal_sub_r: "   ",
                modal_side_l: " ",
                modal_side_r: " ",
                modal_bot_l: "   ",
                modal_bot_r: "   ",
                modal_foot_l: "     ",
                modal_foot_fill: " ",
                modal_foot_r: "     ",
                // Structural application bar lines stay visible even with
                // no decorative framing: the top/bottom bars are UI
                // hierarchy, not decoration. Transitions also draw the
                // rule so no gap opens between badges and the line. Same
                // 2-cell / 1-cell widths as the other styles, so geometry
                // never moves.
                bar_left_trans: "──",
                bar_right_trans: "──",
                bar_fill: "─",
                input_trans: "──",
                input_main_trans: "──",
                input_fill: "─",
            },
        }
    }
}

impl Default for AppChromeStyle {
    fn default() -> Self {
        AppChromeStyle::Modern
    }
}

impl std::fmt::Display for AppChromeStyle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.label())
    }
}

/// Fixed-width decorative runs for one style. Widths are part of the
/// contract: every field has identical `.chars().count()` in all styles
/// (see width tests), so layout/hitbox math derived from these runs is
/// style-invariant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AppChrome {
    pub diagonal_blocks: bool,
    pub box_border: bool,
    pub modal_tl: &'static str,
    pub modal_title_right: &'static str,
    pub modal_fill: &'static str,
    pub modal_tr_mid: &'static str,
    pub modal_tr_end: &'static str,
    pub modal_sub_l: &'static str,
    pub modal_sub_r: &'static str,
    pub modal_side_l: &'static str,
    pub modal_side_r: &'static str,
    pub modal_bot_l: &'static str,
    pub modal_bot_r: &'static str,
    pub modal_foot_l: &'static str,
    pub modal_foot_fill: &'static str,
    pub modal_foot_r: &'static str,
    pub bar_left_trans: &'static str,
    pub bar_right_trans: &'static str,
    pub bar_fill: &'static str,
    pub input_trans: &'static str,
    pub input_main_trans: &'static str,
    pub input_fill: &'static str,
}

/// Current chrome from live settings. Call sites use this (never a cached
/// copy) so switching styles re-renders on the very next frame.
pub fn current_chrome() -> AppChrome {
    crate::settings::get_app_chrome_style().chrome()
}

/// Live-preview content for the App Style settings screen: a miniature
/// sample of what the style does to application chrome, painted in the
/// CURRENT palette so the preview reflects style + palette together.
/// Pure function of style + palette + width — snapshot-tested per style.
///
/// - Modern shows diagonal clusters plus a box corner row.
/// - Border Line shows box corners with no diagonal decoration.
/// - None shows plain hierarchy text plus the structural bar line (top
///   and bottom application bars stay visible — they are hierarchy,
///   not decoration).
pub fn style_preview_lines(
    style: AppChromeStyle,
    palette: &crate::app_palette::AppPalette,
    width: usize,
) -> Vec<ratatui::text::Line<'static>> {
    use ratatui::style::{Modifier, Style};
    use ratatui::text::{Line, Span};
    let chrome = style.chrome();
    // Painted on the menu modal background (surface in None, canvas
    // otherwise) so the preview never bands against its panel.
    let bg = crate::app::modal_bg();
    let frame_fg = Style::default().bg(bg).fg(palette.accent_c());
    let title_fg = Style::default()
        .bg(bg)
        .fg(palette.foreground_c())
        .add_modifier(Modifier::BOLD);
    let body_fg = Style::default().bg(bg).fg(palette.muted_c());
    let w = width.max(20);
    let mut lines = Vec::new();
    lines.push(Line::from(Span::styled(
        format!("{} — {}", style.label(), style.description()),
        title_fg,
    )));
    lines.push(Line::from(""));
    let title = " Preview ";
    match style {
        AppChromeStyle::Modern => {
            // Diagonal clusters + box corners, mirroring the modal frame.
            let mut top = String::from(chrome.modal_tl);
            top.push_str(title);
            while top.chars().count() < w.saturating_sub(chrome.modal_tr_end.chars().count()) {
                top.push_str(chrome.modal_fill);
            }
            top.push_str(chrome.modal_tr_end);
            lines.push(Line::from(Span::styled(top, frame_fg)));
            for body in [
                "Top bar, badges and menu frame.",
                "Popups keep drawing-box borders.",
            ] {
                lines.push(Line::from(Span::styled(body.to_string(), body_fg)));
            }
            let mut bottom = String::from(chrome.modal_foot_l);
            while bottom.chars().count() < w.saturating_sub(chrome.modal_foot_r.chars().count()) {
                bottom.push_str(chrome.modal_foot_fill);
            }
            bottom.push_str(chrome.modal_foot_r);
            lines.push(Line::from(Span::styled(bottom, frame_fg)));
        }
        AppChromeStyle::BorderLine => {
            // Plain box corners, no diagonal clusters anywhere.
            let mut top = String::from("┌");
            top.push_str(title);
            while top.chars().count() < w.saturating_sub(1) {
                top.push('─');
            }
            top.push('┐');
            lines.push(Line::from(Span::styled(top, frame_fg)));
            for body in [
                "Top bar, badges and menu frame.",
                "Popups keep drawing-box borders.",
            ] {
                lines.push(Line::from(Span::styled(format!("│{body}"), body_fg)));
            }
            let bottom: String = std::iter::once('└')
                .chain(std::iter::repeat('─').take(w.saturating_sub(2)))
                .chain(std::iter::once('┘'))
                .collect();
            lines.push(Line::from(Span::styled(bottom, frame_fg)));
        }
        AppChromeStyle::None => {
            // Hierarchy only: section title + plain body, zero framing —
            // plus the structural application bar line, which stays
            // visible because bars are hierarchy, not decoration.
            lines.push(Line::from(Span::styled("Preview", title_fg)));
            for body in [
                "Top bar, badges and menu frame.",
                "Spacing and text carry hierarchy.",
            ] {
                lines.push(Line::from(Span::styled(body.to_string(), body_fg)));
            }
            let rule: String = std::iter::repeat('─').take(w).collect();
            lines.push(Line::from(Span::styled(
                rule,
                Style::default().bg(bg).fg(palette.separator_c()),
            )));
        }
    }
    lines
}

/// Container frame under the style, for symmetric ratatui `Block`
/// containers (popups, panels). Modern/BorderLine pass through untouched
/// (those boxes are already drawing-box as coded); None swaps glyphs for
/// spaces — same rect, same inner area, identical geometry, invisible
/// frame. No layout, hitbox or scroll math changes in any style.
pub fn frame_container<'a>(block: ratatui::widgets::Block<'a>) -> ratatui::widgets::Block<'a> {
    match crate::settings::get_app_chrome_style() {
        AppChromeStyle::None => block.border_set(ratatui::symbols::border::Set {
            top_left: " ",
            top_right: " ",
            bottom_left: " ",
            bottom_right: " ",
            vertical_left: " ",
            vertical_right: " ",
            horizontal_top: " ",
            horizontal_bottom: " ",
        }),
        _ => block,
    }
}

impl AppChrome {
    fn w(run: &'static str) -> u16 {
        run.chars().count() as u16
    }
    /// Cell widths backing layout/hitbox math. All styles agree (tested);
    /// renderers must derive geometry from these, never literals.
    pub fn modal_tl_w(&self) -> u16 {
        Self::w(self.modal_tl)
    }
    pub fn modal_title_right_w(&self) -> u16 {
        Self::w(self.modal_title_right)
    }
    pub fn modal_tr_mid_w(&self) -> u16 {
        Self::w(self.modal_tr_mid)
    }
    pub fn modal_tr_end_w(&self) -> u16 {
        Self::w(self.modal_tr_end)
    }
    pub fn modal_sub_w(&self) -> u16 {
        Self::w(self.modal_sub_l)
    }
    pub fn modal_side_w(&self) -> u16 {
        Self::w(self.modal_side_l)
    }
    pub fn modal_foot_w(&self) -> u16 {
        Self::w(self.modal_foot_l)
    }
    pub fn bar_trans_w(&self) -> u16 {
        Self::w(self.bar_left_trans)
    }
    pub fn input_trans_w(&self) -> u16 {
        Self::w(self.input_trans)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_style_is_modern() {
        assert_eq!(AppChromeStyle::default(), AppChromeStyle::Modern);
    }

    #[test]
    fn all_styles_have_labels_and_descriptions() {
        for style in AppChromeStyle::all() {
            assert!(!style.label().is_empty());
            assert!(!style.description().is_empty());
            assert_eq!(style.to_string(), style.label());
        }
        let labels: Vec<&str> = AppChromeStyle::all().iter().map(|s| s.label()).collect();
        assert_eq!(labels, ["Modern", "Border Line", "None"]);
    }

    #[test]
    fn cycling_covers_all_styles() {
        assert_eq!(
            AppChromeStyle::Modern.cycle_next(),
            AppChromeStyle::BorderLine
        );
        assert_eq!(
            AppChromeStyle::BorderLine.cycle_next(),
            AppChromeStyle::None
        );
        assert_eq!(AppChromeStyle::None.cycle_next(), AppChromeStyle::Modern);
        assert_eq!(AppChromeStyle::Modern.cycle_prev(), AppChromeStyle::None);
        assert_eq!(
            AppChromeStyle::None.cycle_prev(),
            AppChromeStyle::BorderLine
        );
        assert_eq!(
            AppChromeStyle::BorderLine.cycle_prev(),
            AppChromeStyle::Modern
        );
    }

    #[test]
    fn semantics_are_not_glyph_renames() {
        assert!(AppChromeStyle::Modern.show_diagonal_blocks());
        assert!(AppChromeStyle::Modern.show_box_border());
        assert!(!AppChromeStyle::BorderLine.show_diagonal_blocks());
        assert!(AppChromeStyle::BorderLine.show_box_border());
        assert!(!AppChromeStyle::None.show_diagonal_blocks());
        assert!(!AppChromeStyle::None.show_box_border());
    }

    #[test]
    fn every_run_keeps_width_across_styles() {
        // THE geometry invariant: identical cell counts per run in all
        // styles, so hitboxes/layout never move with the style.
        let themes: Vec<AppChrome> = AppChromeStyle::all().iter().map(|s| s.chrome()).collect();
        let widths = |f: fn(&AppChrome) -> &'static str| -> Vec<usize> {
            themes.iter().map(|t| f(t).chars().count()).collect()
        };
        for (name, ws) in [
            ("modal_tl", widths(|t| t.modal_tl)),
            ("modal_title_right", widths(|t| t.modal_title_right)),
            ("modal_fill", widths(|t| t.modal_fill)),
            ("modal_tr_mid", widths(|t| t.modal_tr_mid)),
            ("modal_tr_end", widths(|t| t.modal_tr_end)),
            ("modal_sub_l", widths(|t| t.modal_sub_l)),
            ("modal_sub_r", widths(|t| t.modal_sub_r)),
            ("modal_side_l", widths(|t| t.modal_side_l)),
            ("modal_side_r", widths(|t| t.modal_side_r)),
            ("modal_bot_l", widths(|t| t.modal_bot_l)),
            ("modal_bot_r", widths(|t| t.modal_bot_r)),
            ("modal_foot_l", widths(|t| t.modal_foot_l)),
            ("modal_foot_fill", widths(|t| t.modal_foot_fill)),
            ("modal_foot_r", widths(|t| t.modal_foot_r)),
            ("bar_left_trans", widths(|t| t.bar_left_trans)),
            ("bar_right_trans", widths(|t| t.bar_right_trans)),
            ("bar_fill", widths(|t| t.bar_fill)),
            ("input_trans", widths(|t| t.input_trans)),
            ("input_main_trans", widths(|t| t.input_main_trans)),
            ("input_fill", widths(|t| t.input_fill)),
        ] {
            assert_eq!(ws[0], ws[1], "{name} modern/borderline width");
            assert_eq!(ws[1], ws[2], "{name} borderline/none width");
        }
        // Spot-check documented widths.
        let m = AppChromeStyle::Modern.chrome();
        assert_eq!(m.modal_tl.chars().count(), 3);
        assert_eq!(m.modal_foot_l.chars().count(), 5);
        assert_eq!(m.modal_side_l.chars().count(), 1);
    }

    #[test]
    fn modern_keeps_diagonal_blocks_borderline_does_not() {
        let m = AppChromeStyle::Modern.chrome().modal_tl;
        assert!(m.contains('🭈'), "modern keeps diagonal clusters");
        let b = AppChromeStyle::BorderLine.chrome();
        for run in [b.modal_tl, b.modal_tr_end, b.modal_foot_l, b.modal_foot_r] {
            assert!(
                !run.chars()
                    .any(|c| ('🭀'..='🭿').contains(&c) || ('🬀'..='🬿').contains(&c)),
                "border line has no block decoration: {run}"
            );
        }
        let n = AppChromeStyle::None.chrome();
        // Decorative frame runs stay blank; the structural bar fills are
        // the deliberate exception (see bar_lines_present_in_every_style).
        for run in [
            n.modal_tl,
            n.modal_title_right,
            n.modal_fill,
            n.modal_sub_l,
            n.modal_side_l,
            n.modal_foot_l,
        ] {
            assert!(
                run.chars().all(|c| c == ' '),
                "none style renders no framing glyphs: {run:?}"
            );
        }
    }

    #[test]
    fn serde_roundtrip_all_styles() {
        for style in AppChromeStyle::all() {
            let json = serde_json::to_string(&style).expect("serialize");
            let back: AppChromeStyle = serde_json::from_str(&json).expect("deserialize");
            assert_eq!(back, style);
        }
        assert!(serde_json::from_str::<AppChromeStyle>("\"Retro\"").is_err());
    }

    #[test]
    fn preview_renders_for_every_style_and_width() {
        use crate::app_palette::AppPalette;
        let pal = AppPalette::rose_pine();
        for style in AppChromeStyle::all() {
            for width in [0, 12, 20, 40, 80] {
                let lines = style_preview_lines(style, &pal, width);
                assert!(!lines.is_empty(), "{style:?} w={width}");
                assert!(
                    lines[0].to_string().contains(style.label()),
                    "{style:?} preview must name the style"
                );
            }
        }
    }

    #[test]
    fn preview_matches_style_semantics() {
        use crate::app_palette::AppPalette;
        let pal = AppPalette::rose_pine();
        let text = |style| {
            style_preview_lines(style, &pal, 40)
                .iter()
                .map(|l| l.to_string())
                .collect::<Vec<_>>()
                .join("\n")
        };
        let modern = text(AppChromeStyle::Modern);
        assert!(modern.contains('🭈'), "modern shows diagonal clusters");
        assert!(
            modern.contains('┌') || modern.contains('🭈'),
            "modern frames"
        );
        let border = text(AppChromeStyle::BorderLine);
        assert!(border.contains('┌'), "border-line box corners");
        assert!(border.contains('┐') && border.contains('└'), "complete box");
        assert!(
            !border
                .chars()
                .any(|c| ('🭀'..='🭿').contains(&c) || ('🬀'..='🬿').contains(&c)),
            "no diagonal decoration in border-line"
        );
        let none = text(AppChromeStyle::None);
        assert!(none.contains("Preview"), "hierarchy text preserved");
        // The structural bar line stays (hierarchy, not decoration)…
        assert!(none.contains('─'), "none keeps the application bar line");
        // …but no decorative box frame or diagonal blocks leak in.
        for glyph in ['🭈', '┌', '┐', '└', '┘', '┏', '│'] {
            assert!(
                !none.contains(glyph),
                "none style must not frame: found {glyph}"
            );
        }
        assert!(
            !none
                .chars()
                .any(|c| ('🭀'..='🭿').contains(&c) || ('🬀'..='🬿').contains(&c)),
            "no diagonal decoration in none"
        );
    }
}
