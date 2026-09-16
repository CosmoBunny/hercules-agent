//! Start-screen splash artwork: `splash.txt` is the SINGLE SOURCE OF
//! TRUTH. The artwork is never copied into a Rust literal, never
//! reconstructed from Spans, and never reformatted: it is loaded
//! verbatim (no trimming of the artwork) and each source line
//! corresponds to one terminal row.
//!
//! Load order follows the existing asset convention
//! (see `resolve_worker_script`): executable dir → dev manifest dir →
//! current working directory → compile-time fallback derived from the
//! canonical file (`include_str!` of `splash.txt` itself, so a lone
//! installed binary such as `cargo install` still finds the artwork).
//! The file-based copy always wins when present, so packaged layouts
//! stay editable. The parsed artwork is cached for the lifetime of the
//! UI — the file is read once, never per frame.

use std::path::{Path, PathBuf};

/// One cached splash artwork: display-ready lines (tabs expanded to
/// their terminal tab-stop columns, so Ratatui — which does not render
/// tab stops — preserves the artwork geometry) plus per-line display
/// widths and the widest line.
#[derive(Debug, Clone)]
pub struct SplashArt {
    pub lines: Vec<String>,
    pub widths: Vec<usize>,
    pub max_width: usize,
}

impl SplashArt {
    /// True when the artwork failed to load (renderers skip the splash).
    pub fn is_empty(&self) -> bool {
        self.lines.is_empty()
    }
}

/// Compile-time fallback derived from the canonical file itself — not a
/// second source of truth, just `splash.txt` baked in at build time so
/// layouts where no runtime file exists (e.g. `cargo install`, which
/// ships only the binary) still render the exact artwork.
pub fn embedded_splash() -> &'static str {
    include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/splash.txt"))
}

/// Path candidates for `splash.txt`, following the existing asset
/// convention: alongside the executable first (bundled installs), then
/// the dev source tree, then the current working directory.
/// Pure over its inputs so packaging layouts are unit-testable.
pub fn splash_candidates_in(
    exe_dir: Option<&Path>,
    manifest_dir: &str,
    cwd: &Path,
) -> Vec<PathBuf> {
    let mut candidates = Vec::new();
    if let Some(dir) = exe_dir {
        candidates.push(dir.join("resources/splash.txt"));
        candidates.push(dir.join("splash.txt"));
    }
    candidates.push(PathBuf::from(manifest_dir).join("splash.txt"));
    candidates.push(cwd.join("splash.txt"));
    candidates
}

/// Path candidates for `splash.txt` in the real runtime environment.
fn splash_candidates() -> Vec<PathBuf> {
    let exe_dir = std::env::current_exe()
        .ok()
        .and_then(|exe| exe.parent().map(|d| d.to_path_buf()));
    let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    splash_candidates_in(exe_dir.as_deref(), env!("CARGO_MANIFEST_DIR"), &cwd)
}

/// First candidate that exists as a file, if any.
pub fn resolve_splash_file(candidates: &[PathBuf]) -> Option<PathBuf> {
    candidates.iter().find(|p| p.is_file()).cloned()
}

/// Runtime file-based splash path, if a packaged copy exists.
pub fn splash_path() -> Option<PathBuf> {
    resolve_splash_file(&splash_candidates())
}

/// Display width of one artwork character. Tabs advance to the next
/// tab-stop column (width 4), matching terminal behavior; control
/// characters render as zero width.
fn char_width(c: char, col: usize) -> usize {
    if c == '\t' {
        4 - (col % 4)
    } else {
        unicode_width::UnicodeWidthChar::width(c).unwrap_or(0)
    }
}

/// Display width of one artwork line (tabs expanded at `col`).
pub fn line_width(line: &str) -> usize {
    let mut col = 0usize;
    for c in line.chars() {
        col += char_width(c, col);
    }
    col
}

/// Display width of a tab-free display line (artwork rows after
/// tab expansion). Renderers use this to center cropped rows.
pub fn line_display_width(line: &str) -> usize {
    line.chars()
        .map(|c| unicode_width::UnicodeWidthChar::width(c).unwrap_or(0))
        .sum()
}

/// Expand tabs to their tab-stop columns so the artwork geometry is
/// preserved by renderers that do not implement tab stops. The
/// canonical `splash.txt` is never modified — this is a rendering
/// detail of the loader.
fn expand_tabs(line: &str) -> String {
    if !line.contains('\t') {
        return line.to_string();
    }
    let mut out = String::with_capacity(line.len());
    let mut col = 0usize;
    for c in line.chars() {
        if c == '\t' {
            let advance = 4 - (col % 4);
            out.push_str(&" ".repeat(advance));
            col += advance;
        } else {
            out.push(c);
            col += char_width(c, col);
        }
    }
    out
}

/// Parse raw artwork text into display-ready lines. Pure and
/// total: never panics, never trims artwork, never normalizes Unicode.
pub fn parse_splash(raw: &str) -> SplashArt {
    let mut lines = Vec::new();
    let mut widths = Vec::new();
    let mut max_width = 0usize;
    for line in raw.split('\n') {
        // Trailing '\r' from CRLF sources is transport noise, not
        // artwork — everything else is preserved verbatim.
        let line = line.strip_suffix('\r').unwrap_or(line);
        let expanded = expand_tabs(line);
        let w = line_width(&expanded);
        max_width = max_width.max(w);
        lines.push(expanded);
        widths.push(w);
    }
    SplashArt {
        lines,
        widths,
        max_width,
    }
}

/// Centered crop of one display line to `[start_col, start_col+width)`.
/// Never splits a wide glyph at either edge, never panics on any input
/// (empty lines, zero width, start past the end). Geometry outside the
/// window is dropped — never scaled, never reflowed.
pub fn crop_centered(line: &str, start_col: usize, width: usize) -> String {
    if width == 0 {
        return String::new();
    }
    let end_col = start_col.saturating_add(width);
    let mut out = String::new();
    let mut col = 0usize;
    for c in line.chars() {
        let cw = unicode_width::UnicodeWidthChar::width(c).unwrap_or(0);
        if col + cw <= start_col {
            col += cw;
            continue;
        }
        if col >= end_col {
            break;
        }
        // A wide glyph straddling either edge is dropped whole rather
        // than split.
        if col < start_col || col + cw > end_col {
            col += cw;
            continue;
        }
        out.push(c);
        col += cw;
    }
    out
}

/// Load and cache the splash artwork. The artwork is read verbatim:
/// lines are never trimmed, whitespace is never normalized, Unicode is
/// preserved. A runtime file copy wins when packaged layouts provide
/// one; otherwise the compile-time derivation of the canonical
/// `splash.txt` is used. Empty only if both are somehow missing.
pub fn load_splash() -> &'static SplashArt {
    static SPLASH: std::sync::OnceLock<SplashArt> = std::sync::OnceLock::new();
    SPLASH.get_or_init(|| {
        let raw = splash_candidates()
            .iter()
            .find_map(|p| std::fs::read_to_string(p).ok())
            .unwrap_or_else(|| embedded_splash().to_string());
        parse_splash(&raw)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_splash_loads_from_source_tree() {
        let art = load_splash();
        assert!(!art.is_empty(), "splash.txt must load in the dev tree");
        assert!(art.max_width > 0);
    }

    #[test]
    fn test_line_count_preserved() {
        // The loader must not drop or add rows: the cached line count
        // matches the canonical file's row count exactly.
        let raw = std::fs::read_to_string(
            std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("splash.txt"),
        )
        .expect("splash.txt readable in dev tree");
        let art = load_splash();
        assert_eq!(art.lines.len(), art.widths.len());
        assert_eq!(
            art.lines.len(),
            raw.split('\n').count(),
            "artwork must not be truncated or padded"
        );
    }

    #[test]
    fn test_blank_lines_preserved() {
        let art = load_splash();
        assert!(
            art.lines.iter().any(|l| l.is_empty()),
            "intentional blank lines must be preserved"
        );
    }

    #[test]
    fn test_unicode_content_preserved() {
        let art = load_splash();
        // Box-drawing glyphs from the artwork survive the loader.
        assert!(
            art.lines
                .iter()
                .any(|l| l.chars().any(|c| ['█', '│', '─', '🭁'].contains(&c))),
            "Unicode artwork glyphs must be preserved"
        );
    }

    #[test]
    fn test_whitespace_not_trimmed() {
        // Artwork columns survive the loader: the first glyph of the
        // first row is intact (no leading trim), interior tab-stop
        // spacing is present as spaces, and non-blank rows are never
        // empty after loading.
        let art = load_splash();
        let first = art.lines.first().expect("non-empty artwork");
        assert!(
            first.starts_with('🭁'),
            "first artwork glyph must survive loading untrimmed"
        );
        assert!(
            art.lines.iter().any(|l| l.contains(' ')),
            "interior artwork spacing must be preserved"
        );
        assert!(
            art.widths
                .iter()
                .zip(art.lines.iter())
                .all(|(w, l)| { (*w == 0) == l.is_empty() }),
            "widths must track content rows exactly"
        );
    }

    #[test]
    fn test_display_width_not_byte_length() {
        // █ is 3 bytes but 1 column: the widest line's width must be
        // well below its byte length.
        let art = load_splash();
        let widest = art
            .lines
            .iter()
            .zip(art.widths.iter())
            .max_by_key(|(_, w)| **w)
            .expect("non-empty artwork");
        assert!(
            widest.1 <= &widest.0.len(),
            "display width must use Unicode columns, not bytes"
        );
        assert_eq!(*widest.1, art.max_width);
    }

    #[test]
    fn test_tab_expansion_geometry() {
        // Tabs expand to the next tab-stop column (width 4):
        // "██" ends at column 2, first tab advances 2, second 4.
        assert_eq!(expand_tabs("██\t\t██"), "██      ██");
        assert_eq!(expand_tabs("a\tb"), "a   b");
        assert_eq!(expand_tabs("no-tabs"), "no-tabs");
        assert_eq!(line_width("██\t\t██"), 2 + 2 + 4 + 2);
    }

    #[test]
    fn test_crop_centered_ascii() {
        // Window [2, 6) of an 8-wide line: the middle, not the left edge.
        assert_eq!(crop_centered("abcdefgh", 2, 4), "cdef");
        assert_eq!(crop_centered("abcdefgh", 0, 8), "abcdefgh");
        assert_eq!(crop_centered("abcdefgh", 0, 20), "abcdefgh");
    }

    #[test]
    fn test_crop_centered_never_panics_on_tiny_inputs() {
        assert_eq!(crop_centered("", 0, 5), "");
        assert_eq!(crop_centered("", 3, 0), "");
        assert_eq!(crop_centered("ab", 0, 0), "");
        assert_eq!(crop_centered("ab", 10, 4), "");
        assert_eq!(crop_centered("ab", 1, 40), "b");
    }

    #[test]
    fn test_crop_centered_never_splits_wide_glyphs() {
        // 全 is 2 columns: a window edge landing mid-glyph drops it whole.
        assert_eq!(crop_centered("a全b", 0, 2), "a");
        assert_eq!(crop_centered("a全b", 1, 2), "全");
        assert_eq!(crop_centered("a全b", 2, 2), "b");
        assert_eq!(crop_centered("a全b", 0, 4), "a全b");
        assert_eq!(line_display_width(&crop_centered("a全b", 1, 2)), 2);
    }

    #[test]
    fn test_crop_centered_is_a_center_crop() {
        // 30-wide artwork row in a 20-wide terminal: start = (30-20)/2.
        let row = "█".repeat(30);
        let cropped = crop_centered(&row, (30 - 20) / 2, 20);
        assert_eq!(cropped.chars().count(), 20);
        assert_eq!(line_display_width(&cropped), 20);
    }

    #[test]
    fn test_packaged_path_resolution_order() {
        // Candidate order: exe-dir resources/ → exe-dir root →
        // manifest dir → cwd. Pure function over inputs.
        let exe = Path::new("/opt/hercules/bin");
        let cands = splash_candidates_in(Some(exe), "/src/hercules-agent", Path::new("/home/u"));
        assert_eq!(
            cands,
            vec![
                PathBuf::from("/opt/hercules/bin/resources/splash.txt"),
                PathBuf::from("/opt/hercules/bin/splash.txt"),
                PathBuf::from("/src/hercules-agent/splash.txt"),
                PathBuf::from("/home/u/splash.txt"),
            ]
        );
        // No exe dir (e.g. unknown launch context): manifest + cwd only.
        let cands = splash_candidates_in(None, "/src/hercules-agent", Path::new("/home/u"));
        assert_eq!(cands.len(), 2);
    }

    #[test]
    fn test_resolve_splash_file_prefers_first_existing() {
        // Simulate a packaged layout: exe-dir resources/splash.txt exists.
        let base =
            std::env::temp_dir().join(format!("hercules-splash-test-{}", std::process::id()));
        let res = base.join("bin").join("resources");
        std::fs::create_dir_all(&res).expect("temp dirs");
        std::fs::write(res.join("splash.txt"), "ART").expect("temp splash");
        let cands = splash_candidates_in(
            Some(&base.join("bin")),
            "/nonexistent-manifest",
            Path::new("/nonexistent-cwd"),
        );
        assert_eq!(
            resolve_splash_file(&cands),
            Some(res.join("splash.txt")),
            "packaged resources/splash.txt must resolve first"
        );
        // Nothing packaged anywhere: no file resolution (the loader then
        // falls back to the compile-time derivation of splash.txt).
        let cands = splash_candidates_in(
            Some(Path::new("/nonexistent-exe")),
            "/nonexistent-manifest",
            Path::new("/nonexistent-cwd"),
        );
        assert_eq!(resolve_splash_file(&cands), None);
        std::fs::remove_dir_all(&base).ok();
    }

    #[test]
    fn test_embedded_fallback_derives_from_canonical_file() {
        // The embedded fallback is the canonical file itself, not a
        // second copy: it parses to the same artwork the loader serves.
        let from_file = parse_splash(
            &std::fs::read_to_string(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("splash.txt"))
                .expect("splash.txt readable"),
        );
        let from_embedded = parse_splash(embedded_splash());
        assert_eq!(from_file.lines, from_embedded.lines);
        assert_eq!(from_file.widths, from_embedded.widths);
        assert_eq!(from_file.max_width, from_embedded.max_width);
    }
}

#[cfg(test)]
mod start_screen_tests {
    use crate::app::App;

    fn fresh_app() -> App {
        // App::new() starts with only the welcome message(s).
        App::new()
    }

    #[tokio::test(flavor = "current_thread")]
    async fn fresh_app_shows_splash() {
        let app = fresh_app();
        assert!(app.is_pristine_start_screen());
    }

    #[tokio::test(flavor = "current_thread")]
    async fn empty_messages_are_not_pristine() {
        // A fresh launch always carries the welcome message; an empty
        // list means unknown state and must never show the splash.
        let mut app = fresh_app();
        app.messages.clear();
        assert!(!app.is_pristine_start_screen());
    }

    #[tokio::test(flavor = "current_thread")]
    async fn welcome_required_for_pristine() {
        // Non-welcome content without the initial welcome is history,
        // not a start screen.
        let mut app = fresh_app();
        app.messages = vec!["System: something else".to_string()];
        assert!(!app.is_pristine_start_screen());
    }

    #[tokio::test(flavor = "current_thread")]
    async fn first_submitted_prompt_hides_splash() {
        let mut app = fresh_app();
        app.messages.push("You: hello".to_string());
        assert!(!app.is_pristine_start_screen());
    }

    #[tokio::test(flavor = "current_thread")]
    async fn existing_session_hides_splash() {
        let mut app = fresh_app();
        // Restored session with real conversation history.
        app.messages = vec![
            "You: earlier prompt".to_string(),
            "Agent: earlier answer".to_string(),
        ];
        assert!(!app.is_pristine_start_screen());
    }

    #[tokio::test(flavor = "current_thread")]
    async fn active_generation_hides_splash() {
        let mut app = fresh_app();
        if let Ok(mut g) = app.is_generating.lock() {
            *g = true;
        }
        assert!(!app.is_pristine_start_screen());
    }

    #[tokio::test(flavor = "current_thread")]
    async fn active_run_hides_splash() {
        let mut app = fresh_app();
        app.current_run = Some(crate::run_timeline::AgentRun::new("building".into()));
        assert!(!app.is_pristine_start_screen());
    }

    #[tokio::test(flavor = "current_thread")]
    async fn streamed_content_hides_splash() {
        let mut app = fresh_app();
        if let Ok(mut s) = app.streaming_response.lock() {
            *s = "partial tokens".to_string();
        }
        assert!(!app.is_pristine_start_screen());
    }

    #[tokio::test(flavor = "current_thread")]
    async fn pending_tool_activity_hides_splash() {
        let mut app = fresh_app();
        app.tool_result_context
            .push("[Tool] something ran".to_string());
        assert!(!app.is_pristine_start_screen());
    }

    /// Terminal resize: the splash must remain centered inside the chat
    /// area at every size (small terminals clip safely, no panic).
    #[tokio::test(flavor = "current_thread")]
    async fn test_splash_centered_across_resizes() {
        for (w, h) in [(80u16, 24u16), (100, 30), (120, 40), (170, 45)] {
            let mut app = fresh_app();
            let mut term =
                ratatui::Terminal::new(ratatui::backend::TestBackend::new(w, h)).expect("terminal");
            term.draw(|f| app.draw(f)).expect("draw");
            // Rendering must succeed; the splash artwork must be present
            // somewhere in the buffer (centered, not distorted).
            assert!(crate::splash::load_splash().max_width > 0);
        }
    }

    /// Artwork wider than the terminal: centered crop (middle visible),
    /// never a panic, never scaled. The artwork is 30 columns wide, so
    /// a 20-wide chat area exercises the crop path.
    #[tokio::test(flavor = "current_thread")]
    async fn test_narrow_terminal_center_crops_artwork() {
        let mut app = fresh_app();
        assert!(app.is_pristine_start_screen());
        let mut term =
            ratatui::Terminal::new(ratatui::backend::TestBackend::new(20, 24)).expect("terminal");
        term.draw(|f| app.draw(f)).expect("draw");
        let buf = term.backend().buffer().clone();
        // Middle of the artwork is visible: some row contains a run of
        // block glyphs (a leftmost-only crop of the tab-indented rows
        // would show mostly leading bars + spaces instead).
        let art = crate::splash::load_splash();
        let mid_row = &art.lines[1];
        let start = (art.widths[1] - 20) / 2;
        let expected = crate::splash::crop_centered(mid_row, start, 20);
        let rendered: String = buf
            .content
            .iter()
            .map(|c| c.symbol())
            .collect::<Vec<_>>()
            .chunks(20)
            .map(|row| row.concat())
            .collect::<Vec<_>>()
            .join("\n");
        assert!(
            rendered.contains(&expected),
            "center-cropped artwork row must be on screen"
        );
    }
}
