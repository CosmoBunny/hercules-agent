//! Application color palette: Rose Pine / Catppuccin / Tokyo Night /
//! Gruvbox / Custom.
//!
//! Scope boundary (read before extending): this controls APPLICATION
//! CHROME COLORS ONLY. AI responses, Markdown, code blocks, terminal
//! output, tool output and agent messages are never touched here.
//!
//! Palette and [`crate::app_chrome::AppChromeStyle`] are fully
//! independent axes:
//! - App Style controls frame/decorative GEOMETRY (glyphs).
//! - Color Palette controls COLORS.
//! Renderers take glyphs from `current_chrome()` and colors from
//! `current_palette()` — never hard-code RGB literals in chrome code.

use ratatui::style::Color;

/// Compact RGB triple. Stored as an array so serde works out of the box;
/// formatted as `#RRGGBB` in the Custom editor.
pub type Rgb = [u8; 3];

/// Color palette selection. Persisted in `RuntimeSettings`; default is
/// [`AppPaletteStyle::RosePine`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum AppPaletteStyle {
    RosePine,
    Catppuccin,
    TokyoNight,
    Gruvbox,
    Custom,
    /// The previous Hercules look: nordic canvas with a white accent.
    Simple,
}

impl AppPaletteStyle {
    pub fn all() -> [AppPaletteStyle; 6] {
        use AppPaletteStyle::*;
        [RosePine, Catppuccin, TokyoNight, Gruvbox, Custom, Simple]
    }

    /// Exact user-visible labels (pinned by test).
    pub fn label(self) -> &'static str {
        match self {
            AppPaletteStyle::RosePine => "Rose Pine",
            AppPaletteStyle::Catppuccin => "Catppuccin",
            AppPaletteStyle::TokyoNight => "Tokyo Night",
            AppPaletteStyle::Gruvbox => "Gruvbox",
            AppPaletteStyle::Custom => "Custom",
            AppPaletteStyle::Simple => "Simple",
        }
    }

    pub fn description(self) -> &'static str {
        match self {
            AppPaletteStyle::RosePine => "Muted lavender, rose and cyan on dark plum",
            AppPaletteStyle::Catppuccin => "Dark base with mauve and blue accents",
            AppPaletteStyle::TokyoNight => "Deep navy with cool blue and cyan accents",
            AppPaletteStyle::Gruvbox => "Warm dark background with orange accents",
            AppPaletteStyle::Custom => "Your own colors, edited below",
            AppPaletteStyle::Simple => "Classic Hercules look with a white accent",
        }
    }

    pub fn cycle_next(self) -> AppPaletteStyle {
        use AppPaletteStyle::*;
        match self {
            RosePine => Catppuccin,
            Catppuccin => TokyoNight,
            TokyoNight => Gruvbox,
            Gruvbox => Custom,
            Custom => Simple,
            Simple => RosePine,
        }
    }

    pub fn cycle_prev(self) -> AppPaletteStyle {
        use AppPaletteStyle::*;
        match self {
            RosePine => Simple,
            Simple => Custom,
            Custom => Gruvbox,
            Gruvbox => TokyoNight,
            TokyoNight => Catppuccin,
            Catppuccin => RosePine,
        }
    }

    /// Centralized palette for this selection. `Custom` resolves to the
    /// persisted custom colors (live settings), so edits apply instantly.
    pub fn palette(self) -> AppPalette {
        match self {
            AppPaletteStyle::RosePine => AppPalette::rose_pine(),
            AppPaletteStyle::Catppuccin => AppPalette::catppuccin(),
            AppPaletteStyle::TokyoNight => AppPalette::tokyo_night(),
            AppPaletteStyle::Gruvbox => AppPalette::gruvbox(),
            AppPaletteStyle::Simple => AppPalette::simple(),
            AppPaletteStyle::Custom => crate::settings::get_custom_palette(),
        }
    }
}

impl Default for AppPaletteStyle {
    fn default() -> Self {
        AppPaletteStyle::RosePine
    }
}

impl std::fmt::Display for AppPaletteStyle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.label())
    }
}

/// Centralized application palette. Every application-owned UI component
/// takes its colors from here — no hard-coded RGB in chrome renderers.
///
/// Field semantics:
/// - `background`: main application background.
/// - `surface` / `surface_alt`: popup/modal panels and raised layers.
/// - `menu_bg`: menu bar layer — always a darker shade than `background`
///   so the menu reads as a separate application layer.
/// - `menu_selected_bg`: selected menu item (with `selection_fg` text).
/// - `selection_bg` / `selection_fg`: generic selection (settings rows…).
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct AppPalette {
    pub background: Rgb,
    pub surface: Rgb,
    pub surface_alt: Rgb,
    pub foreground: Rgb,
    pub muted: Rgb,
    pub border: Rgb,
    pub separator: Rgb,
    pub accent: Rgb,
    pub accent_alt: Rgb,
    pub selection_bg: Rgb,
    pub selection_fg: Rgb,
    pub menu_bg: Rgb,
    pub menu_selected_bg: Rgb,
    pub success: Rgb,
    pub warning: Rgb,
    pub error: Rgb,
    pub info: Rgb,
}

impl AppPalette {
    /// Rose Pine inspired: dark plum background, muted lavender/rose/cyan
    /// accent family. <https://rosepinetheme.com/palette>
    pub fn rose_pine() -> AppPalette {
        AppPalette {
            background: [0x19, 0x17, 0x24],   // base #191724
            surface: [0x1f, 0x1d, 0x2e],      // surface #1f1d2e
            surface_alt: [0x26, 0x23, 0x3a],  // overlay #26233a
            foreground: [0xe0, 0xde, 0xf4],   // text #e0def4
            muted: [0x6e, 0x6a, 0x86],        // muted #6e6a86
            border: [0x40, 0x3d, 0x52],       // highlightMed #403d52
            separator: [0x26, 0x23, 0x3a],    // overlay #26233a
            accent: [0xc4, 0xa7, 0xe7],       // iris #c4a7e7
            accent_alt: [0x9c, 0xcf, 0xd8],   // foam #9ccfd8
            selection_bg: [0x52, 0x4f, 0x67], // highlightHigh #524f67
            selection_fg: [0xe0, 0xde, 0xf4], // text #e0def4
            menu_bg: [0x12, 0x11, 0x1c],      // darker than background
            menu_selected_bg: [0x52, 0x4f, 0x67],
            success: [0x31, 0x74, 0x8f], // pine #31748f
            warning: [0xf6, 0xc1, 0x77], // gold #f6c177
            error: [0xeb, 0x6f, 0x92],   // love #eb6f92
            info: [0x9c, 0xcf, 0xd8],    // foam #9ccfd8
        }
    }

    /// Catppuccin Mocha inspired: dark base, mauve/blue accents.
    /// <https://catppuccin.com/palette>
    pub fn catppuccin() -> AppPalette {
        AppPalette {
            background: [0x1e, 0x1e, 0x2e],   // base #1e1e2e
            surface: [0x18, 0x18, 0x25],      // mantle #181825
            surface_alt: [0x31, 0x32, 0x44],  // surface0 #313244
            foreground: [0xcd, 0xd6, 0xf4],   // text #cdd6f4
            muted: [0xa6, 0xad, 0xc8],        // subtext0 #a6adc8
            border: [0x45, 0x47, 0x5a],       // surface1 #45475a
            separator: [0x31, 0x32, 0x44],    // surface0 #313244
            accent: [0xcb, 0xa6, 0xf7],       // mauve #cba6f7
            accent_alt: [0x89, 0xb4, 0xfa],   // blue #89b4fa
            selection_bg: [0x58, 0x5b, 0x70], // surface2 #585b70
            selection_fg: [0xcd, 0xd6, 0xf4], // text #cdd6f4
            menu_bg: [0x11, 0x11, 0x1b],      // crust #11111b
            menu_selected_bg: [0x45, 0x47, 0x5a],
            success: [0xa6, 0xe3, 0xa1], // green #a6e3a1
            warning: [0xf9, 0xe2, 0xaf], // yellow #f9e2af
            error: [0xf3, 0x8b, 0xa8],   // red #f38ba8
            info: [0x94, 0xe2, 0xd5],    // teal #94e2d5
        }
    }

    /// Tokyo Night inspired: deep navy background, cool blue/cyan accents.
    pub fn tokyo_night() -> AppPalette {
        AppPalette {
            background: [0x1a, 0x1b, 0x26],   // bg #1a1b26
            surface: [0x16, 0x16, 0x1e],      // bg_dark #16161e
            surface_alt: [0x29, 0x2e, 0x42],  // bg_highlight #292e42
            foreground: [0xc0, 0xca, 0xf5],   // fg #c0caf5
            muted: [0x56, 0x5f, 0x89],        // comment #565f89
            border: [0x3b, 0x42, 0x61],       // fg_gutter #3b4261
            separator: [0x29, 0x2e, 0x42],    // bg_highlight #292e42
            accent: [0x7a, 0xa2, 0xf7],       // blue #7aa2f7
            accent_alt: [0x7d, 0xcf, 0xff],   // cyan #7dcfff
            selection_bg: [0x28, 0x34, 0x57], // visual #283457
            selection_fg: [0xc0, 0xca, 0xf5], // fg #c0caf5
            menu_bg: [0x16, 0x16, 0x1e],      // bg_dark #16161e
            menu_selected_bg: [0x33, 0x46, 0x7c],
            success: [0x9e, 0xce, 0x6a], // green #9ece6a
            warning: [0xe0, 0xaf, 0x68], // yellow #e0af68
            error: [0xf7, 0x76, 0x8e],   // red #f7768e
            info: [0x7d, 0xcf, 0xff],    // cyan #7dcfff
        }
    }

    /// Gruvbox dark inspired: warm dark background, orange accents.
    pub fn gruvbox() -> AppPalette {
        AppPalette {
            background: [0x28, 0x28, 0x28],   // bg0 #282828
            surface: [0x3c, 0x38, 0x36],      // bg1 #3c3836
            surface_alt: [0x50, 0x49, 0x45],  // bg2 #504945
            foreground: [0xeb, 0xdb, 0xb2],   // fg1 #ebdbb2
            muted: [0x92, 0x83, 0x74],        // gray #928374
            border: [0x66, 0x5c, 0x54],       // bg3 #665c54
            separator: [0x50, 0x49, 0x45],    // bg2 #504945
            accent: [0xfe, 0x80, 0x19],       // orange #fe8019
            accent_alt: [0xfa, 0xbd, 0x2f],   // yellow #fabd2f
            selection_bg: [0x50, 0x49, 0x45], // bg2 #504945
            selection_fg: [0xfb, 0xf1, 0xc7], // fg0 #fbf1c7
            menu_bg: [0x1d, 0x20, 0x21],      // bg0_h #1d2021
            menu_selected_bg: [0x50, 0x49, 0x45],
            success: [0xb8, 0xbb, 0x26], // green #b8bb26
            warning: [0xfa, 0xbd, 0x2f], // yellow #fabd2f
            error: [0xfb, 0x49, 0x34],   // red #fb4934
            info: [0x83, 0xa5, 0x98],    // blue #83a598
        }
    }

    /// Fresh Custom palettes start from Rose Pine so every slot is sane
    /// before the user edits anything.
    pub fn custom_default() -> AppPalette {
        AppPalette::rose_pine()
    }

    /// Simple: the classic Hercules look — nordic canvas, white accent.
    pub fn simple() -> AppPalette {
        AppPalette {
            background: [0x2e, 0x34, 0x40],   // #2E3440 Polar Night
            surface: [0x3b, 0x42, 0x52],      // #3B4252
            surface_alt: [0x4c, 0x56, 0x6a],  // #4C566A
            foreground: [0xec, 0xef, 0xf4],   // #ECEFF4 Snow Storm
            muted: [0x81, 0xa1, 0xc1],        // #81A1C1 Frost Blue
            border: [0x4c, 0x56, 0x6a],       // #4C566A
            separator: [0x3b, 0x42, 0x52],    // #3B4252
            accent: [0xec, 0xef, 0xf4],       // white accent #ECEFF4
            accent_alt: [0x88, 0xc0, 0xd0],   // #88C0D0 Frost Cyan
            selection_bg: [0x4c, 0x56, 0x6a], // #4C566A
            selection_fg: [0xec, 0xef, 0xf4], // #ECEFF4
            menu_bg: [0x24, 0x29, 0x33],      // #242933 darker layer
            menu_selected_bg: [0x4c, 0x56, 0x6a],
            success: [0xa3, 0xbe, 0x8c], // #A3BE8C green
            warning: [0xeb, 0xcb, 0x8b], // #EBCB8B yellow
            error: [0xbf, 0x61, 0x6a],   // #BF616A red
            info: [0x88, 0xc0, 0xd0],    // #88C0D0 cyan
        }
    }

    fn color(c: Rgb) -> Color {
        Color::Rgb(c[0], c[1], c[2])
    }

    pub fn background_c(self) -> Color {
        Self::color(self.background)
    }
    pub fn surface_c(self) -> Color {
        Self::color(self.surface)
    }
    pub fn surface_alt_c(self) -> Color {
        Self::color(self.surface_alt)
    }
    pub fn foreground_c(self) -> Color {
        Self::color(self.foreground)
    }
    pub fn muted_c(self) -> Color {
        Self::color(self.muted)
    }
    pub fn border_c(self) -> Color {
        Self::color(self.border)
    }
    pub fn separator_c(self) -> Color {
        Self::color(self.separator)
    }
    pub fn accent_c(self) -> Color {
        Self::color(self.accent)
    }
    pub fn accent_alt_c(self) -> Color {
        Self::color(self.accent_alt)
    }
    pub fn selection_bg_c(self) -> Color {
        Self::color(self.selection_bg)
    }
    pub fn selection_fg_c(self) -> Color {
        Self::color(self.selection_fg)
    }
    pub fn menu_bg_c(self) -> Color {
        Self::color(self.menu_bg)
    }
    pub fn menu_selected_bg_c(self) -> Color {
        Self::color(self.menu_selected_bg)
    }
    pub fn success_c(self) -> Color {
        Self::color(self.success)
    }
    pub fn warning_c(self) -> Color {
        Self::color(self.warning)
    }
    pub fn error_c(self) -> Color {
        Self::color(self.error)
    }
    pub fn info_c(self) -> Color {
        Self::color(self.info)
    }

    /// Editable-slot access for the Custom editor.
    pub fn get_field(self, field: CustomField) -> Rgb {
        match field {
            CustomField::Background => self.background,
            CustomField::Surface => self.surface,
            CustomField::Foreground => self.foreground,
            CustomField::Muted => self.muted,
            CustomField::Accent => self.accent,
            CustomField::Selection => self.selection_bg,
            CustomField::MenuBackground => self.menu_bg,
            CustomField::MenuSelected => self.menu_selected_bg,
            CustomField::Border => self.border,
            CustomField::Success => self.success,
            CustomField::Warning => self.warning,
            CustomField::Error => self.error,
        }
    }

    pub fn set_field(&mut self, field: CustomField, rgb: Rgb) {
        match field {
            CustomField::Background => self.background = rgb,
            CustomField::Surface => self.surface = rgb,
            CustomField::Foreground => self.foreground = rgb,
            CustomField::Muted => self.muted = rgb,
            CustomField::Accent => self.accent = rgb,
            CustomField::Selection => self.selection_bg = rgb,
            CustomField::MenuBackground => self.menu_bg = rgb,
            CustomField::MenuSelected => self.menu_selected_bg = rgb,
            CustomField::Border => self.border = rgb,
            CustomField::Success => self.success = rgb,
            CustomField::Warning => self.warning = rgb,
            CustomField::Error => self.error = rgb,
        }
    }
}

/// The twelve user-editable Custom slots (exact labels pinned by test).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CustomField {
    Background,
    Surface,
    Foreground,
    Muted,
    Accent,
    Selection,
    MenuBackground,
    MenuSelected,
    Border,
    Success,
    Warning,
    Error,
}

impl CustomField {
    pub fn all() -> [CustomField; 12] {
        use CustomField::*;
        [
            Background,
            Surface,
            Foreground,
            Muted,
            Accent,
            Selection,
            MenuBackground,
            MenuSelected,
            Border,
            Success,
            Warning,
            Error,
        ]
    }

    pub fn label(self) -> &'static str {
        match self {
            CustomField::Background => "Background",
            CustomField::Surface => "Surface",
            CustomField::Foreground => "Foreground",
            CustomField::Muted => "Muted",
            CustomField::Accent => "Accent",
            CustomField::Selection => "Selection",
            CustomField::MenuBackground => "Menu Background",
            CustomField::MenuSelected => "Menu Selected Background",
            CustomField::Border => "Border",
            CustomField::Success => "Success",
            CustomField::Warning => "Warning",
            CustomField::Error => "Error",
        }
    }
}

/// Parse `#RRGGBB` or `RRGGBB` (case-insensitive, surrounding whitespace
/// tolerated). Anything else is rejected — the UI surfaces the error and
/// keeps the old color, so invalid input can never crash or corrupt.
pub fn parse_hex_color(s: &str) -> Result<Rgb, &'static str> {
    let t = s.trim().strip_prefix('#').unwrap_or_else(|| s.trim());
    if t.len() != 6 {
        return Err("expected #RRGGBB (6 hex digits)");
    }
    if !t.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err("expected #RRGGBB (6 hex digits)");
    }
    let v = |i: usize| u8::from_str_radix(&t[i..i + 2], 16).map_err(|_| "invalid hex");
    Ok([v(0)?, v(2)?, v(4)?])
}

/// Canonical `#RRGGBB` (uppercase) for display in the Custom editor.
pub fn to_hex_color(c: Rgb) -> String {
    format!("#{:02X}{:02X}{:02X}", c[0], c[1], c[2])
}

/// Relative luminance (0..1) — used to pin "menu darker than background".
pub fn luminance(c: Rgb) -> f32 {
    let f = |b: u8| {
        let v = b as f32 / 255.0;
        if v <= 0.03928 {
            v / 12.92
        } else {
            ((v + 0.055) / 1.055).powf(2.4)
        }
    };
    0.2126 * f(c[0]) + 0.7152 * f(c[1]) + 0.0722 * f(c[2])
}

/// Readable text over a saturated badge color: dark palette background on
/// light fills (e.g. the white Simple accent), light foreground on dark
/// fills. Keeps badge chips legible in every palette.
pub fn contrasting_text_on(bg: Rgb, pal: &AppPalette) -> Color {
    if luminance(bg) > 0.35 {
        pal.background_c()
    } else {
        pal.foreground_c()
    }
}

/// Current palette from live settings. Call sites use this (never a cached
/// copy) so switching palettes re-renders on the very next frame.
pub fn current_palette() -> AppPalette {
    crate::settings::get_color_palette().palette()
}
