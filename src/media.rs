use std::path::{Path, PathBuf};
use std::fs;
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone, PartialEq)]
pub enum MediaType {
    Image,
    Video,
    Document,
    Other,
}

impl MediaType {
    pub fn label(&self) -> &'static str {
        match self {
            MediaType::Image => "Image",
            MediaType::Video => "Video",
            MediaType::Document => "PDF/Doc",
            MediaType::Other => "File",
        }
    }

    pub fn from_extension(ext: &str) -> Self {
        let e = ext.to_ascii_lowercase();
        match e.as_str() {
            "png" | "jpg" | "jpeg" | "webp" | "svg" | "bmp" | "gif" | "tiff" | "ico" => {
                MediaType::Image
            }
            "mp4" | "mkv" | "webm" | "mov" | "avi" | "flv" | "wmv" | "m4v" => {
                MediaType::Video
            }
            "pdf" | "txt" | "md" | "json" | "csv" | "docx" | "epub" | "log" | "xml" | "toml" | "yaml" | "yml" => {
                MediaType::Document
            }
            _ => MediaType::Other,
        }
    }
}

#[derive(Debug, Clone)]
pub struct MediaAttachment {
    pub id: usize,
    pub original_name: String,
    pub staged_path: PathBuf,
    pub media_type: MediaType,
    pub size_bytes: u64,
    pub ocr_text: Option<String>,
    pub ocr_in_progress: bool,
}

pub fn media_staging_dir() -> PathBuf {
    PathBuf::from("/tmp/hercules/media")
}

/// Returns the persistent local session media directory:
/// e.g. `~/.local/share/hercules/sessions/{session_id}/media` or `~/.local/share/hercules/media/{session_id}`
pub fn session_media_dir(session_id: &str) -> PathBuf {
    crate::session::sessions_dir()
        .join(session_id)
        .join("media")
}

/// Returns the staging directory for media based on user configuration (Local session storage or /tmp)
pub fn media_staging_dir_for_session(session_id: Option<&str>) -> PathBuf {
    match crate::settings::get_media_storage_location() {
        crate::settings::MediaStorageLocation::Local => {
            if let Some(sid) = session_id {
                let dir = session_media_dir(sid);
                let _ = fs::create_dir_all(&dir);
                dir
            } else {
                let dir = crate::session::sessions_dir().join("default").join("media");
                let _ = fs::create_dir_all(&dir);
                dir
            }
        }
        crate::settings::MediaStorageLocation::Tmp => {
            let dir = if let Some(sid) = session_id {
                PathBuf::from(format!("/tmp/hercules/media/{}", sid))
            } else {
                PathBuf::from("/tmp/hercules/media")
            };
            let _ = fs::create_dir_all(&dir);
            dir
        }
    }
}

/// Deletes the local media storage folder for a given session ID
pub fn delete_session_media(session_id: &str) {
    let local_dir = session_media_dir(session_id);
    if local_dir.exists() {
        let _ = fs::remove_dir_all(&local_dir);
    }
    // Also remove parent session folder if empty
    if let Some(parent) = local_dir.parent() {
        if parent.exists() {
            let _ = fs::remove_dir(parent);
        }
    }
    let tmp_dir = PathBuf::from(format!("/tmp/hercules/media/{}", session_id));
    if tmp_dir.exists() {
        let _ = fs::remove_dir_all(&tmp_dir);
    }
}

/// Deletes entire session media storage across all sessions
pub fn delete_all_sessions_media() {
    let dir = crate::session::sessions_dir();
    if let Ok(entries) = fs::read_dir(&dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                let _ = fs::remove_dir_all(&path);
            }
        }
    }
    let tmp_dir = PathBuf::from("/tmp/hercules/media");
    if tmp_dir.exists() {
        let _ = fs::remove_dir_all(&tmp_dir);
    }
}

pub fn stage_image_bytes(bytes: &[u8], ext: &str) -> Result<MediaAttachment, String> {
    stage_image_bytes_for_session(bytes, ext, None)
}

pub fn stage_image_bytes_for_session(bytes: &[u8], ext: &str, session_id: Option<&str>) -> Result<MediaAttachment, String> {
    let dir = media_staging_dir_for_session(session_id);
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();
    let filename = format!("clip_{now}.{ext}");
    let path = dir.join(&filename);

    fs::write(&path, bytes).map_err(|e| format!("Failed to write clipboard media: {e}"))?;

    Ok(MediaAttachment {
        id: 0, // will be assigned by App state
        original_name: filename,
        staged_path: path,
        media_type: MediaType::Image,
        size_bytes: bytes.len() as u64,
        ocr_text: None,
        ocr_in_progress: false,
    })
}

pub fn stage_file_path(path: &Path) -> Result<MediaAttachment, String> {
    if !path.exists() {
        return Err(format!("File does not exist: {}", path.display()));
    }

    let meta = fs::metadata(path).map_err(|e| format!("Failed to read metadata: {e}"))?;
    let ext = path
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_string();
    let media_type = MediaType::from_extension(&ext);
    let original_name = path
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("file")
        .to_string();

    Ok(MediaAttachment {
        id: 0,
        original_name,
        staged_path: path.to_path_buf(),
        media_type,
        size_bytes: meta.len(),
        ocr_text: None,
        ocr_in_progress: false,
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TerminalGraphicsProtocol {
    Kitty,
    Sixel,
    ITerm2,
    UnicodeHalfBlock,
}

pub fn base64_encode(data: &[u8]) -> String {
    const TABLE: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity((data.len() + 2) / 3 * 4);
    for chunk in data.chunks(3) {
        let b0 = chunk[0];
        let b1 = chunk.get(1).copied().unwrap_or(0);
        let b2 = chunk.get(2).copied().unwrap_or(0);

        out.push(TABLE[(b0 >> 2) as usize] as char);
        out.push(TABLE[(((b0 & 0x03) << 4) | (b1 >> 4)) as usize] as char);

        if chunk.len() > 1 {
            out.push(TABLE[(((b1 & 0x0f) << 2) | (b2 >> 6)) as usize] as char);
        } else {
            out.push('=');
        }

        if chunk.len() > 2 {
            out.push(TABLE[(b2 & 0x3f) as usize] as char);
        } else {
            out.push('=');
        }
    }
    out
}

impl TerminalGraphicsProtocol {
    pub fn detect() -> Self {
        if let Ok(term) = std::env::var("TERM") {
            if term.contains("kitty") {
                return TerminalGraphicsProtocol::Kitty;
            }
        }
        if let Ok(prog) = std::env::var("TERM_PROGRAM") {
            let prog_low = prog.to_ascii_lowercase();
            if prog_low.contains("iterm") || prog_low.contains("wezterm") {
                return TerminalGraphicsProtocol::ITerm2;
            }
            if prog_low.contains("foot") || prog_low.contains("contour") || prog_low.contains("mlterm") {
                return TerminalGraphicsProtocol::Sixel;
            }
        }
        TerminalGraphicsProtocol::UnicodeHalfBlock
    }

    pub fn render_preview(&self, path: &Path, max_w: u32, max_h: u32) -> Result<String, String> {
        match self {
            TerminalGraphicsProtocol::Kitty => {
                let data = fs::read(path).map_err(|e| e.to_string())?;
                let b64 = base64_encode(&data);
                Ok(format!("\x1b_Ga=T,f=100;{b64}\x1b\\"))
            }
            TerminalGraphicsProtocol::ITerm2 => {
                let data = fs::read(path).map_err(|e| e.to_string())?;
                let b64 = base64_encode(&data);
                Ok(format!("\x1b]1337;File=inline=1;width={max_w};height={max_h}:{b64}\x07"))
            }
            _ => {
                Ok(format!("[Graphical Preview: {} ({}x{})]", path.display(), max_w, max_h))
            }
        }
    }
}

/// Opens a file in the user's default OS desktop application / image viewer (xdg-open / open / start).
pub fn open_with_system_viewer(path: &Path) -> Result<(), String> {
    if !path.exists() {
        return Err(format!("File does not exist: {}", path.display()));
    }
    #[cfg(target_os = "linux")]
    {
        std::process::Command::new("xdg-open")
            .arg(path)
            .spawn()
            .map_err(|e| format!("Failed to run xdg-open: {e}"))?;
        Ok(())
    }
    #[cfg(target_os = "macos")]
    {
        std::process::Command::new("open")
            .arg(path)
            .spawn()
            .map_err(|e| format!("Failed to run open: {e}"))?;
        Ok(())
    }
    #[cfg(target_os = "windows")]
    {
        std::process::Command::new("cmd")
            .args(["/C", "start", "", &path.to_string_lossy()])
            .spawn()
            .map_err(|e| format!("Failed to run start: {e}"))?;
        Ok(())
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
    {
        Err("Unsupported operating system for default viewer".to_string())
    }
}

/// Inspects pixel dimensions (width, height) of an image if possible using identify/chafa/python
pub fn get_image_dimensions(path: &Path) -> Option<(usize, usize)> {
    // 1. Try python PIL
    let script = "import sys\nfrom PIL import Image\ntry:\n    im = Image.open(sys.argv[1])\n    print(f'{im.width} {im.height}')\nexcept:\n    pass";
    if let Ok(out) = std::process::Command::new("python3")
        .arg("-c")
        .arg(script)
        .arg(path)
        .output()
    {
        if out.status.success() {
            let s = String::from_utf8_lossy(&out.stdout);
            let parts: Vec<&str> = s.split_whitespace().collect();
            if parts.len() == 2 {
                if let (Ok(w), Ok(h)) = (parts[0].parse::<usize>(), parts[1].parse::<usize>()) {
                    if w > 0 && h > 0 {
                        return Some((w, h));
                    }
                }
            }
        }
    }
    None
}

/// Parses raw ANSI escape strings (containing \x1b[38;2;R;G;Bm, \x1b[48;2;R;G;Bm, \x1b[0m, etc.) into styled Ratatui Spans.
pub fn parse_ansi_to_line(ansi_str: &str) -> ratatui::text::Line<'static> {
    use ratatui::style::{Color, Style, Modifier};
    let mut spans: Vec<ratatui::text::Span<'static>> = Vec::new();
    let mut cur_style = Style::default();
    let mut cur_text = String::new();

    let mut chars = ansi_str.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\x1b' {
            if chars.peek() == Some(&'[') {
                chars.next(); // consume '['
                let mut seq = String::new();
                while let Some(&next_c) = chars.peek() {
                    chars.next();
                    if next_c.is_ascii_alphabetic() {
                        seq.push(next_c);
                        break;
                    } else {
                        seq.push(next_c);
                    }
                }

                if !cur_text.is_empty() {
                    spans.push(ratatui::text::Span::styled(std::mem::take(&mut cur_text), cur_style));
                }

                // Parse CSI SGR parameters (e.g. 0, 38;2;r;g;b, 48;2;r;g;b)
                if seq.ends_with('m') {
                    let sgr = &seq[..seq.len() - 1];
                    if sgr == "0" || sgr.is_empty() {
                        cur_style = Style::default();
                    } else {
                        let parts: Vec<&str> = sgr.split(';').collect();
                        let mut i = 0;
                        while i < parts.len() {
                            match parts[i] {
                                "0" => { cur_style = Style::default(); i += 1; }
                                "1" => { cur_style = cur_style.add_modifier(Modifier::BOLD); i += 1; }
                                "38" if i + 4 < parts.len() && parts[i + 1] == "2" => {
                                    if let (Ok(r), Ok(g), Ok(b)) = (parts[i + 2].parse::<u8>(), parts[i + 3].parse::<u8>(), parts[i + 4].parse::<u8>()) {
                                        cur_style = cur_style.fg(Color::Rgb(r, g, b));
                                    }
                                    i += 5;
                                }
                                "48" if i + 4 < parts.len() && parts[i + 1] == "2" => {
                                    if let (Ok(r), Ok(g), Ok(b)) = (parts[i + 2].parse::<u8>(), parts[i + 3].parse::<u8>(), parts[i + 4].parse::<u8>()) {
                                        cur_style = cur_style.bg(Color::Rgb(r, g, b));
                                    }
                                    i += 5;
                                }
                                _ => { i += 1; }
                            }
                        }
                    }
                }
                continue;
            }
        }
        cur_text.push(c);
    }

    if !cur_text.is_empty() {
        spans.push(ratatui::text::Span::styled(cur_text, cur_style));
    }

    ratatui::text::Line::from(spans)
}

/// Generates a clipped ANSI/Unicode color block thumbnail representation of an image.
/// Uses the pure Rust `image` crate to decode and sample pixels directly into native Ratatui `Line`s
/// with upper half-block characters (`▀`) combining foreground (upper pixel) and background (lower pixel).
/// Calculates dynamic width based on the image's aspect ratio up to `max_h` (max 12 rows).
pub fn generate_image_thumbnail_lines(path: &Path, max_avail_w: usize, max_h: usize) -> (Vec<ratatui::text::Line<'static>>, usize) {
    use ratatui::style::{Color, Style};
    use ratatui::text::{Line, Span};

    let target_h = max_h.clamp(4, 12);

    if let Ok(dyn_img) = image::open(path) {
        let (orig_w, orig_h) = (dyn_img.width(), dyn_img.height());
        if orig_w > 0 && orig_h > 0 {
            // Character aspect ratio compensation (1 char height ~= 2 char widths in terminal fonts)
            let ratio = (orig_w as f32) / (orig_h as f32);
            let calc_w = (ratio * (target_h as f32) * 2.1).round() as usize;
            let target_w = calc_w.clamp(16, max_avail_w.min(76));

            // In half-block rendering, 1 terminal row covers 2 vertical pixels
            let pixel_h = (target_h * 2) as u32;
            let pixel_w = target_w as u32;

            let resized = dyn_img.resize_exact(pixel_w, pixel_h, image::imageops::FilterType::Triangle);
            let rgb = resized.to_rgb8();

            let mut lines = Vec::new();
            let mut actual_w = 0;

            for y in (0..pixel_h).step_by(2) {
                let mut spans = Vec::new();
                for x in 0..pixel_w {
                    let top_p = rgb.get_pixel(x, y);
                    let fg = Color::Rgb(top_p[0], top_p[1], top_p[2]);

                    if y + 1 < pixel_h {
                        let bot_p = rgb.get_pixel(x, y + 1);
                        let bg = Color::Rgb(bot_p[0], bot_p[1], bot_p[2]);
                        spans.push(Span::styled("▀", Style::default().fg(fg).bg(bg)));
                    } else {
                        spans.push(Span::styled("▀", Style::default().fg(fg)));
                    }
                }
                let line = Line::from(spans);
                actual_w = actual_w.max(line.width());
                lines.push(line);
            }

            if !lines.is_empty() {
                return (lines, actual_w.max(target_w));
            }
        }
    }

    // Fallback if image decode fails
    (vec![Line::from(format!("  [Thumbnail: {}]", path.file_name().unwrap_or_default().to_string_lossy()))], 30)
}

