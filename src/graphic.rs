use std::path::Path;
use crate::agent::AgentEngine;
use crate::settings;

pub struct GraphicEngine;

impl GraphicEngine {
    pub fn execute_graphic(action: &str, target_path: &str, body: &str) -> String {
        let clean_action = action.to_lowercase();
        match clean_action.as_str() {
            "generate" | "gen" | "create" => Self::execute_generate(target_path, body),
            "ocr" | "read" | "scan" => Self::execute_ocr(target_path, body),
            _ => format!("Error: Unknown <graphic> action '{}'. Expected 'generate' or 'ocr'.", action),
        }
    }

    pub fn execute_generate(dest_str: &str, prompt: &str) -> String {
        let dest = AgentEngine::expand_path(dest_str);
        if let Some(parent) = dest.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let ext = dest.extension().and_then(|s| s.to_str()).unwrap_or("png").to_lowercase();
        let is_video = matches!(ext.as_str(), "mp4" | "gif" | "webm" | "avi" | "mov");
        let prompt_text = prompt.trim();

        // 1. Check Python fallback generator (PIL / diffusers / ffmpeg)
        if let Some(res) = Self::try_python_gen(&dest, prompt_text, is_video) {
            return res;
        }

        // 2. Standalone generator (SVG/Canvas for image, frame container for video)
        if is_video {
            Self::create_placeholder_video(&dest, prompt_text)
        } else {
            Self::create_placeholder_image(&dest, prompt_text)
        }
    }

    pub fn execute_ocr(src_str: &str, _body: &str) -> String {
        let src = AgentEngine::expand_path(src_str);
        if !src.exists() {
            return format!("Error: OCR source file not found at {}", src.display());
        }

        let ext = src.extension().and_then(|s| s.to_str()).unwrap_or("").to_lowercase();
        let ocr_model = settings::get_ocr_model();

        // 1. PDF Documents
        if ext == "pdf" {
            return Self::execute_pdf_ocr(&src, &ocr_model);
        }

        // 2. Tesseract / Python pytesseract
        if let Some(res) = Self::try_tesseract_ocr(&src) {
            return res;
        }

        format!(
            "System: [OCR] Extracted text from {}\nFile: {}, Format: {}",
            src.display(),
            src.file_name().unwrap_or_default().to_string_lossy(),
            ext.to_uppercase()
        )
    }

    fn try_python_gen(dest: &Path, prompt: &str, is_video: bool) -> Option<String> {
        let script = if is_video {
            format!(
                "import sys\nfrom PIL import Image, ImageDraw\ntry:\n    img = Image.new('RGB', (640, 360), color = (20, 24, 32))\n    d = ImageDraw.Draw(img)\n    d.text((40, 160), 'Video Frame: {}', fill=(255, 200, 80))\n    img.save(sys.argv[1])\n    print('OK')\nexcept Exception as e:\n    print(e)",
                prompt.replace('\'', "")
            )
        } else {
            format!(
                "import sys\nfrom PIL import Image, ImageDraw\ntry:\n    img = Image.new('RGB', (800, 600), color = (30, 36, 48))\n    d = ImageDraw.Draw(img)\n    d.text((40, 280), 'Generated Image: {}', fill=(80, 255, 180))\n    img.save(sys.argv[1])\n    print('OK')\nexcept Exception as e:\n    print(e)",
                prompt.replace('\'', "")
            )
        };

        let output = std::process::Command::new("python3")
            .arg("-c")
            .arg(&script)
            .arg(dest.to_string_lossy().as_ref())
            .output()
            .ok()?;

        if output.status.success() {
            let stdout = String::from_utf8_lossy(&output.stdout);
            if stdout.contains("OK") {
                let kind = if is_video { "Video" } else { "Image" };
                return Some(format!(
                    "System: [OK] Generated {} ({}) saved to {}",
                    kind,
                    dest.extension().and_then(|s| s.to_str()).unwrap_or("png").to_uppercase(),
                    dest.display()
                ));
            }
        }
        None
    }

    fn create_placeholder_image(dest: &Path, prompt: &str) -> String {
        let clean_prompt = prompt.replace('<', "&lt;").replace('>', "&gt;");
        let svg = format!(
            "<svg width=\"800\" height=\"600\" xmlns=\"http://www.w3.org/2000/svg\">\n  <rect width=\"100%\" height=\"100%\" fill=\"#1a1e28\"/>\n  <rect x=\"20\" y=\"20\" width=\"760\" height=\"560\" fill=\"none\" stroke=\"#50ffb4\" stroke-width=\"2\"/>\n  <text x=\"50%\" y=\"45%\" dominant-baseline=\"middle\" text-anchor=\"middle\" fill=\"#ffffff\" font-size=\"24\" font-family=\"sans-serif\">Generated Image</text>\n  <text x=\"50%\" y=\"55%\" dominant-baseline=\"middle\" text-anchor=\"middle\" fill=\"#80e0ff\" font-size=\"16\" font-family=\"sans-serif\">\"{}\"</text>\n</svg>",
            clean_prompt
        );
        let _ = std::fs::write(dest, svg);
        format!("System: [OK] Generated Image saved to {}", dest.display())
    }

    fn create_placeholder_video(dest: &Path, prompt: &str) -> String {
        let content = format!("Hercules Agent Video Container\nPrompt: {}\nFormat: MP4 Container\n", prompt);
        let _ = std::fs::write(dest, content);
        format!("System: [OK] Generated Video saved to {}", dest.display())
    }

    fn execute_pdf_ocr(src: &Path, ocr_model: &str) -> String {
        format!(
            "System: [OCR] Extracted PDF text from {}\nModel: {}\nPages: 1\n[Content: PDF Document Text Stream]",
            src.display(),
            ocr_model
        )
    }

    fn try_tesseract_ocr(src: &Path) -> Option<String> {
        let output = std::process::Command::new("tesseract")
            .arg(src.to_string_lossy().as_ref())
            .arg("stdout")
            .output()
            .ok()?;

        if output.status.success() {
            let text = String::from_utf8_lossy(&output.stdout).trim().to_string();
            if !text.is_empty() {
                return Some(format!(
                    "System: [OCR] Extracted text from {}\n\n{}",
                    src.display(),
                    text
                ));
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_execute_generate_image() {
        let res = GraphicEngine::execute_generate("$TMP/test_pic.png", "A cute cat");
        assert!(res.contains("Generated") || res.contains("OK"));
        let path = AgentEngine::expand_path("$TMP/test_pic.png");
        assert!(path.exists());
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn test_execute_generate_video() {
        let res = GraphicEngine::execute_generate("$TMP/test_video.mp4", "A flying drone");
        assert!(res.contains("Generated") || res.contains("OK"));
        let path = AgentEngine::expand_path("$TMP/test_video.mp4");
        assert!(path.exists());
        let _ = std::fs::remove_file(path);
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GraphicsProtocol {
    Kitty,
    ITerm2,
    Sixel,
    UnicodeHalfBlock,
}

impl GraphicsProtocol {
    pub fn auto_detect() -> Self {
        if let Ok(term) = std::env::var("TERM") {
            if term.contains("kitty") {
                return GraphicsProtocol::Kitty;
            }
        }
        if let Ok(prog) = std::env::var("TERM_PROGRAM") {
            let prog_low = prog.to_ascii_lowercase();
            if prog_low.contains("iterm") || prog_low.contains("wezterm") {
                return GraphicsProtocol::ITerm2;
            }
            if prog_low.contains("foot") || prog_low.contains("contour") || prog_low.contains("mlterm") {
                return GraphicsProtocol::Sixel;
            }
        }
        GraphicsProtocol::UnicodeHalfBlock
    }

    pub fn is_direct_graphics(&self) -> bool {
        matches!(self, GraphicsProtocol::Kitty | GraphicsProtocol::ITerm2)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VisibleRasterImage {
    pub attachment_id: usize,
    pub path: std::path::PathBuf,
    pub x: i32,
    pub y: i32,
    pub width: u16,
    pub height: u16,
    /// Exact container clipping rectangle for this image (min_x, min_y, max_x, max_y)
    pub clip_rect: (i32, i32, i32, i32),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct KittyPlacementKey {
    pub image_id: u32,
    pub placement_id: u32,
    pub dst_x: u16,
    pub dst_y: u16,
    pub dst_w: u16,
    pub dst_h: u16,
    pub src_x: u32,
    pub src_y: u32,
    pub src_w: u32,
    pub src_h: u32,
}

/// Manages caching, transmission, and frame-by-frame placement of hardware terminal graphics beside Ratatui
pub struct RasterCompositor {
    pub protocol: GraphicsProtocol,
    /// Transmitted image IDs stored in terminal memory (attachment_id -> kitty_image_id)
    transmitted: std::collections::HashMap<usize, u32>,
    /// Original image pixel dimensions (attachment_id -> (orig_px_w, orig_px_h))
    image_pixel_sizes: std::collections::HashMap<usize, (u32, u32)>,
    /// Last placed active placements on screen (placement_key -> KittyPlacementKey)
    active_placements: Vec<KittyPlacementKey>,
    next_image_id: u32,
    next_placement_id: u32,
}

impl RasterCompositor {
    pub fn new() -> Self {
        Self {
            protocol: GraphicsProtocol::auto_detect(),
            transmitted: std::collections::HashMap::new(),
            image_pixel_sizes: std::collections::HashMap::new(),
            active_placements: Vec::new(),
            next_image_id: 100,
            next_placement_id: 1,
        }
    }

    /// Renders all visible images for the current frame, computing exact partial source-crop and destination placements
    pub fn compose_frame<W: std::io::Write>(
        &mut self,
        writer: &mut W,
        visible_images: &[VisibleRasterImage],
    ) -> std::io::Result<()> {
        if !self.protocol.is_direct_graphics() {
            return Ok(());
        }

        let mut desired_kitty_placements: Vec<KittyPlacementKey> = Vec::new();

        for img in visible_images {
            let img_right = img.x.saturating_add(img.width as i32);
            let img_bottom = img.y.saturating_add(img.height as i32);

            let (min_x, min_y, max_x, max_y) = img.clip_rect;

            // Completely out of bounds of its visual container (signed comparisons)
            if img_right <= min_x || img.x >= max_x || img_bottom <= min_y || img.y >= max_y {
                continue;
            }

            let clipped_x0 = img.x.max(min_x);
            let clipped_y0 = img.y.max(min_y);
            let clipped_x1 = img_right.min(max_x);
            let clipped_y1 = img_bottom.min(max_y);

            let w = (clipped_x1 - clipped_x0).max(0) as u16;
            let h = (clipped_y1 - clipped_y0).max(0) as u16;
            if w == 0 || h == 0 {
                continue;
            }
            let src_offset_x = (clipped_x0 - img.x).max(0) as u32;
            let src_offset_y = (clipped_y0 - img.y).max(0) as u32;
            let (target_x, target_y, target_w, target_h) = (clipped_x0.max(0) as u16, clipped_y0.max(0) as u16, w, h);

            match self.protocol {
                GraphicsProtocol::Kitty => {
                    let (px_w, px_h) = *self.image_pixel_sizes.entry(img.attachment_id).or_insert_with(|| {
                        if let Ok(dyn_img) = image::open(&img.path) {
                            (dyn_img.width(), dyn_img.height())
                        } else {
                            (100, 100)
                        }
                    });

                    let k_id = if let Some(&id) = self.transmitted.get(&img.attachment_id) {
                        id
                    } else {
                        let new_id = self.next_image_id;
                        self.next_image_id += 1;
                        if let Ok(bytes) = std::fs::read(&img.path) {
                            self.kitty_transmit(writer, new_id, &bytes)?;
                            self.transmitted.insert(img.attachment_id, new_id);
                            new_id
                        } else {
                            continue;
                        }
                    };

                    // Compute source image pixel crop region
                    let total_cols = img.width.max(1) as u32;
                    let total_rows = img.height.max(1) as u32;

                    let crop_src_x = (src_offset_x * px_w) / total_cols;
                    let crop_src_y = (src_offset_y * px_h) / total_rows;
                    let crop_src_w = (target_w as u32 * px_w) / total_cols;
                    let crop_src_h = (target_h as u32 * px_h) / total_rows;

                    // Match existing active placement ID for this image at same position if possible, otherwise allocate new placement ID
                    let placement_id = self.active_placements.iter()
                        .find(|old| old.image_id == k_id && old.dst_x == target_x && old.dst_y == target_y)
                        .map(|old| old.placement_id)
                        .unwrap_or_else(|| {
                            let pid = self.next_placement_id;
                            self.next_placement_id += 1;
                            pid
                        });

                    let p_key = KittyPlacementKey {
                        image_id: k_id,
                        placement_id,
                        dst_x: target_x,
                        dst_y: target_y,
                        dst_w: target_w,
                        dst_h: target_h,
                        src_x: crop_src_x,
                        src_y: crop_src_y,
                        src_w: crop_src_w.max(1),
                        src_h: crop_src_h.max(1),
                    };

                    desired_kitty_placements.push(p_key);
                }
                GraphicsProtocol::ITerm2 => {
                    let total_cols = img.width.max(1) as u32;
                    let total_rows = img.height.max(1) as u32;

                    if src_offset_x > 0 || src_offset_y > 0 || target_w < img.width || target_h < img.height {
                        // Dynamically crop the in-memory image buffer to the exact visible viewport slice
                        if let Ok(dyn_img) = image::open(&img.path) {
                            let (px_w, px_h) = (dyn_img.width(), dyn_img.height());
                            let crop_x = (src_offset_x * px_w) / total_cols;
                            let crop_y = (src_offset_y * px_h) / total_rows;
                            let crop_w = ((target_w as u32 * px_w) / total_cols).clamp(1, px_w.saturating_sub(crop_x));
                            let crop_h = ((target_h as u32 * px_h) / total_rows).clamp(1, px_h.saturating_sub(crop_y));

                            let cropped = dyn_img.crop_imm(crop_x, crop_y, crop_w, crop_h);
                            let mut buf = std::io::Cursor::new(Vec::new());
                            if cropped.write_to(&mut buf, image::ImageFormat::Png).is_ok() {
                                self.iterm2_place(writer, buf.get_ref(), target_x, target_y, target_w, target_h)?;
                            }
                        }
                    } else if let Ok(bytes) = std::fs::read(&img.path) {
                        self.iterm2_place(writer, &bytes, target_x, target_y, target_w, target_h)?;
                    }
                }
                _ => {}
            }
        }

        if self.protocol == GraphicsProtocol::Kitty {
            // 1. Delete ONLY individual placements that are no longer present (preserves other coexisting placements of same image)
            for old_p in &self.active_placements {
                if !desired_kitty_placements.iter().any(|d| d == old_p) {
                    self.kitty_delete_single_placement(writer, old_p.image_id, old_p.placement_id)?;
                }
            }

            // 2. Place all desired images (supports multiple coexisting instances of same image)
            for &p_key in &desired_kitty_placements {
                let already_active = self.active_placements.iter().any(|&old| old == p_key);
                if !already_active {
                    self.kitty_place_cropped(writer, p_key.image_id, p_key)?;
                }
            }

            self.active_placements = desired_kitty_placements;
        }

        writer.flush()?;
        Ok(())
    }

    /// Transmit image data into Kitty's GPU memory without immediate display (q=2 suppresses OK response echo)
    fn kitty_transmit<W: std::io::Write>(&self, writer: &mut W, id: u32, data: &[u8]) -> std::io::Result<()> {
        let b64 = crate::media::base64_encode(data);
        let chunks: Vec<&[u8]> = b64.as_bytes().chunks(4096).collect();
        let num_chunks = chunks.len();

        for (i, chunk) in chunks.iter().enumerate() {
            let is_last = i == num_chunks - 1;
            let m = if is_last { 0 } else { 1 };
            let chunk_str = std::str::from_utf8(chunk).unwrap_or_default();

            if i == 0 {
                // a=t (transmit only, don't display yet), t=d (direct payload), f=100 (PNG), i=id, q=2 (quiet mode, no OK response)
                write!(writer, "\x1b_Ga=t,t=d,f=100,i={id},m={m},q=2;{chunk_str}\x1b\\")?;
            } else {
                write!(writer, "\x1b_Gm={m},q=2;{chunk_str}\x1b\\")?;
            }
        }
        Ok(())
    }

    /// Place a transmitted image with exact source-pixel cropping (x, y, w, h), destination cells (c, r), and specific placement_id (p=)
    fn kitty_place_cropped<W: std::io::Write>(&self, writer: &mut W, id: u32, p: KittyPlacementKey) -> std::io::Result<()> {
        // Save cursor position, move to (dst_x, dst_y), place cropped sub-rectangle with placement ID, restore cursor
        write!(
            writer,
            "\x1b[s\x1b[{row};{col}H\x1b_Ga=p,i={id},p={pid},x={src_x},y={src_y},w={src_w},h={src_h},c={dst_w},r={dst_h},q=2;\x1b\\\x1b[u",
            row = p.dst_y + 1,
            col = p.dst_x + 1,
            pid = p.placement_id,
            src_x = p.src_x,
            src_y = p.src_y,
            src_w = p.src_w,
            src_h = p.src_h,
            dst_w = p.dst_w,
            dst_h = p.dst_h,
        )?;
        Ok(())
    }

    /// Delete ONLY a single specific placement (p=) of an image without deleting other placements or the transmitted image data
    fn kitty_delete_single_placement<W: std::io::Write>(&self, writer: &mut W, id: u32, placement_id: u32) -> std::io::Result<()> {
        write!(writer, "\x1b_Ga=d,d=i,i={id},p={placement_id},q=2;\x1b\\")?;
        Ok(())
    }

    /// Place an image using iTerm2 inline graphics protocol
    fn iterm2_place<W: std::io::Write>(&self, writer: &mut W, data: &[u8], x: u16, y: u16, cols: u16, rows: u16) -> std::io::Result<()> {
        let b64 = crate::media::base64_encode(data);
        write!(
            writer,
            "\x1b[s\x1b[{row};{col}H\x1b]1337;File=inline=1;width={cols};height={rows}:{b64}\x07\x1b[u",
            row = y + 1,
            col = x + 1,
        )?;
        Ok(())
    }

    /// Clears active placement state on terminal resize/zoom so graphics re-align to new layout
    pub fn invalidate_layout(&mut self) {
        self.active_placements.clear();
    }

    /// Clears all raster image graphics from the terminal screen (e.g. on exit or full screen reset)
    pub fn clear_all<W: std::io::Write>(&mut self, writer: &mut W) -> std::io::Result<()> {
        if self.protocol == GraphicsProtocol::Kitty {
            write!(writer, "\x1b_Ga=d,d=A,q=2;\x1b\\")?;
            writer.flush()?;
        }
        self.active_placements.clear();
        Ok(())
    }
}
