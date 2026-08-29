use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum OcrEngineMode {
    Auto,
    Tesseract,
    Native,
    Pdftotext,
}

impl OcrEngineMode {
    pub fn label(&self) -> &'static str {
        match self {
            OcrEngineMode::Auto => "Auto (Fastest / Best)",
            OcrEngineMode::Tesseract => "Tesseract (System)",
            OcrEngineMode::Native => "Native (Built-in)",
            OcrEngineMode::Pdftotext => "pdftotext (Documents)",
        }
    }
}

pub struct OcrService;

impl OcrService {
    /// Extract text asynchronously from an image, PDF, or video file.
    pub async fn extract_text(path: &Path, mode: OcrEngineMode) -> Result<String, String> {
        let ext = path
            .extension()
            .and_then(|s| s.to_str())
            .unwrap_or("")
            .to_ascii_lowercase();

        match ext.as_str() {
            "pdf" => Self::extract_pdf(path).await,
            "mp4" | "mkv" | "webm" | "mov" | "avi" => Self::extract_video(path).await,
            _ => Self::extract_image(path, mode).await,
        }
    }

    /// Extract text from images using tesseract CLI or fallbacks.
    pub async fn extract_image(path: &Path, _mode: OcrEngineMode) -> Result<String, String> {
        let p_str = path.to_string_lossy().to_string();

        // 1. Try tesseract CLI (standard, reliable across Linux distributions)
        let output = tokio::process::Command::new("tesseract")
            .arg(&p_str)
            .arg("stdout")
            .arg("-l")
            .arg("eng")
            .output()
            .await;

        if let Ok(out) = output {
            if out.status.success() {
                let text = String::from_utf8_lossy(&out.stdout).trim().to_string();
                if !text.is_empty() {
                    return Ok(text);
                }
            }
        }

        // 2. Try ocrmypdf if available
        let output_ocr = tokio::process::Command::new("gocr")
            .arg("-i")
            .arg(&p_str)
            .output()
            .await;

        if let Ok(out) = output_ocr {
            if out.status.success() {
                let text = String::from_utf8_lossy(&out.stdout).trim().to_string();
                if !text.is_empty() {
                    return Ok(text);
                }
            }
        }

        Err(format!(
            "OCR: No readable text found in {} or tesseract not installed.",
            path.display()
        ))
    }

    /// Extract vector text from PDF or OCR if scanned.
    pub async fn extract_pdf(path: &Path) -> Result<String, String> {
        let p_str = path.to_string_lossy().to_string();

        // Try pdftotext (poppler-utils)
        let output = tokio::process::Command::new("pdftotext")
            .arg(&p_str)
            .arg("-")
            .output()
            .await;

        if let Ok(out) = output {
            if out.status.success() {
                let text = String::from_utf8_lossy(&out.stdout).trim().to_string();
                if !text.is_empty() {
                    return Ok(text);
                }
            }
        }

        // Fallback: convert first page to image via pdftoppm then OCR
        let tmp_img = format!("/tmp/hercules_pdf_page_{}", std::process::id());
        let _ = tokio::process::Command::new("pdftoppm")
            .arg("-png")
            .arg("-f")
            .arg("1")
            .arg("-l")
            .arg("1")
            .arg(&p_str)
            .arg(&tmp_img)
            .output()
            .await;

        let rendered_str = format!("{tmp_img}-1.png");
        let rendered = Path::new(&rendered_str);
        if rendered.exists() {
            let res = Self::extract_image(rendered, OcrEngineMode::Auto).await;
            let _ = std::fs::remove_file(rendered);
            return res;
        }

        Err(format!("Could not extract text from PDF: {}", path.display()))
    }

    /// Extract text from video by taking a screenshot frame via ffmpeg and running OCR.
    pub async fn extract_video(path: &Path) -> Result<String, String> {
        let p_str = path.to_string_lossy().to_string();
        let tmp_frame = format!("/tmp/hercules_vid_frame_{}.png", std::process::id());

        // Grab keyframe at 1.0 second mark
        let output = tokio::process::Command::new("ffmpeg")
            .arg("-y")
            .arg("-ss")
            .arg("00:00:01")
            .arg("-i")
            .arg(&p_str)
            .arg("-vframes")
            .arg("1")
            .arg("-q:v")
            .arg("2")
            .arg(&tmp_frame)
            .output()
            .await;

        if let Ok(out) = output {
            if out.status.success() && Path::new(&tmp_frame).exists() {
                let res = Self::extract_image(Path::new(&tmp_frame), OcrEngineMode::Auto).await;
                let _ = std::fs::remove_file(&tmp_frame);
                return res;
            }
        }

        Err(format!("Could not extract video keyframe from: {}", path.display()))
    }
}
