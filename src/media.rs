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
    let dir = PathBuf::from("/tmp/hercules/media");
    let _ = fs::create_dir_all(&dir);
    dir
}

pub fn stage_image_bytes(bytes: &[u8], ext: &str) -> Result<MediaAttachment, String> {
    let dir = media_staging_dir();
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
