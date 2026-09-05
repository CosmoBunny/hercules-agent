//! Hugging Face repository discovery: metadata-first, cheap.
//!
//! Separates MODEL DISCOVERY (file inventory + small JSON) from DOWNLOAD,
//! LOADING and GENERATION. Only explicitly required files are ever
//! fetched; repository/model names are never interpolated into shells
//! (URL path segments are percent-encoded, downloads stream to files).

use super::error::ModelError;
use super::manifest::{RepoFile, RepoManifest};

/// Files worth fetching as metadata (small JSON), never weights.
const METADATA_FILES: &[&str] = &[
    "config.json",
    "generation_config.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "model.safetensors.index.json",
];

fn encode_path(path: &str) -> String {
    // Percent-encoding for URL PATHS (slashes separate components).
    // Repository ids, revisions and filenames are validated separately
    // before reaching here; this never touches a shell.
    let mut out = String::with_capacity(path.len());
    for b in path.bytes() {
        if b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.' | b'/') {
            out.push(b as char);
        } else {
            out.push_str(&format!("%{b:02X}"));
        }
    }
    out
}

/// Centralized HF authentication: settings store (which already merges
/// `HF_TOKEN` env) — never scattered env reads across the resolver.
#[derive(Debug, Clone, Default)]
pub struct HfAuth {
    pub token: Option<String>,
}

impl HfAuth {
    pub fn from_settings() -> Self {
        Self {
            token: crate::settings::get_hf_token(),
        }
    }

    pub fn anonymous() -> Self {
        Self { token: None }
    }

    fn authed(&self, builder: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        match &self.token {
            Some(tok) if !tok.is_empty() => builder.bearer_auth(tok),
            _ => builder,
        }
    }
}

/// Validate a `org/name` repository id without shell or regex.
fn valid_repo_id(repo: &str) -> bool {
    let mut parts = repo.split('/');
    match (parts.next(), parts.next(), parts.next()) {
        (Some(org), Some(name), None) => {
            !org.is_empty()
                && !name.is_empty()
                && repo
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.' | b'/'))
        }
        _ => false,
    }
}

/// Revisions are branch/tag/commit names: no path separators allowed.
fn valid_revision(rev: &str) -> bool {
    !rev.is_empty()
        && rev
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.'))
}

/// Filenames must be relative in-repo paths: no leading `/`, no `..`.
fn valid_filename(name: &str) -> bool {
    !name.is_empty()
        && !name.starts_with('/')
        && !name.split('/').any(|seg| seg == "..")
        && name
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.' | b'/' | b' '))
}

/// One entry of the HF `/api/models/{repo}?blobs=true` siblings list.
#[derive(Debug, serde::Deserialize)]
struct Sibling {
    #[serde(default)]
    rfilename: String,
    #[serde(default)]
    size: Option<u64>,
}

#[derive(Debug, serde::Deserialize)]
struct RepoApi {
    #[serde(default)]
    siblings: Vec<Sibling>,
}

/// List repository files (names + sizes) via the HF API. Cheap: one JSON
/// response, no weights touched.
pub async fn list_repo_files(
    auth: &HfAuth,
    repo: &str,
    revision: &str,
) -> Result<Vec<RepoFile>, ModelError> {
    if !valid_repo_id(repo) {
        return Err(ModelError::RepositoryNotFound {
            repo: repo.to_string(),
        });
    }
    if !valid_revision(revision) {
        return Err(ModelError::FileUnavailable {
            repo: repo.to_string(),
            file: format!("invalid revision: {revision}"),
        });
    }
    let url = format!(
        "https://huggingface.co/api/models/{}?blobs=true&revision={}",
        encode_path(repo),
        encode_path(revision)
    );
    let client = reqwest::Client::new();
    let resp =
        auth.authed(client.get(&url))
            .send()
            .await
            .map_err(|e| ModelError::RepositoryNotFound {
                repo: format!("{repo}: {e}"),
            })?;
    if resp.status() == reqwest::StatusCode::NOT_FOUND {
        return Err(ModelError::RepositoryNotFound {
            repo: repo.to_string(),
        });
    }
    if resp.status() == reqwest::StatusCode::UNAUTHORIZED
        || resp.status() == reqwest::StatusCode::FORBIDDEN
    {
        return Err(ModelError::AuthenticationRequired {
            repo: repo.to_string(),
        });
    }
    // reqwest has no `json` feature enabled: parse the body manually.
    let body = resp.text().await.map_err(|e| ModelError::FileUnavailable {
        repo: repo.to_string(),
        file: format!("metadata listing: {e}"),
    })?;
    let api: RepoApi = serde_json::from_str(&body).map_err(|e| ModelError::FileUnavailable {
        repo: repo.to_string(),
        file: format!("metadata listing: {e}"),
    })?;
    Ok(api
        .siblings
        .into_iter()
        .map(|s| RepoFile {
            name: s.rfilename,
            size_bytes: s.size,
        })
        .collect())
}

/// Fetch one small metadata file. Returns None when absent (many repos
/// lack optional metadata); errors only on transport failure.
pub async fn fetch_metadata_file(
    auth: &HfAuth,
    repo: &str,
    revision: &str,
    filename: &str,
) -> Result<Option<String>, ModelError> {
    if !valid_filename(filename) {
        return Err(ModelError::FileUnavailable {
            repo: repo.to_string(),
            file: format!("invalid filename: {filename}"),
        });
    }
    let url = format!(
        "https://huggingface.co/{}/raw/{}/{}",
        encode_path(repo),
        encode_path(revision),
        encode_path(filename)
    );
    let client = reqwest::Client::new();
    let resp =
        auth.authed(client.get(&url))
            .send()
            .await
            .map_err(|e| ModelError::FileUnavailable {
                repo: repo.to_string(),
                file: format!("{filename}: {e}"),
            })?;
    if resp.status() == reqwest::StatusCode::NOT_FOUND {
        return Ok(None);
    }
    let text = resp.text().await.map_err(|e| ModelError::FileUnavailable {
        repo: repo.to_string(),
        file: format!("{filename}: {e}"),
    })?;
    Ok(Some(text))
}

/// Full metadata-first inspection: file list + config.json, no weights.
/// This is what the resolver decides on.
pub async fn inspect_repository(
    auth: &HfAuth,
    repo: &str,
    revision: &str,
) -> Result<RepoManifest, ModelError> {
    let files = list_repo_files(auth, repo, revision).await?;
    let mut config_json: Option<String> = None;
    for meta in METADATA_FILES {
        if files.iter().any(|f| f.name == *meta) {
            if let Some(text) = fetch_metadata_file(auth, repo, revision, meta).await? {
                if *meta == "config.json" {
                    config_json = Some(text);
                }
            }
        }
    }
    // Fallback: config.json may exist even when the siblings listing omits
    // sizes or entries (API variance across repos).
    if config_json.is_none() {
        config_json = fetch_metadata_file(auth, repo, revision, "config.json").await?;
    }
    Ok(RepoManifest::from_files(
        repo,
        revision,
        files,
        config_json.as_deref(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_repo_id_validation() {
        assert!(valid_repo_id("org/model-name"));
        assert!(valid_repo_id("mlx-community/Llama-3-8B-MLX"));
        assert!(!valid_repo_id("no-slash"));
        assert!(!valid_repo_id("a/b/c"));
        assert!(!valid_repo_id("org/model; rm -rf"));
        assert!(!valid_repo_id("org/model$(x)"));
    }

    #[test]
    fn test_path_encoding() {
        assert_eq!(encode_path("org/model"), "org/model");
        assert_eq!(encode_path("a b"), "a%20b");
    }

    #[test]
    fn test_revision_and_filename_validation() {
        assert!(valid_revision("main"));
        assert!(valid_revision("v1.0"));
        assert!(!valid_revision(""));
        assert!(!valid_revision("../evil"));
        assert!(!valid_revision("a/b"));
        assert!(valid_filename("config.json"));
        assert!(valid_filename("subdir/tokenizer.json"));
        assert!(!valid_filename("/abs/path"));
        assert!(!valid_filename("../escape"));
        assert!(!valid_filename(""));
    }

    #[test]
    fn test_auth_anonymous_by_default() {
        let a = HfAuth::anonymous();
        assert!(a.token.is_none());
    }
}
