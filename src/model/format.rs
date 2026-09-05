//! Explicit model formats: weight container vs. model layout.
//!
//! A weight format is a fact about stored artifacts, never a compatibility
//! verdict: SafeTensors does not imply Transformers-runnable, and GGUF
//! does not imply every llama.cpp architecture is supported. Those are
//! separate compatibility conditions (see `compatibility`).
//!
//! MLX is intentionally NOT a weight format: MLX models store SafeTensors
//! underneath. It is a [`ModelLayout`] — a runtime representation — so
//! `SafeTensors + MLX layout → MLX-LM` while
//! `SafeTensors + Standard HF layout → Transformers`.

/// Weight containers Hercules can inventory.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ModelFormat {
    /// GGUF single file or sharded (`-00001-of-N.gguf`).
    Gguf,
    /// `model.safetensors` or sharded (`model-00001-of-N.safetensors`,
    /// plus `model.safetensors.index.json`). Includes MLX repos, whose
    /// weights are SafeTensors — the MLX-ness is the layout, not the file.
    SafeTensors,
    /// Legacy PyTorch blobs (`*.bin`, `*.pt`, `*.pth`, `pytorch_model.bin`).
    PyTorch,
    /// None of the above were found.
    Unknown,
}

/// Model layout / runtime representation, orthogonal to weight bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ModelLayout {
    /// Standard Hugging Face Transformers layout (config + tokenizer +
    /// weights the Transformers loaders understand).
    StandardHf,
    /// Apple MLX layout (MLX-LM compatible). Detected from repository
    /// metadata/naming, never from weight extensions alone.
    Mlx,
    /// Could not be determined.
    Unknown,
}

/// How a layout verdict was reached. Naming heuristics are cheap but
/// fallible (`some-project-mlx-test` is not an MLX repo); only repository
/// metadata counts as confirmation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LayoutSource {
    /// Repo/file naming suggests MLX (`mlx-community` org, `-MLX` names).
    Heuristic,
    /// `config.json` metadata itself evidences the layout.
    Metadata,
    /// No evidence either way (GGUF-only repos, bare file lists).
    Undetermined,
}

impl ModelLayout {
    pub fn label(self) -> &'static str {
        match self {
            Self::StandardHf => "Standard-HF",
            Self::Mlx => "MLX",
            Self::Unknown => "Unknown",
        }
    }

    /// Candidate-grade detection from repository id + artifact files.
    /// Tightened on purpose: the `mlx` marker must be the model org, the
    /// repo-name suffix, or a full filename token — never an arbitrary
    /// substring of an unrelated name. `repo_has_hf_meta` carries
    /// repo-wide `config.json`/tokenizer presence, since per-artifact
    /// file groups exclude shared metadata files.
    pub fn detect(repo: &str, files: &[String], repo_has_hf_meta: bool) -> (Self, LayoutSource) {
        let repo_low = repo.to_lowercase();
        let org_is_mlx = repo_low.split('/').next() == Some("mlx-community");
        let name_is_mlx = repo_low
            .rsplit('/')
            .next()
            .map(|n| n.ends_with("-mlx") || n.ends_with("_mlx"))
            .unwrap_or(false);
        let file_token_is_mlx = files.iter().any(|f| {
            f.to_lowercase()
                .split(['/', '-', '_', '.'])
                .any(|tok| tok == "mlx")
                && !f.to_lowercase().ends_with("config.json")
        });
        if org_is_mlx || name_is_mlx || file_token_is_mlx {
            return (ModelLayout::Mlx, LayoutSource::Heuristic);
        }
        let lower: Vec<String> = files.iter().map(|f| f.to_lowercase()).collect();
        let has_hf_meta = repo_has_hf_meta
            || lower.iter().any(|f| {
                f.ends_with("config.json")
                    || f.ends_with("tokenizer.json")
                    || f.ends_with("tokenizer_config.json")
            });
        if has_hf_meta {
            (ModelLayout::StandardHf, LayoutSource::Metadata)
        } else {
            (ModelLayout::Unknown, LayoutSource::Undetermined)
        }
    }

    /// Metadata confirmation: raw `config.json` evidences an MLX layout
    /// only through actual JSON structure — an `mlx` token in a KEY at any
    /// depth (e.g. `"mlx_version"`, `"mlx_lm"`). A prose *value* merely
    /// mentioning MLX ("compatible with MLX") does NOT confirm.
    pub fn confirm_mlx_metadata(config_json: &str) -> bool {
        let v: serde_json::Value = match serde_json::from_str(config_json) {
            Ok(v) => v,
            Err(_) => return false,
        };
        fn keys_mention_mlx(v: &serde_json::Value) -> bool {
            match v {
                serde_json::Value::Object(map) => map
                    .iter()
                    .any(|(k, val)| k.to_lowercase().contains("mlx") || keys_mention_mlx(val)),
                serde_json::Value::Array(arr) => arr.iter().any(keys_mention_mlx),
                _ => false,
            }
        }
        keys_mention_mlx(&v)
    }
}

impl ModelFormat {
    pub fn label(self) -> &'static str {
        match self {
            Self::Gguf => "GGUF",
            Self::SafeTensors => "SafeTensors",
            Self::PyTorch => "PyTorch",
            Self::Unknown => "Unknown",
        }
    }

    /// Inventory which formats a repository file list contains.
    /// Pure filename inventory — compatibility is decided elsewhere.
    pub fn detect_in_files(files: &[String]) -> Vec<ModelFormat> {
        let mut out = Vec::new();
        let lower: Vec<String> = files.iter().map(|f| f.to_lowercase()).collect();
        let any = |suffixes: &[&str]| {
            lower
                .iter()
                .any(|f| suffixes.iter().any(|s| f.ends_with(s)))
        };
        if any(&[".gguf"]) {
            out.push(ModelFormat::Gguf);
        }
        if any(&[".safetensors"]) {
            out.push(ModelFormat::SafeTensors);
        }
        if any(&[".bin", ".pt", ".pth"]) {
            out.push(ModelFormat::PyTorch);
        }
        if out.is_empty() {
            out.push(ModelFormat::Unknown);
        }
        out
    }

    /// True when the list holds a sharded variant of this format.
    pub fn is_sharded(files: &[String], format: ModelFormat) -> bool {
        let lower: Vec<String> = files.iter().map(|f| f.to_lowercase()).collect();
        match format {
            ModelFormat::Gguf => lower
                .iter()
                .any(|f| f.ends_with(".gguf") && f.contains("-of-")),
            ModelFormat::SafeTensors => lower.iter().any(|f| {
                (f.ends_with(".safetensors") && f.contains("-of-"))
                    || f.ends_with("model.safetensors.index.json")
            }),
            _ => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gguf_detection() {
        let files = vec!["model-Q4_K_M.gguf".to_string()];
        assert_eq!(
            ModelFormat::detect_in_files(&files),
            vec![ModelFormat::Gguf]
        );
        assert!(!ModelFormat::is_sharded(&files, ModelFormat::Gguf));
    }

    #[test]
    fn test_sharded_gguf_detection() {
        let files = vec!["model-00001-of-00013.gguf".to_string()];
        let found = ModelFormat::detect_in_files(&files);
        assert!(found.contains(&ModelFormat::Gguf));
        assert!(ModelFormat::is_sharded(&files, ModelFormat::Gguf));
    }

    #[test]
    fn test_safetensors_detection() {
        let files = vec!["model.safetensors".to_string(), "config.json".to_string()];
        assert!(ModelFormat::detect_in_files(&files).contains(&ModelFormat::SafeTensors));
    }

    #[test]
    fn test_sharded_safetensors_detection() {
        let files = vec![
            "model-00001-of-00004.safetensors".to_string(),
            "model.safetensors.index.json".to_string(),
        ];
        assert!(ModelFormat::is_sharded(&files, ModelFormat::SafeTensors));
    }

    #[test]
    fn test_pytorch_and_unknown_detection() {
        let files = vec!["pytorch_model.bin".to_string()];
        assert!(ModelFormat::detect_in_files(&files).contains(&ModelFormat::PyTorch));
        let files = vec!["README.md".to_string()];
        assert_eq!(
            ModelFormat::detect_in_files(&files),
            vec![ModelFormat::Unknown]
        );
    }
}
