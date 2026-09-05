//! Repository manifest: everything known about a model repo BEFORE any
//! gigabyte is downloaded. Discovery is metadata-first and cheap: file
//! inventory, parsed `config.json`, sizes. Loading/generation never start
//! from an uninspected repo.

use super::format::{LayoutSource, ModelFormat, ModelLayout};

/// One usable artifact family inside a repository: its weight files,
/// their layout, and their known byte cost. Files group by format AND
/// quant/stem, so `model-Q4_K_M.gguf` and `model-Q8_0.gguf` are separate
/// artifacts while `-00001-of-N` shards stay one artifact.
#[derive(Debug, Clone)]
pub struct Artifact {
    pub format: ModelFormat,
    pub files: Vec<String>,
    pub bytes: u64,
    pub layout: ModelLayout,
    pub layout_source: LayoutSource,
    pub sharded: bool,
}

impl Artifact {
    /// Stable identity: `gguf:model-q4_k_m`. Two quants never share an
    /// id; shards of one checkpoint always do. The resolver selects and
    /// the runtime factory receives this id — never a bare format.
    pub fn id(&self) -> String {
        format!(
            "{}:{}",
            self.format.label().to_lowercase(),
            artifact_group_key(self.files.first().map(|s| s.as_str()).unwrap_or("unknown"))
        )
    }
}

/// Group key: lowercase stem, extension and `-NNNNN-of-MMMMM` shard
/// suffix stripped. Shards collapse; distinct quants stay separate.
pub fn artifact_group_key(name: &str) -> String {
    let base = name.rsplit('/').next().unwrap_or(name).to_lowercase();
    let stem = match base.rfind('.') {
        Some(i) => &base[..i],
        None => &base,
    };
    // Strip multipart shard suffix: `-00001-of-00013`.
    let mut key = stem.to_string();
    if let Some(of_pos) = key.rfind("-of-") {
        let (head, _) = key.split_at(of_pos);
        if let Some(dash) = head.rfind('-') {
            let num = &head[dash + 1..];
            if !num.is_empty() && num.bytes().all(|b| b.is_ascii_digit()) {
                key = head[..dash].to_string();
            }
        }
    }
    key
}

/// One file in a repository (from the HF API `siblings` list).
#[derive(Debug, Clone)]
pub struct RepoFile {
    pub name: String,
    pub size_bytes: Option<u64>,
}

/// Parsed subset of HF `config.json`. Only the fields resolution needs;
/// everything else is ignored. All fields optional — metadata is
/// untrusted input and often incomplete.
#[derive(Debug, Clone, Default)]
pub struct ParsedConfig {
    pub architectures: Vec<String>,
    pub model_type: Option<String>,
    pub quantization: Option<String>,
    pub torch_dtype: Option<String>,
}

impl ParsedConfig {
    /// Parse from raw JSON text. Never fails hard: unknown shapes yield
    /// an empty config (detection continues with less information).
    pub fn parse(text: &str) -> Self {
        let v: serde_json::Value = match serde_json::from_str(text) {
            Ok(v) => v,
            Err(_) => return Self::default(),
        };
        let architectures = v
            .get("architectures")
            .and_then(|a| a.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|x| x.as_str().map(|s| s.to_string()))
                    .collect()
            })
            .unwrap_or_default();
        let model_type = v
            .get("model_type")
            .and_then(|x| x.as_str())
            .map(|s| s.to_string());
        // quantization_config is polymorphic across transformers versions:
        // accept {"quant_method": "..."} or a bare string.
        let quantization = v
            .get("quantization_config")
            .and_then(|q| {
                q.get("quant_method")
                    .and_then(|m| m.as_str())
                    .or_else(|| q.as_str())
                    .map(|s| s.to_string())
            })
            .or_else(|| {
                v.get("quantization")
                    .and_then(|q| q.as_str())
                    .map(|s| s.to_string())
            });
        let torch_dtype = v
            .get("torch_dtype")
            .and_then(|x| x.as_str())
            .map(|s| s.to_string());
        Self {
            architectures,
            model_type,
            quantization,
            torch_dtype,
        }
    }

    /// Primary architecture string, if any.
    pub fn primary_arch(&self) -> Option<&str> {
        self.architectures
            .first()
            .map(|s| s.as_str())
            .or(self.model_type.as_deref())
    }
}

/// Inspected repository: files + metadata + derived facts.
#[derive(Debug, Clone)]
pub struct RepoManifest {
    pub repo: String,
    pub revision: String,
    pub files: Vec<RepoFile>,
    pub config: ParsedConfig,
    /// Usable artifact families (GGUF and/or SafeTensors and/or PyTorch),
    /// each with its own files, layout verdict and byte cost.
    pub artifacts: Vec<Artifact>,
    pub tokenizer_present: bool,
}

impl RepoManifest {
    /// Build from a file list + optional raw `config.json` text.
    /// Pure and offline — the unit-testable core of discovery.
    pub fn from_files(
        repo: &str,
        revision: &str,
        files: Vec<RepoFile>,
        config_json: Option<&str>,
    ) -> Self {
        let names: Vec<String> = files.iter().map(|f| f.name.clone()).collect();
        let config = config_json.map(ParsedConfig::parse).unwrap_or_default();
        let lower_all: Vec<String> = names.iter().map(|f| f.to_lowercase()).collect();
        let repo_has_hf_meta = lower_all.iter().any(|f| {
            f.ends_with("config.json")
                || f.ends_with("tokenizer.json")
                || f.ends_with("tokenizer_config.json")
        });
        let tokenizer_present = lower_all.iter().any(|f| {
            f.ends_with("tokenizer.json")
                || f.ends_with("tokenizer_config.json")
                || f.ends_with("special_tokens_map.json")
        });
        let mut artifacts = Vec::new();
        // Group by (format, quant stem): distinct quants are separate
        // artifacts; `-00001-of-N` shards of one checkpoint stay together.
        let mut groups: Vec<(ModelFormat, String, Vec<&RepoFile>)> = Vec::new();
        for f in &files {
            let l = f.name.to_lowercase();
            let format = if l.ends_with(".gguf") {
                ModelFormat::Gguf
            } else if l.ends_with(".safetensors") {
                ModelFormat::SafeTensors
            } else if l.ends_with(".bin") || l.ends_with(".pt") || l.ends_with(".pth") {
                ModelFormat::PyTorch
            } else {
                continue;
            };
            let key = artifact_group_key(&f.name);
            match groups
                .iter_mut()
                .find(|(ff, k, _)| *ff == format && *k == key)
            {
                Some((_, _, members)) => members.push(f),
                None => groups.push((format, key, vec![f])),
            }
        }
        for (format, _key, members) in groups {
            let art_names: Vec<String> = members.iter().map(|f| f.name.clone()).collect();
            let art_bytes: u64 = members.iter().filter_map(|f| f.size_bytes).sum();
            // Layout is per artifact: a GGUF sidecar never inherits the
            // SafeTensors HF layout, and vice versa. Repo-wide metadata
            // presence is shared context for HF-family artifacts only —
            // GGUF is self-contained, so its layout stays Undetermined.
            let meta_ctx = repo_has_hf_meta && format != ModelFormat::Gguf;
            let (mut layout, mut layout_source) = ModelLayout::detect(repo, &art_names, meta_ctx);
            // Heuristic MLX verdicts stay candidate-grade unless the
            // metadata itself evidences MLX. diagnose() prints the qualifier.
            if layout == ModelLayout::Mlx
                && layout_source == LayoutSource::Heuristic
                && config_json
                    .map(ModelLayout::confirm_mlx_metadata)
                    .unwrap_or(false)
            {
                layout_source = LayoutSource::Metadata;
            }
            artifacts.push(Artifact {
                bytes: art_bytes,
                sharded: ModelFormat::is_sharded(&art_names, format),
                files: art_names,
                format,
                layout,
                layout_source,
            });
        }
        Self {
            repo: repo.to_string(),
            revision: revision.to_string(),
            files,
            config,
            artifacts,
            tokenizer_present,
        }
    }

    /// Detected weight formats across all artifacts.
    pub fn weight_formats(&self) -> Vec<ModelFormat> {
        self.artifacts.iter().map(|a| a.format).collect()
    }

    /// Total bytes across all weight artifacts (upper bound, not a plan:
    /// selection downloads exactly one artifact — see per-artifact bytes).
    pub fn weight_bytes(&self) -> u64 {
        self.artifacts.iter().map(|a| a.bytes).sum()
    }

    /// Primary weight format for single-artifact displays.
    pub fn primary_weights(&self) -> ModelFormat {
        self.artifacts
            .first()
            .map(|a| a.format)
            .unwrap_or(ModelFormat::Unknown)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn files(names: &[&str]) -> Vec<RepoFile> {
        names
            .iter()
            .map(|n| RepoFile {
                name: n.to_string(),
                size_bytes: Some(1000),
            })
            .collect()
    }

    #[test]
    fn test_manifest_parses_config_arch() {
        let m = RepoManifest::from_files(
            "org/model",
            "main",
            files(&["config.json", "model.safetensors", "tokenizer.json"]),
            Some(
                r#"{"architectures":["LlamaForCausalLM"],"model_type":"llama","torch_dtype":"float16"}"#,
            ),
        );
        assert_eq!(m.config.primary_arch(), Some("LlamaForCausalLM"));
        assert_eq!(m.config.torch_dtype.as_deref(), Some("float16"));
        assert!(m.tokenizer_present);
        assert!(m.weight_formats().contains(&ModelFormat::SafeTensors));
        let art = m
            .artifacts
            .iter()
            .find(|a| a.format == ModelFormat::SafeTensors)
            .unwrap();
        assert_eq!(art.layout, ModelLayout::StandardHf);
        assert_eq!(art.bytes, 1000);
        assert_eq!(m.weight_bytes(), 1000);
    }

    #[test]
    fn test_manifest_bad_config_never_fails() {
        let m = RepoManifest::from_files("o/m", "main", files(&["a.gguf"]), Some("not json{{"));
        assert_eq!(m.config.primary_arch(), None);
        assert_eq!(m.primary_weights(), ModelFormat::Gguf);
    }

    #[test]
    fn test_manifest_sharded_weights_sized() {
        let m = RepoManifest::from_files(
            "o/m",
            "main",
            vec![
                RepoFile {
                    name: "model-00001-of-00004.safetensors".into(),
                    size_bytes: Some(5_000),
                },
                RepoFile {
                    name: "model-00002-of-00004.safetensors".into(),
                    size_bytes: Some(5_000),
                },
                RepoFile {
                    name: "config.json".into(),
                    size_bytes: Some(100),
                },
            ],
            None,
        );
        assert!(
            m.artifacts
                .iter()
                .find(|a| a.format == ModelFormat::SafeTensors)
                .map(|a| a.sharded)
                .unwrap_or(false)
        );
        assert_eq!(m.weight_bytes(), 10_000);
    }

    #[test]
    fn test_manifest_multi_artifact_layouts_split() {
        // GGUF sidecar must NOT inherit the SafeTensors HF layout.
        let m = RepoManifest::from_files(
            "org/both",
            "main",
            files(&["config.json", "model.safetensors", "model-Q4_K_M.gguf"]),
            None,
        );
        assert_eq!(m.artifacts.len(), 2);
        let st = m
            .artifacts
            .iter()
            .find(|a| a.format == ModelFormat::SafeTensors)
            .unwrap();
        let gg = m
            .artifacts
            .iter()
            .find(|a| a.format == ModelFormat::Gguf)
            .unwrap();
        assert_eq!(st.layout, ModelLayout::StandardHf);
        assert_eq!(gg.layout, ModelLayout::Unknown);
    }

    #[test]
    fn test_manifest_mlx_layout() {
        let m = RepoManifest::from_files(
            "mlx-community/Llama-3-8B-MLX",
            "main",
            files(&["config.json", "model.safetensors"]),
            None,
        );
        // Name heuristic already routes to the MLX layout.
        let art = m
            .artifacts
            .iter()
            .find(|a| a.format == ModelFormat::SafeTensors)
            .unwrap();
        assert_eq!(art.layout, ModelLayout::Mlx);
        assert!(m.weight_formats().contains(&ModelFormat::SafeTensors));
    }

    #[test]
    fn test_manifest_mlx_heuristic_vs_confirmed() {
        // Without metadata evidence the verdict stays candidate-grade…
        let m = RepoManifest::from_files(
            "mlx-community/Llama-3-8B-MLX",
            "main",
            files(&["config.json", "model.safetensors"]),
            Some(r#"{"architectures":["LlamaForCausalLM"]}"#),
        );
        let art = m
            .artifacts
            .iter()
            .find(|a| a.format == ModelFormat::SafeTensors)
            .unwrap();
        assert_eq!(art.layout, ModelLayout::Mlx);
        assert_eq!(art.layout_source, LayoutSource::Heuristic);
        // …and config evidence upgrades it to confirmed.
        let m = RepoManifest::from_files(
            "mlx-community/Llama-3-8B-MLX",
            "main",
            files(&["config.json", "model.safetensors"]),
            Some(r#"{"architectures":["LlamaForCausalLM"],"mlx_version":"0.1"}"#),
        );
        let art = m
            .artifacts
            .iter()
            .find(|a| a.format == ModelFormat::SafeTensors)
            .unwrap();
        assert_eq!(art.layout_source, LayoutSource::Metadata);
    }

    #[test]
    fn test_artifact_group_key_splits_quants_keeps_shards() {
        assert_eq!(artifact_group_key("model-Q4_K_M.gguf"), "model-q4_k_m");
        assert_eq!(artifact_group_key("model-Q8_0.gguf"), "model-q8_0");
        assert_eq!(artifact_group_key("model-00001-of-00013.gguf"), "model");
        assert_eq!(artifact_group_key("MODEL-00002-OF-00013.GGUF"), "model");
    }

    #[test]
    fn test_two_gguf_quants_are_separate_artifacts_with_ids() {
        let m = RepoManifest::from_files(
            "org/quants",
            "main",
            files(&["model-Q4_K_M.gguf", "model-Q8_0.gguf"]),
            None,
        );
        assert_eq!(m.artifacts.len(), 2);
        let mut ids: Vec<String> = m.artifacts.iter().map(|a| a.id()).collect();
        ids.sort();
        assert_eq!(ids, vec!["gguf:model-q4_k_m", "gguf:model-q8_0"]);
    }

    #[test]
    fn test_shards_share_one_artifact_and_id() {
        let m = RepoManifest::from_files(
            "org/shard",
            "main",
            files(&["model-00001-of-00002.gguf", "model-00002-of-00002.gguf"]),
            None,
        );
        assert_eq!(m.artifacts.len(), 1);
        assert!(m.artifacts[0].sharded);
        assert_eq!(m.artifacts[0].id(), "gguf:model");
        assert_eq!(m.artifacts[0].files.len(), 2);
    }
}
