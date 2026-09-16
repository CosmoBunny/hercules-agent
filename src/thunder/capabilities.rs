//! Host capability + model advertisement.
//!
//! A host explicitly shares selected models with explicit limits. Only
//! shared models are advertised; installation never implies sharing, and
//! no filesystem paths cross the wire (model identity, not location).

use super::error::ThunderError;

/// One remotely usable model: identity + capability description.
///
/// `context_length` is the host's EFFECTIVE serving limit for this
/// model (native backend limit, or the host's enforced default bound
/// when the backend reports none) — not necessarily the model's
/// native length. `architecture` is "unknown" when the host could not
/// source it; `quantization` is None when unreported (never guessed).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ThunderModel {
    /// Stable identity within the host, e.g. `qwen3-30b`.
    pub id: String,
    pub name: String,
    pub architecture: String,
    pub format: String,
    pub backend: String,
    pub quantization: Option<String>,
    pub context_length: u32,
    pub streaming: bool,
    pub cancellation: bool,
}

impl ThunderModel {
    pub fn validate(&self) -> Result<(), ThunderError> {
        if self.id.trim().is_empty() || self.name.trim().is_empty() {
            return Err(ThunderError::InvalidRequest {
                detail: "model id/name must not be empty".to_string(),
            });
        }
        if self
            .id
            .bytes()
            .any(|b| !(b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_')))
        {
            return Err(ThunderError::InvalidRequest {
                detail: format!("bad model id: {}", self.id),
            });
        }
        if self.context_length == 0 {
            return Err(ThunderError::InvalidRequest {
                detail: "context_length must be > 0".to_string(),
            });
        }
        Ok(())
    }
}

/// What a host offers: hardware + models + policy limits.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct HostCapabilities {
    pub peer_id: String,
    pub hardware: crate::model::HardwareInfo,
    pub models: Vec<ThunderModel>,
    pub inference: bool,
    pub forwarding: bool,
    pub max_concurrent_requests: u32,
    pub max_context_tokens: u32,
    pub max_new_tokens: u32,
    pub streaming: bool,
    pub cancellation: bool,
}

impl HostCapabilities {
    pub fn validate(&self) -> Result<(), ThunderError> {
        if self.max_concurrent_requests == 0 || self.max_concurrent_requests > 64 {
            return Err(ThunderError::InvalidRequest {
                detail: "max_concurrent_requests out of range 1..=64".to_string(),
            });
        }
        if self.models.len() > 64 {
            return Err(ThunderError::InvalidRequest {
                detail: "too many advertised models".to_string(),
            });
        }
        for m in &self.models {
            m.validate()?;
        }
        Ok(())
    }

    pub fn find_model(&self, id: &str) -> Option<&ThunderModel> {
        self.models.iter().find(|m| m.id == id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model(id: &str) -> ThunderModel {
        ThunderModel {
            id: id.to_string(),
            name: id.to_string(),
            architecture: "Qwen3".to_string(),
            format: "SafeTensors".to_string(),
            backend: "Transformers".to_string(),
            quantization: None,
            context_length: 32768,
            streaming: true,
            cancellation: true,
        }
    }

    #[test]
    fn test_model_validation_rejects_paths_and_empties() {
        assert!(model("qwen3-30b").validate().is_ok());
        let mut m = model("");
        assert!(m.validate().is_err());
        m = model("/home/alice/model.safetensors");
        assert!(m.validate().is_err());
        m = model("qwen3-30b");
        m.context_length = 0;
        assert!(m.validate().is_err());
    }
}
