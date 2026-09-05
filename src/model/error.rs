//! Structured model errors — typed, never string-matched.
//!
//! Every failure in discovery/download/load/generate maps to one variant
//! with the underlying detail preserved, so diagnostics and UI can tell
//! "unsupported architecture" apart from "download failed".

/// All model-pipeline failures.
#[derive(Debug, Clone)]
pub enum ModelError {
    RepositoryNotFound {
        repo: String,
    },
    AuthenticationRequired {
        repo: String,
    },
    FileUnavailable {
        repo: String,
        file: String,
    },
    ArchitectureUnsupported {
        architecture: String,
        backend: String,
    },
    FormatUnsupported {
        format: String,
        backend: String,
    },
    BackendUnavailable {
        backend: String,
        reason: String,
    },
    DependencyMissing {
        backend: String,
        dependency: String,
    },
    InsufficientMemory {
        needed_bytes: u64,
        available_bytes: u64,
    },
    LoadFailed {
        backend: String,
        detail: String,
    },
    GenerationFailed {
        backend: String,
        detail: String,
    },
    Cancelled,
}

impl ModelError {
    /// One-line user-facing message.
    pub fn message(&self) -> String {
        match self {
            Self::RepositoryNotFound { repo } => format!("Model repository not found: {repo}"),
            Self::AuthenticationRequired { repo } => {
                format!("Repository requires authentication: {repo}")
            }
            Self::FileUnavailable { repo, file } => {
                format!("File unavailable: {file} in {repo}")
            }
            Self::ArchitectureUnsupported {
                architecture,
                backend,
            } => {
                format!("{backend} does not support architecture {architecture}")
            }
            Self::FormatUnsupported { format, backend } => {
                format!("{backend} does not run {format} weights")
            }
            Self::BackendUnavailable { backend, reason } => {
                format!("{backend} is unavailable: {reason}")
            }
            Self::DependencyMissing {
                backend,
                dependency,
            } => {
                format!("{backend} needs missing dependency: {dependency}")
            }
            Self::InsufficientMemory {
                needed_bytes,
                available_bytes,
            } => format!("Insufficient memory: need {needed_bytes} bytes, have {available_bytes}"),
            Self::LoadFailed { backend, detail } => {
                format!("{backend} failed to load the model: {detail}")
            }
            Self::GenerationFailed { backend, detail } => {
                format!("{backend} generation failed: {detail}")
            }
            Self::Cancelled => "Model operation was cancelled".to_string(),
        }
    }
}

impl std::fmt::Display for ModelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message())
    }
}

impl std::error::Error for ModelError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_messages_are_specific() {
        let e = ModelError::ArchitectureUnsupported {
            architecture: "X".into(),
            backend: "llama.cpp".into(),
        };
        assert!(e.message().contains("architecture"));
        assert!(!e.message().contains("download"));
        let e = ModelError::Cancelled;
        assert_eq!(e.message(), "Model operation was cancelled");
    }
}
