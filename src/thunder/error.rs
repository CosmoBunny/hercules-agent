//! Typed Thunder errors — never collapsed into generic strings.

/// All Shared Thunder failures.
#[derive(Debug, Clone)]
pub enum ThunderError {
    IdentityCorrupt,
    Persistence { detail: String },
    PairingFailed { detail: String },
    AuthenticationFailed,
    EncryptionFailed { detail: String },
    ProtocolMismatch { detail: String },
    PermissionDenied,
    ModelUnavailable { model: String },
    HostBusy,
    RequestTimeout,
    RequestCancelled,
    ConnectionLost,
    InvalidRequest { detail: String },
    ContextTooLarge,
    HopLimitExceeded,
    PeerUnavailable { peer: String },
}

impl ThunderError {
    pub fn message(&self) -> String {
        match self {
            Self::IdentityCorrupt => "Thunder identity is corrupt; re-pairing required".to_string(),
            Self::Persistence { detail } => format!("Thunder persistence failed: {detail}"),
            Self::PairingFailed { detail } => format!("Thunder pairing failed: {detail}"),
            Self::AuthenticationFailed => "Thunder peer authentication failed".to_string(),
            Self::EncryptionFailed { detail } => format!("Thunder encryption failed: {detail}"),
            Self::ProtocolMismatch { detail } => format!("Thunder protocol mismatch: {detail}"),
            Self::PermissionDenied => "Thunder peer denied the request".to_string(),
            Self::ModelUnavailable { model } => format!("Thunder model unavailable: {model}"),
            Self::HostBusy => "Thunder host is busy".to_string(),
            Self::RequestTimeout => "Thunder request timed out".to_string(),
            Self::RequestCancelled => "Thunder request cancelled".to_string(),
            Self::ConnectionLost => "Thunder connection lost".to_string(),
            Self::InvalidRequest { detail } => format!("Thunder invalid request: {detail}"),
            Self::ContextTooLarge => "Thunder context exceeds host limits".to_string(),
            Self::HopLimitExceeded => "Thunder hop limit exceeded".to_string(),
            Self::PeerUnavailable { peer } => format!("Shared Thunder host unavailable: {peer}"),
        }
    }
}

impl std::fmt::Display for ThunderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message())
    }
}

impl std::error::Error for ThunderError {}
