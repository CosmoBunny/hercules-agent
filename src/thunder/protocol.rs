//! Versioned Thunder wire protocol.
//!
//! Every message carries protocol version + request id. Strict IDs like
//! the Transformers worker: a response for another request is rejected.
//! Hop fields (T13) are present but clamped to max_hops=1 until enabled.

use super::error::ThunderError;

pub const PROTOCOL_VERSION: u32 = 1;
/// Max decoded message: 4 MiB (prompts + context bounded elsewhere too).
pub const MAX_MESSAGE_BYTES: usize = 4 * 1024 * 1024;
/// Request IDs: `r` + digits only.
pub const MAX_HOPS: u32 = 1;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ThunderMessage {
    pub protocol_version: u32,
    pub request_id: String,
    #[serde(flatten)]
    pub payload: MessageKind,
    /// Multi-hop routing (T13): enforced, default single hop.
    #[serde(default)]
    pub origin_peer: Option<String>,
    #[serde(default)]
    pub hop_count: u32,
    #[serde(default)]
    pub max_hops: u32,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum MessageKind {
    Hello {
        peer_id: String,
        public_key: Vec<u8>,
        ephemeral: Vec<u8>,
        session_id: String,
        signature: Vec<u8>,
    },
    Authenticate {
        peer_id: String,
    },
    Pair {
        code: String,
        peer_id: String,
        name: String,
        public_key: Vec<u8>,
    },
    Capabilities {
        caps: super::capabilities::HostCapabilities,
    },
    Models {
        models: Vec<super::capabilities::ThunderModel>,
    },
    Ping {
        sent_epoch_ms: u64,
    },
    Generate {
        model_id: String,
        system_prompt: String,
        prompt: String,
        context: super::context::ContextEnvelope,
        max_new_tokens: u32,
        temperature: Option<f32>,
    },
    Token {
        text: String,
    },
    Done {
        full_text: String,
        /// T13: true when the answer came via a forwarded (multi-hop)
        /// request — receivers can attribute the real origin.
        forwarded: bool,
    },
    /// Request cancellation — the top-level request_id identifies the
    /// request (an inner id would duplicate the field on the wire).
    ///
    /// SEMANTICS: `Cancelled` means "cancellation ACCEPTED" — the host
    /// has signalled the worker — NOT "inference computation has
    /// definitely stopped". The underlying backend may still be
    /// unwinding after the confirmation is sent.
    Cancel,
    /// Cancellation acknowledgement: accepted, computation may still be
    /// unwinding (see `Cancel`).
    Cancelled,
    Error {
        code: String,
        message: String,
    },
    Shutdown,
}

impl ThunderMessage {
    pub fn new(request_id: impl Into<String>, payload: MessageKind) -> Self {
        Self {
            protocol_version: PROTOCOL_VERSION,
            request_id: request_id.into(),
            payload,
            origin_peer: None,
            hop_count: 0,
            max_hops: MAX_HOPS,
        }
    }

    pub fn validate(&self) -> Result<(), ThunderError> {
        if self.protocol_version != PROTOCOL_VERSION {
            return Err(ThunderError::ProtocolMismatch {
                detail: format!("version {}", self.protocol_version),
            });
        }
        if !valid_request_id(&self.request_id) {
            return Err(ThunderError::InvalidRequest {
                detail: "bad request_id".to_string(),
            });
        }
        if self.hop_count > self.max_hops || self.max_hops > MAX_HOPS {
            return Err(ThunderError::HopLimitExceeded);
        }
        match &self.payload {
            MessageKind::Generate {
                model_id,
                prompt,
                max_new_tokens,
                ..
            } => {
                if model_id.trim().is_empty() || model_id.len() > 128 {
                    return Err(ThunderError::InvalidRequest {
                        detail: "bad model_id".to_string(),
                    });
                }
                if prompt.len() > MAX_MESSAGE_BYTES / 2 {
                    return Err(ThunderError::ContextTooLarge);
                }
                if *max_new_tokens == 0 || *max_new_tokens > 131_072 {
                    return Err(ThunderError::InvalidRequest {
                        detail: "max_new_tokens out of range".to_string(),
                    });
                }
            }
            _ => {}
        }
        Ok(())
    }

    pub fn encode(&self) -> Result<Vec<u8>, ThunderError> {
        let bytes = serde_json::to_vec(self).map_err(|e| ThunderError::InvalidRequest {
            detail: format!("encode: {e}"),
        })?;
        if bytes.len() > MAX_MESSAGE_BYTES {
            return Err(ThunderError::ContextTooLarge);
        }
        Ok(bytes)
    }

    pub fn decode(bytes: &[u8]) -> Result<Self, ThunderError> {
        if bytes.len() > MAX_MESSAGE_BYTES {
            return Err(ThunderError::ContextTooLarge);
        }
        let msg: Self =
            serde_json::from_slice(bytes).map_err(|_| ThunderError::InvalidRequest {
                detail: "malformed message".to_string(),
            })?;
        msg.validate()?;
        Ok(msg)
    }
}

pub fn valid_request_id(id: &str) -> bool {
    let b = id.as_bytes();
    (2..=64).contains(&b.len()) && b[0] == b'r' && b[1..].iter().all(|c| c.is_ascii_digit())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_request_id_rules() {
        assert!(valid_request_id("r183"));
        assert!(!valid_request_id("183"));
        assert!(!valid_request_id("r"));
        assert!(!valid_request_id("r../x"));
    }

    #[test]
    fn test_version_and_hop_enforced() {
        let mut m = ThunderMessage::new("r1", MessageKind::Ping { sent_epoch_ms: 0 });
        m.protocol_version = 99;
        assert!(matches!(
            m.validate(),
            Err(ThunderError::ProtocolMismatch { .. })
        ));
        let mut m = ThunderMessage::new("r1", MessageKind::Ping { sent_epoch_ms: 0 });
        m.hop_count = 2;
        assert!(matches!(m.validate(), Err(ThunderError::HopLimitExceeded)));
    }

    #[test]
    fn test_generate_bounds() {
        let bad = ThunderMessage::new(
            "r1",
            MessageKind::Generate {
                model_id: "".to_string(),
                system_prompt: String::new(),
                prompt: "hi".to_string(),
                context: crate::thunder::context::ContextEnvelope::empty(),
                max_new_tokens: 64,
                temperature: None,
            },
        );
        assert!(matches!(
            bad.validate(),
            Err(ThunderError::InvalidRequest { .. })
        ));
        let bad = ThunderMessage::new(
            "r1",
            MessageKind::Generate {
                model_id: "m".to_string(),
                system_prompt: String::new(),
                prompt: "hi".to_string(),
                context: crate::thunder::context::ContextEnvelope::empty(),
                max_new_tokens: 0,
                temperature: None,
            },
        );
        assert!(matches!(
            bad.validate(),
            Err(ThunderError::InvalidRequest { .. })
        ));
    }

    #[test]
    fn test_unknown_message_rejected() {
        assert!(
            ThunderMessage::decode(
                br#"{"protocol_version":1,"request_id":"r1","type":"teleport"}"#
            )
            .is_err()
        );
        assert!(ThunderMessage::decode(b"{oops").is_err());
    }
}
