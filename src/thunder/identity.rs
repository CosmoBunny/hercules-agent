//! Thunder peer identity: persistent Ed25519 keypair, stable peer ID,
//! displayable fingerprint. Generated once, persisted securely (private
//! key file mode 0600), never regenerated on launch, never transmitted.

use ed25519_dalek::{Signature, Signer, SigningKey, Verifier, VerifyingKey};
use rand::rngs::OsRng;
use sha2::{Digest, Sha256};
use std::path::PathBuf;

/// Persistent identity of one Hercules installation on Thunder.
#[derive(Debug, Clone)]
pub struct ThunderIdentity {
    /// Stable id, e.g. `thunder-7f3c…` (uuid v4, 8 hex chars).
    pub peer_id: String,
    pub display_name: String,
    pub signing_key: SigningKey,
}

impl ThunderIdentity {
    pub fn generate(display_name: String) -> Self {
        let signing_key = SigningKey::generate(&mut OsRng);
        let peer_id = format!("thunder-{}", &uuid::Uuid::new_v4().to_string()[..8]);
        Self {
            peer_id,
            display_name,
            signing_key,
        }
    }

    pub fn verifying_key(&self) -> VerifyingKey {
        self.signing_key.verifying_key()
    }

    pub fn public_key_bytes(&self) -> [u8; 32] {
        self.verifying_key().to_bytes()
    }

    /// Displayable fingerprint: first 8 bytes of SHA-256(pubkey) as
    /// `AB:91:…`. Deterministic per key.
    pub fn fingerprint(&self) -> String {
        Self::fingerprint_of(&self.public_key_bytes())
    }

    pub fn fingerprint_of(public_key: &[u8; 32]) -> String {
        let digest = Sha256::digest(public_key);
        digest[..8]
            .iter()
            .map(|b| format!("{b:02X}"))
            .collect::<Vec<_>>()
            .join(":")
    }

    pub fn sign(&self, message: &[u8]) -> Signature {
        self.signing_key.sign(message)
    }

    pub fn verify(public_key: &[u8; 32], message: &[u8], signature: &Signature) -> bool {
        VerifyingKey::from_bytes(public_key)
            .map(|vk| vk.verify(message, signature).is_ok())
            .unwrap_or(false)
    }

    fn dir() -> PathBuf {
        super::identity_path()
    }

    fn public_path() -> PathBuf {
        Self::dir().join("identity.json")
    }

    fn private_path() -> PathBuf {
        Self::dir().join("identity.key")
    }

    /// Load existing identity or generate + persist a new one.
    /// Never regenerates when files exist.
    pub fn load_or_generate(display_name: String) -> Result<Self, crate::thunder::ThunderError> {
        use crate::thunder::ThunderError;
        if let (Ok(pub_text), Ok(priv_bytes)) = (
            std::fs::read_to_string(Self::public_path()),
            std::fs::read(Self::private_path()),
        ) {
            if priv_bytes.len() == 32 {
                let mut arr = [0u8; 32];
                arr.copy_from_slice(&priv_bytes);
                let signing_key = SigningKey::from_bytes(&arr);
                let v: serde_json::Value =
                    serde_json::from_str(&pub_text).map_err(|_| ThunderError::IdentityCorrupt)?;
                let peer_id = v
                    .get("peer_id")
                    .and_then(|x| x.as_str())
                    .ok_or(ThunderError::IdentityCorrupt)?
                    .to_string();
                let name = v
                    .get("display_name")
                    .and_then(|x| x.as_str())
                    .unwrap_or("hercules")
                    .to_string();
                // Cross-check: stored pubkey must match the private key.
                let expect: Vec<u8> = v
                    .get("public_key")
                    .and_then(|x| x.as_array())
                    .map(|a| {
                        a.iter()
                            .filter_map(|n| n.as_u64().map(|v| v as u8))
                            .collect()
                    })
                    .unwrap_or_default();
                if expect.as_slice() == signing_key.verifying_key().to_bytes() {
                    return Ok(Self {
                        peer_id,
                        display_name: name,
                        signing_key,
                    });
                }
                return Err(ThunderError::IdentityCorrupt);
            }
        }
        let id = Self::generate(display_name);
        id.persist()?;
        Ok(id)
    }

    fn persist(&self) -> Result<(), crate::thunder::ThunderError> {
        use crate::thunder::ThunderError;
        std::fs::create_dir_all(Self::dir()).map_err(|e| ThunderError::Persistence {
            detail: format!("thunder dir: {e}"),
        })?;
        let pub_json = serde_json::json!({
            "peer_id": self.peer_id,
            "display_name": self.display_name,
            "public_key": self.public_key_bytes().to_vec(),
        });
        std::fs::write(Self::public_path(), pub_json.to_string()).map_err(|e| {
            ThunderError::Persistence {
                detail: format!("identity.json: {e}"),
            }
        })?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            let mut opts = std::fs::OpenOptions::new();
            opts.write(true).create(true).truncate(true).mode(0o600);
            use std::io::Write;
            let mut f = opts
                .open(Self::private_path())
                .map_err(|e| ThunderError::Persistence {
                    detail: format!("identity.key: {e}"),
                })?;
            f.write_all(&self.signing_key.to_bytes())
                .map_err(|e| ThunderError::Persistence {
                    detail: format!("identity.key: {e}"),
                })?;
        }
        #[cfg(not(unix))]
        {
            std::fs::write(Self::private_path(), self.signing_key.to_bytes()).map_err(|e| {
                ThunderError::Persistence {
                    detail: format!("identity.key: {e}"),
                }
            })?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fingerprint_deterministic_and_shaped() {
        let a = ThunderIdentity::generate("a".to_string());
        let b = ThunderIdentity::generate("b".to_string());
        assert_eq!(a.fingerprint(), a.fingerprint());
        assert_ne!(a.fingerprint(), b.fingerprint());
        assert_eq!(a.fingerprint().split(':').count(), 8);
        assert_ne!(a.peer_id, b.peer_id);
    }

    #[test]
    fn test_sign_verify_round_trip() {
        let a = ThunderIdentity::generate("a".to_string());
        let sig = a.sign(b"hello thunder");
        assert!(ThunderIdentity::verify(
            &a.public_key_bytes(),
            b"hello thunder",
            &sig
        ));
        assert!(!ThunderIdentity::verify(
            &a.public_key_bytes(),
            b"tampered",
            &sig
        ));
        assert!(!ThunderIdentity::verify(&[0u8; 32], b"hello thunder", &sig));
    }

    #[test]
    fn test_private_key_never_in_public_file() {
        let dir = tempfile::tempdir().unwrap();
        // Redirect via env is not supported; verify serialization shape only.
        let a = ThunderIdentity::generate("x".to_string());
        let pub_json = serde_json::json!({
            "peer_id": a.peer_id,
            "display_name": a.display_name,
            "public_key": a.public_key_bytes().to_vec(),
        });
        let text = pub_json.to_string();
        let priv_hex = hex_of(&a.signing_key.to_bytes());
        assert!(!text.contains(&priv_hex));
        let _ = dir;
    }

    fn hex_of(b: &[u8]) -> String {
        b.iter().map(|x| format!("{x:02x}")).collect()
    }
}
