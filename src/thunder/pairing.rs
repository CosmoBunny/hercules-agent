//! Explicit peer pairing: short code + fingerprint confirmation.
//!
//! Flow: host creates a time-boxed pairing code and shows it out of band.
//! Receiver connects with the code + its identity; host shows the
//! receiver fingerprint for Trust/Reject. Both sides persist TrustedPeer.
//! Unknown peers are never trusted silently.

use rand::{Rng, rngs::OsRng};
use std::collections::HashMap;
use std::path::PathBuf;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use super::error::ThunderError;

#[cfg(test)]
static STORE_TEST_SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Serialize tests that redirect process-global `XDG_DATA_HOME` for the
/// trust store. Shared across test modules (`app::thunder_tests`,
/// `thunder::runtime::tests`) so no two tests swap the variable
/// concurrently. Hold for the whole test body, starting BEFORE any
/// `set_var` call.
#[cfg(test)]
pub(crate) fn store_test_guard() -> std::sync::MutexGuard<'static, ()> {
    STORE_TEST_SERIAL.lock().unwrap_or_else(|e| e.into_inner())
}

/// A peer this installation trusts, with explicit permissions.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TrustedPeer {
    pub peer_id: String,
    pub public_key: [u8; 32],
    pub name: String,
    pub permissions: PeerPermissions,
    pub paired_at_epoch: u64,
}

/// What a trusted peer may do. Everything except inference defaults off.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PeerPermissions {
    pub inference: bool,
    pub forwarding: bool,
}

impl Default for PeerPermissions {
    fn default() -> Self {
        Self {
            inference: true,
            forwarding: false,
        }
    }
}

/// A short pairing code shown by the host (`7K4Q-92PX`). 40 bits of
/// entropy — brute force across a network round-trip per guess is
/// infeasible within its 10-minute lifetime.
#[derive(Debug, Clone)]
pub struct PairingCode {
    pub code: String,
    pub created: Instant,
    pub lifetime: Duration,
}

impl PairingCode {
    const ALPHABET: &'static [u8] = b"ABCDEFGHJKMNPQRSTUVWXYZ23456789";

    pub fn generate() -> Self {
        let mut rng = OsRng;
        let raw: String = (0..8)
            .map(|_| Self::ALPHABET[rng.gen_range(0..Self::ALPHABET.len())] as char)
            .collect();
        Self {
            code: format!("{}-{}", &raw[..4], &raw[4..]),
            created: Instant::now(),
            lifetime: Duration::from_secs(600),
        }
    }

    pub fn is_valid(&self) -> bool {
        self.created.elapsed() < self.lifetime
    }

    /// Constant-shape comparison (codes are short; timing is irrelevant
    /// next to network RTT, but avoid early-exit anyway).
    pub fn matches(&self, input: &str) -> bool {
        let norm = |s: &str| s.trim().to_uppercase().replace('-', "");
        let a = norm(&self.code);
        let b = norm(input);
        self.is_valid() && a.len() == b.len() && a == b
    }
}

/// Persistent trusted-peer store (`peers.json`).
///
/// Test isolation note: the store path derives from process-global
/// `XDG_DATA_HOME`, so tests that redirect it MUST serialize through
/// [`store_test_guard`]. Without the guard, parallel tests swap the
/// variable mid-render and read each other's trust stores (flake).
#[derive(Debug, Default, Clone)]
pub struct PeerStore {
    peers: HashMap<String, TrustedPeer>,
}

impl PeerStore {
    fn path() -> PathBuf {
        super::identity_path().join("peers.json")
    }

    pub fn load() -> Self {
        let mut store = Self::default();
        if let Ok(text) = std::fs::read_to_string(Self::path()) {
            if let Ok(list) = serde_json::from_str::<Vec<TrustedPeer>>(&text) {
                for p in list {
                    store.peers.insert(p.peer_id.clone(), p);
                }
            }
        }
        store
    }

    pub fn save(&self) -> Result<(), ThunderError> {
        let list: Vec<&TrustedPeer> = self.peers.values().collect();
        let text = serde_json::to_string_pretty(&list).map_err(|e| ThunderError::Persistence {
            detail: format!("peers.json: {e}"),
        })?;
        std::fs::write(Self::path(), text).map_err(|e| ThunderError::Persistence {
            detail: format!("peers.json: {e}"),
        })
    }

    /// Trust a peer after explicit user approval (fingerprint confirmed).
    /// Returns false — never stores — for unknown/unverified keys.
    pub fn trust(
        &mut self,
        peer_id: String,
        public_key: [u8; 32],
        name: String,
        permissions: PeerPermissions,
        verified: bool,
    ) -> bool {
        if !verified {
            return false;
        }
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        self.peers.insert(
            peer_id.clone(),
            TrustedPeer {
                peer_id,
                public_key,
                name,
                permissions,
                paired_at_epoch: now,
            },
        );
        true
    }

    pub fn reject(&mut self, peer_id: &str) -> bool {
        self.peers.remove(peer_id).is_some()
    }

    /// Adjust permissions for an already-trusted peer (test/admin helper).
    /// Unknown peers are never created by this call.
    pub fn set_permissions(&mut self, peer_id: &str, permissions: PeerPermissions) -> bool {
        match self.peers.get_mut(peer_id) {
            Some(p) => {
                p.permissions = permissions;
                true
            }
            None => false,
        }
    }

    pub fn get(&self, peer_id: &str) -> Option<&TrustedPeer> {
        self.peers.get(peer_id)
    }

    pub fn is_trusted(&self, peer_id: &str) -> bool {
        self.peers.contains_key(peer_id)
    }

    /// Read-only snapshot of all trusted peers, sorted by peer id
    /// (stable UI order). Additive accessor for UI listing — no
    /// behavior change to trust semantics.
    pub fn all(&self) -> Vec<&TrustedPeer> {
        let mut v: Vec<&TrustedPeer> = self.peers.values().collect();
        v.sort_by(|a, b| a.peer_id.cmp(&b.peer_id));
        v
    }

    /// Number of trusted peers (diagnostics/UI sizing).
    pub fn peers_len(&self) -> usize {
        self.peers.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pairing_code_shape_and_match() {
        let c = PairingCode::generate();
        assert_eq!(c.code.len(), 9);
        assert_eq!(&c.code[4..5], "-");
        assert!(c.matches(&c.code));
        assert!(c.matches(&c.code.to_lowercase().replace('-', "")));
        assert!(!c.matches("AAAA-0000"));
    }

    #[test]
    fn test_expired_code_rejected() {
        let mut c = PairingCode::generate();
        c.created = Instant::now() - Duration::from_secs(601);
        assert!(!c.is_valid());
        assert!(!c.matches(&c.code.clone()));
    }

    #[test]
    fn test_unverified_peer_never_trusted() {
        let mut store = PeerStore::default();
        assert!(!store.trust(
            "x".into(),
            [1u8; 32],
            "mallory".into(),
            PeerPermissions::default(),
            false
        ));
        assert!(!store.is_trusted("x"));
        assert!(store.trust(
            "x".into(),
            [1u8; 32],
            "alice".into(),
            PeerPermissions::default(),
            true
        ));
        assert!(store.is_trusted("x"));
        // Default permissions: inference on, forwarding off.
        assert!(store.get("x").unwrap().permissions.inference);
        assert!(!store.get("x").unwrap().permissions.forwarding);
    }
}
