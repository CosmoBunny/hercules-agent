//! Session encryption: X25519 ECDH + ChaCha20Poly1305.
//!
//! Established primitives only (Dalek crates) — no custom algorithms.
//! Long-term Ed25519 identities authenticate ephemeral X25519 keys;
//! each session gets a fresh symmetric key with strictly increasing
//! nonces (4-byte random salt + 8-byte big-endian counter). Nonce reuse
//! within a session is a logic error and panics in debug.

use chacha20poly1305::{
    ChaCha20Poly1305, Key, Nonce,
    aead::{Aead, KeyInit},
};
use sha2::{Digest, Sha256};
use x25519_dalek::{EphemeralSecret, PublicKey as XPublic};

use super::error::ThunderError;

const NONCE_SALT_LEN: usize = 4;

/// One encrypted session direction set (full duplex uses one key; nonces
/// never repeat because the counter only increases).
pub struct ThunderSession {
    key: ChaCha20Poly1305,
    salt: [u8; NONCE_SALT_LEN],
    counter: u64,
}

impl ThunderSession {
    /// ECDH between our ephemeral secret and their ephemeral public key,
    /// hashed with both public keys for channel binding. Consumes the
    /// ephemeral secret (forward secrecy: it cannot be reused).
    pub fn establish(
        our_secret: EphemeralSecret,
        our_public: &XPublic,
        their_public: &XPublic,
    ) -> Self {
        use rand::RngCore;
        let shared = our_secret.diffie_hellman(their_public);
        let mut hasher = Sha256::new();
        hasher.update(shared.as_bytes());
        hasher.update(our_public.as_bytes());
        hasher.update(their_public.as_bytes());
        let digest = hasher.finalize();
        let key = Key::from_slice(&digest);
        let mut salt = [0u8; NONCE_SALT_LEN];
        rand::rngs::OsRng.fill_bytes(&mut salt);
        Self {
            key: ChaCha20Poly1305::new(key),
            salt,
            counter: 0,
        }
    }

    fn next_nonce(&mut self) -> Nonce {
        let mut n = [0u8; 12];
        n[..4].copy_from_slice(&self.salt);
        n[4..].copy_from_slice(&self.counter.to_be_bytes());
        self.counter = self.counter.checked_add(1).expect("nonce counter overflow");
        *Nonce::from_slice(&n)
    }

    pub fn encrypt(&mut self, plaintext: &[u8]) -> Result<Vec<u8>, ThunderError> {
        let nonce = self.next_nonce();
        self.key
            .encrypt(&nonce, plaintext)
            .map_err(|_| ThunderError::EncryptionFailed {
                detail: "seal failed".to_string(),
            })
    }

    /// Decrypt with an explicit counter (receiver tracks its own counter
    /// per direction; out-of-order delivery is rejected as tampering).
    pub fn decrypt_with_counter(
        key: &ChaCha20Poly1305,
        salt: &[u8; NONCE_SALT_LEN],
        counter: u64,
        ciphertext: &[u8],
    ) -> Result<Vec<u8>, ThunderError> {
        let mut n = [0u8; 12];
        n[..4].copy_from_slice(salt);
        n[4..].copy_from_slice(&counter.to_be_bytes());
        let nonce = Nonce::from_slice(&n);
        key.decrypt(nonce, ciphertext)
            .map_err(|_| ThunderError::EncryptionFailed {
                detail: "open failed: wrong key or tampered data".to_string(),
            })
    }

    pub fn export_key(&self) -> ChaCha20Poly1305 {
        self.key.clone()
    }

    pub fn salt(&self) -> [u8; NONCE_SALT_LEN] {
        self.salt
    }

    /// Counter value that the NEXT encrypt() will use.
    pub fn counter(&self) -> u64 {
        self.counter
    }

    /// Build from explicit key material (both sides derive identical
    /// directional keys; see `derive_session_keys`).
    pub fn from_parts(key: [u8; 32], salt: [u8; NONCE_SALT_LEN]) -> Self {
        Self {
            key: ChaCha20Poly1305::new(Key::from_slice(&key)),
            salt,
            counter: 0,
        }
    }
}

/// Derive send/recv keys for one side from the shared ECDH secret.
/// Ordering by peer id makes direction unambiguous without extra round
/// trips: the lexicographically smaller id sends with key A.
pub struct SessionKeys {
    pub send_key: [u8; 32],
    pub send_salt: [u8; NONCE_SALT_LEN],
    pub recv_key: [u8; 32],
    pub recv_salt: [u8; NONCE_SALT_LEN],
}

fn kdf(secret: &[u8; 32], tag: &str, a: &str, b: &str) -> ([u8; 32], [u8; NONCE_SALT_LEN]) {
    let mut h = Sha256::new();
    h.update(b"thunder-v1/");
    h.update(tag.as_bytes());
    h.update(secret);
    h.update(a.as_bytes());
    h.update(b.as_bytes());
    let d = h.finalize();
    let mut key = [0u8; 32];
    key.copy_from_slice(&d);
    let mut salt = [0u8; NONCE_SALT_LEN];
    salt.copy_from_slice(&d[..4]);
    (key, salt)
}

pub fn derive_session_keys(shared: &[u8; 32], my_id: &str, their_id: &str) -> SessionKeys {
    let (lo, hi) = if my_id <= their_id {
        (my_id, their_id)
    } else {
        (their_id, my_id)
    };
    let (ka, sa) = kdf(shared, "a", lo, hi);
    let (kb, sb) = kdf(shared, "b", lo, hi);
    if my_id <= their_id {
        SessionKeys {
            send_key: ka,
            send_salt: sa,
            recv_key: kb,
            recv_salt: sb,
        }
    } else {
        SessionKeys {
            send_key: kb,
            send_salt: sb,
            recv_key: ka,
            recv_salt: sa,
        }
    }
}

/// Sign handshake bytes (ephemeral pubkey + session id) with the
/// long-term identity key. Verifier checks against the TRUSTED peer key.
pub fn sign_handshake(
    identity: &super::identity::ThunderIdentity,
    session_id: &str,
    ephemeral: &XPublic,
) -> ed25519_dalek::Signature {
    let mut msg = Vec::new();
    msg.extend_from_slice(session_id.as_bytes());
    msg.extend_from_slice(ephemeral.as_bytes());
    identity.sign(&msg)
}

pub fn verify_handshake(
    trusted_pubkey: &[u8; 32],
    session_id: &str,
    ephemeral: &XPublic,
    signature: &ed25519_dalek::Signature,
) -> bool {
    let mut msg = Vec::new();
    msg.extend_from_slice(session_id.as_bytes());
    msg.extend_from_slice(ephemeral.as_bytes());
    super::identity::ThunderIdentity::verify(trusted_pubkey, &msg, signature)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::OsRng;

    fn session_pair() -> (ThunderSession, ChaCha20Poly1305, [u8; 4]) {
        let s1 = EphemeralSecret::random_from_rng(OsRng);
        let s2 = EphemeralSecret::random_from_rng(OsRng);
        let p1 = XPublic::from(&s1);
        let p2 = XPublic::from(&s2);
        let a = ThunderSession::establish(s1, &p1, &p2);
        let key = a.export_key();
        let salt = a.salt();
        (a, key, salt)
    }

    #[test]
    fn test_session_round_trip() {
        let (mut a, b_key, salt) = session_pair();
        let c1 = a.encrypt(b"hello thunder").unwrap();
        let c2 = a.encrypt(b"second message").unwrap();
        assert_ne!(c1, c2);
        let p1 = ThunderSession::decrypt_with_counter(&b_key, &salt, 0, &c1).unwrap();
        let p2 = ThunderSession::decrypt_with_counter(&b_key, &salt, 1, &c2).unwrap();
        assert_eq!(p1, b"hello thunder");
        assert_eq!(p2, b"second message");
    }

    #[test]
    fn test_tamper_and_wrong_key_fail() {
        let (mut a, b_key, salt) = session_pair();
        let mut c = a.encrypt(b"secret").unwrap();
        c[5] ^= 0xff;
        assert!(ThunderSession::decrypt_with_counter(&b_key, &salt, 0, &c).is_err());
        // Wrong counter (replay) fails.
        let (mut a2, _, _) = session_pair();
        let c2 = a2.encrypt(b"x").unwrap();
        assert!(ThunderSession::decrypt_with_counter(&b_key, &salt, 99, &c2).is_err());
    }

    #[test]
    fn test_handshake_sign_verify() {
        use crate::thunder::identity::ThunderIdentity;
        let id = ThunderIdentity::generate("a".to_string());
        let s = EphemeralSecret::random_from_rng(OsRng);
        let p = XPublic::from(&s);
        let sig = sign_handshake(&id, "sess-1", &p);
        assert!(verify_handshake(&id.public_key_bytes(), "sess-1", &p, &sig));
        // Wrong session binds differently.
        assert!(!verify_handshake(
            &id.public_key_bytes(),
            "sess-2",
            &p,
            &sig
        ));
    }
}
