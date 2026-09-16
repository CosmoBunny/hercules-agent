//! Transport abstraction + encrypted TCP implementation.
//!
//! The rest of Thunder operates above `ThunderTransport`, never TCP
//! directly. Frames: `[len u32 BE][counter u64 BE][ciphertext]` where the
//! ciphertext is a versioned `ThunderMessage`. Each direction has its own
//! key/counter derived from one ECDH (see crypto::derive_session_keys).

use std::collections::HashMap;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

use super::crypto::{self, ThunderSession, derive_session_keys, sign_handshake, verify_handshake};
use super::error::ThunderError;
use super::identity::ThunderIdentity;
use super::protocol::{MAX_MESSAGE_BYTES, MessageKind, ThunderMessage};

/// Transport boundary: connect/send/receive/close. Inference, routing and
/// discovery never touch sockets through this trait.
#[async_trait::async_trait]
pub trait ThunderTransport: Send {
    async fn send(&mut self, msg: &ThunderMessage) -> Result<(), ThunderError>;
    async fn receive(&mut self) -> Result<ThunderMessage, ThunderError>;
    async fn close(mut self);
}

/// An authenticated, encrypted peer connection (one direction pair).
pub struct ThunderConnection {
    stream: TcpStream,
    send: ThunderSession,
    recv_key: chacha20poly1305::ChaCha20Poly1305,
    recv_salt: [u8; 4],
    recv_counter: u64,
    pub peer_id: String,
    pub peer_name: String,
    /// The peer's Ed25519 public key as presented (and self-signature
    /// verified) during the handshake. Retained so pairing can
    /// cryptographically bind a Pair request to this channel: the key
    /// claimed in Pair must equal this key.
    pub peer_public_key: [u8; 32],
}

impl ThunderConnection {
    fn new(
        stream: TcpStream,
        send: ThunderSession,
        recv_key: chacha20poly1305::ChaCha20Poly1305,
        recv_salt: [u8; 4],
        peer_id: String,
        peer_name: String,
        peer_public_key: [u8; 32],
    ) -> Self {
        Self {
            stream,
            send,
            recv_key,
            recv_salt,
            recv_counter: 0,
            peer_id,
            peer_name,
            peer_public_key,
        }
    }

    /// Outbound: connect + mutual handshake against a TRUSTED peer key.
    pub async fn connect(
        identity: &ThunderIdentity,
        trusted: &super::pairing::TrustedPeer,
        addr: std::net::SocketAddr,
    ) -> Result<Self, ThunderError> {
        let stream =
            tokio::time::timeout(std::time::Duration::from_secs(10), TcpStream::connect(addr))
                .await
                .map_err(|_| ThunderError::PeerUnavailable {
                    peer: trusted.peer_id.clone(),
                })?
                .map_err(|_| ThunderError::PeerUnavailable {
                    peer: trusted.peer_id.clone(),
                })?;
        tokio::time::timeout(
            std::time::Duration::from_secs(10),
            Self::handshake(stream, identity, Some(trusted)),
        )
        .await
        .map_err(|_| ThunderError::PeerUnavailable {
            peer: trusted.peer_id.clone(),
        })?
    }

    /// Mutual handshake. Outbound callers pass the trusted peer (unknown
    /// peers rejected before crypto). Inbound (host side) passes None and
    /// returns the verified peer id for the host to authorize.
    pub async fn handshake(
        mut stream: TcpStream,
        identity: &ThunderIdentity,
        trusted: Option<&super::pairing::TrustedPeer>,
    ) -> Result<Self, ThunderError> {
        use rand::rngs::OsRng;
        use x25519_dalek::{EphemeralSecret, PublicKey as XPublic};
        let session_id = format!("s{}", uuid::Uuid::new_v4());
        let secret = EphemeralSecret::random_from_rng(OsRng);
        let public = XPublic::from(&secret);
        let sig = sign_handshake(identity, &session_id, &public);
        let hello = ThunderMessage::new(
            "r0000000000000001",
            MessageKind::Hello {
                peer_id: identity.peer_id.clone(),
                public_key: identity.public_key_bytes().to_vec(),
                ephemeral: public.as_bytes().to_vec(),
                session_id: session_id.clone(),
                signature: sig.to_bytes().to_vec(),
            },
        );
        write_frame(&mut stream, &hello.encode()?).await?;
        let bytes = read_frame(&mut stream).await?;
        let reply = ThunderMessage::decode(&bytes)?;
        let (their_id, their_name, their_ephemeral, their_sig, their_key, their_session) =
            match reply.payload {
                MessageKind::Hello {
                    peer_id,
                    public_key,
                    ephemeral,
                    session_id,
                    signature,
                } => {
                    let name = String::new();
                    (peer_id, name, ephemeral, signature, public_key, session_id)
                }
                _ => {
                    return Err(ThunderError::ProtocolMismatch {
                        detail: "expected hello".to_string(),
                    });
                }
            };
        if their_key.len() != 32 || their_ephemeral.len() != 32 || their_sig.len() != 64 {
            return Err(ThunderError::AuthenticationFailed);
        }
        let mut ed = [0u8; 32];
        ed.copy_from_slice(&their_key);
        // Outbound: must match the trusted key. Inbound: accept the key,
        // verify self-signature, let the host authorize by peer id.
        if let Some(t) = trusted {
            if t.peer_id != their_id || t.public_key != ed {
                return Err(ThunderError::AuthenticationFailed);
            }
        }
        let mut ep = [0u8; 32];
        ep.copy_from_slice(&their_ephemeral);
        let ephpub = XPublic::from(ep);
        let mut sigb = [0u8; 64];
        sigb.copy_from_slice(&their_sig);
        let sig = ed25519_dalek::Signature::from_bytes(&sigb);
        // Verify against the SENDER's session id (each side signs its own
        // hello; the signature still binds sender + ephemeral + session).
        if !verify_handshake(&ed, &their_session, &ephpub, &sig) {
            return Err(ThunderError::AuthenticationFailed);
        }
        let shared = secret.diffie_hellman(&ephpub);
        let keys = derive_session_keys(shared.as_bytes(), &identity.peer_id, &their_id);
        let send = ThunderSession::from_parts(keys.send_key, keys.send_salt);
        Ok(Self::new(
            stream,
            send,
            {
                use chacha20poly1305::{Key, KeyInit};
                chacha20poly1305::ChaCha20Poly1305::new(Key::from_slice(&keys.recv_key))
            },
            keys.recv_salt,
            their_id.clone(),
            their_name,
            ed,
        ))
    }
}

async fn write_frame(stream: &mut TcpStream, plaintext: &[u8]) -> Result<(), ThunderError> {
    // Plaintext handshake frames only (pre-session). Encrypted frames go
    // through ThunderConnection::send.
    if plaintext.len() > MAX_MESSAGE_BYTES {
        return Err(ThunderError::ContextTooLarge);
    }
    let len = (plaintext.len() as u32).to_be_bytes();
    stream
        .write_all(&len)
        .await
        .map_err(|_| ThunderError::ConnectionLost)?;
    stream
        .write_all(plaintext)
        .await
        .map_err(|_| ThunderError::ConnectionLost)?;
    Ok(())
}

async fn read_frame(stream: &mut TcpStream) -> Result<Vec<u8>, ThunderError> {
    let mut lenb = [0u8; 4];
    stream
        .read_exact(&mut lenb)
        .await
        .map_err(|_| ThunderError::ConnectionLost)?;
    let len = u32::from_be_bytes(lenb) as usize;
    if len == 0 || len > MAX_MESSAGE_BYTES {
        return Err(ThunderError::InvalidRequest {
            detail: "bad frame length".to_string(),
        });
    }
    let mut buf = vec![0u8; len];
    stream
        .read_exact(&mut buf)
        .await
        .map_err(|_| ThunderError::ConnectionLost)?;
    Ok(buf)
}

#[async_trait::async_trait]
impl ThunderTransport for ThunderConnection {
    async fn send(&mut self, msg: &ThunderMessage) -> Result<(), ThunderError> {
        msg.validate()?;
        let bytes = msg.encode()?;
        let counter = self.send.counter();
        let ciphertext = self.send.encrypt(&bytes)?;
        let mut frame = Vec::with_capacity(4 + 8 + ciphertext.len());
        frame.extend_from_slice(&(8 + ciphertext.len() as u32).to_be_bytes());
        frame.extend_from_slice(&counter.to_be_bytes());
        frame.extend_from_slice(&ciphertext);
        self.stream
            .write_all(&frame)
            .await
            .map_err(|_| ThunderError::ConnectionLost)?;
        Ok(())
    }

    async fn receive(&mut self) -> Result<ThunderMessage, ThunderError> {
        let mut lenb = [0u8; 4];
        self.stream
            .read_exact(&mut lenb)
            .await
            .map_err(|_| ThunderError::ConnectionLost)?;
        let len = u32::from_be_bytes(lenb) as usize;
        if len < 8 || len > MAX_MESSAGE_BYTES + 8 {
            return Err(ThunderError::InvalidRequest {
                detail: "bad encrypted frame".to_string(),
            });
        }
        let mut counterb = [0u8; 8];
        self.stream
            .read_exact(&mut counterb)
            .await
            .map_err(|_| ThunderError::ConnectionLost)?;
        let counter = u64::from_be_bytes(counterb);
        let mut ct = vec![0u8; len - 8];
        self.stream
            .read_exact(&mut ct)
            .await
            .map_err(|_| ThunderError::ConnectionLost)?;
        // Strict ordering: exact expected counter only (no gaps/replays).
        if counter != self.recv_counter {
            return Err(ThunderError::EncryptionFailed {
                detail: "out-of-order frame".to_string(),
            });
        }
        let bytes =
            ThunderSession::decrypt_with_counter(&self.recv_key, &self.recv_salt, counter, &ct)?;
        self.recv_counter += 1;
        ThunderMessage::decode(&bytes)
    }

    async fn close(mut self) {
        let _ = self.stream.shutdown().await;
    }
}

impl ThunderConnection {
    /// Shutdown the socket in place (keeps ownership with the caller —
    /// required by the single-owner receive pump).
    pub async fn shutdown(&mut self) {
        let _ = self.stream.shutdown().await;
    }
}

/// Inbound listener helper for the host side.
pub struct ThunderListener {
    listener: TcpListener,
}

/// Outbound-only half: the send owner (frame counter + write half).
pub struct ThunderSendHalf {
    write: tokio::net::tcp::OwnedWriteHalf,
    session: ThunderSession,
    peer_id: String,
}

impl ThunderSendHalf {
    /// Send one validated message (frame: len + counter + ciphertext).
    pub async fn send(&mut self, msg: &ThunderMessage) -> Result<(), ThunderError> {
        msg.validate()?;
        let bytes = msg.encode()?;
        let counter = self.session.counter();
        let ciphertext = self.session.encrypt(&bytes)?;
        let mut frame = Vec::with_capacity(4 + 8 + ciphertext.len());
        frame.extend_from_slice(&(8 + ciphertext.len() as u32).to_be_bytes());
        frame.extend_from_slice(&counter.to_be_bytes());
        frame.extend_from_slice(&ciphertext);
        self.write
            .write_all(&frame)
            .await
            .map_err(|_| ThunderError::ConnectionLost)?;
        Ok(())
    }

    pub fn peer_id(&self) -> &str {
        &self.peer_id
    }
}

/// Inbound-only half: the receive owner (single reader, never raced by
/// a select! — read_exact is not cancellation safe, so a mid-frame
/// cancellation would corrupt the stream).
pub struct ThunderRecvHalf {
    read: tokio::net::tcp::OwnedReadHalf,
    recv_key: chacha20poly1305::ChaCha20Poly1305,
    recv_salt: [u8; 4],
    recv_counter: u64,
    peer_id: String,
}

impl ThunderRecvHalf {
    /// Receive one message (single reader by construction).
    pub async fn receive(&mut self) -> Result<ThunderMessage, ThunderError> {
        let mut lenb = [0u8; 4];
        self.read
            .read_exact(&mut lenb)
            .await
            .map_err(|_| ThunderError::ConnectionLost)?;
        let len = u32::from_be_bytes(lenb) as usize;
        if len < 8 || len > MAX_MESSAGE_BYTES + 8 {
            return Err(ThunderError::InvalidRequest {
                detail: "bad encrypted frame".to_string(),
            });
        }
        let mut counterb = [0u8; 8];
        self.read
            .read_exact(&mut counterb)
            .await
            .map_err(|_| ThunderError::ConnectionLost)?;
        let counter = u64::from_be_bytes(counterb);
        let mut ct = vec![0u8; len - 8];
        self.read
            .read_exact(&mut ct)
            .await
            .map_err(|_| ThunderError::ConnectionLost)?;
        // Strict ordering: exact expected counter only (no gaps/replays).
        if counter != self.recv_counter {
            return Err(ThunderError::EncryptionFailed {
                detail: "out-of-order frame".to_string(),
            });
        }
        let bytes =
            ThunderSession::decrypt_with_counter(&self.recv_key, &self.recv_salt, counter, &ct)?;
        self.recv_counter += 1;
        ThunderMessage::decode(&bytes)
    }

    pub fn peer_id(&self) -> &str {
        &self.peer_id
    }
}

impl ThunderConnection {
    /// Split into the send owner and the single receive owner. Used by
    /// clients that must send (Generate/Cancel) while one dedicated task
    /// reads — no two readers, no mid-frame cancellation.
    pub fn into_split(self) -> (ThunderSendHalf, ThunderRecvHalf) {
        let peer_id = self.peer_id.clone();
        let (read, write) = self.stream.into_split();
        (
            ThunderSendHalf {
                write,
                session: self.send,
                peer_id: peer_id.clone(),
            },
            ThunderRecvHalf {
                read,
                recv_key: self.recv_key,
                recv_salt: self.recv_salt,
                recv_counter: self.recv_counter,
                peer_id,
            },
        )
    }
}

impl ThunderListener {
    pub async fn bind(port: u16) -> Result<(Self, u16), ThunderError> {
        let listener = TcpListener::bind(("0.0.0.0", port)).await.map_err(|e| {
            ThunderError::InvalidRequest {
                detail: format!("thunder bind: {e}"),
            }
        })?;
        let actual = listener.local_addr().map(|a| a.port()).unwrap_or(port);
        Ok((Self { listener }, actual))
    }

    /// The actual bound socket address (0.0.0.0:<port> for the
    /// all-interfaces listener). Hosts must derive their advertised
    /// endpoint from THIS value — never manufacture one.
    pub fn local_addr(&self) -> Result<std::net::SocketAddr, ThunderError> {
        self.listener
            .local_addr()
            .map_err(|_| ThunderError::PeerUnavailable {
                peer: "listener".to_string(),
            })
    }

    pub async fn accept(&self) -> Result<TcpStream, ThunderError> {
        self.listener
            .accept()
            .await
            .map(|(s, _)| s)
            .map_err(|_| ThunderError::ConnectionLost)
    }
}

/// Pending inbound peer names by id (filled from discovery/advertisement).
pub type PeerNameTable = HashMap<String, String>;
