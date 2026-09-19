//! SharedThunderRuntime (T7): remote inference as a first-class runtime.
//!
//! Wraps a `ThunderReceiver` so Shared Thunder behaves like any other
//! Hercules backend: same `generate`/`generate_stream` signatures, same
//! cancellation contract (`is_generating` flag checked per token), plain
//! `String` results. No local model is ever loaded; the remote host owns
//! compute. Identity persists per installation; pairing is explicit.

use std::sync::Arc;
use std::time::Duration;

use super::context::ContextEnvelope;
use super::error::ThunderError;
use super::identity::ThunderIdentity;
use super::pairing::TrustedPeer;
use super::receiver::ThunderReceiver;

pub const REMOTE_MAX_NEW_TOKENS: u32 = 512;

/// One atomic remote inference target. The UI selection becomes THIS
/// value: peer identity, endpoint and model travel together from
/// selection → runtime → connection. Never an arbitrary first peer.
#[derive(Debug, Clone)]
pub struct RemoteTarget {
    /// Exact selected peer (matched against the trusted key at handshake).
    pub peer_id: String,
    pub address: std::net::SocketAddr,
    /// Remote model id (a `ThunderModel::id`, never a local path).
    pub model_id: String,
    /// The trusted record for THIS peer only.
    pub trusted_peer: TrustedPeer,
}

impl RemoteTarget {
    /// Parse and resolve one selection reference:
    /// `peer_id::model_id@host:port`. The peer is looked up BY ID in the
    /// persistent trust store — never an arbitrary first peer.
    pub fn parse(model_ref: &str) -> Result<Self, ThunderError> {
        let (peer_id, rest) =
            model_ref
                .split_once("::")
                .ok_or_else(|| ThunderError::InvalidRequest {
                    detail: format!(
                        "shared-thunder ref must be peer_id::model_id@host:port: {model_ref}"
                    ),
                })?;
        let peer_id = peer_id.trim().to_string();
        let (model_id, addr) =
            rest.rsplit_once('@')
                .ok_or_else(|| ThunderError::InvalidRequest {
                    detail: format!(
                        "shared-thunder ref must be peer_id::model_id@host:port: {model_ref}"
                    ),
                })?;
        let model_id = model_id.trim().to_string();
        if peer_id.is_empty() || model_id.is_empty() {
            return Err(ThunderError::InvalidRequest {
                detail: "empty peer_id or model_id in shared-thunder ref".to_string(),
            });
        }
        let address: std::net::SocketAddr =
            addr.trim()
                .parse()
                .map_err(|_| ThunderError::InvalidRequest {
                    detail: format!("bad host:port in shared-thunder ref: {addr}"),
                })?;
        let store = crate::thunder::pairing::PeerStore::load();
        let trusted_peer = store
            .get(&peer_id)
            .ok_or_else(|| ThunderError::PeerUnavailable {
                peer: peer_id.clone(),
            })?;
        Ok(Self {
            peer_id,
            address,
            model_id,
            trusted_peer: trusted_peer.clone(),
        })
    }
}

/// A remote Shared Thunder host exposed through the backend surface.
#[derive(Clone)]
pub struct SharedThunderBackend {
    pub identity: Arc<ThunderIdentity>,
    /// The atomic selected target — never decomposed into
    /// (model_id + endpoint) + (arbitrary trusted peer).
    pub target: RemoteTarget,
    conn: Arc<tokio::sync::Mutex<Option<Arc<ThunderReceiver>>>>,
}

impl SharedThunderBackend {
    pub fn new(identity: Arc<ThunderIdentity>, target: RemoteTarget) -> Self {
        Self {
            identity,
            target,
            conn: Arc::new(tokio::sync::Mutex::new(None)),
        }
    }

    /// Parse a selection reference of the form
    /// `peer_id::model_id@host:port` and bind the EXACT trusted peer by
    /// id. No `get_first_trusted()`, no arbitrary peer, no invented host.
    pub fn from_ref(model_ref: &str) -> Result<Self, ThunderError> {
        let target = RemoteTarget::parse(model_ref)?;
        let identity = Arc::new(ThunderIdentity::load_or_generate("Hercules".to_string())?);
        Ok(Self::new(identity, target))
    }

    pub fn with_model_id(&self, model_id: &str) -> Self {
        let mut b = self.clone();
        b.target.model_id = model_id.trim().to_string();
        b
    }

    pub fn name(&self) -> String {
        format!(
            "Shared Thunder ({}) @ {}",
            self.target.model_id, self.target.trusted_peer.name
        )
    }

    /// Persistent connection (lazily established, reused across calls).
    async fn receiver(&self) -> Result<Arc<ThunderReceiver>, ThunderError> {
        let mut guard = self.conn.lock().await;
        if let Some(r) = guard.as_ref() {
            return Ok(r.clone());
        }
        let r = Arc::new(
            ThunderReceiver::connect(
                &self.identity,
                &self.target.trusted_peer,
                self.target.address,
            )
            .await?,
        );
        *guard = Some(r.clone());
        Ok(r)
    }

    pub async fn generate(&self, prompt: &str) -> Result<String, String> {
        let receiver = self.receiver().await.map_err(thunder_to_string)?;
        receiver
            .generate(
                &self.target.model_id,
                "",
                prompt,
                ContextEnvelope::empty(),
                REMOTE_MAX_NEW_TOKENS,
            )
            .await
            .map_err(thunder_to_string)
    }

    /// Streaming generate over the backend stream contract: pushes tokens
    /// into `stream_target` as they arrive; checks `is_generating` per
    /// token and cancels the remote request when the user cancelled.
    pub async fn generate_stream(
        &self,
        prompt: &str,
        stream_target: Arc<std::sync::Mutex<String>>,
        is_generating: Arc<std::sync::Mutex<bool>>,
    ) -> Result<String, String> {
        let receiver = self.receiver().await.map_err(thunder_to_string)?;
        let mut stream = receiver
            .generate_stream(
                &self.target.model_id,
                "",
                prompt,
                ContextEnvelope::empty(),
                REMOTE_MAX_NEW_TOKENS,
            )
            .await
            .map_err(thunder_to_string)?;
        let mut cancelled = false;
        // The cancel flag is watched on its own tick — a host that
        // streams nothing must still be cancellable promptly.
        let mut ticker = tokio::time::interval(Duration::from_millis(100));
        ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        loop {
            let ev = tokio::select! {
                ev = stream.events.recv() => match ev {
                    Some(ev) => ev,
                    None => break,
                },
                _ = ticker.tick() => {
                    if !cancelled {
                        if let Ok(active) = is_generating.lock() {
                            if !*active {
                                cancelled = true;
                                stream.cancel();
                            }
                        }
                    }
                    continue;
                }
            };
            match ev {
                super::receiver::StreamEvent::Token(t) => {
                    if let Ok(mut target) = stream_target.lock() {
                        target.push_str(&t);
                    }
                }
                super::receiver::StreamEvent::Done(_) => break,
                super::receiver::StreamEvent::Error(super::ThunderError::RequestCancelled) => {
                    return Err("[Generation Cancelled by User (CTRL+C)]".to_string());
                }
                super::receiver::StreamEvent::Error(e) => return Err(thunder_to_string(e)),
            }
        }
        let full = stream_target.lock().map(|t| t.clone()).unwrap_or_default();
        if full.is_empty() {
            Err("Shared Thunder stream completed with no tokens.".to_string())
        } else {
            Ok(full)
        }
    }

    /// Health probe: connection handshake only — no inference.
    pub async fn probe(&self) -> Result<(), ThunderError> {
        self.receiver().await.map(|_| ())
    }
}

fn thunder_to_string(e: ThunderError) -> String {
    format!("[Shared Thunder Error] {}", e)
}

#[cfg(test)]
mod tests {
    use super::super::capabilities::{HostCapabilities, ThunderModel};
    use super::super::host::{SlowExecutor, StubExecutor, ThunderHost};
    use super::super::identity::ThunderIdentity;
    use super::super::pairing::{PeerPermissions, PeerStore};
    use super::*;

    fn caps(host_id: &str) -> HostCapabilities {
        HostCapabilities {
            peer_id: host_id.to_string(),
            hardware: crate::model::HardwareInfo::detect(),
            models: vec![ThunderModel {
                id: "qwen3-30b".to_string(),
                name: "Qwen3-30B".to_string(),
                architecture: "Qwen3".to_string(),
                format: "SafeTensors".to_string(),
                backend: "Transformers".to_string(),
                quantization: None,
                context_length: 32768,
                streaming: true,
                cancellation: true,
            }],
            inference: true,
            forwarding: false,
            max_concurrent_requests: 4,
            max_context_tokens: 32768,
            max_new_tokens: 512,
            streaming: true,
            cancellation: true,
        }
    }

    fn trusted_pair(
        host_id: &ThunderIdentity,
        recv_id: &ThunderIdentity,
    ) -> (PeerStore, TrustedPeer) {
        let mut host_peers = PeerStore::default();
        assert!(host_peers.trust(
            recv_id.peer_id.clone(),
            recv_id.public_key_bytes(),
            "recv".to_string(),
            PeerPermissions::default(),
            true
        ));
        let trusted = TrustedPeer {
            peer_id: host_id.peer_id.clone(),
            public_key: host_id.public_key_bytes(),
            name: "host".to_string(),
            permissions: PeerPermissions::default(),
            paired_at_epoch: 0,
        };
        (host_peers, trusted)
    }

    async fn spawn_host(
        executor: Arc<dyn super::super::host::ThunderExecutor>,
    ) -> (std::net::SocketAddr, ThunderIdentity, ThunderIdentity) {
        let host_id = ThunderIdentity::generate("host".to_string());
        let recv_id = ThunderIdentity::generate("recv".to_string());
        let (host_peers, _) = trusted_pair(&host_id, &recv_id);
        let host = ThunderHost::new(caps(&host_id.peer_id), executor).unwrap();
        let addr =
            super::super::host::tests_serve_forever(host, host_id.clone(), host_peers.clone())
                .await;
        (addr, host_id, recv_id)
    }

    fn recv_trusted(host_id: &ThunderIdentity) -> TrustedPeer {
        TrustedPeer {
            peer_id: host_id.peer_id.clone(),
            public_key: host_id.public_key_bytes(),
            name: "host".to_string(),
            permissions: PeerPermissions::default(),
            paired_at_epoch: 0,
        }
    }

    fn make_backend(
        addr: std::net::SocketAddr,
        host_id: &ThunderIdentity,
        recv_id: ThunderIdentity,
        model_id: &str,
    ) -> SharedThunderBackend {
        SharedThunderBackend::new(
            Arc::new(recv_id),
            RemoteTarget {
                peer_id: host_id.peer_id.clone(),
                address: addr,
                model_id: model_id.to_string(),
                trusted_peer: recv_trusted(host_id),
            },
        )
    }

    #[tokio::test]
    async fn test_shared_runtime_generate_round_trip() {
        let (addr, host_id, recv_id) = spawn_host(Arc::new(StubExecutor {
            reply: "remote hello".to_string(),
            tokens: vec![],
        }))
        .await;
        let backend = make_backend(addr, &host_id, recv_id, "qwen3-30b");
        let out = backend.generate("hi").await.unwrap();
        assert_eq!(out, "remote hello");
        assert!(backend.name().contains("Shared Thunder"));
        assert!(backend.name().contains("qwen3-30b"));
    }

    #[tokio::test]
    async fn test_shared_runtime_stream_into_target() {
        let (addr, host_id, recv_id) = spawn_host(Arc::new(StubExecutor {
            reply: String::new(),
            tokens: vec!["a".to_string(), "b".to_string(), "c".to_string()],
        }))
        .await;
        let backend = make_backend(addr, &host_id, recv_id, "qwen3-30b");
        let target = Arc::new(std::sync::Mutex::new(String::new()));
        let flag = Arc::new(std::sync::Mutex::new(true));
        let full = backend
            .generate_stream("hi", target.clone(), flag)
            .await
            .unwrap();
        assert_eq!(full, "abc");
        assert_eq!(target.lock().unwrap().as_str(), "abc");
    }

    #[tokio::test]
    async fn test_shared_runtime_cancel_via_flag() {
        let (addr, host_id, recv_id) = spawn_host(Arc::new(SlowExecutor {
            delay: Duration::from_secs(30),
        }))
        .await;
        let backend = make_backend(addr, &host_id, recv_id, "qwen3-30b");
        let target = Arc::new(std::sync::Mutex::new(String::new()));
        let flag = Arc::new(std::sync::Mutex::new(true));
        let flag2 = flag.clone();
        // User cancels shortly after the stream starts.
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(400)).await;
            if let Ok(mut f) = flag2.lock() {
                *f = false;
            }
        });
        let t0 = std::time::Instant::now();
        let res = backend.generate_stream("hi", target, flag).await;
        assert!(res.is_err(), "cancelled generation must error");
        assert!(res.unwrap_err().contains("Cancelled"));
        assert!(
            t0.elapsed() < Duration::from_secs(20),
            "must not wait for slow host"
        );
    }

    #[test]
    fn test_from_ref_binds_exact_peer_never_arbitrary() {
        // Malformed refs fail typed, never invented.
        assert!(SharedThunderBackend::from_ref("no-address").is_err());
        assert!(SharedThunderBackend::from_ref("p::m@not-an-address").is_err());
        assert!(SharedThunderBackend::from_ref("::m@127.0.0.1:9000").is_err());
        assert!(SharedThunderBackend::from_ref("p::@127.0.0.1:9000").is_err());
        // Unknown peer id: PeerUnavailable — NEVER the first trusted peer.
        assert!(matches!(
            SharedThunderBackend::from_ref("nobody::m@127.0.0.1:9000"),
            Err(ThunderError::PeerUnavailable { .. })
        ));
    }

    #[test]
    fn test_from_ref_resolves_the_selected_peer_not_the_first() {
        // Persisted store: alice at :1, bob at :2. Selecting bob must
        // bind BOB's trusted record — never alice (the first peer).
        let _store_guard = crate::thunder::pairing::store_test_guard();
        let dir =
            std::env::temp_dir().join(format!("hercules-thunder-target-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        // SAFETY: test-only process-wide env override; no other test
        // thread reads XDG_DATA_HOME concurrently by contract.
        unsafe {
            std::env::set_var("XDG_DATA_HOME", &dir);
        }
        let alice_key = [1u8; 32];
        let bob_key = [2u8; 32];
        let peers = vec![
            TrustedPeer {
                peer_id: "thunder-alice".to_string(),
                public_key: alice_key,
                name: "Alice-PC".to_string(),
                permissions: PeerPermissions::default(),
                paired_at_epoch: 0,
            },
            TrustedPeer {
                peer_id: "thunder-bob".to_string(),
                public_key: bob_key,
                name: "Bob-PC".to_string(),
                permissions: PeerPermissions::default(),
                paired_at_epoch: 0,
            },
        ];
        let peer_dir = dir.join("hercules").join("thunder");
        std::fs::create_dir_all(&peer_dir).unwrap();
        std::fs::write(
            peer_dir.join("peers.json"),
            serde_json::to_string(&peers).unwrap(),
        )
        .unwrap();
        let b = SharedThunderBackend::from_ref("thunder-bob::qwen3-30b@192.168.1.20:47832")
            .expect("selected peer must resolve");
        // EXACT binding: bob's id, bob's key, bob's address, bob's model.
        assert_eq!(b.target.peer_id, "thunder-bob");
        assert_eq!(b.target.trusted_peer.public_key, bob_key);
        assert_eq!(b.target.trusted_peer.name, "Bob-PC");
        assert_eq!(b.target.address.to_string(), "192.168.1.20:47832");
        assert_eq!(b.target.model_id, "qwen3-30b");
        unsafe {
            std::env::remove_var("XDG_DATA_HOME");
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
}
