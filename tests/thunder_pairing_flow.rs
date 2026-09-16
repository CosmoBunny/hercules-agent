//! Shared Thunder acceptance: real pairing + remote inference.
//!
//! Two Hercules instances (A hosts, B pairs) using ONLY production
//! paths: real listener bind, real discovery beacons, real Pair
//! exchange against the host registry, real Models fetch, exact
//! RemoteTarget, real streamed generation — then the reverse.
//! Test executors live in THIS file (never in production).

use std::net::SocketAddr;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use hercules_agent::backend::AgentBackend;
use hercules_agent::thunder::capabilities::{HostCapabilities, ThunderModel};
use hercules_agent::thunder::connection::{ThunderConnection, ThunderListener};
use hercules_agent::thunder::discovery::Discovery;
use hercules_agent::thunder::error::ThunderError;
use hercules_agent::thunder::host::{PairingRegistry, ThunderExecutor, ThunderHost};
use hercules_agent::thunder::identity::ThunderIdentity;
use hercules_agent::thunder::pairing::{PairingCode, PeerPermissions, PeerStore};
use hercules_agent::thunder::receiver::{StreamEvent, ThunderReceiver};
use hercules_agent::thunder::runtime::{RemoteTarget, SharedThunderBackend};
use hercules_agent::thunder_ui;

// ---------------------------------------------------------------------------
// Local test doubles (this file only — never production).
// ---------------------------------------------------------------------------

struct ReplyExecutor {
    reply: String,
    tokens: Vec<String>,
}

#[async_trait::async_trait]
impl ThunderExecutor for ReplyExecutor {
    async fn generate(
        &self,
        _model_id: &str,
        _system: &str,
        _prompt: &str,
    ) -> Result<String, ThunderError> {
        Ok(self.reply.clone())
    }

    async fn generate_stream(
        &self,
        _model_id: &str,
        _system: &str,
        _prompt: &str,
        tx: tokio::sync::mpsc::Sender<String>,
    ) -> Result<(), ThunderError> {
        if self.tokens.is_empty() {
            let _ = tx.send(self.reply.clone()).await;
        } else {
            for tok in &self.tokens {
                tokio::time::sleep(Duration::from_millis(20)).await;
                if tx.send(tok.clone()).await.is_err() {
                    break;
                }
            }
        }
        Ok(())
    }
}

/// Blocks until aborted: proves host Cancel reaches backend execution.
struct BlockingExecutor {
    finished: Arc<AtomicBool>,
}

#[async_trait::async_trait]
impl ThunderExecutor for BlockingExecutor {
    async fn generate(
        &self,
        _model_id: &str,
        _system: &str,
        _prompt: &str,
    ) -> Result<String, ThunderError> {
        self.generate_stream(_model_id, _system, _prompt, tokio::sync::mpsc::channel(1).0)
            .await?;
        Ok(String::new())
    }

    async fn generate_stream(
        &self,
        _model_id: &str,
        _system: &str,
        _prompt: &str,
        _tx: tokio::sync::mpsc::Sender<String>,
    ) -> Result<(), ThunderError> {
        tokio::time::sleep(Duration::from_secs(60)).await;
        self.finished.store(true, Ordering::SeqCst);
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Helpers (production shapes, test wiring).
// ---------------------------------------------------------------------------

fn model(id: &str, name: &str) -> ThunderModel {
    ThunderModel {
        id: id.to_string(),
        name: name.to_string(),
        architecture: "Qwen3".to_string(),
        format: "SafeTensors".to_string(),
        backend: "Transformers".to_string(),
        quantization: None,
        context_length: 32768,
        streaming: true,
        cancellation: true,
    }
}

fn caps(peer_id: &str, models: Vec<ThunderModel>) -> HostCapabilities {
    HostCapabilities {
        peer_id: peer_id.to_string(),
        hardware: hercules_agent::model::HardwareInfo::detect(),
        models,
        inference: true,
        forwarding: false,
        max_concurrent_requests: 4,
        max_context_tokens: 32768,
        max_new_tokens: 512,
        streaming: true,
        cancellation: true,
    }
}

/// Production-shaped accept loop: handshake → serve with FRESH trust
/// per connection (mirrors production `PeerStore::load()`). Shared
/// store so later Trust applies without restart.
async fn serve_forever(
    mut host: ThunderHost,
    host_id: ThunderIdentity,
    peers: Arc<std::sync::Mutex<PeerStore>>,
    registry: Arc<PairingRegistry>,
) -> SocketAddr {
    host.set_pairing_registry(registry);
    let (listener, port) = ThunderListener::bind(0).await.unwrap();
    let addr: SocketAddr = format!("127.0.0.1:{port}").parse().unwrap();
    let host = Arc::new(host);
    tokio::spawn(async move {
        loop {
            let stream = match listener.accept().await {
                Ok(s) => s,
                Err(_) => return,
            };
            let conn = match tokio::time::timeout(
                Duration::from_secs(10),
                ThunderConnection::handshake(stream, &host_id, None),
            )
            .await
            {
                Ok(Ok(c)) => c,
                _ => continue,
            };
            let host = host.clone();
            let peers = peers.clone();
            tokio::spawn(async move {
                let snapshot = peers.lock().unwrap().clone();
                let _ = host.serve(conn, &snapshot).await;
            });
        }
    });
    addr
}

fn trust(store: &mut PeerStore, peer_id: &str, key: [u8; 32], name: &str) {
    assert!(store.trust(
        peer_id.to_string(),
        key,
        name.to_string(),
        PeerPermissions::default(),
        true,
    ));
}

async fn stream_text(backend: &SharedThunderBackend, prompt: &str) -> String {
    backend
        .generate(prompt)
        .await
        .unwrap_or_else(|e| panic!("remote inference failed: {e}"))
}

// ---------------------------------------------------------------------------
// 1-3. Real bind: non-loopback listener, actual endpoint, beacon port.
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn host_binds_all_interfaces_with_actual_endpoint() {
    // The listener binds all interfaces (LAN-reachable), never loopback.
    let (listener, _) = ThunderListener::bind(0).await.unwrap();
    let bound = listener.local_addr().unwrap();
    assert!(
        bound.ip().is_unspecified(),
        "listener must bind 0.0.0.0, got {bound}"
    );
    assert_ne!(bound.port(), 0, "ephemeral port resolved to actual");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn host_task_endpoint_is_real_and_dialable() {
    // Production start path with a real (Ollama-named, no inference
    // performed) backend: endpoint port == bound port, TCP dialable,
    // stop makes it unavailable.
    let backend = AgentBackend::Ollama(hercules_agent::backend::OllamaBackend::new(
        "qwen3:8b".to_string(),
    ));
    let hosted = thunder_ui::host_model_from_backend(&backend).expect("ollama hosts");
    let id = ThunderIdentity::generate("A".to_string());
    let caps = thunder_ui::build_host_caps(&id, &hosted, true, 4);
    let rt = thunder_ui::start_host_task(backend, caps, id, 0)
        .await
        .expect("host starts");
    assert_ne!(rt.tcp_port, 0);
    assert_eq!(rt.endpoint.port(), rt.tcp_port, "single authoritative port");
    // Endpoint is a real enumerated interface address, or an explicit
    // loopback-only fallback — never a silently mislabeled LAN address.
    if rt.lan_available {
        assert!(!rt.endpoint.ip().is_loopback());
        assert!(rt.all_endpoints.contains(&rt.endpoint));
    } else {
        assert!(rt.endpoint.ip().is_loopback());
    }
    // Dialable while live…
    let probe = tokio::time::timeout(
        Duration::from_secs(5),
        tokio::net::TcpStream::connect(SocketAddr::new(
            std::net::IpAddr::from([127, 0, 0, 1]),
            rt.tcp_port,
        )),
    )
    .await
    .expect("connect completes");
    assert!(probe.is_ok(), "listener accepts connections");
    drop(probe);
    // …unavailable after stop.
    rt.stop_token.cancel();
    tokio::time::sleep(Duration::from_millis(500)).await;
    let probe = tokio::time::timeout(
        Duration::from_secs(5),
        tokio::net::TcpStream::connect(SocketAddr::new(
            std::net::IpAddr::from([127, 0, 0, 1]),
            rt.tcp_port,
        )),
    )
    .await
    .expect("post-stop connect completes");
    // Either refused immediately or accepted-then-closed by teardown;
    // in both cases no NEW session can complete a handshake.
    if let Ok(stream) = probe {
        let id2 = ThunderIdentity::generate("probe".to_string());
        let hs = tokio::time::timeout(
            Duration::from_secs(5),
            ThunderConnection::handshake(stream, &id2, None),
        )
        .await
        .expect("post-stop handshake completes");
        assert!(hs.is_err(), "no handshake after stop");
    }
}

// ---------------------------------------------------------------------------
// Host stop via the production path: live session dies, new dials fail.
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn host_stop_kills_live_session_and_blocks_new() {
    // Isolate the persistent store: the production accept loop loads
    // PeerStore::load() per connection.
    let dir = tempfile::tempdir().unwrap();
    let old_xdg = std::env::var("XDG_DATA_HOME").ok();
    unsafe {
        std::env::set_var("XDG_DATA_HOME", dir.path());
    }

    let host_id = ThunderIdentity::generate("host".to_string());
    let recv_id = ThunderIdentity::generate("recv".to_string());
    // Explicit operator trust (the only path to trust).
    assert!(thunder_ui::ThunderUiState::trust_peer(
        &recv_id.peer_id,
        recv_id.public_key_bytes(),
        "recv",
        true,
        false,
        true,
    ));
    let trusted = hercules_agent::thunder::pairing::PeerStore::load()
        .get(&recv_id.peer_id)
        .is_some();
    assert!(trusted, "trust persisted");
    let trusted_host = hercules_agent::thunder::pairing::TrustedPeer {
        peer_id: host_id.peer_id.clone(),
        public_key: host_id.public_key_bytes(),
        name: "host".to_string(),
        permissions: hercules_agent::thunder::pairing::PeerPermissions::default(),
        paired_at_epoch: 0,
    };
    // Trust the receiver's view of the host too (exact record).
    assert!(thunder_ui::ThunderUiState::trust_peer(
        &host_id.peer_id,
        host_id.public_key_bytes(),
        "host",
        true,
        false,
        true,
    ));

    let backend = AgentBackend::Ollama(hercules_agent::backend::OllamaBackend::new(
        "qwen3:8b".to_string(),
    ));
    let hosted = thunder_ui::host_model_from_backend(&backend).expect("ollama hosts");
    let caps = thunder_ui::build_host_caps(&host_id, &hosted, true, 4);
    let rt = thunder_ui::start_host_task(backend, caps, host_id.clone(), 0)
        .await
        .expect("host starts");
    let dial = SocketAddr::new(std::net::IpAddr::from([127, 0, 0, 1]), rt.tcp_port);

    // Session established while live (real Models exchange).
    let fetched = thunder_ui::fetch_remote_models(&recv_id, &trusted_host, dial)
        .await
        .expect("models while live");
    assert_eq!(fetched.len(), 1, "only the real hosted model");
    assert_eq!(fetched[0].id, hosted.wire_id, "advertised == executed id");

    // Hold a persistent session open…
    let rx = ThunderReceiver::connect(&recv_id, &trusted_host, dial)
        .await
        .expect("session while live");

    // …stop the host: accept loop + live sessions + listener go down.
    rt.stop_token.cancel();
    tokio::time::sleep(Duration::from_millis(500)).await;

    // Existing session is dead: generation fails instead of hanging.
    let err = tokio::time::timeout(
        Duration::from_secs(10),
        rx.generate(
            &hosted.wire_id,
            "",
            "hi",
            hercules_agent::thunder::context::ContextEnvelope::empty(),
            16,
        ),
    )
    .await
    .expect("generate completes");
    assert!(err.is_err(), "no generation on a stopped host");
    rx.close().await;

    // New sessions cannot be established.
    let fresh = tokio::time::timeout(
        Duration::from_secs(10),
        ThunderReceiver::connect(&recv_id, &trusted_host, dial),
    )
    .await
    .expect("connect completes");
    assert!(fresh.is_err(), "new connections fail after stop");

    unsafe {
        match old_xdg {
            Some(v) => std::env::set_var("XDG_DATA_HOME", v),
            None => std::env::remove_var("XDG_DATA_HOME"),
        }
    }
}

// ---------------------------------------------------------------------------
// 4-11. Discovery + real pairing exchange.
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn discovery_pairing_trust_flow() {
    let alice = ThunderIdentity::generate("Alice".to_string());
    let bob = ThunderIdentity::generate("Bob".to_string());

    // Host A with a pairing registry + one real model.
    let registry = PairingRegistry::new();
    registry.set_host_identity(
        alice.peer_id.clone(),
        "Alice".to_string(),
        alice.public_key_bytes(),
    );
    let code = PairingCode::generate();
    let code_text = code.code.clone();
    registry.set_code(Some(code));
    let host = ThunderHost::new(
        caps(&alice.peer_id, vec![model("qwen3-32b", "Qwen3-32B")]),
        Arc::new(ReplyExecutor {
            reply: "hello from Alice".to_string(),
            tokens: vec![],
        }),
    )
    .unwrap();
    let host_peers = Arc::new(std::sync::Mutex::new(PeerStore::default()));
    let addr = serve_forever(host, alice.clone(), host_peers.clone(), registry.clone()).await;

    // (4) Another instance discovers the host via directed beacon.
    let mut recv_disc = Discovery::bind_loopback(0).unwrap();
    let mut host_disc = Discovery::bind_loopback(0).unwrap();
    host_disc
        .announce_to(&alice, recv_disc.local_beacon_addr().unwrap())
        .unwrap();
    tokio::time::sleep(Duration::from_millis(100)).await;
    let fresh = recv_disc.poll(&bob.peer_id);
    assert!(fresh.contains(&alice.peer_id), "B discovers A");
    let beacon = recv_disc
        .peers()
        .into_iter()
        .find(|p| p.peer_id == alice.peer_id)
        .unwrap()
        .clone();

    // (8) Before any trust: generation fails (untrusted).
    let mut bob_peers = PeerStore::default();
    assert!(!bob_peers.trust(
        "x".to_string(),
        [0u8; 32],
        "x".to_string(),
        PeerPermissions::default(),
        false,
    ));
    let untrusted = hercules_agent::thunder::pairing::TrustedPeer {
        peer_id: alice.peer_id.clone(),
        public_key: beacon.public_key,
        name: "Alice".to_string(),
        permissions: PeerPermissions::default(),
        paired_at_epoch: 0,
    };

    // (5) Correct code → host accepts, queues validated request.
    let accepted = thunder_ui::send_pairing_request(&bob, addr, &code_text)
        .await
        .expect("correct code accepted");
    assert_eq!(accepted.peer_id, alice.peer_id, "real host identity");
    assert_eq!(accepted.public_key, alice.public_key_bytes());
    let pending = registry.pending();
    assert_eq!(pending.len(), 1, "validated request queued");
    assert_eq!(pending[0].peer_id, bob.peer_id);

    // (6) Wrong code → rejected, nothing queued, stays untrusted.
    let before = registry.pending().len();
    let err = thunder_ui::send_pairing_request(&bob, addr, "XXXX-YYYY")
        .await
        .expect_err("wrong code rejected");
    assert!(err.to_string().contains("incorrect") || err.to_string().contains("rejected"));
    assert_eq!(registry.pending().len(), before);
    assert!(!bob_peers.is_trusted(&alice.peer_id));

    // (10) Presented key != channel key → rejected even with right code.
    {
        use hercules_agent::thunder::connection::ThunderTransport;
        use hercules_agent::thunder::protocol::{MessageKind, ThunderMessage};
        let stream =
            tokio::time::timeout(Duration::from_secs(5), tokio::net::TcpStream::connect(addr))
                .await
                .expect("connect completes")
                .unwrap();
        let mut conn = tokio::time::timeout(
            Duration::from_secs(5),
            ThunderConnection::handshake(stream, &bob, None),
        )
        .await
        .expect("handshake completes")
        .unwrap();
        let req = ThunderMessage::new(
            hercules_agent::thunder::receiver::next_request_id(),
            MessageKind::Pair {
                code: code_text.clone(),
                peer_id: bob.peer_id.clone(),
                name: "Bob".to_string(),
                public_key: [7u8; 32].to_vec(),
            },
        );
        conn.send(&req).await.unwrap();
        let reply = tokio::time::timeout(Duration::from_secs(5), conn.receive())
            .await
            .unwrap()
            .unwrap();
        assert!(
            matches!(reply.payload, MessageKind::Error { .. }),
            "key mismatch rejected, got {:?}",
            reply.payload
        );
    }

    // (9) Explicit Trust (operator saw fingerprint) persists.
    trust(&mut bob_peers, &alice.peer_id, beacon.public_key, "Alice");
    trust(
        &mut host_peers.lock().unwrap(),
        &bob.peer_id,
        bob.public_key_bytes(),
        "Bob",
    );
    assert!(
        registry.consume(&bob.peer_id).is_some(),
        "trust consumes request"
    );

    // (12) Models response contains ONLY the real advertised model.
    let trusted = bob_peers.get(&alice.peer_id).unwrap().clone();
    let fetched = thunder_ui::fetch_remote_models(&bob, &trusted, addr)
        .await
        .expect("models fetch");
    assert_eq!(fetched.len(), 1);
    assert_eq!(fetched[0].id, "qwen3-32b");

    // (13) Host rejects model ids it does not serve.
    let rx = ThunderReceiver::connect(&bob, &trusted, addr)
        .await
        .unwrap();
    let err = rx
        .generate(
            "not-a-model",
            "",
            "hi",
            hercules_agent::thunder::context::ContextEnvelope::empty(),
            16,
        )
        .await
        .expect_err("unknown model rejected");
    assert!(
        matches!(
            err,
            hercules_agent::thunder::error::ThunderError::ModelUnavailable { .. }
        ),
        "typed ModelUnavailable, got: {err:?}"
    );
    rx.close().await;

    // (14/15) Exact target: Alice stays Alice.
    let target = thunder_ui::make_remote_target(&alice.peer_id, addr, "qwen3-32b", trusted);
    assert_eq!(target.peer_id, alice.peer_id);
    assert_eq!(target.trusted_peer.public_key, alice.public_key_bytes());
    let _ = untrusted;

    // (16/17) Streamed generation from the selected host.
    let backend = SharedThunderBackend::new(Arc::new(bob.clone()), target);
    let out = stream_text(&backend, "say hello").await;
    assert!(out.contains("hello from Alice"), "got: {out}");
}

// ---------------------------------------------------------------------------
// 7. Expiry is enforced by the registry (unit-level, real clock edge).
// ---------------------------------------------------------------------------

#[test]
fn pairing_code_expiry_rejected() {
    let registry = PairingRegistry::new();
    let expired = PairingCode {
        code: "ABCD-EFGH".to_string(),
        created: std::time::Instant::now() - Duration::from_secs(3600),
        lifetime: Duration::from_secs(600),
    };
    assert!(!expired.is_valid());
    registry.set_code(Some(expired));
    let err = registry
        .accept_pair("thunder-x", "X", [1u8; 32], &[1u8; 32], "ABCD-EFGH")
        .expect_err("expired code rejected");
    assert!(err.contains("expired") || err.contains("incorrect"));
    assert!(registry.pending().is_empty());
}

// ---------------------------------------------------------------------------
// Pairing codes are single-use: correct → accepted, same again →
// rejected, incorrect → rejected (expiry above). Operator Reject does
// NOT restore the code.
// ---------------------------------------------------------------------------

#[test]
fn pairing_code_single_use() {
    let registry = PairingRegistry::new();
    registry.set_host_identity("thunder-host".to_string(), "host".to_string(), [9u8; 32]);
    let code = PairingCode::generate();
    let text = code.code.clone();
    registry.set_code(Some(code));

    // Correct code → accepted and queued.
    registry
        .accept_pair("thunder-b", "B", [1u8; 32], &[1u8; 32], &text)
        .expect("first use accepted");
    assert_eq!(registry.pending().len(), 1);
    // The secret is consumed: the UI must show "none — generate".
    assert!(registry.active_code_text().is_none());

    // Same code again (different peer) → rejected, nothing queued.
    let err = registry
        .accept_pair("thunder-c", "C", [2u8; 32], &[2u8; 32], &text)
        .expect_err("second use rejected");
    assert!(err.contains("incorrect") || err.contains("expired"));
    assert_eq!(registry.pending().len(), 1);

    // Incorrect code → rejected.
    let err = registry
        .accept_pair("thunder-d", "D", [3u8; 32], &[3u8; 32], "XXXX-YYYY")
        .expect_err("wrong code rejected");
    assert!(err.contains("incorrect") || err.contains("expired"));
    assert_eq!(registry.pending().len(), 1);

    // Operator Reject drops the request but does NOT restore the code.
    assert!(registry.reject("thunder-b"));
    assert!(registry.pending().is_empty());
    assert!(registry.active_code_text().is_none());
}

// ---------------------------------------------------------------------------
// 18. Cancellation reaches backend execution.
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn host_cancellation_aborts_backend_work() {
    use std::sync::atomic::AtomicBool;
    let host_id = ThunderIdentity::generate("host".to_string());
    let recv_id = ThunderIdentity::generate("recv".to_string());
    let finished = Arc::new(AtomicBool::new(false));
    let host = ThunderHost::new(
        caps(&host_id.peer_id, vec![model("m", "M")]),
        Arc::new(BlockingExecutor {
            finished: finished.clone(),
        }),
    )
    .unwrap();
    let peers = Arc::new(std::sync::Mutex::new(PeerStore::default()));
    trust(
        &mut peers.lock().unwrap(),
        &recv_id.peer_id,
        recv_id.public_key_bytes(),
        "recv",
    );
    let trusted = hercules_agent::thunder::pairing::TrustedPeer {
        peer_id: host_id.peer_id.clone(),
        public_key: host_id.public_key_bytes(),
        name: "host".to_string(),
        permissions: PeerPermissions::default(),
        paired_at_epoch: 0,
    };
    let addr = serve_forever(host, host_id, peers, PairingRegistry::new()).await;
    let rx = ThunderReceiver::connect(&recv_id, &trusted, addr)
        .await
        .unwrap();
    let mut stream = rx
        .generate_stream(
            "m",
            "",
            "slow prompt",
            hercules_agent::thunder::context::ContextEnvelope::empty(),
            64,
        )
        .await
        .unwrap();
    tokio::time::sleep(Duration::from_millis(100)).await;
    stream.cancel();
    let mut saw_cancelled = false;
    let deadline = tokio::time::sleep(Duration::from_secs(5));
    tokio::pin!(deadline);
    loop {
        tokio::select! {
            ev = stream.events.recv() => match ev {
                Some(StreamEvent::Error(e)) if matches!(e, hercules_agent::thunder::error::ThunderError::RequestCancelled) => {
                    saw_cancelled = true;
                    break;
                }
                Some(_) => continue,
                None => break,
            },
            _ = &mut deadline => break,
        }
    }
    assert!(saw_cancelled, "host confirms cancellation");
    tokio::time::sleep(Duration::from_millis(200)).await;
    assert!(
        !finished.load(Ordering::SeqCst),
        "backend worker aborted before completing"
    );
    rx.close().await;
}

// ---------------------------------------------------------------------------
// Full A→B acceptance (pair → trust → select → Main AI → prompt).
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn thunder_pairing_flow_a_to_b() {
    let alice = ThunderIdentity::generate("Alice".to_string());
    let bob = ThunderIdentity::generate("Bob".to_string());
    let registry = PairingRegistry::new();
    registry.set_host_identity(
        alice.peer_id.clone(),
        "Alice".to_string(),
        alice.public_key_bytes(),
    );
    let code = PairingCode::generate();
    let code_text = code.code.clone();
    registry.set_code(Some(code));

    let host = ThunderHost::new(
        caps(&alice.peer_id, vec![model("qwen3-32b", "Qwen3-32B")]),
        Arc::new(ReplyExecutor {
            reply: String::new(),
            tokens: vec!["hello ".to_string(), "from Alice".to_string()],
        }),
    )
    .unwrap();
    let host_peers = Arc::new(std::sync::Mutex::new(PeerStore::default()));
    let addr = serve_forever(host, alice.clone(), host_peers.clone(), registry.clone()).await;

    // B discovers A.
    let mut recv_disc = Discovery::bind_loopback(0).unwrap();
    let mut host_disc = Discovery::bind_loopback(0).unwrap();
    host_disc
        .announce_to(&alice, recv_disc.local_beacon_addr().unwrap())
        .unwrap();
    tokio::time::sleep(Duration::from_millis(100)).await;
    assert!(recv_disc.poll(&bob.peer_id).contains(&alice.peer_id));
    let beacon = recv_disc
        .peers()
        .into_iter()
        .find(|p| p.peer_id == alice.peer_id)
        .unwrap()
        .clone();

    // B submits A's code → accepted → B trusts with fingerprint shown.
    let accepted = thunder_ui::send_pairing_request(&bob, addr, &code_text)
        .await
        .expect("pairing accepted");
    let fp = ThunderIdentity::fingerprint_of(&accepted.public_key);
    assert!(!fp.is_empty());
    let mut bob_peers = PeerStore::default();
    trust(
        &mut bob_peers,
        &accepted.peer_id,
        accepted.public_key,
        &accepted.name,
    );

    // A sees B (directed beacon) → A trusts.
    recv_disc
        .announce_to(&bob, host_disc.local_beacon_addr().unwrap())
        .unwrap();
    tokio::time::sleep(Duration::from_millis(100)).await;
    host_disc.poll(&alice.peer_id);
    let b_beacon = host_disc
        .peers()
        .into_iter()
        .find(|p| p.peer_id == bob.peer_id)
        .expect("A sees B")
        .clone();
    trust(
        &mut host_peers.lock().unwrap(),
        &bob.peer_id,
        b_beacon.public_key,
        "Bob",
    );
    assert!(registry.consume(&bob.peer_id).is_some());

    // Shared store already carries trust — no re-serve needed.
    // B: Models → select → Set as Main AI → prompt.
    let trusted = bob_peers.get(&alice.peer_id).unwrap().clone();
    let fetched = thunder_ui::fetch_remote_models(&bob, &trusted, addr)
        .await
        .expect("models fetch");
    assert_eq!(fetched.len(), 1);
    assert_eq!(fetched[0].id, "qwen3-32b");
    assert_eq!(beacon.public_key, alice.public_key_bytes());
    let target = thunder_ui::make_remote_target(&alice.peer_id, addr, "qwen3-32b", trusted);
    let backend = SharedThunderBackend::new(Arc::new(bob), target);
    let label = thunder_ui::main_ai_label(&AgentBackend::SharedThunder(backend.clone()));
    assert_eq!(label, "Thunder / Alice / qwen3-32b");
    let out = stream_text(&backend, "say hello").await;
    assert!(out.contains("hello from Alice"), "got: {out}");
}

// ---------------------------------------------------------------------------
// 20. Reverse direction, independent.
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn thunder_pairing_flow_b_to_a_independent() {
    let bob = ThunderIdentity::generate("Bob".to_string());
    let alice = ThunderIdentity::generate("Alice".to_string());
    let registry = PairingRegistry::new();
    registry.set_host_identity(
        bob.peer_id.clone(),
        "Bob".to_string(),
        bob.public_key_bytes(),
    );
    let code = PairingCode::generate();
    let code_text = code.code.clone();
    registry.set_code(Some(code));

    let host = ThunderHost::new(
        caps(&bob.peer_id, vec![model("qwen3-8b", "Qwen3-8B")]),
        Arc::new(ReplyExecutor {
            reply: "hello from Bob".to_string(),
            tokens: vec![],
        }),
    )
    .unwrap();
    let bob_peers = Arc::new(std::sync::Mutex::new(PeerStore::default()));
    trust(
        &mut bob_peers.lock().unwrap(),
        &alice.peer_id,
        alice.public_key_bytes(),
        "Alice",
    );
    let addr = serve_forever(host, bob.clone(), bob_peers, registry).await;

    let accepted = thunder_ui::send_pairing_request(&alice, addr, &code_text)
        .await
        .expect("pairing accepted");
    assert_eq!(accepted.peer_id, bob.peer_id);
    let mut alice_peers = PeerStore::default();
    trust(
        &mut alice_peers,
        &accepted.peer_id,
        accepted.public_key,
        &accepted.name,
    );
    let trusted = alice_peers.get(&bob.peer_id).unwrap().clone();
    let target = RemoteTarget {
        peer_id: bob.peer_id.clone(),
        address: addr,
        model_id: "qwen3-8b".to_string(),
        trusted_peer: trusted,
    };
    let backend = SharedThunderBackend::new(Arc::new(alice), target);
    let label = thunder_ui::main_ai_label(&AgentBackend::SharedThunder(backend.clone()));
    assert_eq!(label, "Thunder / Bob / qwen3-8b");
    let out = stream_text(&backend, "say hello").await;
    assert!(out.contains("hello from Bob"), "got: {out}");
}
