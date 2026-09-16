//! Receiver client: connect to a trusted host, request generation,
//! enforce request IDs. Streaming + cancellation are first-class.
//!
//! Single receive owner (invariant): exactly ONE pump task per
//! ThunderConnection performs both `conn.send` and `conn.receive`.
//! Everything else — streaming consumers, cancellation, multi-agent
//! pools — receives events from per-request channels. No second reader
//! can ever consume a message intended for another request.

use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex;

use super::connection::ThunderConnection;
use super::context::ContextEnvelope;
use super::error::ThunderError;
use super::identity::ThunderIdentity;
use super::pairing::TrustedPeer;
use super::protocol::{MessageKind, ThunderMessage};

pub const REQUEST_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(300);

static NEXT_REQUEST: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

pub fn next_request_id() -> String {
    format!(
        "r{}",
        NEXT_REQUEST.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    )
}

/// Commands into the single-owner pump (the ONLY path to the socket).
enum Command {
    Send(ThunderMessage),
    Close,
}

/// A connected receiver session to one trusted host. The connection is
/// owned exclusively by the pump task; callers only hold channels.
pub struct ThunderReceiver {
    /// Consumed once by the pump (split into send/recv halves).
    conn: Arc<Mutex<Option<ThunderConnection>>>,
    cmd_tx: tokio::sync::mpsc::Sender<Command>,
    /// Taken once by the pump.
    cmd_rx: std::sync::Mutex<Option<tokio::sync::mpsc::Receiver<Command>>>,
    /// Cached peer id (no lock is ever needed to read it).
    peer_id: String,
    /// Explicit lifecycle: cancelled by `close()`, observed by BOTH pump
    /// tasks — the receive task terminates deterministically instead of
    /// relying on the socket read failing.
    shutdown: Arc<tokio_util::sync::CancellationToken>,
    /// Authenticated frames with unknown request ids: counted for
    /// diagnostics (never misdelivered, never tears the connection down —
    /// late frames can legitimately occur during shutdown).
    unknown_frames: Arc<std::sync::atomic::AtomicU64>,
    /// Per-request event channels: request_id → sender. The pump routes
    /// every inbound message by request id; nothing is dropped silently
    /// for registered requests.
    registry: Arc<std::sync::Mutex<HashMap<String, tokio::sync::mpsc::Sender<StreamEvent>>>>,
}

impl ThunderReceiver {
    pub async fn connect(
        identity: &ThunderIdentity,
        trusted: &TrustedPeer,
        addr: std::net::SocketAddr,
    ) -> Result<Self, ThunderError> {
        let conn = ThunderConnection::connect(identity, trusted, addr).await?;
        let (cmd_tx, cmd_rx) = tokio::sync::mpsc::channel(64);
        let peer_id = conn.peer_id.clone();
        let rx = Self {
            conn: Arc::new(Mutex::new(Some(conn))),
            cmd_tx,
            cmd_rx: std::sync::Mutex::new(Some(cmd_rx)),
            peer_id,
            shutdown: Arc::new(tokio_util::sync::CancellationToken::new()),
            unknown_frames: Arc::new(std::sync::atomic::AtomicU64::new(0)),
            registry: Arc::new(std::sync::Mutex::new(HashMap::new())),
        };
        let conn = rx
            .conn
            .lock()
            .await
            .take()
            .expect("connection consumed once");
        rx.spawn_pump(conn);
        Ok(rx)
    }

    /// Count of authenticated frames with unknown request ids (stale
    /// responses, late frames during shutdown). Diagnostics only: such
    /// frames are never misdelivered and never tear the connection down.
    pub fn unknown_frames_dropped(&self) -> u64 {
        self.unknown_frames
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// The ONE receive owner per connection. A dedicated task owns the
    /// receive half (single reader — no mid-frame cancellation possible)
    /// and routes every inbound message by request id. Sends go through
    /// a separate send-owner task; the two never race on one socket.
    /// Lifecycle: BOTH tasks observe `shutdown` — `close()` terminates
    /// them deterministically (the receive task stops even while blocked
    /// in a read; the in-flight frame, if any, is abandoned with the
    /// connection).
    fn spawn_pump(&self, conn: ThunderConnection) {
        let (mut send_half, mut recv_half) = conn.into_split();
        // Send owner: the ONLY writer on this socket. On Close it cancels
        // the shared token FIRST (so the receive task exits even before
        // the socket is torn down), then drops the write half.
        let mut cmd_rx = self.take_cmd_rx();
        let registry = self.registry.clone();
        let shutdown = self.shutdown.clone();
        tokio::spawn(async move {
            while let Some(cmd) = cmd_rx.recv().await {
                match cmd {
                    Command::Send(msg) => {
                        if send_half.send(&msg).await.is_err() {
                            shutdown.cancel();
                            Self::notify_all(
                                &registry,
                                StreamEvent::Error(ThunderError::ConnectionLost),
                            );
                            return;
                        }
                    }
                    Command::Close => {
                        shutdown.cancel();
                        return;
                    }
                }
            }
            shutdown.cancel();
        });
        // Receive owner: the ONLY reader on this socket. Blocking here
        // can never block a sender (separate halves) and a frame can
        // never be dropped mid-read (no racing select) — EXCEPT after
        // shutdown, where abandoning the in-flight frame is correct.
        let registry2 = self.registry.clone();
        let shutdown2 = self.shutdown.clone();
        let unknown = self.unknown_frames.clone();
        tokio::spawn(async move {
            loop {
                let msg = tokio::select! {
                    msg = recv_half.receive() => msg,
                    _ = shutdown2.cancelled() => return,
                };
                match msg {
                    Ok(m) => {
                        let rid = m.request_id.clone();
                        let terminal = matches!(
                            m.payload,
                            MessageKind::Done { .. }
                                | MessageKind::Error { .. }
                                | MessageKind::Cancelled
                        );
                        let ev = stream_event_from(m);
                        let known = registry2.lock().ok().and_then(|r| r.get(&rid).cloned());
                        match known {
                            Some(tx) => {
                                if tx.send(ev).await.is_err() {
                                    // Consumer gone: stop routing this request.
                                    if let Ok(mut r) = registry2.lock() {
                                        r.remove(&rid);
                                    }
                                }
                            }
                            // Unknown request id: counted, never misdelivered,
                            // never tears the connection down (late frames can
                            // legitimately occur during shutdown/cancel).
                            None => {
                                unknown.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            }
                        }
                        if terminal {
                            if let Ok(mut r) = registry2.lock() {
                                r.remove(&rid);
                            }
                        }
                    }
                    Err(_) => {
                        // Remote closed: mark the receiver observably
                        // dead (later calls fail fast instead of
                        // hanging) and wake every in-flight request.
                        shutdown2.cancel();
                        Self::notify_all(
                            &registry2,
                            StreamEvent::Error(ThunderError::ConnectionLost),
                        );
                        return;
                    }
                }
            }
        });
    }

    fn take_cmd_rx(&self) -> tokio::sync::mpsc::Receiver<Command> {
        self.cmd_rx
            .lock()
            .expect("pump already spawned")
            .take()
            .expect("pump already spawned")
    }

    fn notify_all(
        registry: &Arc<std::sync::Mutex<HashMap<String, tokio::sync::mpsc::Sender<StreamEvent>>>>,
        ev: StreamEvent,
    ) {
        if let Ok(mut r) = registry.lock() {
            for (_, tx) in r.drain() {
                let _ = tx.try_send(ev.clone());
            }
        }
    }

    /// Register a per-request event channel (called before the request
    /// is sent, so no early reply is lost).
    fn register(&self, request_id: &str) -> tokio::sync::mpsc::Receiver<StreamEvent> {
        let (tx, rx) = tokio::sync::mpsc::channel(256);
        self.registry
            .lock()
            .expect("registry poisoned")
            .insert(request_id.to_string(), tx);
        rx
    }

    fn deregister(&self, request_id: &str) {
        if let Ok(mut r) = self.registry.lock() {
            r.remove(request_id);
        }
    }

    /// Fail fast when the connection is observably dead (remote FIN /
    /// error seen by the pump, or explicit close). Without this, calls
    /// issued after the session died would hang forever: the request
    /// registers a channel nobody will ever answer.
    fn check_alive(&self) -> Result<(), ThunderError> {
        if self.shutdown.is_cancelled() {
            return Err(ThunderError::ConnectionLost);
        }
        Ok(())
    }

    /// Send one message through the pump (send-only — never reads).
    async fn send(&self, msg: ThunderMessage) -> Result<(), ThunderError> {
        self.cmd_tx
            .send(Command::Send(msg))
            .await
            .map_err(|_| ThunderError::ConnectionLost)
    }

    pub async fn peer_id(&self) -> String {
        // Cached: the pump owns the conn lock while blocked in receive;
        // locking here would block until the next inbound frame.
        self.peer_id.clone()
    }

    /// Full generation round trip (blocking form over the stream).
    /// Rejects any response whose request id differs (pump routing).
    pub async fn generate(
        &self,
        model_id: &str,
        system_prompt: &str,
        prompt: &str,
        context: ContextEnvelope,
        max_new_tokens: u32,
    ) -> Result<String, ThunderError> {
        let mut stream = self
            .generate_stream(model_id, system_prompt, prompt, context, max_new_tokens)
            .await?;
        let mut full = String::new();
        while let Some(ev) = stream.events.recv().await {
            match ev {
                StreamEvent::Token(t) => full.push_str(&t),
                StreamEvent::Done(text) => {
                    if full.is_empty() {
                        full = text;
                    }
                    break;
                }
                StreamEvent::Error(e) => return Err(e),
            }
        }
        Ok(full)
    }

    /// Streaming generation: tokens arrive on `events` as the host
    /// produces them. Cancel via the handle — the pump forwards the
    /// Cancel to the host and routes the Cancelled confirmation back on
    /// the same per-request channel (single owner, no competing reader).
    pub async fn generate_stream(
        &self,
        model_id: &str,
        system_prompt: &str,
        prompt: &str,
        context: ContextEnvelope,
        max_new_tokens: u32,
    ) -> Result<ThunderStream, ThunderError> {
        let request_id = next_request_id();
        let req = ThunderMessage::new(
            request_id.clone(),
            MessageKind::Generate {
                model_id: model_id.to_string(),
                system_prompt: system_prompt.to_string(),
                prompt: prompt.to_string(),
                context,
                max_new_tokens,
                temperature: None,
            },
        );
        let events = self.register(&request_id);
        if let Err(e) = self.send(req).await {
            self.deregister(&request_id);
            return Err(e);
        }
        // The session may have died under us (FIN processed after
        // register): never hand out a channel nobody will answer.
        if self.check_alive().is_err() {
            self.deregister(&request_id);
            return Err(ThunderError::ConnectionLost);
        }
        Ok(ThunderStream {
            request_id,
            events,
            cmd_tx: self.cmd_tx.clone(),
            cancelled: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
        })
    }

    /// Test/internal helper: send an arbitrary validated request and
    /// await its terminal event on a private channel (still routed by
    /// the single pump — never a second reader).
    #[doc(hidden)]
    pub async fn send_and_await(&self, msg: ThunderMessage) -> Result<StreamEvent, ThunderError> {
        let request_id = msg.request_id.clone();
        self.check_alive()?;
        let mut events = self.register(&request_id);
        if let Err(e) = self.send(msg).await {
            self.deregister(&request_id);
            return Err(e);
        }
        if self.check_alive().is_err() {
            self.deregister(&request_id);
            return Err(ThunderError::ConnectionLost);
        }
        let ev = events.recv().await.ok_or(ThunderError::ConnectionLost)?;
        Ok(ev)
    }

    /// Test/internal helper: send WITHOUT registering a request channel
    /// (used to exercise the unknown-request-id diagnostics path).
    #[doc(hidden)]
    pub async fn send_raw(&self, msg: ThunderMessage) -> Result<(), ThunderError> {
        self.cmd_tx
            .send(Command::Send(msg))
            .await
            .map_err(|_| ThunderError::ConnectionLost)
    }

    pub async fn close(self) {
        // Explicit lifecycle: cancel the shared token (both pump tasks
        // observe it) AND stop the send task. The send task drops the
        // write half, which closes the socket; the receive task exits
        // on the token — deterministically, even while blocked in a
        // read. If the send task is already gone, the token alone
        // terminates the receive task.
        self.shutdown.cancel();
        let _ = self.cmd_tx.send(Command::Close).await;
    }
}

/// Live stream handle for one remote generation.
pub struct ThunderStream {
    request_id: String,
    pub events: tokio::sync::mpsc::Receiver<StreamEvent>,
    cmd_tx: tokio::sync::mpsc::Sender<Command>,
    cancelled: Arc<std::sync::atomic::AtomicBool>,
}

/// Stream events delivered in order, routed by the single pump.
#[derive(Debug, Clone)]
pub enum StreamEvent {
    Token(String),
    /// Carries the full text for single-shot (non-streamed) replies.
    Done(String),
    Error(ThunderError),
}

impl ThunderStream {
    /// Cancel: the pump forwards Cancel to the host; the Cancelled
    /// confirmation arrives on `events` as `Error(RequestCancelled)`.
    ///
    /// SEMANTICS: `Error(RequestCancelled)` means the host ACCEPTED the
    /// cancellation — the remote computation may still be unwinding.
    /// Idempotent and send-only — never a second reader.
    pub fn cancel(&mut self) {
        if self
            .cancelled
            .swap(true, std::sync::atomic::Ordering::Relaxed)
        {
            return;
        }
        let cancel = ThunderMessage::new(self.request_id.clone(), MessageKind::Cancel);
        let _ = self.cmd_tx.try_send(Command::Send(cancel));
    }
}

fn stream_event_from(m: ThunderMessage) -> StreamEvent {
    match m.payload {
        MessageKind::Token { text } => StreamEvent::Token(text),
        MessageKind::Done { full_text, .. } => StreamEvent::Done(full_text),
        MessageKind::Cancelled => {
            // Cancel confirmation translated for the consumer.
            StreamEvent::Error(ThunderError::RequestCancelled)
        }
        MessageKind::Error { code, message } => {
            StreamEvent::Error(thunder_error_from_parts(&code, &message))
        }
        _ => StreamEvent::Error(ThunderError::InvalidRequest {
            detail: "unexpected message for request".to_string(),
        }),
    }
}

fn thunder_error_from_parts(code: &str, message: &str) -> ThunderError {
    match code {
        "auth" => ThunderError::AuthenticationFailed,
        "denied" => ThunderError::PermissionDenied,
        "no_model" => ThunderError::ModelUnavailable {
            model: message.to_string(),
        },
        "busy" => ThunderError::HostBusy,
        "too_large" => ThunderError::ContextTooLarge,
        "cancelled" => ThunderError::RequestCancelled,
        _ => ThunderError::InvalidRequest {
            detail: message.to_string(),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::super::host::{StubExecutor, ThunderHost};
    use super::super::identity::ThunderIdentity;
    use super::super::pairing::{PeerPermissions, PeerStore};
    use super::*;

    fn trusted(host_id: &ThunderIdentity) -> TrustedPeer {
        TrustedPeer {
            peer_id: host_id.peer_id.clone(),
            public_key: host_id.public_key_bytes(),
            name: "host".to_string(),
            permissions: PeerPermissions::default(),
            paired_at_epoch: 0,
        }
    }

    async fn spawn_host() -> (std::net::SocketAddr, ThunderIdentity) {
        let host_id = ThunderIdentity::generate("host".to_string());
        let recv_id = ThunderIdentity::generate("recv".to_string());
        let mut peers = PeerStore::default();
        assert!(peers.trust(
            recv_id.peer_id.clone(),
            recv_id.public_key_bytes(),
            "recv".to_string(),
            PeerPermissions::default(),
            true
        ));
        let caps = crate::thunder::capabilities::HostCapabilities {
            peer_id: host_id.peer_id.clone(),
            hardware: crate::model::HardwareInfo::detect(),
            models: vec![],
            inference: true,
            forwarding: false,
            max_concurrent_requests: 4,
            max_context_tokens: 32768,
            max_new_tokens: 512,
            streaming: true,
            cancellation: true,
        };
        let host = ThunderHost::new(
            caps,
            std::sync::Arc::new(StubExecutor {
                reply: "ok".to_string(),
                tokens: vec![],
            }),
        )
        .unwrap();
        let addr = crate::thunder::host::tests_serve_forever(host, host_id.clone(), peers).await;
        (addr, host_id)
    }

    #[tokio::test]
    async fn test_unknown_request_id_counted_never_tears_down() {
        let (addr, host_id) = spawn_host().await;
        let recv_id = ThunderIdentity::generate("recv".to_string());
        let rx = ThunderReceiver::connect(&recv_id, &trusted(&host_id), addr)
            .await
            .unwrap();
        // Ping with an unregistered request id: counted, connection stays up.
        let ping = ThunderMessage::new("r555", MessageKind::Ping { sent_epoch_ms: 1 });
        rx.send_raw(ping).await.unwrap();
        tokio::time::sleep(std::time::Duration::from_millis(150)).await;
        assert_eq!(rx.unknown_frames_dropped(), 1);
        // The connection was NOT torn down: a normal request still gets
        // routed (a Ping reply is not a generation message, so it maps
        // to a typed unexpected-message error — liveness is the point).
        let ping = ThunderMessage::new("r556", MessageKind::Ping { sent_epoch_ms: 2 });
        let ev = rx.send_and_await(ping).await.unwrap();
        assert!(matches!(ev, StreamEvent::Error(_)));
        assert_eq!(rx.unknown_frames_dropped(), 1);
        rx.close().await;
    }

    #[tokio::test]
    async fn test_close_terminates_pumps_deterministically() {
        let (addr, host_id) = spawn_host().await;
        let recv_id = ThunderIdentity::generate("recv".to_string());
        let rx = ThunderReceiver::connect(&recv_id, &trusted(&host_id), addr)
            .await
            .unwrap();
        // Hold one stream open (consumer alive), then close the receiver.
        let stream = rx
            .generate_stream("m", "", "hi", ContextEnvelope::empty(), 64)
            .await
            .unwrap();
        rx.close().await;
        // The receive task exits on the shared token: the pending
        // consumer sees the channel end (or a ConnectionLost), never a
        // silent hang.
        let ev = tokio::time::timeout(std::time::Duration::from_secs(5), async {
            let mut s = stream;
            loop {
                match s.events.recv().await {
                    Some(ev) => match ev {
                        StreamEvent::Error(_) => return true,
                        _ => continue,
                    },
                    None => return true,
                }
            }
        })
        .await
        .expect("pump must terminate after close");
        assert!(ev);
    }
}
