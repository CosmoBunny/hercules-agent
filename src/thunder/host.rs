//! Host side: validates requests, executes against the EXISTING
//! backend, enforces concurrency limits. Inference only — never shell,
//! filesystem, tools, or credentials. Unknown/untrusted peers rejected
//! before any model work.

use std::sync::Arc;
use tokio::sync::Semaphore;

use super::capabilities::HostCapabilities;
use super::connection::ThunderConnection;
use super::error::ThunderError;
use super::pairing::PeerStore;
use super::protocol::{MessageKind, ThunderMessage};

/// Forwards a request upstream when this host cannot serve it (T13).
/// Multi-hop only: `hops_left` bounds recursion; a host never forwards
/// unless the requesting peer granted forwarding permission AND this
/// host advertises forwarding.
#[async_trait::async_trait]
pub trait PeerForwarder: Send + Sync {
    async fn forward(
        &self,
        model_id: &str,
        prompt: &str,
        hops_left: u32,
    ) -> Result<String, ThunderError>;
}

/// Executes one prompt on the host's real backend. Thin adapter — the
/// actual generation implementations are untouched. Streaming is
/// first-class: token events flow as they arrive.
#[async_trait::async_trait]
pub trait ThunderExecutor: Send + Sync {
    async fn generate(
        &self,
        model_id: &str,
        system_prompt: &str,
        prompt: &str,
    ) -> Result<String, ThunderError> {
        let (tx, mut rx) = tokio::sync::mpsc::channel(64);
        self.generate_stream(model_id, system_prompt, prompt, tx)
            .await?;
        let mut full = String::new();
        while let Some(tok) = rx.recv().await {
            full.push_str(&tok);
        }
        Ok(full)
    }

    async fn generate_stream(
        &self,
        model_id: &str,
        system_prompt: &str,
        prompt: &str,
        tx: tokio::sync::mpsc::Sender<String>,
    ) -> Result<(), ThunderError>;
}

/// Adapter over the existing AgentBackend (real runtimes only).
pub struct AgentBackendExecutor {
    pub backend: crate::backend::AgentBackend,
    /// The single wire model id this host serves. Requests for any
    /// other id are rejected with ModelUnavailable BEFORE execution —
    /// the advertised model always equals the executed model.
    pub advertised_model: String,
}

#[async_trait::async_trait]
impl ThunderExecutor for AgentBackendExecutor {
    async fn generate(
        &self,
        model_id: &str,
        system_prompt: &str,
        prompt: &str,
    ) -> Result<String, ThunderError> {
        if model_id != self.advertised_model {
            return Err(ThunderError::ModelUnavailable {
                model: model_id.to_string(),
            });
        }
        let full = if system_prompt.trim().is_empty() {
            prompt.to_string()
        } else {
            format!("{system_prompt}\n\n{prompt}")
        };
        self.backend
            .generate(&full)
            .await
            .map_err(|e| ThunderError::InvalidRequest {
                detail: format!("backend failed: {e}"),
            })
    }

    async fn generate_stream(
        &self,
        model_id: &str,
        system_prompt: &str,
        prompt: &str,
        tx: tokio::sync::mpsc::Sender<String>,
    ) -> Result<(), ThunderError> {
        if model_id != self.advertised_model {
            return Err(ThunderError::ModelUnavailable {
                model: model_id.to_string(),
            });
        }
        // Bridge the existing Arc<Mutex<String>> stream target into the
        // channel: run the real backend, poll for deltas, forward them.
        // Cancellation flows through the shared flag (host sets it false
        // on Cancel), exactly like local generation.
        let full = if system_prompt.trim().is_empty() {
            prompt.to_string()
        } else {
            format!("{system_prompt}\n\n{prompt}")
        };
        let target = std::sync::Arc::new(std::sync::Mutex::new(String::new()));
        let flag = std::sync::Arc::new(std::sync::Mutex::new(true));
        let backend = self.backend.clone();
        let t2 = target.clone();
        let f2 = flag.clone();
        let gen_task = tokio::spawn(async move { backend.generate_stream(&full, t2, f2).await });
        let mut sent = 0usize;
        loop {
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
            let current = target.lock().map(|t| t.clone()).unwrap_or_default();
            if current.len() > sent {
                let delta = current[sent..].to_string();
                sent = current.len();
                if tx.send(delta).await.is_err() {
                    break;
                }
            }
            if gen_task.is_finished() {
                let current = target.lock().map(|t| t.clone()).unwrap_or_default();
                if current.len() > sent {
                    let _ = tx.send(current[sent..].to_string()).await;
                }
                break;
            }
        }
        match gen_task.await {
            Ok(Ok(_)) => Ok(()),
            Ok(Err(e)) => Err(ThunderError::InvalidRequest {
                detail: format!("backend failed: {e}"),
            }),
            Err(e) => Err(ThunderError::InvalidRequest {
                detail: format!("backend task failed: {e}"),
            }),
        }
    }
}

/// Test/stub executor: deterministic output, no model needed.
#[cfg(test)]
pub struct StubExecutor {
    pub reply: String,
    /// Optional per-token streaming (empty = single shot).
    pub tokens: Vec<String>,
}

#[cfg(test)]
#[async_trait::async_trait]
impl ThunderExecutor for StubExecutor {
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
                tokio::time::sleep(std::time::Duration::from_millis(20)).await;
                if tx.send(tok.clone()).await.is_err() {
                    break;
                }
            }
        }
        Ok(())
    }
}

/// Failing executor for failover tests (typed error, no model needed).
#[cfg(test)]
pub struct FailingExecutor {
    pub detail: String,
}

#[cfg(test)]
#[async_trait::async_trait]
impl ThunderExecutor for FailingExecutor {
    async fn generate(
        &self,
        _model_id: &str,
        _system: &str,
        _prompt: &str,
    ) -> Result<String, ThunderError> {
        Err(ThunderError::InvalidRequest {
            detail: self.detail.clone(),
        })
    }

    async fn generate_stream(
        &self,
        _model_id: &str,
        _system: &str,
        _prompt: &str,
        _tx: tokio::sync::mpsc::Sender<String>,
    ) -> Result<(), ThunderError> {
        Err(ThunderError::InvalidRequest {
            detail: self.detail.clone(),
        })
    }
}

/// Slow executor for concurrency tests.
#[cfg(test)]
pub struct SlowExecutor {
    pub delay: std::time::Duration,
}

#[cfg(test)]
#[async_trait::async_trait]
impl ThunderExecutor for SlowExecutor {
    async fn generate(
        &self,
        _model_id: &str,
        _system: &str,
        _prompt: &str,
    ) -> Result<String, ThunderError> {
        tokio::time::sleep(self.delay).await;
        Ok("slow done".to_string())
    }

    async fn generate_stream(
        &self,
        model_id: &str,
        system: &str,
        prompt: &str,
        _tx: tokio::sync::mpsc::Sender<String>,
    ) -> Result<(), ThunderError> {
        self.generate(model_id, system, prompt).await?;
        Ok(())
    }
}

/// One validated pairing attempt awaiting an explicit operator
/// Trust/Reject decision. Created ONLY by `PairingRegistry::accept_pair`
/// after the pairing code validated — never synthesized by UI code.
#[derive(Debug, Clone)]
pub struct PendingPairingRequest {
    pub peer_id: String,
    pub name: String,
    pub public_key: [u8; 32],
    pub received_epoch: u64,
}

/// Host-side pairing state: the active pairing secret plus the queue of
/// validated pairing attempts.
///
/// Protocol (TOFU with out-of-band code):
/// 1. Host generates a `PairingCode`, stores it here, displays it.
/// 2. Receiver opens a connection and sends `Pair { code, peer_id,
///    name, public_key }` over the encrypted channel.
/// 3. `accept_pair` checks: registry has an active code; the presented
///    code matches via `PairingCode::matches` (expiry included);
///    identity fields well-formed; presented key equals the
///    handshake-verified channel key. Anything else is rejected with a
///    reason — the peer stays untrusted.
/// 4. The operator sees fingerprint + identity and chooses TRUST
///    (persists `TrustedPeer`) or REJECT (drops the request).
/// The accept reply echoes `Pair` with an EMPTY code plus the host's
/// identity so the receiver learns the real peer id/key for its own
/// fingerprint confirmation.
#[derive(Debug, Default)]
pub struct PairingRegistry {
    host_peer_id: std::sync::Mutex<String>,
    host_name: std::sync::Mutex<String>,
    host_key: std::sync::Mutex<[u8; 32]>,
    code: std::sync::Mutex<Option<super::pairing::PairingCode>>,
    pending: std::sync::Mutex<Vec<PendingPairingRequest>>,
}

impl PairingRegistry {
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    /// Snapshot of the host identity advertised in Pair accept replies.
    pub fn set_host_identity(&self, peer_id: String, name: String, public_key: [u8; 32]) {
        *self.host_peer_id.lock().unwrap() = peer_id;
        *self.host_name.lock().unwrap() = name;
        *self.host_key.lock().unwrap() = public_key;
    }

    /// Install (or clear) the active pairing secret.
    pub fn set_code(&self, code: Option<super::pairing::PairingCode>) {
        *self.code.lock().unwrap() = code;
    }

    /// Active (non-expired) code text for display, if any.
    pub fn active_code_text(&self) -> Option<String> {
        self.code
            .lock()
            .unwrap()
            .as_ref()
            .filter(|c| c.is_valid())
            .map(|c| c.code.clone())
    }

    /// Validate one pairing attempt. On success the request is queued
    /// for the operator, the active pairing code is CONSUMED (single-use:
    /// the same code cannot accept a second peer), and the host identity
    /// triple is returned for the accept reply. On failure NOTHING is
    /// queued, the code is left untouched, and the peer stays untrusted.
    /// Operator Reject drops the queued request but does NOT restore the
    /// code — the host must generate a fresh code for the next peer.
    pub fn accept_pair(
        &self,
        peer_id: &str,
        name: &str,
        public_key: [u8; 32],
        channel_key: &[u8; 32],
        code: &str,
    ) -> Result<(String, String, [u8; 32]), String> {
        if peer_id.trim().is_empty() || name.trim().is_empty() {
            return Err("malformed pairing request: empty identity".to_string());
        }
        if public_key == [0u8; 32] {
            return Err("malformed pairing request: missing public key".to_string());
        }
        if public_key != *channel_key {
            return Err("pairing rejected: presented key differs from channel key".to_string());
        }
        if code.trim().is_empty() {
            return Err("pairing rejected: missing pairing code".to_string());
        }
        {
            let mut guard = self.code.lock().unwrap();
            let valid = guard.as_ref().map(|c| c.matches(code)).unwrap_or(false);
            if !valid {
                return Err("pairing rejected: incorrect or expired code".to_string());
            }
            // Single-use: a successful exchange consumes the secret.
            *guard = None;
        }
        let mut pending = self.pending.lock().unwrap();
        if !pending.iter().any(|p| p.peer_id == peer_id) {
            pending.push(PendingPairingRequest {
                peer_id: peer_id.to_string(),
                name: name.to_string(),
                public_key,
                received_epoch: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_secs())
                    .unwrap_or(0),
            });
        }
        Ok((
            self.host_peer_id.lock().unwrap().clone(),
            self.host_name.lock().unwrap().clone(),
            *self.host_key.lock().unwrap(),
        ))
    }

    /// Validated requests awaiting operator decision.
    pub fn pending(&self) -> Vec<PendingPairingRequest> {
        self.pending.lock().unwrap().clone()
    }

    /// Consume a validated request (Trust path). Returns None unless a
    /// validated Pair exchange produced this entry.
    pub fn consume(&self, peer_id: &str) -> Option<PendingPairingRequest> {
        let mut pending = self.pending.lock().unwrap();
        pending
            .iter()
            .position(|p| p.peer_id == peer_id)
            .map(|i| pending.remove(i))
    }

    /// Drop a validated request without trusting (Reject path).
    pub fn reject(&self, peer_id: &str) -> bool {
        let mut pending = self.pending.lock().unwrap();
        let before = pending.len();
        pending.retain(|p| p.peer_id != peer_id);
        pending.len() != before
    }
}

/// A Thunder host: advertised capabilities + executor + concurrency gate.
#[derive(Clone)]
pub struct ThunderHost {
    caps: HostCapabilities,
    executor: Arc<dyn ThunderExecutor>,
    semaphore: Arc<Semaphore>,
    forwarder: Option<Arc<dyn PeerForwarder>>,
    pairing: Option<Arc<PairingRegistry>>,
}

impl ThunderHost {
    pub fn new(
        caps: HostCapabilities,
        executor: Arc<dyn ThunderExecutor>,
    ) -> Result<Self, ThunderError> {
        Self::build(caps, executor, None)
    }

    pub fn with_forwarder(
        caps: HostCapabilities,
        executor: Arc<dyn ThunderExecutor>,
        forwarder: Arc<dyn PeerForwarder>,
    ) -> Result<Self, ThunderError> {
        Self::build(caps, executor, Some(forwarder))
    }

    fn build(
        caps: HostCapabilities,
        executor: Arc<dyn ThunderExecutor>,
        forwarder: Option<Arc<dyn PeerForwarder>>,
    ) -> Result<Self, ThunderError> {
        caps.validate()?;
        let max = caps.max_concurrent_requests.max(1) as usize;
        Ok(Self {
            caps,
            executor,
            semaphore: Arc::new(Semaphore::new(max)),
            forwarder,
            pairing: None,
        })
    }

    pub fn capabilities(&self) -> &HostCapabilities {
        &self.caps
    }

    /// Attach the pairing registry for this host. Pair requests are
    /// validated against the registry's active code; without a registry
    /// the host answers Pair with "pairing not enabled".
    pub fn set_pairing_registry(&mut self, registry: Arc<PairingRegistry>) {
        self.pairing = Some(registry);
    }

    /// Serve one established connection until shutdown/disconnect.
    /// Every request is validated: trusted peer, permission, shared model,
    /// limits, concurrency. Anything else gets a typed Error reply.
    /// Generation streams: tokens fan out as they arrive; Cancel stops
    /// the worker task and its backend flag.
    pub async fn serve(
        &self,
        conn: ThunderConnection,
        peers: &PeerStore,
    ) -> Result<(), ThunderError> {
        // No host-wide shutdown: per-request Cancel still works, the
        // session lives until disconnect. Production host tasks use
        // `serve_with_shutdown` so Host stop closes live sessions.
        self.serve_with_shutdown(conn, peers, tokio_util::sync::CancellationToken::new())
            .await
    }

    /// Serve one connection until disconnect OR host-wide shutdown.
    /// On shutdown: every active generation worker is signalled +
    /// aborted, the session is closed, and the task returns — no
    /// detached connection tasks survive "Host stopped".
    pub async fn serve_with_shutdown(
        &self,
        mut conn: ThunderConnection,
        peers: &PeerStore,
        shutdown: tokio_util::sync::CancellationToken,
    ) -> Result<(), ThunderError> {
        use super::connection::ThunderTransport;
        use std::collections::HashMap;
        struct Active {
            flag: std::sync::Arc<std::sync::Mutex<bool>>,
            token: tokio_util::sync::CancellationToken,
            task: tokio::task::JoinHandle<()>,
        }
        enum Forward {
            Token { rid: String, text: String },
            Finished { rid: String, full: String },
            Failed { rid: String, err: ThunderError },
        }
        let (fwd_tx, mut fwd_rx) = tokio::sync::mpsc::channel::<Forward>(256);
        let mut active: HashMap<String, Active> = HashMap::new();
        loop {
            tokio::select! {
                _ = shutdown.cancelled() => {
                    // Host stop: abort every active generation worker,
                    // close the session, leave nothing detached.
                    for (_, a) in active.drain() {
                        if let Ok(mut f) = a.flag.lock() {
                            *f = false;
                        }
                        a.token.cancel();
                        a.task.abort();
                    }
                    let _ = conn.shutdown().await;
                    return Ok(());
                }
                msg = conn.receive() => {
                    let msg = match msg {
                        Ok(m) => m,
                        Err(_) => {
                            return Ok(())
                        },
                    };
                    let req_id = msg.request_id.clone();
                    // Cancel short-circuits: stop worker, confirm, forget.
                    // SEMANTICS: Cancelled means "cancellation ACCEPTED"
                    // (worker signalled: flag cleared, token cancelled,
                    // task aborted) — NOT "computation has definitely
                    // stopped". The nested backend exec task may still be
                    // unwinding until the shared flag is observed.
                    if let MessageKind::Cancel = &msg.payload {
                        if let Some(a) = active.remove(&req_id) {
                            if let Ok(mut f) = a.flag.lock() {
                                *f = false;
                            }
                            a.token.cancel();
                            a.task.abort();
                        }
                        let out = ThunderMessage::new(&req_id, MessageKind::Cancelled);
                        if conn.send(&out).await.is_err() {
                            return Ok(());
                        }
                        continue;
                    }
                    match self.handle(&msg, &conn.peer_id, &conn.peer_public_key, peers).await {
                        Ok(HandleOut::Reply(payload)) => {
                            let out = ThunderMessage {
                                protocol_version: super::protocol::PROTOCOL_VERSION,
                                request_id: req_id,
                                payload,
                                origin_peer: None,
                                hop_count: 0,
                                max_hops: super::protocol::MAX_HOPS,
                            };
                            let done = matches!(out.payload, MessageKind::Shutdown);
                            if conn.send(&out).await.is_err() {
                                return Ok(());
                            }
                            if done {
                                return Ok(());
                            }
                        }
                        Ok(HandleOut::Dispatch(params)) => {
                            // Validated generation: stream via worker task.
                            let rid = req_id.clone();
                            // Duplicate in-flight request ids are rejected:
                            // request IDs must be unique per connection.
                            if active.contains_key(&rid) {
                                let out = ThunderMessage::new(
                                    req_id,
                                    MessageKind::Error {
                                        code: "error".to_string(),
                                        message: "duplicate request id".to_string(),
                                    },
                                );
                                let _ = conn.send(&out).await;
                                continue;
                            }
                            let flag = std::sync::Arc::new(std::sync::Mutex::new(true));
                            let token = tokio_util::sync::CancellationToken::new();
                            let permit = match self.semaphore.clone().try_acquire_owned() {
                                Ok(p) => p,
                                Err(_) => {
                                    let out = ThunderMessage::new(
                                        req_id,
                                        MessageKind::Error {
                                            code: "busy".to_string(),
                                            message: "host busy".to_string(),
                                        },
                                    );
                                    let _ = conn.send(&out).await;
                                    continue;
                                }
                            };
                            let executor = self.executor.clone();
                            let fwd = fwd_tx.clone();
                            let f2 = flag.clone();
                            let t2 = token.clone();
                            let rid_task = rid.clone();
                            let task = tokio::spawn(async move {
                                let (tx, mut rx) = tokio::sync::mpsc::channel::<String>(256);
                                let exec = tokio::spawn(async move {
                                    executor
                                        .generate_stream(
                                            &params.model_id,
                                            &params.system_prompt,
                                            &params.prompt,
                                            tx,
                                        )
                                        .await
                                });
                                let mut full = String::new();
                                let result: Result<String, ThunderError> = loop {
                                    tokio::select! {
                                        _ = t2.cancelled() => {
                                            if let Ok(mut f) = f2.lock() {
                                                *f = false;
                                            }
                                            break Err(ThunderError::RequestCancelled);
                                        }
                                        item = rx.recv() => {
                                            match item {
                                                Some(tok) => {
                                                    full.push_str(&tok);
                                                    let _ = fwd.send(Forward::Token {
                                                        rid: rid_task.clone(),
                                                        text: tok,
                                                    }).await;
                                                }
                                                None => break match exec.await {
                                                    Ok(Ok(())) => Ok(full.clone()),
                                                    Ok(Err(e)) => Err(e),
                                                    Err(e) => Err(ThunderError::InvalidRequest {
                                                        detail: format!("worker failed: {e}"),
                                                    }),
                                                },
                                            }
                                        }
                                    }
                                };
                                drop(permit);
                                match result {
                                    Ok(text) => {
                                        let _ = fwd.send(Forward::Finished { rid: rid_task.clone(), full: text }).await;
                                    }
                                    Err(e) => {
                                        let _ = fwd.send(Forward::Failed { rid: rid_task.clone(), err: e }).await;
                                    }
                                }
                            });
                            active.insert(rid.clone(), Active { flag, token, task });
                        }
                        Ok(HandleOut::None) => {}
                        Err(e) => {
                            let (code, message) = thunder_error_parts(&e);
                            let out = ThunderMessage::new(
                                req_id,
                                MessageKind::Error { code, message },
                            );
                            let _ = conn.send(&out).await;
                        }
                    }
                }
                Some(fwd) = fwd_rx.recv() => {
                    match fwd {
                        Forward::Token { rid, text } => {
                            let out = ThunderMessage::new(rid, MessageKind::Token { text });
                            if conn.send(&out).await.is_err() {
                                return Ok(());
                            }
                        }
                        Forward::Finished { rid, full } => {
                            active.remove(&rid);
                            let out = ThunderMessage::new(
                                rid,
                                MessageKind::Done {
                                    full_text: full,
                                    forwarded: false,
                                },
                            );
                            if conn.send(&out).await.is_err() {
                                return Ok(());
                            }
                        }
                        Forward::Failed { rid, err } => {
                            active.remove(&rid);
                            let (code, message) = thunder_error_parts(&err);
                            let out = ThunderMessage::new(rid, MessageKind::Error { code, message });
                            let _ = conn.send(&out).await;
                        }
                    }
                }
            }
        }
    }
}

/// Validated generation ready for worker dispatch (no execution here).
#[derive(Debug, Clone)]
struct DispatchParams {
    model_id: String,
    system_prompt: String,
    prompt: String,
}

enum HandleOut {
    Reply(MessageKind),
    Dispatch(DispatchParams),
    None,
}

impl ThunderHost {
    async fn handle(
        &self,
        msg: &ThunderMessage,
        peer_id: &str,
        channel_key: &[u8; 32],
        peers: &PeerStore,
    ) -> Result<HandleOut, ThunderError> {
        // Pairing is pre-trust by design: an untrusted receiver proves
        // knowledge of the out-of-band code here. Every other message
        // requires a persisted TrustedPeer below.
        if let MessageKind::Pair {
            code,
            peer_id: claimed_id,
            name,
            public_key,
        } = &msg.payload
        {
            let registry = self.pairing.as_ref().ok_or(ThunderError::InvalidRequest {
                detail: "pairing not enabled on this host".to_string(),
            })?;
            // Channel binding: the Pair identity must be the handshake
            // identity of THIS connection — never a third peer's.
            if claimed_id != peer_id {
                return Err(ThunderError::InvalidRequest {
                    detail: "pairing rejected: identity differs from channel".to_string(),
                });
            }
            let mut key = [0u8; 32];
            if public_key.len() != 32 {
                return Err(ThunderError::InvalidRequest {
                    detail: "pairing rejected: malformed public key".to_string(),
                });
            }
            key.copy_from_slice(public_key);
            match registry.accept_pair(claimed_id, name, key, channel_key, code) {
                Ok((host_id, host_name, host_key)) => {
                    return Ok(HandleOut::Reply(MessageKind::Pair {
                        code: String::new(),
                        peer_id: host_id,
                        name: host_name,
                        public_key: host_key.to_vec(),
                    }));
                }
                Err(reason) => {
                    return Err(ThunderError::InvalidRequest { detail: reason });
                }
            }
        }
        let trusted = peers
            .get(peer_id)
            .ok_or(ThunderError::AuthenticationFailed)?;
        match &msg.payload {
            MessageKind::Ping { sent_epoch_ms } => {
                let _ = sent_epoch_ms;
                Ok(HandleOut::Reply(MessageKind::Ping { sent_epoch_ms: 0 }))
            }
            MessageKind::Capabilities { .. } => Ok(HandleOut::Reply(MessageKind::Capabilities {
                caps: self.caps.clone(),
            })),
            MessageKind::Models { .. } => Ok(HandleOut::Reply(MessageKind::Models {
                models: self.caps.models.clone(),
            })),
            MessageKind::Generate {
                model_id,
                system_prompt,
                prompt,
                context,
                max_new_tokens,
                ..
            } => {
                if !trusted.permissions.inference {
                    return Err(ThunderError::PermissionDenied);
                }
                let model = self.caps.find_model(model_id);
                if model.is_none() {
                    // T13 forwarding: only with explicit permission on
                    // both sides, and only when this request still has a
                    // hop available (hop_count < max_hops). The forwarded
                    // message arrives upstream with hop_count + 1, so
                    // hops_left bounds ITS further recursion — 0 means
                    // the upstream is a terminal hop.
                    let hops_left = msg.max_hops.saturating_sub(msg.hop_count + 1);
                    let forwarder = self.forwarder.as_ref();
                    if forwarder.is_some()
                        && msg.hop_count < msg.max_hops
                        && trusted.permissions.forwarding
                        && self.caps.forwarding
                    {
                        let text = forwarder
                            .unwrap()
                            .forward(model_id, prompt, hops_left)
                            .await?;
                        return Ok(HandleOut::Reply(MessageKind::Done {
                            full_text: text,
                            forwarded: true,
                        }));
                    }
                    return Err(ThunderError::ModelUnavailable {
                        model: model_id.clone(),
                    });
                }
                if prompt.len() + context.total_bytes() > self.caps.max_context_tokens as usize * 4
                {
                    return Err(ThunderError::ContextTooLarge);
                }
                if *max_new_tokens > self.caps.max_new_tokens {
                    return Err(ThunderError::InvalidRequest {
                        detail: "max_new_tokens exceeds host limit".to_string(),
                    });
                }
                let _ = model;
                // No execution here: the serve loop spawns the worker task
                // (permit acquired there, held for the whole generation).
                Ok(HandleOut::Dispatch(DispatchParams {
                    model_id: model_id.clone(),
                    system_prompt: system_prompt.clone(),
                    prompt: prompt.clone(),
                }))
            }
            MessageKind::Shutdown => Ok(HandleOut::Reply(MessageKind::Shutdown)),
            _ => Err(ThunderError::InvalidRequest {
                detail: "unsupported message for host".to_string(),
            }),
        }
    }
}

fn thunder_error_parts(e: &ThunderError) -> (String, String) {
    let code = match e {
        ThunderError::AuthenticationFailed => "auth",
        ThunderError::PermissionDenied => "denied",
        ThunderError::ModelUnavailable { .. } => "no_model",
        ThunderError::HostBusy => "busy",
        ThunderError::ContextTooLarge => "too_large",
        ThunderError::RequestCancelled => "cancelled",
        _ => "error",
    };
    (code.to_string(), e.message())
}

#[cfg(test)]
pub(crate) async fn tests_serve_forever(
    host: ThunderHost,
    host_id: crate::thunder::identity::ThunderIdentity,
    peers: PeerStore,
) -> std::net::SocketAddr {
    use super::connection::ThunderListener;
    let (listener, port) = ThunderListener::bind(0).await.unwrap();
    let addr: std::net::SocketAddr = format!("127.0.0.1:{port}").parse().unwrap();
    let host = std::sync::Arc::new(host);
    tokio::spawn(async move {
        loop {
            let stream = match listener.accept().await {
                Ok(s) => s,
                Err(_) => return,
            };
            let conn = match ThunderConnection::handshake(stream, &host_id, None).await {
                Ok(c) => c,
                Err(_) => continue,
            };
            let host = host.clone();
            let peers = peers.clone();
            tokio::spawn(async move {
                let _ = host.serve(conn, &peers).await;
            });
        }
    });
    addr
}

#[cfg(test)]
mod tests {
    use super::super::capabilities::{HostCapabilities, ThunderModel};
    use super::super::connection::{ThunderConnection, ThunderListener};
    use super::super::identity::ThunderIdentity;
    use super::super::pairing::{PeerPermissions, PeerStore};
    use super::super::receiver::ThunderReceiver;
    use super::*;

    fn test_ids() -> (ThunderIdentity, ThunderIdentity) {
        (
            ThunderIdentity::generate("host".to_string()),
            ThunderIdentity::generate("recv".to_string()),
        )
    }

    fn test_caps(host_id: &str) -> HostCapabilities {
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

    fn trusted_pair() -> (ThunderIdentity, ThunderIdentity, PeerStore, PeerStore) {
        let (host_id, recv_id) = test_ids();
        let mut host_peers = PeerStore::default();
        assert!(host_peers.trust(
            recv_id.peer_id.clone(),
            recv_id.public_key_bytes(),
            "recv".to_string(),
            PeerPermissions::default(),
            true
        ));
        let mut recv_peers = PeerStore::default();
        assert!(recv_peers.trust(
            host_id.peer_id.clone(),
            host_id.public_key_bytes(),
            "host".to_string(),
            PeerPermissions::default(),
            true
        ));
        (host_id, recv_id, host_peers, recv_peers)
    }

    async fn serve_once(
        host: ThunderHost,
        host_id: ThunderIdentity,
        peers: PeerStore,
    ) -> std::net::SocketAddr {
        serve_many(std::sync::Arc::new(host), host_id, peers).await
    }

    /// Accept loop: every connection gets its own serve task (required
    /// once tests open multiple receivers against one host).
    async fn serve_many(
        host: std::sync::Arc<ThunderHost>,
        host_id: ThunderIdentity,
        peers: PeerStore,
    ) -> std::net::SocketAddr {
        let (listener, port) = ThunderListener::bind(0).await.unwrap();
        let addr: std::net::SocketAddr = format!("127.0.0.1:{port}").parse().unwrap();
        tokio::spawn(async move {
            loop {
                let stream = match listener.accept().await {
                    Ok(s) => s,
                    Err(_) => return,
                };
                let conn = match ThunderConnection::handshake(stream, &host_id, None).await {
                    Ok(c) => c,
                    Err(_) => continue,
                };
                let host = host.clone();
                let peers = peers.clone();
                tokio::spawn(async move {
                    let _ = host.serve(conn, &peers).await;
                });
            }
        });
        addr
    }

    fn trusted_for(host_id: &ThunderIdentity) -> crate::thunder::pairing::TrustedPeer {
        crate::thunder::pairing::TrustedPeer {
            peer_id: host_id.peer_id.clone(),
            public_key: host_id.public_key_bytes(),
            name: "host".to_string(),
            permissions: PeerPermissions::default(),
            paired_at_epoch: 0,
        }
    }

    #[tokio::test]
    async fn test_shutdown_closes_session_and_aborts_generation() {
        use super::super::receiver::ThunderReceiver;
        let (host_id, recv_id, host_peers, _) = trusted_pair();
        let host = std::sync::Arc::new(
            ThunderHost::new(
                test_caps(&host_id.peer_id),
                Arc::new(SlowExecutor {
                    delay: std::time::Duration::from_secs(60),
                }),
            )
            .unwrap(),
        );
        let shutdown = tokio_util::sync::CancellationToken::new();
        // Single connection served with the host-wide shutdown token.
        let (listener, port) = ThunderListener::bind(0).await.unwrap();
        let addr: std::net::SocketAddr = format!("127.0.0.1:{port}").parse().unwrap();
        let serve_done = tokio::spawn({
            let host = host.clone();
            let shutdown = shutdown.clone();
            let peers = host_peers.clone();
            let host_id = host_id.clone();
            async move {
                let stream = listener.accept().await.unwrap();
                let conn = ThunderConnection::handshake(stream, &host_id, None)
                    .await
                    .unwrap();
                host.serve_with_shutdown(conn, &peers, shutdown).await
            }
        });
        let rx = ThunderReceiver::connect(&recv_id, &trusted_for(&host_id), addr)
            .await
            .unwrap();
        // Active generation in flight…
        let mut stream = rx
            .generate_stream(
                "qwen3-30b",
                "",
                "slow prompt",
                super::super::context::ContextEnvelope::empty(),
                64,
            )
            .await
            .unwrap();
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        // …host stop aborts it and closes the session.
        shutdown.cancel();
        let done = tokio::time::timeout(std::time::Duration::from_secs(5), serve_done)
            .await
            .expect("serve returns on shutdown");
        assert!(done.unwrap().is_ok());
        // The receiver observes the closed session (error or EOF), never
        // a completed generation.
        let mut saw_end = false;
        let deadline = tokio::time::sleep(std::time::Duration::from_secs(5));
        tokio::pin!(deadline);
        loop {
            tokio::select! {
                ev = stream.events.recv() => match ev {
                    Some(_) => continue,
                    None => { saw_end = true; break; }
                },
                _ = &mut deadline => break,
            }
        }
        assert!(saw_end, "live session closes on host stop");
        rx.close().await;
    }

    #[tokio::test]
    async fn test_generate_round_trip() {
        let (host_id, recv_id, host_peers, _) = trusted_pair();
        let host = ThunderHost::new(
            test_caps(&host_id.peer_id),
            Arc::new(StubExecutor {
                reply: "stub says hi".to_string(),
                tokens: vec![],
            }),
        )
        .unwrap();
        let addr = serve_once(host, host_id.clone(), host_peers).await;
        let rx = ThunderReceiver::connect(&recv_id, &trusted_for(&host_id), addr)
            .await
            .unwrap();
        let out = rx
            .generate(
                "qwen3-30b",
                "sys",
                "hello",
                super::super::context::ContextEnvelope::empty(),
                64,
            )
            .await
            .unwrap();
        assert_eq!(out, "stub says hi");
        rx.close().await;
    }

    #[tokio::test]
    async fn test_untrusted_peer_rejected() {
        let (host_id, recv_id, _, _) = trusted_pair();
        // Receiver trusts host, but host does NOT trust receiver.
        let no_trust = PeerStore::default();
        let host = ThunderHost::new(
            test_caps(&host_id.peer_id),
            Arc::new(StubExecutor {
                reply: "x".to_string(),
                tokens: vec![],
            }),
        )
        .unwrap();
        let addr = serve_once(host, host_id.clone(), no_trust).await;
        let rx = ThunderReceiver::connect(&recv_id, &trusted_for(&host_id), addr)
            .await
            .unwrap();
        let err = rx
            .generate(
                "qwen3-30b",
                "",
                "hi",
                super::super::context::ContextEnvelope::empty(),
                16,
            )
            .await
            .unwrap_err();
        assert!(matches!(err, ThunderError::AuthenticationFailed), "{err:?}");
    }

    #[tokio::test]
    async fn test_unshared_model_and_denied_permission() {
        let (host_id, recv_id, mut host_peers, _) = trusted_pair();
        // Permission denied variant.
        let mut perms = PeerPermissions::default();
        perms.inference = false;
        assert!(host_peers.set_permissions(&recv_id.peer_id, perms));
        let host = ThunderHost::new(
            test_caps(&host_id.peer_id),
            Arc::new(StubExecutor {
                reply: "x".to_string(),
                tokens: vec![],
            }),
        )
        .unwrap();
        let addr = serve_once(host, host_id.clone(), host_peers).await;
        let rx = ThunderReceiver::connect(&recv_id, &trusted_for(&host_id), addr)
            .await
            .unwrap();
        let err = rx
            .generate(
                "qwen3-30b",
                "",
                "hi",
                super::super::context::ContextEnvelope::empty(),
                16,
            )
            .await
            .unwrap_err();
        assert!(matches!(err, ThunderError::PermissionDenied), "{err:?}");
        rx.close().await;
    }

    #[tokio::test]
    async fn test_unknown_model_rejected() {
        let (host_id, recv_id, host_peers, _) = trusted_pair();
        let host = ThunderHost::new(
            test_caps(&host_id.peer_id),
            Arc::new(StubExecutor {
                reply: "x".to_string(),
                tokens: vec![],
            }),
        )
        .unwrap();
        let addr = serve_once(host, host_id.clone(), host_peers).await;
        let rx = ThunderReceiver::connect(&recv_id, &trusted_for(&host_id), addr)
            .await
            .unwrap();
        let err = rx
            .generate(
                "not-shared",
                "",
                "hi",
                super::super::context::ContextEnvelope::empty(),
                16,
            )
            .await
            .unwrap_err();
        assert!(
            matches!(err, ThunderError::ModelUnavailable { .. }),
            "{err:?}"
        );
        rx.close().await;
    }

    #[tokio::test]
    async fn test_streaming_tokens_incremental() {
        use super::super::receiver::StreamEvent;
        let (host_id, recv_id, host_peers, _) = trusted_pair();
        let host = ThunderHost::new(
            test_caps(&host_id.peer_id),
            Arc::new(StubExecutor {
                reply: String::new(),
                tokens: vec!["Hello".to_string(), " ".to_string(), "world".to_string()],
            }),
        )
        .unwrap();
        let addr = serve_once(host, host_id.clone(), host_peers).await;
        let rx = ThunderReceiver::connect(&recv_id, &trusted_for(&host_id), addr)
            .await
            .unwrap();
        let mut stream = rx
            .generate_stream(
                "qwen3-30b",
                "",
                "hi",
                super::super::context::ContextEnvelope::empty(),
                64,
            )
            .await
            .unwrap();
        let mut got = Vec::new();
        while let Some(ev) = stream.events.recv().await {
            match ev {
                StreamEvent::Token(t) => got.push(t),
                StreamEvent::Done(_) => break,
                StreamEvent::Error(e) => panic!("unexpected stream error: {e:?}"),
            }
        }
        // Tokens arrived incrementally, in order, before Done.
        assert_eq!(got, vec!["Hello", " ", "world"]);
        rx.close().await;
    }

    #[tokio::test]
    async fn test_cancel_mid_stream_confirmed() {
        use super::super::receiver::StreamEvent;
        let (host_id, recv_id, host_peers, _) = trusted_pair();
        let host = ThunderHost::new(
            test_caps(&host_id.peer_id),
            Arc::new(SlowExecutor {
                delay: std::time::Duration::from_secs(30),
            }),
        )
        .unwrap();
        let addr = serve_once(host, host_id.clone(), host_peers).await;
        let rx = ThunderReceiver::connect(&recv_id, &trusted_for(&host_id), addr)
            .await
            .unwrap();
        let mut stream = rx
            .generate_stream(
                "qwen3-30b",
                "",
                "hi",
                super::super::context::ContextEnvelope::empty(),
                64,
            )
            .await
            .unwrap();
        // Cancel before anything arrives: host confirms Cancelled.
        stream.cancel();
        let ev = tokio::time::timeout(std::time::Duration::from_secs(15), stream.events.recv())
            .await
            .expect("cancel confirmation must arrive")
            .expect("stream must produce an event");
        assert!(
            matches!(ev, StreamEvent::Error(ThunderError::RequestCancelled)),
            "{ev:?}"
        );
        rx.close().await;
    }

    #[tokio::test]
    async fn test_host_busy_when_saturated() {
        let (host_id, recv_id, host_peers, _) = trusted_pair();
        let mut caps = test_caps(&host_id.peer_id);
        caps.max_concurrent_requests = 1;
        let host = ThunderHost::new(
            caps,
            Arc::new(SlowExecutor {
                delay: std::time::Duration::from_secs(5),
            }),
        )
        .unwrap();
        let addr = serve_once(host, host_id.clone(), host_peers).await;
        let rx1 = ThunderReceiver::connect(&recv_id, &trusted_for(&host_id), addr)
            .await
            .unwrap();
        let rx2 = ThunderReceiver::connect(&recv_id, &trusted_for(&host_id), addr)
            .await
            .unwrap();
        // First request occupies the only slot (runs in background).
        let rx1_handle = tokio::spawn(async move {
            rx1.generate(
                "qwen3-30b",
                "",
                "slow one",
                super::super::context::ContextEnvelope::empty(),
                64,
            )
            .await
        });
        // Give the first request time to occupy the worker.
        tokio::time::sleep(std::time::Duration::from_millis(300)).await;
        let err = rx2
            .generate(
                "qwen3-30b",
                "",
                "second",
                super::super::context::ContextEnvelope::empty(),
                64,
            )
            .await
            .unwrap_err();
        assert!(matches!(err, ThunderError::HostBusy), "{err:?}");
        let _ = rx1_handle.await;
        rx2.close().await;
    }

    #[allow(dead_code)]
    struct NoForwarder;

    #[async_trait::async_trait]
    impl PeerForwarder for NoForwarder {
        async fn forward(
            &self,
            _model_id: &str,
            _prompt: &str,
            _hops_left: u32,
        ) -> Result<String, ThunderError> {
            Err(ThunderError::HopLimitExceeded)
        }
    }

    struct UpForwarder {
        reply: String,
        seen_hops: std::sync::Mutex<Vec<u32>>,
    }

    #[async_trait::async_trait]
    impl PeerForwarder for UpForwarder {
        async fn forward(
            &self,
            _model_id: &str,
            _prompt: &str,
            hops_left: u32,
        ) -> Result<String, ThunderError> {
            if let Ok(mut v) = self.seen_hops.lock() {
                v.push(hops_left);
            }
            Ok(self.reply.clone())
        }
    }

    #[tokio::test]
    async fn test_forward_requires_permission_and_hops() {
        // Host has a forwarder but the peer lacks forwarding permission:
        // ModelUnavailable, forwarder never called.
        let (host_id, recv_id, mut host_peers, _) = trusted_pair();
        let fwd = Arc::new(UpForwarder {
            reply: "up".to_string(),
            seen_hops: std::sync::Mutex::new(Vec::new()),
        });
        let mut caps = test_caps(&host_id.peer_id);
        caps.forwarding = true;
        let host = ThunderHost::with_forwarder(
            caps,
            Arc::new(StubExecutor {
                reply: "x".to_string(),
                tokens: vec![],
            }),
            fwd.clone() as Arc<dyn PeerForwarder>,
        )
        .unwrap();
        assert!(host_peers.set_permissions(
            &recv_id.peer_id,
            PeerPermissions {
                inference: true,
                forwarding: false
            }
        ));
        let addr = serve_once(host, host_id.clone(), host_peers).await;
        let rx = ThunderReceiver::connect(&recv_id, &trusted_for(&host_id), addr)
            .await
            .unwrap();
        let err = rx
            .generate(
                "not-shared-model",
                "",
                "hi",
                super::super::context::ContextEnvelope::empty(),
                16,
            )
            .await
            .unwrap_err();
        assert!(
            matches!(err, ThunderError::ModelUnavailable { .. }),
            "{err:?}"
        );
        assert!(fwd.seen_hops.lock().unwrap().is_empty());
        rx.close().await;
    }

    #[tokio::test]
    async fn test_forward_success_is_attributed() {
        let (host_id, recv_id, mut host_peers, _) = trusted_pair();
        let fwd = Arc::new(UpForwarder {
            reply: "upstream answer".to_string(),
            seen_hops: std::sync::Mutex::new(Vec::new()),
        });
        let mut caps = test_caps(&host_id.peer_id);
        caps.forwarding = true;
        let host = ThunderHost::with_forwarder(
            caps,
            Arc::new(StubExecutor {
                reply: "x".to_string(),
                tokens: vec![],
            }),
            fwd.clone() as Arc<dyn PeerForwarder>,
        )
        .unwrap();
        assert!(host_peers.set_permissions(
            &recv_id.peer_id,
            PeerPermissions {
                inference: true,
                forwarding: true
            }
        ));
        let addr = serve_once(host, host_id.clone(), host_peers).await;
        let rx = ThunderReceiver::connect(&recv_id, &trusted_for(&host_id), addr)
            .await
            .unwrap();
        // Unknown model triggers the forward path.
        let out = rx
            .generate(
                "not-shared-model",
                "",
                "hi",
                super::super::context::ContextEnvelope::empty(),
                16,
            )
            .await
            .unwrap();
        assert_eq!(out, "upstream answer");
        // The forwarder saw hops_left = max_hops - hop_count - 1 = 0: it
        // was invoked, but as a terminal hop it cannot recurse further.
        assert_eq!(*fwd.seen_hops.lock().unwrap(), vec![0]);
        rx.close().await;
    }

    #[tokio::test]
    async fn test_forward_hop_limit_enforced() {
        // With MAX_HOPS=1, hop 1 (a forwarded request arriving with
        // hop_count 1) is rejected at validate() — the upstream can never
        // be asked to forward again.
        let (host_id, recv_id, host_peers, _) = trusted_pair();
        let fwd = Arc::new(UpForwarder {
            reply: "should-not-happen".to_string(),
            seen_hops: std::sync::Mutex::new(Vec::new()),
        });
        let mut caps = test_caps(&host_id.peer_id);
        caps.forwarding = true;
        let host = ThunderHost::with_forwarder(
            caps,
            Arc::new(StubExecutor {
                reply: "x".to_string(),
                tokens: vec![],
            }),
            fwd.clone() as Arc<dyn PeerForwarder>,
        )
        .unwrap();
        let addr = serve_once(host, host_id.clone(), host_peers).await;
        let rx = ThunderReceiver::connect(&recv_id, &trusted_for(&host_id), addr)
            .await
            .unwrap();
        let req = ThunderMessage::new(
            "r777".to_string(),
            MessageKind::Generate {
                model_id: "not-shared-model".to_string(),
                system_prompt: String::new(),
                prompt: "hi".to_string(),
                context: super::super::context::ContextEnvelope::empty(),
                max_new_tokens: 16,
                temperature: None,
            },
        );
        let mut forwarded = req.clone();
        forwarded.hop_count = 1;
        // Routed by the single pump (send_and_await registers a private
        // channel) — never a second reader on the connection.
        let reply = rx.send_and_await(forwarded).await.unwrap();
        assert!(
            matches!(reply, super::super::receiver::StreamEvent::Error(_)),
            "{reply:?}"
        );
        assert!(fwd.seen_hops.lock().unwrap().is_empty());
        let _ = req;
        rx.close().await;
    }
}
