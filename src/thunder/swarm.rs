//! Sequential failover pool (T12 first cut): one receiver, several
//! paired hosts tried in order — first success wins.
//!
//! This is FAILOVER, not parallel multi-agent execution: hosts run one
//! at a time, never simultaneously (Planner ├─ Alice ├─ Bob ├─ Carol
//! in parallel is a later refinement). Every host is a trusted,
//! explicitly paired peer; nothing is auto-trusted.

use std::sync::Arc;

use super::context::ContextEnvelope;
use super::error::ThunderError;
use super::receiver::ThunderReceiver;

/// One pool entry: a connected receiver for one trusted host.
pub struct PoolHost {
    pub receiver: Arc<ThunderReceiver>,
}

impl From<ThunderReceiver> for PoolHost {
    fn from(receiver: ThunderReceiver) -> Self {
        Self {
            receiver: Arc::new(receiver),
        }
    }
}

/// A pool of paired hosts. Order is preference order.
#[derive(Default)]
pub struct HostPool {
    hosts: Vec<PoolHost>,
}

impl HostPool {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&mut self, receiver: ThunderReceiver) {
        self.hosts.push(receiver.into());
    }

    pub fn len(&self) -> usize {
        self.hosts.len()
    }

    pub fn is_empty(&self) -> bool {
        self.hosts.is_empty()
    }

    /// Generate on the first host that succeeds. Failed peers are
    /// returned alongside the error so callers can explain the outage.
    pub async fn generate(
        &self,
        model_id: &str,
        system_prompt: &str,
        prompt: &str,
        context: ContextEnvelope,
        max_new_tokens: u32,
    ) -> (Result<String, ThunderError>, Vec<String>) {
        let mut failed = Vec::new();
        let mut last_err = ThunderError::PeerUnavailable {
            peer: "host pool is empty".to_string(),
        };
        for host in &self.hosts {
            match host
                .receiver
                .generate(
                    model_id,
                    system_prompt,
                    prompt,
                    context.clone(),
                    max_new_tokens,
                )
                .await
            {
                // An empty success is no answer: treat it as a failure
                // and keep trying the remaining hosts.
                Ok(text) if !text.trim().is_empty() => return (Ok(text), failed),
                Ok(_) => {
                    failed.push(host.receiver.peer_id().await);
                    last_err = ThunderError::ConnectionLost;
                }
                Err(e) => {
                    failed.push(host.receiver.peer_id().await);
                    last_err = e;
                }
            }
        }
        (Err(last_err), failed)
    }
}

#[cfg(test)]
mod tests {
    use super::super::capabilities::{HostCapabilities, ThunderModel};
    use super::super::host::{FailingExecutor, StubExecutor, ThunderHost};
    use super::super::identity::ThunderIdentity;
    use super::super::pairing::{PeerPermissions, PeerStore, TrustedPeer};
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

    fn trusted(host_id: &ThunderIdentity) -> TrustedPeer {
        TrustedPeer {
            peer_id: host_id.peer_id.clone(),
            public_key: host_id.public_key_bytes(),
            name: "host".to_string(),
            permissions: PeerPermissions::default(),
            paired_at_epoch: 0,
        }
    }

    async fn spawn(
        executor: Arc<dyn super::super::host::ThunderExecutor>,
    ) -> (std::net::SocketAddr, ThunderIdentity, ThunderIdentity) {
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
        let host = ThunderHost::new(caps(&host_id.peer_id), executor).unwrap();
        let addr = super::super::host::tests_serve_forever(host, host_id.clone(), peers).await;
        (addr, host_id, recv_id)
    }

    async fn connect_to(
        addr: std::net::SocketAddr,
        host_id: &ThunderIdentity,
        recv_id: &ThunderIdentity,
    ) -> ThunderReceiver {
        ThunderReceiver::connect(recv_id, &trusted(host_id), addr)
            .await
            .unwrap()
    }

    #[tokio::test]
    async fn test_first_success_wins() {
        // Host A answers; host B would too — A is preferred.
        let (addr_a, id_a, rid_a) = spawn(Arc::new(StubExecutor {
            reply: "from-a".to_string(),
            tokens: vec![],
        }))
        .await;
        let (addr_b, id_b, rid_b) = spawn(Arc::new(StubExecutor {
            reply: "from-b".to_string(),
            tokens: vec![],
        }))
        .await;
        let mut pool = HostPool::new();
        pool.push(connect_to(addr_a, &id_a, &rid_a).await);
        pool.push(connect_to(addr_b, &id_b, &rid_b).await);
        let (res, failed) = pool
            .generate("qwen3-30b", "", "hi", ContextEnvelope::empty(), 64)
            .await;
        assert_eq!(res.unwrap(), "from-a");
        assert!(failed.is_empty());
    }

    #[tokio::test]
    async fn test_failover_to_second_host() {
        // Host A FAILS (typed error) — the pool falls to B and reports
        // A among the failed peers.
        let (addr_a, id_a, rid_a) = spawn(Arc::new(FailingExecutor {
            detail: "host a is down".to_string(),
        }))
        .await;
        let (addr_b, id_b, rid_b) = spawn(Arc::new(StubExecutor {
            reply: "from-b".to_string(),
            tokens: vec![],
        }))
        .await;
        let mut pool = HostPool::new();
        pool.push(connect_to(addr_a, &id_a, &rid_a).await);
        pool.push(connect_to(addr_b, &id_b, &rid_b).await);
        let t0 = std::time::Instant::now();
        let (res, failed) = pool
            .generate("qwen3-30b", "", "hi", ContextEnvelope::empty(), 64)
            .await;
        assert!(res.is_ok(), "failover must succeed");
        assert_eq!(res.unwrap(), "from-b");
        assert_eq!(failed, vec![id_a.peer_id.clone()]);
        assert!(t0.elapsed() < std::time::Duration::from_secs(60));
    }

    #[tokio::test]
    async fn test_empty_pool_typed_error() {
        let pool = HostPool::new();
        assert!(pool.is_empty());
        let (res, failed) = pool
            .generate("m", "", "hi", ContextEnvelope::empty(), 64)
            .await;
        assert!(res.is_err());
        assert!(failed.is_empty());
    }
}
